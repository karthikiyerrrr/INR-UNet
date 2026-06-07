"""SyntheticRenderSource: maps a dataset index to a cropped RenderedSample."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from inr_unet.data.generation.renderer import TEMRenderer
from inr_unet.data.generation.sampler import AugmentationSampler
from inr_unet.data.occupancy import FiniteSupportProvider
from inr_unet.data.providers import CachingProvider, CIFProvider, SyntheticLatticeProvider

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from inr_unet.data.generation.structures import ColumnList, RenderOutput, RenderParams

# Decorrelated RNG stream (length-3 SeedSequence root; see plan determinism note).
_CROP_SALT = 2017
# Keep columns within this Angstrom margin of the valid window (covers peak/disk support).
_LABEL_MARGIN_A = 6.0
# Decorrelated RNG stream for the frame-vs-background (vacuum-aim) decision on partial scenes.
_EMPTY_SALT = 6029


@dataclass(frozen=True)
class RenderedSample:
    image: torch.Tensor          # [S, S] in [0, 1], a physical-extent tile resampled to crop_size
    positions_A: torch.Tensor    # [M, 2] column centers, crop-local Angstroms
    radii_A: torch.Tensor        # [M]
    input_pixel_size_A: float    # the render's pixel size (physical resolution of the input)
    valid_extent_A: float        # physical size of the valid (non-padded) region, in Angstroms


class SyntheticRenderSource:
    """Owns provider + sampler + renderer + crop; maps idx -> RenderedSample."""

    def __init__(self, cfg: DictConfig) -> None:
        syn = cfg.data.synthetic
        self.master_seed = int(syn.master_seed)
        self.crop_size = int(syn.crop_size)
        self.draws_per_scene = int(syn.draws_per_scene)
        self.min_columns_in_crop = int(syn.min_columns_in_crop)
        self.empty_crop_fraction = float(syn.empty_crop_fraction)
        self.crop_redraw_cap = int(syn.crop_redraw_cap)
        self.tile_fov_A_min = float(syn.tile_fov_A_min)
        self.tile_fov_A_max = float(syn.tile_fov_A_max)
        self.provider = self._build_provider(cfg)
        self.sampler = AugmentationSampler(cfg.generation.sampler, self.master_seed)
        self.renderer = TEMRenderer(cfg.generation)

    def _build_provider(self, cfg: DictConfig):
        syn = cfg.data.synthetic
        if str(cfg.data.provider) == "cif":
            inner = CIFProvider(
                manifest_path=str(cfg.data.cif.manifest_path),
                n_scenes=int(syn.n_scenes),
                master_seed=self.master_seed,
                # projection-collapse exponent (per-scene, fixed); distinct from the
                # sampler's per-render ADF weighting exponent (sampler.z_exponent_*).
                n_exponent=float(cfg.data.cif.z_eff_exponent),
                group_tol_A=float(cfg.data.cif.group_tol_A),
                scene_fov_A=(
                    float(cfg.data.cif.scene_fov_A_min),
                    float(cfg.data.cif.scene_fov_A_max),
                ),
                rotation_jitter_deg=float(cfg.data.cif.rotation_jitter_deg),
                partial_fov_prob=float(cfg.data.cif.partial_fov_prob),
                supercell_nx_range=tuple(cfg.data.cif.supercell_nx_range),
                supercell_ny_range=tuple(cfg.data.cif.supercell_ny_range),
                strip_prob=float(cfg.data.cif.strip_prob),
            )
        else:
            inner = SyntheticLatticeProvider(syn, self.master_seed)
        if str(cfg.data.occupancy.mode) != "full":
            inner = FiniteSupportProvider(inner, cfg.data.occupancy, self.master_seed)
        if bool(cfg.data.cache_scenes):
            return CachingProvider(inner)
        return inner

    def __len__(self) -> int:
        return len(self.provider) * self.draws_per_scene

    def get(self, idx: int) -> RenderedSample:
        if not 0 <= idx < len(self):
            raise IndexError(f"idx {idx} out of range [0, {len(self)})")
        scene = self.provider.get(idx // self.draws_per_scene)
        condition, params = self.sampler.sample(idx, max_fov_A=scene.fov_A)
        if scene.is_partial and scene.positions_A.shape[0] > 0:
            params = replace(params, position_offset_A=self._partial_offset(scene, params, idx))
        out = self.renderer.render(scene, condition, params)
        return self._crop(out, idx)

    def _partial_offset(
        self, scene: ColumnList, params: RenderParams, idx: int
    ) -> torch.Tensor:
        """Render offset for a partial scene: 0 frames the patch; an ``empty_crop_fraction``
        fraction of draws instead aim the window into vacuum (clear of the patch + label margin
        along a random axis) so the tile is pure background."""
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, int(idx), _EMPTY_SALT])
        )
        if rng.random() >= self.empty_crop_fraction:
            return torch.zeros(2)  # frame the patch
        struct_center = scene.fov_A / 2.0
        radius = float((scene.positions_A - struct_center).norm(dim=1).max())
        fov_render = params.output_size * params.pixel_size_A
        # push the window fully past the patch: min in-window column coord then exceeds the
        # keep band [-margin, fov_render + margin], so every patch column is dropped.
        mag = fov_render / 2.0 + radius + _LABEL_MARGIN_A + 1.0
        axis = int(rng.integers(2))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        offset = torch.zeros(2)
        offset[axis] = sign * mag
        return offset

    def _crop(self, out: RenderOutput, idx: int) -> RenderedSample:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, int(idx), _CROP_SALT])
        )
        h = out.meta.output_size
        px = out.meta.pixel_size_A
        positions = out.positions_A
        radii = out.radii_A
        render_extent_A = h * px
        # Bounded physical tile window, clamped to the render extent (small renders use the whole
        # field). Round to whole pixels, then recompute the extent so reported fields are exact.
        tile_fov_A = float(rng.uniform(self.tile_fov_A_min, self.tile_fov_A_max))
        tile_fov_A = min(tile_fov_A, render_extent_A)
        w = max(1, min(int(round(tile_fov_A / px)), h))
        tile_fov_A = w * px
        oy, ox = self._choose_offset(positions, h, w, px, rng)
        window = out.image[oy:oy + w, ox:ox + w]
        image = F.interpolate(
            window[None, None], size=(self.crop_size, self.crop_size),
            mode="bilinear", align_corners=False, antialias=True,
        )[0, 0]
        positions = positions - torch.tensor([ox * px, oy * px], dtype=positions.dtype)
        keep = (
            (positions >= -_LABEL_MARGIN_A) & (positions <= tile_fov_A + _LABEL_MARGIN_A)
        ).all(dim=1)
        return RenderedSample(
            image=image.contiguous(),
            positions_A=positions[keep],
            radii_A=radii[keep],
            input_pixel_size_A=float(tile_fov_A / self.crop_size),
            valid_extent_A=float(tile_fov_A),
        )

    def _choose_offset(
        self, positions: torch.Tensor, h: int, w: int, px: float, rng: np.random.Generator
    ) -> tuple[int, int]:
        """Seek an offset holding >= min_columns_in_crop columns (unless an empty tile is drawn)."""
        allow_empty = rng.random() < self.empty_crop_fraction
        oy = ox = 0
        for _ in range(self.crop_redraw_cap):
            oy = int(rng.integers(0, h - w + 1))
            ox = int(rng.integers(0, h - w + 1))
            if allow_empty:
                break
            shifted = positions - torch.tensor([ox * px, oy * px], dtype=positions.dtype)
            in_crop = ((shifted >= 0.0) & (shifted <= w * px)).all(dim=1).sum().item()
            if in_crop >= self.min_columns_in_crop:
                break
        return oy, ox
