"""SyntheticRenderSource: maps a dataset index to a cropped RenderedSample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from inr_unet.data.generation.renderer import TEMRenderer
from inr_unet.data.generation.sampler import AugmentationSampler
from inr_unet.data.providers import SyntheticLatticeProvider

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from inr_unet.data.generation.structures import RenderOutput

# Decorrelated RNG stream (length-3 SeedSequence root; see plan determinism note).
_CROP_SALT = 2017
# Keep columns within this Angstrom margin of the valid window (covers peak/disk support).
_LABEL_MARGIN_A = 6.0


@dataclass(frozen=True)
class RenderedSample:
    image: torch.Tensor          # [S, S] in [0, 1], reflect-padded if the render was < S
    positions_A: torch.Tensor    # [M, 2] column centers, crop-local Angstroms
    radii_A: torch.Tensor        # [M]
    input_pixel_size_A: float    # the render's pixel size (physical resolution of the input)
    valid_extent_A: float        # physical size of the valid (non-padded) region, in Angstroms


def _reflect_pad_to(img: torch.Tensor, size: int) -> torch.Tensor:
    """Pad a square [H, W] image to [size, size] with the render kept at the top-left."""
    h, w = img.shape
    py, px = size - h, size - w
    mode = "reflect" if (py < h and px < w) else "replicate"
    out = F.pad(img[None, None], (0, px, 0, py), mode=mode)
    return out[0, 0]


class SyntheticRenderSource:
    """Owns provider + sampler + renderer + crop; maps idx -> RenderedSample."""

    def __init__(self, cfg: DictConfig) -> None:
        syn = cfg.data.synthetic
        self.master_seed = int(syn.master_seed)
        self.crop_size = int(syn.crop_size)
        self.draws_per_scene = int(syn.draws_per_scene)
        self.provider = SyntheticLatticeProvider(syn, self.master_seed)
        self.sampler = AugmentationSampler(cfg.generation.sampler, self.master_seed)
        self.renderer = TEMRenderer(cfg.generation)

    def __len__(self) -> int:
        return len(self.provider) * self.draws_per_scene

    def get(self, idx: int) -> RenderedSample:
        if not 0 <= idx < len(self):
            raise IndexError(f"idx {idx} out of range [0, {len(self)})")
        scene = self.provider.get(idx // self.draws_per_scene)
        condition, params = self.sampler.sample(idx, max_fov_A=scene.fov_A)
        out = self.renderer.render(scene, condition, params)
        return self._crop(out, idx)

    def _crop(self, out: RenderOutput, idx: int) -> RenderedSample:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, int(idx), _CROP_SALT])
        )
        h = out.meta.output_size
        px = out.meta.pixel_size_A
        s = self.crop_size
        positions = out.positions_A
        radii = out.radii_A
        if h >= s:
            # render is square (h == w == output_size), so the same range bounds both offsets
            oy = int(rng.integers(0, h - s + 1))
            ox = int(rng.integers(0, h - s + 1))
            image = out.image[oy:oy + s, ox:ox + s]
            valid_extent_A = s * px
            positions = positions - torch.tensor([ox * px, oy * px], dtype=positions.dtype)
        else:
            image = _reflect_pad_to(out.image, s)
            valid_extent_A = h * px
        keep = (
            (positions >= -_LABEL_MARGIN_A) & (positions <= valid_extent_A + _LABEL_MARGIN_A)
        ).all(dim=1)
        return RenderedSample(
            image=image.contiguous(),
            positions_A=positions[keep],
            radii_A=radii[keep],
            input_pixel_size_A=float(px),
            valid_extent_A=float(valid_extent_A),
        )
