"""Datasets for S/TEM segmentation, including LIIF-style coordinate sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

import inr_unet.data.generation.labels  # noqa: F401  (registers LABEL_FIELDS entries)
from inr_unet.data.generation.labels import GaussianField
from inr_unet.data.generation.structures import Grid
from inr_unet.data.render_source import SyntheticRenderSource
from inr_unet.registry import LABEL_FIELDS

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Decorrelated RNG stream (length-3 SeedSequence root; see plan determinism note).
_QUERY_SALT = 3023


def build_label_field(syn: DictConfig):
    """Construct the configured label field.

    GaussianField needs sigma params from config and is handled directly; all other
    fields (e.g. CircularField) take no constructor args and are built via the registry.
    """
    if syn.label_kind == "gaussian":
        return GaussianField(
            fwhm_A=float(syn.gaussian_fwhm_A),
            sigma_floor_px=float(syn.gaussian_sigma_floor_px),
        )
    return LABEL_FIELDS.get(syn.label_kind)()


class STEMSegDataset(Dataset):
    """Yields fixed-grid ``(image, mask)`` pairs for the baseline UNet."""

    def __init__(self, cfg: DictConfig) -> None:
        self.source = SyntheticRenderSource(cfg)
        self.label_field = build_label_field(cfg.data.synthetic)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.source.get(idx)
        grid = Grid(self.source.crop_size, s.input_pixel_size_A, device=str(s.image.device))
        mask = self.label_field.rasterize(s.positions_A, s.radii_A, grid)
        return s.image[None], mask[None]


class LIIFSegDataset(Dataset):
    """Yields ``(image, coords, cell, gt)`` for resolution-agnostic training.

    ``coords`` are (x, y) normalized to [-1, 1] over the full crop; ``cell`` is the
    dimensionless target-pixel size in that normalized frame (``2 * tgt_px / crop_extent_A``,
    matching LIIF's ``2 / n_target_px``); ``gt`` is the analytic label sampled at the
    continuous query coordinates.
    """

    def __init__(self, cfg: DictConfig) -> None:
        syn = cfg.data.synthetic
        self.source = SyntheticRenderSource(cfg)
        self.label_field = build_label_field(syn)
        self.sample_q = int(syn.sample_q)
        self.tgt_px_min = float(syn.target_pixel_size_A_min)
        self.tgt_px_max = float(syn.target_pixel_size_A_max)
        self.master_seed = int(syn.master_seed)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.source.get(idx)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, int(idx), _QUERY_SALT])
        )
        q = self.sample_q
        xy_A = torch.as_tensor(
            rng.uniform(0.0, s.valid_extent_A, size=(q, 2)), dtype=torch.float32
        )
        tgt_px = float(rng.uniform(self.tgt_px_min, self.tgt_px_max))
        gt = self.label_field.values_at(s.positions_A, s.radii_A, xy_A, tgt_px)[:, None]
        crop_extent_A = self.source.crop_size * s.input_pixel_size_A
        coords = 2.0 * (xy_A / crop_extent_A) - 1.0
        cell = torch.full((q, 2), 2.0 * tgt_px / crop_extent_A)
        return s.image[None], coords, cell, gt
