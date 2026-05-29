"""TEMRenderer: orchestrates the forward-model stages into a RenderOutput."""

from __future__ import annotations

import math

import torch
from omegaconf import DictConfig

from inr_unet.data.generation.background import BACKGROUNDS
from inr_unet.data.generation.labels import circular_mask, column_radius, gaussian_mask
from inr_unet.data.generation.noise import apply_poisson, apply_scan_noise
from inr_unet.data.generation.psf import build_psf, fft_convolve
from inr_unet.data.generation.structures import (
    ColumnList,
    Grid,
    ImagingCondition,
    RenderMeta,
    RenderOutput,
    RenderParams,
)
from inr_unet.registry import POTENTIALS

_MARGIN_A = 6.0  # covers PSF tails + splat support for columns just outside the FOV


def _transform_columns(columns: ColumnList, params: RenderParams) -> ColumnList:
    """Rotate about the FOV center, translate, and crop to the FOV plus a margin."""
    fov = params.output_size * params.pixel_size_A
    device = params.device
    pos = columns.positions_A.to(device)
    if pos.shape[0] == 0:
        return ColumnList(pos, columns.z.to(device), columns.count.to(device), fov)
    center = fov / 2.0
    theta = math.radians(params.rotation_deg)
    rot = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        device=device,
        dtype=pos.dtype,
    )
    moved = (pos - center) @ rot.T + center + params.position_offset_A.to(device)
    keep = (
        (moved[:, 0] >= -_MARGIN_A)
        & (moved[:, 0] <= fov + _MARGIN_A)
        & (moved[:, 1] >= -_MARGIN_A)
        & (moved[:, 1] <= fov + _MARGIN_A)
    )
    return ColumnList(moved[keep], columns.z.to(device)[keep], columns.count.to(device)[keep], fov)


def _signal_scale(clean: torch.Tensor) -> float:
    flat = clean.flatten()
    k = max(1, int(0.01 * flat.numel()))
    return float(torch.topk(flat, k).values.mean())


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else torch.zeros_like(x)


class TEMRenderer:
    """Render an ADF-STEM image and label rasters from projected columns."""

    def __init__(self, cfg: DictConfig) -> None:
        self.potential_backend = cfg.potential_backend
        self.sigma_potential_A = cfg.sigma_potential_A
        self.aperture_soft = cfg.aperture_soft

    def render(
        self,
        columns: ColumnList,
        condition: ImagingCondition,
        params: RenderParams,
    ) -> RenderOutput:
        grid = Grid(params.output_size, params.pixel_size_A, device=params.device)
        columns.validate(grid)
        cols = _transform_columns(columns, params)

        sigma = POTENTIALS.get(self.potential_backend)(
            cols, params, grid, sigma_phys_A=self.sigma_potential_A
        )
        psf = build_psf(condition, grid, soft=self.aperture_soft)
        clean = fft_convolve(sigma, psf)  # no_background_no_noise

        generator = torch.Generator(device=params.device)
        generator.manual_seed(params.seed)

        signal = _signal_scale(clean)
        bg = BACKGROUNDS.get(params.background.kind)(
            grid, params.background.params, signal, generator
        )
        no_noise = clean + bg
        warped = apply_scan_noise(no_noise, params.noise, condition, grid, generator)
        noisy = apply_poisson(warped, params.noise, generator)
        image = _normalize01(noisy)

        r = column_radius(condition)
        radii = torch.full((cols.positions_A.shape[0],), r, device=params.device)
        return RenderOutput(
            image=image,
            gaussian_mask=gaussian_mask(cols.positions_A, grid),
            circular_mask=circular_mask(cols.positions_A, radii, grid),
            no_noise=no_noise,
            no_background_no_noise=clean,
            positions_A=cols.positions_A,
            radii_A=radii,
            meta=RenderMeta(params.pixel_size_A, params.output_size, condition.name),
        )
