"""Coordinate-derived label rasters, re-rasterized at the render resolution."""

from __future__ import annotations

import math

import torch

from inr_unet.data.generation.psf import wavelength_A
from inr_unet.data.generation.structures import Grid, ImagingCondition

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_GAUSSIAN_MASK_FWHM_A = 0.2


def column_radius(cond: ImagingCondition) -> float:
    """Column radius r in A (brief Section 1.5, corrected)."""
    lam = wavelength_A(cond.energy_keV)
    alpha = cond.alpha_max_mrad * 1e-3
    return 0.5 * math.sqrt((lam / alpha) ** 2 + (cond.source_size_A / 2.0) ** 2)


def circular_mask(positions_A: torch.Tensor, radii_A: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Binary union of disks, one per column, radius radii_A[k]."""
    size = grid.output_size
    out = torch.zeros((size, size), device=grid.device)
    if positions_A.shape[0] == 0:
        return out
    yy, xx = grid.pixel_coords()
    px = positions_A[:, 0] / grid.pixel_size_A
    py = positions_A[:, 1] / grid.pixel_size_A
    r_px = radii_A / grid.pixel_size_A
    for i in range(positions_A.shape[0]):
        d2 = (xx - float(px[i])) ** 2 + (yy - float(py[i])) ** 2
        out = torch.maximum(out, (d2 <= float(r_px[i]) ** 2).to(out.dtype))
    return out


def gaussian_mask(positions_A: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Equalized Gaussian peaks (FWHM 0.2 A), normalized to [0, 1]."""
    size = grid.output_size
    out = torch.zeros((size, size), device=grid.device)
    if positions_A.shape[0] == 0:
        return out
    sigma_px = (_GAUSSIAN_MASK_FWHM_A / _FWHM_PER_SIGMA) / grid.pixel_size_A
    yy, xx = grid.pixel_coords()
    px = positions_A[:, 0] / grid.pixel_size_A
    py = positions_A[:, 1] / grid.pixel_size_A
    for i in range(positions_A.shape[0]):
        d2 = (xx - float(px[i])) ** 2 + (yy - float(py[i])) ** 2
        out = torch.maximum(out, torch.exp(-d2 / (2.0 * sigma_px**2)))
    return out
