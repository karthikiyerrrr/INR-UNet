"""Coordinate-derived label rasters, re-rasterized at the render resolution."""

from __future__ import annotations

import math
from typing import Protocol

import torch

from inr_unet.data.generation.psf import wavelength_A
from inr_unet.data.generation.structures import Grid, ImagingCondition
from inr_unet.registry import LABEL_FIELDS

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_GAUSSIAN_MASK_FWHM_A = 0.2


class LabelField(Protocol):
    def values_at(
        self,
        positions_A: torch.Tensor,
        radii_A: torch.Tensor,
        query_xy_A: torch.Tensor,
        pixel_size_A: float | None = None,
    ) -> torch.Tensor: ...

    def rasterize(
        self,
        positions_A: torch.Tensor,
        radii_A: torch.Tensor,
        grid: Grid,
    ) -> torch.Tensor: ...


def _rasterize(
    field: LabelField,
    positions_A: torch.Tensor,
    radii_A: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    """Evaluate ``field`` at every pixel center; return an [H, W] raster."""
    yy, xx = grid.pixel_coords()
    query_xy_A = torch.stack(
        [(xx * grid.pixel_size_A).reshape(-1), (yy * grid.pixel_size_A).reshape(-1)],
        dim=-1,
    )
    vals = field.values_at(positions_A, radii_A, query_xy_A, grid.pixel_size_A)
    return vals.reshape(grid.output_size, grid.output_size)


@LABEL_FIELDS.register("gaussian")
class GaussianField:
    """Equalized Gaussian peaks (physical FWHM with a pixel floor), evaluated in Angstroms.

    Peaks are equalized to 1.0 (localization-only). The target width is physical
    (``fwhm_A``) but ``sigma`` is floored at ``sigma_floor_px * pixel_size_A`` so coarse-pixel
    rasters are not sub-Nyquist. ``pixel_size_A`` is the render/query pixel size; when ``None``
    (no floor) the width is purely physical.
    """

    def __init__(self, fwhm_A: float = _GAUSSIAN_MASK_FWHM_A, sigma_floor_px: float = 1.0) -> None:
        self.fwhm_A = fwhm_A
        self.sigma_floor_px = sigma_floor_px

    def values_at(
        self,
        positions_A: torch.Tensor,
        radii_A: torch.Tensor,
        query_xy_A: torch.Tensor,
        pixel_size_A: float | None = None,
    ) -> torch.Tensor:
        q = query_xy_A.shape[0]
        if positions_A.shape[0] == 0:
            return torch.zeros(q, device=query_xy_A.device, dtype=query_xy_A.dtype)
        sigma = self.fwhm_A / _FWHM_PER_SIGMA
        if pixel_size_A is not None:
            sigma = max(sigma, self.sigma_floor_px * float(pixel_size_A))
        d2 = ((query_xy_A[:, None, :] - positions_A[None, :, :]) ** 2).sum(dim=-1)  # [Q, M]
        return torch.exp(-d2 / (2.0 * sigma**2)).max(dim=1).values

    def rasterize(
        self,
        positions_A: torch.Tensor,
        radii_A: torch.Tensor,
        grid: Grid,
    ) -> torch.Tensor:
        return _rasterize(self, positions_A, radii_A, grid)


@LABEL_FIELDS.register("circular")
class CircularField:
    """Binary union of disks, radius ``radii_A[k]`` per column."""

    def values_at(
        self,
        positions_A: torch.Tensor,
        radii_A: torch.Tensor,
        query_xy_A: torch.Tensor,
        pixel_size_A: float | None = None,
    ) -> torch.Tensor:
        q = query_xy_A.shape[0]
        if positions_A.shape[0] == 0:
            return torch.zeros(q, device=query_xy_A.device, dtype=query_xy_A.dtype)
        d2 = ((query_xy_A[:, None, :] - positions_A[None, :, :]) ** 2).sum(dim=-1)  # [Q, M]
        inside = d2 <= (radii_A[None, :] ** 2)
        return inside.any(dim=1).to(query_xy_A.dtype)

    def rasterize(
        self,
        positions_A: torch.Tensor,
        radii_A: torch.Tensor,
        grid: Grid,
    ) -> torch.Tensor:
        return _rasterize(self, positions_A, radii_A, grid)


def gaussian_mask(positions_A: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Equalized Gaussian peaks (FWHM 0.2 A), normalized to [0, 1]."""
    return GaussianField().rasterize(positions_A, torch.empty(0), grid)  # radii unused


def circular_mask(positions_A: torch.Tensor, radii_A: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Binary union of disks, one per column, radius radii_A[k]."""
    return CircularField().rasterize(positions_A, radii_A, grid)


def column_radius(cond: ImagingCondition) -> float:
    """Column radius r in A (brief Section 1.5, corrected)."""
    lam = wavelength_A(cond.energy_keV)
    alpha = cond.alpha_max_mrad * 1e-3
    return 0.5 * math.sqrt((lam / alpha) ** 2 + (cond.source_size_A / 2.0) ** 2)
