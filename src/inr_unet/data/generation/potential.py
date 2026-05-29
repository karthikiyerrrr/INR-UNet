"""Projected-potential (sigma) builders. Default: Z^n-weighted Gaussian splat."""

from __future__ import annotations

import math

import torch

from inr_unet.data.generation.structures import ColumnList, Grid, RenderParams
from inr_unet.registry import POTENTIALS


@POTENTIALS.register("z_power")
def build_potential(
    columns: ColumnList,
    params: RenderParams,
    grid: Grid,
    *,
    sigma_phys_A: float,
) -> torch.Tensor:
    """Sigma [H, W]: each column an analytic Gaussian of fixed physical width,
    weighted by count * Z**z_exponent, splatted at its sub-pixel position."""
    device = grid.device
    out = torch.zeros((grid.output_size, grid.output_size), device=device)
    n = columns.positions_A.shape[0]
    if n == 0:
        return out

    pos = columns.positions_A.to(device)
    weights = columns.count.to(device) * columns.z.to(device) ** params.z_exponent
    sigma_px = sigma_phys_A / grid.pixel_size_A
    norm = 1.0 / (2.0 * math.pi * sigma_px**2)
    rad = max(1, int(math.ceil(4.0 * sigma_px)))
    size = grid.output_size

    px = pos[:, 0] / grid.pixel_size_A  # column (x)
    py = pos[:, 1] / grid.pixel_size_A  # row (y)
    for i in range(n):
        cx, cy = float(px[i]), float(py[i])
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, ix - rad), min(size, ix + rad + 1)
        y0, y1 = max(0, iy - rad), min(size, iy + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = torch.arange(x0, x1, device=device, dtype=torch.float32)
        ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
        gx = torch.exp(-((xs - cx) ** 2) / (2.0 * sigma_px**2))
        gy = torch.exp(-((ys - cy) ** 2) / (2.0 * sigma_px**2))
        out[y0:y1, x0:x1] += weights[i] * norm * torch.outer(gy, gx)
    return out
