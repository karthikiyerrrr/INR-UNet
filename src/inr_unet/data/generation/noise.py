"""Acquisition noise: per-row scan warp (grid_sample) and Poisson shot noise."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from inr_unet.data.generation.structures import Grid, ImagingCondition, NoiseSpec


def apply_scan_noise(
    image: torch.Tensor,
    noise: NoiseSpec,
    cond: ImagingCondition,
    grid: Grid,
    generator: torch.Generator,
) -> torch.Tensor:
    """Geometric per-row beam displacement, constant across each horizontal row."""
    size = grid.output_size
    device = grid.device
    rows = torch.arange(size, device=device, dtype=torch.float32)
    a = torch.randn(size, generator=generator, device=device)  # per-row i.i.d.
    sigma_px = cond.sigma_jitter_A / grid.pixel_size_A
    phase = 2.0 * math.pi * noise.scan_freq_cyc_per_row * rows
    dx = a * sigma_px * torch.sin(phase)
    dy = noise.scan_beta * a * sigma_px * torch.sin(phase + noise.scan_phi0)

    base_y, base_x = grid.pixel_coords()
    src_x = base_x - dx[:, None]
    src_y = base_y - dy[:, None]
    gx = 2.0 * src_x / (size - 1) - 1.0
    gy = 2.0 * src_y / (size - 1) - 1.0
    samp = torch.stack([gx, gy], dim=-1)[None]  # [1, H, W, 2] (x, y)
    out = F.grid_sample(
        image[None, None], samp, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return out[0, 0]


def apply_poisson(
    image: torch.Tensor, noise: NoiseSpec, generator: torch.Generator
) -> torch.Tensor:
    """Dose-dependent shot noise; returns counts renormalized by n_peak."""
    img = image - image.min()
    mx = img.max()
    img = img / mx if mx > 0 else img
    n_bg = noise.n_bg_frac * noise.n_peak
    lam = n_bg + (noise.n_peak - n_bg) * img
    counts = torch.poisson(lam, generator=generator)
    return counts / noise.n_peak
