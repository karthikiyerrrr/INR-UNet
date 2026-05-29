"""Low-frequency background families, amplitudes expressed as fractions of signal."""

from __future__ import annotations

import math

import torch

from inr_unet.data.generation.structures import Grid
from inr_unet.registry import BACKGROUNDS


def _rand(generator: torch.Generator, device: str) -> torch.Tensor:
    return torch.rand(1, generator=generator, device=device)


@BACKGROUNDS.register("constant")
def constant_bg(grid: Grid, params: dict, signal_scale: float, generator: torch.Generator) -> torch.Tensor:
    c = float(params.get("c", 0.2))
    return torch.full((grid.output_size, grid.output_size), c * signal_scale, device=grid.device)


@BACKGROUNDS.register("linear_ramp")
def linear_ramp(grid: Grid, params: dict, signal_scale: float, generator: torch.Generator) -> torch.Tensor:
    c0 = float(params.get("c0", 0.15))
    g = float(params.get("g", 0.3))
    theta = _rand(generator, grid.device) * 2.0 * math.pi
    yy, xx = grid.normalized_coords()
    ramp = c0 + (g / 2.0) * (xx * torch.cos(theta) + yy * torch.sin(theta))
    return ramp * signal_scale


@BACKGROUNDS.register("nonlinear")
def nonlinear_bg(grid: Grid, params: dict, signal_scale: float, generator: torch.Generator) -> torch.Tensor:
    n_blobs = int(params.get("n_blobs", 3))
    yy, xx = grid.normalized_coords()
    bg = torch.zeros_like(xx)
    for _ in range(n_blobs):
        cx = _rand(generator, grid.device) * 2.0 - 1.0
        cy = _rand(generator, grid.device) * 2.0 - 1.0
        amp = 0.10 + 0.35 * _rand(generator, grid.device)
        width = 0.4 + 0.8 * _rand(generator, grid.device)
        bg = bg + amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * width**2))
    bg = torch.clamp(bg, max=0.8)
    return bg * signal_scale
