"""Label geometry: radius formula, circular-mask area, equalized Gaussian mask."""

import math

import pytest
import torch

from inr_unet.data.generation.labels import circular_mask, column_radius, gaussian_mask
from inr_unet.data.generation.structures import Grid, ImagingCondition


def test_column_radius_formula():
    cond = ImagingCondition(200.0, 24.0, 0.9, name="t")
    lam = 0.025079
    alpha = 24.0e-3
    expected = 0.5 * math.sqrt((lam / alpha) ** 2 + (0.9 / 2.0) ** 2)
    assert column_radius(cond) == pytest.approx(expected, rel=1e-3)


def test_circular_mask_area():
    grid = Grid(output_size=128, pixel_size_A=0.05)  # 6.4 A FOV
    pos = torch.tensor([[3.2, 3.2]])
    r_A = 1.0
    radii = torch.tensor([r_A])
    mask = circular_mask(pos, radii, grid)
    area_px = mask.sum().item()
    expected_px = math.pi * (r_A / grid.pixel_size_A) ** 2
    assert abs(area_px - expected_px) / expected_px < 0.05


def test_gaussian_mask_equalized_across_columns():
    grid = Grid(output_size=64, pixel_size_A=0.05)
    pos = torch.tensor([[1.0, 1.0], [2.4, 2.4]])
    mask = gaussian_mask(pos, grid)
    assert mask.max().item() == pytest.approx(1.0, abs=1e-5)
    # both columns reach the same peak (equal weights), so two distinct near-1 peaks
    peaks = mask[mask > 0.99]
    assert peaks.numel() >= 2
