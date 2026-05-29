"""Scan-warp geometry and Poisson dose statistics."""

import math

import pytest
import torch

from inr_unet.data.generation.noise import apply_poisson, apply_scan_noise
from inr_unet.data.generation.structures import Grid, ImagingCondition, NoiseSpec


def test_scan_noise_zero_jitter_is_identity():
    grid = Grid(output_size=32, pixel_size_A=0.1)
    cond = ImagingCondition(200.0, 24.0, 0.9, sigma_jitter_A=0.0, name="t")
    img = torch.rand(32, 32)
    gen = torch.Generator().manual_seed(0)
    out = apply_scan_noise(img, NoiseSpec(), cond, grid, gen)
    assert torch.allclose(out, img, atol=1e-4)


def test_scan_noise_deterministic_given_seed():
    grid = Grid(output_size=32, pixel_size_A=0.1)
    cond = ImagingCondition(200.0, 24.0, 0.9, name="t")
    img = torch.rand(32, 32)
    a = apply_scan_noise(img, NoiseSpec(), cond, grid, torch.Generator().manual_seed(5))
    b = apply_scan_noise(img, NoiseSpec(), cond, grid, torch.Generator().manual_seed(5))
    assert torch.allclose(a, b)


def test_poisson_peak_snr_matches_sqrt_npeak():
    # Peak-intensity field -> Poisson peak-SNR = mean/std ~ sqrt(n_peak).
    # Anchor the min in one corner so [0,1] normalization maps the bulk to peak = 1
    # (a perfectly uniform field has an undefined [0,1] normalization).
    img = torch.ones(200, 200)
    img[0, 0] = 0.0
    noise = NoiseSpec(n_peak=144.0, n_bg_frac=0.0)
    out = apply_poisson(img, noise, torch.Generator().manual_seed(0))
    bulk = out[1:]  # exclude the anchored corner row; bulk is all at peak dose
    snr = bulk.mean() / bulk.std()
    assert snr.item() == pytest.approx(math.sqrt(144.0), rel=0.1)
