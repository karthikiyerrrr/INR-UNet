"""Calibration statistics: intensity histogram and radial power spectrum."""

import math

import torch

from inr_unet.data.generation.calibration import (
    intensity_histogram,
    radial_power_spectrum,
)


def test_intensity_histogram_counts_and_range():
    img = torch.rand(32, 48)
    counts, edges = intensity_histogram(img, bins=16)
    assert counts.shape == (16,)
    assert edges.shape == (17,)
    assert int(counts.sum()) == img.numel()
    assert math.isclose(float(edges[0]), float(img.min()), rel_tol=0, abs_tol=1e-6)
    assert math.isclose(float(edges[-1]), float(img.max()), rel_tol=0, abs_tol=1e-6)


def test_radial_power_spectrum_sinusoid_peaks_at_its_frequency():
    n, k = 64, 8
    ax = torch.arange(n, dtype=torch.float32)
    xx = ax[None, :].expand(n, n)
    img = torch.cos(2.0 * math.pi * k * xx / n)  # k cycles across the width
    freq, power = radial_power_spectrum(img)
    peak_bin = int(torch.argmax(power[1:])) + 1
    assert abs(peak_bin - k) <= 1


def test_radial_power_spectrum_flat_concentrates_at_dc():
    img = torch.full((40, 40), 0.37)
    freq, power = radial_power_spectrum(img)
    assert int(torch.argmax(power)) == 0
    assert float(power[0]) > float(power[1:].max()) * 10.0
