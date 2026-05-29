"""PSF optics: wavelength, Airy probe at zero aberration, Nyquist guard, convolution."""

import pytest
import torch

from inr_unet.data.generation.psf import build_psf, fft_convolve, wavelength_A
from inr_unet.data.generation.structures import Grid, ImagingCondition


def test_wavelength_relativistic():
    assert wavelength_A(200.0) == pytest.approx(0.02508, abs=1e-4)
    assert wavelength_A(100.0) == pytest.approx(0.03701, abs=1e-4)


def test_psf_normalized_and_centered():
    cond = ImagingCondition(200.0, 24.0, 0.9, name="t")
    grid = Grid(output_size=64, pixel_size_A=0.2)
    psf = build_psf(cond, grid, soft=False)
    assert psf.shape == (64, 64)
    assert psf.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (psf >= 0).all()
    # peak at the center pixel for a centered PSF
    peak = torch.argmax(psf)
    assert (int(peak) // 64, int(peak) % 64) == (32, 32)


def test_psf_nyquist_guard():
    # 30 mrad at 200 keV needs pixel_size < ~0.42 A; 0.6 A must raise.
    cond = ImagingCondition(200.0, 30.0, 0.9, name="t")
    grid = Grid(output_size=64, pixel_size_A=0.6)
    with pytest.raises(ValueError, match="Nyquist"):
        build_psf(cond, grid)


def test_fft_convolve_delta_recovers_psf():
    cond = ImagingCondition(200.0, 24.0, 0.9, name="t")
    grid = Grid(output_size=64, pixel_size_A=0.2)
    psf = build_psf(cond, grid, soft=False)
    sigma = torch.zeros(64, 64)
    sigma[32, 32] = 1.0  # unit impulse at center
    out = fft_convolve(sigma, psf)
    assert torch.allclose(out, psf, atol=1e-5)
