"""PSF optics: wavelength, Airy probe at zero aberration, Nyquist guard, convolution."""

import math

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


def _psf_principal(psf: torch.Tensor, pixel_size_A: float):
    """Return (axis_ratio, major_axis_angle_deg in [0,180)) from the PSF 2nd moments."""
    n = psf.shape[0]
    ax = (torch.arange(n, dtype=torch.float32) - n // 2) * pixel_size_A
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    cy = float((psf * yy).sum())
    cx = float((psf * xx).sum())
    dy, dx = yy - cy, xx - cx
    mxx = float((psf * dx * dx).sum())
    myy = float((psf * dy * dy).sum())
    mxy = float((psf * dx * dy).sum())
    m = torch.tensor([[mxx, mxy], [mxy, myy]])
    evals, evecs = torch.linalg.eigh(m)
    major = evecs[:, -1]  # eigenvector of the larger eigenvalue: (x, y) components
    angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    ratio = float((evals[-1] / evals[0]) ** 0.5)
    return ratio, angle


def test_psf_byte_identical_when_astig_zero():
    # Adding the (zeroed) A1 term must not perturb the radial-only PSF, even with defocus.
    grid = Grid(output_size=96, pixel_size_A=0.15)
    for defocus in (0.0, 37.0):
        radial = ImagingCondition(200.0, 24.0, 0.9, defocus_A=defocus, name="t")
        astig0 = ImagingCondition(
            200.0, 24.0, 0.9, defocus_A=defocus,
            astig_a1_A=0.0, astig_a1_azimuth_rad=1.0, name="t",
        )
        assert torch.equal(build_psf(radial, grid, soft=False),
                           build_psf(astig0, grid, soft=False))


def test_psf_astigmatism_elliptical_and_oriented():
    # Visible ellipticity needs BOTH defocus and A1; the major axis tracks the azimuth.
    grid = Grid(output_size=128, pixel_size_A=0.15)

    pure_defocus = ImagingCondition(200.0, 24.0, 0.9, defocus_A=50.0, name="t")
    ratio_df, _ = _psf_principal(build_psf(pure_defocus, grid, soft=False), 0.15)
    assert ratio_df < 1.05  # pure defocus stays isotropic

    horiz = ImagingCondition(200.0, 24.0, 0.9, defocus_A=50.0,
                             astig_a1_A=50.0, astig_a1_azimuth_rad=0.0, name="t")
    vert = ImagingCondition(200.0, 24.0, 0.9, defocus_A=50.0,
                            astig_a1_A=50.0, astig_a1_azimuth_rad=math.pi / 2, name="t")
    r_h, ang_h = _psf_principal(build_psf(horiz, grid, soft=False), 0.15)
    r_v, ang_v = _psf_principal(build_psf(vert, grid, soft=False), 0.15)

    assert r_h > 1.15 and r_v > 1.15
    assert min(ang_h, 180.0 - ang_h) < 20.0   # azimuth 0 -> major axis ~0 deg
    assert abs(ang_v - 90.0) < 20.0           # azimuth pi/2 -> major axis ~90 deg
