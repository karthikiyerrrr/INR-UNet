"""Sigma builder: Z^n weighting and scale-invariant physical width."""

import math

import torch

from inr_unet.data.generation.potential import build_potential
from inr_unet.data.generation.structures import ColumnList, Grid, RenderParams
from inr_unet.registry import POTENTIALS


def _one_column(x_A=2.5, y_A=2.5, z=78.0, count=1.0, fov_A=5.0):
    return ColumnList(
        positions_A=torch.tensor([[x_A, y_A]]),
        z=torch.tensor([float(z)]),
        count=torch.tensor([float(count)]),
        fov_A=fov_A,
    )


def test_z_power_registered():
    assert "z_power" in POTENTIALS


def test_weight_scales_with_count_and_z():
    grid = Grid(output_size=64, pixel_size_A=0.1)
    params = RenderParams(output_size=64, pixel_size_A=0.1, z_exponent=2.0)
    light = build_potential(_one_column(z=8.0, count=1.0), params, grid, sigma_phys_A=0.4)
    heavy = build_potential(_one_column(z=16.0, count=1.0), params, grid, sigma_phys_A=0.4)
    # total integrated weight ratio == (16/8)^2 == 4
    assert (heavy.sum() / light.sum()).item() == math.pow(2.0, 2.0)


def test_physical_width_is_scale_invariant():
    # Same physical splat width (A) must give the same physical FWHM at two pixel sizes.
    # FWHM is measured with sub-pixel linear interpolation of the half-max crossings;
    # integer-pixel counting quantizes too coarsely to compare across resolutions.
    def fwhm_A(pixel_size_A):
        n = 128
        grid = Grid(output_size=n, pixel_size_A=pixel_size_A)
        params = RenderParams(output_size=n, pixel_size_A=pixel_size_A)
        c = _one_column(x_A=n * pixel_size_A / 2, y_A=n * pixel_size_A / 2, fov_A=n * pixel_size_A)
        img = build_potential(c, params, grid, sigma_phys_A=0.6)
        row = img[img.shape[0] // 2]
        half = (row.max() / 2).item()
        above = (row >= half).nonzero().flatten()
        lo, hi = int(above[0]), int(above[-1])
        left = lo - (row[lo].item() - half) / (row[lo].item() - row[lo - 1].item())
        right = hi + (row[hi].item() - half) / (row[hi].item() - row[hi + 1].item())
        return (right - left) * pixel_size_A

    analytic = 2.0 * math.sqrt(2.0 * math.log(2.0)) * 0.6  # FWHM = 2.3548 * sigma_phys
    fwhm_fine = fwhm_A(0.10)
    fwhm_coarse = fwhm_A(0.20)
    assert abs(fwhm_fine - fwhm_coarse) / fwhm_fine < 0.05
    assert abs(fwhm_fine - analytic) / analytic < 0.05
