"""Background families: registration, shape, amplitude scaling, determinism."""

import torch

from inr_unet.data.generation.background import constant_bg, linear_ramp, nonlinear_bg
from inr_unet.data.generation.structures import Grid
from inr_unet.registry import BACKGROUNDS


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_families_registered():
    assert {"constant", "linear_ramp", "nonlinear"} <= set(BACKGROUNDS.keys())


def test_constant_scales_with_signal():
    grid = Grid(output_size=16, pixel_size_A=0.1)
    bg = constant_bg(grid, {"c": 0.25}, signal_scale=8.0, generator=_gen())
    assert bg.shape == (16, 16)
    assert torch.allclose(bg, torch.full((16, 16), 2.0))  # 0.25 * 8.0


def test_linear_ramp_deterministic_given_seed():
    grid = Grid(output_size=16, pixel_size_A=0.1)
    a = linear_ramp(grid, {}, signal_scale=5.0, generator=_gen(3))
    b = linear_ramp(grid, {}, signal_scale=5.0, generator=_gen(3))
    assert torch.allclose(a, b)
    assert a.shape == (16, 16)


def test_nonlinear_capped_and_nonnegative():
    grid = Grid(output_size=32, pixel_size_A=0.1)
    bg = nonlinear_bg(grid, {"n_blobs": 4}, signal_scale=10.0, generator=_gen(1))
    assert bg.shape == (32, 32)
    assert (bg >= 0).all()
    assert bg.max() <= 0.8 * 10.0 + 1e-5
