"""AugmentationSampler: determinism, axis ranges, constraints, renderer round-trip."""

import math

import pytest
from omegaconf import OmegaConf

from inr_unet.config import SamplerConfig
from inr_unet.data.generation.psf import build_psf
from inr_unet.data.generation.sampler import AugmentationSampler
from inr_unet.data.generation.structures import (
    IMAGING_CONDITIONS,
    Grid,
    ImagingCondition,
    NoiseSpec,
)


def _sampler(master_seed=123):
    cfg = OmegaConf.structured(SamplerConfig)
    return AugmentationSampler(cfg, master_seed=master_seed)


def test_seed_streams_deterministic_and_distinct():
    s = _sampler()
    rng_a, seed_a = s._streams(5)
    rng_b, seed_b = s._streams(5)
    rng_c, seed_c = s._streams(6)
    # same index -> same derived render seed and same first draw
    assert seed_a == seed_b
    assert rng_a.random() == rng_b.random()
    # different index -> different render seed
    assert seed_a != seed_c
    # different master seed -> different render seed for the same index
    assert _sampler(master_seed=999)._streams(5)[1] != seed_a


def test_draw_condition_is_a_preset():
    s = _sampler()
    rng, _ = s._streams(0)
    seen = set()
    for _ in range(100):
        cond = s._draw_condition(rng)
        assert isinstance(cond, ImagingCondition)
        seen.add(cond.name)
    assert seen == {"cond1", "cond2", "cond3", "cond4", "cond5"}


def test_draw_noise_within_ranges():
    s = _sampler()
    rng, _ = s._streams(0)
    for _ in range(100):
        n = s._draw_noise(rng)
        assert isinstance(n, NoiseSpec)
        assert 30.0 <= n.n_peak <= 3000.0
        assert 0.0 <= n.n_bg_frac <= 0.05
        assert 0.02 <= n.scan_freq_cyc_per_row <= 0.5
        assert 0.3 <= n.scan_beta <= 0.5
        assert 0.0 <= n.scan_phi0 <= 2.0 * math.pi


def test_draw_background_families_and_params():
    s = _sampler()
    rng, _ = s._streams(0)
    seen = set()
    for _ in range(300):
        bg = s._draw_background(rng)
        seen.add(bg.kind)
        if bg.kind == "constant":
            assert 0.05 <= bg.params["c"] <= 0.40
        elif bg.kind == "linear_ramp":
            assert 0.05 <= bg.params["c0"] <= 0.30
            assert 0.10 <= bg.params["g"] <= 0.60
        elif bg.kind == "nonlinear":
            assert 2 <= bg.params["n_blobs"] <= 5
    assert seen == {"constant", "linear_ramp", "nonlinear"}


def test_rotation_choices_filtered_to_feasible():
    s = _sampler()
    rng, _ = s._streams(0)
    # max_fov 30 with smallest fov 8: 8*(|cos|+|sin|) + 12 <= 30 -> factor <= 2.25.
    # All rotations 0..90 satisfy this (max factor ~1.414), so all are allowed.
    rots = {s._draw_rotation(rng, 30.0) for _ in range(200)}
    assert rots <= {0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0}
    assert 0.0 in rots


def test_too_small_structure_raises():
    s = _sampler()
    rng, _ = s._streams(0)
    # smallest fov 8: even at 0 deg, 8*1 + 12 = 20 > 10 -> no feasible rotation.
    with pytest.raises(ValueError):
        s._draw_rotation(rng, 10.0)


def test_draw_scale_respects_aliasing_and_fov():
    s = _sampler()
    rng, _ = s._streams(0)
    cond = IMAGING_CONDITIONS["cond1"]
    for _ in range(100):
        fov_A, pixel_size_A, output_size = s._draw_scale(rng, cond, 0.0, 80.0)
        assert fov_A in list(s.cfg.fov_set_A)
        assert 0.05 <= pixel_size_A <= 0.30
        # rendered FOV never exceeds sampled FOV (floor), and aliasing guard holds
        assert output_size * pixel_size_A <= fov_A + 1e-9
        build_psf(cond, Grid(output_size, pixel_size_A))  # raises if k_cut >= k_Nyq


def test_draw_offset_within_slack():
    s = _sampler()
    rng, _ = s._streams(0)
    fov_A, rotation_deg, max_fov_A = 10.0, 30.0, 40.0
    factor = abs(math.cos(math.radians(rotation_deg))) + abs(math.sin(math.radians(rotation_deg)))
    slack = max((max_fov_A - fov_A * factor) / 2.0 - 6.0, 0.0)
    bound = 0.5 * slack  # offset_jitter_frac default 0.5
    for _ in range(100):
        off = s._draw_offset(rng, fov_A, rotation_deg, max_fov_A)
        assert off.shape == (2,)
        assert float(off.abs().max()) <= bound + 1e-6


def test_draw_scale_raises_when_no_fov_fits():
    s = _sampler()
    rng, _ = s._streams(0)
    cond = IMAGING_CONDITIONS["cond1"]
    # at 0 deg even the smallest FOV (8) needs 8 + 2*6 = 20 A > 11 -> no candidate
    with pytest.raises(ValueError):
        s._draw_scale(rng, cond, 0.0, 11.0)


def test_draw_scale_raises_when_aliasing_below_min_pixel():
    cfg = OmegaConf.structured(SamplerConfig)
    cfg.pixel_size_A_min = 0.5  # coarser than any condition's aliasing limit (~0.35 A)
    cfg.pixel_size_A_max = 0.6
    s = AugmentationSampler(cfg, master_seed=0)
    rng, _ = s._streams(0)
    with pytest.raises(ValueError):
        s._draw_scale(rng, IMAGING_CONDITIONS["cond1"], 0.0, 80.0)
