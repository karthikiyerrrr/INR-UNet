"""AugmentationSampler: determinism, axis ranges, constraints, renderer round-trip."""

import math

from omegaconf import OmegaConf

from inr_unet.config import SamplerConfig
from inr_unet.data.generation.sampler import AugmentationSampler
from inr_unet.data.generation.structures import ImagingCondition, NoiseSpec


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
