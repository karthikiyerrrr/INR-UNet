"""SyntheticRenderSource: deterministic idx -> RenderedSample (physical-extent tile)."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from inr_unet.config import ExperimentConfig
from inr_unet.data.generation.structures import RenderMeta, RenderOutput
from inr_unet.data.render_source import RenderedSample, SyntheticRenderSource


def _cfg(crop_size=64, n_scenes=2, draws_per_scene=2, tile_min=12.0, tile_max=32.0):
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.synthetic.crop_size = crop_size
    cfg.data.synthetic.n_scenes = n_scenes
    cfg.data.synthetic.draws_per_scene = draws_per_scene
    cfg.data.synthetic.tile_fov_A_min = tile_min
    cfg.data.synthetic.tile_fov_A_max = tile_max
    # keep renders modest for test speed
    cfg.generation.sampler.pixel_size_A_min = 0.20
    cfg.generation.sampler.pixel_size_A_max = 0.30
    return cfg


def _fake_out(h, px, positions):
    img = torch.arange(h * h, dtype=torch.float32).reshape(h, h) / (h * h)
    return RenderOutput(
        image=img, gaussian_mask=torch.zeros(h, h), circular_mask=torch.zeros(h, h),
        no_noise=img, no_background_no_noise=img,
        positions_A=torch.as_tensor(positions, dtype=torch.float32),
        radii_A=torch.full((len(positions),), 0.3),
        meta=RenderMeta(pixel_size_A=px, output_size=h, condition_name="cond1"),
    )


def test_len_is_scenes_times_draws():
    assert len(SyntheticRenderSource(_cfg(n_scenes=3, draws_per_scene=4))) == 12


def test_get_is_deterministic():
    src = SyntheticRenderSource(_cfg())
    a, b = src.get(1), src.get(1)
    assert torch.equal(a.image, b.image)
    assert torch.equal(a.positions_A, b.positions_A)
    assert a.input_pixel_size_A == b.input_pixel_size_A
    assert a.valid_extent_A == b.valid_extent_A


def test_image_is_crop_size_square():
    s = 64
    src = SyntheticRenderSource(_cfg(crop_size=s))
    for idx in range(len(src)):
        assert src.get(idx).image.shape == (s, s)


def test_reported_fields_consistent():
    """valid_extent == tile window; input pixel == valid_extent / crop_size."""
    src = SyntheticRenderSource(_cfg(crop_size=64))
    s = src.get(0)
    assert s.input_pixel_size_A == pytest.approx(s.valid_extent_A / 64)


def test_tile_extent_within_bounds_for_large_render():
    src = SyntheticRenderSource(_cfg(crop_size=64, tile_min=12.0, tile_max=20.0))
    # h*px = 256*0.25 = 64 A render >> tile_max, so the tile is a bounded sub-window
    out = _fake_out(h=256, px=0.25, positions=[[10.0, 10.0]])
    s = src._crop(out, idx=0)
    assert s.image.shape == (64, 64)
    assert 12.0 - 1e-6 <= s.valid_extent_A <= 20.0 + 0.25  # within [min, max] (+ one-pixel rounding)
    assert s.input_pixel_size_A == pytest.approx(s.valid_extent_A / 64)


def test_small_render_uses_whole_field():
    """A render smaller than tile_min uses the whole field (extent clamped to render extent)."""
    src = SyntheticRenderSource(_cfg(crop_size=64, tile_min=12.0, tile_max=32.0))
    out = _fake_out(h=20, px=0.25, positions=[[1.0, 1.0]])  # render extent = 5 A << tile_min
    s = src._crop(out, idx=0)
    assert s.image.shape == (64, 64)
    assert s.valid_extent_A == pytest.approx(20 * 0.25)  # whole render


def test_crop_offset_shifts_positions():
    """Positions become tile-local: shifted by the chosen pixel offset times px."""
    cfg = _cfg(crop_size=32, tile_min=6.0, tile_max=6.0)
    cfg.data.synthetic.min_columns_in_crop = 0  # loop breaks on first draw -> single offset pair
    src = SyntheticRenderSource(cfg)
    px = 0.1
    out = _fake_out(h=128, px=px, positions=[[3.0, 3.0]])
    s = src._crop(out, idx=5)
    w = round(6.0 / px)
    rng = np.random.default_rng(np.random.SeedSequence([0, 5, 2017]))
    _ = float(rng.uniform(6.0, 6.0))               # tile_fov_A draw
    _ = rng.random() < src.empty_crop_fraction     # allow_empty draw
    oy = int(rng.integers(0, 128 - w + 1))
    ox = int(rng.integers(0, 128 - w + 1))
    expected_x, expected_y = 3.0 - ox * px, 3.0 - oy * px
    assert s.positions_A.shape[0] == 1, "column must survive keep-filter"
    assert torch.allclose(
        s.positions_A, torch.tensor([[expected_x, expected_y]]), atol=1e-4
    )


def test_returns_rendered_sample_type():
    assert isinstance(SyntheticRenderSource(_cfg()).get(0), RenderedSample)


def test_content_aware_crop_is_always_on():
    """No redraw_enabled gate: a partial CIF scene still gets column-seeking crops under full."""
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.provider = "cif"
    cfg.data.occupancy.mode = "full"
    cfg.data.cif.partial_fov_prob = 1.0
    cfg.data.synthetic.n_scenes = 2
    cfg.data.synthetic.draws_per_scene = 4
    src = SyntheticRenderSource(cfg)
    found = sum(int(src.get(i).positions_A.shape[0] > 0) for i in range(len(src)))
    assert found >= 1  # column-seeking offset lands on the patch at least once


def test_cif_provider_end_to_end_sample():
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.provider = "cif"
    cfg.data.synthetic.n_scenes = 2
    cfg.data.synthetic.draws_per_scene = 2
    src = SyntheticRenderSource(cfg)
    s = src.get(0)
    assert s.image.shape == (cfg.data.synthetic.crop_size, cfg.data.synthetic.crop_size)


def test_occupancy_wrapper_applied_when_active():
    from inr_unet.data.occupancy import FiniteSupportProvider

    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.occupancy.mode = "blob"
    assert isinstance(SyntheticRenderSource(cfg).provider, FiniteSupportProvider)


def test_synthetic_full_provider_unwrapped():
    from inr_unet.data.providers import SyntheticLatticeProvider

    cfg = OmegaConf.structured(ExperimentConfig)
    assert isinstance(SyntheticRenderSource(cfg).provider, SyntheticLatticeProvider)
