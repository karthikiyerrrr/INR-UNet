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
    assert 12.0 - 1e-6 <= s.valid_extent_A <= 20.0 + 0.25  # within [min, max] (+ 1px rounding)
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
    cfg.data.cache_scenes = False  # disable cache so .provider is the raw inner provider
    assert isinstance(SyntheticRenderSource(cfg).provider, FiniteSupportProvider)


def test_synthetic_full_provider_unwrapped():
    from inr_unet.data.providers import SyntheticLatticeProvider

    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.cache_scenes = False  # disable cache so .provider is the raw inner provider
    assert isinstance(SyntheticRenderSource(cfg).provider, SyntheticLatticeProvider)


def test_partial_scene_zeros_render_offset():
    """For a partial scene, the offset reaching the renderer is forced to zero (centers the
    window on the patch), regardless of the jitter the sampler drew."""
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.provider = "cif"
    cfg.data.occupancy.mode = "full"
    cfg.data.cif.partial_fov_prob = 1.0
    cfg.data.synthetic.n_scenes = 2
    cfg.data.synthetic.draws_per_scene = 2
    cfg.data.synthetic.empty_crop_fraction = 0.0  # force frame draws so the offset is zeroed
    src = SyntheticRenderSource(cfg)

    captured = {"offset": None}
    real_render = src.renderer.render

    def spy(scene, condition, params):
        captured["offset"] = params.position_offset_A.clone()
        return real_render(scene, condition, params)

    src.renderer.render = spy
    sample = src.get(0)
    assert sample.positions_A.shape[0] > 0  # partial patch is framed (precondition)
    assert captured["offset"] is not None, "renderer was never called with a partial scene"
    assert torch.equal(captured["offset"], torch.zeros(2))


def test_full_scene_keeps_sampler_offset():
    """A full (non-partial) scene must NOT have its offset zeroed; the guard leaves it alone."""
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.provider = "cif"
    cfg.data.occupancy.mode = "full"
    cfg.data.cif.partial_fov_prob = 0.0  # all full scenes
    cfg.data.synthetic.n_scenes = 2
    cfg.data.synthetic.draws_per_scene = 2
    src = SyntheticRenderSource(cfg)

    captured = {"offset": None}
    real_render = src.renderer.render

    def spy(scene, condition, params):
        assert scene.is_partial is False  # precondition: this scene is full
        captured["offset"] = params.position_offset_A.clone()
        return real_render(scene, condition, params)

    src.renderer.render = spy
    src.get(0)
    # the offset reaching the renderer is whatever the sampler drew (the guard did not touch it)
    _, expected = src.sampler.sample(0, max_fov_A=src.provider.get(0).fov_A)
    assert captured["offset"] is not None
    assert torch.equal(captured["offset"], expected.position_offset_A)


def test_partial_scenes_are_framed_not_empty():
    """Partial-FOV tiles hold columns at >= 70%; the centered-patch + zeroed-offset fix
    maintains this floor end-to-end (provider -> sampler -> render -> crop)."""
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.provider = "cif"
    cfg.data.occupancy.mode = "full"
    cfg.data.cif.partial_fov_prob = 1.0
    cfg.data.synthetic.n_scenes = 3
    cfg.data.synthetic.draws_per_scene = 6
    src = SyntheticRenderSource(cfg)
    nonempty = sum(int(src.get(i).positions_A.shape[0] > 0) for i in range(len(src)))
    assert nonempty / len(src) >= 0.7  # floor leaves headroom for empty_crop_fraction misses


def test_empty_crop_fraction_controls_partial_empties():
    """empty_crop_fraction reliably produces vacuum (background) tiles from partial scenes by
    aiming the render into vacuum; 0.0 yields none (patches stay framed)."""
    def empty_rate(ecf):
        cfg = OmegaConf.structured(ExperimentConfig)
        cfg.data.provider = "cif"
        cfg.data.occupancy.mode = "full"
        cfg.data.cif.partial_fov_prob = 1.0
        cfg.data.synthetic.empty_crop_fraction = ecf
        cfg.data.synthetic.n_scenes = 4
        cfg.data.synthetic.draws_per_scene = 6
        src = SyntheticRenderSource(cfg)
        empt = sum(int(src.get(i).positions_A.shape[0] == 0) for i in range(len(src)))
        return empt / len(src)

    assert empty_rate(0.0) == 0.0    # no background draws -> centered patches always framed
    assert empty_rate(0.5) >= 0.25   # background draws reliably materialize as vacuum tiles


def test_partial_background_draw_aims_render_into_vacuum():
    """With empty_crop_fraction=1.0 every partial draw is a background draw: the render offset is
    pushed off the patch and the resulting tile is empty (all-zero label)."""
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.provider = "cif"
    cfg.data.occupancy.mode = "full"
    cfg.data.cif.partial_fov_prob = 1.0
    cfg.data.synthetic.empty_crop_fraction = 1.0
    cfg.data.synthetic.n_scenes = 2
    cfg.data.synthetic.draws_per_scene = 2
    src = SyntheticRenderSource(cfg)

    captured = {"offset": None}
    real_render = src.renderer.render

    def spy(scene, condition, params):
        captured["offset"] = params.position_offset_A.clone()
        return real_render(scene, condition, params)

    src.renderer.render = spy
    sample = src.get(0)
    assert captured["offset"] is not None
    assert float(captured["offset"].abs().max()) > 0.0  # aimed off the patch
    assert sample.positions_A.shape[0] == 0              # tile is pure background
    assert torch.isfinite(sample.image).all()
    assert float(sample.image.min()) >= 0.0 and float(sample.image.max()) <= 1.0
