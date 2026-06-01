"""SyntheticRenderSource: deterministic idx -> RenderedSample with fixed-size crop/pad."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from inr_unet.config import ExperimentConfig
from inr_unet.data.generation.structures import RenderMeta, RenderOutput
from inr_unet.data.render_source import RenderedSample, SyntheticRenderSource, _reflect_pad_to


def _cfg(crop_size=48, n_scenes=2, draws_per_scene=2):
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.synthetic.crop_size = crop_size
    cfg.data.synthetic.n_scenes = n_scenes
    cfg.data.synthetic.draws_per_scene = draws_per_scene
    # keep renders modest for test speed
    cfg.generation.sampler.pixel_size_A_min = 0.20
    cfg.generation.sampler.pixel_size_A_max = 0.30
    return cfg


def test_len_is_scenes_times_draws():
    src = SyntheticRenderSource(_cfg(n_scenes=3, draws_per_scene=4))
    assert len(src) == 12


def test_get_is_deterministic():
    src = SyntheticRenderSource(_cfg())
    a, b = src.get(1), src.get(1)
    assert torch.equal(a.image, b.image)
    assert torch.equal(a.positions_A, b.positions_A)
    assert a.input_pixel_size_A == b.input_pixel_size_A
    assert a.valid_extent_A == b.valid_extent_A


def test_image_is_crop_size_square():
    s = 48
    src = SyntheticRenderSource(_cfg(crop_size=s))
    for idx in range(len(src)):
        assert src.get(idx).image.shape == (s, s)


def test_positions_within_valid_window():
    src = SyntheticRenderSource(_cfg())
    s = src.get(0)
    margin = 6.0
    if s.positions_A.shape[0] > 0:
        assert float(s.positions_A.min()) >= -margin - 1e-4
        assert float(s.positions_A.max()) <= s.valid_extent_A + margin + 1e-4


def test_out_of_range_raises():
    src = SyntheticRenderSource(_cfg(n_scenes=1, draws_per_scene=1))
    with pytest.raises(IndexError):
        src.get(1)


def test_crop_pad_branch_for_small_render():
    """A render smaller than S takes the reflect-pad path; valid_extent = H*px."""
    src = SyntheticRenderSource(_cfg(crop_size=48))
    h, px = 20, 0.25
    img = torch.arange(h * h, dtype=torch.float32).reshape(h, h) / (h * h)
    fake = RenderOutput(
        image=img,
        gaussian_mask=torch.zeros(h, h),
        circular_mask=torch.zeros(h, h),
        no_noise=img,
        no_background_no_noise=img,
        positions_A=torch.tensor([[1.0, 1.0], [3.0, 2.0]]),
        radii_A=torch.tensor([0.3, 0.3]),
        meta=RenderMeta(pixel_size_A=px, output_size=h, condition_name="cond1"),
    )
    sample = src._crop(fake, idx=0)
    assert sample.image.shape == (48, 48)
    assert sample.valid_extent_A == pytest.approx(h * px)
    # top-left h x h block is the original render untouched
    assert torch.allclose(sample.image[:h, :h], img)
    # positions unchanged in the pad branch
    assert torch.equal(sample.positions_A, fake.positions_A)


def test_crop_branch_shifts_positions():
    """A render larger than S takes the crop path; positions become crop-local."""
    src = SyntheticRenderSource(_cfg(crop_size=16))
    h, px = 64, 0.1
    img = torch.zeros(h, h)
    fake = RenderOutput(
        image=img, gaussian_mask=img, circular_mask=img, no_noise=img,
        no_background_no_noise=img,
        positions_A=torch.tensor([[3.0, 3.0]]),
        radii_A=torch.tensor([0.3]),
        meta=RenderMeta(pixel_size_A=px, output_size=h, condition_name="cond1"),
    )
    sample = src._crop(fake, idx=5)
    assert sample.image.shape == (16, 16)
    assert sample.valid_extent_A == pytest.approx(16 * px)
    # deterministic crop offset for (master_seed=0, idx=5)
    rng = np.random.default_rng(np.random.SeedSequence([0, 5, 2017]))
    oy = int(rng.integers(0, h - 16 + 1))
    ox = int(rng.integers(0, h - 16 + 1))
    expected = torch.tensor([[3.0 - ox * px, 3.0 - oy * px]])
    # column survives the keep-filter only if within the window; assert mapping is correct when kept
    if sample.positions_A.shape[0] == 1:
        assert torch.allclose(sample.positions_A, expected, atol=1e-5)


def test_reflect_pad_to_reflect_branch_keeps_render_top_left():
    h, size = 40, 48  # pad = 8 < h, so the reflect branch (not replicate) is taken
    img = torch.arange(h * h, dtype=torch.float32).reshape(h, h)
    out = _reflect_pad_to(img, size)
    assert out.shape == (size, size)
    # render preserved untouched at the top-left
    assert torch.equal(out[:h, :h], img)
    # bottom/right padding is a reflection of the interior (row h is a reflection of row h-2)
    assert torch.equal(out[h, :h], img[h - 2, :])
    assert torch.equal(out[:h, h], img[:, h - 2])


def test_returns_rendered_sample_type():
    src = SyntheticRenderSource(_cfg())
    assert isinstance(src.get(0), RenderedSample)


def test_redraw_disabled_when_occupancy_full_reproduces_offsets():
    # With occupancy.mode == "full" the redraw path must not engage; results stay deterministic.
    from omegaconf import OmegaConf

    from inr_unet.config import ExperimentConfig
    from inr_unet.data.render_source import SyntheticRenderSource

    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.synthetic.n_scenes = 2
    cfg.data.synthetic.draws_per_scene = 4
    src = SyntheticRenderSource(cfg)
    assert src.redraw_enabled is False
    a = src.get(3)
    b = SyntheticRenderSource(cfg).get(3)
    import torch

    assert torch.equal(a.image, b.image)


def test_redraw_enabled_when_occupancy_active():
    from omegaconf import OmegaConf

    from inr_unet.config import ExperimentConfig
    from inr_unet.data.render_source import SyntheticRenderSource

    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.occupancy.mode = "facet_polygon"
    src = SyntheticRenderSource(cfg)
    assert src.redraw_enabled is True


def test_cif_provider_end_to_end_sample():
    from omegaconf import OmegaConf

    from inr_unet.config import ExperimentConfig
    from inr_unet.data.render_source import SyntheticRenderSource

    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.provider = "cif"
    cfg.data.synthetic.n_scenes = 2
    cfg.data.synthetic.draws_per_scene = 2
    src = SyntheticRenderSource(cfg)
    s = src.get(0)
    assert s.image.shape == (cfg.data.synthetic.crop_size, cfg.data.synthetic.crop_size)


def test_occupancy_wrapper_applied_when_active():
    from omegaconf import OmegaConf

    from inr_unet.config import ExperimentConfig
    from inr_unet.data.occupancy import FiniteSupportProvider
    from inr_unet.data.render_source import SyntheticRenderSource

    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.occupancy.mode = "blob"
    src = SyntheticRenderSource(cfg)
    assert isinstance(src.provider, FiniteSupportProvider)


def test_synthetic_full_provider_unwrapped():
    from omegaconf import OmegaConf

    from inr_unet.config import ExperimentConfig
    from inr_unet.data.providers import SyntheticLatticeProvider
    from inr_unet.data.render_source import SyntheticRenderSource

    cfg = OmegaConf.structured(ExperimentConfig)
    src = SyntheticRenderSource(cfg)
    assert isinstance(src.provider, SyntheticLatticeProvider)
