"""Persistent render cache: keying, materialization, ragged reconstruction, alignment."""

import pytest  # noqa: F401
import torch  # noqa: F401
from omegaconf import OmegaConf

from inr_unet.config import ExperimentConfig
from inr_unet.data.cache import CachedRenderSource, RenderCache, cache_key  # noqa: F401
from inr_unet.data.dataset import LIIFSegDataset, STEMSegDataset  # noqa: F401
from inr_unet.data.render_source import SyntheticRenderSource  # noqa: F401


def _cfg(crop_size=32, n_scenes=2, draws_per_scene=2):
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.num_workers = 0            # single-process build keeps tests fast + fork-free
    cfg.data.synthetic.crop_size = crop_size
    cfg.data.synthetic.n_scenes = n_scenes
    cfg.data.synthetic.draws_per_scene = draws_per_scene
    cfg.generation.sampler.pixel_size_A_min = 0.20   # modest renders for test speed
    cfg.generation.sampler.pixel_size_A_max = 0.30
    return cfg


def test_key_excludes_label_and_query_knobs():
    a, b = _cfg(), _cfg()
    b.data.synthetic.label_kind = "circular"
    b.data.synthetic.sample_q = 128
    b.data.synthetic.target_pixel_size_A_max = 0.5
    b.data.synthetic.gaussian_fwhm_A = 0.9
    assert cache_key(a) == cache_key(b)


def test_key_sensitive_to_render_config():
    a, b = _cfg(), _cfg()
    b.data.synthetic.tile_fov_A_max = 99.0
    assert cache_key(a) != cache_key(b)


def test_key_sensitive_to_split():
    a, b = _cfg(), _cfg()
    b.train.split.val_frac = 0.25
    assert cache_key(a) != cache_key(b)
