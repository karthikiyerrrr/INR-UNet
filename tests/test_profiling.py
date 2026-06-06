"""profile_datagen times datagen and projects an epoch ETA."""

import math

from inr_unet.config import load_config
from inr_unet.profiling import DatagenProfile, format_profile, profile_datagen
from inr_unet.train import build_splits

CONFIG = "configs/default.yaml"


def _small_cfg():
    cfg = load_config(CONFIG)
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 3
    cfg.data.synthetic.draws_per_scene = 2
    cfg.data.synthetic.sample_q = 64
    cfg.data.num_workers = 0
    cfg.model.encoder.base_channels = 8
    cfg.model.encoder.depth = 3
    cfg.model.encoder.feature_dim = 16
    cfg.model.decoder.hidden_dim = 32
    cfg.model.decoder.num_layers = 3
    return cfg


def test_profile_datagen_fields():
    cfg = _small_cfg()
    p = profile_datagen(cfg, n=4)
    assert isinstance(p, DatagenProfile)
    assert p.n_samples == 4
    assert p.num_workers == cfg.data.num_workers
    assert p.n_train == len(build_splits(cfg).train)
    for v in (p.cold_s, p.warm_mean_s, p.warm_median_s, p.projection_mean_s,
              p.render_crop_mean_s, p.est_secs_per_epoch, p.est_min_per_epoch):
        assert v >= 0.0
    assert math.isclose(p.est_min_per_epoch, p.est_secs_per_epoch / 60.0, rel_tol=1e-9)
    assert p.render_crop_mean_s == max(0.0, p.warm_mean_s - p.projection_mean_s)


def test_format_profile_is_readable():
    cfg = _small_cfg()
    text = format_profile(profile_datagen(cfg, n=2))
    assert isinstance(text, str)
    assert "datagen profile" in text
    assert "epoch" in text
