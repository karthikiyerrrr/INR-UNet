"""profile_datagen times the full ds[i] over shuffled indices and splits the cost three ways."""

import math

from inr_unet.config import load_config
from inr_unet.profiling import (
    DatagenProfile,
    TrainStepProfile,
    format_profile,
    format_train_step_profile,
    profile_datagen,
    profile_train_step,
)
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
              p.render_crop_mean_s, p.values_at_mean_s, p.est_secs_per_epoch,
              p.est_min_per_epoch):
        assert v >= 0.0
    assert math.isclose(p.est_min_per_epoch, p.est_secs_per_epoch / 60.0, rel_tol=1e-9)


def test_format_profile_shows_three_way_split():
    cfg = _small_cfg()
    text = format_profile(profile_datagen(cfg, n=2))
    assert isinstance(text, str)
    assert "datagen profile" in text
    assert "projection" in text
    assert "render+crop" in text
    assert "values_at" in text
    assert "epoch" in text


def _both_model_cfgs():
    inr = _small_cfg()
    base = _small_cfg()
    base.model.name = "unet_baseline"
    return inr, base


def test_profile_train_step_fields_inr():
    cfg = _small_cfg()
    cfg.data.batch_size = 2
    p = profile_train_step(cfg, n_batches=3, warmup=1)
    assert isinstance(p, TrainStepProfile)
    assert p.model_name == "inr_unet"
    assert p.n_batches == 3
    assert p.device == "cpu"
    for v in (p.data_ms, p.forward_ms, p.loss_ms, p.backward_ms, p.step_ms,
              p.data_median_ms, p.forward_median_ms, p.loss_median_ms,
              p.backward_median_ms, p.step_median_ms):
        assert v >= 0.0


def test_profile_train_step_fields_baseline():
    _, base = _both_model_cfgs()
    base.data.batch_size = 2
    p = profile_train_step(base, n_batches=3, warmup=1)
    assert p.model_name == "unet_baseline"
    assert p.n_batches == 3
    for v in (p.data_ms, p.forward_ms, p.loss_ms, p.backward_ms, p.step_ms):
        assert v >= 0.0


def test_format_train_step_profile_lists_all_phases():
    cfg = _small_cfg()
    cfg.data.batch_size = 2
    text = format_train_step_profile(profile_train_step(cfg, n_batches=2, warmup=1))
    assert isinstance(text, str)
    assert "train-step profile" in text
    for phase in ("data", "forward", "loss", "backward", "step"):
        assert phase in text
    assert "inr_unet" in text
