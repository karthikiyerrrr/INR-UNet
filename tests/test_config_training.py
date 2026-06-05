"""The training/split/eval config knobs load with defaults and accept overrides."""

from inr_unet.config import load_config

CONFIG = "configs/default.yaml"


def test_train_defaults_present():
    cfg = load_config(CONFIG)
    t = cfg.train
    assert t.weight_decay == 1.0e-4
    assert t.warmup_frac == 0.05
    assert t.scheduler == "cosine"
    assert t.grad_clip_norm == 1.0
    assert t.grad_accum_steps == 1
    assert t.amp is True
    assert t.eval_every == 1
    assert t.early_stop_patience == 15


def test_split_and_eval_nested_defaults():
    cfg = load_config(CONFIG)
    assert cfg.train.split.train_frac == 0.7
    assert cfg.train.split.val_frac == 0.15
    assert cfg.train.split.seed == 0
    assert cfg.train.eval.eval_draws_per_scene == 4
    assert cfg.train.eval.threshold == 0.5
    assert cfg.train.eval.min_distance_px == 2.0
    assert cfg.train.eval.match_tol_px == 2.0
    assert cfg.train.eval.n_val_panels == 6


def test_overrides_merge():
    cfg = load_config(CONFIG)
    cfg.train.scheduler = "none"
    cfg.train.split.val_frac = 0.2
    assert cfg.train.scheduler == "none"
    assert cfg.train.split.val_frac == 0.2
