"""train() produces a well-formed result, checkpoints, and resumes exactly."""

import torch

from inr_unet.config import load_config
from inr_unet.train import TrainResult, load_checkpoint, train

CONFIG = "configs/default.yaml"


def _cfg(epochs):
    cfg = load_config(CONFIG)
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 6
    cfg.data.synthetic.draws_per_scene = 2
    cfg.data.synthetic.sample_q = 64
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.model.encoder.base_channels = 8
    cfg.model.encoder.depth = 3
    cfg.model.encoder.feature_dim = 16
    cfg.model.decoder.hidden_dim = 32
    cfg.model.decoder.num_layers = 3
    cfg.train.amp = False
    cfg.train.scheduler = "none"   # isolate resume from schedule-total bookkeeping
    cfg.train.epochs = epochs
    cfg.train.early_stop_patience = 999
    cfg.train.eval_every = 1   # one EpochRecord per epoch (pins len(history) == epochs)
    cfg.train.split.train_frac = 0.5
    cfg.train.split.val_frac = 0.25
    cfg.train.eval.eval_draws_per_scene = 1
    cfg.train.profile_datagen_samples = 0
    cfg.train.log_every = -1
    return cfg


def test_train_returns_result_and_checkpoints(tmp_path):
    result = train(_cfg(2), run_dir=tmp_path / "run")
    assert isinstance(result, TrainResult)
    assert len(result.history) == 2
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()
    assert result.val_panels["input"].ndim == 3
    assert all(r.epoch_time_s >= 0.0 for r in result.history)


def test_resume_equals_uninterrupted(tmp_path):
    full = train(_cfg(4), run_dir=tmp_path / "A")
    train(_cfg(2), run_dir=tmp_path / "B")
    resumed = train(_cfg(4), run_dir=tmp_path / "B",
                    resume_from=tmp_path / "B" / "checkpoints" / "last.pt")
    wa = load_checkpoint(tmp_path / "A" / "checkpoints" / "last.pt")["model"]
    wb = load_checkpoint(tmp_path / "B" / "checkpoints" / "last.pt")["model"]
    for k in wa:
        assert torch.allclose(wa[k], wb[k], atol=1e-6), k
    assert [r.train_loss for r in full.history] == [r.train_loss for r in resumed.history]
    assert [r.val_f1 for r in full.history] == [r.val_f1 for r in resumed.history]


def test_train_prints_probe_and_heartbeat(tmp_path, capsys):
    cfg = _cfg(1)
    cfg.train.profile_datagen_samples = 2
    cfg.train.log_every = 1
    train(cfg, run_dir=tmp_path / "run")
    out = capsys.readouterr().out
    assert "datagen profile" in out
    assert "samp/s" in out


def test_train_skips_probe_when_disabled(tmp_path, capsys):
    cfg = _cfg(1)  # _cfg already sets profile_datagen_samples=0, log_every=-1
    train(cfg, run_dir=tmp_path / "run")
    out = capsys.readouterr().out
    assert "datagen profile" not in out


def test_history_records_eval_time(tmp_path):
    result = train(_cfg(2), run_dir=tmp_path / "run")
    assert all(r.eval_time_s >= 0.0 for r in result.history)
    assert any(r.eval_time_s > 0.0 for r in result.history)  # eval actually ran
