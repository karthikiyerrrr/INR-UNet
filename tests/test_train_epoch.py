"""train_one_epoch lowers loss on a tiny overfittable loader."""

import torch
from torch.utils.data import DataLoader, Subset

from inr_unet.config import load_config
from inr_unet.data import LIIFSegDataset, STEMSegDataset
from inr_unet.losses import make_loss
from inr_unet.registry import build_model
from inr_unet.train import (
    PATHS,
    build_optimizer,
    build_scheduler,
    build_splits,
    evaluate,
    train_one_epoch,
)

CONFIG = "configs/default.yaml"


def _cfg():
    cfg = load_config(CONFIG)
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 2
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
    cfg.train.lr = 1e-2
    return cfg


def test_epoch_runs_and_reduces_loss():
    torch.manual_seed(0)
    cfg = _cfg()
    ds = LIIFSegDataset(cfg)
    loader = DataLoader(Subset(ds, list(range(len(ds)))), batch_size=2, shuffle=True)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=20)
    loss_fn = make_loss(cfg)
    first, t_first = train_one_epoch(model, loader, loss_fn, opt, sched, cfg, device="cpu")
    assert t_first >= 0.0
    last = first
    for _ in range(9):
        last, _ = train_one_epoch(model, loader, loss_fn, opt, sched, cfg, device="cpu")
    assert last < first


def test_epoch_heartbeat_prints(capsys):
    torch.manual_seed(0)
    cfg = _cfg()
    cfg.train.log_every = 1
    ds = LIIFSegDataset(cfg)
    loader = DataLoader(Subset(ds, list(range(len(ds)))), batch_size=2, shuffle=True)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=20)
    loss_fn = make_loss(cfg)
    train_one_epoch(model, loader, loss_fn, opt, sched, cfg, device="cpu", epoch=0)
    out = capsys.readouterr().out
    assert "samp/s" in out
    assert "first batch" in out


def test_epoch_heartbeat_silent_when_off(capsys):
    torch.manual_seed(0)
    cfg = _cfg()
    cfg.train.log_every = -1
    ds = LIIFSegDataset(cfg)
    loader = DataLoader(Subset(ds, list(range(len(ds)))), batch_size=2, shuffle=True)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=20)
    loss_fn = make_loss(cfg)
    train_one_epoch(model, loader, loss_fn, opt, sched, cfg, device="cpu", epoch=0)
    out = capsys.readouterr().out
    assert "samp/s" not in out
    assert "first batch" not in out


def test_evaluate_renders_each_tile_once(monkeypatch):
    torch.manual_seed(0)
    cfg = _cfg()
    cfg.data.synthetic.n_scenes = 3  # build_splits requires >= 3 scenes
    ds = LIIFSegDataset(cfg)
    model = build_model(cfg)
    val = build_splits(cfg).val
    calls: list[int] = []
    real_get = ds.source.get

    def counting_get(idx):
        calls.append(idx)
        return real_get(idx)

    monkeypatch.setattr(ds.source, "get", counting_get)
    metrics = evaluate(model, ds, val, cfg, device="cpu")
    assert metrics.n_tiles == len(val)
    assert sorted(calls) == sorted(val)  # exactly one source.get per tile (was two)


def test_dense_epoch_runs_and_reduces_loss():
    torch.manual_seed(0)
    cfg = _cfg()
    cfg.model.name = "unet_baseline"
    ds = STEMSegDataset(cfg)
    loader = DataLoader(Subset(ds, list(range(len(ds)))), batch_size=2, shuffle=True)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=20)
    loss_fn = make_loss(cfg)
    path = PATHS["unet_baseline"]
    first, t_first = train_one_epoch(
        model, loader, loss_fn, opt, sched, cfg, device="cpu", path=path)
    assert t_first >= 0.0
    last = first
    for _ in range(9):
        last, _ = train_one_epoch(model, loader, loss_fn, opt, sched, cfg, device="cpu", path=path)
    assert last < first
