"""train_one_epoch lowers loss on a tiny overfittable loader."""

import torch
from torch.utils.data import DataLoader, Subset

from inr_unet.config import load_config
from inr_unet.data import LIIFSegDataset
from inr_unet.losses import make_loss
from inr_unet.registry import build_model
from inr_unet.train import build_optimizer, build_scheduler, train_one_epoch

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
    first = train_one_epoch(model, loader, loss_fn, opt, sched, cfg, device="cpu")
    last = first
    for _ in range(9):
        last = train_one_epoch(model, loader, loss_fn, opt, sched, cfg, device="cpu")
    assert last < first
