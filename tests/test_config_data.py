"""build_train_loader applies worker knobs only when num_workers > 0."""

import torch

from inr_unet.config import load_config
from inr_unet.data import LIIFSegDataset
from inr_unet.train import build_train_loader

CONFIG = "configs/default.yaml"


def _cfg():
    cfg = load_config(CONFIG)
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 4
    cfg.data.synthetic.draws_per_scene = 2
    cfg.data.synthetic.sample_q = 64
    cfg.data.batch_size = 2
    return cfg


def test_loader_no_worker_kwargs_at_nw0():
    cfg = _cfg()
    cfg.data.num_workers = 0
    ds = LIIFSegDataset(cfg)
    loader = build_train_loader(ds, list(range(len(ds))), cfg, torch.Generator(), "cpu")
    assert loader.num_workers == 0
    assert loader.persistent_workers is False


def test_loader_sets_knobs_at_nw_gt0():
    cfg = _cfg()
    cfg.data.num_workers = 2
    cfg.data.prefetch_factor = 3
    cfg.data.persistent_workers = True
    ds = LIIFSegDataset(cfg)
    loader = build_train_loader(ds, list(range(len(ds))), cfg, torch.Generator(), "cpu")
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3


def test_loader_pin_memory_only_on_cuda():
    cfg = _cfg()
    cfg.data.num_workers = 0
    cfg.data.pin_memory = True
    ds = LIIFSegDataset(cfg)
    loader = build_train_loader(ds, list(range(len(ds))), cfg, torch.Generator(), "cpu")
    assert loader.pin_memory is False  # device == "cpu" disables it


def test_loader_pin_memory_passes_true_for_cuda_device():
    # device == "cuda" propagates pin_memory; DataLoader stores it at construction (no GPU needed).
    cfg = _cfg()
    cfg.data.num_workers = 0
    cfg.data.pin_memory = True
    ds = LIIFSegDataset(cfg)
    loader = build_train_loader(ds, list(range(len(ds))), cfg, torch.Generator(), "cuda")
    assert loader.pin_memory is True
