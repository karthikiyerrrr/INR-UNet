"""TrainPath strategy: PATHS resolves by model.name; step_loss/eval_tile work per path."""

import torch
from torch.utils.data import DataLoader, Subset

from inr_unet.config import load_config
from inr_unet.data import LIIFSegDataset, STEMSegDataset
from inr_unet.losses import make_loss
from inr_unet.registry import build_model
from inr_unet.train import PATHS, DensePath, QueryPath

CONFIG = "configs/default.yaml"


def _cfg(name):
    cfg = load_config(CONFIG)
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 3
    cfg.data.synthetic.draws_per_scene = 2
    cfg.data.synthetic.sample_q = 64
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.model.name = name
    cfg.model.encoder.base_channels = 8
    cfg.model.encoder.depth = 3
    cfg.model.encoder.feature_dim = 16
    cfg.model.decoder.hidden_dim = 32
    cfg.model.decoder.num_layers = 3
    return cfg


def test_paths_resolve_by_name():
    assert isinstance(PATHS["inr_unet"], QueryPath)
    assert isinstance(PATHS["unet_baseline"], DensePath)


def test_query_path_dataset_and_step():
    cfg = _cfg("inr_unet")
    path = PATHS["inr_unet"]
    ds = path.build_dataset(cfg, None)
    assert isinstance(ds, LIIFSegDataset)
    model = build_model(cfg)
    loss_fn = make_loss(cfg)
    batch = next(iter(DataLoader(Subset(ds, [0, 1]), batch_size=2)))
    loss, bs = path.step_loss(model, batch, loss_fn, "cpu")
    assert bs == 2
    assert torch.isfinite(loss)


def test_dense_path_dataset_and_step():
    cfg = _cfg("unet_baseline")
    path = PATHS["unet_baseline"]
    ds = path.build_dataset(cfg, None)
    assert isinstance(ds, STEMSegDataset)
    model = build_model(cfg)
    loss_fn = make_loss(cfg)
    batch = next(iter(DataLoader(Subset(ds, [0, 1]), batch_size=2)))
    loss, bs = path.step_loss(model, batch, loss_fn, "cpu")
    assert bs == 2
    assert torch.isfinite(loss)


def test_eval_tile_returns_loss_and_heatmap():
    for name in ("inr_unet", "unet_baseline"):
        cfg = _cfg(name)
        path = PATHS[name]
        ds = path.build_dataset(cfg, None)
        model = build_model(cfg)
        loss_fn = make_loss(cfg)
        s = ds.source.get(0)
        val_loss, heatmap = path.eval_tile(model, ds, s, 0, loss_fn, "cpu")
        assert isinstance(val_loss, float)
        assert heatmap.shape == (ds.source.crop_size, ds.source.crop_size)
