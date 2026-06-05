"""Smoke test: both datasets emit sane pairs under the locked default.yaml."""

import torch

from inr_unet.config import load_config
from inr_unet.data import LIIFSegDataset, STEMSegDataset

CONFIG = "configs/default.yaml"


def _small_locked_cfg():
    # the real locked distribution, shrunk for test speed
    cfg = load_config(CONFIG)
    cfg.data.synthetic.n_scenes = 3
    cfg.data.synthetic.draws_per_scene = 3
    cfg.data.synthetic.sample_q = 128
    return cfg


def test_unet_pairs_are_sane():
    cfg = _small_locked_cfg()
    ds = STEMSegDataset(cfg)
    s = cfg.data.synthetic.crop_size
    for i in range(len(ds)):
        image, mask = ds[i]
        assert image.shape == (1, s, s)
        assert mask.shape == (1, s, s)
        assert torch.isfinite(image).all()
        assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_liif_pairs_are_sane_and_some_label_nonempty():
    cfg = _small_locked_cfg()
    ds = LIIFSegDataset(cfg)
    s = cfg.data.synthetic.crop_size
    any_signal = False
    for i in range(len(ds)):
        image, coords, cell, gt = ds[i]
        assert image.shape == (1, s, s)
        assert coords.shape == (cfg.data.synthetic.sample_q, 2)
        assert float(coords.min()) >= -1.0 and float(coords.max()) <= 1.0
        tgt = float(cell[0, 0])
        px_min = cfg.data.synthetic.target_pixel_size_A_min
        px_max = cfg.data.synthetic.target_pixel_size_A_max
        assert px_min <= tgt <= px_max
        assert torch.isfinite(gt).all()
        assert float(gt.min()) >= 0.0 and float(gt.max()) <= 1.0
        any_signal = any_signal or float(gt.max()) > 0.5
    assert any_signal  # content-aware crop yields labelled columns somewhere in the set
