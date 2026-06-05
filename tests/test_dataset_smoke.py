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
    any_signal = False
    for i in range(len(ds)):
        image, mask = ds[i]
        assert image.shape == (1, s, s)
        assert mask.shape == (1, s, s)
        assert torch.isfinite(image).all()
        assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
        any_signal = any_signal or float(mask.max()) > 0.5
    assert any_signal, "content-aware crop should find a labelled column in 9 draws"


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
        cell_val = float(cell[0, 0])
        sample = ds.source.get(i)
        crop_extent_A = ds.source.crop_size * sample.input_pixel_size_A
        px_min = cfg.data.synthetic.target_pixel_size_A_min
        px_max = cfg.data.synthetic.target_pixel_size_A_max
        lo = 2.0 * px_min / crop_extent_A
        hi = 2.0 * px_max / crop_extent_A
        assert lo <= cell_val <= hi  # dimensionless, not raw A/px
        assert torch.isfinite(gt).all()
        assert float(gt.min()) >= 0.0 and float(gt.max()) <= 1.0
        any_signal = any_signal or float(gt.max()) > 0.0
    assert any_signal  # content-aware crop yields labelled columns somewhere in the set
