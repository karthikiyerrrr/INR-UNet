"""STEMSegDataset and LIIFSegDataset: thin formatters over SyntheticRenderSource."""

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from inr_unet.config import ExperimentConfig
from inr_unet.data import LIIFSegDataset, STEMSegDataset


def _cfg(crop_size=48, sample_q=128, n_scenes=2, draws_per_scene=2):
    cfg = OmegaConf.structured(ExperimentConfig)
    cfg.data.synthetic.crop_size = crop_size
    cfg.data.synthetic.sample_q = sample_q
    cfg.data.synthetic.n_scenes = n_scenes
    cfg.data.synthetic.draws_per_scene = draws_per_scene
    cfg.generation.sampler.pixel_size_A_min = 0.20
    cfg.generation.sampler.pixel_size_A_max = 0.30
    return cfg


def test_stem_shapes_and_range():
    ds = STEMSegDataset(_cfg(crop_size=48))
    assert len(ds) == 4
    image, mask = ds[0]
    assert image.shape == (1, 48, 48)
    assert mask.shape == (1, 48, 48)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_liif_shapes_and_conventions():
    cfg = _cfg(crop_size=48, sample_q=200)
    ds = LIIFSegDataset(cfg)
    image, coords, cell, gt = ds[0]
    assert image.shape == (1, 48, 48)
    assert coords.shape == (200, 2)
    assert cell.shape == (200, 2)
    assert gt.shape == (200, 1)
    assert float(coords.min()) >= -1.0 and float(coords.max()) <= 1.0
    assert float(gt.min()) >= 0.0 and float(gt.max()) <= 1.0
    # cell is a single physical pixel size, in the configured range
    assert torch.unique(cell).numel() == 1
    tgt = float(cell[0, 0])
    syn = cfg.data.synthetic
    assert syn.target_pixel_size_A_min <= tgt <= syn.target_pixel_size_A_max


def test_both_datasets_deterministic():
    cfg = _cfg()
    s0 = STEMSegDataset(cfg)[1]
    s1 = STEMSegDataset(cfg)[1]
    assert torch.equal(s0[0], s1[0]) and torch.equal(s0[1], s1[1])
    l0 = LIIFSegDataset(cfg)[1]
    l1 = LIIFSegDataset(cfg)[1]
    for a, b in zip(l0, l1, strict=True):
        assert torch.equal(a, b)


def test_default_collate_batches_cleanly():
    cfg = _cfg(crop_size=32, sample_q=64)
    stem = STEMSegDataset(cfg)
    batch = default_collate([stem[0], stem[1]])
    assert batch[0].shape == (2, 1, 32, 32)
    assert batch[1].shape == (2, 1, 32, 32)
    liif = LIIFSegDataset(cfg)
    img, coords, cell, gt = default_collate([liif[0], liif[1]])
    assert img.shape == (2, 1, 32, 32)
    assert coords.shape == (2, 64, 2)
    assert cell.shape == (2, 64, 2)
    assert gt.shape == (2, 64, 1)


def test_liif_gt_ties_to_coordinates():
    """A query placed exactly at a known column returns gt ~ 1 (analytic, not rasterized)."""
    cfg = _cfg(sample_q=16)
    ds = LIIFSegDataset(cfg)
    s = ds.source.get(0)
    if s.positions_A.shape[0] == 0:
        pytest.skip("no columns survived the crop for this draw")
    col = s.positions_A[0]
    val = ds.label_field.values_at(s.positions_A, s.radii_A, col[None, :])
    assert float(val[0]) == pytest.approx(1.0, abs=1e-5)
