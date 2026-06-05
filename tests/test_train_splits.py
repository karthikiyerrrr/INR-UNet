"""build_splits stratifies by structure class with scene-level disjointness."""

import pytest

from inr_unet.config import load_config
from inr_unet.train import build_splits

CONFIG = "configs/default.yaml"


def _cif_cfg(n_scenes=36, draws=2):
    cfg = load_config(CONFIG)
    cfg.data.provider = "cif"
    cfg.data.synthetic.n_scenes = n_scenes
    cfg.data.synthetic.draws_per_scene = draws
    cfg.train.eval.eval_draws_per_scene = 1
    return cfg


def _scene_of(idx, draws):
    return idx // draws


def test_splits_cover_every_structure_class():
    cfg = _cif_cfg()
    draws = cfg.data.synthetic.draws_per_scene
    s = build_splits(cfg)
    # 12 manifest entries -> class = scene % 12; each split must contain all 12 classes
    for name, idxs in (("train", s.train), ("val", s.val), ("test", s.test)):
        classes = {_scene_of(i, draws) % 12 for i in idxs}
        assert classes == set(range(12)), (name, sorted(classes))


def test_splits_are_scene_disjoint():
    cfg = _cif_cfg()
    draws = cfg.data.synthetic.draws_per_scene
    s = build_splits(cfg)
    scenes = {n: {_scene_of(i, draws) for i in idxs}
              for n, idxs in (("train", s.train), ("val", s.val), ("test", s.test))}
    assert scenes["train"].isdisjoint(scenes["val"])
    assert scenes["train"].isdisjoint(scenes["test"])
    assert scenes["val"].isdisjoint(scenes["test"])


def test_val_test_use_capped_draws():
    cfg = _cif_cfg(draws=4)
    cfg.train.eval.eval_draws_per_scene = 2
    s = build_splits(cfg)
    # each val scene contributes exactly eval_draws_per_scene sample indices
    val_scenes = {i // 4 for i in s.val}
    assert len(s.val) == 2 * len(val_scenes)


def test_splits_are_deterministic():
    cfg = _cif_cfg()
    assert build_splits(cfg) == build_splits(cfg)


def test_too_few_scenes_per_class_raises():
    cfg = _cif_cfg(n_scenes=24)  # 24 / 12 classes = 2 scenes per class (< 3)
    with pytest.raises(ValueError):
        build_splits(cfg)
