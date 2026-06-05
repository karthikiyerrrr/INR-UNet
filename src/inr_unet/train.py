"""Training loop for INRUNet: structure-stratified splits, resumable multi-scene
training on the LIIF query path, and dense peak-localization evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class Splits:
    """Flattened ``LIIFSegDataset`` sample indices for each partition (no scene shared)."""

    train: list[int]
    val: list[int]
    test: list[int]


def _n_scene_classes(cfg: DictConfig) -> int:
    """Number of distinct structure classes: CIF manifest entries, else 1 (synthetic)."""
    if str(cfg.data.provider) == "cif":
        manifest = yaml.safe_load(Path(cfg.data.cif.manifest_path).read_text())
        return len(manifest["entries"])
    return 1


def build_splits(cfg: DictConfig) -> Splits:
    """Partition scenes into train/val/test, stratified by structure class.

    A scene's class is ``scene_idx % n_classes`` (CIFProvider cycles entries by modulo), so
    splitting each class's scene list independently guarantees every structure appears in every
    split with fully disjoint seeds. Splitting is at the scene level (all draws of a scene stay
    together) to avoid augmentation leakage. Train uses every draw; val/test are capped at
    ``eval_draws_per_scene`` to bound per-epoch eval cost.
    """
    n_scenes = int(cfg.data.synthetic.n_scenes)
    draws = int(cfg.data.synthetic.draws_per_scene)
    n_classes = _n_scene_classes(cfg)
    train_frac = float(cfg.train.split.train_frac)
    val_frac = float(cfg.train.split.val_frac)
    eval_draws = min(int(cfg.train.eval.eval_draws_per_scene), draws)
    rng = np.random.default_rng(int(cfg.train.split.seed))

    train_scenes: list[int] = []
    val_scenes: list[int] = []
    test_scenes: list[int] = []
    for c in range(n_classes):
        scenes = list(range(c, n_scenes, n_classes))
        rng.shuffle(scenes)
        n = len(scenes)
        if n < 3:
            raise ValueError(
                f"structure class {c} has only {n} scene(s); need >=3 for a non-empty "
                f"train/val/test split (increase n_scenes or reduce classes)"
            )
        # Guarantee at least 1 scene in val and test per class; train gets the rest.
        n_val = max(1, int(round(val_frac * n)))
        n_test = max(1, n - int(round(train_frac * n)) - n_val)
        n_train = n - n_val - n_test
        train_scenes += scenes[:n_train]
        val_scenes += scenes[n_train:n_train + n_val]
        test_scenes += scenes[n_train + n_val:]

    if not val_scenes or not test_scenes:
        raise ValueError(
            f"split produced an empty val/test set (n_scenes={n_scenes}, classes={n_classes}); "
            "increase n_scenes or adjust fractions"
        )

    def expand(scenes: list[int], per_scene: int) -> list[int]:
        return sorted(s * draws + d for s in scenes for d in range(per_scene))

    return Splits(
        train=expand(train_scenes, draws),
        val=expand(val_scenes, eval_draws),
        test=expand(test_scenes, eval_draws),
    )
