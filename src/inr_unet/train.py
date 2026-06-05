"""Training loop for INRUNet: structure-stratified splits, resumable multi-scene
training on the LIIF query path, and dense peak-localization evaluation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

from inr_unet.localization import peak_localization
from inr_unet.losses import make_loss

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


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_val_f1: float,
    best_epoch: int,
    history: list,
) -> None:
    """Atomically write full training state (incl. torch + numpy RNG) to ``path``."""
    path = Path(path)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint dict written by :func:`save_checkpoint` (CPU map)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def build_optimizer(model, cfg: DictConfig):
    """AdamW with config lr + weight decay."""
    return torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )


def make_lr_lambda(cfg: DictConfig, total_steps: int):
    """LR multiplier: linear warmup to 1.0 then cosine decay to 0; constant for 'none'."""
    sched = str(cfg.train.scheduler)
    warmup = int(float(cfg.train.warmup_frac) * total_steps)

    def fn(step: int) -> float:
        if sched == "none":
            return 1.0
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def build_scheduler(optimizer, cfg: DictConfig, total_steps: int):
    """LambdaLR driving :func:`make_lr_lambda`; step once per optimizer step."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(cfg, total_steps))


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregate held-out metrics for one eval pass."""

    loss: float
    precision: float        # macro (mean over tiles)
    recall: float
    f1: float
    mean_offset_A: float     # mean over non-empty tiles' mean_offset_A (NaN if none matched)
    median_offset_A: float
    micro_precision: float   # pooled: sum matched / sum pred
    micro_recall: float
    n_tiles: int
    n_empty: int             # tiles with n_gt == 0


def evaluate(model, dataset, indices: list[int], cfg: DictConfig, device: str) -> EvalMetrics:
    """Run the dense path on each eval tile, score peak_localization, aggregate macro + micro.

    ``dataset`` is a ``LIIFSegDataset``; its ``.source.get(idx)`` gives the dense image + atom
    positions, and ``dataset[idx]`` gives the query-path tensors for the val loss. Pixel size is
    ``input_pixel_size_A`` (= tile_fov_A / crop_size), never a sampler value.

    Two forward passes per tile, both under ``no_grad``: the query path mirrors the training loss
    (random coords + cell), while the dense path produces the full heatmap that peak_localization
    needs. They are distinct computations, so the encoder pass is not shared. ``mean_offset_A`` is
    averaged over *matched* peaks only (tiles with no match contribute nothing), so it reflects
    localization accuracy where the model fired, not coverage.
    """
    model.eval()
    ec = cfg.train.eval
    loss_fn = make_loss(cfg)
    total_loss, n = 0.0, 0
    per_tile: list[dict] = []
    micro_pred = micro_gt = micro_match = 0
    offsets: list[float] = []
    n_empty = 0
    with torch.no_grad():
        for idx in indices:
            img, coords, cell, gt = (t[None].to(device) for t in dataset[idx])
            total_loss += float(loss_fn(model(img, coords, cell), gt))
            n += 1
            s = dataset.source.get(idx)
            dense = torch.sigmoid(model(s.image[None, None].to(device)))[0, 0]
            m = peak_localization(
                dense, s.positions_A, s.input_pixel_size_A,
                threshold=float(ec.threshold),
                min_distance_px=float(ec.min_distance_px),
                match_tol_px=float(ec.match_tol_px),
            )
            per_tile.append(m)
            micro_pred += m["n_pred"]
            micro_gt += m["n_gt"]
            micro_match += m["n_matched"]
            if m["n_gt"] == 0:
                n_empty += 1
            elif not math.isnan(m["mean_offset_A"]):
                offsets.append(m["mean_offset_A"])

    macro_p = float(np.mean([m["precision"] for m in per_tile]))
    macro_r = float(np.mean([m["recall"] for m in per_tile]))
    macro_f1 = float(np.mean([m["f1"] for m in per_tile]))
    return EvalMetrics(
        loss=total_loss / max(1, n),
        precision=macro_p,
        recall=macro_r,
        f1=macro_f1,
        mean_offset_A=float(np.mean(offsets)) if offsets else float("nan"),
        median_offset_A=float(np.median(offsets)) if offsets else float("nan"),
        micro_precision=micro_match / micro_pred if micro_pred else 0.0,
        # NaN, not 1.0: micro_gt == 0 means the whole eval set had no atoms (degenerate config),
        # which is undefined recall rather than perfect recall.
        micro_recall=micro_match / micro_gt if micro_gt else float("nan"),
        n_tiles=len(per_tile),
        n_empty=n_empty,
    )
