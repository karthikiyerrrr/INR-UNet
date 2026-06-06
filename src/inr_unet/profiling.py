"""Datagen profiling: measure per-sample render cost and project an epoch ETA.

Standalone diagnostic (call from a notebook) and the one-time startup probe inside ``train()``.
Runs single-threaded in the main process (no DataLoader workers) so it measures raw render cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from inr_unet.data import LIIFSegDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class DatagenProfile:
    """Timing summary for one datagen profiling pass."""

    n_samples: int             # warm samples timed
    cold_s: float              # first source.get(idx): pays structure load + first projection
    warm_mean_s: float         # mean of n warm source.get(idx)
    warm_median_s: float
    projection_mean_s: float   # mean warmed provider.get(scene_idx): isolates project_structure
    render_crop_mean_s: float  # warm_mean_s - projection_mean_s (render + sampler + crop share)
    est_secs_per_epoch: float  # warm_mean_s * n_train / max(1, num_workers); optimistic
    est_min_per_epoch: float
    n_train: int
    num_workers: int


def _take(indices: list[int], n: int, start: int) -> list[int]:
    """``n`` indices from ``indices`` beginning at ``start``, wrapping if the list is short."""
    return [indices[(start + i) % len(indices)] for i in range(n)]


def profile_datagen(cfg: DictConfig, *, n: int = 8, indices: list[int] | None = None
                    ) -> DatagenProfile:
    """Time datagen and project epoch cost.

    ``est_secs_per_epoch`` assumes perfect worker scaling and ignores collate/copy/transfer, so it
    is an optimistic lower bound. Whatever ``generation.sampler.device`` is set to is reflected.
    """
    from inr_unet.train import build_splits

    dataset = LIIFSegDataset(cfg)
    source = dataset.source
    if indices is None:
        indices = build_splits(cfg).train
    if not indices:
        raise ValueError("no indices to profile (empty train split)")
    n_train = len(indices)

    t0 = time.perf_counter()
    source.get(indices[0])
    cold_s = time.perf_counter() - t0

    warm = []
    for idx in _take(indices, n, start=1):
        t0 = time.perf_counter()
        source.get(idx)
        warm.append(time.perf_counter() - t0)
    warm_mean_s = float(np.mean(warm))
    warm_median_s = float(np.median(warm))

    scene_idx = indices[0] // source.draws_per_scene
    source.provider.get(scene_idx)  # warm structure load so we time projection, not file I/O
    proj = []
    for _ in range(n):
        t0 = time.perf_counter()
        source.provider.get(scene_idx)
        proj.append(time.perf_counter() - t0)
    projection_mean_s = float(np.mean(proj))
    render_crop_mean_s = max(0.0, warm_mean_s - projection_mean_s)

    est_secs_per_epoch = warm_mean_s * n_train / max(1, int(cfg.data.num_workers))
    return DatagenProfile(
        n_samples=len(warm),
        cold_s=cold_s,
        warm_mean_s=warm_mean_s,
        warm_median_s=warm_median_s,
        projection_mean_s=projection_mean_s,
        render_crop_mean_s=render_crop_mean_s,
        est_secs_per_epoch=est_secs_per_epoch,
        est_min_per_epoch=est_secs_per_epoch / 60.0,
        n_train=n_train,
        num_workers=int(cfg.data.num_workers),
    )


def format_profile(p: DatagenProfile) -> str:
    """Human-readable block for the startup probe / notebook diagnostic."""
    pct = (100.0 * p.projection_mean_s / p.warm_mean_s) if p.warm_mean_s else 0.0
    return (
        f"datagen profile (single-process, {p.n_samples} warm samples)\n"
        f"  cold sample:        {p.cold_s:.2f} s\n"
        f"  warm mean / median: {p.warm_mean_s:.2f} / {p.warm_median_s:.2f} s\n"
        f"  projection share:   {p.projection_mean_s:.2f} s ({pct:.0f}%)\n"
        f"  render+crop share:  {p.render_crop_mean_s:.2f} s\n"
        f"  est epoch time:     ~{p.est_min_per_epoch:.1f} min   "
        f"(n_train={p.n_train}, num_workers={p.num_workers}, optimistic)"
    )
