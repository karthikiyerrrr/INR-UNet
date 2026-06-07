"""Datagen profiling: time the full ``ds[i]`` and split per-sample cost into render / projection /
values_at, then project an epoch ETA.

Standalone diagnostic (call from a notebook) and the one-time startup probe inside ``train()``.
Runs single-threaded in the main process. Caching is disabled internally so each timed call pays
the real (uncached) cost and the part subtraction stays valid; the reported epoch ETA is therefore
an uncached upper bound (a real run with ``data.cache_scenes`` on and workers is faster).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import OmegaConf

from inr_unet.data import LIIFSegDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class DatagenProfile:
    """Timing summary for one datagen profiling pass (all times in seconds)."""

    n_samples: int             # warm full-ds[i] samples timed
    cold_s: float              # first full ds[i]: pays structure load + first projection
    warm_mean_s: float         # mean warm full ds[i]
    warm_median_s: float
    projection_mean_s: float   # mean provider.get(scene_idx): project_structure (uncached)
    render_crop_mean_s: float  # source.get(idx) - provider.get (render + sampler + crop)
    values_at_mean_s: float    # ds[i] - source.get(idx) (query-point label sampling)
    est_secs_per_epoch: float  # warm_mean_s * n_train / max(1, num_workers); uncached upper bound
    est_min_per_epoch: float
    n_train: int
    num_workers: int


def _take(indices: list[int], n: int, start: int) -> list[int]:
    """``n`` indices from ``indices`` beginning at ``start``, wrapping if the list is short."""
    return [indices[(start + i) % len(indices)] for i in range(n)]


def profile_datagen(cfg: DictConfig, *, n: int = 8, indices: list[int] | None = None
                    ) -> DatagenProfile:
    """Time full ``ds[i]`` over shuffled train indices and break the cost into three parts.

    ``est_secs_per_epoch`` assumes perfect worker scaling and uncached per-sample cost, so it is an
    upper bound; real runs (cache on, persistent workers) are faster.
    """
    from inr_unet.train import build_splits

    cfg = OmegaConf.merge(cfg, {"data": {"cache_scenes": False}})
    dataset = LIIFSegDataset(cfg)
    source = dataset.source
    if indices is None:
        indices = build_splits(cfg).train
    if not indices:
        raise ValueError("no indices to profile (empty train split)")
    n_train = len(indices)

    shuffled = list(indices)
    np.random.default_rng(0).shuffle(shuffled)

    t0 = time.perf_counter()
    dataset[shuffled[0]]
    cold_s = time.perf_counter() - t0

    full, proj, rend, vals = [], [], [], []
    for idx in _take(shuffled, n, start=1):
        scene_idx = idx // source.draws_per_scene
        t0 = time.perf_counter()
        source.provider.get(scene_idx)
        t_proj = time.perf_counter() - t0
        t0 = time.perf_counter()
        source.get(idx)
        t_src = time.perf_counter() - t0
        t0 = time.perf_counter()
        dataset[idx]
        t_full = time.perf_counter() - t0
        full.append(t_full)
        proj.append(t_proj)
        rend.append(max(0.0, t_src - t_proj))
        vals.append(max(0.0, t_full - t_src))

    warm_mean_s = float(np.mean(full))
    est_secs_per_epoch = warm_mean_s * n_train / max(1, int(cfg.data.num_workers))
    return DatagenProfile(
        n_samples=len(full),
        cold_s=cold_s,
        warm_mean_s=warm_mean_s,
        warm_median_s=float(np.median(full)),
        projection_mean_s=float(np.mean(proj)),
        render_crop_mean_s=float(np.mean(rend)),
        values_at_mean_s=float(np.mean(vals)),
        est_secs_per_epoch=est_secs_per_epoch,
        est_min_per_epoch=est_secs_per_epoch / 60.0,
        n_train=n_train,
        num_workers=int(cfg.data.num_workers),
    )


def format_profile(p: DatagenProfile) -> str:
    """Human-readable block for the startup probe / notebook diagnostic."""
    parts = p.projection_mean_s + p.render_crop_mean_s + p.values_at_mean_s

    def pct(x: float) -> float:
        return (100.0 * x / parts) if parts else 0.0

    return (
        f"datagen profile (full ds[i], {p.n_samples} shuffled warm samples)\n"
        f"  cold sample:        {p.cold_s:.2f} s\n"
        f"  warm mean / median: {p.warm_mean_s:.3f} / {p.warm_median_s:.3f} s\n"
        f"  projection:         {p.projection_mean_s:.3f} s ({pct(p.projection_mean_s):.0f}%)\n"
        f"  render+crop:        {p.render_crop_mean_s:.3f} s ({pct(p.render_crop_mean_s):.0f}%)\n"
        f"  values_at:          {p.values_at_mean_s:.3f} s ({pct(p.values_at_mean_s):.0f}%)\n"
        f"  est epoch time:     ~{p.est_min_per_epoch:.1f} min   "
        f"(n_train={p.n_train}, num_workers={p.num_workers}, uncached upper bound)"
    )
