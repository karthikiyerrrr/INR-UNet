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
import torch
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

    # Nested-subtraction split: source.get embeds exactly one provider.get, and ds[i] embeds one
    # source.get, so t_proj=P, t_src=P+render, t_full=P+render+values_at. The deltas isolate each
    # part because projection is deterministic per scene_idx (same work each call, up to jitter).
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


@dataclass(frozen=True)
class TrainStepProfile:
    """Per-phase wall-clock for one training step, averaged over warm batches (ms)."""

    model_name: str
    device: str
    n_batches: int
    data_ms: float        # next(loader) + .to(device): dataset[i] (CPU rasterize/sample) + collate
    forward_ms: float     # model(...) only
    loss_ms: float        # loss_fn(...) only
    backward_ms: float    # loss.backward()
    step_ms: float        # optimizer.step + scheduler.step + zero_grad
    data_median_ms: float
    forward_median_ms: float
    loss_median_ms: float
    backward_median_ms: float
    step_median_ms: float


def _sync(device: str) -> float:
    """perf_counter, CUDA-synchronized so GPU phase timings are real (no-op on CPU)."""
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _forward_and_target(model, parts) -> tuple:
    """Split a moved batch into (model output, target). 4 tensors = LIIF query path
    (img, coords, cell, gt); 2 = dense path (img, mask). Mirrors QueryPath/DensePath.step_loss
    without importing them, so train.py stays untouched."""
    if len(parts) == 4:
        img, coords, cell, gt = parts
        return model(img, coords, cell), gt
    img, mask = parts
    return model(img), mask


def profile_train_step(cfg: DictConfig, *, n_batches: int = 20, warmup: int = 3
                       ) -> TrainStepProfile:
    """Time each phase of one training step (data / forward / loss / backward / step) for the
    model in ``cfg``, building the model, dataset, loader, optimizer, and scheduler exactly as
    ``train()`` does so the dense-vs-query asymmetry is measured in situ. Discards ``warmup``
    batches, then averages ``n_batches``.
    """
    from inr_unet.losses import make_loss
    from inr_unet.registry import build_model
    from inr_unet.train import (
        PATHS,
        build_optimizer,
        build_or_load_cache,
        build_scheduler,
        build_splits,
        build_train_loader,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    model.train()
    path = PATHS[str(cfg.model.name)]
    splits = build_splits(cfg)
    cached = build_or_load_cache(cfg, splits)
    dataset = path.build_dataset(cfg, cached)
    loader_cfg = OmegaConf.merge(cfg, {"data": {"num_workers": 0}}) if cached else cfg
    gen = torch.Generator()
    gen.manual_seed(0)
    loader = build_train_loader(dataset, splits.train, loader_cfg, gen, device)
    loss_fn = make_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps=max(1, warmup + n_batches))

    data, fwd, loss_t, bwd, step = [], [], [], [], []
    it = iter(loader)
    for i in range(warmup + n_batches):
        t0 = _sync(device)
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            batch = next(it)
        parts = [t.to(device) for t in batch]
        t_data = _sync(device) - t0

        t0 = _sync(device)
        out, target = _forward_and_target(model, parts)
        t_fwd = _sync(device) - t0

        t0 = _sync(device)
        ls = loss_fn(out, target)
        t_loss = _sync(device) - t0

        t0 = _sync(device)
        ls.backward()
        t_bwd = _sync(device) - t0

        t0 = _sync(device)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        t_step = _sync(device) - t0

        if i >= warmup:
            data.append(t_data)
            fwd.append(t_fwd)
            loss_t.append(t_loss)
            bwd.append(t_bwd)
            step.append(t_step)

    def ms(xs: list) -> float:
        return 1000.0 * float(np.mean(xs))

    def med(xs: list) -> float:
        return 1000.0 * float(np.median(xs))

    return TrainStepProfile(
        model_name=str(cfg.model.name),
        device=device,
        n_batches=len(data),
        data_ms=ms(data),
        forward_ms=ms(fwd),
        loss_ms=ms(loss_t),
        backward_ms=ms(bwd),
        step_ms=ms(step),
        data_median_ms=med(data),
        forward_median_ms=med(fwd),
        loss_median_ms=med(loss_t),
        backward_median_ms=med(bwd),
        step_median_ms=med(step),
    )


def format_train_step_profile(p: TrainStepProfile) -> str:
    """Human-readable per-phase block (mean / median ms) for the speed A/B notebook."""
    total = p.data_ms + p.forward_ms + p.loss_ms + p.backward_ms + p.step_ms

    def pct(x: float) -> float:
        return (100.0 * x / total) if total else 0.0

    rows = [
        ("data", p.data_ms, p.data_median_ms),
        ("forward", p.forward_ms, p.forward_median_ms),
        ("loss", p.loss_ms, p.loss_median_ms),
        ("backward", p.backward_ms, p.backward_median_ms),
        ("step", p.step_ms, p.step_median_ms),
    ]
    lines = [
        f"train-step profile  model={p.model_name}  device={p.device}  "
        f"({p.n_batches} warm batches)"
    ]
    for name, mean, median in rows:
        lines.append(
            f"  {name:<9} {mean:7.2f} ms mean / {median:7.2f} ms median ({pct(mean):4.0f}%)"
        )
    lines.append(f"  {'total':<9} {total:7.2f} ms/step")
    return "\n".join(lines)
