"""Training loop for INRUNet: structure-stratified splits, resumable multi-scene
training on the LIIF query path, and dense peak-localization evaluation."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from inr_unet.data import LIIFSegDataset
from inr_unet.data.generation.structures import Grid
from inr_unet.localization import peak_localization
from inr_unet.losses import make_loss
from inr_unet.profiling import format_profile, profile_datagen
from inr_unet.registry import build_model

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
        # CPU torch RNG only; the model has no CUDA-only stochastic ops (no dropout), so CUDA
        # RNG need not be saved for bit-exact resume.
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


def build_train_loader(dataset, indices: list[int], cfg: DictConfig, generator, device: str
                       ) -> DataLoader:
    """Training DataLoader over ``indices``. Worker knobs (persistent_workers, prefetch_factor)
    are valid only when num_workers > 0; pin_memory is applied only on CUDA."""
    nw = int(cfg.data.num_workers)
    kwargs = dict(
        batch_size=int(cfg.data.batch_size), shuffle=True, generator=generator,
        num_workers=nw, pin_memory=bool(cfg.data.pin_memory) and device == "cuda",
    )
    if nw > 0:
        kwargs["persistent_workers"] = bool(cfg.data.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
    return DataLoader(Subset(dataset, indices), **kwargs)


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
            s = dataset.source.get(idx)
            coords, cell, gt = (t[None].to(device) for t in dataset.query_from_sample(s, idx))
            img = s.image[None, None].to(device)
            total_loss += float(loss_fn(model(img, coords, cell), gt))
            n += 1
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


def _mmss(seconds: float) -> str:
    """Format a duration as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, cfg: DictConfig, device: str,
                    *, epoch: int | None = None) -> tuple[float, float]:
    """One pass over ``loader`` on the LIIF query path.

    Returns ``(mean per-batch loss, wall-time seconds)``. Prints first-batch latency and a periodic
    heartbeat (running loss, samples/s, ETA). Cadence is ``cfg.train.log_every`` batches when > 0,
    else ~20 lines/epoch; < 0 disables all per-batch printing.
    """
    model.train()
    accum = max(1, int(cfg.train.grad_accum_steps))
    use_amp = bool(cfg.train.amp) and device == "cuda"
    clip = float(cfg.train.grad_clip_norm)
    autocast_device = "cuda" if device == "cuda" else "cpu"
    n_batches = len(loader)
    log_every = int(cfg.train.log_every)
    quiet = log_every < 0
    stride = log_every if log_every > 0 else max(1, n_batches // 20)
    prefix = f"[epoch {epoch + 1}]" if epoch is not None else "[train]"
    total, n, seen = 0.0, 0, 0
    t_epoch = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for i, (img, coords, cell, gt) in enumerate(loader):
        img, coords, cell, gt = (t.to(device) for t in (img, coords, cell, gt))
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_amp):
            loss = loss_fn(model(img, coords, cell), gt)
        (loss / accum).backward()
        if (i + 1) % accum == 0 or (i + 1) == n_batches:
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach())
        n += 1
        seen += int(img.shape[0])
        if i == 0 and not quiet:
            print(f"{prefix} first batch in {time.perf_counter() - t_epoch:.1f}s "
                  f"(worker spin-up + first render)", flush=True)
        last_batch = (i + 1) == n_batches
        # heartbeat on stride boundaries and always on the last batch; skip i==0 (the first-batch
        # line already covered it) unless that lone batch is also the last (single-batch epoch).
        if not quiet and (last_batch or (i != 0 and (i + 1) % stride == 0)):
            elapsed = time.perf_counter() - t_epoch
            sps = seen / elapsed if elapsed > 0 else 0.0
            eta = elapsed * (n_batches - (i + 1)) / (i + 1)
            print(f"{prefix} batch {i + 1}/{n_batches} | loss={total / n:.4f} | "
                  f"{sps:.1f} samp/s | elapsed {_mmss(elapsed)} | ETA {_mmss(eta)}", flush=True)
    return total / max(1, n), time.perf_counter() - t_epoch


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_mean_offset_A: float
    val_median_offset_A: float
    lr: float
    epoch_time_s: float


@dataclass(frozen=True)
class TrainResult:
    history: list[EpochRecord]
    best_epoch: int
    best_val_f1: float
    best_checkpoint: Path | None   # None if no eval epoch ran (best.pt was never written)
    val_panels: dict[str, np.ndarray]


def _make_panels(model, dataset, indices: list[int], device: str, k: int
                 ) -> dict[str, np.ndarray]:
    """Collect up to ``k`` (input, gt, pred) dense triplets from val tiles for visualization."""
    model.eval()
    ins, gts, preds = [], [], []
    with torch.no_grad():
        for idx in indices[:k]:
            s = dataset.source.get(idx)
            grid = Grid(dataset.source.crop_size, s.input_pixel_size_A,
                        device=str(s.image.device))
            gt = dataset.label_field.rasterize(s.positions_A, s.radii_A, grid)
            pred = torch.sigmoid(model(s.image[None, None].to(device)))[0, 0].cpu()
            ins.append(s.image.cpu().numpy())
            gts.append(gt.cpu().numpy())
            preds.append(pred.numpy())
    if not ins:
        empty = np.zeros((0, 0, 0), dtype="float32")
        return {"input": empty, "gt": empty, "pred": empty}
    return {"input": np.stack(ins), "gt": np.stack(gts), "pred": np.stack(preds)}


def train(cfg: DictConfig, *, run_dir, resume_from=None) -> TrainResult:
    """Train INRUNet on the LIIF query path with per-epoch dense val eval, val-F1 selection,
    early stopping, and resumable checkpoints. ``run_dir/checkpoints/{last,best}.pt`` are written
    every epoch (put run_dir on persistent storage so a dropped session can resume_from last.pt).

    An ``EpochRecord`` (and progress line) is emitted only on eval epochs, so with
    ``eval_every > 1`` the history is sparse (one row per eval, not per epoch); ``last.pt`` is
    still written every epoch for resume.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg.train.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(cfg).to(device)
    dataset = LIIFSegDataset(cfg)
    splits = build_splits(cfg)
    gen = torch.Generator()
    loader = build_train_loader(dataset, splits.train, cfg, gen, device)
    loss_fn = make_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    accum = max(1, int(cfg.train.grad_accum_steps))
    steps_per_epoch = math.ceil(len(loader) / accum)
    epochs = int(cfg.train.epochs)
    scheduler = build_scheduler(optimizer, cfg, total_steps=epochs * steps_per_epoch)

    start_epoch, best_val_f1, best_epoch = 0, -1.0, -1
    history: list[EpochRecord] = []
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_f1 = float(ckpt["best_val_f1"])
        best_epoch = int(ckpt["best_epoch"])
        history = [EpochRecord(**r) for r in ckpt["history"]]
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])

    eval_every = max(1, int(cfg.train.eval_every))
    patience = int(cfg.train.early_stop_patience)
    if int(cfg.train.profile_datagen_samples) > 0:
        print(format_profile(profile_datagen(cfg, n=int(cfg.train.profile_datagen_samples))),
              flush=True)
    for epoch in range(start_epoch, epochs):
        gen.manual_seed(seed + epoch)
        train_loss, epoch_time_s = train_one_epoch(
            model, loader, loss_fn, optimizer, scheduler, cfg, device, epoch=epoch)
        is_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
        improved = False
        if is_eval:
            m = evaluate(model, dataset, splits.val, cfg, device)
            improved = m.f1 > best_val_f1
            if improved:
                best_val_f1, best_epoch = m.f1, epoch
            history.append(EpochRecord(
                epoch=epoch, train_loss=train_loss, val_loss=m.loss,
                val_precision=m.precision, val_recall=m.recall, val_f1=m.f1,
                val_mean_offset_A=m.mean_offset_A, val_median_offset_A=m.median_offset_A,
                lr=optimizer.param_groups[0]["lr"], epoch_time_s=epoch_time_s,
            ))
            print(
                f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={m.loss:.4f}  val_f1={m.f1:.4f}  best_f1={best_val_f1:.4f}  "
                f"epoch_time={_mmss(epoch_time_s)}",
                flush=True,
            )
        history_dicts = [asdict(r) for r in history]
        save_checkpoint(
            ckpt_dir / "last.pt", epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, best_val_f1=best_val_f1, best_epoch=best_epoch,
            history=history_dicts,
        )
        if is_eval and improved:
            save_checkpoint(
                ckpt_dir / "best.pt", epoch=epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, best_val_f1=best_val_f1, best_epoch=best_epoch,
                history=history_dicts,
            )
        if is_eval and patience and (epoch - best_epoch) >= patience:
            break

    panels = _make_panels(model, dataset, splits.val, device, int(cfg.train.eval.n_val_panels))
    return TrainResult(
        history=history, best_epoch=best_epoch, best_val_f1=best_val_f1,
        best_checkpoint=(ckpt_dir / "best.pt") if best_epoch >= 0 else None,
        val_panels=panels,
    )
