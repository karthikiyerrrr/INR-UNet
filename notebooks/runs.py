"""Read/write the runs/<run_id>/ artifact bundle shared by the Colab and marimo notebooks.

A run directory holds:
  meta.json               run metadata (run_id, created, git_sha, gpu_name, torch_version, purpose)
  config.yaml             the resolved ExperimentConfig (OmegaConf)
  overfit_losses.parquet  polars [step, loss, bce, dice]
  profile.parquet         polars [sweep, knob, value, fwd_ms, bwd_ms, step_ms, steps_per_s,
                                  samples_per_s, peak_mb, oom]
  overfit_sample.npz      float32 arrays: input[H,W], gt[H,W], pred[H,W]

Training-run bundles additionally hold:
  history.parquet         polars epoch-level metrics (epoch, train_loss, val_*, ...)
  checkpoints/            directory of *.pt checkpoint files written by the trainer
  val_panels.npz          float32 arrays of sample images from the final validation pass
  test_metrics.json       final held-out test metrics (optional)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

_META = "meta.json"
_CONFIG = "config.yaml"
_LOSSES = "overfit_losses.parquet"
_PROFILE = "profile.parquet"
_SAMPLE = "overfit_sample.npz"
_SAMPLE_KEYS = ("input", "gt", "pred")
_REQUIRED = (_META, _CONFIG, _LOSSES, _PROFILE, _SAMPLE)

_HISTORY = "history.parquet"
_VAL_PANELS = "val_panels.npz"
_TEST_METRICS = "test_metrics.json"
_TRAIN_REQUIRED = (_META, _CONFIG, _HISTORY)


@dataclass(frozen=True)
class RunArtifacts:
    """One loaded run: every artifact in memory, ready to plot."""

    run_dir: Path
    meta: dict
    config: DictConfig
    losses: pl.DataFrame
    profile: pl.DataFrame
    sample: dict[str, np.ndarray]


def save_run(
    run_dir: str | Path,
    *,
    meta: dict,
    config: DictConfig,
    losses: pl.DataFrame,
    profile: pl.DataFrame,
    sample: dict[str, np.ndarray],
) -> Path:
    """Write a complete run bundle to ``run_dir``; refuse to overwrite an existing run.

    The bundle is written to a temporary directory and atomically renamed into place,
    so a failed or interrupted write never leaves a partial run behind. Float arrays in
    ``sample`` are coerced to float32 on write.
    """
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run dir already exists, refusing to overwrite: {run_dir}")
    missing = [k for k in _SAMPLE_KEYS if k not in sample]
    if missing:
        raise KeyError(f"sample is missing required arrays: {missing}")

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=run_dir.parent))
    try:
        (tmp_dir / _META).write_text(json.dumps(meta, indent=2, sort_keys=True))
        OmegaConf.save(config, tmp_dir / _CONFIG)
        losses.write_parquet(tmp_dir / _LOSSES)
        profile.write_parquet(tmp_dir / _PROFILE)
        np.savez(
            tmp_dir / _SAMPLE,
            **{k: np.asarray(sample[k], dtype="float32") for k in _SAMPLE_KEYS},
        )
        os.replace(tmp_dir, run_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return run_dir


def load_run(run_dir: str | Path) -> RunArtifacts:
    """Load a run bundle written by :func:`save_run`; raise if anything is missing."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run directory: {run_dir}")
    for name in _REQUIRED:
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"run {run_dir} is missing artifact {name}")

    meta = json.loads((run_dir / _META).read_text())
    config = OmegaConf.load(run_dir / _CONFIG)
    losses = pl.read_parquet(run_dir / _LOSSES)
    profile = pl.read_parquet(run_dir / _PROFILE)
    with np.load(run_dir / _SAMPLE) as npz:
        sample = {k: npz[k] for k in _SAMPLE_KEYS}
    return RunArtifacts(run_dir, meta, config, losses, profile, sample)


def list_runs(runs_root: str | Path) -> list[Path]:
    """Return run directories under ``runs_root`` (those with a meta.json), sorted by name."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    return sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / _META).exists())


@dataclass(frozen=True)
class TrainingRunArtifacts:
    """One loaded training run: curves, config, sample panels, and checkpoint paths."""

    run_dir: Path
    meta: dict
    config: DictConfig
    history: pl.DataFrame
    val_panels: dict[str, np.ndarray]
    checkpoints: dict[str, Path]
    test_metrics: dict | None


def save_training_run(
    run_dir: str | Path,
    *,
    meta: dict,
    config: DictConfig,
    history: pl.DataFrame,
    val_panels: dict[str, np.ndarray],
    test_metrics: dict | None = None,
) -> Path:
    """Write/refresh a training bundle. Unlike :func:`save_run` this updates ``run_dir`` in place
    (the trainer has already written ``checkpoints/`` there and a resume finalize re-runs this),
    so it does not refuse an existing directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _META).write_text(json.dumps(meta, indent=2, sort_keys=True))
    OmegaConf.save(config, run_dir / _CONFIG)
    history.write_parquet(run_dir / _HISTORY)
    if val_panels:
        np.savez(
            run_dir / _VAL_PANELS,
            **{k: np.asarray(v, dtype="float32") for k, v in val_panels.items()},
        )
    if test_metrics is not None:
        (run_dir / _TEST_METRICS).write_text(json.dumps(test_metrics, indent=2, sort_keys=True))
    return run_dir


def load_training_run(run_dir: str | Path) -> TrainingRunArtifacts:
    """Load a training bundle written by :func:`save_training_run`."""
    run_dir = Path(run_dir)
    for name in _TRAIN_REQUIRED:
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"training run {run_dir} is missing artifact {name}")
    meta = json.loads((run_dir / _META).read_text())
    config = OmegaConf.load(run_dir / _CONFIG)
    history = pl.read_parquet(run_dir / _HISTORY)
    val_panels: dict[str, np.ndarray] = {}
    if (run_dir / _VAL_PANELS).exists():
        with np.load(run_dir / _VAL_PANELS) as npz:
            val_panels = {k: npz[k] for k in npz.files}
    ckpt_dir = run_dir / "checkpoints"
    checkpoints = (
        {p.stem: p for p in sorted(ckpt_dir.glob("*.pt"))} if ckpt_dir.is_dir() else {}
    )
    test_metrics = (
        json.loads((run_dir / _TEST_METRICS).read_text())
        if (run_dir / _TEST_METRICS).exists()
        else None
    )
    return TrainingRunArtifacts(
        run_dir, meta, config, history, val_panels, checkpoints, test_metrics
    )


def run_kind(run_dir: str | Path) -> str:
    """Classify a run bundle by ``meta['purpose']``: 'training' vs 'overfit' (default)."""
    meta = json.loads((Path(run_dir) / _META).read_text())
    return "training" if meta.get("purpose") == "training" else "overfit"
