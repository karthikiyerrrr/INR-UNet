"""Read/write the runs/<run_id>/ artifact bundle shared by the Colab and marimo notebooks.

A run directory holds:
  meta.json               run metadata (run_id, created, git_sha, gpu_name, torch_version, purpose)
  config.yaml             the resolved ExperimentConfig (OmegaConf)
  overfit_losses.parquet  polars [step, loss, bce, dice]
  profile.parquet         polars [sweep, knob, value, fwd_ms, bwd_ms, step_ms, steps_per_s,
                                  samples_per_s, peak_mb, oom]
  overfit_sample.npz      float32 arrays: input[H,W], gt[H,W], pred[H,W]
"""

from __future__ import annotations

import json
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
    run_dir,
    *,
    meta: dict,
    config: DictConfig,
    losses: pl.DataFrame,
    profile: pl.DataFrame,
    sample: dict[str, np.ndarray],
) -> Path:
    """Write a complete run bundle to ``run_dir``; refuse to overwrite an existing run."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run dir already exists, refusing to overwrite: {run_dir}")
    missing = [k for k in _SAMPLE_KEYS if k not in sample]
    if missing:
        raise KeyError(f"sample is missing required arrays: {missing}")

    run_dir.mkdir(parents=True)
    (run_dir / _META).write_text(json.dumps(meta, indent=2, sort_keys=True))
    OmegaConf.save(config, run_dir / _CONFIG)
    losses.write_parquet(run_dir / _LOSSES)
    profile.write_parquet(run_dir / _PROFILE)
    np.savez(run_dir / _SAMPLE, **{k: np.asarray(sample[k], dtype="float32") for k in _SAMPLE_KEYS})
    return run_dir


def load_run(run_dir) -> RunArtifacts:
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


def list_runs(runs_root) -> list[Path]:
    """Return run directories under ``runs_root`` (those with a meta.json), sorted by name."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    return sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / _META).exists())
