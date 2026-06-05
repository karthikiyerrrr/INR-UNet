"""Round-trip and listing tests for the runs/<id>/ artifact I/O."""

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from inr_unet.config import ExperimentConfig
from notebooks.runs import RunArtifacts, list_runs, load_run, save_run


def _fixture_args():
    meta = {
        "run_id": "20260604-120000",
        "created": "2026-04-04T12:00:00",
        "git_sha": "abc1234",
        "gpu_name": "A100",
        "torch_version": "2.0.0",
        "purpose": "smoke_profile",
    }
    config = OmegaConf.structured(ExperimentConfig)
    losses = pl.DataFrame(
        {
            "step": [0, 1, 2],
            "loss": [1.0, 0.5, 0.1],
            "bce": [0.6, 0.3, 0.05],
            "dice": [0.4, 0.2, 0.05],
        }
    )
    profile = pl.DataFrame(
        {
            "sweep": ["baseline"],
            "knob": ["-"],
            "value": ["-"],
            "fwd_ms": [1.0],
            "bwd_ms": [2.0],
            "step_ms": [3.0],
            "steps_per_s": [333.0],
            "samples_per_s": [2664.0],
            "peak_mb": [1234.0],
            "oom": [False],
        }
    )
    sample = {
        "input": np.random.rand(8, 8).astype("float32"),
        "gt": np.random.rand(8, 8).astype("float32"),
        "pred": np.random.rand(8, 8).astype("float32"),
    }
    return meta, config, losses, profile, sample


def test_save_then_load_roundtrips_all_fields(tmp_path):
    meta, config, losses, profile, sample = _fixture_args()
    run_dir = tmp_path / "20260604-120000"
    save_run(run_dir, meta=meta, config=config, losses=losses, profile=profile, sample=sample)

    loaded = load_run(run_dir)
    assert isinstance(loaded, RunArtifacts)
    assert loaded.meta == meta
    assert OmegaConf.to_container(loaded.config) == OmegaConf.to_container(config)
    assert loaded.losses.equals(losses)
    assert loaded.profile.equals(profile)
    for key in ("input", "gt", "pred"):
        assert np.array_equal(loaded.sample[key], sample[key])


def test_list_runs_sorted_and_ignores_non_runs(tmp_path):
    meta, config, losses, profile, sample = _fixture_args()
    for run_id in ("20260604-090000", "20260604-120000", "20260603-235959"):
        save_run(tmp_path / run_id, meta={**meta, "run_id": run_id}, config=config,
                 losses=losses, profile=profile, sample=sample)
    (tmp_path / "not_a_run").mkdir()           # dir without meta.json -> ignored
    (tmp_path / "stray.txt").write_text("x")   # file -> ignored

    found = [p.name for p in list_runs(tmp_path)]
    assert found == ["20260603-235959", "20260604-090000", "20260604-120000"]


def test_list_runs_missing_root_returns_empty(tmp_path):
    assert list_runs(tmp_path / "does_not_exist") == []


def test_load_run_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope")


def test_load_run_missing_artifact_raises(tmp_path):
    meta, config, losses, profile, sample = _fixture_args()
    run_dir = tmp_path / "20260604-120000"
    save_run(run_dir, meta=meta, config=config, losses=losses, profile=profile, sample=sample)
    (run_dir / "profile.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        load_run(run_dir)


def test_save_run_refuses_overwrite(tmp_path):
    meta, config, losses, profile, sample = _fixture_args()
    run_dir = tmp_path / "20260604-120000"
    save_run(run_dir, meta=meta, config=config, losses=losses, profile=profile, sample=sample)
    with pytest.raises(FileExistsError):
        save_run(run_dir, meta=meta, config=config, losses=losses, profile=profile, sample=sample)


def test_save_run_missing_sample_key_raises(tmp_path):
    meta, config, losses, profile, sample = _fixture_args()
    del sample["pred"]
    with pytest.raises(KeyError):
        save_run(
            tmp_path / "r", meta=meta, config=config, losses=losses, profile=profile, sample=sample
        )
