"""save_training_run/load_training_run round-trip; run_kind discriminates bundles."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from runs import load_training_run, run_kind, save_training_run  # noqa: E402


def _history():
    return pl.DataFrame({"epoch": [0, 1], "train_loss": [0.5, 0.3], "val_f1": [0.1, 0.4]})


def test_training_bundle_round_trip(tmp_path):
    run_dir = tmp_path / "20260605-000000-train"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"x")
    meta = {"run_id": run_dir.name, "purpose": "training", "best_val_f1": 0.4}
    panels = {"input": np.zeros((2, 4, 4)), "gt": np.zeros((2, 4, 4)), "pred": np.zeros((2, 4, 4))}
    save_training_run(run_dir, meta=meta, config=OmegaConf.create({"a": 1}),
                      history=_history(), val_panels=panels,
                      test_metrics={"f1": 0.42})
    art = load_training_run(run_dir)
    assert art.meta["purpose"] == "training"
    assert art.history["val_f1"].to_list() == [0.1, 0.4]
    assert art.val_panels["input"].shape == (2, 4, 4)
    assert art.checkpoints["best"].name == "best.pt"
    assert art.test_metrics == {"f1": 0.42}
    assert run_kind(run_dir) == "training"


def test_save_training_run_updates_in_place(tmp_path):
    run_dir = tmp_path / "run"
    meta = {"purpose": "training"}
    save_training_run(run_dir, meta=meta, config=OmegaConf.create({}),
                      history=_history(), val_panels={})
    # second call (resume finalize) must not raise on an existing dir
    save_training_run(run_dir, meta={**meta, "best_val_f1": 0.9}, config=OmegaConf.create({}),
                      history=_history(), val_panels={})
    assert load_training_run(run_dir).meta["best_val_f1"] == 0.9
