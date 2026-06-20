"""save_resolution_sweep/load_resolution_sweep round-trip; run_kind tags resolution bundles."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from runs import (  # noqa: E402
    load_resolution_sweep,
    run_kind,
    save_resolution_sweep,
)


def _frame(axis):
    return pl.DataFrame({"model": ["inr_unet", "unet_baseline"], axis: [64, 64], "f1": [0.5, 0.6]})


def test_resolution_bundle_round_trip(tmp_path):
    run_dir = tmp_path / "20260620-000000-resgrid"
    meta = {"run_id": run_dir.name, "purpose": "resolution",
            "inr_run_id": "a", "baseline_run_id": "b"}
    panels = {"input": np.zeros((1, 8, 8), dtype="float32"),
              "extent_A": np.array([20.0], dtype="float32"),
              "inr_64": np.zeros((1, 64, 64), dtype="float32")}
    save_resolution_sweep(run_dir, meta=meta, config=OmegaConf.create({"x": 1}),
                          output_sweep=_frame("output_size"), fov_sweep=_frame("fov_A"),
                          panels=panels)
    art = load_resolution_sweep(run_dir)
    assert art.meta["purpose"] == "resolution"
    assert art.output_sweep["f1"].to_list() == [0.5, 0.6]
    assert art.fov_sweep["fov_A"].to_list() == [64, 64]
    assert art.panels["inr_64"].shape == (1, 64, 64)
    assert run_kind(run_dir) == "resolution"


def test_resolution_bundle_without_panels(tmp_path):
    run_dir = tmp_path / "20260620-000001-resgrid"
    save_resolution_sweep(run_dir, meta={"purpose": "resolution"}, config=OmegaConf.create({}),
                          output_sweep=_frame("output_size"), fov_sweep=_frame("fov_A"))
    art = load_resolution_sweep(run_dir)
    assert art.panels == {}
