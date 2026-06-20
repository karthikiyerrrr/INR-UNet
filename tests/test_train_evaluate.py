"""evaluate() runs the dense path, aggregates macro/micro metrics, and computes val loss."""

import math

from inr_unet.config import load_config
from inr_unet.data import LIIFSegDataset, STEMSegDataset
from inr_unet.registry import build_model
from inr_unet.train import PATHS, EvalMetrics, evaluate

CONFIG = "configs/default.yaml"


def _small_cfg():
    cfg = load_config(CONFIG)
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 3
    cfg.data.synthetic.draws_per_scene = 2
    cfg.data.synthetic.sample_q = 64
    cfg.model.encoder.base_channels = 8
    cfg.model.encoder.depth = 3
    cfg.model.encoder.feature_dim = 16
    cfg.model.decoder.hidden_dim = 32
    cfg.model.decoder.num_layers = 3
    return cfg


def test_evaluate_returns_metrics_in_range():
    cfg = _small_cfg()
    ds = LIIFSegDataset(cfg)
    model = build_model(cfg)
    m = evaluate(model, ds, [0, 1, 2], cfg, device="cpu")
    assert isinstance(m, EvalMetrics)
    assert m.n_tiles == 3
    assert 0.0 <= m.precision <= 1.0
    assert 0.0 <= m.recall <= 1.0
    assert 0.0 <= m.f1 <= 1.0
    assert m.loss >= 0.0
    assert 0.0 <= m.micro_precision <= 1.0
    # offsets are NaN only if nothing matched; otherwise non-negative
    assert math.isnan(m.mean_offset_A) or m.mean_offset_A >= 0.0


def test_evaluate_handles_all_empty_tiles():
    # a model that always predicts background -> n_pred 0; empty/full tiles both score cleanly
    cfg = _small_cfg()
    ds = LIIFSegDataset(cfg)

    class Zero:
        def eval(self):
            return self

        def __call__(self, *a, **k):
            import torch
            # query path -> [B,Q,1]; dense path -> [B,1,H,W]
            if len(a) >= 2 and a[1] is not None:
                return torch.full_like(a[1][..., :1], -20.0)
            img = a[0]
            return torch.full((img.shape[0], 1, img.shape[-2], img.shape[-1]), -20.0)

    m = evaluate(Zero(), ds, [0, 1], cfg, device="cpu")
    assert m.n_tiles == 2
    assert m.micro_precision in (0.0, 1.0)
    # the model predicts no peaks -> nothing matches -> no offsets collected -> NaN
    assert math.isnan(m.mean_offset_A)
    assert math.isnan(m.median_offset_A)


def test_evaluate_baseline_metrics_in_range():
    cfg = _small_cfg()
    cfg.model.name = "unet_baseline"
    ds = STEMSegDataset(cfg)
    model = build_model(cfg)
    m = evaluate(model, ds, [0, 1, 2], cfg, device="cpu", path=PATHS["unet_baseline"])
    assert isinstance(m, EvalMetrics)
    assert m.n_tiles == 3
    assert math.isfinite(m.loss)
    assert 0.0 <= m.precision <= 1.0
    assert 0.0 <= m.recall <= 1.0
    assert 0.0 <= m.f1 <= 1.0


def test_evaluate_baseline_all_background():
    cfg = _small_cfg()
    cfg.model.name = "unet_baseline"
    ds = STEMSegDataset(cfg)

    class Zero:
        def eval(self):
            return self

        def __call__(self, *a, **k):
            import torch
            img = a[0]
            return torch.full((img.shape[0], 1, img.shape[-2], img.shape[-1]), -20.0)

    m = evaluate(Zero(), ds, [0, 1], cfg, device="cpu", path=PATHS["unet_baseline"])
    assert m.n_tiles == 2
    assert m.micro_precision in (0.0, 1.0)
    assert math.isnan(m.mean_offset_A)


def test_aggregate_localization_matches_inline():
    from inr_unet.train import aggregate_localization
    per_tile = [
        {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "mean_offset_A": 0.1,
         "n_pred": 1, "n_gt": 2, "n_matched": 1},
        {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_offset_A": float("nan"),
         "n_pred": 1, "n_gt": 0, "n_matched": 0},  # empty tile (n_gt == 0)
    ]
    m = aggregate_localization(per_tile, loss=0.25)
    assert m.n_tiles == 2
    assert m.n_empty == 1
    assert m.loss == 0.25
    assert abs(m.precision - 0.5) < 1e-6          # macro mean of [1.0, 0.0]
    assert abs(m.micro_precision - 0.5) < 1e-6    # 1 matched / 2 pred
    assert abs(m.micro_recall - 0.5) < 1e-6       # 1 matched / 2 gt
    assert abs(m.mean_offset_A - 0.1) < 1e-6      # only the non-empty tile contributes
