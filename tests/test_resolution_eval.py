"""decode_dense produces N×N heatmaps for both models; score_tile uses physical tolerances."""

import numpy as np
import torch

from inr_unet.config import load_config
from inr_unet.registry import build_model
from inr_unet.resolution_eval import decode_dense, score_tile


def _small_cfg():
    cfg = load_config("configs/default.yaml")
    cfg.data.provider = "synthetic"
    cfg.data.synthetic.n_scenes = 3
    cfg.data.synthetic.draws_per_scene = 2
    cfg.model.encoder.base_channels = 8
    cfg.model.encoder.depth = 3
    cfg.model.encoder.feature_dim = 16
    cfg.model.decoder.hidden_dim = 32
    cfg.model.decoder.num_layers = 3
    return cfg


def _img():
    return torch.rand(1, 1, 128, 128)


def test_decode_dense_inr_shapes():
    cfg = _small_cfg()
    model = build_model(cfg)
    for n in (64, 128, 256):
        hm = decode_dense(model, _img(), n)
        assert hm.shape == (n, n)
        assert hm.min() >= 0.0 and hm.max() <= 1.0


def test_decode_dense_baseline_shapes():
    cfg = _small_cfg()
    cfg.model.name = "unet_baseline"
    model = build_model(cfg)
    for n in (64, 128, 256):
        hm = decode_dense(model, _img(), n)
        assert hm.shape == (n, n)


def test_decode_dense_inr_at_128_matches_native():
    cfg = _small_cfg()
    model = build_model(cfg).eval()
    img = _img()
    with torch.no_grad():
        native = torch.sigmoid(model(img))[0, 0]  # native dense branch
    hm = decode_dense(model, img, 128)
    assert torch.allclose(hm, native, atol=1e-5)


def _gaussian_hm(n, extent_A, center_A, sigma_A=0.4):
    """An n×n heatmap with a Gaussian peak centered at physical center_A (x, y)."""
    px = extent_A / n
    ys = (np.arange(n) + 0.5) * px
    xx, yy = np.meshgrid(ys, ys)  # xx varies along cols, yy along rows
    g = np.exp(-(((xx - center_A[0]) ** 2 + (yy - center_A[1]) ** 2) / (2 * sigma_A**2)))
    return torch.from_numpy(g.astype("float32"))


def test_score_tile_physical_tolerance_invariant_across_resolution():
    extent_A = 20.0
    center = (10.0, 10.0)
    gt = torch.tensor([[10.0, 10.0]])
    res = {}
    for n in (64, 128, 256):
        hm = _gaussian_hm(n, extent_A, center)
        res[n] = score_tile(hm, gt, extent_A, match_tol_A=0.5, min_distance_A=0.5, threshold=0.5)
    # one physical peak, fixed Å tolerance -> detected and matched at every resolution
    for n in (64, 128, 256):
        assert res[n]["n_matched"] == 1
        assert res[n]["f1"] == 1.0


def test_sweep_output_resolution_frame():
    from inr_unet.data import LIIFSegDataset
    from inr_unet.resolution_eval import sweep_output_resolution

    cfg = _small_cfg()
    ds = LIIFSegDataset(cfg)
    inr = build_model(cfg)
    base_cfg = _small_cfg()
    base_cfg.model.name = "unet_baseline"
    base = build_model(base_cfg)
    sizes = [64, 128]
    df = sweep_output_resolution(inr, base, ds, [0, 1], sizes,
                                 match_tol_A=0.3, min_distance_A=0.3)
    assert set(df["model"].unique()) == {"inr_unet", "unet_baseline"}
    assert sorted(df["output_size"].unique()) == sizes
    assert df.height == 2 * len(sizes)  # 2 models × 2 sizes
    for col in ("f1", "precision", "recall", "micro_precision", "n_tiles", "n_empty"):
        assert col in df.columns
    assert (df["n_tiles"] == 2).all()


def test_make_resolution_panels_shapes():
    from inr_unet.data import LIIFSegDataset
    from inr_unet.resolution_eval import make_resolution_panels

    cfg = _small_cfg()
    ds = LIIFSegDataset(cfg)
    inr = build_model(cfg)
    base_cfg = _small_cfg()
    base_cfg.model.name = "unet_baseline"
    base = build_model(base_cfg)
    panels = make_resolution_panels(inr, base, ds, [0, 1], [64, 128])
    assert panels["input"].shape == (2, 128, 128)
    assert panels["extent_A"].shape == (2,)
    assert panels["inr_64"].shape == (2, 64, 64)
    assert panels["base_128"].shape == (2, 128, 128)


def test_sweep_input_fov_frame():
    from inr_unet.resolution_eval import sweep_input_fov

    cfg = _small_cfg()
    inr = build_model(cfg)
    base_cfg = _small_cfg()
    base_cfg.model.name = "unet_baseline"
    base = build_model(base_cfg)
    fovs = [12.0, 24.0]
    df = sweep_input_fov(inr, base, cfg, fovs, [0, 1],
                         match_tol_A=0.3, min_distance_A=0.3)
    assert set(df["model"].unique()) == {"inr_unet", "unet_baseline"}
    assert sorted(df["fov_A"].unique()) == fovs
    assert df.height == 2 * len(fovs)
    assert (df["n_tiles"] == 2).all()
    assert "mean_offset_A" in df.columns
