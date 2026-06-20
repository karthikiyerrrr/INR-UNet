"""Contract tests for eval-time LIIF probes."""

import numpy as np
import torch
from omegaconf import OmegaConf

from inr_unet.models.inr_unet import INRUNet
from inr_unet.probe import dense_heatmap, probe_sweep


def _model():
    cfg = OmegaConf.create(
        {
            "encoder": {
                "in_channels": 1,
                "feature_dim": 8,
                "depth": 2,
                "base_channels": 8,
            },
            "decoder": {
                "name": "liif",
                "hidden_dim": 16,
                "num_layers": 3,
                "n_classes": 1,
                "local_ensemble": True,
                "feature_unfold": True,
                "cell_decode": True,
            },
        }
    )
    torch.manual_seed(0)
    return INRUNet(cfg).eval()


def test_dense_heatmap_matches_model_forward():
    model = _model()
    img = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        ref = torch.sigmoid(model(img))[0, 0]
        got = dense_heatmap(model, img)
    assert got.shape == (16, 16)
    assert torch.allclose(got, ref, atol=1e-6)


def test_dense_heatmap_ensemble_override_changes_output():
    model = _model()
    img = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        on = dense_heatmap(model, img, local_ensemble=True)
        off = dense_heatmap(model, img, local_ensemble=False)
    assert not torch.allclose(on, off)
    # the override must not mutate the model's persistent attribute
    assert model.decoder.local_ensemble is True


def test_dense_heatmap_cell_scale_changes_output():
    model = _model()
    img = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        base = dense_heatmap(model, img, cell_scale=1.0)
        scaled = dense_heatmap(model, img, cell_scale=2.0)
    assert not torch.allclose(base, scaled)


class _Sample:
    def __init__(self, image, positions_A, input_pixel_size_A):
        self.image = image
        self.positions_A = positions_A
        self.input_pixel_size_A = input_pixel_size_A


class _Source:
    def __init__(self, samples):
        self._samples = samples

    def get(self, idx):
        return self._samples[idx]


class _Dataset:
    def __init__(self, samples):
        self.source = _Source(samples)


def _toy_dataset(n=2, size=16):
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(n):
        img = torch.from_numpy(rng.standard_normal((size, size)).astype("float32"))
        pos = torch.tensor([[4.0, 4.0], [9.0, 9.0]])  # Angstroms, in-FOV at px=1.0
        samples.append(_Sample(img, pos, 1.0))
    return _Dataset(samples)


def test_probe_sweep_one_row_per_config_with_expected_keys():
    model = _model()
    ds = _toy_dataset()
    configs = [
        {"cell_scale": 1.0, "local_ensemble": None, "label": "native"},
        {"cell_scale": 1.0, "local_ensemble": False, "label": "no_ensemble"},
        {"cell_scale": 2.0, "local_ensemble": None, "label": "cell2x"},
    ]
    rows = probe_sweep(model, ds, [0, 1], configs)
    assert [r["label"] for r in rows] == ["native", "no_ensemble", "cell2x"]
    for r in rows:
        assert set(r) >= {
            "label",
            "cell_scale",
            "local_ensemble",
            "median_offset_A",
            "median_fwhm",
            "median_height",
            "median_floor",
            "f1",
        }


def test_probe_sweep_is_deterministic():
    model = _model()
    ds = _toy_dataset()
    configs = [{"cell_scale": 1.0, "local_ensemble": None, "label": "native"}]
    a = probe_sweep(model, ds, [0, 1], configs)
    b = probe_sweep(model, ds, [0, 1], configs)
    assert a[0]["f1"] == b[0]["f1"]
    assert a[0]["median_height"] == b[0]["median_height"]
