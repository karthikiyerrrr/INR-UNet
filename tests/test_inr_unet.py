"""Contract tests for INRUNet (two forward modes + gradient) and BaselineUNet."""

import torch
from omegaconf import OmegaConf

from inr_unet.models.inr_unet import BaselineUNet, INRUNet


def _model_cfg(**enc_over):
    enc = {"name": "unet", "in_channels": 1, "base_channels": 8, "depth": 3, "feature_dim": 16}
    enc.update(enc_over)
    dec = {
        "name": "liif",
        "hidden_dim": 64,
        "num_layers": 3,
        "n_classes": 1,
        "local_ensemble": True,
        "feature_unfold": True,
        "cell_decode": True,
    }
    return OmegaConf.create({"name": "inr_unet", "encoder": enc, "decoder": dec})


def test_inr_unet_query_mode_shape():
    model = INRUNet(_model_cfg())
    img = torch.randn(2, 1, 32, 32)
    coords = torch.rand(2, 100, 2) * 2 - 1
    cell = torch.ones(2, 100, 2) / 32
    out = model(img, coords, cell)
    assert out.shape == (2, 100, 1)


def test_inr_unet_dense_mode_shape():
    model = INRUNet(_model_cfg())
    out = model(torch.randn(2, 1, 32, 48))  # coords/cell default to None
    assert out.shape == (2, 1, 32, 48)


def test_inr_unet_dense_matches_explicit_query():
    from inr_unet.models.components.liif import make_coord

    model = INRUNet(_model_cfg()).eval()
    img = torch.randn(1, 1, 16, 24)
    with torch.no_grad():
        dense = model(img)  # [1, 1, 16, 24]
        coords = make_coord((16, 24))[None]
        cell = torch.empty(1, coords.shape[1], 2)
        cell[:, :, 0] = 2.0 / 24
        cell[:, :, 1] = 2.0 / 16
        flat = model(img, coords, cell)  # [1, 384, 1]
    assert torch.allclose(dense.permute(0, 2, 3, 1).reshape(1, -1, 1), flat, atol=1e-5)


def test_inr_unet_gradient_reaches_encoder():
    model = INRUNet(_model_cfg())
    img = torch.randn(1, 1, 32, 32)
    coords = torch.rand(1, 50, 2) * 2 - 1
    cell = torch.ones(1, 50, 2) / 32
    model(img, coords, cell).sum().backward()
    enc_grads = [p.grad is not None for p in model.encoder.parameters()]
    assert all(enc_grads) and len(enc_grads) > 0


def test_baseline_unet_shape():
    model = BaselineUNet(_model_cfg())
    out = model(torch.randn(2, 1, 40, 40))
    assert out.shape == (2, 1, 40, 40)
