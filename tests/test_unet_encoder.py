"""Contract tests for UNetEncoder: shapes, arbitrary input sizes, channel schedule."""

import torch
from omegaconf import OmegaConf

from inr_unet.models.components.unet import UNetEncoder


def _cfg(**over):
    base = {"name": "unet", "in_channels": 1, "base_channels": 8, "depth": 3, "feature_dim": 16}
    base.update(over)
    return OmegaConf.create(base)


def test_encoder_returns_feature_map_at_input_resolution():
    enc = UNetEncoder(_cfg())
    out = enc(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 16, 64, 64)


def test_encoder_handles_non_multiple_of_2_pow_depth():
    # depth=3 -> needs multiple of 8; 50x70 is not, must still work and return original size
    enc = UNetEncoder(_cfg())
    out = enc(torch.randn(1, 1, 50, 70))
    assert out.shape == (1, 16, 50, 70)


def test_encoder_channel_schedule_follows_config():
    enc = UNetEncoder(_cfg(base_channels=16, depth=2, feature_dim=32))
    out = enc(torch.randn(1, 1, 32, 32))
    assert out.shape == (1, 32, 32, 32)


def test_encoder_gradient_flows():
    enc = UNetEncoder(_cfg())
    out = enc(torch.randn(1, 1, 32, 32))
    out.sum().backward()
    assert all(p.grad is not None for p in enc.parameters())
