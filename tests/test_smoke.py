"""Smoke and interface-contract tests for the inr_unet package skeleton."""

import importlib
from pathlib import Path

import pytest
import torch

from inr_unet.config import load_config
from inr_unet.models import INRUNet
from inr_unet.registry import MODELS, build_model

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

MODULES = [
    "inr_unet",
    "inr_unet.config",
    "inr_unet.registry",
    "inr_unet.losses",
    "inr_unet.metrics",
    "inr_unet.data",
    "inr_unet.models",
    "inr_unet.models.components.blocks",
    "inr_unet.models.components.unet",
    "inr_unet.models.components.liif",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    """Every key submodule imports cleanly."""
    assert importlib.import_module(module) is not None


def test_models_registered():
    """INRUNet and the baseline are registered under the MODELS registry."""
    assert "inr_unet" in MODELS
    assert "unet_baseline" in MODELS


def test_load_default_config():
    """The default YAML merges over the structured schema."""
    cfg = load_config(CONFIG_PATH)
    assert cfg.model.name == "inr_unet"
    assert cfg.model.encoder.name == "unet"
    assert cfg.model.decoder.name == "liif"


def test_build_model_returns_inr_unet():
    """build_model instantiates an INRUNet from the default config."""
    cfg = load_config(CONFIG_PATH)
    model = build_model(cfg)
    assert isinstance(model, INRUNet)


def test_inr_unet_forward_shape():
    """INRUNet maps an image + query coordinates to per-coordinate logits."""
    cfg = load_config(CONFIG_PATH)
    model = build_model(cfg)
    img = torch.randn(2, 1, 64, 64)
    coords = torch.rand(2, 100, 2) * 2 - 1
    cell = torch.ones(2, 100, 2) / 64
    out = model(img, coords, cell)
    assert out.shape == (2, 100, 1)
