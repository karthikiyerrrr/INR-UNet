"""Config defaults for the regression reframe + config-driven label-field construction."""

from inr_unet.config import load_config
from inr_unet.data.dataset import _build_label_field
from inr_unet.data.generation.labels import CircularField, GaussianField


def test_default_config_exposes_loss_and_sigma_fields():
    cfg = load_config("configs/default.yaml")
    assert cfg.train.loss == "weighted_mse"
    assert cfg.train.fg_weight == 10.0
    assert cfg.data.synthetic.gaussian_fwhm_A == 0.2
    assert cfg.data.synthetic.gaussian_sigma_floor_px == 1.0


def test_build_label_field_gaussian_uses_config():
    cfg = load_config("configs/default.yaml")
    cfg.data.synthetic.label_kind = "gaussian"
    cfg.data.synthetic.gaussian_fwhm_A = 0.4
    cfg.data.synthetic.gaussian_sigma_floor_px = 2.0
    field = _build_label_field(cfg.data.synthetic)
    assert isinstance(field, GaussianField)
    assert field.fwhm_A == 0.4
    assert field.sigma_floor_px == 2.0


def test_build_label_field_circular():
    cfg = load_config("configs/default.yaml")
    cfg.data.synthetic.label_kind = "circular"
    assert isinstance(_build_label_field(cfg.data.synthetic), CircularField)
