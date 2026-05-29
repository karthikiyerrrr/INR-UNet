"""GenerationConfig is part of the structured schema and loads from YAML."""

from pathlib import Path

from inr_unet.config import GenerationConfig, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_generation_defaults():
    cfg = GenerationConfig()
    assert cfg.potential_backend == "z_power"
    assert cfg.sigma_potential_A == 0.4
    assert cfg.z_exponent == 1.7
    assert cfg.aperture_soft is True


def test_default_yaml_has_generation():
    cfg = load_config(CONFIG_PATH)
    assert cfg.generation.potential_backend == "z_power"
    assert cfg.generation.sigma_potential_A == 0.4
