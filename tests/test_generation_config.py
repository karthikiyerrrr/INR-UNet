"""GenerationConfig is part of the structured schema and loads from YAML."""

from pathlib import Path

from inr_unet.config import (
    DataConfig,
    GenerationConfig,
    SamplerConfig,
    SyntheticDatasetConfig,
    load_config,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_generation_defaults():
    cfg = GenerationConfig()
    assert cfg.potential_backend == "z_power"
    assert cfg.sigma_potential_A == 0.4
    assert cfg.aperture_soft is True


def test_default_yaml_has_generation():
    cfg = load_config(CONFIG_PATH)
    assert cfg.generation.potential_backend == "z_power"
    assert cfg.generation.sigma_potential_A == 0.4


def test_sampler_defaults():
    cfg = SamplerConfig()
    assert cfg.fov_set_A == [8.0, 10.0, 20.0, 30.0, 40.0]
    assert cfg.pixel_size_A_min == 0.05
    assert cfg.pixel_size_A_max == 0.30
    assert cfg.z_exponent == 1.7
    assert cfg.n_peak_max == 3000.0
    assert cfg.conditions == ["cond1", "cond2", "cond3", "cond4", "cond5"]


def test_generation_has_sampler():
    cfg = GenerationConfig()
    assert cfg.sampler.n_peak_min == 30.0
    assert cfg.sampler.rotation_step_deg == 15.0


def test_default_yaml_has_sampler():
    cfg = load_config(CONFIG_PATH)
    assert cfg.generation.sampler.fov_set_A == [8.0, 10.0, 20.0, 30.0, 40.0]
    assert cfg.generation.sampler.z_exponent == 1.7
    assert cfg.generation.sampler.bg_weights["nonlinear"] == 1.0


def test_synthetic_dataset_defaults():
    cfg = SyntheticDatasetConfig()
    assert cfg.n_scenes == 64
    assert cfg.draws_per_scene == 16
    assert cfg.crop_size == 48
    assert cfg.sample_q == 2304
    assert cfg.label_kind == "gaussian"
    assert cfg.target_pixel_size_A_min == 0.05
    assert cfg.target_pixel_size_A_max == 0.30
    assert cfg.master_seed == 0


def test_data_has_synthetic():
    cfg = DataConfig()
    assert cfg.synthetic.crop_size == 48
    assert cfg.synthetic.label_kind == "gaussian"


def test_default_yaml_has_synthetic():
    cfg = load_config(CONFIG_PATH)
    assert cfg.data.synthetic.n_scenes == 64
    assert cfg.data.synthetic.crop_size == 48
    assert cfg.data.synthetic.sample_q == 2304
