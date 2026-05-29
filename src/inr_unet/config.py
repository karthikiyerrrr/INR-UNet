"""OmegaConf-backed structured configuration schemas for INR-UNet."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class EncoderConfig:
    name: str = "unet"
    in_channels: int = 1
    base_channels: int = 64
    depth: int = 4
    feature_dim: int = 64


@dataclass
class DecoderConfig:
    name: str = "liif"
    hidden_dim: int = 256
    num_layers: int = 5
    n_classes: int = 1
    local_ensemble: bool = True
    feature_unfold: bool = True
    cell_decode: bool = True


@dataclass
class ModelConfig:
    name: str = "inr_unet"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class DataConfig:
    root: str = "data"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    seed: int = 0


@dataclass
class SamplerConfig:
    # scale
    fov_set_A: list[float] = field(default_factory=lambda: [8.0, 10.0, 20.0, 30.0, 40.0])
    pixel_size_A_min: float = 0.05
    pixel_size_A_max: float = 0.30
    # geometry
    rotation_step_deg: float = 15.0
    rotation_max_deg: float = 90.0
    offset_jitter_frac: float = 0.5
    # imaging condition
    conditions: list[str] = field(
        default_factory=lambda: ["cond1", "cond2", "cond3", "cond4", "cond5"]
    )
    # background family mix + amplitude ranges
    bg_weights: dict[str, float] = field(
        default_factory=lambda: {"constant": 1.0, "linear_ramp": 1.0, "nonlinear": 1.0}
    )
    bg_constant_c_min: float = 0.05
    bg_constant_c_max: float = 0.40
    bg_ramp_c0_min: float = 0.05
    bg_ramp_c0_max: float = 0.30
    bg_ramp_g_min: float = 0.10
    bg_ramp_g_max: float = 0.60
    bg_nonlinear_blobs_min: int = 2
    bg_nonlinear_blobs_max: int = 5
    # noise ranges
    n_peak_min: float = 30.0
    n_peak_max: float = 3000.0
    n_bg_frac_max: float = 0.05
    scan_freq_min: float = 0.02
    scan_freq_max: float = 0.5
    scan_beta_min: float = 0.3
    scan_beta_max: float = 0.5
    # physics constant (NOT randomized; visual-calibration knob)
    z_exponent: float = 1.7
    device: str = "cpu"


@dataclass
class GenerationConfig:
    potential_backend: str = "z_power"
    sigma_potential_A: float = 0.4
    aperture_soft: bool = True
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config and merge it over the structured ExperimentConfig schema."""
    schema = OmegaConf.structured(ExperimentConfig)
    user_cfg = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, user_cfg)
    assert isinstance(merged, DictConfig)
    return merged
