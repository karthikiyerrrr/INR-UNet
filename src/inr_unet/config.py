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
class GenerationConfig:
    potential_backend: str = "z_power"
    sigma_potential_A: float = 0.4
    z_exponent: float = 1.7
    aperture_soft: bool = True


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
