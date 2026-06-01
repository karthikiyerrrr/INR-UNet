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
class SyntheticDatasetConfig:
    n_scenes: int = 64
    draws_per_scene: int = 16
    crop_size: int = 48              # S - encoder input; renders < S are reflect-padded
    sample_q: int = 2304             # Q = 48**2 query points
    label_kind: str = "gaussian"     # LABEL_FIELDS key
    target_pixel_size_A_min: float = 0.05
    target_pixel_size_A_max: float = 0.30
    master_seed: int = 0
    # finite-particle crop control (engaged only when occupancy.mode != "full")
    min_columns_in_crop: int = 1
    empty_crop_fraction: float = 0.15
    crop_redraw_cap: int = 16


@dataclass
class CIFConfig:
    manifest_path: str = "src/inr_unet/data/cif/manifest.yaml"
    group_tol_A: float = 0.4
    scene_fov_A_min: float = 60.0
    scene_fov_A_max: float = 80.0
    rotation_jitter_deg: float = 5.0


@dataclass
class OccupancyConfig:
    mode: str = "full"  # "full" | "facet_polygon" | "blob" | "mix"
    mix_weights: dict[str, float] = field(
        default_factory=lambda: {"facet_polygon": 1.0, "blob": 1.0, "full": 1.0}
    )
    support_frac_min: float = 0.3
    support_frac_max: float = 0.9
    n_facets_min: int = 4
    n_facets_max: int = 6
    facet_offset_jitter_frac: float = 0.15
    half_plane_prob: float = 0.15
    edge_clip_prob: float = 0.3
    blob_roughness: float = 0.2


@dataclass
class DataConfig:
    root: str = "data"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    provider: str = "synthetic"  # "synthetic" | "cif"
    synthetic: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)
    cif: CIFConfig = field(default_factory=CIFConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)


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
