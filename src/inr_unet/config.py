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
    # S - encoder input; each tile is a physical window resampled to S px
    crop_size: int = 48
    sample_q: int = 2304             # Q query points (independent of crop_size)
    label_kind: str = "gaussian"     # LABEL_FIELDS key
    gaussian_fwhm_A: float = 0.2
    gaussian_sigma_floor_px: float = 1.0
    target_pixel_size_A_min: float = 0.05
    target_pixel_size_A_max: float = 0.30
    # physical-extent tile: each crop spans a bounded Angstrom window, resampled to crop_size px
    tile_fov_A_min: float = 12.0
    tile_fov_A_max: float = 32.0
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
    z_eff_exponent: float = 1.7          # exponent for the projection z_eff collapse
    partial_fov_prob: float = 0.0        # P(render a small supercell in a band vs fill the FOV)
    supercell_nx_range: list[int] = field(default_factory=lambda: [1, 3])
    supercell_ny_range: list[int] = field(default_factory=lambda: [1, 2])
    strip_prob: float = 0.0  # P(elongated row-strip | partial); rest of partials are compact blobs


@dataclass
class OccupancyConfig:
    mode: str = "full"  # "full" | "facet_polygon" | "blob" | "mix"
    mix_weights: dict[str, float] = field(
        default_factory=lambda: {"full": 0.87, "facet_polygon": 0.10, "blob": 0.03}
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
    loss: str = "weighted_mse"   # {"weighted_mse", "dice_bce"}
    fg_weight: float = 10.0      # foreground weight lambda in weighted_mse_loss


@dataclass
class SamplerConfig:
    # scale
    fov_set_A: list[float] = field(default_factory=lambda: [8.0, 10.0, 20.0, 30.0, 40.0])
    # pixel size is a free log-uniform variable (the resolution-agnostic / LIIF extension);
    # Lin2021 implicitly fixed pixel size at FOV / 256.
    pixel_size_A_min: float = 0.05
    pixel_size_A_max: float = 0.30
    # FOV and pixel size are drawn independently, so a small FOV with coarse pixels can imply
    # an unrealistically tiny image. Bound the resulting output_size to realistic S/TEM sizes
    # (the reference dataset is 256 px); when the bound bites, pixel size is back-solved from
    # output_size = FOV / pixel_size so the FOV/zoom is preserved.
    output_size_min: int = 128
    output_size_max: int = 512
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
        default_factory=lambda: {
            "constant": 1.0, "linear_ramp": 1.0, "nonlinear": 1.0, "perlin": 1.0
        }
    )
    bg_constant_c_min: float = 0.05
    bg_constant_c_max: float = 0.40
    bg_ramp_c0_min: float = 0.05
    bg_ramp_c0_max: float = 0.30
    bg_ramp_g_min: float = 0.10
    bg_ramp_g_max: float = 0.60
    bg_nonlinear_blobs_min: int = 2
    bg_nonlinear_blobs_max: int = 5
    bg_perlin_amp_min: float = 0.05
    bg_perlin_amp_max: float = 0.35
    bg_perlin_cells_min: int = 2
    bg_perlin_cells_max: int = 6
    # noise ranges
    n_peak_min: float = 30.0
    n_peak_max: float = 3000.0
    n_bg_frac_max: float = 0.05
    scan_freq_min: float = 0.02
    scan_freq_max: float = 0.5
    scan_beta_min: float = 0.3
    scan_beta_max: float = 0.5
    # per-render ADF intensity weighting (count * Z**z_exponent); sampled over [min, max]
    # as an augmentation (measured HAADF exponents span ~1.5-1.9; central estimate 1.7).
    # Distinct from data.cif.z_eff_exponent, which collapses mixed columns at projection time.
    z_exponent_min: float = 1.5
    z_exponent_max: float = 2.0
    # per-render aberration draws (extensions beyond Lin2021, which used zero aberrations).
    # Pair sampled A1 with nonzero defocus: A1 produces no ellipticity at exact focus. 0.0
    # disables that draw.
    defocus_A_max: float = 40.0   # defocus ~ U(-defocus_A_max, +defocus_A_max)
    astig_a1_A_max: float = 40.0  # 2-fold astigmatism magnitude ~ U(0, astig_a1_A_max)
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
