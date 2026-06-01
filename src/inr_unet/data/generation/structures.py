"""Input/output data contracts and the pixel Grid for the forward model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Grid:
    """The render lattice: output_size px at pixel_size_A angstroms/px."""

    output_size: int
    pixel_size_A: float
    device: str = "cpu"

    @property
    def extent_A(self) -> float:
        return self.output_size * self.pixel_size_A

    def pixel_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pixel-center coordinates in pixels, returned as (yy, xx), each [H, W]."""
        ax = torch.arange(self.output_size, device=self.device, dtype=torch.float32)
        return torch.meshgrid(ax, ax, indexing="ij")

    def normalized_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Coordinates normalized to [-1, 1], returned as (yy, xx), each [H, W]."""
        ax = torch.linspace(-1.0, 1.0, self.output_size, device=self.device)
        return torch.meshgrid(ax, ax, indexing="ij")

    def freq_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Spatial frequencies in 1/A (fft layout), returned as (ky, kx), each [H, W]."""
        f = torch.fft.fftfreq(self.output_size, d=self.pixel_size_A, device=self.device)
        return torch.meshgrid(f, f, indexing="ij")


@dataclass(frozen=True)
class ColumnList:
    """Projected atomic columns from upstream CIF/zone-axis tooling.

    positions_A[:, 0] = x (horizontal), positions_A[:, 1] = y (vertical), origin at FOV corner.
    lattice_basis_A, when present, holds the two in-plane lattice vectors (rows, Angstroms)
    for the projected structure; downstream occupancy uses it to align facet edges.
    """

    positions_A: torch.Tensor  # [N, 2]
    z: torch.Tensor            # [N]
    count: torch.Tensor        # [N]
    fov_A: float
    lattice_basis_A: torch.Tensor | None = None  # [2, 2] rows = in-plane lattice vectors (A)

    def validate(self, grid: Grid) -> None:
        n = self.positions_A.shape[0]
        if self.positions_A.ndim != 2 or self.positions_A.shape[1] != 2:
            raise ValueError(f"positions_A must be [N, 2], got {tuple(self.positions_A.shape)}")
        if self.z.shape[0] != n or self.count.shape[0] != n:
            raise ValueError("z and count must each have length N matching positions_A")
        if n > 0 and float(self.count.min()) < 1.0:
            raise ValueError("every column count must be >= 1")
        if self.fov_A < grid.extent_A - 1e-6:
            raise ValueError(
                f"fov_A={self.fov_A} too small for render extent {grid.extent_A} A"
            )
        if self.lattice_basis_A is not None and tuple(self.lattice_basis_A.shape) != (2, 2):
            raise ValueError(
                f"lattice_basis_A must be [2, 2], got {tuple(self.lattice_basis_A.shape)}"
            )


@dataclass(frozen=True)
class ImagingCondition:
    """Physical imaging parameters (brief Section 1.6)."""

    energy_keV: float
    alpha_max_mrad: float
    source_size_A: float
    sigma_jitter_A: float = 0.2
    c3_A: float = 0.0
    c5_A: float = 0.0
    defocus_A: float = 0.0
    astig_a1_A: float = 0.0           # 2-fold astigmatism magnitude (angstroms)
    astig_a1_azimuth_rad: float = 0.0  # astigmatism orientation (radians)
    name: str = "custom"


IMAGING_CONDITIONS: dict[str, ImagingCondition] = {
    "cond1": ImagingCondition(200.0, 24.0, 0.9, name="cond1"),
    "cond2": ImagingCondition(100.0, 30.0, 0.8, name="cond2"),
    # cond3 and cond4 share identical listed parameters in the source paper's Table 1.
    "cond3": ImagingCondition(200.0, 10.5, 0.9, name="cond3"),
    "cond4": ImagingCondition(200.0, 10.5, 0.9, name="cond4"),
    "cond5": ImagingCondition(200.0, 10.0, 1.6, name="cond5"),
}


@dataclass(frozen=True)
class BackgroundSpec:
    """Selects a background family and its (family-specific) parameters."""

    kind: str = "constant"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NoiseSpec:
    """Dose and scan-noise parameters for one render."""

    n_peak: float = 100.0
    n_bg_frac: float = 0.02
    scan_freq_cyc_per_row: float = 0.1
    scan_beta: float = 0.4
    scan_phi0: float = 0.0


@dataclass(frozen=True)
class RenderParams:
    """Per-render draws supplied by the (external) augmentation sampler."""

    output_size: int = 256
    pixel_size_A: float = 0.1
    rotation_deg: float = 0.0
    position_offset_A: torch.Tensor = field(default_factory=lambda: torch.zeros(2))
    z_exponent: float = 1.7
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class RenderMeta:
    """Scalar metadata attached to one render output."""

    pixel_size_A: float
    output_size: int
    condition_name: str


@dataclass(frozen=True)
class RenderOutput:
    """All rasters and coordinate ground truth from one render."""

    image: torch.Tensor                   # [H, W] noisy, normalized
    gaussian_mask: torch.Tensor           # [H, W] equalized seg target
    circular_mask: torch.Tensor           # [H, W] binary disks
    no_noise: torch.Tensor                # [H, W] signal + background, no noise
    no_background_no_noise: torch.Tensor  # [H, W] clean sigma (x) PSF
    positions_A: torch.Tensor             # [M, 2] transformed/cropped centers (A)
    radii_A: torch.Tensor                 # [M] per-column radius (A)
    meta: RenderMeta

    def to_pixels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Column centers and radii converted to pixel units."""
        return self.positions_A / self.meta.pixel_size_A, self.radii_A / self.meta.pixel_size_A
