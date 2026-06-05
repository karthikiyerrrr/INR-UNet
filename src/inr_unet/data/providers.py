"""Scene sources: the ColumnListProvider interface and a synthetic-lattice provider."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch
import yaml

from inr_unet.data.generation.structures import ColumnList

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from inr_unet.config import SyntheticDatasetConfig

# Decorrelated RNG stream (length-3 SeedSequence root; see plan determinism note).
_SCENE_SALT = 1009
# Decorrelated RNG stream for CIF scenes (distinct from the synthetic-lattice salt).
_CIF_SALT = 5011
# Lattice generation ranges (internal; scene FOV is large so the sampler's FOV/rotation grid fits).
_SCENE_FOV_A = (60.0, 80.0)
_SPACING_A = (2.0, 4.0)
_Z_A = (40, 90)
_Z_B = (6, 30)
_ANGLE_JITTER_DEG = 5.0


@runtime_checkable
class ColumnListProvider(Protocol):
    """Supplies a deterministic ColumnList per scene index."""

    def __len__(self) -> int: ...
    def get(self, scene_idx: int) -> ColumnList: ...


def _build_lattice(
    fov_A: float, spacing_A: float, z_a: float, z_b: float, angle_deg: float
) -> ColumnList:
    """Two-species checkerboard square lattice spanning the FOV, with small angle jitter."""
    coords = torch.arange(spacing_A / 2.0, fov_A, spacing_A)
    k = coords.numel()
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    n = pos.shape[0]
    ix = torch.arange(k)
    gi, gj = torch.meshgrid(ix, ix, indexing="ij")
    parity = ((gi + gj) % 2).reshape(-1).bool()
    z = torch.where(parity, torch.full((n,), float(z_a)), torch.full((n,), float(z_b)))
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]], dtype=pos.dtype)
    if n > 0 and angle_deg != 0.0:
        center = fov_A / 2.0
        pos = (pos - center) @ rot.T + center
    # in-plane lattice vectors (spacing_A along x and y) rotated by the same angle
    basis = (rot @ torch.tensor([[spacing_A, 0.0], [0.0, spacing_A]], dtype=pos.dtype).T).T
    return ColumnList(
        positions_A=pos, z=z, count=torch.ones(n), fov_A=float(fov_A), lattice_basis_A=basis
    )


class SyntheticLatticeProvider:
    """Index-keyed deterministic two-species crystal lattices."""

    def __init__(self, cfg: SyntheticDatasetConfig | DictConfig, master_seed: int) -> None:
        self.n_scenes = int(cfg.n_scenes)
        self.master_seed = int(master_seed)

    def __len__(self) -> int:
        return self.n_scenes

    def get(self, scene_idx: int) -> ColumnList:
        if not 0 <= scene_idx < self.n_scenes:
            raise IndexError(f"scene_idx {scene_idx} out of range [0, {self.n_scenes})")
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, int(scene_idx), _SCENE_SALT])
        )
        fov_A = float(rng.uniform(*_SCENE_FOV_A))
        spacing_A = float(rng.uniform(*_SPACING_A))
        z_a = float(rng.integers(_Z_A[0], _Z_A[1] + 1))
        z_b = float(rng.integers(_Z_B[0], _Z_B[1] + 1))
        angle_deg = float(rng.uniform(-_ANGLE_JITTER_DEG, _ANGLE_JITTER_DEG))
        return _build_lattice(fov_A, spacing_A, z_a, z_b, angle_deg)


class CIFProvider:
    """Index-keyed deterministic scenes from a bundled CIF set, projected per zone axis."""

    def __init__(
        self,
        manifest_path: str,
        n_scenes: int,
        master_seed: int,
        n_exponent: float,
        group_tol_A: float,
        scene_fov_A: tuple[float, float],
        rotation_jitter_deg: float,
        partial_fov_prob: float = 0.0,
        supercell_nx_range: tuple[int, int] = (1, 3),
        supercell_ny_range: tuple[int, int] = (1, 2),
        strip_prob: float = 0.0,
    ) -> None:
        self.n_scenes = int(n_scenes)
        self.master_seed = int(master_seed)
        self.n_exponent = float(n_exponent)
        self.group_tol_A = float(group_tol_A)
        self.scene_fov_A = (float(scene_fov_A[0]), float(scene_fov_A[1]))
        self.rotation_jitter_deg = float(rotation_jitter_deg)
        self.partial_fov_prob = float(partial_fov_prob)
        self.supercell_nx_range = (int(supercell_nx_range[0]), int(supercell_nx_range[1]))
        self.supercell_ny_range = (int(supercell_ny_range[0]), int(supercell_ny_range[1]))
        self.strip_prob = float(strip_prob)
        manifest = yaml.safe_load(Path(manifest_path).read_text())
        self._cif_dir = Path(manifest_path).parent
        self._entries = manifest["entries"]
        self._structures: dict[str, object] = {}

    def __len__(self) -> int:
        return self.n_scenes

    def _structure(self, cif_name: str):
        from pymatgen.core import Structure

        if cif_name not in self._structures:
            self._structures[cif_name] = Structure.from_file(self._cif_dir / cif_name)
        return self._structures[cif_name]

    def get(self, scene_idx: int) -> ColumnList:
        if not 0 <= scene_idx < self.n_scenes:
            raise IndexError(f"scene_idx {scene_idx} out of range [0, {self.n_scenes})")
        from inr_unet.data.projection import project_structure

        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, int(scene_idx), _CIF_SALT])
        )
        entry = self._entries[scene_idx % len(self._entries)]
        fov_A = float(rng.uniform(*self.scene_fov_A))
        angle_deg = float(rng.uniform(-self.rotation_jitter_deg, self.rotation_jitter_deg))
        use_partial = bool(rng.random() < self.partial_fov_prob)
        supercell = None
        if use_partial:
            if rng.random() < self.strip_prob:
                # elongated n x 1 row: read in-plane spacing from a (1,1) projection, size n to span
                # ~0.4-0.65 x FOV so the row stays framed; sampler rotation then tilts it.
                long_axis = int(rng.random() < 0.5)
                _, _, _, basis0 = project_structure(
                    self._structure(entry["cif"]),
                    entry["zone_axis"],
                    fov_A=fov_A,
                    n_exponent=self.n_exponent,
                    group_tol_A=self.group_tol_A,
                    supercell=(1, 1),
                )
                spacing = float(torch.linalg.norm(basis0[long_axis]))
                ncell = max(3, round(float(rng.uniform(0.4, 0.65)) * fov_A / max(spacing, 1e-6)))
                supercell = (ncell, 1) if long_axis == 0 else (1, ncell)
            else:
                nx = int(rng.integers(self.supercell_nx_range[0], self.supercell_nx_range[1] + 1))
                ny = int(rng.integers(self.supercell_ny_range[0], self.supercell_ny_range[1] + 1))
                supercell = (nx, ny)
        pos, z, count, basis = project_structure(
            self._structure(entry["cif"]),
            entry["zone_axis"],
            fov_A=fov_A,
            n_exponent=self.n_exponent,
            group_tol_A=self.group_tol_A,
            supercell=supercell,
        )
        if use_partial and pos.shape[0] > 0:
            # center the patch centroid in the FOV so the (center-cropped) render frames it.
            # both rotations below pivot about the FOV center, leaving the centroid fixed.
            pos = pos - pos.mean(dim=0) + fov_A / 2.0
        if pos.shape[0] > 0 and angle_deg != 0.0:
            theta = math.radians(angle_deg)
            c, s = math.cos(theta), math.sin(theta)
            rot = torch.tensor([[c, -s], [s, c]], dtype=pos.dtype)
            center = fov_A / 2.0
            pos = (pos - center) @ rot.T + center
            basis = (rot @ basis.T).T
        return ColumnList(
            positions_A=pos, z=z, count=count, fov_A=fov_A, lattice_basis_A=basis,
            is_partial=use_partial,
        )
