"""Finite-particle occupancy: support-region geometry and the FiniteSupportProvider wrapper.

A support region answers "is this column inside the particle?". Shapes:
- full: keep everything.
- facet_polygon: convex intersection of half-planes whose normals are low-index in-plane
  lattice directions (facet-plausible, NOT an energy-minimized Wulff shape).
- blob: isotropic disk with optional smooth radial roughness (for irregular/amorphous).
The FiniteSupportProvider (below) applies one region per scene at a seeded placement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from inr_unet.data.generation.structures import ColumnList

if TYPE_CHECKING:
    from inr_unet.data.providers import ColumnListProvider

# Decorrelated RNG stream for occupancy (distinct from scene/crop/query/cif salts).
_SUPPORT_SALT = 4099


def _choose_mode(weights: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(weights.keys())
    probs = np.asarray([weights[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def _center(fov_A: float, edge_clip_prob: float, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < edge_clip_prob:
        return rng.uniform(0.0, fov_A, size=2)  # may push the particle partly off-frame
    jitter = rng.uniform(-0.15, 0.15, size=2) * fov_A
    return np.array([fov_A / 2.0, fov_A / 2.0]) + jitter


def _facet_normals(basis: np.ndarray, n_facets: int, rng: np.random.Generator) -> np.ndarray:
    """Unit normals along low-index in-plane lattice directions; ±a, ±b guarantee bounded."""
    a, b = basis[0], basis[1]
    base = [a, -a, b, -b]  # 4 facets -> always bounded
    extra = [a + b, -(a + b), a - b, -(a - b)]
    rng.shuffle(extra)
    dirs = base + extra[: max(0, n_facets - 4)]
    normals = []
    for d in dirs:
        norm = np.linalg.norm(d)
        if norm > 1e-6:
            normals.append(d / norm)
    return np.stack(normals)


def _facet_mask(
    xy: np.ndarray, center: np.ndarray, radius: float, basis: np.ndarray, occ, rng
) -> np.ndarray:
    n_facets = int(rng.integers(occ.n_facets_min, occ.n_facets_max + 1))
    normals = _facet_normals(basis, n_facets, rng)
    if rng.random() < occ.half_plane_prob:
        normals = normals[:1]  # degenerate single crystal edge crossing the FOV
    jitter = occ.facet_offset_jitter_frac
    offsets = radius * (1.0 + rng.uniform(-jitter, jitter, size=normals.shape[0]))
    rel = xy - center
    proj = rel @ normals.T  # [N, F]
    return (proj <= offsets).all(axis=1)


def _blob_mask(
    xy: np.ndarray, center: np.ndarray, radius: float, roughness: float, rng: np.random.Generator
) -> np.ndarray:
    rel = xy - center
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    # smooth low-frequency radial perturbation
    phase = rng.uniform(0, 2 * np.pi, size=3)
    amp = roughness * radius
    boundary = radius + amp * (
        np.sin(2 * theta + phase[0]) + np.sin(3 * theta + phase[1]) + np.sin(5 * theta + phase[2])
    ) / 3.0
    return r <= boundary


def support_mask(
    positions_A: torch.Tensor,
    fov_A: float,
    lattice_basis_A: torch.Tensor | None,
    occ,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Boolean keep-mask over columns for one seeded support region."""
    n = positions_A.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    mode = occ.mode
    if mode == "mix":
        mode = _choose_mode(occ.mix_weights, rng)
    if mode == "full":
        return torch.ones(n, dtype=torch.bool)

    xy = positions_A.numpy()
    frac = float(rng.uniform(occ.support_frac_min, occ.support_frac_max))
    radius = 0.5 * frac * fov_A
    center = _center(fov_A, occ.edge_clip_prob, rng)

    if mode == "facet_polygon" and lattice_basis_A is not None:
        mask = _facet_mask(xy, center, radius, lattice_basis_A.numpy(), occ, rng)
    else:
        mask = _blob_mask(xy, center, radius, occ.blob_roughness, rng)
    return torch.as_tensor(mask, dtype=torch.bool)


class FiniteSupportProvider:
    """Wraps any ColumnListProvider and clips each scene's columns to a support region."""

    def __init__(self, inner: ColumnListProvider, occ, master_seed: int) -> None:
        self.inner = inner
        self.occ = occ
        self.master_seed = int(master_seed)

    def __len__(self) -> int:
        return len(self.inner)

    def get(self, scene_idx: int) -> ColumnList:
        cols = self.inner.get(scene_idx)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, int(scene_idx), _SUPPORT_SALT])
        )
        mask = support_mask(cols.positions_A, cols.fov_A, cols.lattice_basis_A, self.occ, rng)
        return ColumnList(
            positions_A=cols.positions_A[mask],
            z=cols.z[mask],
            count=cols.count[mask],
            fov_A=cols.fov_A,
            lattice_basis_A=cols.lattice_basis_A,
            is_partial=cols.is_partial,
        )
