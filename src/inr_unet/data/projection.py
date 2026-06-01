"""Pure zone-axis projection of a crystal Structure into projected columns.

No RNG, no config: given a pymatgen Structure, a zone axis, and a FOV, return the
projected column positions, an effective per-column z (so count * z**n reproduces the
true ADF power-sum), per-column atom counts, and the in-plane lattice basis.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pymatgen.core import Structure


def _vec2(v: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Components of a 3D Cartesian vector in the in-plane (e1, e2) frame."""
    return np.array([float(v @ e1), float(v @ e2)])


def _in_plane_frame(
    lattice_matrix: np.ndarray, zone_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (beam_hat, e1, e2) with beam_hat along the zone axis; align = |axis . beam|."""
    beam = zone_axis @ lattice_matrix  # Cartesian beam direction (u*a + v*b + w*c)
    beam_hat = beam / np.linalg.norm(beam)
    norms = np.linalg.norm(lattice_matrix, axis=1)
    align = np.abs(lattice_matrix @ beam_hat) / norms  # how parallel each cell vector is to beam
    seed = lattice_matrix[int(np.argmin(align))]
    e1 = seed - (seed @ beam_hat) * beam_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(beam_hat, e1)
    e2 = e2 / np.linalg.norm(e2)
    return beam_hat, e1, e2, align


def _group_columns(
    pts: np.ndarray, zs: np.ndarray, tol_A: float, n: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy clustering of coincident projected atoms into columns.

    Deterministic in input order.
    """
    centers: list[np.ndarray] = []
    members: list[list[float]] = []
    tol2 = tol_A * tol_A
    for p, z in zip(pts, zs, strict=True):
        placed = False
        for ci, c in enumerate(centers):
            dx, dy = p[0] - c[0], p[1] - c[1]
            if dx * dx + dy * dy <= tol2:
                members[ci].append(float(z))
                placed = True
                break
        if not placed:
            centers.append(p.copy())
            members.append([float(z)])
    xy, z_eff, count = [], [], []
    for c, ms in zip(centers, members, strict=True):
        arr = np.asarray(ms)
        nat = arr.shape[0]
        xy.append(c)
        z_eff.append(float((np.sum(arr**n) / nat) ** (1.0 / n)))
        count.append(float(nat))
    if not xy:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return np.asarray(xy), np.asarray(z_eff), np.asarray(count)


def project_structure(
    structure: Structure,
    zone_axis: list[int] | tuple[int, int, int],
    fov_A: float,
    n_exponent: float,
    group_tol_A: float,
    margin_A: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a structure along a zone axis into (positions_A, z_eff, count, lattice_basis_A).

    Tiles the cell to cover fov_A in-plane and exactly one cell deep along the beam,
    projects atoms onto the plane normal to the zone axis, groups coincident atoms into
    columns, and reduces each mixed column to an effective z with count * z**n == sum(z_i**n).
    """
    lat = np.asarray(structure.lattice.matrix, dtype=float)  # rows = a, b, c (Cartesian)
    axis = np.asarray(zone_axis, dtype=float)
    beam_hat, e1, e2, align = _in_plane_frame(lat, axis)
    beam_axis = int(np.argmax(align))  # lattice vector most parallel to the beam
    in_plane_axes = [i for i in range(3) if i != beam_axis]

    span = fov_A + 2.0 * margin_A
    ranges: list[range] = []
    for i in range(3):
        if i == beam_axis:
            ranges.append(range(0, 1))  # single slab along the beam
        else:
            length = np.linalg.norm(_vec2(lat[i], e1, e2))
            reps = int(np.ceil(span / max(length, 1e-6))) + 1
            ranges.append(range(-reps, reps + 1))

    cart = np.asarray([site.coords for site in structure.sites], dtype=float)  # [A, 3]
    zs_cell = np.asarray([site.specie.Z for site in structure.sites], dtype=float)  # [A]

    pts_list, z_list = [], []
    for i, j, k in product(*ranges):
        shift = i * lat[0] + j * lat[1] + k * lat[2]
        for a in range(cart.shape[0]):
            pts_list.append(_vec2(cart[a] + shift, e1, e2))
            z_list.append(zs_cell[a])
    pts = np.asarray(pts_list)
    zs = np.asarray(z_list)

    pts = pts - pts.min(axis=0)
    inside = (
        (pts[:, 0] >= 0.0)
        & (pts[:, 0] <= fov_A)
        & (pts[:, 1] >= 0.0)
        & (pts[:, 1] <= fov_A)
    )
    pts, zs = pts[inside], zs[inside]

    xy, z_eff, count = _group_columns(pts, zs, group_tol_A, n_exponent)
    basis = np.stack(
        [_vec2(lat[in_plane_axes[0]], e1, e2), _vec2(lat[in_plane_axes[1]], e1, e2)]
    )
    return (
        torch.tensor(xy, dtype=torch.float32),
        torch.tensor(z_eff, dtype=torch.float32),
        torch.tensor(count, dtype=torch.float32),
        torch.tensor(basis, dtype=torch.float32),
    )
