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


def _tile_ranges(
    lat: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    beam_hat: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> list[range]:
    """Integer (i, j, k) lattice-translation ranges whose cells cover an oriented box.

    The box is given by its lower/upper corners ``lo``/``hi`` in (e1, e2, beam_hat)
    coordinates. Mapping the 8 box corners to fractional lattice coordinates gives a tight
    per-index integer range, so a beam-parallel lattice vector needs few reps while in-plane
    vectors get many -- no blow-up, no undersampling.
    """
    basis = np.stack([e1, e2, beam_hat])  # rows; orthonormal
    corners_ortho = np.array(
        [
            [hi[0] if (m & 1) else lo[0],
             hi[1] if (m & 2) else lo[1],
             hi[2] if (m & 4) else lo[2]]
            for m in range(8)
        ]
    )
    corners_cart = corners_ortho @ basis  # (e1, e2, beam) coords -> Cartesian
    frac = corners_cart @ np.linalg.inv(lat)  # cart = frac @ lat
    fmin = np.floor(frac.min(axis=0)).astype(int) - 1
    fmax = np.ceil(frac.max(axis=0)).astype(int) + 1
    return [range(int(fmin[d]), int(fmax[d]) + 1) for d in range(3)]


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
    """Order-independent clustering of coincident projected atoms into columns.

    Union-find over the within-tolerance coincidence graph; column centers are member
    means and columns are returned in lexicographic (x, then y) order, so the result is
    invariant to input atom order.
    """
    npts = pts.shape[0]
    if npts == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    parent = list(range(npts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    ii, jj = np.where((d2 <= tol_A * tol_A) & (np.arange(npts)[:, None] < np.arange(npts)[None, :]))
    for a, b in zip(ii, jj, strict=True):
        union(int(a), int(b))

    groups: dict[int, list[int]] = {}
    for i in range(npts):
        groups.setdefault(find(i), []).append(i)

    xy, z_eff, count = [], [], []
    for members in groups.values():
        m = np.asarray(members)
        arr = zs[m]
        nat = arr.shape[0]
        xy.append(pts[m].mean(axis=0))
        z_eff.append(float((np.sum(arr**n) / nat) ** (1.0 / n)))
        count.append(float(nat))

    xy = np.asarray(xy)
    z_eff = np.asarray(z_eff)
    count = np.asarray(count)
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    return xy[order], z_eff[order], count[order]


def project_structure(
    structure: Structure,
    zone_axis: list[int] | tuple[int, int, int],
    fov_A: float,
    n_exponent: float,
    group_tol_A: float,
    margin_A: float = 4.0,
    supercell: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a structure along a zone axis into (positions_A, z_eff, count, lattice_basis_A).

    Tiles the cell over an oriented box covering the FOV in-plane and exactly one beam-period
    deep, projects atoms onto the plane normal to the zone axis, groups coincident atoms into
    columns, and reduces each mixed column to an effective z with count * z**n == sum(z_i**n).

    If ``supercell=(nx, ny)`` is given, tiles exactly nx x ny in-plane cells (one cell deep)
    and returns positions in the supercell's own extent rather than clipping to the FOV.
    """
    lat = np.asarray(structure.lattice.matrix, dtype=float)  # rows = a, b, c (Cartesian)
    axis = np.asarray(zone_axis, dtype=float)
    beam_hat, e1, e2, align = _in_plane_frame(lat, axis)
    beam_axis = int(np.argmax(align))  # lattice vector most parallel to the beam
    in_plane_axes = [i for i in range(3) if i != beam_axis]
    a_2d = _vec2(lat[in_plane_axes[0]], e1, e2)
    b_2d = _vec2(lat[in_plane_axes[1]], e1, e2)

    cart = np.asarray([site.coords for site in structure.sites], dtype=float)  # [A, 3]
    zs_cell = np.asarray([site.specie.Z for site in structure.sites], dtype=float)  # [A]

    area = abs(a_2d[0] * b_2d[1] - a_2d[1] * b_2d[0])
    beam_period = abs(np.linalg.det(lat)) / max(area, 1e-9)  # one cell deep along the beam

    if supercell is None:
        lo = np.array([-margin_A, -margin_A, 0.0])
        hi = np.array([fov_A + margin_A, fov_A + margin_A, beam_period])
        ranges = _tile_ranges(lat, e1, e2, beam_hat, lo, hi)
    else:
        nx, ny = int(supercell[0]), int(supercell[1])
        ranges: list[range] = [range(0)] * 3  # each entry reassigned below
        ranges[in_plane_axes[0]] = range(0, nx)
        ranges[in_plane_axes[1]] = range(0, ny)
        ranges[beam_axis] = range(0, 1)  # exactly one cell deep along the beam

    pts_list, t_list, z_list = [], [], []
    for i, j, k in product(*ranges):
        shift = i * lat[0] + j * lat[1] + k * lat[2]
        for a in range(cart.shape[0]):
            c = cart[a] + shift
            pts_list.append(_vec2(c, e1, e2))
            t_list.append(float(c @ beam_hat))
            z_list.append(zs_cell[a])

    basis = np.stack([a_2d, b_2d])
    if not pts_list:
        return (
            torch.zeros((0, 2), dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
            torch.tensor(basis, dtype=torch.float32),
        )

    pts = np.asarray(pts_list)
    ts = np.asarray(t_list)
    zs = np.asarray(z_list)

    if supercell is None:
        slab = (ts >= 0.0) & (ts < beam_period)  # keep exactly one period deep
        pts, zs = pts[slab], zs[slab]
        pts = pts - pts.min(axis=0)
        inside = (
            (pts[:, 0] >= 0.0)
            & (pts[:, 0] <= fov_A)
            & (pts[:, 1] >= 0.0)
            & (pts[:, 1] <= fov_A)
        )
        pts, zs = pts[inside], zs[inside]
    else:
        pts = pts - pts.min(axis=0)  # small extent; provider places it inside a larger FOV

    xy, z_eff, count = _group_columns(pts, zs, group_tol_A, n_exponent)
    return (
        torch.tensor(xy, dtype=torch.float32),
        torch.tensor(z_eff, dtype=torch.float32),
        torch.tensor(count, dtype=torch.float32),
        torch.tensor(basis, dtype=torch.float32),
    )
