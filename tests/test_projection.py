"""Zone-axis projection of crystal structures into projected columns."""

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from inr_unet.data.projection import _group_columns, project_structure


def _pt_fcc():
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.9242), ["Pt"], [[0, 0, 0]])


def _graphene():
    # Hexagonal in-plane cell (gamma = 120 deg), two carbons in the basal plane.
    return Structure(
        Lattice.hexagonal(2.461, 10.0),
        ["C", "C"],
        [[0.0, 0.0, 0.0], [1 / 3, 2 / 3, 0.0]],
    )


def _empty_occupancy_cells(pos: torch.Tensor, fov_A: float, n: int = 8) -> int:
    """Number of empty cells when [0, fov]^2 is binned into an n x n occupancy grid."""
    p = pos.numpy()
    gx = np.clip((p[:, 0] / fov_A * n).astype(int), 0, n - 1)
    gy = np.clip((p[:, 1] / fov_A * n).astype(int), 0, n - 1)
    grid = np.zeros((n, n), dtype=bool)
    grid[gx, gy] = True
    return n * n - int(grid.sum())


def _srtio3():
    return Structure.from_spacegroup(
        "Pm-3m",
        Lattice.cubic(3.905),
        ["Sr", "Ti", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]],
    )


def test_pt001_is_single_species_square_net():
    pos, z, count, basis = project_structure(
        _pt_fcc(), zone_axis=[0, 0, 1], fov_A=40.0, n_exponent=1.7, group_tol_A=0.4
    )
    assert pos.shape[0] > 0
    assert pos.shape[1] == 2
    # only platinum -> all effective z equal to Z(Pt)=78
    assert torch.allclose(z, torch.full_like(z, 78.0), atol=1e-3)
    # positions inside the FOV window
    assert float(pos.min()) >= -1e-3 and float(pos.max()) <= 40.0 + 1e-3
    assert tuple(basis.shape) == (2, 2)
    # in-plane basis vectors of fcc[001] have length a = 3.9242
    lengths = torch.linalg.norm(basis, dim=1)
    assert torch.allclose(lengths, torch.full_like(lengths, 3.9242), atol=0.05)


def test_counts_are_positive_integers():
    _, _, count, _ = project_structure(
        _pt_fcc(), zone_axis=[0, 0, 1], fov_A=30.0, n_exponent=1.7, group_tol_A=0.4
    )
    assert (count >= 1.0).all()
    assert torch.allclose(count, count.round())


def test_mixed_column_effective_z_matches_power_sum():
    # SrTiO3[001]: the (0,0) column stacks Sr and O along the beam -> mixed.
    pos, z, count, _ = project_structure(
        _srtio3(), zone_axis=[0, 0, 1], fov_A=20.0, n_exponent=1.7, group_tol_A=0.4
    )
    # at least one column must mix species (effective z not equal to any single element Z)
    elem_z = {38.0, 22.0, 8.0}  # Sr, Ti, O
    mixed = [
        (float(zi), float(ci))
        for zi, ci in zip(z.tolist(), count.tolist(), strict=True)
        if all(abs(zi - e) > 1e-3 for e in elem_z)
    ]
    assert mixed, "expected at least one mixed-species column"
    # for a mixed column, count * z_eff**n must be a clean power-sum (>= the heaviest contribution)
    for z_eff, c in mixed:
        assert c * z_eff**1.7 > 8.0**1.7


def test_pt111_is_densely_sampled():
    # Regression for the argmax tiling bug: [111] used to yield only ~9 columns.
    pos, z, count, basis = project_structure(
        _pt_fcc(), zone_axis=[1, 1, 1], fov_A=30.0, n_exponent=1.7, group_tol_A=0.4
    )
    assert pos.shape[0] > 80  # dense hex net, not the old 9
    lengths = torch.linalg.norm(basis, dim=1)
    cosang = torch.dot(basis[0], basis[1]) / (lengths[0] * lengths[1])
    angle = torch.rad2deg(torch.arccos(cosang)).item()
    assert abs(angle - 120.0) < 1.0  # hexagonal frame preserved


def test_hexagonal_lattice_fills_the_square_fov():
    # Regression: graphene/MoS2 ([001], gamma=120 deg) left a triangular vacuum gap in one
    # corner of the square FOV. The full lattice must cover the whole window with no holes.
    for fov in (40.0, 56.0, 80.0):
        pos, *_ = project_structure(_graphene(), [0, 0, 1], fov, 1.7, 0.4)
        assert _empty_occupancy_cells(pos, fov) == 0, f"empty cells at fov={fov}"
        assert float(pos.min()) >= -1e-3 and float(pos.max()) <= fov + 1e-3


def test_low_index_nets_preserved():
    # [001] and [110] keep their dense nets and bases after the tiling rewrite.
    pos1, z1, _, basis1 = project_structure(_pt_fcc(), [0, 0, 1], 30.0, 1.7, 0.4)
    assert pos1.shape[0] > 200
    assert torch.allclose(torch.linalg.norm(basis1, dim=1),
                          torch.tensor([3.924, 3.924]), atol=0.05)
    assert torch.allclose(z1, torch.full_like(z1, 78.0), atol=1e-3)
    assert float(pos1.min()) >= -1e-3 and float(pos1.max()) <= 30.0 + 1e-3

    pos2, _, _, basis2 = project_structure(_pt_fcc(), [1, 1, 0], 30.0, 1.7, 0.4)
    assert pos2.shape[0] > 150
    assert torch.allclose(torch.linalg.norm(basis2, dim=1),
                          torch.tensor([2.775, 3.924]), atol=0.05)


def test_projection_is_deterministic():
    a = project_structure(_pt_fcc(), [0, 0, 1], 30.0, 1.7, 0.4)
    b = project_structure(_pt_fcc(), [0, 0, 1], 30.0, 1.7, 0.4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_supercell_tiles_a_small_patch():
    # NiO-style small supercell: a fixed 2x1 tile, not a FOV-filling net.
    nio = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.18), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    pos_full, *_ = project_structure(nio, [1, 1, 0], 40.0, 1.7, 0.4)
    pos_sc, _, _, _ = project_structure(
        nio, [1, 1, 0], 40.0, 1.7, 0.4, supercell=(2, 1)
    )
    # the supercell patch is far smaller and occupies a sub-region, not the whole FOV
    assert pos_sc.shape[0] < pos_full.shape[0]
    extent = pos_sc.max(dim=0).values - pos_sc.min(dim=0).values
    assert float(extent.max()) < 40.0  # fits in a band, leaving room for margins
    assert pos_sc.shape[0] > 0


def test_group_columns_is_order_independent():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 20.0, size=(200, 2))
    zs = rng.integers(6, 80, size=200).astype(float)
    perm = rng.permutation(200)

    xy_a, z_a, c_a = _group_columns(pts, zs, tol_A=0.4, n=1.7)
    xy_b, z_b, c_b = _group_columns(pts[perm], zs[perm], tol_A=0.4, n=1.7)

    # columns are returned in a deterministic (lexicographic) order, so compare directly
    assert xy_a.shape == xy_b.shape
    assert np.allclose(xy_a, xy_b, atol=1e-6)
    assert np.allclose(z_a, z_b, atol=1e-6)
    assert np.allclose(c_a, c_b, atol=1e-6)
    assert c_a.max() > 1  # grouping actually merged some atoms (not all singletons)
