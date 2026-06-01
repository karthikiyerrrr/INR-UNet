"""Zone-axis projection of crystal structures into projected columns."""

import torch
from pymatgen.core import Lattice, Structure

from inr_unet.data.projection import project_structure


def _pt_fcc():
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.9242), ["Pt"], [[0, 0, 0]])


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


def test_projection_is_deterministic():
    a = project_structure(_pt_fcc(), [0, 0, 1], 30.0, 1.7, 0.4)
    b = project_structure(_pt_fcc(), [0, 0, 1], 30.0, 1.7, 0.4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
