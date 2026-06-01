"""CIFProvider: index-keyed deterministic scenes from the bundled CIF set."""

import pytest
import torch

from inr_unet.data.generation.structures import ColumnList, Grid
from inr_unet.data.providers import CIFProvider, ColumnListProvider


def _provider(n_scenes=6, seed=0):
    return CIFProvider(
        manifest_path="src/inr_unet/data/cif/manifest.yaml",
        n_scenes=n_scenes,
        master_seed=seed,
        n_exponent=1.7,
        group_tol_A=0.4,
        scene_fov_A=(60.0, 80.0),
        rotation_jitter_deg=5.0,
    )


def test_len_matches_n_scenes():
    assert len(_provider(n_scenes=9)) == 9


def test_get_returns_valid_columnlist():
    scene = _provider().get(0)
    assert isinstance(scene, ColumnList)
    assert scene.lattice_basis_A is not None
    assert 60.0 <= scene.fov_A <= 80.0
    scene.validate(Grid(output_size=48, pixel_size_A=0.3))  # 14.4 A extent << fov


def test_get_is_deterministic_across_instances():
    a, b = _provider().get(2), _provider().get(2)
    assert torch.equal(a.positions_A, b.positions_A)
    assert a.fov_A == b.fov_A


def test_distinct_indices_cover_distinct_entries():
    p = _provider(n_scenes=6)
    a, b = p.get(0), p.get(1)
    assert a.positions_A.shape != b.positions_A.shape or not torch.equal(
        a.positions_A, b.positions_A
    )


def test_out_of_range_raises():
    p = _provider(n_scenes=3)
    with pytest.raises(IndexError):
        p.get(3)
    with pytest.raises(IndexError):
        p.get(-1)


def test_satisfies_protocol():
    assert isinstance(_provider(), ColumnListProvider)
