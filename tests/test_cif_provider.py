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


def _provider_partial(n_scenes=6, seed=0):
    return CIFProvider(
        manifest_path="src/inr_unet/data/cif/manifest.yaml",
        n_scenes=n_scenes,
        master_seed=seed,
        n_exponent=1.7,
        group_tol_A=0.4,
        scene_fov_A=(60.0, 80.0),
        rotation_jitter_deg=5.0,
        partial_fov_prob=1.0,
        supercell_nx_range=(2, 2),
        supercell_ny_range=(1, 1),
    )


def test_partial_fov_leaves_margins():
    from inr_unet.data.generation.structures import Grid

    scene = _provider_partial().get(2)
    assert scene.positions_A.shape[0] > 0
    span = scene.positions_A.max(dim=0).values - scene.positions_A.min(dim=0).values
    # a 2x1 supercell occupies only a band inside the (>=60 A) FOV
    assert float(span.max()) < scene.fov_A
    scene.validate(Grid(output_size=48, pixel_size_A=0.3))


def test_partial_fov_is_deterministic():
    a, b = _provider_partial().get(3), _provider_partial().get(3)
    assert torch.equal(a.positions_A, b.positions_A)


def _ratio(scene):
    span = scene.positions_A.max(dim=0).values - scene.positions_A.min(dim=0).values
    return float(span.max() / span.min().clamp(min=1e-6))


def _provider_strip(seed=0):
    return CIFProvider(
        manifest_path="src/inr_unet/data/cif/manifest.yaml",
        n_scenes=6, master_seed=seed, n_exponent=1.7, group_tol_A=0.4,
        scene_fov_A=(60.0, 80.0), rotation_jitter_deg=0.0,
        partial_fov_prob=1.0, strip_prob=1.0,
    )


def _provider_blob(seed=0):
    return CIFProvider(
        manifest_path="src/inr_unet/data/cif/manifest.yaml",
        n_scenes=6, master_seed=seed, n_exponent=1.7, group_tol_A=0.4,
        scene_fov_A=(60.0, 80.0), rotation_jitter_deg=0.0,
        partial_fov_prob=1.0, strip_prob=0.0,
        supercell_nx_range=(2, 2), supercell_ny_range=(2, 2),
    )


def test_strip_is_more_elongated_than_blob():
    strip = _provider_strip().get(2)
    blob = _provider_blob().get(2)
    assert strip.positions_A.shape[0] > 0 and blob.positions_A.shape[0] > 0
    assert _ratio(strip) >= 3.0          # an n x 1 row is strongly elongated
    assert _ratio(strip) > _ratio(blob)  # ... and more so than a compact 2x2 blob


def test_strip_is_deterministic():
    a, b = _provider_strip().get(3), _provider_strip().get(3)
    assert torch.equal(a.positions_A, b.positions_A)
