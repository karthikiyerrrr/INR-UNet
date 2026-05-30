"""SyntheticLatticeProvider: index-keyed deterministic two-species lattices."""

import pytest
import torch

from inr_unet.config import SyntheticDatasetConfig
from inr_unet.data.generation.structures import ColumnList
from inr_unet.data.providers import ColumnListProvider, SyntheticLatticeProvider


def _provider(n_scenes=4, seed=0):
    cfg = SyntheticDatasetConfig(n_scenes=n_scenes)
    return SyntheticLatticeProvider(cfg, master_seed=seed)


def test_len_matches_n_scenes():
    assert len(_provider(n_scenes=7)) == 7


def test_get_is_deterministic():
    p = _provider()
    a, b = p.get(2), p.get(2)
    assert torch.equal(a.positions_A, b.positions_A)
    assert torch.equal(a.z, b.z)
    assert a.fov_A == b.fov_A


def test_distinct_scenes_differ():
    p = _provider()
    a, b = p.get(0), p.get(1)
    assert not (a.positions_A.shape == b.positions_A.shape
                and torch.equal(a.positions_A, b.positions_A))


def test_two_species_present():
    p = _provider()
    scene = p.get(0)
    assert torch.unique(scene.z).numel() == 2


def test_fov_in_expected_band():
    p = _provider()
    scene = p.get(3)
    assert 60.0 <= scene.fov_A <= 80.0


def test_returns_columnlist_with_valid_counts():
    p = _provider()
    scene = p.get(0)
    assert isinstance(scene, ColumnList)
    assert scene.count.shape[0] == scene.positions_A.shape[0]
    assert float(scene.count.min()) >= 1.0


def test_out_of_range_raises():
    p = _provider(n_scenes=3)
    with pytest.raises(IndexError):
        p.get(3)
    with pytest.raises(IndexError):
        p.get(-1)


def test_satisfies_protocol():
    assert isinstance(_provider(), ColumnListProvider)
