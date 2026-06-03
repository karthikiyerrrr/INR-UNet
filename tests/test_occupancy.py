"""Support-region geometry for finite-particle occupancy."""

import numpy as np
import torch

from inr_unet.data.occupancy import support_mask


class _Occ:
    """Minimal stand-in for OccupancyConfig fields used by support_mask."""

    def __init__(self, mode):
        self.mode = mode
        self.mix_weights = {"facet_polygon": 1.0, "blob": 1.0, "full": 1.0}
        self.support_frac_min = 0.5
        self.support_frac_max = 0.5
        self.n_facets_min = 4
        self.n_facets_max = 4
        self.facet_offset_jitter_frac = 0.0
        self.half_plane_prob = 0.0
        self.edge_clip_prob = 0.0
        self.blob_roughness = 0.0


def _grid_positions(fov=60.0, step=2.0):
    xs = np.arange(step / 2, fov, step)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    return torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)


def test_full_mode_keeps_everything():
    pos = _grid_positions()
    mask = support_mask(pos, 60.0, torch.eye(2) * 3.0, _Occ("full"), np.random.default_rng(0))
    assert mask.all()


def test_facet_polygon_keeps_convex_subset_inside_fov():
    pos = _grid_positions()
    basis = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    mask = support_mask(pos, 60.0, basis, _Occ("facet_polygon"), np.random.default_rng(0))
    assert mask.any() and not mask.all()
    # kept points must be spatially clustered (a convex region), not the whole frame
    kept = pos[mask]
    extent = (kept.max(dim=0).values - kept.min(dim=0).values)
    assert (extent <= 60.0 * 0.5 + 6.0).all()  # ~support fraction + lattice slack


def test_blob_mode_works_without_lattice():
    pos = _grid_positions()
    mask = support_mask(pos, 60.0, None, _Occ("blob"), np.random.default_rng(0))
    assert mask.any() and not mask.all()


def test_facet_falls_back_to_blob_without_basis():
    pos = _grid_positions()
    mask = support_mask(pos, 60.0, None, _Occ("facet_polygon"), np.random.default_rng(1))
    assert mask.any() and not mask.all()


def test_deterministic_for_same_seed():
    pos = _grid_positions()
    basis = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    m1 = support_mask(pos, 60.0, basis, _Occ("facet_polygon"), np.random.default_rng(7))
    m2 = support_mask(pos, 60.0, basis, _Occ("facet_polygon"), np.random.default_rng(7))
    assert torch.equal(m1, m2)


def test_finite_support_provider_clips_columns():
    from inr_unet.config import OccupancyConfig, SyntheticDatasetConfig
    from inr_unet.data.occupancy import FiniteSupportProvider
    from inr_unet.data.providers import SyntheticLatticeProvider

    inner = SyntheticLatticeProvider(SyntheticDatasetConfig(n_scenes=4), master_seed=0)
    occ = OccupancyConfig(mode="facet_polygon", support_frac_min=0.4, support_frac_max=0.4)
    wrapped = FiniteSupportProvider(inner, occ, master_seed=0)

    full = inner.get(0)
    clipped = wrapped.get(0)
    assert clipped.positions_A.shape[0] < full.positions_A.shape[0]
    assert clipped.positions_A.shape[0] > 0
    assert clipped.fov_A == full.fov_A
    assert clipped.lattice_basis_A is not None
    assert clipped.z.shape[0] == clipped.positions_A.shape[0]


def test_finite_support_provider_is_deterministic():
    from inr_unet.config import OccupancyConfig, SyntheticDatasetConfig
    from inr_unet.data.occupancy import FiniteSupportProvider
    from inr_unet.data.providers import SyntheticLatticeProvider

    inner = SyntheticLatticeProvider(SyntheticDatasetConfig(n_scenes=4), master_seed=0)
    occ = OccupancyConfig(mode="blob")
    a = FiniteSupportProvider(inner, occ, 0).get(1)
    b = FiniteSupportProvider(inner, occ, 0).get(1)
    assert torch.equal(a.positions_A, b.positions_A)


def test_finite_support_provider_full_is_identity():
    from inr_unet.config import OccupancyConfig, SyntheticDatasetConfig
    from inr_unet.data.occupancy import FiniteSupportProvider
    from inr_unet.data.providers import ColumnListProvider, SyntheticLatticeProvider

    inner = SyntheticLatticeProvider(SyntheticDatasetConfig(n_scenes=4), master_seed=0)
    wrapped = FiniteSupportProvider(inner, OccupancyConfig(mode="full"), 0)
    assert isinstance(wrapped, ColumnListProvider)
    assert torch.equal(wrapped.get(0).positions_A, inner.get(0).positions_A)
    assert len(wrapped) == len(inner)


def test_default_mix_weights_favor_full():
    from inr_unet.config import OccupancyConfig

    w = OccupancyConfig().mix_weights
    total = sum(w.values())
    assert w["full"] / total > 0.8                       # ~87% full
    assert w["full"] > w["facet_polygon"] > w["blob"]    # ordered by dataset frequency


def test_mix_mode_samples_at_configured_proportions():
    pos = _grid_positions()
    basis = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    occ = _Occ("mix")
    occ.mix_weights = {"full": 0.87, "facet_polygon": 0.10, "blob": 0.03}
    rng = np.random.default_rng(0)
    full_count = 0
    trials = 400
    for _ in range(trials):
        mask = support_mask(pos, 60.0, basis, occ, rng)
        if bool(mask.all()):
            full_count += 1
    assert 0.78 < full_count / trials < 0.95             # ~0.87 full draws
