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
