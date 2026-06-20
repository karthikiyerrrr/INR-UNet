"""Contract tests for make_coord and LIIFDecoder: order, shape, continuity, wiring."""

import torch
from omegaconf import OmegaConf

from inr_unet.models.components.liif import LIIFDecoder, make_coord


def _cfg(**over):
    base = {
        "name": "liif",
        "hidden_dim": 64,
        "num_layers": 3,
        "n_classes": 1,
        "local_ensemble": True,
        "feature_unfold": True,
        "cell_decode": True,
    }
    base.update(over)
    return OmegaConf.create(base)


def test_make_coord_shape_range_and_xy_order():
    coord = make_coord((2, 4))  # H=2, W=4
    assert coord.shape == (8, 2)
    assert coord.min() >= -1.0 and coord.max() <= 1.0
    # last dim is (x, y): x (col) varies fastest, y (row) is constant within a row
    grid = make_coord((2, 4), flatten=False)  # [2, 4, 2]
    assert torch.allclose(grid[0, :, 1], grid[0, 0, 1])  # y constant across a row
    assert grid[0, 0, 0] < grid[0, -1, 0]                # x increases across columns


def test_make_coord_respects_device_dtype():
    coord = make_coord((3, 3), dtype=torch.float64)
    assert coord.dtype == torch.float64


def test_decoder_forward_shape():
    dec = LIIFDecoder(_cfg(), in_dim=6)
    feat = torch.randn(2, 6, 8, 8)
    coords = torch.rand(2, 50, 2) * 2 - 1
    cell = torch.ones(2, 50, 2) / 8
    out = dec(feat, coords, cell)
    assert out.shape == (2, 50, 1)


def test_decoder_local_ensemble_smooths_seams():
    # Local ensemble blends 4 neighbours by area, so marching across a feature-cell
    # boundary must produce smaller steps than the same weights with the ensemble off
    # (which uses a single nearest sample -> hard piecewise-constant steps at seams).
    torch.manual_seed(0)
    le = LIIFDecoder(_cfg(local_ensemble=True), in_dim=4).eval()
    no_le = LIIFDecoder(_cfg(local_ensemble=False), in_dim=4).eval()
    no_le.load_state_dict(le.state_dict())  # identical weights; only the ensemble differs
    feat = torch.randn(1, 4, 6, 6)
    xs = torch.linspace(-0.4, 0.4, 200)
    coords = torch.stack([xs, torch.zeros_like(xs)], dim=-1)[None]  # [1, 200, 2] (x, y)
    cell = torch.full((1, 200, 2), 2.0 / 6)
    with torch.no_grad():
        out_le = le(feat, coords, cell)[0, :, 0]
        out_no = no_le(feat, coords, cell)[0, :, 0]
    assert out_le.diff().abs().max() < out_no.diff().abs().max()


def test_decoder_cell_decode_changes_output():
    torch.manual_seed(0)
    dec = LIIFDecoder(_cfg(), in_dim=4).eval()
    feat = torch.randn(1, 4, 8, 8)
    coords = torch.rand(1, 20, 2) * 2 - 1
    with torch.no_grad():
        small = dec(feat, coords, torch.full((1, 20, 2), 2.0 / 64))
        large = dec(feat, coords, torch.full((1, 20, 2), 2.0 / 8))
    assert not torch.allclose(small, large)


def test_decoder_is_deterministic():
    dec = LIIFDecoder(_cfg(), in_dim=4).eval()
    feat = torch.randn(1, 4, 8, 8)
    coords = torch.rand(1, 20, 2) * 2 - 1
    cell = torch.ones(1, 20, 2) / 8
    with torch.no_grad():
        a = dec(feat, coords, cell)
        b = dec(feat, coords, cell)
    assert torch.allclose(a, b)


def test_pos_encode_freqs_zero_is_noop():
    # default-off must be byte-identical to a decoder built without the key
    torch.manual_seed(0)
    with_key = LIIFDecoder(_cfg(pos_encode_freqs=0), in_dim=4).eval()
    without_key = LIIFDecoder(_cfg(), in_dim=4).eval()
    without_key.load_state_dict(with_key.state_dict())  # same shapes => no-op path
    feat = torch.randn(1, 4, 6, 6)
    coords = torch.rand(1, 20, 2) * 2 - 1
    cell = torch.ones(1, 20, 2) / 6
    assert torch.allclose(with_key(feat, coords, cell), without_key(feat, coords, cell))


def test_pos_encode_grows_mlp_input():
    base = LIIFDecoder(_cfg(feature_unfold=False, cell_decode=False), in_dim=4)
    enc = LIIFDecoder(
        _cfg(feature_unfold=False, cell_decode=False, pos_encode_freqs=6), in_dim=4
    )
    first_base = base.imnet.net[0].in_features  # in_dim(4) + rel(2) = 6
    first_enc = enc.imnet.net[0].in_features     # in_dim(4) + rel(2*(1+2*6)=26) = 30
    assert first_base == 6
    assert first_enc == 30


def test_pos_encode_forward_shape():
    dec = LIIFDecoder(_cfg(pos_encode_freqs=6), in_dim=4)
    feat = torch.randn(2, 4, 8, 8)
    coords = torch.rand(2, 50, 2) * 2 - 1
    cell = torch.ones(2, 50, 2) / 8
    assert dec(feat, coords, cell).shape == (2, 50, 1)
