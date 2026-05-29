"""End-to-end renderer: output contract, determinism, source-size acceptance."""

import torch
from omegaconf import OmegaConf

from inr_unet.data import TEMRenderer
from inr_unet.data.generation.structures import (
    IMAGING_CONDITIONS,
    ColumnList,
    RenderOutput,
    RenderParams,
)


def _renderer():
    cfg = OmegaConf.create(
        {"potential_backend": "z_power", "sigma_potential_A": 0.4, "aperture_soft": True}
    )
    return TEMRenderer(cfg)


def _lattice(fov_A=12.8):
    xs = torch.linspace(2.0, fov_A - 2.0, 4)
    pts = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)
    n = pts.shape[0]
    return ColumnList(positions_A=pts, z=torch.full((n,), 78.0), count=torch.ones(n), fov_A=fov_A)


def test_render_output_contract():
    r = _renderer()
    p = RenderParams(output_size=64, pixel_size_A=0.2, seed=1)
    out = r.render(_lattice(), IMAGING_CONDITIONS["cond1"], p)
    assert isinstance(out, RenderOutput)
    for field in ("image", "gaussian_mask", "circular_mask", "no_noise", "no_background_no_noise"):
        assert getattr(out, field).shape == (64, 64)
    assert out.meta.condition_name == "cond1"
    assert out.positions_A.shape[1] == 2
    assert out.radii_A.shape[0] == out.positions_A.shape[0]


def test_render_deterministic_given_seed():
    r = _renderer()
    p = RenderParams(output_size=48, pixel_size_A=0.2, seed=7)
    a = r.render(_lattice(), IMAGING_CONDITIONS["cond1"], p)
    b = r.render(_lattice(), IMAGING_CONDITIONS["cond1"], p)
    assert torch.allclose(a.image, b.image)


def test_render_seed_changes_noise():
    r = _renderer()
    base = _lattice()
    a = r.render(
        base, IMAGING_CONDITIONS["cond1"], RenderParams(output_size=48, pixel_size_A=0.2, seed=1)
    )
    b = r.render(
        base, IMAGING_CONDITIONS["cond1"], RenderParams(output_size=48, pixel_size_A=0.2, seed=2)
    )
    assert not torch.allclose(a.image, b.image)


def test_source_size_acceptance():
    # A single column's blob FWHM should be of order 2r from the radius formula.
    from inr_unet.data.generation.labels import column_radius

    r = _renderer()
    cond = IMAGING_CONDITIONS["cond1"]
    fov = 6.4
    cols = ColumnList(
        positions_A=torch.tensor([[fov / 2, fov / 2]]),
        z=torch.tensor([78.0]),
        count=torch.tensor([1.0]),
        fov_A=fov,
    )
    p = RenderParams(output_size=128, pixel_size_A=fov / 128, seed=0)
    out = r.render(cols, cond, p)
    clean = out.no_background_no_noise
    row = clean[clean.shape[0] // 2]
    half = row.max() / 2
    above = (row >= half).nonzero().flatten()
    fwhm_A = float(above[-1] - above[0]) * p.pixel_size_A
    expected = 2.0 * column_radius(cond)
    assert 0.6 * expected <= fwhm_A <= 1.5 * expected


def test_empty_columns_renders_background_only():
    r = _renderer()
    empty = ColumnList(
        positions_A=torch.zeros(0, 2), z=torch.zeros(0), count=torch.zeros(0), fov_A=12.8
    )
    out = r.render(
        empty, IMAGING_CONDITIONS["cond1"], RenderParams(output_size=32, pixel_size_A=0.4, seed=0)
    )
    assert out.no_background_no_noise.abs().sum() == 0.0
    assert out.image.shape == (32, 32)


def test_render_centers_window_in_larger_structure():
    # Structure confined to the center band [7, 13] of a 20 A box.
    # With the old corner-pivot the rotated render window misses the structure entirely
    # (gmax==0); with the structure-center pivot the window lands on the structure.
    r = _renderer()
    xs = torch.arange(7.0, 13.5, 0.5)
    pts = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)
    n = pts.shape[0]
    cols = ColumnList(
        positions_A=pts, z=torch.full((n,), 78.0), count=torch.ones(n), fov_A=20.0
    )
    # render FOV = 64 * 0.1 = 6.4 A, rotated 45 deg, centered in the 20 A structure
    p = RenderParams(output_size=64, pixel_size_A=0.1, rotation_deg=45.0, seed=0)
    out = r.render(cols, IMAGING_CONDITIONS["cond1"], p)
    clean = out.no_background_no_noise
    gmax = float(clean.max())
    # window must land on the structure (not in empty space)
    assert gmax > 0.0
    h = clean.shape[0] // 2
    quadrants = [clean[:h, :h], clean[:h, h:], clean[h:, :h], clean[h:, h:]]
    # every quadrant is covered by structure -> no spurious vacuum wedge from rotation
    for q in quadrants:
        assert float(q.max()) > 0.2 * gmax
