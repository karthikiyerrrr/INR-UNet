"""Data contracts for the forward model: dataclasses, presets, Grid, validation."""

import pytest
import torch

from inr_unet.data import TEMRenderer
from inr_unet.data.generation.structures import (
    IMAGING_CONDITIONS,
    BackgroundSpec,
    ColumnList,
    Grid,
    NoiseSpec,
    RenderParams,
)


def _columns(fov_A=25.6):
    return ColumnList(
        positions_A=torch.tensor([[5.0, 5.0], [10.0, 12.0]]),
        z=torch.tensor([78.0, 8.0]),
        count=torch.tensor([4.0, 2.0]),
        fov_A=fov_A,
    )


def test_imaging_condition_presets():
    assert set(IMAGING_CONDITIONS) == {"cond1", "cond2", "cond3", "cond4", "cond5"}
    c1 = IMAGING_CONDITIONS["cond1"]
    assert c1.energy_keV == 200.0
    assert c1.alpha_max_mrad == 24.0
    assert c1.source_size_A == 0.9
    assert c1.sigma_jitter_A == 0.1
    assert c1.name == "cond1"


def test_columnlist_validate_ok():
    cols = _columns()
    cols.validate(Grid(output_size=256, pixel_size_A=0.1))  # 25.6 A FOV, fits


def test_columnlist_validate_fov_too_small():
    cols = _columns(fov_A=10.0)
    with pytest.raises(ValueError, match="fov_A"):
        cols.validate(Grid(output_size=256, pixel_size_A=0.1))


def test_columnlist_validate_count_positive():
    bad = ColumnList(
        positions_A=torch.tensor([[1.0, 1.0]]),
        z=torch.tensor([8.0]),
        count=torch.tensor([0.0]),
        fov_A=25.6,
    )
    with pytest.raises(ValueError, match="count"):
        bad.validate(Grid(output_size=256, pixel_size_A=0.1))


def test_grid_extent_and_normalized_coords():
    g = Grid(output_size=4, pixel_size_A=0.5)
    assert g.extent_A == 2.0
    yy, xx = g.normalized_coords()
    assert yy.shape == (4, 4)
    assert torch.isclose(xx.min(), torch.tensor(-1.0))
    assert torch.isclose(xx.max(), torch.tensor(1.0))


def test_renderparams_defaults():
    p = RenderParams()
    assert p.output_size == 256
    assert isinstance(p.background, BackgroundSpec)
    assert isinstance(p.noise, NoiseSpec)
    assert p.position_offset_A.shape == (2,)


def test_columnlist_accepts_optional_lattice_basis():
    import torch

    from inr_unet.data.generation.structures import ColumnList, Grid

    basis = torch.tensor([[3.9, 0.0], [0.0, 3.9]])
    cols = ColumnList(
        positions_A=torch.zeros(1, 2),
        z=torch.tensor([78.0]),
        count=torch.tensor([1.0]),
        fov_A=60.0,
        lattice_basis_A=basis,
    )
    cols.validate(Grid(output_size=10, pixel_size_A=1.0))
    assert torch.equal(cols.lattice_basis_A, basis)


def test_columnlist_defaults_lattice_basis_to_none():
    import torch

    from inr_unet.data.generation.structures import ColumnList

    cols = ColumnList(
        positions_A=torch.zeros(0, 2),
        z=torch.zeros(0),
        count=torch.zeros(0),
        fov_A=60.0,
    )
    assert cols.lattice_basis_A is None


def test_columnlist_rejects_misshaped_lattice_basis():
    import pytest
    import torch

    from inr_unet.data.generation.structures import ColumnList, Grid

    cols = ColumnList(
        positions_A=torch.zeros(1, 2),
        z=torch.tensor([78.0]),
        count=torch.tensor([1.0]),
        fov_A=60.0,
        lattice_basis_A=torch.zeros(3, 2),
    )
    with pytest.raises(ValueError):
        cols.validate(Grid(output_size=10, pixel_size_A=1.0))


def test_imaging_condition_astig_defaults_zero():
    from inr_unet.data.generation.structures import ImagingCondition

    cond = ImagingCondition(200.0, 24.0, 0.9, name="t")
    assert cond.astig_a1_A == 0.0
    assert cond.astig_a1_azimuth_rad == 0.0
    # every named preset stays radial-only (defocus and astig all zero)
    for c in IMAGING_CONDITIONS.values():
        assert c.defocus_A == 0.0
        assert c.astig_a1_A == 0.0
        assert c.astig_a1_azimuth_rad == 0.0


def test_renderer_is_implemented():
    from omegaconf import OmegaConf

    from inr_unet.data.generation.structures import RenderOutput

    cfg = OmegaConf.create(
        {"potential_backend": "z_power", "sigma_potential_A": 0.4, "aperture_soft": True}
    )
    r = TEMRenderer(cfg)
    out = r.render(
        _columns(),
        IMAGING_CONDITIONS["cond1"],
        RenderParams(output_size=64, pixel_size_A=0.4, seed=0),
    )
    assert isinstance(out, RenderOutput)
