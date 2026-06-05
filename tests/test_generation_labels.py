"""Label geometry: radius formula, circular-mask area, equalized Gaussian mask."""

import math

import pytest
import torch

from inr_unet.data.generation.labels import (
    CircularField,
    GaussianField,
    circular_mask,
    column_radius,
    gaussian_mask,
)
from inr_unet.data.generation.structures import Grid, ImagingCondition
from inr_unet.registry import LABEL_FIELDS


def test_column_radius_formula():
    cond = ImagingCondition(200.0, 24.0, 0.9, name="t")
    lam = 0.025079
    alpha = 24.0e-3
    expected = 0.5 * math.sqrt((lam / alpha) ** 2 + (0.9 / 2.0) ** 2)
    assert column_radius(cond) == pytest.approx(expected, rel=1e-3)


def test_circular_mask_area():
    grid = Grid(output_size=128, pixel_size_A=0.05)  # 6.4 A FOV
    pos = torch.tensor([[3.2, 3.2]])
    r_A = 1.0
    radii = torch.tensor([r_A])
    mask = circular_mask(pos, radii, grid)
    area_px = mask.sum().item()
    expected_px = math.pi * (r_A / grid.pixel_size_A) ** 2
    assert abs(area_px - expected_px) / expected_px < 0.05


def test_gaussian_mask_equalized_across_columns():
    grid = Grid(output_size=64, pixel_size_A=0.05)
    pos = torch.tensor([[1.0, 1.0], [2.4, 2.4]])
    mask = gaussian_mask(pos, grid)
    assert mask.max().item() == pytest.approx(1.0, abs=1e-5)
    # both columns reach the same peak (equal weights), so two distinct near-1 peaks
    peaks = mask[mask > 0.99]
    assert peaks.numel() >= 2


def test_label_fields_registered():
    assert "gaussian" in LABEL_FIELDS
    assert "circular" in LABEL_FIELDS
    assert isinstance(LABEL_FIELDS.get("gaussian")(), GaussianField)
    assert isinstance(LABEL_FIELDS.get("circular")(), CircularField)


def test_gaussian_values_at_peak_and_decay():
    pos = torch.tensor([[1.0, 1.0], [2.4, 2.4]])
    radii = torch.zeros(2)
    field = GaussianField()
    # query exactly on a column -> 1.0; far away -> ~0
    q = torch.tensor([[1.0, 1.0], [2.4, 2.4], [10.0, 10.0]])
    vals = field.values_at(pos, radii, q)
    assert vals[0].item() == pytest.approx(1.0, abs=1e-5)
    assert vals[1].item() == pytest.approx(1.0, abs=1e-5)
    assert vals[2].item() < 1e-3


def test_circular_values_at_binary():
    pos = torch.tensor([[5.0, 5.0]])
    radii = torch.tensor([1.0])
    field = CircularField()
    q = torch.tensor([[5.0, 5.0], [5.5, 5.0], [7.0, 5.0]])  # inside, inside, outside
    vals = field.values_at(pos, radii, q)
    assert vals[0].item() == 1.0
    assert vals[1].item() == 1.0
    assert vals[2].item() == 0.0


def test_rasterize_matches_values_at_on_grid():
    grid = Grid(output_size=32, pixel_size_A=0.1)
    pos = torch.tensor([[1.0, 1.5], [2.0, 0.7]])
    radii = torch.full((2,), 0.3)
    for field in (GaussianField(), CircularField()):
        raster = field.rasterize(pos, radii, grid)
        yy, xx = grid.pixel_coords()
        q = torch.stack([(xx * grid.pixel_size_A).reshape(-1),
                         (yy * grid.pixel_size_A).reshape(-1)], dim=-1)
        flat = field.values_at(
            pos, radii, q, grid.pixel_size_A
        ).reshape(grid.output_size, grid.output_size)
        assert torch.allclose(raster, flat, atol=1e-6)


def test_values_at_empty_columns():
    field = GaussianField()
    q = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    vals = field.values_at(torch.zeros((0, 2)), torch.zeros((0,)), q)
    assert vals.shape == (2,)
    assert torch.all(vals == 0)


def test_gaussian_values_at_discriminates_axes():
    field = GaussianField()
    pos = torch.tensor([[1.0, 3.0]])  # x=1, y=3
    radii = torch.zeros(1)
    on = field.values_at(pos, radii, torch.tensor([[1.0, 3.0]]))
    swapped = field.values_at(pos, radii, torch.tensor([[3.0, 1.0]]))
    assert on[0].item() == pytest.approx(1.0, abs=1e-5)
    assert swapped[0].item() < 1e-3  # (3,1) is far from the column at (1,3)


def test_circular_values_at_distinct_radii():
    field = CircularField()
    pos = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    radii = torch.tensor([0.5, 2.0])
    # 1.5 A from each column: outside the r=0.5 disk, inside the r=2.0 disk
    q = torch.tensor([[1.5, 0.0], [8.5, 0.0]])
    vals = field.values_at(pos, radii, q)
    assert vals[0].item() == 0.0  # near small-radius column, outside
    assert vals[1].item() == 1.0  # near large-radius column, inside


def test_circular_values_at_empty_columns():
    field = CircularField()
    q = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    vals = field.values_at(torch.zeros((0, 2)), torch.zeros((0,)), q)
    assert vals.shape == (3,)
    assert torch.all(vals == 0)


def test_gaussian_sigma_floored_at_coarse_pixel():
    field = GaussianField()  # fwhm 0.2 A -> physical sigma ~0.085 A
    pos = torch.tensor([[0.0, 0.0]])
    radii = torch.zeros(1)
    q = torch.tensor([[0.2, 0.0]])  # 0.2 A off-center
    physical = field.values_at(pos, radii, q).item()                       # no floor
    floored = field.values_at(pos, radii, q, pixel_size_A=0.30).item()     # floor 0.30 px-A
    assert floored > physical + 0.3  # floored sigma (0.30 A) is much wider


def test_gaussian_sigma_not_floored_at_fine_pixel():
    field = GaussianField()
    pos = torch.tensor([[0.0, 0.0]])
    radii = torch.zeros(1)
    q = torch.tensor([[0.2, 0.0]])
    physical = field.values_at(pos, radii, q).item()
    fine = field.values_at(pos, radii, q, pixel_size_A=0.05).item()  # floor 0.05 < 0.085
    assert fine == pytest.approx(physical, abs=1e-6)


def test_gaussian_peak_height_unchanged_by_floor():
    field = GaussianField()
    pos = torch.tensor([[0.0, 0.0]])
    radii = torch.zeros(1)
    on_peak = field.values_at(pos, radii, torch.tensor([[0.0, 0.0]]), pixel_size_A=0.30)
    assert on_peak[0].item() == pytest.approx(1.0, abs=1e-5)


def test_gaussian_coarse_raster_is_a_blob_not_a_spike():
    grid = Grid(output_size=32, pixel_size_A=0.30)  # coarse: physical sigma would be sub-Nyquist
    pos = torch.tensor([[4.8, 4.8]])  # near grid center (16 px * 0.30)
    raster = GaussianField().rasterize(pos, torch.zeros(1), grid)
    assert (raster > 0.5).sum().item() >= 3  # a sampled blob, not a single-pixel spike


def test_gaussian_field_respects_constructor_params():
    narrow = GaussianField(fwhm_A=0.1, sigma_floor_px=0.0)
    wide = GaussianField(fwhm_A=0.4, sigma_floor_px=0.0)
    pos = torch.tensor([[0.0, 0.0]])
    radii = torch.zeros(1)
    q = torch.tensor([[0.15, 0.0]])
    assert wide.values_at(pos, radii, q).item() > narrow.values_at(pos, radii, q).item()
