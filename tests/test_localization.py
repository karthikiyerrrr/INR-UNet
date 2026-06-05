"""Known-value tests for peak detection + sub-pixel localization."""

import math

import pytest
import torch

from inr_unet.localization import peak_localization


def _splat(size, peaks_xy_px, sigma_px=1.2):
    """Render unit-height Gaussian blobs at (x, y) pixel centers on a [size, size] grid."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    hm = torch.zeros(size, size)
    for x, y in peaks_xy_px:
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        hm = torch.maximum(hm, torch.exp(-d2 / (2 * sigma_px**2)))
    return hm


def test_perfect_detection_and_recall():
    ps = 0.1
    peaks_px = [(8, 8), (20, 24), (30, 12)]
    hm = _splat(40, peaks_px)
    gt_A = torch.tensor([[x * ps, y * ps] for x, y in peaks_px])
    out = peak_localization(hm, gt_A, ps)
    assert out["recall"] == 1.0
    assert out["precision"] == 1.0
    assert out["f1"] == 1.0
    assert out["mean_offset_A"] < 0.02  # << 1 px


def test_subpixel_offset_recovered():
    ps = 0.1
    hm = _splat(40, [(20.3, 15.0)])  # fractional x offset
    gt_A = torch.tensor([[20.3 * ps, 15.0 * ps]])
    out = peak_localization(hm, gt_A, ps)
    assert out["n_matched"] == 1
    assert out["mean_offset_A"] < 0.05 * ps + 0.01  # sub-pixel refinement lands close


def test_threshold_suppresses_weak_peaks():
    ps = 0.1
    hm = 0.3 * _splat(40, [(10, 10)])  # peak height 0.3
    gt_A = torch.tensor([[10 * ps, 10 * ps]])
    out = peak_localization(hm, gt_A, ps, threshold=0.5)
    assert out["n_pred"] == 0
    assert out["recall"] == 0.0


def test_spurious_peak_lowers_precision():
    ps = 0.1
    hm = _splat(40, [(10, 10), (30, 30)])
    gt_A = torch.tensor([[10 * ps, 10 * ps]])  # only one real atom; (30,30) is spurious
    out = peak_localization(hm, gt_A, ps, match_tol_px=2.0)
    assert out["n_pred"] == 2
    assert out["recall"] == 1.0
    assert out["precision"] == pytest.approx(0.5)


def test_empty_prediction_and_empty_gt():
    ps = 0.1
    hm = torch.zeros(16, 16)
    out = peak_localization(hm, torch.zeros(0, 2), ps)
    assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0
    assert math.isnan(out["mean_offset_A"]) or out["mean_offset_A"] == 0.0


def test_off_fov_gt_positions_excluded_from_recall():
    ps = 0.1
    hm = _splat(40, [(10, 10), (30, 30)])  # 40x40 grid -> in-FOV extent ~[0, 3.9] A
    # two real in-FOV columns + one far outside the heatmap (x = 100 px = 10 A)
    gt_A = torch.tensor([[10 * ps, 10 * ps], [30 * ps, 30 * ps], [100 * ps, 5 * ps]])
    out = peak_localization(hm, gt_A, ps)
    assert out["n_gt"] == 2  # off-FOV column dropped, not charged as a miss
    assert out["recall"] == 1.0
    assert out["precision"] == 1.0
