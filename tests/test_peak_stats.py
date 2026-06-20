"""Contract tests for the lifted peak-sharpness helpers."""

import numpy as np

from inr_unet.peak_stats import (
    collect_peak_stats,
    gt_peaks,
    off_peak_floor,
    peak_fwhm,
    subpixel_offset,
)


def _gaussian(h, w, cy, cx, sigma, amp=1.0):
    ys, xs = np.mgrid[0:h, 0:w]
    return amp * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma**2))


def test_gt_peaks_finds_single_maximum():
    field = _gaussian(21, 21, 10, 10, 2.0)
    peaks = gt_peaks(field)
    assert peaks == [(10, 10)]


def test_subpixel_offset_centered_peak_is_zero():
    field = _gaussian(21, 21, 10, 10, 2.0)
    dx, dy = subpixel_offset(field, 10, 10)
    assert abs(dx) < 1e-6 and abs(dy) < 1e-6


def test_subpixel_offset_recovers_known_shift():
    # peak displaced +0.3 px in x: the parabolic vertex should lean positive in x
    field = _gaussian(21, 21, 10.0, 10.3, 2.0)
    dx, dy = subpixel_offset(field, 10, 10)
    assert dx > 0.1 and abs(dy) < 1e-6


def test_peak_fwhm_matches_gaussian_formula():
    # FWHM of a 2D Gaussian = 2*sqrt(2 ln 2) * sigma; the half-max-area estimator
    # should land within ~15% of it for a well-sampled peak
    sigma = 2.5
    field = _gaussian(31, 31, 15, 15, sigma)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert abs(peak_fwhm(field, 15, 15, rad=6) - expected) < 0.15 * expected


def test_off_peak_floor_excludes_peak_neighborhood():
    field = _gaussian(21, 21, 10, 10, 2.0) + 0.05
    floor = off_peak_floor(field, [(10, 10)])
    assert abs(floor - 0.05) < 0.02


def test_collect_peak_stats_pools_per_peak_and_per_tile():
    gt = _gaussian(21, 21, 10, 10, 2.0)
    pred = _gaussian(21, 21, 10, 10, 2.5, amp=0.8)
    peaks = gt_peaks(gt)
    stats = collect_peak_stats([pred], [peaks], [gt])
    assert set(stats) == {"offset", "height", "fwhm", "floor"}
    assert stats["offset"].shape == (1,)
    assert stats["floor"].shape == (1,)
    assert abs(float(stats["height"][0]) - 0.8) < 1e-6
