"""Pixel-space peak-sharpness metrics: sub-pixel offset, effective FWHM, off-peak floor.

Diagnostic counterparts to ``localization.peak_localization``: where localization scores
detection + matching against ground-truth atom positions, these pool per-peak shape statistics
(how tall, how wide, how off-center the predicted peaks are) for comparing heatmap sharpness
across models. Operates on dense NumPy heatmaps; ground-truth peak locations come from the
shared label heatmap so two runs are measured at identical reference points.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

HM_THR = 0.3        # gt fraction-of-max to count as a peak
HM_MIN_DIST = 2     # px; local-maximum neighborhood radius
HM_FWHM_RAD = 4     # px; window radius for half-max-area FWHM
HM_FLOOR_RAD = 3    # px; peak-exclusion radius for the off-peak floor


def gt_peaks(
    gt: np.ndarray, thr: float = HM_THR, min_dist: int = HM_MIN_DIST
) -> list[tuple[int, int]]:
    """(row, col) integer peak locations = local maxima of the shared gt heatmap."""
    mx = maximum_filter(gt, size=2 * int(min_dist) + 1)
    rs, cs = np.where((gt == mx) & (gt >= thr * float(gt.max() or 1.0)))
    return list(zip(rs.tolist(), cs.tolist(), strict=False))


def subpixel_offset(field: np.ndarray, r: int, c: int) -> tuple[float, float]:
    """Parabolic sub-pixel vertex offset (dx, dy) of ``field`` around integer peak (r, c)."""

    def axis(lo, mid, hi):
        denom = lo - 2.0 * mid + hi
        if denom == 0:
            return 0.0
        return float(np.clip(0.5 * (lo - hi) / denom, -0.5, 0.5))

    h, w = field.shape
    dy = axis(field[max(0, r - 1), c], field[r, c], field[min(h - 1, r + 1), c])
    dx = axis(field[r, max(0, c - 1)], field[r, c], field[r, min(w - 1, c + 1)])
    return dx, dy


def peak_fwhm(pred: np.ndarray, r: int, c: int, rad: int = HM_FWHM_RAD) -> float:
    """Effective FWHM diameter (px) from the half-max area in a window around (r, c)."""
    peak = float(pred[r, c])
    if peak <= 0:
        return float("nan")
    win = pred[max(0, r - rad) : r + rad + 1, max(0, c - rad) : c + rad + 1]
    area = float((win >= 0.5 * peak).sum())
    return 2.0 * float(np.sqrt(area / np.pi))


def off_peak_floor(
    pred: np.ndarray, peaks: list[tuple[int, int]], rad: int = HM_FLOOR_RAD
) -> float:
    """Mean prediction away from every peak (a proxy for the false-positive floor)."""
    mask = np.ones(pred.shape, dtype=bool)
    for r, c in peaks:
        mask[max(0, r - rad) : r + rad + 1, max(0, c - rad) : c + rad + 1] = False
    return float(pred[mask].mean()) if mask.any() else float("nan")


def collect_peak_stats(preds, peaks_per_tile, gts) -> dict[str, np.ndarray]:
    """Pool per-peak offset/height/fwhm and per-tile floor over a run's tiles."""
    off, ht, fw, fl = [], [], [], []
    for pred, peaks, gt in zip(preds, peaks_per_tile, gts, strict=False):
        for r, c in peaks:
            dxg, dyg = subpixel_offset(gt, r, c)
            dxp, dyp = subpixel_offset(pred, r, c)
            off.append(float(np.hypot(dxp - dxg, dyp - dyg)))
            ht.append(float(pred[r, c]))
            fw.append(peak_fwhm(pred, r, c))
        fl.append(off_peak_floor(pred, peaks))
    return {
        "offset": np.array(off),
        "height": np.array(ht),
        "fwhm": np.array(fw),
        "floor": np.array(fl),
    }
