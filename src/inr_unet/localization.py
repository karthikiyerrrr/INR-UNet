"""Peak detection + sub-pixel localization metric for equalized Gaussian heatmaps.

Decoupled from the training loss: detect local maxima, refine to sub-pixel, greedily match
to ground-truth atom positions, and report detection precision/recall/F1 plus the mean
sub-pixel localization offset (in Angstroms).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _local_maxima(
    heatmap: torch.Tensor, threshold: float, min_distance_px: float
) -> tuple[list[int], list[int], list[float]]:
    """Return (rows, cols, values) of NMS-filtered local maxima, descending by value."""
    h = heatmap[None, None]
    pooled = F.max_pool2d(h, kernel_size=3, stride=1, padding=1)
    is_max = (h == pooled) & (h > threshold)
    ys, xs = torch.nonzero(is_max[0, 0], as_tuple=True)
    if ys.numel() == 0:
        return [], [], []
    vals = heatmap[ys, xs]
    order = torch.argsort(vals, descending=True)
    ys, xs, vals = ys[order].tolist(), xs[order].tolist(), vals[order].tolist()
    kept_r, kept_c, kept_v = [], [], []
    for r, c, v in zip(ys, xs, vals, strict=True):
        if all(
            (r - kr) ** 2 + (c - kc) ** 2 >= min_distance_px**2
            for kr, kc in zip(kept_r, kept_c, strict=True)
        ):
            kept_r.append(r)
            kept_c.append(c)
            kept_v.append(v)
    return kept_r, kept_c, kept_v


def _subpixel_offset(heatmap: torch.Tensor, r: int, c: int) -> tuple[float, float]:
    """Parabolic sub-pixel offset (dx, dy) around the integer maximum at (r, c)."""
    h, w = heatmap.shape

    def refine(lo: torch.Tensor, mid: torch.Tensor, hi: torch.Tensor) -> float:
        denom = lo - 2 * mid + hi
        if denom.abs() < 1e-12:
            return 0.0
        return float((0.5 * (lo - hi) / denom).clamp(-0.5, 0.5))

    dx = refine(heatmap[r, c - 1], heatmap[r, c], heatmap[r, c + 1]) if 0 < c < w - 1 else 0.0
    dy = refine(heatmap[r - 1, c], heatmap[r, c], heatmap[r + 1, c]) if 0 < r < h - 1 else 0.0
    return dx, dy


def peak_localization(
    pred_heatmap: torch.Tensor,
    gt_positions_A: torch.Tensor,
    pixel_size_A: float,
    *,
    threshold: float = 0.5,
    min_distance_px: float = 2.0,
    match_tol_px: float = 2.0,
) -> dict[str, float]:
    """Detect peaks in ``pred_heatmap`` and match them to ``gt_positions_A`` (x, y in Angstroms).

    Returns precision, recall, f1, mean_offset_A (over matched peaks; NaN if none),
    n_pred, n_gt, n_matched. Matching is greedy by descending peak value, in pixel space.
    """
    hm = pred_heatmap.detach().float().cpu()
    rows, cols, _ = _local_maxima(hm, threshold, min_distance_px)
    pred_xy_px = [
        (c + dx, r + dy)
        for r, c in zip(rows, cols, strict=True)
        for dx, dy in [_subpixel_offset(hm, r, c)]
    ]  # (x=col, y=row), descending confidence

    gt = gt_positions_A.detach().float().cpu()
    n_pred, n_gt = len(pred_xy_px), int(gt.shape[0])

    if n_gt == 0:
        precision = 1.0 if n_pred == 0 else 0.0
        return {
            "precision": precision,
            "recall": 1.0,
            "f1": precision,
            "mean_offset_A": float("nan"),
            "n_pred": n_pred,
            "n_gt": 0,
            "n_matched": 0,
        }

    gt_xy_px = gt / pixel_size_A  # [M, 2] (x, y)
    used = torch.zeros(n_gt, dtype=torch.bool)
    offsets_px: list[float] = []
    for px, py in pred_xy_px:
        d = torch.sqrt((gt_xy_px[:, 0] - px) ** 2 + (gt_xy_px[:, 1] - py) ** 2)
        d[used] = float("inf")
        j = int(torch.argmin(d))
        if float(d[j]) <= match_tol_px:
            used[j] = True
            offsets_px.append(float(d[j]))

    n_matched = len(offsets_px)
    precision = n_matched / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = n_matched / n_gt
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_offset_A = float(np.mean(offsets_px)) * pixel_size_A if offsets_px else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_offset_A": mean_offset_A,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_matched": n_matched,
    }
