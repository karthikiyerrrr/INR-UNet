"""Eval-time ablation probes for the LIIF head.

Re-runs a trained INR-UNet at a chosen output cell size and/or with the local ensemble
toggled, so the broadening of predicted peaks can be attributed to specific decoder
mechanisms without retraining. ``dense_heatmap`` mirrors ``INRUNet``'s dense forward; the
defaults reproduce it exactly.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch

from inr_unet.localization import peak_localization
from inr_unet.models.components.liif import make_coord
from inr_unet.peak_stats import collect_peak_stats, gt_peaks


@contextlib.contextmanager
def _ensemble(decoder, value: bool | None):
    """Temporarily set ``decoder.local_ensemble``; restore it on exit."""
    if value is None:
        yield
        return
    prev = decoder.local_ensemble
    decoder.local_ensemble = bool(value)
    try:
        yield
    finally:
        decoder.local_ensemble = prev


def dense_heatmap(
    model,
    img: torch.Tensor,
    *,
    cell_scale: float = 1.0,
    local_ensemble: bool | None = None,
) -> torch.Tensor:
    """Dense LIIF forward for ``img`` [1, 1, H, W] -> sigmoid heatmap [H, W].

    ``cell_scale`` multiplies the per-pixel cell size handed to the decoder (1.0 = the native
    eval cell ``2/W, 2/H``); ``local_ensemble`` overrides the decoder's blend flag for this call
    only. With both at their defaults the output equals ``torch.sigmoid(model(img))[0, 0]``.
    """
    feat = model.encoder(img)
    b, _, h, w = img.shape
    coords = make_coord((h, w), device=img.device, dtype=img.dtype)
    coords = coords.unsqueeze(0).expand(b, -1, -1)
    cell = torch.empty_like(coords)
    cell[:, :, 0] = (2.0 / w) * cell_scale
    cell[:, :, 1] = (2.0 / h) * cell_scale
    with _ensemble(model.decoder, local_ensemble):
        logits = model.decoder(feat, coords, cell)  # [B, H*W, n_classes]
    n_classes = logits.shape[-1]
    logits = logits.permute(0, 2, 1).reshape(b, n_classes, h, w)
    return torch.sigmoid(logits)[0, 0]


def _median(arr) -> float:
    """Median of a 1-D array, NaN-safe; NaN when empty or all-NaN."""
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else float("nan")


def probe_sweep(model, dataset, indices, configs, *, device: str = "cpu") -> list[dict]:
    """Aggregate localization + peak-sharpness metrics per probe config over ``indices``.

    Each entry of ``configs`` is a dict with ``cell_scale`` (float), ``local_ensemble``
    (bool | None) and ``label`` (str). Returns one row per config; the gt reference peaks come
    from each tile's analytic Gaussian heatmap (rasterized via the dataset's label field) so all
    configs are scored at identical peak locations.
    """
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for cfg in configs:
            preds, peaks_per_tile, gts = [], [], []
            offsets, f1s = [], []
            for idx in indices:
                s = dataset.source.get(idx)
                img = s.image[None, None].to(device)
                hm = dense_heatmap(
                    model,
                    img,
                    cell_scale=float(cfg["cell_scale"]),
                    local_ensemble=cfg["local_ensemble"],
                ).cpu()
                hm_np = hm.numpy()
                m = peak_localization(hm, s.positions_A, s.input_pixel_size_A)
                if not math.isnan(m["mean_offset_A"]):
                    offsets.append(m["mean_offset_A"])
                f1s.append(m["f1"])
                gt_hm = _gt_heatmap(dataset, s, hm_np.shape)
                preds.append(hm_np)
                gts.append(gt_hm)
                peaks_per_tile.append(gt_peaks(gt_hm))
            stats = collect_peak_stats(preds, peaks_per_tile, gts)
            rows.append(
                {
                    "label": cfg["label"],
                    "cell_scale": float(cfg["cell_scale"]),
                    "local_ensemble": cfg["local_ensemble"],
                    "median_offset_A": _median(offsets),
                    "median_fwhm": _median(stats["fwhm"]),
                    "median_height": _median(stats["height"]),
                    "median_floor": _median(stats["floor"]),
                    "f1": float(np.mean(f1s)) if f1s else float("nan"),
                }
            )
    return rows


def _gt_heatmap(dataset, s, shape) -> np.ndarray:
    """Dense gt heatmap for tile ``s`` at ``shape`` (H, W) in pixel space.

    Uses ``dataset.label_field.rasterize`` when the dataset exposes one (the real eval path);
    otherwise splats unit Gaussians at the gt pixel centers so the probe runs standalone.
    """
    h, w = shape
    field = getattr(dataset, "label_field", None)
    if field is not None and hasattr(field, "rasterize"):
        from inr_unet.data.generation.structures import Grid  # local import: optional dep

        grid = Grid(w, s.input_pixel_size_A, device="cpu")
        return field.rasterize(s.positions_A, s.radii_A, grid).cpu().numpy()
    gt = np.zeros((h, w), dtype="float32")
    px = s.positions_A.cpu().numpy() / float(s.input_pixel_size_A)
    ys, xs = np.mgrid[0:h, 0:w]
    for x, y in px:
        if 0 <= x <= w - 1 and 0 <= y <= h - 1:
            gt = np.maximum(gt, np.exp(-((ys - y) ** 2 + (xs - x) ** 2) / (2 * 1.5**2)))
    return gt.astype("float32")
