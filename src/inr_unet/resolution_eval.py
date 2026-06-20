"""Eval-only cross-resolution comparison of INRUNet vs. the baseline UNet.

decode_dense renders either model to a dense [N, N] sigmoid heatmap at an arbitrary grid;
score_tile scores it against atom positions under physical-unit (Angstrom) detection
tolerances, so detection F1 is comparable across output resolutions.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from inr_unet.localization import peak_localization
from inr_unet.models.components.liif import make_coord
from inr_unet.models.inr_unet import INRUNet
from inr_unet.train import aggregate_localization


def decode_dense(model, img: torch.Tensor, output_size: int) -> torch.Tensor:
    """Dense [output_size, output_size] sigmoid heatmap for ``img`` ([1, 1, 128, 128]).

    INRUNet queries an output_size×output_size grid (cell = 2/output_size), genuinely decoding
    from the fixed 128² feature map. The baseline forwards at its native 128² grid, then the
    sigmoid heatmap is bilinear-resized to output_size (the LIIF-paper interpolation strawman).
    """
    model.eval()
    n = int(output_size)
    with torch.no_grad():
        if isinstance(model, INRUNet):
            coords = make_coord((n, n), device=img.device, dtype=img.dtype)
            coords = coords.unsqueeze(0).expand(img.shape[0], -1, -1)
            cell = torch.empty_like(coords)
            cell[:, :, 0] = 2.0 / n
            cell[:, :, 1] = 2.0 / n
            logits = model(img, coords, cell)  # [B, n*n, 1]
            return torch.sigmoid(logits)[0, :, 0].reshape(n, n)
        hm = torch.sigmoid(model(img))  # [B, 1, 128, 128]
        if n != hm.shape[-1]:
            hm = F.interpolate(hm, size=(n, n), mode="bilinear", align_corners=False)
        return hm[0, 0]


def score_tile(
    heatmap: torch.Tensor,
    positions_A: torch.Tensor,
    extent_A: float,
    *,
    match_tol_A: float,
    min_distance_A: float,
    threshold: float = 0.5,
) -> dict:
    """Score a dense heatmap against atom positions with Angstrom-defined detection tolerances."""
    pixel_size_A = float(extent_A) / heatmap.shape[-1]
    return peak_localization(
        heatmap,
        positions_A,
        pixel_size_A,
        threshold=threshold,
        min_distance_px=float(min_distance_A) / pixel_size_A,
        match_tol_px=float(match_tol_A) / pixel_size_A,
    )


def _metric_row(model_name: str, axis: dict, m) -> dict:
    """Flatten an EvalMetrics ``m`` into a sweep-frame row, prefixed with model + axis value."""
    return {
        "model": model_name,
        **axis,
        "f1": m.f1,
        "precision": m.precision,
        "recall": m.recall,
        "mean_offset_A": m.mean_offset_A,
        "median_offset_A": m.median_offset_A,
        "micro_precision": m.micro_precision,
        "micro_recall": m.micro_recall,
        "n_tiles": m.n_tiles,
        "n_empty": m.n_empty,
    }


def sweep_output_resolution(
    inr, base, dataset, indices, sizes,
    *, match_tol_A: float, min_distance_A: float, threshold: float = 0.5, device: str = "cpu",
) -> pl.DataFrame:
    """Axis A: fixed 128² input per tile, decode each model at every grid in ``sizes`` and score.

    The baseline is decoded natively then bilinear-resized inside ``decode_dense``. Localization
    uses each tile's own ``valid_extent_A`` so the Å offset is physical and comparable across N.
    """
    rows: list[dict] = []
    for name, model in (("inr_unet", inr), ("unet_baseline", base)):
        for n in sizes:
            per_tile = []
            for idx in indices:
                s = dataset.source.get(idx)
                img = s.image[None, None].to(device)
                hm = decode_dense(model, img, n)
                per_tile.append(score_tile(
                    hm, s.positions_A, s.valid_extent_A,
                    match_tol_A=match_tol_A, min_distance_A=min_distance_A, threshold=threshold,
                ))
            m = aggregate_localization(per_tile, loss=float("nan"))
            rows.append(_metric_row(name, {"output_size": int(n)}, m))
    return pl.DataFrame(rows)


def make_resolution_panels(inr, base, dataset, indices, sizes, *, device: str = "cpu"):
    """Per tile: the 128² input plus INR and resized-baseline heatmaps at each grid in ``sizes``."""
    cols: dict[str, list] = {"input": [], "extent_A": []}
    for n in sizes:
        cols[f"inr_{n}"] = []
        cols[f"base_{n}"] = []
    for idx in indices:
        s = dataset.source.get(idx)
        img = s.image[None, None].to(device)
        cols["input"].append(s.image.cpu().numpy())
        cols["extent_A"].append(float(s.valid_extent_A))
        for n in sizes:
            cols[f"inr_{n}"].append(decode_dense(inr, img, n).cpu().numpy())
            cols[f"base_{n}"].append(decode_dense(base, img, n).cpu().numpy())
    return {
        k: (np.asarray(v, dtype="float32") if k == "extent_A" else np.stack(v).astype("float32"))
        for k, v in cols.items()
    }
