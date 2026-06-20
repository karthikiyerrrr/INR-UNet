"""Eval-only cross-resolution comparison of INRUNet vs. the baseline UNet.

decode_dense renders either model to a dense [N, N] sigmoid heatmap at an arbitrary grid;
score_tile scores it against atom positions under physical-unit (Angstrom) detection
tolerances, so detection F1 is comparable across output resolutions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from inr_unet.localization import peak_localization
from inr_unet.models.components.liif import make_coord
from inr_unet.models.inr_unet import INRUNet


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
