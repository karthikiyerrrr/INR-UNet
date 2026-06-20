"""Eval-time ablation probes for the LIIF head.

Re-runs a trained INR-UNet at a chosen output cell size and/or with the local ensemble
toggled, so the broadening of predicted peaks can be attributed to specific decoder
mechanisms without retraining. ``dense_heatmap`` mirrors ``INRUNet``'s dense forward; the
defaults reproduce it exactly.
"""

from __future__ import annotations

import contextlib

import torch

from inr_unet.models.components.liif import make_coord


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
