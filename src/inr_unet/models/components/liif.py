"""LIIF continuous local implicit representation: decoder + coordinate utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from inr_unet.registry import DECODERS


def make_coord(
    shape: tuple[int, int],
    ranges: tuple[tuple[float, float], ...] | None = None,
    flatten: bool = True,
) -> torch.Tensor:
    """Create a grid of normalized coordinates for a feature map of the given shape."""
    raise NotImplementedError("make_coord is not implemented yet.")


@DECODERS.register("liif")
class LIIFDecoder(nn.Module):
    """Local implicit MLP decoder that queries a feature map at continuous coordinates."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, feat: torch.Tensor, coords: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Map feat [B, D, H, W] + coords [B, Q, 2] + cell [B, Q, 2] to logits [B, Q, n_classes]."""
        raise NotImplementedError("LIIFDecoder.forward is not implemented yet.")
