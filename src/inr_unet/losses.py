"""Segmentation loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DiceLoss.forward is not implemented yet.")


def dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, *, dice_weight: float = 1.0
) -> torch.Tensor:
    """Combined Dice + binary cross-entropy loss."""
    raise NotImplementedError("dice_bce_loss is not implemented yet.")
