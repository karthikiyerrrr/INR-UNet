"""Segmentation loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation; operates on sigmoid(logits)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        tgt = target.reshape(target.shape[0], -1)
        inter = (probs * tgt).sum(dim=1)
        denom = probs.sum(dim=1) + tgt.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


def dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, *, dice_weight: float = 1.0
) -> torch.Tensor:
    """Combined binary cross-entropy + (weighted) Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = DiceLoss()(logits, target)
    return bce + dice_weight * dice
