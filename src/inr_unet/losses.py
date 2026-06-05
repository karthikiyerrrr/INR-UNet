"""Segmentation loss functions."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


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


def weighted_mse_loss(
    logits: torch.Tensor, target: torch.Tensor, *, fg_weight: float = 10.0
) -> torch.Tensor:
    """Foreground-weighted MSE between sigmoid(logits) and a soft target in [0, 1].

    Peaks are a tiny fraction of pixels, so an unweighted MSE is background-dominated.
    The per-pixel weight ``1 + fg_weight * target`` scales the penalty with the target
    value, emphasizing peaks without a hard threshold.
    """
    probs = torch.sigmoid(logits)
    w = 1.0 + fg_weight * target
    return (w * (probs - target) ** 2).sum() / w.sum()


def make_loss(cfg: DictConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the training loss callable ``(logits, target) -> loss`` from config."""
    name = cfg.train.loss
    if name == "weighted_mse":
        return partial(weighted_mse_loss, fg_weight=float(cfg.train.fg_weight))
    if name == "dice_bce":
        return dice_bce_loss
    raise ValueError(f"unknown loss {name!r} (expected 'weighted_mse' or 'dice_bce')")
