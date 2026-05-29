"""Segmentation metrics."""

from __future__ import annotations

import torch


def iou(pred: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Intersection-over-union for binary masks."""
    raise NotImplementedError("iou is not implemented yet.")


def dice_score(pred: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Dice coefficient for binary masks."""
    raise NotImplementedError("dice_score is not implemented yet.")


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Fraction of correctly classified pixels."""
    raise NotImplementedError("pixel_accuracy is not implemented yet.")
