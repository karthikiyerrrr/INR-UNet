"""Segmentation metrics. Inputs are predicted probabilities (or masks) and targets."""

from __future__ import annotations

import torch


def iou(pred: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Intersection-over-union for binary masks; 1.0 when both are empty."""
    p = pred >= threshold
    t = target >= 0.5
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return 1.0 if union == 0 else inter / union


def dice_score(pred: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Dice coefficient for binary masks; 1.0 when both are empty."""
    p = pred >= threshold
    t = target >= 0.5
    inter = (p & t).sum().item()
    denom = p.sum().item() + t.sum().item()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Fraction of correctly classified pixels."""
    p = pred >= threshold
    t = target >= 0.5
    return (p == t).float().mean().item()
