"""Normalization and augmentation transforms for S/TEM images."""

from __future__ import annotations

import torch


class Normalize:
    """Normalize an image tensor using a fixed mean and standard deviation."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Normalize.__call__ is not implemented yet.")
