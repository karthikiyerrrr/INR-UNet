"""Synthetic S/TEM image and segmentation-mask generation (TEMImageNet style)."""

from __future__ import annotations

import numpy as np


class TEMImageNetGenerator:
    """Generate synthetic S/TEM images with paired segmentation masks from simulation parameters."""

    def __init__(self, image_size: int = 256, seed: int | None = None) -> None:
        self.image_size = image_size
        self.seed = seed

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a single ``(image, mask)`` pair, each of shape [H, W]."""
        raise NotImplementedError("TEMImageNetGenerator.generate is not implemented yet.")
