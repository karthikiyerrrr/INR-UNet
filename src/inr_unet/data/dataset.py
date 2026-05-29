"""Datasets for S/TEM segmentation, including LIIF-style coordinate sampling."""

from __future__ import annotations

from torch.utils.data import Dataset


class STEMSegDataset(Dataset):
    """Yields ``(image, mask)`` pairs for S/TEM segmentation."""

    def __init__(self, root: str, image_size: int = 256) -> None:
        self.root = root
        self.image_size = image_size

    def __len__(self) -> int:
        raise NotImplementedError("STEMSegDataset.__len__ is not implemented yet.")

    def __getitem__(self, idx: int):
        raise NotImplementedError("STEMSegDataset.__getitem__ is not implemented yet.")


class LIIFSegDataset(Dataset):
    """Wraps a base dataset to yield ``(lr_image, coords, cell, gt_values)`` for
    resolution-agnostic training."""

    def __init__(self, base: Dataset, sample_q: int | None = None) -> None:
        self.base = base
        self.sample_q = sample_q

    def __len__(self) -> int:
        raise NotImplementedError("LIIFSegDataset.__len__ is not implemented yet.")

    def __getitem__(self, idx: int):
        raise NotImplementedError("LIIFSegDataset.__getitem__ is not implemented yet.")
