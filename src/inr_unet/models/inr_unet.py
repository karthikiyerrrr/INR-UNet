"""Assembled model classes combining the UNet encoder with the LIIF decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from inr_unet.models.components.liif import LIIFDecoder
from inr_unet.models.components.unet import UNetEncoder
from inr_unet.registry import MODELS


@MODELS.register("inr_unet")
class INRUNet(nn.Module):
    """Resolution-agnostic UNet: a UNet encoder feeding a LIIF implicit decoder."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = UNetEncoder(cfg.encoder)
        self.decoder = LIIFDecoder(cfg.decoder)

    def forward(
        self,
        img: torch.Tensor,
        coords: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Segment ``img`` at arbitrary query coords/resolution (defaults to the input grid)."""
        raise NotImplementedError("INRUNet.forward is not implemented yet.")


@MODELS.register("unet_baseline")
class BaselineUNet(nn.Module):
    """Plain fixed-resolution UNet (encoder + conv head) for ablation/comparison."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = UNetEncoder(cfg.encoder)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Segment ``img`` [B, C, H, W] into logits [B, n_classes, H, W] at the input resolution."""
        raise NotImplementedError("BaselineUNet.forward is not implemented yet.")
