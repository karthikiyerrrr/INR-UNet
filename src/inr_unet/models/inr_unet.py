"""Assembled model classes combining the UNet encoder with the LIIF decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from inr_unet.models.components.liif import LIIFDecoder, make_coord
from inr_unet.models.components.unet import UNetEncoder
from inr_unet.registry import MODELS


@MODELS.register("inr_unet")
class INRUNet(nn.Module):
    """Resolution-agnostic UNet: a UNet encoder feeding a LIIF implicit decoder."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = UNetEncoder(cfg.encoder)
        self.decoder = LIIFDecoder(cfg.decoder, in_dim=int(cfg.encoder.feature_dim))

    def forward(
        self,
        img: torch.Tensor,
        coords: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Segment ``img`` at query coords ([B, Q, n_classes]) or densely ([B, n_classes, H, W])."""
        feat = self.encoder(img)
        if coords is not None:
            return self.decoder(feat, coords, cell)

        b, _, h, w = img.shape
        coords = make_coord((h, w), device=img.device, dtype=img.dtype)
        coords = coords.unsqueeze(0).expand(b, -1, -1)
        cell = torch.empty_like(coords)
        cell[:, :, 0] = 2.0 / w
        cell[:, :, 1] = 2.0 / h
        logits = self.decoder(feat, coords, cell)  # [B, H*W, n_classes]
        n_classes = logits.shape[-1]
        return logits.permute(0, 2, 1).reshape(b, n_classes, h, w)


@MODELS.register("unet_baseline")
class BaselineUNet(nn.Module):
    """Plain fixed-resolution UNet (encoder + conv head) for ablation/comparison."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = UNetEncoder(cfg.encoder)
        self.head = nn.Conv2d(int(cfg.encoder.feature_dim), int(cfg.decoder.n_classes), 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Segment ``img`` [B, C, H, W] into logits [B, n_classes, H, W] at input resolution."""
        return self.head(self.encoder(img))
