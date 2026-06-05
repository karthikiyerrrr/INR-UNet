"""LIIF continuous local implicit representation: decoder + coordinate utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from inr_unet.registry import DECODERS


def make_coord(
    shape: tuple[int, int],
    ranges: tuple[tuple[float, float], ...] | None = None,
    flatten: bool = True,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cell-center coordinates in [-1, 1] for an H x W grid, last dim ordered (x, y).

    ``shape`` is (H, W). Row-major flatten yields index = y * W + x, matching the dense
    reshape used by INRUNet.
    """
    seqs = []
    for i, n in enumerate(shape):
        v0, v1 = (-1.0, 1.0) if ranges is None else ranges[i]
        r = (v1 - v0) / (2 * n)
        seqs.append(v0 + r + (2 * r) * torch.arange(n, device=device, dtype=dtype))
    grid = torch.stack(torch.meshgrid(*seqs, indexing="ij"), dim=-1)  # [H, W, 2] = (y, x)
    grid = grid.flip(-1)  # -> (x, y)
    if flatten:
        grid = grid.view(-1, grid.shape[-1])
    return grid


class _MLP(nn.Module):
    """Plain ReLU MLP applied to the last dimension."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int, out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        out = self.net(x.reshape(-1, x.shape[-1]))
        return out.view(*shape, -1)


@DECODERS.register("liif")
class LIIFDecoder(nn.Module):
    """Local implicit MLP decoder that queries a feature map at continuous coordinates.

    Coordinates and cells are expected in (x, y) order (x indexes width). ``in_dim`` is the
    encoder feature dimension; it is supplied by INRUNet (or via ``cfg.in_dim``).
    """

    def __init__(self, cfg: DictConfig, in_dim: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        if in_dim is None:
            in_dim = int(cfg["in_dim"]) if "in_dim" in cfg else None
        if in_dim is None:
            raise ValueError("LIIFDecoder requires in_dim (the encoder feature_dim)")
        self.in_dim = int(in_dim)
        self.local_ensemble = bool(cfg.local_ensemble)
        self.feature_unfold = bool(cfg.feature_unfold)
        self.cell_decode = bool(cfg.cell_decode)
        self.n_classes = int(cfg.n_classes)

        imnet_in = self.in_dim * (9 if self.feature_unfold else 1)
        imnet_in += 2  # relative coordinate
        if self.cell_decode:
            imnet_in += 2
        self.imnet = _MLP(imnet_in, int(cfg.hidden_dim), int(cfg.num_layers), self.n_classes)

    def forward(
        self, feat: torch.Tensor, coords: torch.Tensor, cell: torch.Tensor
    ) -> torch.Tensor:
        """Map feat [B, D, H, W] + coords [B, Q, 2] + cell [B, Q, 2] to logits [B, Q, n_classes]."""
        if self.feature_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
            )
        b, _, hh, ww = feat.shape

        feat_coord = (
            make_coord((hh, ww), flatten=False, device=feat.device, dtype=feat.dtype)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(b, 2, hh, ww)
        )  # [B, 2, H, W] = (x, y)

        if self.local_ensemble:
            vx_lst, vy_lst, eps = [-1, 1], [-1, 1], 1e-6
        else:
            vx_lst, vy_lst, eps = [0], [0], 0.0

        half_x = 1.0 / ww
        half_y = 1.0 / hh

        preds: list[torch.Tensor] = []
        areas: list[torch.Tensor] = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coords.clone()
                coord_[:, :, 0] += vx * half_x + eps
                coord_[:, :, 1] += vy * half_y + eps
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                grid = coord_.unsqueeze(2)  # [B, Q, 1, 2], last dim already (x, y)
                q_feat = F.grid_sample(
                    feat, grid, mode="nearest", align_corners=False, padding_mode="border"
                )[:, :, :, 0].permute(0, 2, 1)  # [B, Q, D]
                q_coord = F.grid_sample(
                    feat_coord, grid, mode="nearest", align_corners=False, padding_mode="border"
                )[:, :, :, 0].permute(0, 2, 1)  # [B, Q, 2] (x, y)

                rel = coords - q_coord
                rel[:, :, 0] *= ww
                rel[:, :, 1] *= hh
                inp = [q_feat, rel]
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= ww
                    rel_cell[:, :, 1] *= hh
                    inp.append(rel_cell)
                preds.append(self.imnet(torch.cat(inp, dim=-1)))
                areas.append(torch.abs(rel[:, :, 0] * rel[:, :, 1]) + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]
        out = torch.zeros_like(preds[0])
        for pred, area in zip(preds, areas, strict=True):
            out = out + pred * (area / tot_area).unsqueeze(-1)
        return out
