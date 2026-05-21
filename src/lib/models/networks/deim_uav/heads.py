"""
CenterNetHead  — stage-1 dense detection (hm, wh, reg)
DETRHead       — stage-2 per-query prediction (box, class, reid)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CenterNetHeadConfig, DETRHeadConfig


# ── Output structs ─────────────────────────────────────────────────────────────

@dataclass
class CenterNetOutput:
    hm:  Tensor   # (B, C, H, W)  sigmoid heatmap
    wh:  Tensor   # (B, 2, H, W)  pixel-scale width/height
    reg: Tensor   # (B, 2, H, W)  sub-pixel offset


@dataclass
class DETROutput:
    boxes:      Tensor            # (B, K, 4)  cxcywh in [0,1] — final decoder layer
    logits:     Tensor            # (B, K, C)  raw class logits — final decoder layer
    reid:       Optional[Tensor]  # (B, K, reid_dim)  L2-normalised; None when no ReID head
    boxes_all:  Tensor            # (num_layers, B, K, 4)  all layers for auxiliary loss
    logits_all: Tensor            # (num_layers, B, K, C)  all layers for auxiliary loss


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_conv_head(in_ch: int, out_ch: int, head_conv: int) -> nn.Module:
    if head_conv > 0:
        return nn.Sequential(
            nn.Conv2d(in_ch, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, out_ch, kernel_size=1, bias=True),
        )
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list = []
    for i in range(num_layers):
        d_in  = in_dim    if i == 0              else hidden_dim
        d_out = out_dim   if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ── CenterNetHead ──────────────────────────────────────────────────────────────

class CenterNetHead(nn.Module):
    def __init__(self, in_ch: int, cfg: CenterNetHeadConfig) -> None:
        super().__init__()
        self.hm  = _make_conv_head(in_ch, cfg.num_classes, cfg.head_conv)
        self.wh  = _make_conv_head(in_ch, 2,               cfg.head_conv)
        self.reg = _make_conv_head(in_ch, 2,               cfg.head_conv)

        last_hm = self.hm[-1] if isinstance(self.hm, nn.Sequential) else self.hm
        nn.init.constant_(last_hm.bias, -4.595)

    def forward(self, feat: Tensor) -> CenterNetOutput:
        return CenterNetOutput(
            hm  = self.hm(feat).sigmoid(),
            wh  = self.wh(feat),
            reg = self.reg(feat),
        )


# ── DETRHead ───────────────────────────────────────────────────────────────────

class DETRHead(nn.Module):
    def __init__(self, cfg: DETRHeadConfig, bbox_reparam: bool = True) -> None:
        super().__init__()
        D = cfg.hidden_dim
        self.bbox_reparam = bbox_reparam
        self.box_mlp  = _make_mlp(D, D, 4, 3)
        self.cls_head = nn.Linear(D, cfg.num_classes)
        self.reid_mlp = _make_mlp(D, D, cfg.reid_dim, 2)

        nn.init.constant_(self.cls_head.bias, -2.0)
        last_box: nn.Linear = list(self.box_mlp.children())[-1]
        nn.init.zeros_(last_box.weight)
        nn.init.zeros_(last_box.bias)

    def forward(self, hs: Tensor, refs_logit: Tensor) -> DETROutput:
        delta = self.box_mlp(hs)
        if self.bbox_reparam:
            cx_cy     = delta[..., :2] * refs_logit[..., 2:] + refs_logit[..., :2]
            wh        = delta[..., 2:].exp() * refs_logit[..., 2:]
            boxes_all = torch.cat([cx_cy, wh], dim=-1).sigmoid()
        else:
            boxes_all = (refs_logit + delta).sigmoid()
        logits_all = self.cls_head(hs)
        reid       = F.normalize(self.reid_mlp(hs[-1]), dim=-1)
        return DETROutput(
            boxes      = boxes_all[-1],
            logits     = logits_all[-1],
            reid       = reid,
            boxes_all  = boxes_all,
            logits_all = logits_all,
        )
