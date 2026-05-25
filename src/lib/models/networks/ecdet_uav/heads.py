"""
CenterNetHead  — stage-1 dense detection (hm, wh, reg)

Note: ECTransformer (decoder) uses its own internal prediction heads (dec_score_head,
dec_bbox_head). Box/cls predictions come from ECTransformer directly, not from here.
ReID embeddings are extracted by HybridECDet.reid_mlp applied to 'hs' (final decoder
hidden states) that ECTransformer exposes in its output dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import CenterNetHeadConfig


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
    reid:       Optional[Tensor]  # (B, K, reid_dim)  L2-normalised; None when reid_dim=0
    boxes_all:  Tensor            # (num_layers, B, K, 4)  all layers for auxiliary loss
    logits_all: Tensor            # (num_layers, B, K, C)  all layers for auxiliary loss
    dn_outputs:       Optional[list] = None  # list of {pred_logits, pred_boxes} per DN layer
    dn_meta:          Optional[dict] = None  # {dn_num_group, dn_positive_idx, dn_num_split}
    enc_aux_outputs:  Optional[list] = None  # [{pred_logits, pred_boxes}] encoder top-K supervision


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_conv_head(in_ch: int, out_ch: int, head_conv: int) -> nn.Module:
    if head_conv > 0:
        return nn.Sequential(
            nn.Conv2d(in_ch, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, out_ch, kernel_size=1, bias=True),
        )
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)


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
