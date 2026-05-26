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
import torch.nn.functional as F
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
    dn_outputs:       Optional[list]           = None  # list of {pred_logits, pred_boxes} per DN layer
    dn_meta:          Optional[dict]           = None  # {dn_num_group, dn_positive_idx, dn_num_split}
    enc_aux_outputs:  Optional[list]           = None  # [{pred_logits, pred_boxes}] encoder top-K supervision
    heatmap_out:      Optional[CenterNetOutput] = None  # hm/wh/reg from CenterNetHead on S4


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
            wh  = F.relu(self.wh(feat)),  # targets are always positive pixel-scale values
            reg = self.reg(feat),
        )


def extract_peaks_as_ref_points(
    hm_out: CenterNetOutput,
    topk: int,
    stride: int = 4,
) -> Tensor:
    """
    Extract top-K heatmap peaks and convert to ECTransformer reference points.

    Peak NMS: 3×3 max-pool keeps only local maxima.
    Positions are returned in inverse-sigmoid space (logit) matching the
    convention used by ECTransformer._get_decoder_input().

    Args:
        hm_out: CenterNetOutput with hm (B,C,H,W) sigmoid, wh (B,2,H,W) in
                feature-map pixel scale, reg (B,2,H,W) sub-pixel offset.
        topk:   Number of peaks to return (= num_queries).
        stride: Feature map stride relative to input image (default 4 for S4).

    Returns:
        Tensor (B, topk, 4) in inverse-sigmoid space (cx, cy, w, h).
    """
    hm, wh, reg = hm_out.hm, hm_out.wh, hm_out.reg
    B, C, H, W = hm.shape

    # NMS: keep only local maxima (equal to 3×3 neighbourhood max)
    hm_max = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    hm_nms = (hm_max == hm).float() * hm

    # class-agnostic score: take max across classes
    scores, _ = hm_nms.max(dim=1)          # (B, H, W)
    scores_flat = scores.flatten(1)         # (B, H*W)

    K = min(topk, H * W)
    _, topk_idx = scores_flat.topk(K, dim=1, sorted=False)  # (B, K)

    topk_y = (topk_idx // W).float()   # row in feature map
    topk_x = (topk_idx  % W).float()   # col in feature map

    def _gather2(feat2d: Tensor) -> Tensor:
        """Gather (B,2,H,W) at topk positions → (B,K,2)."""
        flat = feat2d.permute(0, 2, 3, 1).reshape(B, H * W, 2)
        return flat.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))

    topk_reg = _gather2(reg)   # (B, K, 2)  sub-pixel offset
    topk_wh  = _gather2(wh)    # (B, K, 2)  feature-map pixel scale

    # Recover normalised center: cx = (ix + reg_x) / W_feat
    cx = ((topk_x + topk_reg[..., 0]) / W).clamp(1e-4, 1 - 1e-4)
    cy = ((topk_y + topk_reg[..., 1]) / H).clamp(1e-4, 1 - 1e-4)

    # Recover normalised wh: wh_pred is in feature-map pixel scale
    bw = (topk_wh[..., 0] / W).clamp(1e-4, 1.0)
    bh = (topk_wh[..., 1] / H).clamp(1e-4, 1.0)

    ref = torch.stack([cx, cy, bw, bh], dim=-1)  # (B, K, 4) in (0, 1)

    # Pad to exactly `topk` if fewer peaks exist
    if K < topk:
        pad = ref[:, :1, :].expand(-1, topk - K, -1)
        ref = torch.cat([ref, pad], dim=1)

    # Inverse sigmoid → logit space (ECTransformer convention)
    return torch.log(ref / (1.0 - ref + 1e-6))
