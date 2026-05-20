"""
TokenScorer  — stride-8 objectness scorer with multi-scale s16+s8 fusion
DETRHead     — stage-2 per-query prediction (box, class, reid)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TokenScorerConfig, DETRHeadConfig


# ── Output structs ─────────────────────────────────────────────────────────────

@dataclass
class ScorerOutput:
    score_map: Tensor   # (B, 1, H/8, W/8)  sigmoid objectness
    feat_s8:   Tensor   # (B, D, H/8, W/8)  stride-8 features for QueryGenerator


@dataclass
class DETROutput:
    boxes:      Tensor   # (B, K, 4)  cxcywh in [0,1] — final decoder layer
    logits:     Tensor   # (B, K, C)  raw class logits — final decoder layer
    reid:       Tensor   # (B, K, reid_dim)  L2-normalised embeddings
    boxes_all:  Tensor   # (num_layers, B, K, 4)  all layers for auxiliary loss
    logits_all: Tensor   # (num_layers, B, K, C)  all layers for auxiliary loss


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list = []
    for i in range(num_layers):
        d_in  = in_dim    if i == 0             else hidden_dim
        d_out = out_dim   if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ── TokenScorer ────────────────────────────────────────────────────────────────

class TokenScorer(nn.Module):
    """
    Lightweight objectness scorer at stride-8.

    Multi-scale fusion (tiny-object rescue):
      Tiny objects (<10px) may occupy only ~0.5 cells at stride-16.  We fuse a
      parallel s16-branch score into the s8 score map BEFORE sigmoid so that even
      the smallest objects leave a detectable peak:

        logit_fused = logit_s8(feat_s8) + bilinear_upsample(logit_s16(feat_s16))
        score_map   = sigmoid(logit_fused)

    Both branches share the same bias init (sigmoid(-4.595) ≈ 0.01) so the
    network starts with near-zero scores everywhere — identical to CenterNet
    focal-loss stability convention.
    """

    def __init__(self, hidden_dim: int, cfg: TokenScorerConfig) -> None:
        super().__init__()
        # stride-16 → stride-8  (single ConvTranspose, lighter than the old 2-step s16→s4)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2,
                               padding=0, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # s8 objectness head
        h = cfg.head_conv
        self.scorer_s8 = nn.Sequential(
            nn.Conv2d(hidden_dim, h, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.scorer_s8[-1].bias, -4.595)

        # s16 branch for multi-scale fusion
        self.use_multiscale_fusion = cfg.use_multiscale_fusion
        if cfg.use_multiscale_fusion:
            self.scorer_s16 = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
            nn.init.constant_(self.scorer_s16.bias, -4.595)

    def forward(self, finest_s16: Tensor) -> ScorerOutput:
        """
        finest_s16 : (B, D, H/16, W/16)
        Returns ScorerOutput with score_map (B,1,H/8,W/8) and feat_s8 (B,D,H/8,W/8).
        """
        feat_s8  = self.upsample(finest_s16)         # (B, D, H/8, W/8)
        logit_s8 = self.scorer_s8(feat_s8)           # (B, 1, H/8, W/8)

        if self.use_multiscale_fusion:
            logit_s16    = self.scorer_s16(finest_s16)  # (B, 1, H/16, W/16)
            logit_s16_up = F.interpolate(
                logit_s16, scale_factor=2,
                mode='bilinear', align_corners=False,
            )                                            # (B, 1, H/8, W/8)
            score_map = (logit_s8 + logit_s16_up).sigmoid()
        else:
            score_map = logit_s8.sigmoid()

        return ScorerOutput(score_map=score_map, feat_s8=feat_s8)


# ── DETRHead ───────────────────────────────────────────────────────────────────

class DETRHead(nn.Module):
    """
    Stage-2 per-query prediction head applied to decoder hidden states.

    Input : hs        (num_layers, B, K, D) — stacked decoder outputs
            refs_logit (num_layers, B, K, 4) — refined ref points, unsigmoid cxcywh
    Output: DETROutput
    """

    def __init__(self, cfg: DETRHeadConfig, bbox_reparam: bool = True) -> None:
        super().__init__()
        D = cfg.hidden_dim
        self.bbox_reparam = bbox_reparam

        self.box_mlp  = _make_mlp(D, D, 4, 3)
        self.cls_head = nn.Linear(D, cfg.num_classes)
        self.reid_mlp = _make_mlp(D, D, cfg.reid_dim, 2)

        # Varifocal bias init: sigmoid(-2.0) ≈ 0.12 prevents zero-gradient cold-start
        nn.init.constant_(self.cls_head.bias, -2.0)

        # Zero-init last box layer (standard DETR practice)
        last_box: nn.Linear = list(self.box_mlp.children())[-1]
        nn.init.zeros_(last_box.weight)
        nn.init.zeros_(last_box.bias)

    def forward(self, hs: Tensor, refs_logit: Tensor) -> DETROutput:
        """
        hs         : (num_layers, B, K, D)
        refs_logit : (num_layers, B, K, 4)  unsigmoid cxcywh
        """
        delta = self.box_mlp(hs)   # (L, B, K, 4)

        if self.bbox_reparam:
            cx_cy     = delta[..., :2] * refs_logit[..., 2:] + refs_logit[..., :2]
            wh        = delta[..., 2:].exp() * refs_logit[..., 2:]
            boxes_all = torch.cat([cx_cy, wh], dim=-1).sigmoid()
        else:
            boxes_all = (refs_logit + delta).sigmoid()

        logits_all = self.cls_head(hs)
        reid       = F.normalize(self.reid_mlp(hs[-1]), dim=-1)   # (B, K, reid_dim)

        return DETROutput(
            boxes      = boxes_all[-1],
            logits     = logits_all[-1],
            reid       = reid,
            boxes_all  = boxes_all,
            logits_all = logits_all,
        )
