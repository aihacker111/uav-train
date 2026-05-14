"""
QueryGenerator: converts a CenterNet heatmap into dynamic DETR queries.

Pipeline:
  hm (B,C,H,W)  →  local-max NMS  →  top-K peaks  →  QueryBundle
                                                         ├── ref_points  (B,K,4) unsigmoid cxcywh
                                                         ├── content     (B,K,D) projected feature
                                                         ├── scores      (B,K)
                                                         └── classes     (B,K)

Reference points are in logit (unsigmoid) space.  The decoder (bbox_reparam=False)
applies sigmoid internally before feeding MSDeformAttn, keeping coordinates in [0,1].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import QueryGenConfig

_EPS = 1e-6


# ── Output struct ─────────────────────────────────────────────────────────────

@dataclass
class QueryBundle:
    ref_points: Tensor   # (B, K, 4)  unsigmoid cxcywh — reference points for decoder
    content:    Tensor   # (B, K, D)  content queries for decoder
    scores:     Tensor   # (B, K)     heatmap confidence
    classes:    Tensor   # (B, K)     predicted class index


# ── QueryGenerator ────────────────────────────────────────────────────────────

class QueryGenerator(nn.Module):
    """
    Converts a multi-class heatmap + a feature map into decoder queries.

    The generator performs three steps:
      1. Local-max NMS on the heatmap to suppress non-peak responses.
      2. Top-K selection across all classes and spatial locations.
      3. Feature projection at peak positions to initialise content queries.
    """

    def __init__(self, hidden_dim: int, cfg: QueryGenConfig) -> None:
        super().__init__()
        self.top_k      = cfg.top_k
        self.nms_kernel = cfg.nms_kernel
        self.score_thr  = cfg.score_threshold

        # Learnable linear to project gathered spatial features → query content
        self.content_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.content_proj.weight)
        nn.init.zeros_(self.content_proj.bias)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _local_max_nms(hm: Tensor, kernel: int) -> Tensor:
        """Retain only local maxima; set all non-peak values to zero."""
        pad   = kernel // 2
        hm_max = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=pad)
        return hm * (hm == hm_max).float()

    @staticmethod
    def _sigmoid_to_logit(x: Tensor) -> Tensor:
        """Safe inverse-sigmoid (logit), clamped to avoid inf."""
        x = x.clamp(_EPS, 1.0 - _EPS)
        return torch.log(x / (1.0 - x))

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, hm: Tensor, feat: Tensor) -> QueryBundle:
        """
        Args:
            hm   : (B, num_classes, H, W)  sigmoid heatmap from CenterNetHead
            feat : (B, D, H, W)            spatial feature map (neck output)
        Returns:
            QueryBundle with K = self.top_k candidates per image
        """
        B, C, H, W = hm.shape
        D = feat.shape[1]

        # ── Step 1: local-max NMS ─────────────────────────────────────────────
        hm_peaks = self._local_max_nms(hm, self.nms_kernel)    # (B, C, H, W)

        # ── Step 2: top-K over all classes × spatial ──────────────────────────
        hm_flat   = hm_peaks.view(B, -1)                        # (B, C*H*W)
        k         = min(self.top_k, hm_flat.shape[1])
        scores, flat_idx = torch.topk(hm_flat, k, dim=1, sorted=True)  # (B, K)

        classes     = flat_idx // (H * W)                       # (B, K) — class id
        spatial_idx = flat_idx %  (H * W)                       # (B, K) — H*W index

        y_idx = (spatial_idx // W).float()                      # (B, K)
        x_idx = (spatial_idx %  W).float()                      # (B, K)

        # Normalised centre coordinates in [0, 1]
        cx = (x_idx + 0.5) / W
        cy = (y_idx + 0.5) / H

        # Default box size: small initial box; decoder will refine
        default_w = torch.full_like(cx, 0.05)
        default_h = torch.full_like(cy, 0.05)

        ref_sig  = torch.stack([cx, cy, default_w, default_h], dim=-1)  # (B, K, 4) in [0,1]
        ref_logit = self._sigmoid_to_logit(ref_sig)                      # (B, K, 4) unsigmoid

        # ── Step 3: gather & project content features ─────────────────────────
        #  feat: (B, D, H, W) → (B, D, H*W)
        feat_flat = feat.view(B, D, H * W)

        #  spatial_idx: (B, K) → expand to (B, D, K) for gather
        idx_exp   = spatial_idx.unsqueeze(1).expand(B, D, k)   # (B, D, K)
        peak_feat = feat_flat.gather(2, idx_exp)                # (B, D, K)
        peak_feat = peak_feat.permute(0, 2, 1)                  # (B, K, D)

        content = self.content_proj(peak_feat)                  # (B, K, D)

        return QueryBundle(
            ref_points=ref_logit,
            content=content,
            scores=scores,
            classes=classes,
        )
