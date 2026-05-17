"""
QueryGenerator: converts a CenterNet heatmap into dynamic DETR queries.

Pipeline:
  hm (B,C,H,W)  →  local-max NMS  →  score-threshold filter  →  top-K peaks
  wh (B,2,H,W)  →  gather at peaks  →  normalised ref box size   ─┐
                                                                    ├→ QueryBundle
  feat (B,D,H,W) → gather + project                               ─┘

Changes vs. v1:
  • Reference box size now comes from Stage-1 wh prediction instead of a
    fixed 0.05 default — gives the decoder a much better starting point,
    especially for the wide size range of VisDrone objects.
  • Soft score masking: content of below-threshold queries is zeroed out
    before entering the decoder.  K stays fixed for batching / ONNX, but
    the decoder only receives signal from confident peaks.

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
    Converts a multi-class heatmap + feature map into decoder queries.

    Steps:
      1. Local-max NMS on the heatmap to suppress non-peak responses.
      2. Zero out sub-threshold peaks (score_threshold filter).
      3. Top-K selection across all classes and spatial locations.
      4. Gather Stage-1 wh predictions at peak locations for reference size.
      5. Project feature at peak positions to initialise content queries.
      6. Soft-mask content of below-threshold queries (zeros → decoder ignores).
    """

    def __init__(self, hidden_dim: int, cfg: QueryGenConfig) -> None:
        super().__init__()
        self.top_k      = cfg.top_k
        self.nms_kernel = cfg.nms_kernel
        self.score_thr  = cfg.score_threshold

        self.content_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.content_proj.weight)
        nn.init.zeros_(self.content_proj.bias)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _local_max_nms(hm: Tensor, kernel: int) -> Tensor:
        pad    = kernel // 2
        hm_max = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=pad)
        return hm * (hm == hm_max).float()

    @staticmethod
    def _sigmoid_to_logit(x: Tensor) -> Tensor:
        x = x.clamp(_EPS, 1.0 - _EPS)
        return torch.log(x / (1.0 - x))

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hm:     Tensor,              # (B, C, H, W)  sigmoid heatmap
        feat:   Tensor,              # (B, D, H, W)  spatial feature map
        wh_map: Tensor | None = None,  # (B, 2, H, W)  Stage-1 wh output (pixel units at stride-4)
    ) -> QueryBundle:
        """
        Args:
            hm     : sigmoid heatmap from CenterNetHead
            feat   : stride-4 feature map (detached from Stage-1 grad)
            wh_map : (optional) wh head output from Stage-1 — used to initialise
                     reference box sizes. If None, falls back to fixed 0.05.
        """
        B, C, H, W = hm.shape
        D = feat.shape[1]

        # ── Step 1: local-max NMS ─────────────────────────────────────────────
        hm_peaks = self._local_max_nms(hm, self.nms_kernel)

        # ── Step 2: suppress sub-threshold peaks before top-K ─────────────────
        # Zeroing (not filtering) keeps K fixed for batching and ONNX export.
        hm_peaks = hm_peaks * (hm_peaks >= self.score_thr).float()

        # ── Step 3: top-K over all classes × spatial ──────────────────────────
        hm_flat   = hm_peaks.reshape(B, -1)                         # (B, C*H*W)
        k         = min(self.top_k, hm_flat.shape[1])
        scores, flat_idx = torch.topk(hm_flat, k, dim=1, sorted=True)

        classes     = flat_idx // (H * W)
        spatial_idx = flat_idx %  (H * W)

        y_idx = (spatial_idx // W).float()
        x_idx = (spatial_idx %  W).float()

        cx = (x_idx + 0.5) / W
        cy = (y_idx + 0.5) / H

        # ── Step 4: reference box size from Stage-1 wh ────────────────────────
        if wh_map is not None:
            # wh_map: (B, 2, H, W) in stride-4 pixel units.
            # Normalise to [0,1] w.r.t. the original image (H_orig=H*4, W_orig=W*4):
            #   w_norm = w_stride4 / W_hm  (= w_orig / W_orig)
            wh_flat  = wh_map.detach().reshape(B, 2, H * W)          # (B, 2, H*W)
            idx_exp2 = spatial_idx.unsqueeze(1).expand(B, 2, k)      # (B, 2, K)
            peak_wh  = wh_flat.gather(2, idx_exp2).permute(0, 2, 1)  # (B, K, 2)

            ref_w = (peak_wh[..., 0] / W).clamp(_EPS, 1.0 - _EPS)   # (B, K)
            ref_h = (peak_wh[..., 1] / H).clamp(_EPS, 1.0 - _EPS)
        else:
            ref_w = torch.full_like(cx, 0.05)
            ref_h = torch.full_like(cy, 0.05)

        ref_sig   = torch.stack([cx, cy, ref_w, ref_h], dim=-1)      # (B, K, 4) in [0,1]
        ref_logit = self._sigmoid_to_logit(ref_sig)                   # (B, K, 4) unsigmoid

        # ── Step 5: gather & project content features ─────────────────────────
        feat_flat = feat.reshape(B, D, H * W)
        idx_exp   = spatial_idx.unsqueeze(1).expand(B, D, k)
        peak_feat = feat_flat.gather(2, idx_exp).permute(0, 2, 1)    # (B, K, D)
        content   = self.content_proj(peak_feat)                      # (B, K, D)

        # ── Step 6: soft-mask below-threshold queries ─────────────────────────
        # Queries with score < score_thr have zeroed content — the decoder
        # receives no signal from them without changing K or breaking batching.
        score_mask = (scores >= self.score_thr).float().unsqueeze(-1)  # (B, K, 1)
        content    = content * score_mask

        return QueryBundle(
            ref_points=ref_logit,
            content=content,
            scores=scores,
            classes=classes,
        )
