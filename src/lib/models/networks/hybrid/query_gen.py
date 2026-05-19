"""
QueryGenerator: converts a CenterNet heatmap into dynamic DETR queries.

Two selection modes controlled by QueryGenConfig.use_gumbel:

  Gumbel-Top-K (training, use_gumbel=True):
    K independent Gumbel perturbations are added to the max-class heatmap
    scores; each query selects a distinct spatial location via argmax.
    A Straight-Through Estimator (STE) makes the whole pipeline differentiable:
      Forward : hard one-hot selection per query (fast, deterministic).
      Backward: gradient flows through soft Gumbel-Softmax weights → heatmap
                scores and Stage-1 wh head (partial detach — backbone feat is
                still detached to protect representation learning).
    Memory note: the (B, K, N) weight tensor uses ~180 MB at B=4, K=200,
    N=56 320 (fp32). With AMP this halves; reduce K if memory is tight.

  Hard TopK (inference or use_gumbel=False):
    TopK over the max-class heatmap. No NMS — decoder self-attention handles
    any spatial duplicates naturally, and is faster than max-pool NMS.

Temperature τ is cosine-annealed epoch-by-epoch via set_tau():
  τ large (early): soft distribution → stable gradients, high query diversity.
  τ small (late) : sharp distribution → near-hard selection, precise seeds.
  Coordinated with the Stage-1/2 curriculum in HybridLoss for consistent pacing.

Reference points are in logit (unsigmoid) space.  The decoder (bbox_reparam=False)
applies sigmoid internally before feeding MSDeformAttn, keeping coordinates in [0,1].
"""
from __future__ import annotations

import math
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

    Gumbel path (training):
      1. Build score distribution over spatial locations (max over classes).
      2. Sample K independent Gumbel perturbations → K diverse soft selections.
      3. STE: hard argmax in forward, soft Gumbel-Softmax gradient in backward.
      4. Aggregate content / position / wh via weighted sum from soft weights.
      5. Soft-mask below-threshold query content.

    Hard path (inference):
      1. Max over classes → TopK over spatial.
      2. Gather content / position / wh at selected locations.
      3. Soft-mask below-threshold query content.
    """

    def __init__(self, hidden_dim: int, cfg: QueryGenConfig) -> None:
        super().__init__()
        self.top_k      = cfg.top_k
        self.score_thr  = cfg.score_threshold
        self.use_gumbel = cfg.use_gumbel
        self.tau_start  = cfg.tau_start
        self.tau_end    = cfg.tau_end
        self._tau       = cfg.tau_start   # updated each epoch by set_tau()

        self.content_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.content_proj.weight)
        nn.init.zeros_(self.content_proj.bias)

    # ── Tau annealing ─────────────────────────────────────────────────────────

    def set_tau(self, epoch: int, total_epochs: int) -> None:
        """
        Cosine-anneal temperature τ from tau_start → tau_end over training.
        Uses the same cosine schedule as the Stage-1/2 curriculum weight so
        both ramps are perfectly in sync: as Stage-2 weight rises, τ falls and
        query selection sharpens — Stage-2 gradient becomes increasingly focused.
        No-op when use_gumbel is False or total_epochs ≤ 0.
        """
        if not self.use_gumbel or total_epochs <= 0:
            self._tau = self.tau_start
            return
        t        = min(1.0, epoch / total_epochs)
        smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))
        # Exponential interpolation in log-space keeps τ > 0 at all times
        self._tau = self.tau_start * (self.tau_end / self.tau_start) ** smooth_t

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid_to_logit(x: Tensor) -> Tensor:
        x = x.clamp(_EPS, 1.0 - _EPS)
        return torch.log(x / (1.0 - x))

    def _gumbel_topk_weights(self, scores: Tensor, K: int) -> Tensor:
        """
        Vectorized Gumbel-Top-K with Straight-Through Estimator.

        scores : (B, N) — max-class heatmap score per spatial location.
        returns: (B, K, N) weight matrix.
          Forward : W_hard — one-hot at argmax(scores + G_k) for each query k.
          Backward: gradient flows through W_soft = softmax((scores+G)/τ).

        Gumbel-Max theorem: argmax(log p_i + G_i) ~ Categorical(p).
        K independent noise tensors G_k give K diverse non-collapsing queries
        without any explicit NMS or diversity regularization.
        """
        B, N = scores.shape
        # K independent Gumbel(0,1) samples: (B, K, N)
        # U ~ Uniform(eps, 1-eps) → G = -log(-log(U))
        U = scores.new_empty(B, K, N).uniform_().clamp_(_EPS, 1.0 - _EPS)
        G = -(-U.log()).log()

        perturbed = scores.unsqueeze(1) + G          # (B, K, N)

        # Soft weights: differentiable path for backward
        W_soft = (perturbed / self._tau).softmax(dim=-1)   # (B, K, N)

        # Hard weights: argmax per query — used in forward (STE)
        hard_idx = perturbed.argmax(dim=-1, keepdim=True)  # (B, K, 1)
        W_hard   = torch.zeros_like(W_soft).scatter_(-1, hard_idx, 1.0)

        # STE trick: W_hard in forward, W_soft gradient in backward
        #   forward : W_soft + (W_hard - W_soft)           = W_hard  ✓
        #   backward: dL/dW_soft + d(W_hard-W_soft).detach = dL/dW_soft ✓
        return W_soft + (W_hard - W_soft).detach()

    # ── Forward paths ─────────────────────────────────────────────────────────

    def _forward_gumbel(
        self,
        hm: Tensor, feat: Tensor, wh_map: Tensor | None,
        B: int, C: int, Hs: int, Ws: int, D: int, N: int, K: int,
    ) -> QueryBundle:
        """Gumbel-Top-K path (training): gradient flows to heatmap + wh head."""

        # Score distribution: max over class dimension
        # Gradient propagates back through hm.max() → heatmap → Stage-1 head
        s = hm.max(dim=1).values.reshape(B, N)          # (B, N)

        # Gumbel-Top-K selection weights (STE): (B, K, N)
        W = self._gumbel_topk_weights(s, K)

        # ── Class scores and confidence ───────────────────────────────────────
        # Weighted combination of class channels → class probability per query
        hm_spatial  = hm.reshape(B, C, N).permute(0, 2, 1)   # (B, N, C)
        class_scores = W @ hm_spatial                          # (B, K, C)
        scores  = class_scores.max(dim=-1).values              # (B, K)
        classes = class_scores.argmax(dim=-1)                  # (B, K)

        # ── Weighted positions ────────────────────────────────────────────────
        # grid is constant — no gradient needed, expand via broadcast
        gy = torch.arange(Hs, device=hm.device, dtype=hm.dtype)
        gx = torch.arange(Ws, device=hm.device, dtype=hm.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        pos = torch.stack(
            [(grid_x + 0.5) / Ws, (grid_y + 0.5) / Hs], dim=-1,
        ).reshape(1, N, 2)                                     # (1, N, 2)

        cx_cy = W @ pos.expand(B, -1, -1)                     # (B, K, 2)
        cx, cy = cx_cy[..., 0], cx_cy[..., 1]

        # ── Weighted wh ───────────────────────────────────────────────────────
        # wh_map detached: Stage-2 CIoU gradient would otherwise diffuse across
        # all N spatial locations via W, conflicting with Stage-1 SmoothL1
        # supervision which only targets GT peak positions.
        # Stage-2 → Stage-1 feedback is preserved through hm (W @ hm_spatial).
        if wh_map is not None:
            wh_spatial = wh_map.detach().reshape(B, 2, N).permute(0, 2, 1)  # (B, N, 2)
            peak_wh    = W @ wh_spatial                              # (B, K, 2)
            ref_w = (peak_wh[..., 0] / Ws).clamp(_EPS, 1.0 - _EPS)
            ref_h = (peak_wh[..., 1] / Hs).clamp(_EPS, 1.0 - _EPS)
        else:
            ref_w = cx.new_full((B, K), 0.05)
            ref_h = cx.new_full((B, K), 0.05)

        # ── Weighted content features ─────────────────────────────────────────
        # feat is detached (backbone features protected from Stage-2 gradient).
        # Gradient still flows through W → s → heatmap (the key path).
        feat_spatial = feat.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)
        content      = self.content_proj(W @ feat_spatial)      # (B, K, D)

        ref_sig   = torch.stack([cx, cy, ref_w, ref_h], dim=-1)
        ref_logit = self._sigmoid_to_logit(ref_sig)

        score_mask = (scores >= self.score_thr).float().unsqueeze(-1)
        content    = content * score_mask

        return QueryBundle(ref_points=ref_logit, content=content,
                           scores=scores, classes=classes)

    def _forward_hard(
        self,
        hm: Tensor, feat: Tensor, wh_map: Tensor | None,
        B: int, C: int, Hs: int, Ws: int, D: int, N: int, K: int,
    ) -> QueryBundle:
        """Hard TopK path (inference): fast, no NMS, no gradient needed."""

        # Max over classes → TopK over spatial locations
        hm_max_scores, hm_max_cls = hm.max(dim=1)      # (B, Hs, Ws) each
        s_flat   = hm_max_scores.reshape(B, N)          # (B, N)
        cls_flat = hm_max_cls.reshape(B, N)             # (B, N)

        scores, spatial_idx = torch.topk(s_flat, K, dim=1, sorted=True)
        classes = cls_flat.gather(1, spatial_idx)

        y_idx = (spatial_idx // Ws).float()
        x_idx = (spatial_idx %  Ws).float()
        cx = (x_idx + 0.5) / Ws
        cy = (y_idx + 0.5) / Hs

        if wh_map is not None:
            wh_flat  = wh_map.detach().reshape(B, 2, N)
            idx_exp2 = spatial_idx.unsqueeze(1).expand(B, 2, K)
            peak_wh  = wh_flat.gather(2, idx_exp2).permute(0, 2, 1)  # (B, K, 2)
            ref_w = (peak_wh[..., 0] / Ws).clamp(_EPS, 1.0 - _EPS)
            ref_h = (peak_wh[..., 1] / Hs).clamp(_EPS, 1.0 - _EPS)
        else:
            ref_w = torch.full_like(cx, 0.05)
            ref_h = torch.full_like(cy, 0.05)

        ref_sig   = torch.stack([cx, cy, ref_w, ref_h], dim=-1)
        ref_logit = self._sigmoid_to_logit(ref_sig)

        feat_flat = feat.reshape(B, D, N)
        idx_exp   = spatial_idx.unsqueeze(1).expand(B, D, K)
        peak_feat = feat_flat.gather(2, idx_exp).permute(0, 2, 1)  # (B, K, D)
        content   = self.content_proj(peak_feat)

        score_mask = (scores >= self.score_thr).float().unsqueeze(-1)
        content    = content * score_mask

        return QueryBundle(ref_points=ref_logit, content=content,
                           scores=scores, classes=classes)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hm:     Tensor,                  # (B, C, H, W)  sigmoid heatmap
        feat:   Tensor,                  # (B, D, H, W)  spatial features (backbone detached)
        wh_map: Tensor | None = None,    # (B, 2, H, W)  Stage-1 wh output (pixel units)
    ) -> QueryBundle:
        B, C, Hs, Ws = hm.shape
        D = feat.shape[1]
        N = Hs * Ws
        K = min(self.top_k, N)

        if self.use_gumbel and self.training:
            return self._forward_gumbel(hm, feat, wh_map, B, C, Hs, Ws, D, N, K)
        return self._forward_hard(hm, feat, wh_map, B, C, Hs, Ws, D, N, K)
