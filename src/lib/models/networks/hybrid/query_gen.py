"""
QueryGenerator: converts a TokenScorer score map into dynamic DETR queries.

Input:
  hm   : (B, 1, H, W)  sigmoid objectness score map from TokenScorer
  feat : (B, D, H, W)  stride-8 spatial features (from TokenScorer.feat_s8)

Two selection modes controlled by QueryGenConfig.use_gumbel:

  Gumbel-Top-K (training, use_gumbel=True):
    K independent Gumbel perturbations are added to the score map; each query
    selects a distinct spatial location via argmax (STE for gradients).
    Forward : hard one-hot selection per query.
    Backward: gradient flows through soft Gumbel-Softmax → score map.

  Hard TopK (inference or use_gumbel=False):
    TopK over the score map. No NMS — decoder self-attention handles duplicates.

Temperature τ is cosine-annealed epoch-by-epoch via set_tau().

Reference points are in logit (unsigmoid) space; width/height default to 0.05
(decoder refines actual box dimensions via MSDeformAttn cross-attention).
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
    scores:     Tensor   # (B, K)     objectness confidence
    classes:    Tensor   # (B, K)     predicted class index (always 0 for single-ch scorer)


# ── QueryGenerator ────────────────────────────────────────────────────────────

class QueryGenerator(nn.Module):
    """
    Converts a score map + feature map into decoder queries.

    Gumbel path (training):
      1. Score distribution over spatial locations.
      2. K independent Gumbel perturbations → K diverse soft selections (STE).
      3. Aggregate content features via weighted sum.
      4. Soft-mask below-threshold query content.

    Hard path (inference):
      1. TopK over score map.
      2. Gather content features at selected locations.
      3. Soft-mask below-threshold query content.

    wh reference: always defaulted to 0.05 — box size is refined by the decoder.
    """

    def __init__(self, hidden_dim: int, cfg: QueryGenConfig) -> None:
        super().__init__()
        self.top_k      = cfg.top_k
        self.score_thr  = cfg.score_threshold
        self.use_gumbel = cfg.use_gumbel
        self.tau_start  = cfg.tau_start
        self.tau_end    = cfg.tau_end
        self._tau       = cfg.tau_start

        self.use_spatial_partition  = cfg.use_spatial_partition
        self.sp_grid_rows           = cfg.sp_grid_rows
        self.sp_grid_cols           = cfg.sp_grid_cols
        self.sp_queries_per_region  = cfg.sp_queries_per_region
        self.sp_overlap_ratio       = cfg.sp_overlap_ratio
        self.sp_global_queries      = cfg.sp_global_queries

        self.content_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.content_proj.weight)
        nn.init.zeros_(self.content_proj.bias)

    # ── Tau annealing ─────────────────────────────────────────────────────────

    def set_tau(self, epoch: int, total_epochs: int) -> None:
        if not self.use_gumbel or total_epochs <= 0:
            self._tau = self.tau_start
            return
        t        = min(1.0, epoch / total_epochs)
        smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))
        self._tau = self.tau_start * (self.tau_end / self.tau_start) ** smooth_t

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid_to_logit(x: Tensor) -> Tensor:
        x = x.clamp(_EPS, 1.0 - _EPS)
        return torch.log(x / (1.0 - x))

    def _gumbel_topk_weights(self, scores: Tensor, K: int) -> Tensor:
        """
        Vectorized Gumbel-Top-K with STE.
        scores : (B, N) — score per spatial location.
        returns: (B, K, N) weight matrix (hard in forward, soft gradient in backward).
        """
        B, N = scores.shape
        U = scores.new_empty(B, K, N).uniform_().clamp_(_EPS, 1.0 - _EPS)
        G = -(-U.log()).log()
        perturbed = scores.unsqueeze(1) + G          # (B, K, N)
        W_soft = (perturbed / self._tau).softmax(dim=-1)
        hard_idx = perturbed.argmax(dim=-1, keepdim=True)
        W_hard   = torch.zeros_like(W_soft).scatter_(-1, hard_idx, 1.0)
        return W_soft + (W_hard - W_soft).detach()

    # ── Plain forward paths ───────────────────────────────────────────────────

    def _forward_gumbel(
        self,
        hm: Tensor, feat: Tensor,
        B: int, C: int, Hs: int, Ws: int, D: int, N: int, K: int,
    ) -> QueryBundle:
        s = hm.max(dim=1).values.reshape(B, N)       # (B, N)
        W = self._gumbel_topk_weights(s, K)           # (B, K, N)

        hm_spatial   = hm.reshape(B, C, N).permute(0, 2, 1)
        class_scores = W @ hm_spatial                 # (B, K, C)
        scores  = class_scores.max(dim=-1).values
        classes = class_scores.argmax(dim=-1)

        gy = torch.arange(Hs, device=hm.device, dtype=hm.dtype)
        gx = torch.arange(Ws, device=hm.device, dtype=hm.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        pos = torch.stack(
            [(grid_x + 0.5) / Ws, (grid_y + 0.5) / Hs], dim=-1,
        ).reshape(1, N, 2)
        cx_cy = W @ pos.expand(B, -1, -1)             # (B, K, 2)
        cx, cy = cx_cy[..., 0], cx_cy[..., 1]

        ref_w = cx.new_full((B, K), 0.05)
        ref_h = cx.new_full((B, K), 0.05)

        feat_spatial = feat.reshape(B, D, N).permute(0, 2, 1)
        content      = self.content_proj(W @ feat_spatial)

        ref_sig   = torch.stack([cx, cy, ref_w, ref_h], dim=-1)
        ref_logit = self._sigmoid_to_logit(ref_sig)
        score_mask = (scores >= self.score_thr).float().unsqueeze(-1)
        content    = content * score_mask

        return QueryBundle(ref_points=ref_logit, content=content,
                           scores=scores, classes=classes)

    def _forward_hard(
        self,
        hm: Tensor, feat: Tensor,
        B: int, C: int, Hs: int, Ws: int, D: int, N: int, K: int,
    ) -> QueryBundle:
        hm_max_scores, hm_max_cls = hm.max(dim=1)
        s_flat   = hm_max_scores.reshape(B, N)
        cls_flat = hm_max_cls.reshape(B, N)

        scores, spatial_idx = torch.topk(s_flat, K, dim=1, sorted=True)
        classes = cls_flat.gather(1, spatial_idx)

        y_idx = (spatial_idx // Ws).float()
        x_idx = (spatial_idx %  Ws).float()
        cx = (x_idx + 0.5) / Ws
        cy = (y_idx + 0.5) / Hs

        ref_w = torch.full_like(cx, 0.05)
        ref_h = torch.full_like(cy, 0.05)

        ref_sig   = torch.stack([cx, cy, ref_w, ref_h], dim=-1)
        ref_logit = self._sigmoid_to_logit(ref_sig)

        feat_flat = feat.reshape(B, D, N)
        idx_exp   = spatial_idx.unsqueeze(1).expand(B, D, K)
        peak_feat = feat_flat.gather(2, idx_exp).permute(0, 2, 1)
        content   = self.content_proj(peak_feat)

        score_mask = (scores >= self.score_thr).float().unsqueeze(-1)
        content    = content * score_mask

        return QueryBundle(ref_points=ref_logit, content=content,
                           scores=scores, classes=classes)

    # ── Spatial-partitioned paths ─────────────────────────────────────────────

    def _forward_gumbel_partitioned(
        self,
        hm: Tensor, feat: Tensor,
        B: int, C: int, Hs: int, Ws: int, D: int, N: int,
    ) -> QueryBundle:
        R     = self.sp_grid_rows
        Cols  = self.sp_grid_cols
        K_loc = self.sp_queries_per_region
        K_glb = self.sp_global_queries
        ovlp  = self.sp_overlap_ratio

        all_ref, all_content, all_scores, all_classes = [], [], [], []

        for r in range(R):
            for c in range(Cols):
                h0 = max(0.0, r / R         - ovlp / 2)
                h1 = min(1.0, (r + 1) / R   + ovlp / 2)
                w0 = max(0.0, c / Cols       - ovlp / 2)
                w1 = min(1.0, (c + 1) / Cols + ovlp / 2)

                rh0 = int(h0 * Hs)
                rh1 = min(Hs, max(rh0 + 1, int(math.ceil(h1 * Hs))))
                rw0 = int(w0 * Ws)
                rw1 = min(Ws, max(rw0 + 1, int(math.ceil(w1 * Ws))))
                rHs, rWs = rh1 - rh0, rw1 - rw0
                rN  = rHs * rWs
                K_r = min(K_loc, rN)

                hm_r   = hm[:, :, rh0:rh1, rw0:rw1]
                feat_r = feat[:, :, rh0:rh1, rw0:rw1]

                s_r  = hm_r.max(dim=1).values.reshape(B, rN)
                W_r  = self._gumbel_topk_weights(s_r, K_r)

                hm_r_sp      = hm_r.reshape(B, C, rN).permute(0, 2, 1)
                cls_scores_r = W_r @ hm_r_sp
                scores_r  = cls_scores_r.max(dim=-1).values
                classes_r = cls_scores_r.argmax(dim=-1)

                gy_r = torch.arange(rHs, device=hm.device, dtype=hm.dtype)
                gx_r = torch.arange(rWs, device=hm.device, dtype=hm.dtype)
                grid_y_r, grid_x_r = torch.meshgrid(gy_r, gx_r, indexing='ij')
                gx_g = ((rw0 + grid_x_r + 0.5) / Ws).reshape(1, rN)
                gy_g = ((rh0 + grid_y_r + 0.5) / Hs).reshape(1, rN)
                pos_r = torch.stack([gx_g, gy_g], dim=-1).expand(B, -1, -1)

                cxcy_r     = W_r @ pos_r
                cx_r, cy_r = cxcy_r[..., 0], cxcy_r[..., 1]

                ref_w_r = cx_r.new_full((B, K_r), 0.05)
                ref_h_r = cx_r.new_full((B, K_r), 0.05)

                feat_r_sp = feat_r.reshape(B, D, rN).permute(0, 2, 1)
                content_r = self.content_proj(W_r @ feat_r_sp)

                ref_logit_r  = self._sigmoid_to_logit(
                    torch.stack([cx_r, cy_r, ref_w_r, ref_h_r], dim=-1))
                score_mask_r = (scores_r >= self.score_thr).float().unsqueeze(-1)
                content_r    = content_r * score_mask_r

                all_ref.append(ref_logit_r)
                all_content.append(content_r)
                all_scores.append(scores_r)
                all_classes.append(classes_r)

        if K_glb > 0:
            K_g = min(K_glb, N)
            s_g  = hm.max(dim=1).values.reshape(B, N)
            W_g  = self._gumbel_topk_weights(s_g, K_g)

            hm_sp_g      = hm.reshape(B, C, N).permute(0, 2, 1)
            cls_scores_g = W_g @ hm_sp_g
            scores_g  = cls_scores_g.max(dim=-1).values
            classes_g = cls_scores_g.argmax(dim=-1)

            gy = torch.arange(Hs, device=hm.device, dtype=hm.dtype)
            gx = torch.arange(Ws, device=hm.device, dtype=hm.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
            pos_g  = torch.stack([(grid_x + 0.5) / Ws, (grid_y + 0.5) / Hs], dim=-1).reshape(1, N, 2)
            cxcy_g = W_g @ pos_g.expand(B, -1, -1)
            cx_g, cy_g = cxcy_g[..., 0], cxcy_g[..., 1]

            ref_w_g = cx_g.new_full((B, K_g), 0.05)
            ref_h_g = cx_g.new_full((B, K_g), 0.05)

            feat_sp_g = feat.reshape(B, D, N).permute(0, 2, 1)
            content_g = self.content_proj(W_g @ feat_sp_g)

            ref_logit_g  = self._sigmoid_to_logit(
                torch.stack([cx_g, cy_g, ref_w_g, ref_h_g], dim=-1))
            score_mask_g = (scores_g >= self.score_thr).float().unsqueeze(-1)
            content_g    = content_g * score_mask_g

            all_ref.append(ref_logit_g)
            all_content.append(content_g)
            all_scores.append(scores_g)
            all_classes.append(classes_g)

        return QueryBundle(
            ref_points = torch.cat(all_ref,     dim=1),
            content    = torch.cat(all_content, dim=1),
            scores     = torch.cat(all_scores,  dim=1),
            classes    = torch.cat(all_classes, dim=1),
        )

    def _forward_hard_partitioned(
        self,
        hm: Tensor, feat: Tensor,
        B: int, C: int, Hs: int, Ws: int, D: int, N: int,
    ) -> QueryBundle:
        R     = self.sp_grid_rows
        Cols  = self.sp_grid_cols
        K_loc = self.sp_queries_per_region
        K_glb = self.sp_global_queries
        ovlp  = self.sp_overlap_ratio

        hm_max_scores, hm_max_cls = hm.max(dim=1)

        all_ref, all_content, all_scores, all_classes = [], [], [], []

        for r in range(R):
            for c in range(Cols):
                h0 = max(0.0, r / R         - ovlp / 2)
                h1 = min(1.0, (r + 1) / R   + ovlp / 2)
                w0 = max(0.0, c / Cols       - ovlp / 2)
                w1 = min(1.0, (c + 1) / Cols + ovlp / 2)

                rh0 = int(h0 * Hs)
                rh1 = min(Hs, max(rh0 + 1, int(math.ceil(h1 * Hs))))
                rw0 = int(w0 * Ws)
                rw1 = min(Ws, max(rw0 + 1, int(math.ceil(w1 * Ws))))
                rHs, rWs = rh1 - rh0, rw1 - rw0
                rN  = rHs * rWs
                K_r = min(K_loc, rN)

                s_r_flat   = hm_max_scores[:, rh0:rh1, rw0:rw1].reshape(B, rN)
                cls_r_flat = hm_max_cls[:, rh0:rh1, rw0:rw1].reshape(B, rN)

                scores_r, local_idx = torch.topk(s_r_flat, K_r, dim=1, sorted=True)
                classes_r = cls_r_flat.gather(1, local_idx)

                y_loc = (local_idx // rWs).float()
                x_loc = (local_idx %  rWs).float()
                cx_r  = (rw0 + x_loc + 0.5) / Ws
                cy_r  = (rh0 + y_loc + 0.5) / Hs

                ref_w_r = torch.full_like(cx_r, 0.05)
                ref_h_r = torch.full_like(cy_r, 0.05)

                feat_r_flat = feat[:, :, rh0:rh1, rw0:rw1].reshape(B, D, rN)
                idx_exp_r   = local_idx.unsqueeze(1).expand(B, D, K_r)
                peak_feat_r = feat_r_flat.gather(2, idx_exp_r).permute(0, 2, 1)
                content_r   = self.content_proj(peak_feat_r)

                ref_logit_r  = self._sigmoid_to_logit(
                    torch.stack([cx_r, cy_r, ref_w_r, ref_h_r], dim=-1))
                score_mask_r = (scores_r >= self.score_thr).float().unsqueeze(-1)
                content_r    = content_r * score_mask_r

                all_ref.append(ref_logit_r)
                all_content.append(content_r)
                all_scores.append(scores_r)
                all_classes.append(classes_r)

        if K_glb > 0:
            K_g = min(K_glb, N)
            s_flat   = hm_max_scores.reshape(B, N)
            cls_flat = hm_max_cls.reshape(B, N)

            scores_g, spatial_idx = torch.topk(s_flat, K_g, dim=1, sorted=True)
            classes_g = cls_flat.gather(1, spatial_idx)

            y_idx = (spatial_idx // Ws).float()
            x_idx = (spatial_idx %  Ws).float()
            cx_g = (x_idx + 0.5) / Ws
            cy_g = (y_idx + 0.5) / Hs

            ref_w_g = torch.full_like(cx_g, 0.05)
            ref_h_g = torch.full_like(cy_g, 0.05)

            feat_flat_g = feat.reshape(B, D, N)
            idx_exp_g   = spatial_idx.unsqueeze(1).expand(B, D, K_g)
            peak_feat_g = feat_flat_g.gather(2, idx_exp_g).permute(0, 2, 1)
            content_g   = self.content_proj(peak_feat_g)

            ref_logit_g  = self._sigmoid_to_logit(
                torch.stack([cx_g, cy_g, ref_w_g, ref_h_g], dim=-1))
            score_mask_g = (scores_g >= self.score_thr).float().unsqueeze(-1)
            content_g    = content_g * score_mask_g

            all_ref.append(ref_logit_g)
            all_content.append(content_g)
            all_scores.append(scores_g)
            all_classes.append(classes_g)

        return QueryBundle(
            ref_points = torch.cat(all_ref,     dim=1),
            content    = torch.cat(all_content, dim=1),
            scores     = torch.cat(all_scores,  dim=1),
            classes    = torch.cat(all_classes, dim=1),
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hm:   Tensor,   # (B, C, H, W)  sigmoid score map (C=1 for objectness)
        feat: Tensor,   # (B, D, H, W)  spatial features
    ) -> QueryBundle:
        B, C, Hs, Ws = hm.shape
        D = feat.shape[1]
        N = Hs * Ws

        if self.use_spatial_partition:
            if self.use_gumbel and self.training:
                return self._forward_gumbel_partitioned(hm, feat, B, C, Hs, Ws, D, N)
            return self._forward_hard_partitioned(hm, feat, B, C, Hs, Ws, D, N)

        K = min(self.top_k, N)
        if self.use_gumbel and self.training:
            return self._forward_gumbel(hm, feat, B, C, Hs, Ws, D, N, K)
        return self._forward_hard(hm, feat, B, C, Hs, Ws, D, N, K)
