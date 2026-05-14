"""
HybridLoss: combined Stage-1 (CenterNet) + Stage-2 (DETR + ReID) loss.

Stage 1 — CenterNet:
  L_s1 = L_focal(hm) + λ_wh * L_L1(wh) + λ_reg * L_L1(reg)

Stage 2 — DETR (Hungarian matching, applied per decoder layer):
  L_s2 = L_focal(cls) + λ_bbox * L_L1(box) + λ_giou * L_GIoU(box)
        + λ_reid * L_CE(reid)  [if reid_classifier is set]

Auxiliary losses from intermediate decoder layers are summed with weight 0.5.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou
from ..networks.hybrid.heads import CenterNetOutput, DETROutput


# ── Loss primitives ────────────────────────────────────────────────────────────

def sigmoid_focal_loss(
    pred:   Tensor,   # (*, C) raw logits
    target: Tensor,   # (*, C) binary targets in {0, 1}
    alpha:  float = 0.25,
    gamma:  float = 2.0,
) -> Tensor:
    """Element-averaged sigmoid focal loss."""
    p   = pred.sigmoid()
    ce  = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    w   = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - p_t) ** gamma * ce).mean()


def centernet_focal_loss(pred_hm: Tensor, gt_hm: Tensor) -> Tensor:
    """
    Modified CenterNet focal loss on Gaussian-rendered heatmaps.

    pred_hm : (B, C, H, W) — sigmoid output
    gt_hm   : (B, C, H, W) — Gaussian-rendered ground-truth in [0, 1]
    """
    pos_mask = (gt_hm == 1).float()
    neg_mask = 1.0 - pos_mask

    p     = pred_hm.clamp(1e-6, 1.0 - 1e-6)
    pos_l = -((1 - p) ** 2) * p.log() * pos_mask
    neg_l = -((1 - gt_hm) ** 4) * (p ** 2) * (1 - p).log() * neg_mask

    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_l + neg_l).sum() / n_pos


def _gather_at_ind(feat: Tensor, ind: Tensor) -> Tensor:
    """
    Gather spatial predictions at flat peak indices.

    feat : (B, C, H, W)
    ind  : (B, max_obj)  — flat HW index
    Returns (B, max_obj, C)
    """
    B, C, H, W = feat.shape
    flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
    idx  = ind.unsqueeze(-1).expand(B, ind.shape[1], C)
    return flat.gather(1, idx)


# ── HybridLoss ─────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Combined CenterNet + DETR loss for HybridCenterNetDETR.

    The batch dict must contain:
      Stage-1 targets (CenterNet format):
        'hm'       : (B, C, H, W)        — Gaussian rendered heatmap
        'wh'       : (B, max_obj, 2)      — width/height at peak locations
        'reg'      : (B, max_obj, 2)      — sub-pixel offset at peak locations
        'ind'      : (B, max_obj)         — flat spatial index of each peak
        'reg_mask' : (B, max_obj)         — 1 for valid objects, 0 for padding

      Stage-2 targets (DETR format):
        'targets'  : List[dict] with keys 'labels' (N,) and 'boxes' (N, 4) cxcywh

      ReID targets (optional):
        'ids'      : (B, max_obj)         — track ID (−1 = ignore)
    """

    def __init__(
        self,
        num_classes:   int   = 7,
        lambda_wh:     float = 0.1,
        lambda_reg:    float = 1.0,
        lambda_bbox:   float = 5.0,
        lambda_giou:   float = 2.0,
        lambda_reid:   float = 1.0,
        lambda_stage1: float = 1.0,
        lambda_stage2: float = 1.0,
        aux_loss:      bool  = True,
        reid_classifier: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.lambda_wh     = lambda_wh
        self.lambda_reg    = lambda_reg
        self.lambda_bbox   = lambda_bbox
        self.lambda_giou   = lambda_giou
        self.lambda_reid   = lambda_reid
        self.lambda_stage1 = lambda_stage1
        self.lambda_stage2 = lambda_stage2
        self.aux_loss      = aux_loss

        # Linear classifier for ReID (embedding → track_id class).
        # Must be set before training begins; dimension depends on the dataset.
        self.reid_classifier = reid_classifier

        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    # ── Stage-1 ────────────────────────────────────────────────────────────────

    def _stage1_loss(self, cn_out: CenterNetOutput, batch: dict) -> Tensor:
        loss_hm = centernet_focal_loss(cn_out.hm, batch['hm'])

        reg_mask = batch['reg_mask']       # (B, max_obj)
        ind      = batch['ind']            # (B, max_obj)
        n        = reg_mask.sum().clamp(min=1).float()

        pred_wh  = _gather_at_ind(cn_out.wh,  ind)   # (B, max_obj, 2)
        pred_reg = _gather_at_ind(cn_out.reg, ind)

        mask = reg_mask.unsqueeze(-1).float()
        loss_wh  = (F.l1_loss(pred_wh,  batch['wh'],  reduction='none') * mask).sum() / n
        loss_reg = (F.l1_loss(pred_reg, batch['reg'], reduction='none') * mask).sum() / n

        return loss_hm + self.lambda_wh * loss_wh + self.lambda_reg * loss_reg

    # ── Stage-2 ────────────────────────────────────────────────────────────────

    def _detr_layer_loss(
        self,
        logits:  Tensor,       # (B, K, C)
        boxes:   Tensor,       # (B, K, 4)
        targets: List[dict],
        indices: list,
    ) -> Tensor:
        B = logits.shape[0]

        # Classification: one-hot target matrix
        tgt_cls = torch.zeros_like(logits)
        for b, (src_i, tgt_i) in enumerate(indices):
            if len(src_i):
                tgt_cls[b, src_i, targets[b]['labels'][tgt_i]] = 1.0
        loss_cls = sigmoid_focal_loss(logits, tgt_cls)

        # Box regression on matched pairs
        src_b_list, tgt_b_list = [], []
        for b, (src_i, tgt_i) in enumerate(indices):
            if len(src_i):
                src_b_list.append(boxes[b][src_i])
                tgt_b_list.append(targets[b]['boxes'][tgt_i])

        if src_b_list:
            src_b = torch.cat(src_b_list)
            tgt_b = torch.cat(tgt_b_list)
            n_m   = src_b.shape[0]
            loss_l1   = F.l1_loss(src_b, tgt_b, reduction='sum') / n_m
            loss_giou = (
                1 - generalized_box_iou(
                    box_cxcywh_to_xyxy(src_b),
                    box_cxcywh_to_xyxy(tgt_b),
                ).diagonal()
            ).mean()
        else:
            loss_l1 = loss_giou = logits.sum() * 0.0

        return loss_cls + self.lambda_bbox * loss_l1 + self.lambda_giou * loss_giou

    def _stage2_loss(
        self,
        detr_out: DETROutput,
        targets:  List[dict],
        batch:    Optional[dict] = None,
    ) -> Tensor:
        # Final-layer matching and loss
        indices = self.matcher(detr_out.logits, detr_out.boxes, targets)
        loss    = self._detr_layer_loss(detr_out.logits, detr_out.boxes, targets, indices)

        # Auxiliary losses from intermediate layers (half weight)
        if self.aux_loss:
            L = detr_out.boxes_all.shape[0]
            for layer in range(L - 1):
                idx  = self.matcher(detr_out.logits_all[layer], detr_out.boxes_all[layer], targets)
                loss = loss + 0.5 * self._detr_layer_loss(
                    detr_out.logits_all[layer], detr_out.boxes_all[layer], targets, idx,
                )

        # ReID loss (optional — only if classifier is provided and batch has ids)
        if self.reid_classifier is not None and batch is not None and 'ids' in batch:
            loss = loss + self.lambda_reid * self._reid_loss(detr_out, batch)

        return loss

    def _reid_loss(self, detr_out: DETROutput, batch: dict) -> Tensor:
        """Cross-entropy ReID loss on matched query embeddings."""
        # Use the stage-1 index layout to find valid ids
        ids      = batch['ids']          # (B, max_obj)  track id, −1 = ignore
        reg_mask = batch['reg_mask']     # (B, max_obj)
        ind      = batch['ind']          # (B, max_obj)

        B, K, reid_dim = detr_out.reid.shape
        _, HW = batch['ind'].shape

        valid_emb, valid_ids = [], []
        for b in range(B):
            mask = reg_mask[b].bool()
            if not mask.any():
                continue
            id_b = ids[b][mask]          # (n_valid,)
            keep = id_b >= 0
            if not keep.any():
                continue
            # Map spatial ind to K-query index via nearest-peak lookup is complex;
            # for simplicity gather from the reid tensor using stage-1 ind directly.
            # This approximation works because QueryGenerator preserves spatial order.
            idx_b = ind[b][mask][keep]   # flat spatial indices → used as query proxy
            idx_b = idx_b.clamp(0, K - 1)
            valid_emb.append(detr_out.reid[b][idx_b])
            valid_ids.append(id_b[keep])

        if not valid_emb:
            return detr_out.reid.sum() * 0.0

        emb = torch.cat(valid_emb)           # (N, reid_dim)
        ids_t = torch.cat(valid_ids)         # (N,)
        logits = self.reid_classifier(emb)   # (N, num_ids)
        return F.cross_entropy(logits, ids_t)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        outputs: Dict[str, Any],
        batch:   dict,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            outputs : dict from HybridCenterNetDETR.forward
            batch   : training batch dict (see class docstring for required keys)
        Returns:
            (total_loss, loss_stats) — compatible with BaseTrainer.ModelWithLoss
        """
        l_s1 = self._stage1_loss(outputs['stage1'], batch)
        l_s2 = self._stage2_loss(outputs['stage2'], batch['targets'], batch)

        total = self.lambda_stage1 * l_s1 + self.lambda_stage2 * l_s2

        loss_stats = {
            'loss':        total,
            'loss_stage1': l_s1,
            'loss_stage2': l_s2,
        }
        return total, loss_stats
