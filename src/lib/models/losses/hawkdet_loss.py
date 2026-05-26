"""
HawkDetLoss — QFL + DFL + GIoU [+ ReID CE] with TAL assignment across 4 scales.

Per image, all scale predictions are flattened, TAL assignment is run
once, then QFL (classification), DFL (distribution regression), and
GIoU (IoU regression) are computed jointly on positives.

Loss terms:
  L = lambda_cls * L_QFL  +  lambda_dfl * L_DFL  +  lambda_giou * L_GIoU
    [ + lambda_reid * L_ReID ]   (when reid_dim > 0 and num_ids > 0)

References:
  GFL   — Generalized Focal Loss, Li et al. NeurIPS 2020 (QFL + DFL)
  GFocalV2 — Li et al. CVPR 2021 (DGQP, distribution quality)
  TOOD  — Feng et al. ICCV 2021 (TAL assignment)
  YOLOv8 — Ultralytics 2023 (DFL + TAL production implementation)
  FairMOT — Zhang et al. ECCV 2020 (dense ReID branch)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..networks.hawkdet.head import make_anchors, dfl_decode, ltrb_to_cxcywh
from ..networks.hawkdet.assigner import (
    TALAssigner,
    cxcywh_to_xyxy_norm,
    cxcywh_to_xyxy_pixel,
)


# ── Loss primitives ──────────────────────────────────────────────────────────

def qfl_loss(pred_logits: Tensor, target_scores: Tensor, beta: float = 2.0) -> Tensor:
    """Quality Focal Loss (GFL, NeurIPS 2020).

    BCE with a dynamic quality-aware modulating factor |σ − y|^beta.

    Args:
        pred_logits   : (N, C) raw logits
        target_scores : (N, C) soft quality targets in [0, 1]
        beta          : modulating exponent
    Returns:
        scalar — sum over all elements (normalised externally by num_pos).
    """
    sigma  = pred_logits.sigmoid()
    weight = (sigma.detach() - target_scores).abs().pow(beta)
    bce    = F.binary_cross_entropy_with_logits(
        pred_logits, target_scores, reduction='none'
    )
    return (weight * bce).sum()


def dfl_loss(pred_dist: Tensor, target_bins: Tensor, reg_max: int) -> Tensor:
    """Distribution Focal Loss (GFL, NeurIPS 2020).

    Interpolates CE between the two adjacent integer bins that bracket the
    continuous target distance.

    Args:
        pred_dist   : (N, 4*(reg_max+1)) raw logits
        target_bins : (N, 4)             continuous target distances in [0, reg_max]
        reg_max     : int
    Returns:
        scalar — sum (normalised externally by num_pos).
    """
    N   = pred_dist.shape[0]
    tgt = target_bins.clamp(0, reg_max)          # (N, 4)

    lo = tgt.long()                              # (N, 4)
    hi = (lo + 1).clamp(max=reg_max)             # (N, 4)
    w_hi = tgt - lo.float()                      # (N, 4) weight for hi bin
    w_lo = 1.0 - w_hi                            # (N, 4)

    # Reshape predictions to (4N, reg_max+1) for cross_entropy
    logits = pred_dist.reshape(N, 4, reg_max + 1).permute(1, 0, 2)  # (4, N, R+1)
    logits = logits.reshape(4 * N, reg_max + 1)

    ce_lo = F.cross_entropy(logits, lo.t().reshape(-1),  reduction='none').reshape(4, N)
    ce_hi = F.cross_entropy(logits, hi.t().reshape(-1), reduction='none').reshape(4, N)

    # (4, N) → scalar
    return (w_lo.t() * ce_lo + w_hi.t() * ce_hi).sum()


def giou_loss(pred_xyxy: Tensor, tgt_xyxy: Tensor) -> Tensor:
    """GIoU loss for N matched pairs."""
    ix1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    iy1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    ix2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    iy2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    inter     = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(0)
    tgt_area  = (tgt_xyxy[:, 2]  - tgt_xyxy[:, 0]).clamp(0) * (tgt_xyxy[:, 3]  - tgt_xyxy[:, 1]).clamp(0)
    union     = pred_area + tgt_area - inter
    iou       = inter / (union + 1e-7)

    enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    enc    = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    return (1.0 - iou + (enc - union) / (enc + 1e-7)).mean()


# ── HawkDetLoss ───────────────────────────────────────────────────────────────

class HawkDetLoss(nn.Module):
    """
    HawkDet multi-scale detection loss.

    Expects model output (training mode):
        {'cls': List[(B,C,Hi,Wi)], 'reg': List[(B,4*(R+1),Hi,Wi)]}
    Batch dict must contain:
        'input'   : (B, 3, H, W)  — used for input_hw
        'targets' : List[dict] with 'boxes' (N,4) cxcywh [0,1] and 'labels' (N,)
    """

    strides: List[int] = [4, 8, 16, 32]

    def __init__(
        self,
        num_classes: int   = 7,
        reg_max:     int   = 16,
        lambda_cls:  float = 1.0,
        lambda_dfl:  float = 1.5,
        lambda_giou: float = 2.5,
        tal_topk:    int   = 13,
        tal_alpha:   float = 1.0,
        tal_beta:    float = 6.0,
        qfl_beta:    float = 2.0,
        reid_dim:    int   = 0,
        num_ids:     int   = 0,
        lambda_reid: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.lambda_cls  = lambda_cls
        self.lambda_dfl  = lambda_dfl
        self.lambda_giou = lambda_giou
        self.lambda_reid = lambda_reid
        self.qfl_beta    = qfl_beta
        self.reid_dim    = reid_dim

        self.assigner = TALAssigner(topk=tal_topk, alpha=tal_alpha, beta=tal_beta)

        # ReID classifier: projects L2-normalised embeddings → identity logits
        if reid_dim > 0 and num_ids > 0:
            self.reid_classifier = nn.Linear(reid_dim, num_ids)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        outputs: Dict,
        batch:   Dict,
    ):
        cls_list  = outputs['cls']              # List[(B, C, Hi, Wi)]
        reg_list  = outputs['reg']              # List[(B, 4*(R+1), Hi, Wi)]
        reid_list = outputs.get('reid')         # List[(B, D, Hi, Wi)] or None
        targets   = batch['targets']
        input_hw  = tuple(batch['input'].shape[-2:])  # (H, W)
        device    = cls_list[0].device
        B         = cls_list[0].shape[0]

        # ── Flatten all scales ───────────────────────────────────────────────
        flat_cls, flat_reg, flat_reid_parts, all_anchors, all_strides = [], [], [], [], []

        for i, (cls, reg) in enumerate(zip(cls_list, reg_list)):
            _, C, H, W = cls.shape
            stride     = self.strides[i]
            anchors    = make_anchors(H, W, stride, device)     # (H*W, 2)

            flat_cls.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, C))
            flat_reg.append(reg.permute(0, 2, 3, 1).reshape(B, H * W, 4 * (self.reg_max + 1)))
            all_anchors.append(anchors)
            all_strides.append(cls.new_full((H * W,), stride))
            if reid_list is not None:
                r = reid_list[i]
                flat_reid_parts.append(r.permute(0, 2, 3, 1).reshape(B, H * W, -1))

        flat_cls     = torch.cat(flat_cls,     dim=1)   # (B, N, C)
        flat_reg     = torch.cat(flat_reg,     dim=1)   # (B, N, 4*(R+1))
        anchors_all  = torch.cat(all_anchors,  dim=0)   # (N, 2)
        strides_all  = torch.cat(all_strides,  dim=0)   # (N,)
        flat_reid    = torch.cat(flat_reid_parts, dim=1) if flat_reid_parts else None  # (B, N, D)
        N = flat_cls.shape[1]

        # ── Decode pred boxes for TAL metric (no grad) ──────────────────────
        with torch.no_grad():
            dist     = dfl_decode(flat_reg, self.reg_max)          # (B, N, 4)
            pred_boxes = ltrb_to_cxcywh(
                dist, anchors_all, strides_all[None, :, None], input_hw
            )                                                        # (B, N, 4)
            pred_scores_detach = flat_cls.sigmoid()                  # (B, N, C)

        # ── Accumulate losses ────────────────────────────────────────────────
        total_cls   = flat_cls.new_tensor(0.0)
        total_dfl   = flat_cls.new_tensor(0.0)
        total_giou  = flat_cls.new_tensor(0.0)
        total_reid  = flat_cls.new_tensor(0.0)
        num_pos     = 0
        num_reid    = 0

        for b in range(B):
            gt = targets[b]
            gt_boxes  = gt['boxes'].to(device)    # (G, 4) cxcywh
            gt_labels = gt['labels'].to(device)
            if gt_labels.ndim == 2:
                gt_labels = gt_labels[:, 0]
            gt_labels = gt_labels.long()

            G = gt_boxes.shape[0]

            if G == 0:
                # All-negative image: QFL with zero quality targets
                total_cls = total_cls + self.lambda_cls * qfl_loss(
                    flat_cls[b],
                    flat_cls[b].new_zeros(N, self.num_classes),
                    self.qfl_beta,
                )
                continue

            a_labels, a_boxes, a_scores, a_gt_idx = self.assigner.assign(
                pred_boxes=pred_boxes[b],
                pred_scores=pred_scores_detach[b],
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                anchors=anchors_all,
                input_hw=input_hw,
            )

            is_pos   = (a_labels >= 0)     # (N,)
            n_pos    = is_pos.sum().item()
            num_pos += n_pos

            # QFL on ALL N predictions
            total_cls = total_cls + self.lambda_cls * qfl_loss(
                flat_cls[b], a_scores, self.qfl_beta
            )

            if n_pos == 0:
                continue

            # ── DFL + GIoU on positives only ────────────────────────────────
            pos_anchors = anchors_all[is_pos]         # (P, 2)
            pos_strides = strides_all[is_pos]         # (P,)
            pos_reg     = flat_reg[b][is_pos]         # (P, 4*(R+1))
            pos_gt      = a_boxes[is_pos]             # (P, 4) cxcywh

            # Convert GT to ltrb in bin space for DFL supervision
            gt_xyxy_px = cxcywh_to_xyxy_pixel(pos_gt, input_hw)  # (P, 4) pixel
            gt_l = (pos_anchors[:, 0] - gt_xyxy_px[:, 0]).clamp(0) / pos_strides
            gt_t = (pos_anchors[:, 1] - gt_xyxy_px[:, 1]).clamp(0) / pos_strides
            gt_r = (gt_xyxy_px[:, 2] - pos_anchors[:, 0]).clamp(0) / pos_strides
            gt_b = (gt_xyxy_px[:, 3] - pos_anchors[:, 1]).clamp(0) / pos_strides
            gt_bins = torch.stack([gt_l, gt_t, gt_r, gt_b], dim=-1).clamp(0, self.reg_max)

            total_dfl = total_dfl + self.lambda_dfl * dfl_loss(pos_reg, gt_bins, self.reg_max)

            # GIoU: decode DFL predictions at positive locations
            pos_dist    = dfl_decode(pos_reg, self.reg_max)          # (P, 4)
            pos_dist_px = pos_dist * pos_strides[:, None]             # pixels
            H_in, W_in  = input_hw
            x1 = (pos_anchors[:, 0] - pos_dist_px[:, 0]) / W_in
            y1 = (pos_anchors[:, 1] - pos_dist_px[:, 1]) / H_in
            x2 = (pos_anchors[:, 0] + pos_dist_px[:, 2]) / W_in
            y2 = (pos_anchors[:, 1] + pos_dist_px[:, 3]) / H_in
            pred_xyxy   = torch.stack([x1, y1, x2, y2], dim=-1)
            tgt_xyxy    = cxcywh_to_xyxy_norm(pos_gt)

            total_giou = total_giou + self.lambda_giou * giou_loss(pred_xyxy, tgt_xyxy)

            # ── ReID CE on positives ─────────────────────────────────────────
            if flat_reid is not None and hasattr(self, 'reid_classifier'):
                gt_ids = gt.get('ids')
                if gt_ids is not None:
                    gt_ids      = gt_ids.to(device).long()
                    pos_reid    = flat_reid[b][is_pos]               # (P, D)
                    pos_gt_idx  = a_gt_idx[is_pos]                   # (P,)
                    pos_ids     = gt_ids[pos_gt_idx]                 # (P,) track IDs
                    valid       = pos_ids >= 0
                    if valid.any():
                        emb    = F.normalize(pos_reid[valid], dim=-1)
                        logits = self.reid_classifier(emb)
                        total_reid = total_reid + F.cross_entropy(logits, pos_ids[valid])
                        num_reid  += 1

        # ── Normalise ─────────────────────────────────────────────────────────
        denom      = max(num_pos, 1)
        total_cls  = total_cls  / denom
        total_dfl  = total_dfl  / denom
        total_giou = total_giou / B       # GIoU.mean() already over positives; avg over batch

        total = total_cls + total_dfl + total_giou

        loss_reid = flat_cls.new_tensor(0.0)
        if num_reid > 0:
            loss_reid = self.lambda_reid * total_reid / num_reid
            total     = total + loss_reid

        stats: Dict = {
            'loss':      total,
            'loss_cls':  total_cls.detach(),
            'loss_dfl':  total_dfl.detach(),
            'loss_giou': total_giou.detach(),
            'loss_reid': loss_reid.detach(),
            'num_pos':   flat_cls.new_tensor(num_pos / max(B, 1)),
        }
        return total, stats
