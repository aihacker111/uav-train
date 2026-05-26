"""
TALAssigner — Task-Aligned Label Assignment for HawkDet.

For each GT box, selects the top-k predictions with the highest alignment
score:
    t = cls_score ^ alpha  *  IoU ^ beta

Constraints:
  - Anchor point must lie inside the GT box (centre-in-box).
  - Each prediction is assigned to at most one GT (best IoU wins conflicts).

Soft classification targets equal the IoU of the assigned GT at the
matched class index (QFL-style quality target).

Reference:
  TOOD — Task-aligned One-stage Object Detection, Feng et al. ICCV 2021
  YOLOv8 — Ultralytics (TAL-based label assignment, 2023)
"""
from __future__ import annotations

import torch
from torch import Tensor


# ── Box utilities ─────────────────────────────────────────────────────────────

def box_iou_pairwise(b1: Tensor, b2: Tensor) -> Tensor:
    """Element-wise IoU for N matched pairs.

    Args:
        b1, b2: (N, 4) xyxy (any consistent unit — pixels or normalised).
    Returns:
        (N,) IoU values.
    """
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    return inter / (a1 + a2 - inter + 1e-7)


def box_iou_matrix(b1: Tensor, b2: Tensor) -> Tensor:
    """Pairwise IoU between two box sets.

    Args:
        b1: (N, 4) xyxy
        b2: (M, 4) xyxy
    Returns:
        (N, M)
    """
    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)  # (N,)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)  # (M,)

    ix1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    iy1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    ix2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    iy2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)           # (N, M)
    union = a1[:, None] + a2[None, :] - inter                       # (N, M)
    return inter / (union + 1e-7)


def cxcywh_to_xyxy_norm(boxes: Tensor) -> Tensor:
    """Normalised cxcywh → normalised xyxy."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def cxcywh_to_xyxy_pixel(boxes: Tensor, input_hw: tuple) -> Tensor:
    """Normalised cxcywh → pixel-space xyxy."""
    H, W  = input_hw
    xyxy  = cxcywh_to_xyxy_norm(boxes)
    scale = boxes.new_tensor([W, H, W, H])
    return xyxy * scale


# ── TALAssigner ───────────────────────────────────────────────────────────────

class TALAssigner:
    """Task-Aligned Label Assignment (TOOD / YOLOv8).

    Args:
        topk  : number of positives to select per GT (across all scales).
        alpha : cls-score exponent in alignment metric.
        beta  : IoU exponent in alignment metric.
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0) -> None:
        self.topk  = topk
        self.alpha = alpha
        self.beta  = beta

    @torch.no_grad()
    def assign(
        self,
        pred_boxes:  Tensor,   # (P, 4) cxcywh normalised
        pred_scores: Tensor,   # (P, C) sigmoid probabilities
        gt_boxes:    Tensor,   # (G, 4) cxcywh normalised
        gt_labels:   Tensor,   # (G,)   int64
        anchors:     Tensor,   # (P, 2) pixel-space (x, y)
        input_hw:    tuple,    # (H, W) input image
    ):
        """
        Returns:
            assigned_labels : (P,)    int64  — -1 = background, ≥0 = class index
            assigned_boxes  : (P, 4)         — GT box for each positive (zeros for bg)
            assigned_scores : (P, C)         — soft QFL targets (IoU at positive, 0 bg)
        """
        P, C  = pred_scores.shape
        G     = gt_boxes.shape[0]
        device = pred_boxes.device

        assigned_labels  = gt_labels.new_full((P,), -1)
        assigned_boxes   = pred_boxes.new_zeros(P, 4)
        assigned_scores  = pred_scores.new_zeros(P, C)
        assigned_gt_idx  = gt_labels.new_full((P,), -1)

        if G == 0:
            return assigned_labels, assigned_boxes, assigned_scores, assigned_gt_idx

        H, W = input_hw

        # ── Centre-in-box: anchor (pixel) inside GT (pixel xyxy) ───────────
        gt_xyxy_px = cxcywh_to_xyxy_pixel(gt_boxes, input_hw)   # (G, 4)
        ax, ay     = anchors[:, 0], anchors[:, 1]                # (P,)

        in_box = (
            (ax[:, None] >= gt_xyxy_px[None, :, 0]) &
            (ax[:, None] <= gt_xyxy_px[None, :, 2]) &
            (ay[:, None] >= gt_xyxy_px[None, :, 1]) &
            (ay[:, None] <= gt_xyxy_px[None, :, 3])
        )  # (P, G)

        # ── IoU between all pred boxes and GT boxes ─────────────────────────
        pred_xyxy = cxcywh_to_xyxy_norm(pred_boxes)   # (P, 4) normalised
        gt_xyxy   = cxcywh_to_xyxy_norm(gt_boxes)     # (G, 4) normalised
        iou_mat   = box_iou_matrix(pred_xyxy, gt_xyxy)  # (P, G)

        # ── Alignment metric: t = cls_score^alpha * iou^beta ───────────────
        # For each (pred, gt) pair: score at gt's class
        gt_cls       = gt_labels.long()                      # (G,)
        cls_at_gt    = pred_scores[:, gt_cls]                # (P, G)
        align_metric = cls_at_gt.pow(self.alpha) * iou_mat.pow(self.beta)
        align_metric = align_metric * in_box.float()         # zero outside GT box

        # ── Per-GT top-k ────────────────────────────────────────────────────
        topk = min(self.topk, P)
        _, topk_idx = align_metric.topk(topk, dim=0, largest=True)  # (K, G)

        is_topk = torch.zeros(P, G, dtype=torch.bool, device=device)
        is_topk.scatter_(0, topk_idx, True)
        is_topk = is_topk & in_box  # must also satisfy centre-in-box

        # ── Resolve conflicts: each pred → one GT (highest IoU) ─────────────
        is_pos = is_topk.any(dim=-1)   # (P,)
        if not is_pos.any():
            return assigned_labels, assigned_boxes, assigned_scores, assigned_gt_idx

        # Among all GTs a pred is topk for, keep the one with highest IoU
        masked_iou  = iou_mat * is_topk.float()     # (P, G)
        best_gt_idx = masked_iou.argmax(dim=-1)     # (P,) — best GT per pred

        pos_mask    = is_pos.nonzero(as_tuple=True)[0]          # (num_pos,)
        best_gts    = best_gt_idx[pos_mask]                     # (num_pos,)

        assigned_labels[pos_mask] = gt_labels[best_gts]
        assigned_boxes[pos_mask]  = gt_boxes[best_gts]
        assigned_gt_idx[pos_mask] = best_gts

        # Soft target: IoU with matched GT at matched class (QFL quality target)
        iou_pos                           = iou_mat[pos_mask, best_gts].clamp(0, 1)
        cls_pos                           = gt_labels[best_gts].long()
        assigned_scores[pos_mask, cls_pos] = iou_pos

        return assigned_labels, assigned_boxes, assigned_scores, assigned_gt_idx
