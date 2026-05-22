"""
HungarianMatcher for Stage-2 DETR loss.

Matching cost:
  C = λ_cls * focal_cost + λ_bbox * L1_cost + λ_giou * GIoU_cost

All box tensors are in normalised cxcywh format.

Interface follows DEIMv2:
  forward(outputs: dict, targets, epoch=0) -> {'indices': [(src, tgt), ...]}
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch import Tensor


# ── Box utilities ──────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(b1: Tensor, b2: Tensor) -> Tuple[Tensor, Tensor]:
    """Pairwise IoU between b1 (N, 4) and b2 (M, 4) in xyxy. Returns (N, M) iou, (N, M) union."""
    area1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)

    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    iou   = inter / (union + 1e-7)
    return iou, union


# Keep _box_iou as alias for backward compat
_box_iou = box_iou


def generalized_box_iou(b1: Tensor, b2: Tensor) -> Tensor:
    """GIoU matrix of shape (N, M) for boxes b1 (N, 4) and b2 (M, 4) in xyxy."""
    iou, union = box_iou(b1, b2)
    enc_x1 = torch.min(b1[:, None, 0], b2[None, :, 0])
    enc_y1 = torch.min(b1[:, None, 1], b2[None, :, 1])
    enc_x2 = torch.max(b1[:, None, 2], b2[None, :, 2])
    enc_y2 = torch.max(b1[:, None, 3], b2[None, :, 3])
    enc    = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)
    return iou - (enc - union) / (enc + 1e-7)


# ── HungarianMatcher ───────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching between K predictions and N targets per image.

    Interface: forward(outputs: dict, targets, epoch=0) -> {'indices': [(src, tgt), ...]}
    Compatible with DEIMv2 DEIMCriterion calling convention.
    """

    def __init__(
        self,
        cost_class:  float = 2.0,
        cost_bbox:   float = 5.0,
        cost_giou:   float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class  = cost_class
        self.cost_bbox   = cost_bbox
        self.cost_giou   = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(
        self,
        outputs:  dict,           # must have 'pred_logits' (B,K,C) and 'pred_boxes' (B,K,4)
        targets:  List[dict],     # list of {'labels': (N,), 'boxes': (N, 4)}
        epoch:    int = 0,
    ) -> Dict:
        """
        Returns {'indices': [(src_tensor, tgt_tensor), ...]} — one tuple per batch image.
        src_tensor: (n,) long — matched prediction indices
        tgt_tensor: (n,) long — matched target indices
        """
        pred_logits = outputs['pred_logits']
        pred_boxes  = outputs['pred_boxes']
        B, K = pred_logits.shape[:2]

        prob  = pred_logits.flatten(0, 1).sigmoid()    # (B*K, C)
        boxes = pred_boxes.flatten(0, 1)               # (B*K, 4)

        tgt_ids  = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes']  for t in targets])

        # Focal classification cost
        a, g  = self.focal_alpha, self.focal_gamma
        neg   = (1 - a) * (prob ** g) * (-(1 - prob + 1e-8).log())
        pos   =       a * ((1 - prob) ** g) * (-(prob + 1e-8).log())
        cost_cls  = pos[:, tgt_ids] - neg[:, tgt_ids]   # (B*K, num_tgt)

        # L1 box cost
        cost_l1   = torch.cdist(boxes, tgt_bbox, p=1)

        # GIoU cost
        boxes_xyxy    = box_cxcywh_to_xyxy(boxes)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou_mat = -generalized_box_iou(boxes_xyxy, tgt_bbox_xyxy)

        C = (self.cost_class * cost_cls
             + self.cost_bbox  * cost_l1
             + self.cost_giou  * cost_giou_mat)
        C = C.view(B, K, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)

        sizes   = [len(t['boxes']) for t in targets]
        indices = []
        for b, c_b in enumerate(C.split(sizes, dim=2)):
            i, j = linear_sum_assignment(c_b[b].numpy())
            indices.append((
                torch.as_tensor(i, dtype=torch.long),
                torch.as_tensor(j, dtype=torch.long),
            ))

        return {'indices': indices}
