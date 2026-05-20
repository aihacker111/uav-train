"""
HybridLoss: simplified loss for HybridDETR.

  Scorer loss (Stage-1 replacement):
    L_score = centernet_focal_loss(score_map, gt_score_s8)
    gt_score_s8 is derived from batch['hm'] (stride-4 GT) via max-pool + class-max.

  Stage-2 DETR loss (unchanged from previous version):
    L_s2 = varifocal(cls) + λ_bbox * L1(box) + λ_ciou * CIoU(box)
         + λ_reid * (CE + triplet)(reid)   [if reid_classifier is set]
    Auxiliary losses from intermediate decoder layers with progressive weights.

  DN reconstruction loss (new):
    Direct supervision on denoising queries — no Hungarian matching needed.
    L_dn = λ_dn_l1 * L1(pred_box, gt_box) + λ_dn_cls * focal(pred_cls, gt_cls)
    Applied across all decoder layers (like aux loss).

Removed:
  wh/reg losses, consistency loss, curriculum stage weighting.
"""
from __future__ import annotations

import math
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou
from ..networks.hybrid.heads import DETROutput
from ..networks.hybrid.dn_gen import DNMeta
from ..base_losses import TripletLoss


# ── Loss primitives ────────────────────────────────────────────────────────────

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


def varifocal_loss(
    pred:  Tensor,
    q:     Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    p    = pred.sigmoid()
    ce   = F.binary_cross_entropy_with_logits(pred, q.clamp(0, 1), reduction='none')

    pos_mask = (q > 0).float()
    neg_mask = 1.0 - pos_mask

    loss = (pos_mask * q + neg_mask * alpha * p.pow(gamma)) * ce
    n_pos = pos_mask.sum().clamp(min=1)
    return loss.sum() / n_pos


def _sigmoid_focal_loss(
    pred:   Tensor,
    target: Tensor,
    alpha:  float = 0.25,
    gamma:  float = 2.0,
) -> Tensor:
    """Standard sigmoid focal loss for DN cls supervision."""
    p   = pred.sigmoid()
    ce  = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    w   = target * (alpha * (1 - p).pow(gamma)) + (1 - target) * ((1 - alpha) * p.pow(gamma))
    return (w * ce).mean()


def _paired_iou(b1: Tensor, b2: Tensor) -> Tensor:
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1    = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2    = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    return inter / (a1 + a2 - inter + 1e-7)


def ciou_loss(pred_xyxy: Tensor, tgt_xyxy: Tensor,
              iou: Tensor | None = None) -> Tensor:
    if iou is None:
        iou = _paired_iou(pred_xyxy, tgt_xyxy)

    enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    c2 = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + 1e-7

    pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    tgt_cx  = (tgt_xyxy[:, 0]  + tgt_xyxy[:, 2])  / 2
    tgt_cy  = (tgt_xyxy[:, 1]  + tgt_xyxy[:, 3])  / 2
    rho2    = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)

    pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(1e-7)
    pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(1e-7)
    tgt_w  = (tgt_xyxy[:, 2]  - tgt_xyxy[:, 0]).clamp(1e-7)
    tgt_h  = (tgt_xyxy[:, 3]  - tgt_xyxy[:, 1]).clamp(1e-7)
    v      = (2 / math.pi) ** 2 * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)).pow(2)

    with torch.no_grad():
        alpha_c = v / (1 - iou + v + 1e-7)

    return (1 - iou + rho2 / c2 + alpha_c * v).mean()


# ── HybridLoss ─────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Simplified loss for HybridDETR.

    Scorer loss:   centernet focal on objectness score_map (stride-8).
    Stage-2 loss:  varifocal + CIoU + ReID (Hungarian matching, aux layers).
    DN loss:       L1 + focal on DN reconstruction (direct assignment).
    """

    def __init__(
        self,
        num_classes:     int   = 7,
        lambda_bbox:     float = 2.0,
        lambda_ciou:     float = 2.0,
        lambda_reid:     float = 1.0,
        lambda_triplet:  float = 0.5,
        lambda_dn_l1:    float = 1.0,
        lambda_dn_cls:   float = 1.0,
        aux_loss:        bool  = True,
        reid_classifier: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.num_classes    = num_classes
        self.lambda_bbox    = lambda_bbox
        self.lambda_ciou    = lambda_ciou
        self.lambda_reid    = lambda_reid
        self.lambda_triplet = lambda_triplet
        self.lambda_dn_l1   = lambda_dn_l1
        self.lambda_dn_cls  = lambda_dn_cls
        self.aux_loss       = aux_loss

        self.reid_classifier = reid_classifier
        self.triplet_loss    = TripletLoss(margin=0.3)
        self.matcher         = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    # ── Scorer loss ────────────────────────────────────────────────────────────

    def _scorer_loss(self, score_map: Tensor, batch: dict) -> Tensor:
        """
        Focal loss on stride-8 objectness score_map.

        GT is derived from batch['hm'] (stride-4 Gaussian heatmap, C classes):
          1. max over class channels → single-channel objectness (B,1,H/4,W/4)
          2. max_pool2d kernel=2 → stride-8 (B,1,H/8,W/8)

        Using max-pool preserves the peak value of 1.0 at GT centers, so the
        centernet focal loss pos_mask (gt == 1) fires correctly.
        """
        dev    = score_map.device
        gt_s4  = batch['hm'].to(dev, non_blocking=True)          # (B, C, H/4, W/4)
        gt_s8  = F.max_pool2d(
            gt_s4.max(dim=1, keepdim=True).values, kernel_size=2, stride=2,
        )                                                          # (B, 1, H/8, W/8)
        return centernet_focal_loss(score_map, gt_s8)

    # ── Stage-2 ────────────────────────────────────────────────────────────────

    def _detr_layer_loss(
        self,
        logits:  Tensor,
        boxes:   Tensor,
        targets: List[dict],
        indices: list,
    ) -> Dict[str, Tensor]:
        dev     = logits.device
        Q_MIN   = 0.1
        tgt_cls = torch.zeros_like(logits)
        src_b_list, tgt_b_list, iou_list = [], [], []

        for b, m in enumerate(indices):
            src_i, tgt_i, iou_cached = m['src_i'], m['tgt_i'], m['iou']
            if not len(src_i):
                continue
            si  = src_i.to(dev)
            ti  = tgt_i.to(dev)
            iou = iou_cached.to(dev)
            tgt_cls[b, si, targets[b]['labels'][ti]] = iou.clamp(min=Q_MIN)
            src_b_list.append(boxes[b][si])
            tgt_b_list.append(targets[b]['boxes'][ti])
            iou_list.append(iou)

        loss_cls = varifocal_loss(logits, tgt_cls)

        if src_b_list:
            src_b   = torch.cat(src_b_list)
            tgt_b   = torch.cat(tgt_b_list)
            iou_all = torch.cat(iou_list)
            n_m     = src_b.shape[0]
            loss_bbox = F.smooth_l1_loss(src_b, tgt_b, beta=0.05, reduction='sum') / n_m
            loss_ciou = ciou_loss(box_cxcywh_to_xyxy(src_b), box_cxcywh_to_xyxy(tgt_b), iou=iou_all)
        else:
            loss_bbox = loss_ciou = logits.sum() * 0.0

        total = loss_cls + self.lambda_bbox * loss_bbox + self.lambda_ciou * loss_ciou
        return {'total': total, 'cls': loss_cls, 'bbox': loss_bbox, 'ciou': loss_ciou}

    def _stage2_loss(
        self,
        detr_out: DETROutput,
        targets:  List[dict],
    ) -> Dict[str, Tensor]:
        indices = self.matcher(detr_out.logits, detr_out.boxes, targets)
        d       = self._detr_layer_loss(detr_out.logits, detr_out.boxes, targets, indices)
        total   = d['total']

        if self.aux_loss:
            L     = detr_out.boxes_all.shape[0]
            n_aux = L - 1
            for layer in range(n_aux):
                aux   = self._detr_layer_loss(
                    detr_out.logits_all[layer], detr_out.boxes_all[layer], targets, indices,
                )
                aux_w = 0.25 + 0.25 * (layer / max(n_aux - 1, 1))
                total = total + aux_w * aux['total']

        return {'total': total, 'cls': d['cls'], 'bbox': d['bbox'],
                'ciou': d['ciou'], 'indices': indices}

    # ── ReID ───────────────────────────────────────────────────────────────────

    def _reid_loss(
        self,
        detr_out: DETROutput,
        targets:  List[dict],
        indices:  list,
    ) -> Tensor:
        valid_emb, valid_ids = [], []
        dev = detr_out.reid.device

        for b, m in enumerate(indices):
            src_i, tgt_i = m['src_i'], m['tgt_i']
            if len(src_i) == 0 or 'ids' not in targets[b]:
                continue
            track_ids = targets[b]['ids'][tgt_i.to(dev)]
            keep      = track_ids >= 0
            if not keep.any():
                continue
            valid_emb.append(detr_out.reid[b][src_i.to(dev)[keep]])
            valid_ids.append(track_ids[keep])

        if not valid_emb:
            return detr_out.reid.sum() * 0.0

        emb    = torch.cat(valid_emb)
        ids_t  = torch.cat(valid_ids).to(emb.device)
        logits = self.reid_classifier(emb)
        loss_ce = F.cross_entropy(logits, ids_t)

        unique_ids = ids_t.unique()
        if unique_ids.numel() >= 2 and emb.shape[0] >= 2:
            loss_tri = self.triplet_loss(emb, ids_t)
        else:
            loss_tri = emb.sum() * 0.0

        return loss_ce + self.lambda_triplet * loss_tri

    # ── DN loss ────────────────────────────────────────────────────────────────

    def _dn_loss(
        self,
        dn_out:  DETROutput,
        dn_meta: DNMeta,
        targets: List[dict],
    ) -> Dict[str, Tensor]:
        """
        Direct-assignment DN reconstruction loss — no Hungarian matching.

        For each DN query slot (group g, GT index j), we know it was initialised
        from GT box j. The loss supervises the decoder output at that slot to
        reconstruct the original GT box and class.

        Layout of dn_out queries per image:
          [group_0: gt_0 .. gt_{max_gt-1} | group_1: ... | group_G: ...]
          Slots beyond n_gt_b within each group are padding (ignored by valid_mask).
        """
        dev    = dn_out.boxes_all.device
        B      = len(targets)
        K_dn   = dn_meta.dn_num_queries
        G      = dn_meta.dn_num_groups
        max_gt = K_dn // max(G, 1)
        L      = dn_out.boxes_all.shape[0]

        zero = dn_out.boxes_all.sum() * 0.0

        if K_dn == 0 or G == 0:
            return {'total': zero, 'l1': zero, 'cls': zero}

        # Build (B, K_dn) target boxes / labels and validity mask
        gt_boxes_tgt  = torch.zeros(B, K_dn, 4,                  device=dev)
        gt_labels_tgt = torch.zeros(B, K_dn, dtype=torch.long,   device=dev)
        valid_mask    = torch.zeros(B, K_dn, dtype=torch.bool,    device=dev)

        for b in range(B):
            n_gt = int(dn_meta.batch_sizes[b].item())
            if n_gt == 0:
                continue
            gt_b   = dn_meta.gt_boxes[b][:n_gt].to(dev)    # (n_gt, 4)
            lbl_b  = dn_meta.gt_labels[b][:n_gt].to(dev)   # (n_gt,)
            for g in range(G):
                s, e = g * max_gt, g * max_gt + n_gt
                gt_boxes_tgt[b, s:e]  = gt_b
                gt_labels_tgt[b, s:e] = lbl_b
                valid_mask[b, s:e]    = True

        # Expand valid_mask across all decoder layers: (L, B, K_dn)
        valid_L = valid_mask.unsqueeze(0).expand(L, -1, -1)

        # Gather valid predictions and targets
        pred_boxes  = dn_out.boxes_all[valid_L]             # (N_valid, 4)
        tgt_boxes   = gt_boxes_tgt.unsqueeze(0).expand(
                          L, -1, -1, -1)[valid_L]           # (N_valid, 4)
        pred_logits = dn_out.logits_all[valid_L]            # (N_valid, C)
        gt_labels_L = gt_labels_tgt.unsqueeze(0).expand(
                          L, -1, -1)[valid_L]               # (N_valid,)

        if pred_boxes.shape[0] == 0:
            return {'total': zero, 'l1': zero, 'cls': zero}

        # Box loss: L1 on normalised cxcywh
        loss_l1  = F.l1_loss(pred_boxes, tgt_boxes)

        # Class loss: sigmoid focal (one-hot target)
        tgt_cls  = torch.zeros_like(pred_logits)
        tgt_cls.scatter_(1, gt_labels_L.unsqueeze(1), 1.0)
        loss_cls = _sigmoid_focal_loss(pred_logits, tgt_cls)

        total = self.lambda_dn_l1 * loss_l1 + self.lambda_dn_cls * loss_cls
        return {'total': total, 'l1': loss_l1, 'cls': loss_cls}

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        outputs: Dict[str, Any],
        batch:   dict,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        dev = outputs['stage2'].logits.device
        targets = [
            {k: v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in batch['targets']
        ]

        # Scorer focal loss
        l_score = self._scorer_loss(outputs['score_map'], batch)

        # Stage-2 DETR loss
        s2 = self._stage2_loss(outputs['stage2'], targets)

        total = l_score + s2['total']

        loss_stats: Dict[str, Tensor] = {
            'loss':       total,
            'loss_score': l_score,
            'loss_s2':    s2['total'],
            'loss_cls':   s2['cls'],
            'loss_bbox':  s2['bbox'],
            'loss_ciou':  s2['ciou'],
        }

        # Gumbel temperature monitoring
        if 'tau_query' in outputs:
            loss_stats['tau_query'] = outputs['tau_query'].detach()

        # ReID loss
        if self.reid_classifier is not None and 'ids' in targets[0]:
            l_reid = self._reid_loss(outputs['stage2'], targets, s2['indices'])
            total  = total + self.lambda_reid * l_reid
            loss_stats['loss_reid'] = l_reid

        # DN reconstruction loss
        if outputs.get('dn_out') is not None and outputs.get('dn_meta') is not None:
            l_dn = self._dn_loss(outputs['dn_out'], outputs['dn_meta'], targets)
            total = total + l_dn['total']
            loss_stats['loss_dn']     = l_dn['total']
            loss_stats['loss_dn_l1']  = l_dn['l1']
            loss_stats['loss_dn_cls'] = l_dn['cls']

        loss_stats['loss'] = total
        return total, loss_stats
