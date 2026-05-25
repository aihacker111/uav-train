"""
HybridLoss: pure DETR loss for HybridECDet.

Stage 2 — DETR (Hungarian matching, applied per decoder layer):
  L = L_varifocal(cls) + λ_bbox * L_L1(box) + λ_ciou * L_CIoU(box)
    + λ_reid * (L_CE + L_triplet)(reid)  [if reid_classifier is set]

Auxiliary losses from intermediate decoder layers use progressive weights:
  layer i weight = 0.4 + 0.6 * i / (num_aux_layers - 1)   range [0.4, 1.0]
  (shallower layers get smaller weight since their predictions are less refined)

  The combined detection loss (final + aux) is then normalised by the sum of all
  layer weights (≈ 4.5 for 6 decoder layers), so L represents the average
  per-decoder-layer loss.

DN (denoising) loss — key for fast DETR convergence:
  λ_dn ramped from 0.5 → 1.0 over dn_warmup_epochs to let the main detection
  loss establish a stable gradient before DN contributes at full weight.
"""
from __future__ import annotations

import math
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou
from ..networks.ecdet_uav.heads import DETROutput
from ..base_losses import TripletLoss


# ── Loss primitives ────────────────────────────────────────────────────────────

def centernet_focal_loss(pred_hm: Tensor, gt_hm: Tensor) -> Tensor:
    """CenterNet focal loss — kept for centernet_aux_loss.py backward compat."""
    pos_mask = (gt_hm == 1).float()
    neg_mask = 1.0 - pos_mask
    p     = pred_hm.clamp(1e-6, 1.0 - 1e-6)
    pos_l = -((1 - p) ** 2) * p.log() * pos_mask
    neg_l = -((1 - gt_hm) ** 4) * (p ** 2) * (1 - p).log() * neg_mask
    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_l + neg_l).sum() / n_pos


def _gather_at_ind(feat: Tensor, ind: Tensor) -> Tensor:
    """Kept for centernet_aux_loss.py backward compat."""
    B, C, H, W = feat.shape
    flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
    idx  = ind.unsqueeze(-1).expand(B, ind.shape[1], C)
    return flat.gather(1, idx)


def mal_loss(
    pred:      Tensor,
    target_q:  Tensor,
    gamma:     float = 1.5,
    num_boxes: int   = 1,
) -> Tensor:
    """
    Modulation Augmented Loss — EdgeCrafter / D-FINE.
    target_q: iou^gamma for positives, 0 for negatives.
    """
    p      = pred.sigmoid().detach()
    is_pos = (target_q > 0).float()
    weight = p.pow(gamma) * (1.0 - is_pos) + is_pos
    loss   = F.binary_cross_entropy_with_logits(pred, target_q, weight=weight, reduction='none')
    return loss.mean(1).sum() * pred.shape[1] / max(num_boxes, 1)


def _paired_iou(b1: Tensor, b2: Tensor) -> Tensor:
    """Element-wise IoU for N matched pairs. b1, b2: (N, 4) xyxy."""
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1    = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2    = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    union = a1 + a2 - inter
    return inter / (union + 1e-7)


def giou_loss(pred_xyxy: Tensor, tgt_xyxy: Tensor,
              iou: Tensor | None = None) -> Tensor:
    """GIoU loss for N matched pairs."""
    if iou is None:
        iou = _paired_iou(pred_xyxy, tgt_xyxy)
    enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    enc    = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)
    inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    inter  = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(0)
    tgt_area  = (tgt_xyxy[:, 2]  - tgt_xyxy[:, 0]).clamp(0)  * (tgt_xyxy[:, 3]  - tgt_xyxy[:, 1]).clamp(0)
    union  = pred_area + tgt_area - inter
    return (1 - iou + (enc - union) / (enc + 1e-7)).mean()


# ── HybridLoss ─────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Pure DETR loss for HybridECDet.

    Batch dict must contain:
      'targets': List[dict] with keys 'labels' (N,) and 'boxes' (N, 4) cxcywh
      'ids'    : (B, max_obj) — optional, track ID for ReID (−1 = ignore)
    """

    def __init__(
        self,
        num_classes:      int   = 7,
        lambda_bbox:      float = 5.0,
        lambda_ciou:      float = 2.0,
        lambda_reid:      float = 1.0,
        lambda_triplet:   float = 0.5,
        lambda_dn:        float = 1.0,
        dn_warmup_epochs: int   = 10,
        lambda_enc_aux:   float = 1.0,   # enc_score_head supervision weight
        mal_gamma:        float = 1.5,
        aux_loss:         bool  = True,
        reid_classifier:  Optional[nn.Linear] = None,
        # kept for call-site backward compat — silently ignored
        lambda_wh:             float = 0.0,
        lambda_reg:            float = 0.0,
        lambda_consist:        float = 0.0,
        consist_warmup_epochs: int   = 0,
        lambda_stage1:         float = 0.0,
        lambda_stage2:         float = 0.0,
        total_epochs:          int   = 0,
    ) -> None:
        super().__init__()
        self.num_classes      = num_classes
        self.lambda_bbox      = lambda_bbox
        self.lambda_ciou      = lambda_ciou
        self.lambda_reid      = lambda_reid
        self.lambda_triplet   = lambda_triplet
        self.lambda_dn        = lambda_dn
        self.dn_warmup_epochs = dn_warmup_epochs
        self.lambda_enc_aux   = lambda_enc_aux
        self.mal_gamma        = mal_gamma
        self.aux_loss         = aux_loss
        self._epoch           = 0

        self.reid_classifier = reid_classifier
        self.triplet_loss    = TripletLoss(margin=0.3)
        self.matcher         = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    def set_epoch(self, epoch: int) -> None:
        """Track epoch for DN warmup schedule."""
        self._epoch = epoch

    # ── DETR layer loss ────────────────────────────────────────────────────────

    def _detr_layer_loss(
        self,
        logits:    Tensor,
        boxes:     Tensor,
        targets:   List[dict],
        indices:   list,
        num_boxes: int = 0,
    ) -> Dict[str, Tensor]:
        dev = logits.device
        B, Q, C = logits.shape

        if num_boxes == 0:
            num_boxes = max(sum(len(t['labels']) for t in targets), 1)

        target_q   = torch.zeros(B, Q, C, device=dev)
        src_b_list, tgt_b_list = [], []

        for b, m in enumerate(indices):
            src_i, tgt_i = m['src_i'], m['tgt_i']
            if not len(src_i):
                continue
            si  = src_i.to(dev)
            ti  = tgt_i.to(dev)

            if 'iou' in m and len(m['iou']):
                iou = m['iou'].to(dev)
            else:
                sb  = boxes[b][si].detach()
                tb  = targets[b]['boxes'][ti].to(dev)
                iou = _paired_iou(box_cxcywh_to_xyxy(sb), box_cxcywh_to_xyxy(tb))

            cls_idx = targets[b]['labels'][ti].to(dev)
            target_q[b, si, cls_idx] = iou.pow(self.mal_gamma).clamp(min=1e-4)

            src_b_list.append(boxes[b][si])
            tgt_b_list.append(targets[b]['boxes'][ti].to(dev))

        loss_cls = mal_loss(logits, target_q, gamma=self.mal_gamma, num_boxes=num_boxes)

        if src_b_list:
            src_b     = torch.cat(src_b_list)
            tgt_b     = torch.cat(tgt_b_list)
            n_m       = src_b.shape[0]
            loss_bbox = F.l1_loss(src_b, tgt_b, reduction='sum') / n_m
            loss_giou = giou_loss(box_cxcywh_to_xyxy(src_b), box_cxcywh_to_xyxy(tgt_b))
        else:
            loss_bbox = loss_giou = logits.sum() * 0.0

        total = loss_cls + self.lambda_bbox * loss_bbox + self.lambda_ciou * loss_giou
        return {'total': total, 'cls': loss_cls, 'bbox': loss_bbox, 'giou': loss_giou}

    def _stage2_loss(
        self,
        detr_out: DETROutput,
        targets:  List[dict],
    ) -> Dict[str, Tensor]:
        num_boxes = max(sum(len(t['labels']) for t in targets), 1)
        indices   = self.matcher(detr_out.logits, detr_out.boxes, targets)
        d         = self._detr_layer_loss(detr_out.logits, detr_out.boxes, targets, indices,
                                          num_boxes=num_boxes)
        layer_weight_sum = 1.0
        total = d['total']

        if self.aux_loss:
            L     = detr_out.boxes_all.shape[0]
            n_aux = L - 1
            for layer in range(n_aux):
                aux   = self._detr_layer_loss(
                    detr_out.logits_all[layer], detr_out.boxes_all[layer],
                    targets, indices, num_boxes=num_boxes,
                )
                aux_w = 0.4 + 0.6 * (layer / max(n_aux - 1, 1))
                total = total + aux_w * aux['total']
                layer_weight_sum += aux_w

        total = total / layer_weight_sum

        if detr_out.dn_outputs is not None and detr_out.dn_meta is not None:
            l_dn     = self._dn_loss(detr_out.dn_outputs, targets, detr_out.dn_meta,
                                     num_boxes=num_boxes)
            dn_scale = min(1.0, 0.5 + 0.5 * self._epoch / max(1, self.dn_warmup_epochs))
            total    = total + dn_scale * self.lambda_dn * l_dn

        # enc_score_head supervision: forces encoder to score object positions highly
        # so _select_topk reliably picks S4 positions for small objects.
        # NOTE: enc_aux logits have Q_enc = num_queries (top-K only), while the main
        # decoder may have extra fallback queries → Q_dec > Q_enc.  Filter indices so
        # only src_i < Q_enc are supervised; fallback-query matches have no enc anchor.
        l_enc_aux = None
        if detr_out.enc_aux_outputs is not None:
            l_enc_aux = detr_out.logits.sum() * 0.0
            for enc_out in detr_out.enc_aux_outputs:
                Q_enc = enc_out['pred_logits'].shape[1]
                enc_indices = [
                    {
                        'src_i': m['src_i'][m['src_i'] < Q_enc],
                        'tgt_i': m['tgt_i'][m['src_i'] < Q_enc],
                    }
                    for m in indices
                ]
                d_enc = self._detr_layer_loss(
                    enc_out['pred_logits'], enc_out['pred_boxes'],
                    targets, enc_indices, num_boxes=num_boxes,
                )
                l_enc_aux = l_enc_aux + d_enc['total']
            total = total + self.lambda_enc_aux * l_enc_aux

        return {'total': total, 'cls': d['cls'], 'bbox': d['bbox'], 'giou': d['giou'],
                'enc_aux': l_enc_aux, 'indices': indices}

    def _dn_loss(
        self,
        dn_outputs: list,
        targets:    List[dict],
        dn_meta:    dict,
        num_boxes:  int = 1,
    ) -> Tensor:
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group    = dn_meta['dn_num_group']

        dn_indices = []
        for b, pos_idx in enumerate(dn_positive_idx):
            num_gt = len(targets[b]['labels'])
            if num_gt > 0 and len(pos_idx) > 0:
                gt_idx = torch.arange(num_gt, device=pos_idx.device).tile(dn_num_group)
                dn_indices.append({'src_i': pos_idx, 'tgt_i': gt_idx})
            else:
                dev = targets[b]['labels'].device
                dn_indices.append({
                    'src_i': torch.zeros(0, dtype=torch.long, device=dev),
                    'tgt_i': torch.zeros(0, dtype=torch.long, device=dev),
                })

        dn_num_boxes = max(num_boxes * dn_num_group, 1)
        total_dn     = None
        for layer_out in dn_outputs:
            d = self._detr_layer_loss(
                layer_out['pred_logits'], layer_out['pred_boxes'],
                targets, dn_indices, num_boxes=dn_num_boxes,
            )
            total_dn = d['total'] if total_dn is None else total_dn + d['total']

        return (total_dn / len(dn_outputs)) if total_dn is not None else \
               dn_outputs[0]['pred_logits'].sum() * 0.0

    # ── ReID loss ──────────────────────────────────────────────────────────────

    def _reid_loss(
        self,
        detr_out: DETROutput,
        targets:  List[dict],
        indices:  list,
    ) -> Tensor:
        if detr_out.reid is None:
            return detr_out.logits.sum() * 0.0

        valid_emb, valid_ids = [], []
        dev = detr_out.reid.device

        for b, m in enumerate(indices):
            src_i, tgt_i = m['src_i'], m['tgt_i']
            if len(src_i) == 0 or 'ids' not in targets[b]:
                continue

            src_i     = src_i.to(dev)
            tgt_i     = tgt_i.to(dev)
            track_ids = targets[b]['ids'][tgt_i]
            keep      = track_ids >= 0
            if not keep.any():
                continue

            valid_emb.append(detr_out.reid[b][src_i[keep]])
            valid_ids.append(track_ids[keep])

        if not valid_emb:
            return detr_out.reid.sum() * 0.0

        emb    = torch.cat(valid_emb)
        ids_t  = torch.cat(valid_ids).to(emb.device)
        logits = self.reid_classifier(emb)

        loss_ce  = F.cross_entropy(logits, ids_t)
        unique_ids = ids_t.unique()
        loss_tri = (
            self.triplet_loss(emb, ids_t)
            if unique_ids.numel() >= 2 and emb.shape[0] >= 2
            else emb.sum() * 0.0
        )
        return loss_ce + self.lambda_triplet * loss_tri

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

        s2 = self._stage2_loss(outputs['stage2'], targets)
        total = s2['total']

        loss_stats: Dict[str, Tensor] = {
            'loss':      total,
            'loss_cls':  s2['cls'],
            'loss_bbox': s2['bbox'],
            'loss_giou': s2['giou'],
        }
        loss_stats['loss_enc_aux'] = s2['enc_aux'] if s2['enc_aux'] is not None \
                                     else total.detach() * 0.0

        if self.reid_classifier is not None and 'ids' in targets[0]:
            l_reid = self._reid_loss(outputs['stage2'], targets, s2['indices'])
            total  = total + self.lambda_reid * l_reid
            loss_stats['loss_reid'] = l_reid

        loss_stats['loss'] = total
        return total, loss_stats
