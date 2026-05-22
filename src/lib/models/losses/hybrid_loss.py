"""
HybridLoss: Stage-1 (CenterNet) + Stage-2 (DEIMv2 main losses only).

Stage-2 losses — ported trực tiếp từ DEIMv2 DEIMCriterion:
  - VFL   : Varifocal Loss (IoU-quality weighted sigmoid focal)
  - L1    : bounding-box L1 regression
  - GIoU  : standard GIoU (không nhân IoU weight — đúng DEIMv2)
  - DN    : Denoising loss — VFL + L1 + GIoU trên DN queries

Không dùng: aux_outputs, enc_aux, pre_outputs.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from ..networks.deim_uav.heads import CenterNetOutput
from ..base_losses import TripletLoss


# ── CenterNet helpers ──────────────────────────────────────────────────────────

def centernet_focal_loss(pred_hm: Tensor, gt_hm: Tensor) -> Tensor:
    pos_mask = (gt_hm == 1).float()
    neg_mask = 1.0 - pos_mask
    p     = pred_hm.clamp(1e-6, 1.0 - 1e-6)
    pos_l = -((1 - p) ** 2) * p.log() * pos_mask
    neg_l = -((1 - gt_hm) ** 4) * (p ** 2) * (1 - p).log() * neg_mask
    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_l + neg_l).sum() / n_pos


def _gather_at_ind(feat: Tensor, ind: Tensor) -> Tensor:
    B, C, H, W = feat.shape
    flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
    idx  = ind.unsqueeze(-1).expand(B, ind.shape[1], C)
    return flat.gather(1, idx)


# ── HybridLoss ─────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Stage-1: CenterNet heatmap loss.
    Stage-2: DEIMv2 losses trên main decoder output + DN queries.
    """

    def __init__(
        self,
        num_classes:     int   = 7,
        lambda_vfl:      float = 1.0,
        lambda_bbox:     float = 5.0,
        lambda_giou:     float = 2.0,
        lambda_reid:     float = 1.0,
        lambda_triplet:  float = 0.5,
        lambda_cn:       float = 0.5,
        vfl_alpha:       float = 0.2,
        vfl_gamma:       float = 2.0,
        aux_loss:        bool  = True,  # kept for API compat, not used
        reid_classifier: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.num_classes    = num_classes
        self.lambda_vfl     = lambda_vfl
        self.lambda_bbox    = lambda_bbox
        self.lambda_giou    = lambda_giou
        self.lambda_reid    = lambda_reid
        self.lambda_triplet = lambda_triplet
        self.lambda_cn      = lambda_cn
        self.vfl_alpha      = vfl_alpha
        self.vfl_gamma      = vfl_gamma

        self.reid_classifier = reid_classifier
        self.triplet_loss    = TripletLoss(margin=0.3)
        self.matcher         = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    # ── Index helper ───────────────────────────────────────────────────────────

    def _src_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # ── VFL — port trực tiếp từ DEIMv2 loss_labels_vfl ───────────────────────
    # Ref: DEIMv2/engine/deim/deim_criterion.py :: loss_labels_vfl()

    def _loss_vfl(self, outputs, targets, indices, num_boxes, ious=None):
        idx        = self._src_idx(indices)
        src_logits = outputs['pred_logits']

        # Tính IoU nếu chưa có
        if ious is None:
            if idx[0].numel() > 0:
                src_boxes = outputs['pred_boxes'][idx]
                tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
                ious_mat, _ = box_iou(box_cxcywh_to_xyxy(src_boxes),
                                      box_cxcywh_to_xyxy(tgt_boxes))
                ious = torch.diag(ious_mat).detach()
            else:
                ious = src_logits.new_empty(0)

        tgt_cls_o = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        tgt_cls   = torch.full(src_logits.shape[:2], self.num_classes,
                               dtype=torch.int64, device=src_logits.device)
        if idx[0].numel() > 0:
            tgt_cls[idx] = tgt_cls_o

        target = F.one_hot(tgt_cls, num_classes=self.num_classes + 1)[..., :-1].float()

        tgt_score_o = torch.zeros(src_logits.shape[:2],
                                  dtype=src_logits.dtype, device=src_logits.device)
        if idx[0].numel() > 0:
            tgt_score_o[idx] = ious.to(tgt_score_o.dtype)
        tgt_score = tgt_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        weight     = self.vfl_alpha * pred_score.pow(self.vfl_gamma) * (1 - target) + tgt_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, tgt_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    # ── L1 + GIoU — port trực tiếp từ DEIMv2 loss_boxes ─────────────────────
    # Ref: DEIMv2/engine/deim/deim_criterion.py :: loss_boxes()
    # Khác biệt so với code cũ: GIoU KHÔNG nhân IoU weight (boxes_weight=None)

    def _loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._src_idx(indices)
        if idx[0].numel() == 0:
            z = outputs['pred_boxes'].sum() * 0.0
            return {'loss_bbox': z, 'loss_giou': z}

        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        # L1
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none').sum() / num_boxes

        # Standard GIoU — không nhân thêm IoU weight (đúng DEIMv2 mặc định)
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        ))
        loss_giou = loss_giou.sum() / num_boxes

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    # ── DN indices — port từ DEIMv2 get_cdn_matched_indices ───────────────────
    # Ref: DEIMv2/engine/deim/deim_criterion.py :: get_cdn_matched_indices()

    @staticmethod
    def _cdn_indices(dn_meta, targets):
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group    = dn_meta['dn_num_group']
        device          = targets[0]['labels'].device
        result = []
        for i, t in enumerate(targets):
            n = len(t['labels'])
            if n > 0:
                gt_idx = torch.arange(n, dtype=torch.int64, device=device).tile(dn_num_group)
                result.append((dn_positive_idx[i], gt_idx))
            else:
                result.append((torch.zeros(0, dtype=torch.int64, device=device),
                                torch.zeros(0, dtype=torch.int64, device=device)))
        return result

    # ── Stage-1 ────────────────────────────────────────────────────────────────

    def _stage1_loss(self, cn_out: CenterNetOutput, batch: dict) -> Dict[str, Tensor]:
        dev = cn_out.hm.device
        hm       = batch['hm'].to(dev,      non_blocking=True)
        reg_mask = batch['reg_mask'].to(dev, non_blocking=True)
        ind      = batch['ind'].to(dev,      non_blocking=True)
        gt_wh    = batch['wh'].to(dev,       non_blocking=True)
        gt_reg   = batch['reg'].to(dev,      non_blocking=True)

        loss_hm = centernet_focal_loss(cn_out.hm, hm)
        n       = reg_mask.sum().clamp(min=1).float()
        pred_wh  = _gather_at_ind(cn_out.wh,  ind)
        pred_reg = _gather_at_ind(cn_out.reg, ind)
        mask     = reg_mask.unsqueeze(-1).float()
        loss_wh  = (F.smooth_l1_loss(pred_wh,  gt_wh,  beta=1.0, reduction='none') * mask).sum() / n
        loss_reg = (F.smooth_l1_loss(pred_reg, gt_reg, beta=0.5, reduction='none') * mask).sum() / n

        total = loss_hm + 0.1 * loss_wh + loss_reg
        return {'total': total, 'hm': loss_hm, 'wh': loss_wh, 'reg': loss_reg}

    # ── ReID ──────────────────────────────────────────────────────────────────

    def _reid_loss(self, stage2: dict, targets: List[dict], indices) -> Tensor:
        if 'reid' not in stage2 or stage2['reid'] is None:
            return stage2['pred_logits'].sum() * 0.0

        reid = stage2['reid']
        dev  = reid.device
        valid_emb, valid_ids = [], []

        for b, (src_i, tgt_i) in enumerate(indices):
            if len(src_i) == 0 or 'ids' not in targets[b]:
                continue
            src_i     = src_i.to(dev)
            tgt_i     = tgt_i.to(dev)
            track_ids = targets[b]['ids'][tgt_i]
            keep      = track_ids >= 0
            if not keep.any():
                continue
            valid_emb.append(reid[b][src_i[keep]])
            valid_ids.append(track_ids[keep])

        if not valid_emb:
            return reid.sum() * 0.0

        emb     = torch.cat(valid_emb)
        ids_t   = torch.cat(valid_ids).to(dev)
        logits  = self.reid_classifier(emb)
        loss_ce = F.cross_entropy(logits, ids_t)

        if ids_t.unique().numel() >= 2 and emb.shape[0] >= 2:
            loss_tri = self.triplet_loss(emb, ids_t)
        else:
            loss_tri = emb.sum() * 0.0

        return loss_ce + self.lambda_triplet * loss_tri

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        outputs: Dict[str, Any],
        batch:   dict,
    ) -> tuple:
        stage2 = outputs['stage2']
        dev    = stage2['pred_logits'].device
        targets = [
            {k: v.to(dev, non_blocking=True) if isinstance(v, Tensor) else v
             for k, v in t.items()}
            for t in batch['targets']
        ]

        # ── Stage 1 ────────────────────────────────────────────────────────────
        s1 = self._stage1_loss(outputs['stage1'], batch)

        # ── Stage 2 ────────────────────────────────────────────────────────────
        num_boxes = float(max(1, sum(len(t['labels']) for t in targets)))

        # Hungarian matching trên main output (last decoder layer)
        main_out = {k: v for k, v in stage2.items()
                    if k not in ('aux_outputs', 'enc_aux_outputs', 'dn_outputs',
                                 'dn_pre_outputs', 'dn_meta', 'enc_meta', 'pre_outputs')}
        indices = self.matcher(main_out, targets)['indices']

        losses: Dict[str, Tensor] = {}

        # Main output: VFL + L1 + GIoU
        losses.update(self._loss_vfl(stage2, targets, indices, num_boxes))
        losses.update(self._loss_boxes(stage2, targets, indices, num_boxes))

        # ── DN loss — port từ DEIMv2 ────────────────────────────────────────────
        # Ref: DEIMv2/engine/deim/deim_criterion.py :: forward() phần 'dn_outputs'
        # DN queries dùng fixed matching (không cần Hungarian), loss = VFL + L1 + GIoU
        if 'dn_outputs' in stage2 and 'dn_meta' in stage2:
            dn_meta      = stage2['dn_meta']
            indices_dn   = self._cdn_indices(dn_meta, targets)
            dn_num_boxes = float(max(1, num_boxes * dn_meta['dn_num_group']))

            for i, dn_out in enumerate(stage2['dn_outputs']):
                dn_out = dict(dn_out)
                sfx = f'_dn_{i}'
                losses.update({k + sfx: v for k, v in
                    self._loss_vfl(dn_out, targets, indices_dn, dn_num_boxes).items()})
                losses.update({k + sfx: v for k, v in
                    self._loss_boxes(dn_out, targets, indices_dn, dn_num_boxes).items()})

            # dn_pre_outputs (first decoder layer, trước FDR) — chỉ VFL + L1 + GIoU
            if 'dn_pre_outputs' in stage2:
                dn_pre = stage2['dn_pre_outputs']
                losses.update({k + '_dn_pre': v for k, v in
                    self._loss_vfl(dn_pre, targets, indices_dn, dn_num_boxes).items()})
                losses.update({k + '_dn_pre': v for k, v in
                    self._loss_boxes(dn_pre, targets, indices_dn, dn_num_boxes).items()})

        # Weighted sum cho stage-2
        def _base_key(k):
            for sep in ('_dn_',):
                if sep in k:
                    return k[:k.index(sep)]
            return k[:-7] if k.endswith('_dn_pre') else k

        w = {
            'loss_vfl':  self.lambda_vfl,
            'loss_bbox': self.lambda_bbox,
            'loss_giou': self.lambda_giou,
        }
        loss_s2 = sum(v * w.get(_base_key(k), 0.0)
                      for k, v in losses.items() if _base_key(k) in w)

        loss_s1 = s1['total']
        total   = loss_s2 + self.lambda_cn * loss_s1

        # Optional ReID
        if self.reid_classifier is not None and 'ids' in targets[0]:
            l_reid = self._reid_loss(stage2, targets, indices)
            total  = total + self.lambda_reid * l_reid
            losses['loss_reid'] = l_reid

        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

        loss_stats = {
            'loss':      total,
            'loss_s1':   s1['total'].detach(),
            'loss_hm':   s1['hm'].detach(),
            'loss_wh':   s1['wh'].detach(),
            'loss_reg':  s1['reg'].detach(),
            'loss_s2':   loss_s2.detach(),
            'loss_vfl':  losses.get('loss_vfl',  total.new_tensor(0.0)).detach(),
            'loss_bbox': losses.get('loss_bbox', total.new_tensor(0.0)).detach(),
            'loss_giou': losses.get('loss_giou', total.new_tensor(0.0)).detach(),
        }
        if 'loss_reid' in losses:
            loss_stats['loss_reid'] = losses['loss_reid'].detach()

        return total, loss_stats