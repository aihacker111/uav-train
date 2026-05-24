"""
HybridLoss: combined Stage-1 (CenterNet) + Stage-2 (DETR + ReID) loss.

Stage 1 — CenterNet:
  L_s1 = L_focal(hm) + λ_wh * L_L1(wh) + λ_reg * L_L1(reg)

Stage 2 — DETR (Hungarian matching, applied per decoder layer):
  L_s2 = L_varifocal(cls) + λ_bbox * L_L1(box) + λ_ciou * L_CIoU(box)
        + λ_reid * (L_CE + L_triplet)(reid)  [if reid_classifier is set]
        + λ_consist * L_consist(Stage-1 hm ↔ GT centers of Stage-2 matched objects)

Combined loss with curriculum:
  L = eff_λ_s1 * L_s1 + eff_λ_s2 * L_s2

Curriculum schedule (t = epoch / total_epochs ∈ [0, 1]):
  eff_λ_s1: lambda_stage1 → 1.0   (decays: Stage-1 dominant early, balanced late)
  eff_λ_s2: lambda_stage2 → lambda_stage1  (rises: Stage-2 starts very low, climbs)

  Rationale: Stage-2 decoder depends on Stage-1 heatmap proposals. Starting Stage-2
  at a low weight (lambda_stage2=0.3) lets Stage-1 converge first — because feats are
  detached, Stage-1 only updates cn_upsample + cn_head and needs early emphasis to
  reach a useful heatmap before Stage-2 begins relying on those proposals.

Consistency loss — improved vs v1:
  Forces Stage-1 heatmap to have a peak at the GT center of every object that
  Stage-2 successfully matched. Weighted by Stage-2 match IoU quality: objects with
  higher IoU matches get stronger consistency supervision, because those are the objects
  Stage-2 is already confident about and Stage-1 should stay aligned with.
  Ramped from 0 → lambda_consist over consist_warmup_epochs to avoid corrupting
  Stage-1 with noisy Stage-2 matching at epoch 0.

Auxiliary losses from intermediate decoder layers use progressive weights:
  layer i weight = 0.4 + 0.6 * i / (num_aux_layers - 1)   range [0.4, 1.0]
  (shallower layers get smaller weight since their predictions are less refined)
"""
from __future__ import annotations

import math
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou
from ..networks.ecdet_uav.heads import CenterNetOutput, DETROutput
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
    pred:    Tensor,   # (*, C) raw logits
    q:       Tensor,   # (*, C) quality targets — IoU score for positives, 0 for negatives
    alpha:   float = 0.75,
    gamma:   float = 2.0,
) -> Tensor:
    """
    Varifocal Loss (VarifocalNet, Zhang et al. 2021).
    Uses IoU-quality score q as positive weight instead of fixed alpha.
      q > 0: -q * (q*log(p) + (1-q)*log(1-p))
      q = 0: -alpha * p^gamma * log(1-p)
    Normalised by number of positives.
    """
    p    = pred.sigmoid()
    ce   = F.binary_cross_entropy_with_logits(pred, q.clamp(0, 1), reduction='none')

    pos_mask = (q > 0).float()
    neg_mask = 1.0 - pos_mask

    loss = (pos_mask * q + neg_mask * alpha * p.pow(gamma)) * ce
    n_pos = pos_mask.sum().clamp(min=1)
    return loss.sum() / n_pos


def _paired_iou(b1: Tensor, b2: Tensor) -> Tensor:
    """Element-wise IoU for N matched pairs. b1, b2: (N, 4) xyxy. Returns (N,)."""
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1    = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2    = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    union = a1 + a2 - inter
    return inter / (union + 1e-7)


def ciou_loss(pred_xyxy: Tensor, tgt_xyxy: Tensor,
              iou: Tensor | None = None) -> Tensor:
    """CIoU loss for N matched pairs. pred_xyxy, tgt_xyxy: (N, 4) xyxy."""
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
    Combined CenterNet + DETR loss with curriculum weighting and consistency.

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
        num_classes:           int   = 7,
        lambda_wh:             float = 0.1,
        lambda_reg:            float = 1.0,
        lambda_bbox:           float = 2.0,
        lambda_ciou:           float = 2.0,
        lambda_reid:           float = 1.0,
        lambda_triplet:        float = 0.5,
        lambda_consist:        float = 0.05,
        consist_warmup_epochs: int   = 5,
        lambda_stage1:         float = 2.0,
        lambda_stage2:         float = 0.3,   # starts low — Stage-2 learns after Stage-1
        aux_loss:              bool  = True,
        reid_classifier:       Optional[nn.Linear] = None,
        total_epochs:          int   = 0,      # 0 = disable curriculum (static weights)
    ) -> None:
        super().__init__()
        self.num_classes           = num_classes
        self.lambda_wh             = lambda_wh
        self.lambda_reg            = lambda_reg
        self.lambda_bbox           = lambda_bbox
        self.lambda_ciou           = lambda_ciou
        self.lambda_reid           = lambda_reid
        self.lambda_triplet        = lambda_triplet
        self.lambda_consist        = lambda_consist
        self.consist_warmup_epochs = consist_warmup_epochs
        self.lambda_stage1         = lambda_stage1
        self.lambda_stage2         = lambda_stage2
        self.aux_loss              = aux_loss
        self.total_epochs          = total_epochs
        self._epoch                = 0

        self.reid_classifier = reid_classifier
        self.triplet_loss    = TripletLoss(margin=0.3)
        self.matcher         = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    def set_epoch(self, epoch: int) -> None:
        """Call at start of each training epoch to update curriculum + warmup state."""
        self._epoch = epoch

    # ── Curriculum ─────────────────────────────────────────────────────────────

    def _effective_stage_weights(self) -> tuple[float, float]:
        """
        Dynamic per-epoch stage weights (cosine schedule).

        t=0 → eff_s1=lambda_stage1 (2.0), eff_s2=lambda_stage2 (0.3)
          Stage-1 gets 6.7x the weight of Stage-2: cn_head converges fast,
          providing good proposals before the decoder depends on them.

        t=1 → eff_s1=1.0, eff_s2=lambda_stage1 (2.0)
          Roles reverse: Stage-2 decoder is now accurate and takes the lead.

        The cosine interpolation avoids abrupt transitions at the endpoints.
        Falls back to (lambda_stage1, lambda_stage2) when total_epochs == 0.
        """
        if self.total_epochs <= 0:
            return self.lambda_stage1, self.lambda_stage2

        t        = min(1.0, self._epoch / self.total_epochs)
        smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))

        eff_s1 = self.lambda_stage1 + smooth_t * (1.0               - self.lambda_stage1)
        eff_s2 = self.lambda_stage2 + smooth_t * (self.lambda_stage1 - self.lambda_stage2)
        return eff_s1, eff_s2

    # ── Stage-1 loss ───────────────────────────────────────────────────────────

    def _stage1_loss(self, cn_out: CenterNetOutput, batch: dict) -> Dict[str, Tensor]:
        dev = cn_out.hm.device

        hm       = batch['hm'].to(dev,       non_blocking=True)
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

        total = loss_hm + self.lambda_wh * loss_wh + self.lambda_reg * loss_reg
        return {'total': total, 'hm': loss_hm, 'wh': loss_wh, 'reg': loss_reg}

    # ── Stage-2 loss ───────────────────────────────────────────────────────────

    def _detr_layer_loss(
        self,
        logits:  Tensor,
        boxes:   Tensor,
        targets: List[dict],
        indices: list,
    ) -> Dict[str, Tensor]:
        dev   = logits.device
        Q_MIN = 0.1
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
            src_b    = torch.cat(src_b_list)
            tgt_b    = torch.cat(tgt_b_list)
            iou_all  = torch.cat(iou_list)
            n_m      = src_b.shape[0]
            loss_bbox = F.smooth_l1_loss(src_b, tgt_b, beta=0.05, reduction='sum') / n_m
            loss_ciou = ciou_loss(
                box_cxcywh_to_xyxy(src_b),
                box_cxcywh_to_xyxy(tgt_b),
                iou=iou_all,
            )
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
                aux = self._detr_layer_loss(
                    detr_out.logits_all[layer], detr_out.boxes_all[layer], targets, indices,
                )
                # Progressive aux weights: [0.4, 1.0] over layers.
                # Wider range than before (was [0.25, 0.5]) so intermediate layers
                # receive stronger supervision while still below the final layer.
                aux_w = 0.4 + 0.6 * (layer / max(n_aux - 1, 1))
                total = total + aux_w * aux['total']

        return {'total': total, 'cls': d['cls'], 'bbox': d['bbox'], 'ciou': d['ciou'],
                'indices': indices}

    # ── Consistency loss ───────────────────────────────────────────────────────

    def _consistency_loss(
        self,
        cn_out:  CenterNetOutput,
        targets: List[dict],
        indices: list,
    ) -> Tensor:
        """
        Align Stage-1 heatmap with Stage-2 matched objects.

        For every object that Stage-2 Hungarian-matched, the Stage-1 heatmap value
        at that object's GT center should be 1.0. Uses GT centers (not Stage-2
        predicted centers) for a stable signal independent of decoder quality.

        Improvement over v1: each object's BCE is weighted by its Stage-2 match IoU.
          - High IoU (e.g. 0.7): Stage-2 is confident → strong consistency push.
          - Low IoU (e.g. 0.1): Stage-2 barely matched → weak push to avoid
            Stage-1 learning from a noisy assignment.
        Only Stage-1 heatmap head receives gradient (no path to Stage-2 decoder).
        """
        hm = cn_out.hm                        # (B, C, H, W) — sigmoid, detached upstream
        B, C, H, W = hm.shape
        dev = hm.device
        Q_MIN = 0.1

        total    = hm.sum() * 0.0
        n_images = 0

        for b, m in enumerate(indices):
            src_i, tgt_i, iou_cached = m['src_i'], m['tgt_i'], m['iou']
            if len(src_i) == 0:
                continue

            ti      = tgt_i.to(dev)
            quality = iou_cached.to(dev).clamp(min=Q_MIN)   # (n,) IoU-based weight

            gt_boxes = targets[b]['boxes'][ti]               # (n, 4) cxcywh normalised
            cx       = gt_boxes[:, 0].clamp(0, 1)
            cy       = gt_boxes[:, 1].clamp(0, 1)
            cls_idx  = targets[b]['labels'][ti]              # (n,) long

            x_hm = (cx * (W - 1)).long().clamp(0, W - 1)
            y_hm = (cy * (H - 1)).long().clamp(0, H - 1)

            # Sample heatmap at GT centers for each object's class channel
            hm_vals = hm[b, cls_idx, y_hm, x_hm]           # (n,)

            # IoU-quality-weighted BCE pushing heatmap → 1.0 at matched centers
            bce = F.binary_cross_entropy(
                hm_vals.clamp(1e-6, 1 - 1e-6),
                torch.ones_like(hm_vals),
                reduction='none',
            )
            total    = total + (quality * bce).sum() / quality.sum().clamp(min=1)
            n_images += 1

        return total / max(n_images, 1)

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

        s1 = self._stage1_loss(outputs['stage1'], batch)
        s2 = self._stage2_loss(outputs['stage2'], targets)

        eff_s1, eff_s2 = self._effective_stage_weights()
        total = eff_s1 * s1['total'] + eff_s2 * s2['total']

        loss_stats: Dict[str, Tensor] = {
            'loss':      total,
            'loss_s1':   s1['total'],
            'loss_hm':   s1['hm'],
            'loss_wh':   s1['wh'],
            'loss_reg':  s1['reg'],
            'loss_s2':   s2['total'],
            'loss_cls':  s2['cls'],
            'loss_bbox': s2['bbox'],
            'loss_ciou': s2['ciou'],
            'w_s1':      torch.tensor(eff_s1, device=dev),
            'w_s2':      torch.tensor(eff_s2, device=dev),
        }

        if self.reid_classifier is not None and 'ids' in targets[0]:
            l_reid = self._reid_loss(outputs['stage2'], targets, s2['indices'])
            total  = total + self.lambda_reid * l_reid
            loss_stats['loss_reid'] = l_reid

        # Consistency loss: ramped 0 → lambda_consist over consist_warmup_epochs.
        # Prevents noisy epoch-0 Stage-2 matching from corrupting Stage-1 peaks,
        # while still establishing alignment once Stage-2 stabilises.
        consist_scale = min(1.0, self._epoch / max(1, self.consist_warmup_epochs))
        l_consist = self._consistency_loss(outputs['stage1'], targets, s2['indices'])
        total     = total + self.lambda_consist * consist_scale * l_consist
        loss_stats['loss_consist'] = l_consist

        loss_stats['loss'] = total
        return total, loss_stats
