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

  The combined detection loss (final + aux) is then normalised by the sum of all
  layer weights (≈ 4.5 for 6 decoder layers), so L_s2 represents the average
  per-decoder-layer loss.  Without this, adding aux layers multiplies the raw
  loss value by ~4.5×, inflating init loss and making lambda_stage2 hard to tune.
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


def mal_loss(
    pred:      Tensor,   # (B, Q, C) raw logits
    target_q:  Tensor,   # (B, Q, C) iou^gamma at matched positions, 0 elsewhere
    gamma:     float = 1.5,
    num_boxes: int   = 1,
) -> Tensor:
    """
    Modulation Augmented Loss — EdgeCrafter / D-FINE.
    target_q: iou^gamma for positives (no alpha dampening), 0 for negatives.
    weight  = pred_score^gamma * (1 - is_pos) + is_pos
    Normalised by num_boxes (total GT count). Same convention as EdgeCrafter.
    Replaces VFL: MAL omits alpha dampening on negatives → cleaner gradient for
    small objects where focal downweighting hurts.
    """
    p      = pred.sigmoid().detach()
    is_pos = (target_q > 0).float()
    weight = p.pow(gamma) * (1.0 - is_pos) + is_pos
    loss   = F.binary_cross_entropy_with_logits(pred, target_q, weight=weight, reduction='none')
    return loss.mean(1).sum() * pred.shape[1] / max(num_boxes, 1)


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


def giou_loss(pred_xyxy: Tensor, tgt_xyxy: Tensor,
              iou: Tensor | None = None) -> Tensor:
    """GIoU loss for N matched pairs. pred_xyxy, tgt_xyxy: (N, 4) xyxy."""
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
        lambda_wh:             float = 0.4,
        lambda_reg:            float = 1.0,
        lambda_bbox:           float = 5.0,   # EdgeCrafter default (was 2.0)
        lambda_ciou:           float = 2.0,   # GIoU weight (name kept for compat)
        lambda_reid:           float = 1.0,
        lambda_triplet:        float = 0.5,
        lambda_consist:        float = 0.15,
        consist_warmup_epochs: int   = 5,
        lambda_stage1:         float = 2.0,
        lambda_stage2:         float = 0.3,
        lambda_dn:             float = 1.0,   # weight for DN auxiliary loss
        dn_warmup_epochs:      int   = 10,    # ramp λ_dn from 0.5→1.0 over first N epochs
        mal_gamma:             float = 1.5,   # MAL/EdgeCrafter exponent
        aux_loss:              bool  = True,
        reid_classifier:       Optional[nn.Linear] = None,
        total_epochs:          int   = 0,
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
        self.lambda_dn             = lambda_dn
        self.dn_warmup_epochs      = dn_warmup_epochs
        self.mal_gamma             = mal_gamma
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

        # Build MAL quality target: iou^gamma at matched (query, class) positions
        target_q   = torch.zeros(B, Q, C, device=dev)
        src_b_list, tgt_b_list = [], []

        for b, m in enumerate(indices):
            src_i, tgt_i = m['src_i'], m['tgt_i']
            if not len(src_i):
                continue
            si  = src_i.to(dev)
            ti  = tgt_i.to(dev)

            # Reuse cached IoU from matcher when present (avoids redundant compute).
            # Fall back to computing fresh (e.g. for DN indices).
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
        layer_weight_sum = 1.0   # weight of the final layer (implicit 1.0)
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

        # Normalize detection loss by the sum of layer weights so that s2 loss
        # represents the average per-decoder-layer loss, not the raw accumulation.
        # Without this, each additional aux layer multiplies the loss magnitude
        # (e.g. 6 layers × avg_weight ≈ 4.5× per-layer), inflating init loss ~4.5×
        # and making lambda_stage2 hard to tune independently of n_decoder_layers.
        total = total / layer_weight_sum

        # DN (denoising) loss — key for fast DETR convergence.
        # The decoder generates supervised DN outputs but they are wasted without
        # this loss. Adds ~2-3x convergence speedup (same as DINO / RT-DETR).
        if detr_out.dn_outputs is not None and detr_out.dn_meta is not None:
            l_dn  = self._dn_loss(detr_out.dn_outputs, targets, detr_out.dn_meta,
                                   num_boxes=num_boxes)
            # Ramp λ_dn from 0.5→1.0 over dn_warmup_epochs so Stage-1 proposals
            # have time to stabilise before DN dominates the decoder gradient.
            dn_scale = min(1.0, 0.5 + 0.5 * self._epoch / max(1, self.dn_warmup_epochs))
            total = total + dn_scale * self.lambda_dn * l_dn

        return {'total': total, 'cls': d['cls'], 'bbox': d['bbox'], 'giou': d['giou'],
                'indices': indices}

    def _dn_loss(
        self,
        dn_outputs: list,
        targets:    List[dict],
        dn_meta:    dict,
        num_boxes:  int = 1,
    ) -> Tensor:
        """Compute denoising auxiliary loss across all DN decoder layers."""
        dn_positive_idx = dn_meta['dn_positive_idx']   # list[Tensor] one per batch
        dn_num_group    = dn_meta['dn_num_group']

        # Build DN match indices (no Hungarian needed — DN construction fixes the mapping)
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

    # ── Consistency loss ───────────────────────────────────────────────────────

    def _consistency_loss(
        self,
        cn_out:   CenterNetOutput,
        targets:  List[dict],
        indices:  list,
        s2_boxes: Optional[Tensor] = None,  # (B, K, 4) cxcywh [0,1] — detached Stage-2 preds
    ) -> Tensor:
        """
        Align Stage-1 with Stage-2 matched objects via two signals:

        1. Heatmap BCE: push hm → 1.0 at GT centers of Stage-2-matched objects,
           weighted by match IoU (high confidence → stronger push).

        2. WH pseudo-label: use Stage-2's refined box w/h as a soft supervision
           target for Stage-1's wh head at GT center locations.
           Only applied when match IoU >= WH_IOU_THRESHOLD (0.4) to avoid noisy
           matches corrupting Stage-1's size estimates.
           Gradient flows only to cn_out.wh; s2_boxes must be passed detached.

        Both signals are one-directional: Stage-1 learns from Stage-2, never the
        reverse (no gradient path to the decoder).
        """
        hm  = cn_out.hm    # (B, C, H, W) — sigmoid
        wh  = cn_out.wh    # (B, 2, H, W) — S4 pixel units
        B, C, H, W = hm.shape
        dev = hm.device
        Q_MIN            = 0.1
        WH_IOU_THRESHOLD = 0.4
        WH_CONSIST_SCALE = 0.1  # relative weight of wh term vs hm term

        total_hm  = hm.sum() * 0.0
        total_wh  = wh.sum() * 0.0
        n_hm      = 0
        n_wh      = 0

        for b, m in enumerate(indices):
            src_i, tgt_i, iou_cached = m['src_i'], m['tgt_i'], m['iou']
            if len(src_i) == 0:
                continue

            ti      = tgt_i.to(dev)
            si      = src_i.to(dev)
            quality = iou_cached.to(dev).clamp(min=Q_MIN)   # (n,)

            gt_boxes = targets[b]['boxes'][ti]               # (n, 4) cxcywh normalised
            cx       = gt_boxes[:, 0].clamp(0, 1)
            cy       = gt_boxes[:, 1].clamp(0, 1)
            cls_idx  = targets[b]['labels'][ti]              # (n,) long

            x_hm = (cx * (W - 1)).long().clamp(0, W - 1)   # S4 col index
            y_hm = (cy * (H - 1)).long().clamp(0, H - 1)   # S4 row index

            # ── 1. Heatmap BCE ──────────────────────────────────────────────────
            hm_vals = hm[b, cls_idx, y_hm, x_hm]           # (n,)
            bce = F.binary_cross_entropy(
                hm_vals.clamp(1e-6, 1 - 1e-6),
                torch.ones_like(hm_vals),
                reduction='none',
            )
            total_hm = total_hm + (quality * bce).sum() / quality.sum().clamp(min=1)
            n_hm     += 1

            # ── 2. WH pseudo-label ──────────────────────────────────────────────
            if s2_boxes is not None:
                wh_mask = quality >= WH_IOU_THRESHOLD       # (n,) bool
                if wh_mask.any():
                    # Stage-2 normalised w/h → S4 pixel units (same convention as cn_out.wh)
                    s2_wh   = s2_boxes[b, si, 2:4][wh_mask]  # (m, 2) already detached
                    tgt_s4  = s2_wh * s2_wh.new_tensor([W, H])  # (m, 2) S4 pixels

                    # Stage-1 predicted wh at GT center grid cells
                    pred_s4 = wh[b, :, y_hm[wh_mask], x_hm[wh_mask]].T  # (m, 2)

                    q_w  = quality[wh_mask].unsqueeze(-1)    # (m, 1) quality weight
                    l_wh = F.smooth_l1_loss(pred_s4, tgt_s4, beta=1.0, reduction='none')
                    total_wh = total_wh + (q_w * l_wh).sum() / (q_w.sum().clamp(min=1e-6) * 2)
                    n_wh += 1

        loss_hm = total_hm / max(n_hm, 1)
        loss_wh = total_wh / max(n_wh, 1)
        return loss_hm + WH_CONSIST_SCALE * loss_wh

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
            'loss_giou': s2['giou'],
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
        l_consist = self._consistency_loss(
            outputs['stage1'], targets, s2['indices'],
            s2_boxes=outputs['stage2'].boxes.detach(),
        )
        total     = total + self.lambda_consist * consist_scale * l_consist
        loss_stats['loss_consist'] = l_consist

        loss_stats['loss'] = total
        return total, loss_stats
