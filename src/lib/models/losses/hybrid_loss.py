"""
HybridLoss: pure DETR loss for HybridECDet.

Stage 2 — DETR (Hungarian matching, per decoder layer):
  L = L_mal(cls) + λ_bbox * L_L1(box) + λ_ciou * L_GIoU(box)

Heatmap branch (CenterNet on S4):
  L_hm  — CornerNet focal loss (Gaussian radius from IoU criterion, AMOT-style)
  L_wh  — NormRegL1Loss: relative error |pred/target − 1|, scale-invariant (AMOT)
  L_reg — SmoothL1 on sub-pixel offset (AMOT)

Detection vs ReID weighting — learnable uncertainty (Kendall et al. 2018, AMOT):
  total = 0.5 * (exp(−s_det) * det_loss + exp(−s_id) * reid_loss + s_det + s_id)
  s_det, s_id are learnable parameters; model auto-balances the two tasks.

DN (denoising) loss — key for fast DETR convergence:
  λ_dn ramped from 0.5 → 1.0 over dn_warmup_epochs.

Note: Logit Adjustment has been removed — MAL already handles class imbalance
via p^gamma modulation on negatives.
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


def _gaussian_radius(fw: float, fh: float, min_overlap: float = 0.7) -> float:
    """CornerNet IoU-based Gaussian radius — ported from AMOT/src/lib/utils/image.py.

    Returns the largest pixel radius r such that a Gaussian centred at the GT peak
    still has IoU ≥ min_overlap with the GT box when thresholded at that radius.
    Three corner-placement scenarios give three quadratics; r = min of all three.
    """
    a1  = 1.0;  b1 = fw + fh
    c1  = fw * fh * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))
    r1  = (b1 + sq1) / 2.0

    a2  = 4.0;  b2 = 2.0 * (fw + fh)
    c2  = (1.0 - min_overlap) * fw * fh
    sq2 = math.sqrt(max(0.0, b2 * b2 - 4.0 * a2 * c2))
    r2  = (b2 + sq2) / 2.0

    a3  = 4.0 * min_overlap
    b3  = -2.0 * min_overlap * (fw + fh)
    c3  = (min_overlap - 1.0) * fw * fh
    sq3 = math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))
    r3  = (b3 + sq3) / 2.0

    return min(r1, r2, r3)


def render_gaussian_heatmaps(
    targets: List[dict],
    num_classes: int,
    feat_h: int,
    feat_w: int,
) -> tuple:
    """
    Render CenterNet-style Gaussian heatmaps for a batch from normalised GT boxes.

    Args:
        targets:     List[dict] with 'boxes' (N,4) cxcywh [0,1] and 'labels' (N,).
        num_classes: Number of foreground classes.
        feat_h, feat_w: Feature map spatial dimensions (S4 resolution).

    Returns:
        gt_hm  (B, C, H, W)  — Gaussian heatmap peaked at GT centers.
        gt_wh  (B, 2, H, W)  — Width/height in feature-map pixel scale at GT cells.
        gt_reg (B, 2, H, W)  — Sub-pixel offset at GT cells.
        reg_mask (B, H, W)   — Binary mask: 1 at GT grid cells, 0 elsewhere.
    """
    B = len(targets)
    device = targets[0]['boxes'].device

    gt_hm    = torch.zeros(B, num_classes, feat_h, feat_w, device=device)
    gt_wh    = torch.zeros(B, 2,           feat_h, feat_w, device=device)
    gt_reg   = torch.zeros(B, 2,           feat_h, feat_w, device=device)
    reg_mask = torch.zeros(B,              feat_h, feat_w, device=device)

    gy = torch.arange(feat_h, dtype=torch.float32, device=device)
    gx = torch.arange(feat_w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)

    for b, t in enumerate(targets):
        boxes  = t['boxes']   # (N, 4) cxcywh normalised
        labels = t['labels']  # (N,) int
        N = len(boxes)
        if N == 0:
            continue

        fx  = boxes[:, 0] * feat_w   # (N,) float center-x in feature map
        fy  = boxes[:, 1] * feat_h   # (N,) float center-y
        fw  = boxes[:, 2] * feat_w   # (N,) float width  in feature-map pixels
        fh  = boxes[:, 3] * feat_h   # (N,) float height

        ix = fx.long().clamp(0, feat_w - 1)  # (N,) snapped integer col
        iy = fy.long().clamp(0, feat_h - 1)  # (N,) snapped integer row

        # Gaussian radius per object — _gaussian_radius is pure Python math,
        # must stay a loop; it is O(N) scalar work and not the bottleneck.
        sigmas = torch.tensor(
            [max(1.0, (2 * max(0, int(_gaussian_radius(fw[i].item(), fh[i].item()))) + 1) / 6.0)
             for i in range(N)],
            dtype=torch.float32, device=device,
        )  # (N,)

        # Vectorised gaussian: one exp() over (N, H, W) instead of N × (H, W).
        # Guarantees gt_hm[iy, ix] = 1.0 exactly (centered at integer cell).
        dx = grid_x.unsqueeze(0) - ix.float().view(N, 1, 1)   # (N, H, W)
        dy = grid_y.unsqueeze(0) - iy.float().view(N, 1, 1)   # (N, H, W)
        gaussians = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigmas.view(N, 1, 1) ** 2))

        # Per-class max — loop over C (10) not N, so at most 10 GPU calls.
        cls = labels.long()
        for c in range(num_classes):
            mask_c = (cls == c)
            if not mask_c.any():
                continue
            gt_hm[b, c] = torch.maximum(gt_hm[b, c], gaussians[mask_c].amax(dim=0))

        # wh / reg / mask — vectorised fancy indexing (no loop over N).
        gt_wh[b, 0, iy, ix]  = fw
        gt_wh[b, 1, iy, ix]  = fh
        gt_reg[b, 0, iy, ix] = fx - ix.float()
        gt_reg[b, 1, iy, ix] = fy - iy.float()
        reg_mask[b, iy, ix]  = 1.0

    return gt_hm, gt_wh, gt_reg, reg_mask


def mal_loss(
    pred:      Tensor,
    target_q:  Tensor,
    gamma:     float = 1.5,
    num_boxes: int   = 1,
) -> Tensor:
    """Modulation Augmented Loss — EdgeCrafter / D-FINE.
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
        mal_gamma:        float = 2.0,
        aux_loss:         bool  = True,
        reid_classifier:  Optional[nn.Linear] = None,
        # Heatmap (CenterNet) branch losses
        lambda_heatmap:   float = 1.0,   # focal loss on heatmap
        lambda_wh:        float = 0.1,   # L1 loss on wh (at GT cell positions)
        lambda_reg:       float = 1.0,   # L1 loss on sub-pixel offset
        # kept for call-site backward compat — silently ignored
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
        self.lambda_heatmap   = lambda_heatmap
        self.lambda_wh        = lambda_wh
        self.lambda_reg       = lambda_reg

        self.reid_classifier = reid_classifier
        self.triplet_loss    = TripletLoss(margin=0.3)
        self.matcher         = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

        # Learnable uncertainty weights — AMOT / Kendall et al. 2018.
        # s = log(σ²): higher s → lower effective weight exp(−s).
        # s = -log(2) ≈ -0.693 → exp(-s) = 2 → net weight = 2/2 = 1.0
        # This initialises the Kendall formula to the same loss scale as the
        # unweighted sum (det + reid), so initial loss ≈ 14–16 as expected.
        self.s_det = nn.Parameter(torch.tensor(-0.693))
        self.s_id  = nn.Parameter(torch.tensor(-0.693))

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

        d = self._detr_layer_loss(detr_out.logits, detr_out.boxes, targets, indices,
                                   num_boxes=num_boxes)
        layer_weight_sum = 1.0
        total = d['total']

        if self.aux_loss:
            L     = detr_out.boxes_all.shape[0]
            n_aux = L - 1
            for layer in range(n_aux):
                # Re-match per layer (as in the original EdgeCrafter criterion.py:344-345).
                # Early layers have different predictions than the final layer, so reusing
                # final-layer indices sends incorrect gradients to shallow decoder layers.
                aux_indices = self.matcher(
                    detr_out.logits_all[layer], detr_out.boxes_all[layer], targets
                )
                aux = self._detr_layer_loss(
                    detr_out.logits_all[layer], detr_out.boxes_all[layer],
                    targets, aux_indices, num_boxes=num_boxes,
                )
                total            = total + aux['total']   # equal weight per layer
                layer_weight_sum = layer_weight_sum + 1.0 # matches EdgeCrafter (no progressive ramp)

        total = total / layer_weight_sum  # average over all decoder layers

        if detr_out.dn_outputs is not None and detr_out.dn_meta is not None:
            l_dn     = self._dn_loss(detr_out.dn_outputs, targets, detr_out.dn_meta,
                                     num_boxes=num_boxes)
            dn_scale = min(1.0, 0.5 + 0.5 * self._epoch / max(1, self.dn_warmup_epochs))
            total    = total + dn_scale * self.lambda_dn * l_dn

        # enc_score_head supervision: forces encoder to score object positions highly
        # so _select_topk reliably picks S4 positions for small objects.
        # NOTE: enc_aux logits have Q_enc = num_queries (top-K only).
        l_enc_aux = None
        if detr_out.enc_aux_outputs is not None:
            l_enc_aux = detr_out.logits.sum() * 0.0
            n_enc = 0
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
                n_enc += 1
            l_enc_aux = l_enc_aux / max(n_enc, 1)  # average over enc heads, not raw sum
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

    # ── Heatmap branch loss ────────────────────────────────────────────────────

    def _heatmap_loss(
        self,
        hm_out,          # CenterNetOutput: .hm (B,C,H,W), .wh (B,2,H,W), .reg (B,2,H,W)
        targets: List[dict],
    ) -> Dict[str, Tensor]:
        """
        CenterNet-style heatmap loss on S4 features — aligned with AMOT.

        loss_hm  — CornerNet focal loss; Gaussian radius from IoU criterion.
        loss_wh  — NormRegL1Loss: |pred/target − 1|, scale-invariant (AMOT).
                   Avoids large objects dominating wh gradient over tiny UAV objects.
        loss_reg — SmoothL1 on sub-pixel offset (AMOT default).
        """
        B, C, H, W = hm_out.hm.shape

        gt_hm, gt_wh, gt_reg, reg_mask = render_gaussian_heatmaps(
            targets, self.num_classes, H, W
        )

        loss_hm = centernet_focal_loss(hm_out.hm, gt_hm)

        # wh and reg: only at positions where there is a GT object
        mask  = reg_mask.bool()                       # (B, H, W)
        n_pos = mask.sum().clamp(min=1).float()

        if mask.any():
            # (B,2,H,W) → gather at positive positions → (P, 2)
            pred_wh  = hm_out.wh.permute(0, 2, 3, 1)[mask]
            pred_reg = hm_out.reg.permute(0, 2, 3, 1)[mask]
            tgt_wh   = gt_wh.permute(0, 2, 3, 1)[mask]
            tgt_reg  = gt_reg.permute(0, 2, 3, 1)[mask]

            # NormRegL1Loss (AMOT): relative error — scale-invariant across object sizes.
            # Equivalent to: |pred_wh / target_wh − 1|, normalized by n_pos.
            wh_ratio = pred_wh / (tgt_wh + 1e-4)
            loss_wh  = F.l1_loss(wh_ratio, torch.ones_like(wh_ratio),
                                 reduction='sum') / n_pos

            # SmoothL1 for offset (AMOT default 'sl1').
            # Sub-pixel targets are in [0,1) so SmoothL1 ≈ L1 in practice,
            # but is more robust to the occasional noisy annotation.
            loss_reg = F.smooth_l1_loss(pred_reg, tgt_reg, reduction='sum') / n_pos
        else:
            loss_wh  = hm_out.wh.sum()  * 0.0
            loss_reg = hm_out.reg.sum() * 0.0

        total = (self.lambda_heatmap * loss_hm
                 + self.lambda_wh    * loss_wh
                 + self.lambda_reg   * loss_reg)

        return {'total': total, 'hm': loss_hm, 'wh': loss_wh, 'reg': loss_reg}

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
        det_total = s2['total']   # DETR detection loss (all decoder layers + DN)

        loss_stats: Dict[str, Tensor] = {
            'loss_cls':  s2['cls'],
            'loss_bbox': s2['bbox'],
            'loss_giou': s2['giou'],
        }
        loss_stats['loss_enc_aux'] = s2['enc_aux'] if s2['enc_aux'] is not None \
                                     else det_total.detach() * 0.0

        # Heatmap branch is part of detection (same task, same s_det weight).
        hm_out = outputs['stage2'].heatmap_out
        if hm_out is not None:
            l_hm = self._heatmap_loss(hm_out, targets)
            det_total = det_total + l_hm['total']
            loss_stats['loss_hm']  = l_hm['hm']
            loss_stats['loss_wh']  = l_hm['wh']
            loss_stats['loss_reg'] = l_hm['reg']

        # Learnable uncertainty weighting — AMOT / Kendall et al. 2018:
        #   total = 0.5 * (exp(−s_det)*det + exp(−s_id)*reid + s_det + s_id)
        # s_det, s_id are learnable: model auto-scales each task's gradient share.
        if self.reid_classifier is not None and 'ids' in targets[0]:
            l_reid = self._reid_loss(outputs['stage2'], targets, s2['indices'])
            total  = 0.5 * (
                torch.exp(-self.s_det) * det_total
                + torch.exp(-self.s_id) * l_reid
                + self.s_det + self.s_id
            )
            loss_stats['loss_reid'] = l_reid
        else:
            total = 0.5 * (torch.exp(-self.s_det) * det_total + self.s_det)

        loss_stats['loss'] = total
        return total, loss_stats
