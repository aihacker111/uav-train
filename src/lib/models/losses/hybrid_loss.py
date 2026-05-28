# """
# HybridLoss: combined Stage-1 (CenterNet) + Stage-2 (DETR + ReID) loss.

# Stage 1 — CenterNet:
#   L_s1 = L_focal(hm) + λ_wh * L_L1(wh) + λ_reg * L_L1(reg)

# Stage 2 — DETR (Hungarian matching, applied per decoder layer):
#   L_s2 = L_varifocal(cls) + λ_bbox * L_L1(box) + λ_ciou * L_CIoU(box)
#         + λ_reid * (L_CE + L_triplet)(reid)  [if reid_classifier is set]
#         + λ_consist * L_consist(Stage-1 hm ↔ GT centers of Stage-2 matched objects)

# Auxiliary losses from intermediate decoder layers use progressive weights:
#   layer i weight = aux_weight_base * (i + 1) / num_aux_layers
#   (shallower layers get smaller weight since their predictions are less refined)

# Stage balance curriculum (when total_epochs > 0):
#   t = epoch / total_epochs  ∈ [0, 1]
#   eff_λ_s1 = λ_stage1  →  1.0   (decays: Stage-1 is dominant early, balanced late)
#   eff_λ_s2 = λ_stage2  →  λ_stage1  (rises: Stage-2 gains more weight over time)
#   This implements a natural curriculum: CenterNet bootstraps Stage-2 first, then
#   the decoder takes over as the primary learning signal.

# Changes vs. v1:
#   • sigmoid_focal_loss → varifocal_loss for Stage-2 cls (IoU-quality-weighted)
#   • GIoU → CIoU for box regression (more stable for tiny boxes)
#   • ReID: CE + TripletLoss (hard mining) instead of CE-only
#   • Consistency loss: uses GT box centers (not Stage-2 predicted centers) for
#     a stable, Stage-2-quality-independent supervision signal on Stage-1 heatmap
#   • Dynamic stage weighting: curriculum from Stage-1-heavy to Stage-2-heavy
#   • Progressive aux weights: shallower decoder layers receive lower loss weight
# """
# from __future__ import annotations

# import math
# from typing import Dict, List, Any, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou
# from ..networks.deim_uav.heads import CenterNetOutput, DETROutput
# from ..base_losses import TripletLoss


# # ── Loss primitives ────────────────────────────────────────────────────────────

# def centernet_focal_loss(pred_hm: Tensor, gt_hm: Tensor) -> Tensor:
#     """
#     Modified CenterNet focal loss on Gaussian-rendered heatmaps.

#     pred_hm : (B, C, H, W) — sigmoid output
#     gt_hm   : (B, C, H, W) — Gaussian-rendered ground-truth in [0, 1]
#     """
#     pos_mask = (gt_hm == 1).float()
#     neg_mask = 1.0 - pos_mask

#     p     = pred_hm.clamp(1e-6, 1.0 - 1e-6)
#     pos_l = -((1 - p) ** 2) * p.log() * pos_mask
#     neg_l = -((1 - gt_hm) ** 4) * (p ** 2) * (1 - p).log() * neg_mask

#     n_pos = pos_mask.sum().clamp(min=1)
#     return (pos_l + neg_l).sum() / n_pos


# def varifocal_loss(
#     pred:    Tensor,   # (*, C) raw logits
#     q:       Tensor,   # (*, C) quality targets — IoU score for positives, 0 for negatives
#     alpha:   float = 0.75,
#     gamma:   float = 2.0,
# ) -> Tensor:
#     """
#     Varifocal Loss (VarifocalNet, Zhang et al. 2021).

#     Unlike sigmoid focal loss that uses a fixed α weight for positives,
#     varifocal uses the IoU-quality score q as the positive weight:

#       VFL(p, q) =
#         q > 0 (positive):  -q * (q·log(p) + (1-q)·log(1-p))
#         q = 0 (negative):  -α * p^γ * log(1-p)

#     This gives stronger gradient to high-quality predictions and naturally
#     down-weights low-quality (low-IoU) matches — better aligned with AP metric.
#     Normalised by number of positive cells (q > 0).
#     """
#     p    = pred.sigmoid()
#     ce   = F.binary_cross_entropy_with_logits(pred, q.clamp(0, 1), reduction='none')

#     pos_mask = (q > 0).float()
#     neg_mask = 1.0 - pos_mask

#     pos_weight = q          # q itself for positives
#     neg_weight = alpha * p.pow(gamma)

#     loss = (pos_mask * pos_weight + neg_mask * neg_weight) * ce
#     n_pos = pos_mask.sum().clamp(min=1)
#     return loss.sum() / n_pos


# def _paired_iou(b1: Tensor, b2: Tensor) -> Tensor:
#     """
#     Element-wise IoU for N matched box pairs.
#     b1, b2 : (N, 4) in xyxy format.
#     Returns (N,) IoU values.
#     """
#     inter_x1 = torch.max(b1[:, 0], b2[:, 0])
#     inter_y1 = torch.max(b1[:, 1], b2[:, 1])
#     inter_x2 = torch.min(b1[:, 2], b2[:, 2])
#     inter_y2 = torch.min(b1[:, 3], b2[:, 3])

#     inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
#     a1    = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
#     a2    = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
#     union = a1 + a2 - inter
#     return inter / (union + 1e-7)


# def ciou_loss(pred_xyxy: Tensor, tgt_xyxy: Tensor,
#               iou: Tensor | None = None) -> Tensor:
#     """
#     CIoU loss for N matched pairs (Complete-IoU).

#     CIoU = 1 - IoU + ρ²(centre_pred, centre_gt)/c² + α·ν
#       ρ²  : squared Euclidean distance between box centres
#       c²  : squared diagonal of the smallest enclosing box
#       ν   : aspect-ratio consistency term
#       α   : trade-off coefficient = ν / (1 - IoU + ν + ε)

#     More stable than GIoU for tiny boxes because it additionally penalises
#     centre misalignment and aspect-ratio difference.

#     pred_xyxy, tgt_xyxy : (N, 4) in xyxy format.
#     Returns scalar mean CIoU loss.
#     """
#     if iou is None:
#         iou = _paired_iou(pred_xyxy, tgt_xyxy)   # (N,)

#     # Enclosing box diagonal squared
#     enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
#     enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
#     enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
#     enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
#     c2 = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + 1e-7

#     # Centre-distance penalty
#     pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
#     pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
#     tgt_cx  = (tgt_xyxy[:, 0]  + tgt_xyxy[:, 2])  / 2
#     tgt_cy  = (tgt_xyxy[:, 1]  + tgt_xyxy[:, 3])  / 2
#     rho2    = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)

#     # Aspect-ratio consistency
#     pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(1e-7)
#     pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(1e-7)
#     tgt_w  = (tgt_xyxy[:, 2]  - tgt_xyxy[:, 0]).clamp(1e-7)
#     tgt_h  = (tgt_xyxy[:, 3]  - tgt_xyxy[:, 1]).clamp(1e-7)
#     v      = (2 / math.pi) ** 2 * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)).pow(2)

#     with torch.no_grad():
#         alpha_c = v / (1 - iou + v + 1e-7)

#     return (1 - iou + rho2 / c2 + alpha_c * v).mean()


# def _gather_at_ind(feat: Tensor, ind: Tensor) -> Tensor:
#     """
#     Gather spatial predictions at flat peak indices.

#     feat : (B, C, H, W)
#     ind  : (B, max_obj)  — flat HW index
#     Returns (B, max_obj, C)
#     """
#     B, C, H, W = feat.shape
#     flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
#     idx  = ind.unsqueeze(-1).expand(B, ind.shape[1], C)
#     return flat.gather(1, idx)


# # ── HybridLoss ─────────────────────────────────────────────────────────────────

# class HybridLoss(nn.Module):
#     """
#     Combined CenterNet + DETR loss.

#     The batch dict must contain:
#       Stage-1 targets (CenterNet format):
#         'hm'       : (B, C, H, W)        — Gaussian rendered heatmap
#         'wh'       : (B, max_obj, 2)      — width/height at peak locations
#         'reg'      : (B, max_obj, 2)      — sub-pixel offset at peak locations
#         'ind'      : (B, max_obj)         — flat spatial index of each peak
#         'reg_mask' : (B, max_obj)         — 1 for valid objects, 0 for padding

#       Stage-2 targets (DETR format):
#         'targets'  : List[dict] with keys 'labels' (N,) and 'boxes' (N, 4) cxcywh

#       ReID targets (optional):
#         'ids'      : (B, max_obj)         — track ID (−1 = ignore)
#     """

#     def __init__(
#         self,
#         num_classes:           int   = 7,
#         lambda_wh:             float = 0.1,
#         lambda_reg:            float = 1.0,
#         lambda_bbox:           float = 2.0,   # matches trainer default (was 5.0, misleading)
#         lambda_ciou:           float = 2.0,
#         lambda_reid:           float = 1.0,
#         lambda_triplet:        float = 0.5,
#         lambda_consist:        float = 0.02,
#         consist_warmup_epochs: int   = 5,     # shorter: GT centers don't need long warmup
#         lambda_stage1:         float = 2.0,
#         lambda_stage2:         float = 1.0,
#         aux_loss:              bool  = True,
#         reid_classifier:       Optional[nn.Linear] = None,
#         total_epochs:          int   = 0,     # 0 = disable dynamic weighting
#     ) -> None:
#         super().__init__()
#         self.num_classes           = num_classes
#         self.lambda_wh             = lambda_wh
#         self.lambda_reg            = lambda_reg
#         self.lambda_bbox           = lambda_bbox
#         self.lambda_ciou           = lambda_ciou
#         self.lambda_reid           = lambda_reid
#         self.lambda_triplet        = lambda_triplet
#         self.lambda_consist        = lambda_consist
#         self.consist_warmup_epochs = consist_warmup_epochs
#         self.lambda_stage1         = lambda_stage1
#         self.lambda_stage2         = lambda_stage2
#         self.aux_loss              = aux_loss
#         self._epoch                = 0      # updated each epoch via set_epoch()
#         self.total_epochs          = total_epochs

#         self.reid_classifier = reid_classifier
#         self.triplet_loss    = TripletLoss(margin=0.3)

#         self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

#     def set_epoch(self, epoch: int) -> None:
#         """Call at the start of each epoch so warmup ramps and dynamic weights update."""
#         self._epoch = epoch

#     def _effective_stage_weights(self) -> tuple[float, float]:
#         """
#         Compute dynamic per-epoch stage weights.

#         Curriculum schedule (t = epoch / total_epochs ∈ [0, 1]):
#           eff_λ_s1: lambda_stage1 → 1.0  (Stage-1 weight decays as it stabilises)
#           eff_λ_s2: lambda_stage2 → lambda_stage1  (Stage-2 rises to share load equally)

#         Rationale:
#           - Early training: Stage-1 (CenterNet) must learn before Stage-2 can get
#             good query seeds. A higher Stage-1 weight drives faster heatmap convergence.
#           - Late training: Stage-1 heatmap is stable; Stage-2 decoder needs more
#             gradient to fine-tune boxes and classification for VisDrone's small objects.
#           - The endpoint eff_λ_s1 = eff_λ_s2 = 1.0 gives balanced supervision
#             so neither stage dominates at convergence.

#         When total_epochs == 0, returns the static configured weights unchanged.
#         """
#         if self.total_epochs <= 0:
#             return self.lambda_stage1, self.lambda_stage2

#         t = min(1.0, self._epoch / self.total_epochs)   # ∈ [0, 1]

#         # Cosine interpolation: smoother than linear at the endpoints
#         # t=0 → start values, t=1 → end values
#         smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))

#         eff_s1 = self.lambda_stage1 + smooth_t * (1.0 - self.lambda_stage1)
#         eff_s2 = self.lambda_stage2 + smooth_t * (self.lambda_stage1 - self.lambda_stage2)
#         return eff_s1, eff_s2

#     # ── Stage-1 ────────────────────────────────────────────────────────────────

#     def _stage1_loss(self, cn_out: CenterNetOutput, batch: dict) -> Dict[str, Tensor]:
#         dev = cn_out.hm.device

#         # Scatter.apply moves batch tensors to the replica device, but add an
#         # explicit guard so this function is safe even if called without scatter.
#         hm       = batch['hm'].to(dev,       non_blocking=True)
#         reg_mask = batch['reg_mask'].to(dev, non_blocking=True)
#         ind      = batch['ind'].to(dev,      non_blocking=True)
#         gt_wh    = batch['wh'].to(dev,       non_blocking=True)
#         gt_reg   = batch['reg'].to(dev,      non_blocking=True)

#         loss_hm  = centernet_focal_loss(cn_out.hm, hm)
#         n        = reg_mask.sum().clamp(min=1).float()

#         pred_wh  = _gather_at_ind(cn_out.wh,  ind)   # (B, max_obj, 2)
#         pred_reg = _gather_at_ind(cn_out.reg, ind)

#         mask     = reg_mask.unsqueeze(-1).float()
#         # SmoothL1: quadratic for |err|<beta (stable for small offsets), L1 beyond
#         loss_wh  = (F.smooth_l1_loss(pred_wh,  gt_wh,  beta=1.0, reduction='none') * mask).sum() / n
#         loss_reg = (F.smooth_l1_loss(pred_reg, gt_reg, beta=0.5, reduction='none') * mask).sum() / n

#         total = loss_hm + self.lambda_wh * loss_wh + self.lambda_reg * loss_reg
#         return {'total': total, 'hm': loss_hm, 'wh': loss_wh, 'reg': loss_reg}

#     # ── Stage-2 ────────────────────────────────────────────────────────────────

#     def _detr_layer_loss(
#         self,
#         logits:  Tensor,       # (B, K, C)
#         boxes:   Tensor,       # (B, K, 4) cxcywh
#         targets: List[dict],
#         indices: list,         # list of {'src_i', 'tgt_i', 'iou'} dicts from matcher
#     ) -> Dict[str, Tensor]:
#         dev = logits.device

#         # ── Collect matched pairs + reuse cached IoU from matcher ──────────────
#         # IoU is pre-computed in HungarianMatcher to avoid a redundant _paired_iou
#         # call here. Saves one full forward over all matched pairs per layer.
#         Q_MIN = 0.1
#         tgt_cls    = torch.zeros_like(logits)
#         src_b_list, tgt_b_list, iou_list = [], [], []

#         for b, m in enumerate(indices):
#             src_i, tgt_i, iou_cached = m['src_i'], m['tgt_i'], m['iou']
#             if not len(src_i):
#                 continue
#             si  = src_i.to(dev)
#             ti  = tgt_i.to(dev)
#             iou = iou_cached.to(dev)

#             # Varifocal quality target: q = IoU for positives (floored at Q_MIN),
#             # 0 for negatives.  Without Q_MIN floor, cls head gets zero positive
#             # gradient until box IoU > 0 — chicken-and-egg early in training.
#             tgt_cls[b, si, targets[b]['labels'][ti]] = iou.clamp(min=Q_MIN)

#             src_b_list.append(boxes[b][si])
#             tgt_b_list.append(targets[b]['boxes'][ti])
#             iou_list.append(iou)

#         loss_cls = varifocal_loss(logits, tgt_cls)

#         # ── CIoU box regression on matched pairs ───────────────────────────────
#         if src_b_list:
#             src_b  = torch.cat(src_b_list)   # (N_total, 4) cxcywh
#             tgt_b  = torch.cat(tgt_b_list)
#             iou_all = torch.cat(iou_list)     # (N_total,) — pre-computed, reused
#             n_m    = src_b.shape[0]
#             # SmoothL1 with beta=0.05: normalized box coords are in [0,1],
#             # threshold at 0.05 ≈ 64px error at 1280px width.
#             loss_bbox = F.smooth_l1_loss(src_b, tgt_b, beta=0.05, reduction='sum') / n_m
#             loss_ciou = ciou_loss(
#                 box_cxcywh_to_xyxy(src_b),
#                 box_cxcywh_to_xyxy(tgt_b),
#                 iou=iou_all,          # pass cached IoU — skips _paired_iou inside
#             )
#         else:
#             loss_bbox = loss_ciou = logits.sum() * 0.0

#         total = loss_cls + self.lambda_bbox * loss_bbox + self.lambda_ciou * loss_ciou
#         return {'total': total, 'cls': loss_cls, 'bbox': loss_bbox, 'ciou': loss_ciou}

#     def _stage2_loss(
#         self,
#         detr_out: DETROutput,
#         targets:  List[dict],
#     ) -> Dict[str, Tensor]:
#         indices = self.matcher(detr_out.logits, detr_out.boxes, targets)
#         d       = self._detr_layer_loss(detr_out.logits, detr_out.boxes, targets, indices)
#         total   = d['total']

#         if self.aux_loss:
#             L = detr_out.boxes_all.shape[0]
#             n_aux = L - 1  # number of intermediate layers
#             for layer in range(n_aux):
#                 # Reuse final-layer indices for aux layers — standard practice in
#                 # DN-DETR, Anchor-DETR; avoids O(K²·N) matching per layer.
#                 # Cached IoU from final-layer matching is also reused, which means
#                 # aux-layer IoU may be slightly stale but the assignment is correct.
#                 aux = self._detr_layer_loss(
#                     detr_out.logits_all[layer], detr_out.boxes_all[layer], targets, indices,
#                 )
#                 # Progressive aux weights: shallower layers get lower weight since
#                 # their predictions are less refined. Linearly from 0.25 (layer 0)
#                 # to 0.5 (layer n_aux-1). Final layer always has weight 1.0.
#                 aux_w = 0.25 + 0.25 * (layer / max(n_aux - 1, 1))
#                 total = total + aux_w * aux['total']

#         return {'total': total, 'cls': d['cls'], 'bbox': d['bbox'], 'ciou': d['ciou'],
#                 'indices': indices}

#     def _reid_loss(
#         self,
#         detr_out: DETROutput,
#         targets:  List[dict],
#         indices:  list,
#     ) -> Tensor:
#         if detr_out.reid is None:
#             return detr_out.logits.sum() * 0.0

#         valid_emb, valid_ids = [], []
#         dev = detr_out.reid.device

#         for b, m in enumerate(indices):
#             src_i, tgt_i = m['src_i'], m['tgt_i']
#             if len(src_i) == 0:
#                 continue
#             if 'ids' not in targets[b]:
#                 continue

#             src_i = src_i.to(dev)
#             tgt_i = tgt_i.to(dev)

#             track_ids = targets[b]['ids'][tgt_i]
#             keep      = track_ids >= 0
#             if not keep.any():
#                 continue

#             valid_emb.append(detr_out.reid[b][src_i[keep]])
#             valid_ids.append(track_ids[keep])

#         if not valid_emb:
#             return detr_out.reid.sum() * 0.0

#         emb    = torch.cat(valid_emb)                  # (N, reid_dim)
#         ids_t  = torch.cat(valid_ids).to(emb.device)  # (N,)
#         logits = self.reid_classifier(emb)             # (N, total_ids)

#         loss_ce = F.cross_entropy(logits, ids_t)

#         # Triplet needs ≥ 2 unique IDs and ≥ 2 samples; skip if batch is too small
#         unique_ids = ids_t.unique()
#         if unique_ids.numel() >= 2 and emb.shape[0] >= 2:
#             loss_tri = self.triplet_loss(emb, ids_t)
#         else:
#             loss_tri = emb.sum() * 0.0

#         return loss_ce + self.lambda_triplet * loss_tri

#     def _consistency_loss(
#         self,
#         cn_out:   CenterNetOutput,
#         targets:  List[dict],
#         indices:  list,
#     ) -> Tensor:
#         """
#         Stage-1 / Stage-2 consistency: the Stage-1 heatmap value at the GT center
#         of each Stage-2 Hungarian-matched object should be high (≈ 1).

#         Uses GT box centers (not Stage-2 predicted centers) for a stable supervision
#         signal that is independent of Stage-2 prediction quality. This avoids the
#         risk of training Stage-1 to place peaks at wrong positions when Stage-2 is
#         still noisy early in training.

#         Only objects that appear in the Stage-2 matching (i.e., objects the decoder
#         sees) are supervised — objects filtered out before Stage-2 are ignored.
#         Only updates Stage-1 heatmap head (no gradient to Stage-2 decoder).
#         """
#         hm = cn_out.hm                       # (B, C, H_hm, W_hm) — sigmoid
#         B, C, H_hm, W_hm = hm.shape
#         dev = hm.device

#         total = hm.sum() * 0.0
#         n_batches = 0

#         for b, m in enumerate(indices):
#             src_i, tgt_i = m['src_i'], m['tgt_i']
#             if len(src_i) == 0:
#                 continue

#             ti = tgt_i.to(dev)

#             # GT box centers from Stage-2 targets — stable regardless of decoder quality.
#             # cxcywh normalised, so cx/cy ∈ [0, 1] directly.
#             gt_boxes = targets[b]['boxes'][ti]           # (n, 4) cxcywh normalised
#             cx       = gt_boxes[:, 0].clamp(0, 1)
#             cy       = gt_boxes[:, 1].clamp(0, 1)

#             # GT class for each matched object
#             cls_idx = targets[b]['labels'][ti]           # (n,) long

#             # Map normalised coords → heatmap pixel indices
#             x_hm = (cx * (W_hm - 1)).long().clamp(0, W_hm - 1)
#             y_hm = (cy * (H_hm - 1)).long().clamp(0, H_hm - 1)

#             # Sample heatmap at GT object centers and push toward 1.0
#             hm_vals = hm[b, cls_idx, y_hm, x_hm]        # (n,)
#             total   = total + F.binary_cross_entropy(
#                 hm_vals, torch.ones_like(hm_vals), reduction='mean',
#             )
#             n_batches += 1

#         return total / max(n_batches, 1)

#     # ── Forward ────────────────────────────────────────────────────────────────

#     def forward(
#         self,
#         outputs: Dict[str, Any],
#         batch:   dict,
#     ) -> tuple[Tensor, Dict[str, Tensor]]:
#         # Move per-image target dicts to the replica device once here so that
#         # _stage2_loss, _reid_loss, and _consistency_loss all see the right device.
#         dev = outputs['stage2'].logits.device
#         # scatter_gather already moved each chunk's tensors to the replica device,
#         # so this is a no-op safety guard for any caller that bypasses scatter.
#         targets = [
#             {k: v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v
#              for k, v in t.items()}
#             for t in batch['targets']
#         ]

#         s1 = self._stage1_loss(outputs['stage1'], batch)
#         s2 = self._stage2_loss(outputs['stage2'], targets)

#         # Dynamic curriculum weights: Stage-1 heavy early, Stage-2 heavy late.
#         # Falls back to static lambda_stage1/2 when total_epochs == 0.
#         eff_s1, eff_s2 = self._effective_stage_weights()
#         total = eff_s1 * s1['total'] + eff_s2 * s2['total']

#         loss_stats: Dict[str, Tensor] = {
#             'loss':      total,
#             'loss_s1':   s1['total'],
#             'loss_hm':   s1['hm'],
#             'loss_wh':   s1['wh'],
#             'loss_reg':  s1['reg'],
#             'loss_s2':   s2['total'],
#             'loss_cls':  s2['cls'],
#             'loss_bbox': s2['bbox'],
#             'loss_ciou': s2['ciou'],
#             'w_s1':      torch.tensor(eff_s1),   # log effective weights for monitoring
#             'w_s2':      torch.tensor(eff_s2),
#         }

#         if self.reid_classifier is not None and 'ids' in targets[0]:
#             l_reid = self._reid_loss(outputs['stage2'], targets, s2['indices'])
#             total  = total + self.lambda_reid * l_reid
#             loss_stats['loss_reid'] = l_reid

#         # Consistency loss: ramped from 0 → lambda_consist over consist_warmup_epochs.
#         # Now uses GT centers (not Stage-2 predictions), so warmup is shorter (5 epochs):
#         # it still prevents matching noise from corrupting Stage-1 on epoch 0, but
#         # the signal itself is stable so it doesn't need a long warmup.
#         consist_scale = min(1.0, self._epoch / max(1, self.consist_warmup_epochs))
#         l_consist = self._consistency_loss(
#             outputs['stage1'], targets, s2['indices'],
#         )
#         total = total + self.lambda_consist * consist_scale * l_consist
#         loss_stats['loss_consist'] = l_consist
#         loss_stats['loss'] = total

#         return total, loss_stats


"""
HybridLoss: combined Stage-1 (CenterNet) + Stage-2 (DETR + ReID) loss.

Stage 1 — CenterNet:
  L_s1 = L_focal(hm) + λ_wh * L_L1(wh) + λ_reg * L_L1(reg)

Stage 2 — DETR (Hungarian matching, applied per decoder layer):
  L_s2 = L_varifocal(cls) + λ_bbox * L_L1(box) + λ_ciou * L_CIoU(box)
        + λ_reid * (L_CE + L_triplet)(reid)  [if reid_classifier is set]

Combined loss:
  L = L_s2 + λ_cn * L_s1

Auxiliary losses from intermediate decoder layers use progressive weights:
  layer i weight = aux_weight_base * (i + 1) / num_aux_layers
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
from ..networks.deim_uav.heads import CenterNetOutput, DETROutput
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

    Unlike sigmoid focal loss that uses a fixed α weight for positives,
    varifocal uses the IoU-quality score q as the positive weight:

      VFL(p, q) =
        q > 0 (positive):  -q * (q·log(p) + (1-q)·log(1-p))
        q = 0 (negative):  -α * p^γ * log(1-p)

    Normalised by number of positive cells (q > 0).
    """
    p    = pred.sigmoid()
    ce   = F.binary_cross_entropy_with_logits(pred, q.clamp(0, 1), reduction='none')

    pos_mask = (q > 0).float()
    neg_mask = 1.0 - pos_mask

    pos_weight = q
    neg_weight = alpha * p.pow(gamma)

    loss = (pos_mask * pos_weight + neg_mask * neg_weight) * ce
    n_pos = pos_mask.sum().clamp(min=1)
    return loss.sum() / n_pos


def _paired_iou(b1: Tensor, b2: Tensor) -> Tensor:
    """
    Element-wise IoU for N matched box pairs.
    b1, b2 : (N, 4) in xyxy format.
    Returns (N,) IoU values.
    """
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1    = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2    = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    union = a1 + a2 - inter
    return inter / (union + 1e-7)


def giou_loss(pred_xyxy: Tensor, tgt_xyxy: Tensor) -> Tensor:
    """
    GIoU loss for N matched pairs — matches the GIoU cost used by HungarianMatcher.
    pred_xyxy, tgt_xyxy : (N, 4) in xyxy format.
    Returns scalar mean GIoU loss.
    """
    inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    inter  = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1  = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(0)
    area2  = (tgt_xyxy[:, 2]  - tgt_xyxy[:, 0]).clamp(0)  * (tgt_xyxy[:, 3]  - tgt_xyxy[:, 1]).clamp(0)
    union  = area1 + area2 - inter
    iou    = inter / (union + 1e-7)

    enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    enc    = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + 1e-7

    return (1 - (iou - (enc - union) / enc)).mean()


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
    Combined CenterNet + DETR loss.

      L = L_s2 + lambda_cn * L_s1

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
        num_classes:     int   = 7,
        lambda_wh:       float = 0.1,
        lambda_reg:      float = 1.0,
        lambda_bbox:     float = 2.0,
        lambda_ciou:     float = 2.0,
        lambda_reid:     float = 1.0,
        lambda_triplet:  float = 0.5,
        lambda_cn:       float = 0.5,   # weight for CenterNet stage-1 loss
        aux_loss:        bool  = True,
        reid_classifier: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.num_classes     = num_classes
        self.lambda_wh       = lambda_wh
        self.lambda_reg      = lambda_reg
        self.lambda_bbox     = lambda_bbox
        self.lambda_ciou     = lambda_ciou
        self.lambda_reid     = lambda_reid
        self.lambda_triplet  = lambda_triplet
        self.lambda_cn       = lambda_cn
        self.aux_loss        = aux_loss

        self.reid_classifier = reid_classifier
        self.triplet_loss    = TripletLoss(margin=0.3)


        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    # ── Stage-1 ────────────────────────────────────────────────────────────────

    def _stage1_loss(self, cn_out: CenterNetOutput, batch: dict) -> Dict[str, Tensor]:
        dev = cn_out.hm.device

        hm       = batch['hm'].to(dev,       non_blocking=True)
        reg_mask = batch['reg_mask'].to(dev, non_blocking=True)
        ind      = batch['ind'].to(dev,      non_blocking=True)
        gt_wh    = batch['wh'].to(dev,       non_blocking=True)
        gt_reg   = batch['reg'].to(dev,      non_blocking=True)

        loss_hm  = centernet_focal_loss(cn_out.hm, hm)

        pred_wh  = _gather_at_ind(cn_out.wh,  ind)   # (B, max_obj, 2)
        pred_reg = _gather_at_ind(cn_out.reg, ind)   # (B, max_obj, 2)

        # Expand mask to (B, max_obj, 2) — normalize by total valid dimensions,
        # matching AMOT's RegL1Loss: F.l1_loss(pred*mask, tgt*mask) / mask.sum()
        mask_wh  = reg_mask.unsqueeze(-1).expand_as(pred_wh).float()
        mask_reg = reg_mask.unsqueeze(-1).expand_as(pred_reg).float()
        loss_wh  = F.l1_loss(pred_wh  * mask_wh,  gt_wh  * mask_wh,  reduction='sum') / (mask_wh.sum()  + 1e-4)
        loss_reg = F.l1_loss(pred_reg * mask_reg, gt_reg * mask_reg, reduction='sum') / (mask_reg.sum() + 1e-4)

        total = loss_hm + self.lambda_wh * loss_wh + self.lambda_reg * loss_reg
        return {'total': total, 'hm': loss_hm, 'wh': loss_wh, 'reg': loss_reg}

    # ── Stage-2 ────────────────────────────────────────────────────────────────

    def _detr_layer_loss(
        self,
        logits:  Tensor,       # (B, K, C)
        boxes:   Tensor,       # (B, K, 4) cxcywh
        targets: List[dict],
        indices: list,         # list of {'src_i', 'tgt_i', 'iou'} dicts from matcher
    ) -> Dict[str, Tensor]:
        dev = logits.device

        Q_MIN = 0.05
        tgt_cls    = torch.zeros_like(logits)
        src_b_list, tgt_b_list = [], []

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

        loss_cls = varifocal_loss(logits, tgt_cls)

        if src_b_list:
            src_b = torch.cat(src_b_list)
            tgt_b = torch.cat(tgt_b_list)
            # L1 box cost matches HungarianMatcher's cost_bbox (torch.cdist p=1)
            loss_bbox = F.l1_loss(src_b, tgt_b, reduction='mean')
            # GIoU loss matches HungarianMatcher's cost_giou (generalized_box_iou)
            loss_giou = giou_loss(
                box_cxcywh_to_xyxy(src_b),
                box_cxcywh_to_xyxy(tgt_b),
            )
        else:
            loss_bbox = loss_giou = logits.sum() * 0.0

        total = loss_cls + self.lambda_bbox * loss_bbox + self.lambda_ciou * loss_giou
        return {'total': total, 'cls': loss_cls, 'bbox': loss_bbox, 'giou': loss_giou}

    def _stage2_loss(
        self,
        detr_out: DETROutput,
        targets:  List[dict],
    ) -> Dict[str, Tensor]:
        indices = self.matcher(detr_out.logits, detr_out.boxes, targets)
        d       = self._detr_layer_loss(detr_out.logits, detr_out.boxes, targets, indices)
        total   = d['total']

        if self.aux_loss:
            L = detr_out.boxes_all.shape[0]
            n_aux = L - 1
            for layer in range(n_aux):
                aux = self._detr_layer_loss(
                    detr_out.logits_all[layer], detr_out.boxes_all[layer], targets, indices,
                )
                aux_w = 0.25 + 0.25 * (layer / max(n_aux - 1, 1))
                total = total + aux_w * aux['total']

        return {'total': total, 'cls': d['cls'], 'bbox': d['bbox'], 'giou': d['giou'],
                'indices': indices}

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
            if len(src_i) == 0:
                continue
            if 'ids' not in targets[b]:
                continue

            src_i = src_i.to(dev)
            tgt_i = tgt_i.to(dev)

            track_ids = targets[b]['ids'][tgt_i]
            keep      = track_ids >= 0
            if not keep.any():
                continue

            valid_emb.append(detr_out.reid[b][src_i[keep]])
            valid_ids.append(track_ids[keep])

        if not valid_emb:
            return detr_out.reid.sum() * 0.0

        emb   = torch.cat(valid_emb)                  # (N, reid_dim) — L2-normalised from DETRHead
        ids_t = torch.cat(valid_ids).to(emb.device)  # (N,)

        # emb_unit  : unit sphere — for TripletLoss (metric learning, distances in [0, 2])
        # emb_normed: scaled by emb_scale — for linear classifier (angular softmax margin)
        #   emb_scale = sqrt(2) * log(nID - 1)  matching AMOT's emb_scale_dict
        n_ids      = self.reid_classifier.out_features
        emb_scale  = math.sqrt(2) * math.log(max(n_ids - 1, 1))
        emb_unit   = F.normalize(emb, p=2, dim=-1)
        emb_normed = emb_scale * emb_unit

        logits  = self.reid_classifier(emb_normed)
        loss_ce = F.cross_entropy(logits, ids_t)

        unique_ids = ids_t.unique()
        if unique_ids.numel() >= 2 and ids_t.numel() >= 2:
            loss_tri = self.triplet_loss(emb_unit, ids_t)   # unit sphere: distances in [0, 2]
        else:
            loss_tri = emb_unit.sum() * 0.0

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

        # Simple weighted combination: DETR loss + lambda_cn * CenterNet loss
        total = s2['total'] + self.lambda_cn * s1['total']

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
        }

        if self.reid_classifier is not None and 'ids' in targets[0]:
            l_reid = self._reid_loss(outputs['stage2'], targets, s2['indices'])
            total  = total + self.lambda_reid * l_reid
            loss_stats['loss_reid'] = l_reid
            loss_stats['loss']      = total

        return total, loss_stats