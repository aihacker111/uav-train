"""
HybridLoss: Stage-1 (CenterNet) + Stage-2 (DEIMv2 criterion) loss.

Stage-2 losses follow DEIMv2 exactly:
  - VFL  (Varifocal Loss): IoU-quality-weighted sigmoid focal
  - L1 + GIoU: bounding box regression
  - FGL  (Fine-Grained Localization): distribution focal loss over reg_max bins
  - DDF  (Decoupled Distillation Focal): KL distillation between decoder layers
  - DN   (Denoising): fixed-matching loss on denoising queries
  - Encoder auxiliary: VFL + box on top-K encoder queries
  - Pre-output: VFL + box on first decoder layer (before FDR refinement)

Stage-1 (CenterNet heatmap) loss is combined as:
  L = L_s2 + lambda_cn * L_s1
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
from ..engine.deim.dfine_utils import bbox2distance


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
    Combined CenterNet + DEIMv2 DETR loss.

    stage2 output dict must contain (training mode):
      'pred_logits', 'pred_boxes'             — last decoder layer
      'pred_corners', 'ref_points'            — for FGL/DDF
      'up', 'reg_scale'                       — decoder distribution params
      'aux_outputs'                           — intermediate decoder layers
      'enc_aux_outputs', 'enc_meta'           — encoder top-K auxiliary
      'pre_outputs'                           — first decoder layer (pre-FDR)
      'dn_outputs', 'dn_pre_outputs', 'dn_meta' — denoising queries
    """

    def __init__(
        self,
        num_classes:     int   = 7,
        reg_max:         int   = 32,
        lambda_vfl:      float = 1.0,
        lambda_bbox:     float = 5.0,
        lambda_giou:     float = 2.0,
        lambda_fgl:      float = 0.5,
        lambda_ddf:      float = 1.0,
        lambda_reid:     float = 1.0,
        lambda_triplet:  float = 0.5,
        lambda_cn:       float = 0.5,
        vfl_alpha:       float = 0.2,
        vfl_gamma:       float = 2.0,
        aux_loss:        bool  = True,
        reid_classifier: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.num_classes    = num_classes
        self.reg_max        = reg_max
        self.lambda_vfl     = lambda_vfl
        self.lambda_bbox    = lambda_bbox
        self.lambda_giou    = lambda_giou
        self.lambda_fgl     = lambda_fgl
        self.lambda_ddf     = lambda_ddf
        self.lambda_reid    = lambda_reid
        self.lambda_triplet = lambda_triplet
        self.lambda_cn      = lambda_cn
        self.vfl_alpha      = vfl_alpha
        self.vfl_gamma      = vfl_gamma
        self.aux_loss       = aux_loss

        self.reid_classifier = reid_classifier
        self.triplet_loss    = TripletLoss(margin=0.3)
        self.matcher         = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

        # FGL/DDF cross-call cache — cleared at the start of each forward()
        self._fgl_targets    = None
        self._fgl_targets_dn = None
        self._ddf_num_pos    = None
        self._ddf_num_neg    = None

    def _clear_cache(self):
        self._fgl_targets    = None
        self._fgl_targets_dn = None
        self._ddf_num_pos    = None
        self._ddf_num_neg    = None

    # ── Index helpers ──────────────────────────────────────────────────────────

    def _src_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Union-set matching across all decoder layers (DEIMv2 _get_go_indices)."""
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices, indices_aux)
            ]
        results = []
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            if ind.numel() == 0:
                results.append((ind.new_empty(0, dtype=torch.long),
                                ind.new_empty(0, dtype=torch.long)))
                continue
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            sort_order     = torch.argsort(counts, descending=True)
            unique_sorted  = unique[sort_order]
            col_to_row: dict = {}
            for pair in unique_sorted:
                r, c = pair[0].item(), pair[1].item()
                if r not in col_to_row:
                    col_to_row[r] = c
            rows = torch.tensor(list(col_to_row.keys()), device=ind.device, dtype=torch.long)
            cols = torch.tensor(list(col_to_row.values()), device=ind.device, dtype=torch.long)
            results.append((rows, cols))
        return results

    # ── Paired IoU for matched pairs ──────────────────────────────────────────

    def _paired_iou(self, outputs, targets, indices) -> Tensor:
        idx = self._src_idx(indices)
        if idx[0].numel() == 0:
            return outputs['pred_boxes'].new_empty(0)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        ious = torch.diag(box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )[0]).detach()
        return ious

    # ── VFL loss ──────────────────────────────────────────────────────────────

    def _loss_vfl(self, outputs, targets, indices, num_boxes, ious=None):
        idx = self._src_idx(indices)
        if ious is None:
            ious = self._paired_iou(outputs, targets, indices)

        src_logits = outputs['pred_logits']
        tgt_cls_o  = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        tgt_cls    = torch.full(src_logits.shape[:2], self.num_classes,
                                dtype=torch.int64, device=src_logits.device)
        if idx[0].numel() > 0:
            tgt_cls[idx] = tgt_cls_o

        target      = F.one_hot(tgt_cls, num_classes=self.num_classes + 1)[..., :-1].float()
        tgt_score_o = torch.zeros(src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device)
        if idx[0].numel() > 0:
            tgt_score_o[idx] = ious.to(tgt_score_o.dtype)
        tgt_score   = tgt_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        weight     = self.vfl_alpha * pred_score.pow(self.vfl_gamma) * (1 - target) + tgt_score

        loss = F.binary_cross_entropy_with_logits(src_logits, tgt_score, weight=weight, reduction='none')
        return {'loss_vfl': loss.mean(1).sum() * src_logits.shape[1] / num_boxes}

    # ── Box loss (L1 + GIoU) ──────────────────────────────────────────────────

    def _loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._src_idx(indices)
        if idx[0].numel() == 0:
            z = outputs['pred_boxes'].sum() * 0.0
            return {'loss_bbox': z, 'loss_giou': z}

        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none').sum() / num_boxes

        # GIoU weighted by per-pair IoU from THIS layer's boxes (matching DEIMv2 boxes_weight_format='iou')
        ious = torch.diag(box_iou(
            box_cxcywh_to_xyxy(src_boxes.detach()),
            box_cxcywh_to_xyxy(tgt_boxes),
        )[0])
        loss_giou_vals = (1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        ))) * ious
        loss_giou = loss_giou_vals.sum() / num_boxes

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    # ── FGL + DDF loss ────────────────────────────────────────────────────────

    def _loss_local(self, outputs, targets, indices, num_boxes, is_dn=False, T=5.0):
        losses = {}
        if 'pred_corners' not in outputs:
            return losses

        idx       = self._src_idx(indices)
        if idx[0].numel() == 0:
            return losses

        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        pred_corners = outputs['pred_corners'][idx].reshape(-1, self.reg_max + 1)
        ref_points   = outputs['ref_points'][idx].detach()

        with torch.no_grad():
            if is_dn:
                if self._fgl_targets_dn is None:
                    self._fgl_targets_dn = bbox2distance(
                        ref_points, box_cxcywh_to_xyxy(tgt_boxes),
                        self.reg_max, outputs['reg_scale'], outputs['up'],
                    )
                target_corners, weight_right, weight_left = self._fgl_targets_dn
            else:
                if self._fgl_targets is None:
                    self._fgl_targets = bbox2distance(
                        ref_points, box_cxcywh_to_xyxy(tgt_boxes),
                        self.reg_max, outputs['reg_scale'], outputs['up'],
                    )
                target_corners, weight_right, weight_left = self._fgl_targets

        ious = torch.diag(box_iou(
            box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]),
            box_cxcywh_to_xyxy(tgt_boxes),
        )[0]).detach()
        iou_weights = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

        losses['loss_fgl'] = self._unimodal_dfocal(
            pred_corners, target_corners, weight_right, weight_left, iou_weights, avg_factor=num_boxes,
        )

        if 'teacher_corners' in outputs:
            pred_all    = outputs['pred_corners'].reshape(-1, self.reg_max + 1)
            teacher_all = outputs['teacher_corners'].reshape(-1, self.reg_max + 1)
            if not torch.equal(pred_all, teacher_all):
                B, K = outputs['pred_logits'].shape[:2]
                w_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]   # (B, K)

                mask = torch.zeros(B, K, dtype=torch.bool, device=w_local.device)
                mask[idx] = True
                mask_flat = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                w_local[idx] = ious.reshape_as(w_local[idx]).to(w_local.dtype)
                w_flat = w_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                kl = (T ** 2) * (nn.KLDivLoss(reduction='none')(
                    F.log_softmax(pred_all / T, dim=1),
                    F.softmax(teacher_all.detach() / T, dim=1),
                )).sum(-1)
                loss_all = w_flat * kl

                if not is_dn:
                    batch_scale = 8.0 / B
                    self._ddf_num_pos = (mask_flat.sum().float()  * batch_scale) ** 0.5
                    self._ddf_num_neg = ((~mask_flat).sum().float() * batch_scale) ** 0.5

                if self._ddf_num_pos is not None and (self._ddf_num_pos + self._ddf_num_neg) > 0:
                    l_pos = loss_all[mask_flat].mean()  if mask_flat.any()  else loss_all.new_tensor(0.0)
                    l_neg = loss_all[~mask_flat].mean() if (~mask_flat).any() else loss_all.new_tensor(0.0)
                    losses['loss_ddf'] = (l_pos * self._ddf_num_pos + l_neg * self._ddf_num_neg) / (
                        self._ddf_num_pos + self._ddf_num_neg)

        return losses

    @staticmethod
    def _unimodal_dfocal(pred, label, weight_right, weight_left, iou_weight=None, avg_factor=None):
        dis_left  = label.long()
        dis_right = dis_left + 1
        loss = (F.cross_entropy(pred, dis_left,  reduction='none') * weight_left.reshape(-1)
              + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1))
        if iou_weight is not None:
            loss = loss * iou_weight.float()
        return loss.sum() / avg_factor if avg_factor is not None else loss.sum()

    # ── DN matching ───────────────────────────────────────────────────────────

    @staticmethod
    def _cdn_indices(dn_meta, targets):
        pos_idx    = dn_meta['dn_positive_idx']
        num_group  = dn_meta['dn_num_group']
        device     = targets[0]['labels'].device
        result = []
        for i, t in enumerate(targets):
            n = len(t['labels'])
            if n > 0:
                gt_idx = torch.arange(n, dtype=torch.int64, device=device).tile(num_group)
                result.append((pos_idx[i], gt_idx))
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
            src_i = src_i.to(dev)
            tgt_i = tgt_i.to(dev)
            track_ids = targets[b]['ids'][tgt_i]
            keep = track_ids >= 0
            if not keep.any():
                continue
            valid_emb.append(reid[b][src_i[keep]])
            valid_ids.append(track_ids[keep])

        if not valid_emb:
            return reid.sum() * 0.0

        emb    = torch.cat(valid_emb)
        ids_t  = torch.cat(valid_ids).to(dev)
        logits = self.reid_classifier(emb)
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
        self._clear_cache()

        # Main matching (last decoder layer)
        main_out = {k: v for k, v in stage2.items()
                    if k not in ('aux_outputs', 'enc_aux_outputs', 'dn_outputs',
                                 'dn_pre_outputs', 'dn_meta', 'enc_meta', 'pre_outputs')}
        indices = self.matcher(main_out, targets)['indices']

        # Gather all per-aux-layer indices for union-set computation
        all_aux_indices: List = []
        aux_outputs   = stage2.get('aux_outputs', [])
        enc_aux_outputs = stage2.get('enc_aux_outputs', [])

        if self.aux_loss:
            for aux in aux_outputs:
                all_aux_indices.append(self.matcher(aux, targets)['indices'])
            if 'pre_outputs' in stage2:
                all_aux_indices.append(self.matcher(stage2['pre_outputs'], targets)['indices'])
            for enc_aux in enc_aux_outputs:
                all_aux_indices.append(self.matcher(enc_aux, targets)['indices'])

        # Union-set indices (used for boxes + local losses across all layers)
        indices_go = (self._get_go_indices(list(indices), list(all_aux_indices))
                      if all_aux_indices else list(indices))

        num_boxes    = float(max(1, sum(len(t['labels']) for t in targets)))
        num_boxes_go = float(max(1, sum(len(x[0]) for x in indices_go)))

        losses: Dict[str, Tensor] = {}

        # Main output losses
        ious_main = self._paired_iou(stage2, targets, indices)
        losses.update(self._loss_vfl(stage2, targets, indices, num_boxes, ious=ious_main))
        losses.update(self._loss_boxes(stage2, targets, indices_go, num_boxes_go))
        losses.update(self._loss_local(stage2, targets, indices_go, num_boxes_go))  # caches fgl_targets

        if self.aux_loss:
            # Intermediate decoder layer losses
            n_aux = len(aux_outputs)
            for i, aux in enumerate(aux_outputs):
                aux_w_up  = dict(aux)
                aux_w_up['up']        = stage2['up']
                aux_w_up['reg_scale'] = stage2['reg_scale']
                aux_idx   = all_aux_indices[i]
                ious_aux  = self._paired_iou(aux_w_up, targets, aux_idx)
                sfx       = f'_aux_{i}'
                losses.update({k + sfx: v for k, v in
                    self._loss_vfl(aux_w_up, targets, aux_idx, num_boxes, ious=ious_aux).items()})
                losses.update({k + sfx: v for k, v in
                    self._loss_boxes(aux_w_up, targets, indices_go, num_boxes_go).items()})
                losses.update({k + sfx: v for k, v in
                    self._loss_local(aux_w_up, targets, indices_go, num_boxes_go).items()})

            # Pre-output losses (first decoder layer, before FDR)
            if 'pre_outputs' in stage2:
                pre_idx = all_aux_indices[n_aux]
                pre_out = stage2['pre_outputs']
                ious_pre = self._paired_iou(pre_out, targets, pre_idx)
                losses.update({k + '_pre': v for k, v in
                    self._loss_vfl(pre_out, targets, pre_idx, num_boxes, ious=ious_pre).items()})
                losses.update({k + '_pre': v for k, v in
                    self._loss_boxes(pre_out, targets, indices_go, num_boxes_go).items()})

            # Encoder auxiliary losses
            pre_offset = n_aux + (1 if 'pre_outputs' in stage2 else 0)
            for i, enc_aux in enumerate(enc_aux_outputs):
                enc_idx  = all_aux_indices[pre_offset + i]
                ious_enc = self._paired_iou(enc_aux, targets, enc_idx)
                sfx      = f'_enc_{i}'
                losses.update({k + sfx: v for k, v in
                    self._loss_vfl(enc_aux, targets, enc_idx, num_boxes, ious=ious_enc).items()})
                losses.update({k + sfx: v for k, v in
                    self._loss_boxes(enc_aux, targets, indices_go, num_boxes_go).items()})

        # Denoising losses
        if 'dn_outputs' in stage2 and 'dn_meta' in stage2:
            dn_meta    = stage2['dn_meta']
            indices_dn = self._cdn_indices(dn_meta, targets)
            dn_num_boxes = float(max(1, num_boxes * dn_meta['dn_num_group']))

            for i, dn_out in enumerate(stage2['dn_outputs']):
                dn_out = dict(dn_out)
                dn_out['is_dn']     = True
                dn_out['up']        = stage2['up']
                dn_out['reg_scale'] = stage2['reg_scale']
                sfx = f'_dn_{i}'
                ious_dn = self._paired_iou(dn_out, targets, indices_dn)
                losses.update({k + sfx: v for k, v in
                    self._loss_vfl(dn_out, targets, indices_dn, dn_num_boxes, ious=ious_dn).items()})
                losses.update({k + sfx: v for k, v in
                    self._loss_boxes(dn_out, targets, indices_dn, dn_num_boxes).items()})
                losses.update({k + sfx: v for k, v in
                    self._loss_local(dn_out, targets, indices_dn, dn_num_boxes, is_dn=True).items()})

            if 'dn_pre_outputs' in stage2:
                dn_pre = stage2['dn_pre_outputs']
                ious_dn_pre = self._paired_iou(dn_pre, targets, indices_dn)
                losses.update({k + '_dn_pre': v for k, v in
                    self._loss_vfl(dn_pre, targets, indices_dn, dn_num_boxes, ious=ious_dn_pre).items()})
                losses.update({k + '_dn_pre': v for k, v in
                    self._loss_boxes(dn_pre, targets, indices_dn, dn_num_boxes).items()})

        # Weighted sum for stage-2
        def _base_key(k):
            for sep in ('_aux_', '_dn_', '_enc_'):
                if sep in k:
                    return k[:k.index(sep)]
            return k[:-4] if k.endswith('_pre') else k

        w = {'loss_vfl': self.lambda_vfl, 'loss_bbox': self.lambda_bbox,
             'loss_giou': self.lambda_giou, 'loss_fgl': self.lambda_fgl,
             'loss_ddf': self.lambda_ddf}
        loss_s2 = sum(v * w.get(_base_key(k), 0.0) for k, v in losses.items()
                      if _base_key(k) in w)

        loss_s1  = s1['total']
        total    = loss_s2 + self.lambda_cn * loss_s1

        # Optional ReID
        if self.reid_classifier is not None and 'ids' in targets[0]:
            l_reid = self._reid_loss(stage2, targets, indices)
            total  = total + self.lambda_reid * l_reid
            losses['loss_reid'] = l_reid

        # Clean nan (matching DEIMv2)
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

        loss_stats = {
            'loss':      total,
            'loss_s1':   s1['total'].detach(),
            'loss_hm':   s1['hm'].detach(),
            'loss_wh':   s1['wh'].detach(),
            'loss_reg':  s1['reg'].detach(),
            'loss_s2':   loss_s2.detach(),
            'loss_vfl':  losses.get('loss_vfl', total.new_tensor(0.0)).detach(),
            'loss_bbox': losses.get('loss_bbox', total.new_tensor(0.0)).detach(),
            'loss_giou': losses.get('loss_giou', total.new_tensor(0.0)).detach(),
            'loss_fgl':  losses.get('loss_fgl',  total.new_tensor(0.0)).detach(),
            'loss_ddf':  losses.get('loss_ddf',  total.new_tensor(0.0)).detach(),
        }
        if 'loss_reid' in losses:
            loss_stats['loss_reid'] = losses['loss_reid'].detach()

        return total, loss_stats
