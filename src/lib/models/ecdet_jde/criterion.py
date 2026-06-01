"""
ECDetJDE Criterion — RF-DETR style: focal + L1 + GIoU + ReID.

Detection losses (simple, stable early-training):
  loss_cls   — sigmoid focal loss (α=0.25, γ=2.0), RF-DETR / RetinaNet style
  loss_bbox  — L1 box regression
  loss_giou  — GIoU box regression
ReID loss: CE + optional Triplet on matched queries.

MAL and FGL/DDF are intentionally removed:
  - MAL ties confidence to IoU → gradient noise when IoU ≈ 0 at init
  - FGL/DDF (DFL distribution) only helps when boxes are already good (IoU>0.3)
  → Both hurt convergence in early training and produce noisy confidence scores.

Optimizations:
  1. IoU computed ONCE per (outputs, indices) pair via _get_shared_meta(),
     shared between focal (empty) and boxes (boxes_weight).
  2. All matcher calls batched upfront at the start of forward() so GPU
     cost-matrix kernels can overlap with earlier CPU linear_sum_assignment calls.
  3. enc_targets built with a dict-comprehension shallow copy instead of
     copy.deepcopy(), avoiding a full tensor copy per forward pass.
  4. ReID embedding gathering restructured: one GPU gather per image (not one
     per class×image), reducing redundant indexing from O(C×B) to O(B).
"""

import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .matcher import HungarianMatcher


import math as _math

def _smooth_ln(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Piecewise smooth_ln from the Repulsion Loss paper (Wang et al. CVPR 2018).

    For x in [0, sigma]:  -log(1 - x)          (standard log penalty)
    For x in (sigma, 1):  (x-sigma)/(1-sigma) - log(1-sigma)  (linear tail)

    The linear tail prevents -log(1-x) → ∞ causing gradient explosion when
    two boxes nearly fully overlap (common in dense UAV scenes).
    """
    return torch.where(
        x <= sigma,
        -torch.log((1.0 - x).clamp(min=1e-7)),
        (x - sigma) / (1.0 - sigma) - _math.log(1.0 - sigma),
    )


def _iog(box_g: torch.Tensor, box_p: torch.Tensor) -> torch.Tensor:
    """Intersection over Ground-truth area (IoG).

    box_g: (N, 4) xyxy — the GT boxes we must NOT trespass into
    box_p: (N, 4) xyxy — predicted boxes
    Returns: (N,) ∈ [0, 1]

    Unlike IoU, IoG measures how much of the GT box is covered by pred.
    Used for RepGT: we want pred to stay OUT of non-assigned GT boxes.
    """
    inter_x1 = torch.max(box_g[:, 0], box_p[:, 0])
    inter_y1 = torch.max(box_g[:, 1], box_p[:, 1])
    inter_x2 = torch.min(box_g[:, 2], box_p[:, 2])
    inter_y2 = torch.min(box_g[:, 3], box_p[:, 3])
    inter = (torch.clamp(inter_x2 - inter_x1, min=0) *
             torch.clamp(inter_y2 - inter_y1, min=0))
    area_g = ((box_g[:, 2] - box_g[:, 0]) *
              (box_g[:, 3] - box_g[:, 1])).clamp(min=1e-6)
    return inter / area_g


def _get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _is_dist_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# ---------------------------------------------------------------------------
# Triplet loss (hard-mining)
# ---------------------------------------------------------------------------
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        if n < 2:
            return inputs.sum() * 0
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            pos = dist[i][mask[i]]
            neg = dist[i][~mask[i]]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            dist_ap.append(pos.max().unsqueeze(0))
            dist_an.append(neg.min().unsqueeze(0))
        if not dist_ap:
            return inputs.sum() * 0
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        return self.ranking_loss(dist_an, dist_ap, torch.ones_like(dist_an))


# ---------------------------------------------------------------------------
# Main criterion
# ---------------------------------------------------------------------------
class ECDetJDECriterion(nn.Module):
    """
    Detection losses mirror EdgeCrafter ECCriterion exactly.
    ReID loss is appended after all detection losses.
    """

    def __init__(self,
                 matcher: HungarianMatcher,
                 num_classes: int,
                 nid_dict: dict,
                 reid_dim: int = 128,
                 weight_dict: dict = None,
                 losses=('mal', 'boxes', 'local'),
                 alpha: float = 0.75,
                 gamma: float = 1.5,
                 reg_max: int = 32,
                 boxes_weight_format=None,
                 use_uni_set: bool = True,
                 id_weight: float = 1.0,
                 use_triplet: bool = False,
                 ):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.nid_dict = nid_dict
        self.reid_dim = reid_dim
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.boxes_weight_format = boxes_weight_format
        self.use_uni_set = use_uni_set
        self.id_weight = id_weight
        self.use_triplet = use_triplet

        self.weight_dict = weight_dict or {
            'loss_cls':  2.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
        }

        # Per-class ReID classifiers (CE head)
        self.classifiers = nn.ModuleDict()
        self.emb_scale_dict = {}
        for cls_id, nid in nid_dict.items():
            self.classifiers[str(cls_id)] = nn.Linear(reid_dim, nid)
            self.emb_scale_dict[cls_id] = math.sqrt(2) * math.log(nid - 1) if nid > 1 else 1.0

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.triplet = TripletLoss(margin=0.3)

    # ------------------------------------------------------------------
    # Detection losses  (RF-DETR style: focal + L1 + GIoU, no MAL/DFL)
    # ------------------------------------------------------------------
    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        """Sigmoid focal loss (RF-DETR style).

        Pure focal loss — no IoU weighting, no soft labels.
        Target = binary one-hot: 1 for matched class, 0 otherwise.
        Much easier to optimize early in training than MAL because gradient
        signal does not depend on box quality (IoU ≈ 0 at init).
        """
        src_logits       = outputs['pred_logits']                         # (B, Q, C)
        idx              = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

        # Build binary one-hot target: shape (B, Q, C)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_onehot  = torch.zeros(
            [*src_logits.shape[:2], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, device=src_logits.device)
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[..., :-1]                           # drop no-obj col

        # Sigmoid focal loss (α=self.alpha, γ=self.gamma)
        prob    = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_onehot, reduction='none')
        p_t     = prob * target_onehot + (1 - prob) * (1 - target_onehot)
        loss    = ce_loss * (1 - p_t).pow(self.gamma)
        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        loss    = alpha_t * loss

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_cls': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        idx       = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight

        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }

    def loss_repulsion(self, outputs, targets, indices, num_boxes):
        """Repulsion Loss — faithful adaptation of Wang et al. CVPR 2018
        to the DETR one-to-one matching context.

        RepGT:
          For each matched prediction P (assigned to GT_i), find the GT box
          with the second-highest IoU to P (call it GT_j, j ≠ i).
          Penalise IoG(GT_j, P) = intersection(GT_j, P) / area(GT_j).
          → Metric is IoG (not IoU) so we penalise how much of GT_j is
            covered by P, regardless of P's size.  Mirrors original paper.
          Skip images with only 1 GT (no second GT exists).

        RepBox:
          For each image, compute IoU between all matched predictions
          (upper-triangle, per-image to avoid cross-image pairs).
          Penalise smooth_ln(IoU) and normalise by number of overlapping pairs.
          → In DETR, 1-to-1 Hungarian means one prediction per GT, so no
            random sampling is needed (unlike anchor-based original).

        smooth_ln (piecewise, σ=0.5):
          x ≤ σ: -log(1-x)
          x > σ: (x-σ)/(1-σ) - log(1-σ)   [linear tail, avoids ∞]
        """
        idx       = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]   # (M_total, 4) cxcywh norm
        zero      = src_boxes.sum() * 0          # scalar zero with grad

        rep_gt_sum,  n_rep_gt  = zero, 0
        rep_box_sum, n_rep_box = zero, 0

        offset = 0
        for b_idx, (src_i, tgt_i) in enumerate(indices):
            m = len(src_i)
            if m == 0:
                offset += m
                continue

            all_gt   = targets[b_idx]['boxes']           # (N_gt, 4) cxcywh
            n_gt     = len(all_gt)
            pred_b   = src_boxes[offset:offset + m]      # (m, 4) cxcywh
            dev      = pred_b.device
            tgt_i_d  = tgt_i.to(dev)

            # ---- RepGT ----
            # Only meaningful when there are ≥2 GT boxes in the image.
            if n_gt >= 2:
                pred_xyxy = box_cxcywh_to_xyxy(pred_b)           # (m, 4)
                gt_xyxy   = box_cxcywh_to_xyxy(all_gt.to(dev))   # (N_gt, 4)

                # IoU between each prediction and ALL GT boxes: (m, N_gt)
                with torch.no_grad():
                    iou_p2gt, _ = box_iou(pred_xyxy, gt_xyxy)

                # Mask out the assigned GT for each prediction → find 2nd-best
                iou_masked = iou_p2gt.clone()
                iou_masked[torch.arange(m, device=dev), tgt_i_d] = -1.0
                _, sec_gt_idx = iou_masked.max(dim=1)            # (m,)

                # IoG(second_GT, pred): how much of the second GT is covered
                iog_vals = _iog(gt_xyxy[sec_gt_idx], pred_xyxy)  # (m,)

                # smooth_ln penalty, averaged over this image's predictions
                rep_gt_sum = rep_gt_sum + _smooth_ln(iog_vals).mean() * m
                n_rep_gt  += m

            # ---- RepBox (per-image) ----
            # Compute IoU between matched predictions within this image only.
            if m >= 2:
                pred_xyxy = box_cxcywh_to_xyxy(pred_b)
                with torch.no_grad():
                    iou_bb, _ = box_iou(pred_xyxy, pred_xyxy)  # (m, m)

                # Upper-triangle, exclude diagonal (self = 1.0)
                iou_bb = iou_bb.triu(diagonal=1)
                active = iou_bb > 0
                if active.any():
                    n_pairs     = active.float().sum().clamp(min=1.0)
                    rep_box_sum = rep_box_sum + _smooth_ln(iou_bb[active]).sum() / n_pairs
                    n_rep_box  += 1   # count images that contributed

            offset += m

        # Normalise consistent with other losses (per GT box)
        denom = max(num_boxes, 1.0)
        rep_gt_term  = (rep_gt_sum  / n_rep_gt)  if n_rep_gt  > 0 else zero
        rep_box_term = (rep_box_sum / n_rep_box) if n_rep_box > 0 else zero

        return {'loss_rep': (rep_gt_term + 0.5 * rep_box_term)}


    # ------------------------------------------------------------------
    # ReID loss — vectorized: one GPU gather per image (not per class×image)
    # ------------------------------------------------------------------
    def loss_reid(self, outputs, targets, indices):
        """CE + optional Triplet on matched queries, per class.

        Restructured so pred_reid[b_idx] is gathered ONCE per image, then
        split by class — reduces GPU gather ops from O(C×B) to O(B).
        """
        if 'pred_reid' not in outputs:
            return {}

        pred_reid = outputs['pred_reid']   # (B, N_det, reid_dim)
        dev = pred_reid.device
        reid_loss = pred_reid.sum() * 0    # scalar zero with grad

        # Single pass over batch: collect embeddings grouped by class
        cls_emb = {cid: [] for cid in self.nid_dict}
        cls_ids = {cid: [] for cid in self.nid_dict}

        for b_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            t       = targets[b_idx]
            tgt_dev = tgt_idx.to(dev)
            src_dev = src_idx.to(dev)

            labels = t['labels'].to(dev)[tgt_dev]       # (M,)
            tids   = t['track_ids'].to(dev)[tgt_dev]    # (M,)
            valid  = tids >= 0
            if not valid.any():
                continue

            # ONE gather per image for all classes
            emb_all = pred_reid[b_idx][src_dev[valid]]  # (M_valid, reid_dim)
            lbl_all = labels[valid]                      # (M_valid,)
            ids_all = tids[valid]                        # (M_valid,)

            for cls_id in self.nid_dict:
                mask = (lbl_all == cls_id)
                if not mask.any():
                    continue
                emb = F.normalize(emb_all[mask]) * self.emb_scale_dict[cls_id]
                cls_emb[cls_id].append(emb)
                cls_ids[cls_id].append(ids_all[mask])

        n_active = 0
        for cls_id in self.nid_dict:
            if not cls_emb[cls_id]:
                continue
            emb_cat = torch.cat(cls_emb[cls_id], dim=0)
            ids_cat = torch.cat(cls_ids[cls_id], dim=0)
            pred    = self.classifiers[str(cls_id)](emb_cat)
            reid_loss = reid_loss + self.ce_loss(pred, ids_cat)
            if self.use_triplet and emb_cat.shape[0] >= 2:
                reid_loss = reid_loss + self.triplet(emb_cat, ids_cat)
            n_active += 1

        # Average across active classes so loss_reid ≈ CE of one class (~log(nIDs))
        # regardless of how many classes appear in the batch.
        if n_active > 1:
            reid_loss = reid_loss / n_active

        return {'loss_reid': reid_loss}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx   = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Union of matched indices across all decoder layers."""
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([i1[0], i2[0]]), torch.cat([i1[1], i2[1]]))
                       for i1, i2 in zip(indices.copy(), indices_aux.copy())]
        results = []
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            sort_idx       = torch.argsort(counts, descending=True)
            unique_sorted  = unique[sort_idx]
            col2row = {}
            for pair in unique_sorted:
                r, c = pair[0].item(), pair[1].item()
                if r not in col2row:
                    col2row[r] = c
            fr = torch.tensor(list(col2row.keys()),   device=ind.device)
            fc = torch.tensor(list(col2row.values()), device=ind.device)
            results.append((fr.long(), fc.long()))
        return results

    def _get_shared_meta(self, outputs, targets, indices):
        """Compute IoU once for boxes weight (GIoU weighting on loss_boxes).

        focal loss does not need IoU — only boxes uses it via boxes_weight.
        Returns empty dict for focal, iou-weight for boxes.
        """
        if self.boxes_weight_format is None:
            return {loss: {} for loss in self.losses}

        idx = self._get_src_permutation_idx(indices)
        src = outputs['pred_boxes'][idx]
        tgt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou = torch.diag(box_iou(
                box_cxcywh_to_xyxy(src.detach()), box_cxcywh_to_xyxy(tgt))[0])
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src.detach()), box_cxcywh_to_xyxy(tgt)))
        else:
            raise AttributeError(f'Unknown boxes_weight_format: {self.boxes_weight_format}')

        return {
            'focal': {},                       # focal loss: no IoU dependency
            'boxes': {'boxes_weight': iou},    # GIoU-weighted L1
            'rep':   {},                       # repulsion: no extra kwargs needed
        }

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'focal': self.loss_labels_focal,
            'boxes': self.loss_boxes,
            'rep':   self.loss_repulsion,
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group    = dn_meta['dn_num_group']
        num_gts  = [len(t['labels']) for t in targets]
        device   = targets[0]['labels'].device
        result   = []
        for i, ng in enumerate(num_gts):
            if ng > 0:
                gt_idx = torch.arange(ng, dtype=torch.int64, device=device).tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                result.append((dn_positive_idx[i], gt_idx))
            else:
                result.append((torch.zeros(0, dtype=torch.int64, device=device),
                               torch.zeros(0, dtype=torch.int64, device=device)))
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, outputs, targets):
        outputs_no_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # ----------------------------------------------------------------
        # PHASE 1: All matcher calls upfront (before any loss computation).
        # Batching them here lets the GPU finish cost-matrix kernels for
        # later layers while the CPU runs linear_sum_assignment for earlier ones,
        # improving CPU-GPU pipeline overlap.
        # ----------------------------------------------------------------
        aux_list = outputs.get('aux_outputs', [])
        if 'pre_outputs' in outputs:
            aux_list = aux_list + [outputs['pre_outputs']]
        enc_list = list(outputs.get('enc_aux_outputs', []))

        indices        = self.matcher(outputs_no_aux, targets)['indices']
        cached_indices = [self.matcher(aux, targets)['indices'] for aux in aux_list]
        cached_enc     = [self.matcher(aux, targets)['indices'] for aux in enc_list]

        if aux_list or enc_list:
            all_aux_indices = cached_indices + cached_enc
            indices_go      = self._get_go_indices(indices, all_aux_indices)

            n_go   = sum(len(x[0]) for x in indices_go)
            n_go_t = torch.as_tensor([n_go], dtype=torch.float,
                                     device=next(iter(outputs.values())).device)
            if _is_dist_available():
                torch.distributed.all_reduce(n_go_t)
            num_boxes_go = torch.clamp(n_go_t / _get_world_size(), min=1).item()
        else:
            indices_go   = indices
            num_boxes_go = 1.0

        num_boxes_t = torch.as_tensor(
            [sum(len(t['labels']) for t in targets)], dtype=torch.float,
            device=next(iter(outputs.values())).device)
        if _is_dist_available():
            torch.distributed.all_reduce(num_boxes_t)
        num_boxes = torch.clamp(num_boxes_t / _get_world_size(), min=1).item()

        # ----------------------------------------------------------------
        # PHASE 2: Compute all losses (RF-DETR style: focal + boxes only).
        # No MAL (IoU-gated), no FGL/DDF (DFL distribution).
        # _get_shared_meta() computes IoU once per layer for boxes_weight.
        # ----------------------------------------------------------------
        losses = {}

        def _apply_losses(out, idx_main, idx_go, nb, nb_go, suffix=''):
            # focal: always use per-layer indices (no union needed)
            # boxes: use union indices (idx_go) for more stable regression
            meta_go   = self._get_shared_meta(out, targets, idx_go)
            meta_main = self._get_shared_meta(out, targets, idx_main)

            for loss in self.losses:
                use_go = self.use_uni_set and loss == 'boxes'
                idx_in = idx_go   if use_go else idx_main
                nb_in  = nb_go    if use_go else nb
                meta   = meta_go.get(loss, {}) if use_go else meta_main.get(loss, {})
                l_dict = self.get_loss(loss, out, targets, idx_in, nb_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                          for k in l_dict if k in self.weight_dict}
                if suffix:
                    l_dict = {k + suffix: v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Main output
        _apply_losses(outputs, indices, indices_go, num_boxes, num_boxes_go)

        # Aux decoder layers
        for i, aux in enumerate(outputs.get('aux_outputs', [])):
            _apply_losses(aux, cached_indices[i], indices_go,
                          num_boxes, num_boxes_go, suffix=f'_aux_{i}')

        # Pre-output (first decoder layer, D-FINE style)
        if 'pre_outputs' in outputs:
            pre_idx_cache = len(cached_indices) - 1
            aux = outputs['pre_outputs']
            _apply_losses(aux, cached_indices[pre_idx_cache], indices_go,
                          num_boxes, num_boxes_go, suffix='_pre')

        # Encoder aux outputs
        if enc_list:
            class_agnostic = outputs.get('enc_meta', {}).get('class_agnostic', False)
            if class_agnostic:
                orig_nc = self.num_classes
                self.num_classes = 1
                # Shallow copy: only replace 'labels', avoid deepcopy of all tensors
                enc_targets = [{**t, 'labels': torch.zeros_like(t['labels'])}
                               for t in targets]
            else:
                enc_targets = targets

            for i, aux in enumerate(enc_list):
                for loss in self.losses:
                    use_go = self.use_uni_set and loss == 'boxes'
                    idx_in = indices_go if use_go else cached_enc[i]
                    nb_in  = num_boxes_go if use_go else num_boxes
                    meta   = self._get_shared_meta(aux, enc_targets, idx_in).get(loss, {})
                    l_dict = self.get_loss(loss, aux, enc_targets, idx_in, nb_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                              for k in l_dict if k in self.weight_dict}
                    losses.update({k + f'_enc_{i}': v for k, v in l_dict.items()})

            if class_agnostic:
                self.num_classes = orig_nc

        # DN losses (denoising groups)
        if 'dn_outputs' in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_nb      = max(num_boxes * outputs['dn_meta']['dn_num_group'], 1)

            for i, aux in enumerate(outputs['dn_outputs']):
                meta_dn = self._get_shared_meta(aux, targets, indices_dn)
                for loss in self.losses:
                    meta   = meta_dn.get(loss, {})
                    l_dict = self.get_loss(loss, aux, targets, indices_dn, dn_nb, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                              for k in l_dict if k in self.weight_dict}
                    losses.update({k + f'_dn_{i}': v for k, v in l_dict.items()})

            if 'dn_pre_outputs' in outputs:
                aux     = outputs['dn_pre_outputs']
                meta_dn = self._get_shared_meta(aux, targets, indices_dn)
                for loss in self.losses:
                    meta   = meta_dn.get(loss, {})
                    l_dict = self.get_loss(loss, aux, targets, indices_dn, dn_nb, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0)
                              for k in l_dict if k in self.weight_dict}
                    losses.update({k + '_dn_pre': v for k, v in l_dict.items()})

        # ReID loss (main output only)
        if self.id_weight > 0:
            reid_dict = self.loss_reid(outputs, targets, indices)
            losses.update({k: v * self.id_weight for k, v in reid_dict.items()})

        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

        # loss_main: main output det+reid only (for display).
        # loss: full sum across all aux heads (for backward).
        _main_keys = {'loss_cls', 'loss_bbox', 'loss_giou', 'loss_rep', 'loss_reid'}
        losses['loss_main'] = sum(losses[k] for k in _main_keys if k in losses)
        losses['loss']      = sum(v for k, v in losses.items() if k != 'loss_main')
        return losses
