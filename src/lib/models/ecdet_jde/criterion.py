"""
ECDetJDE Criterion: ECCrafter detection loss + ReID loss.
Detection loss is kept identical to ECCrafter (focal/mal + boxes + local).
ReID loss (CE + Triplet) is computed only on Hungarian-matched queries.
"""

import copy
import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .utils import bbox2distance
from .matcher import HungarianMatcher


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
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


# ---------------------------------------------------------------------------
# Main criterion
# ---------------------------------------------------------------------------
class ECDetJDECriterion(nn.Module):
    """
    Combined detection + ReID criterion for ECDetJDE.

    Detection losses are identical to ECCrafter (Hungarian matching,
    focal/mal/vfl + boxes + local). ReID loss is added on top:
      reid_loss = CE(classifier(F.normalize(reid_emb)), track_id)
               + TripletLoss(F.normalize(reid_emb), track_id)
    """

    def __init__(self,
                 matcher: HungarianMatcher,
                 num_classes: int,
                 nid_dict: dict,        # {cls_id: num_identities}
                 reid_dim: int = 128,
                 weight_dict: dict = None,
                 losses=('mal', 'boxes'),
                 alpha: float = 0.2,
                 gamma: float = 2.0,
                 reg_max: int = 32,
                 boxes_weight_format=None,
                 use_uni_set: bool = True,
                 id_weight: float = 1.0,
                 use_triplet: bool = True,
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
            'loss_mal': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0,
        }

        # Per-class ReID classifiers (CE head)
        self.classifiers = nn.ModuleDict()
        self.emb_scale_dict = {}
        for cls_id, nid in nid_dict.items():
            self.classifiers[str(cls_id)] = nn.Linear(reid_dim, nid)
            self.emb_scale_dict[cls_id] = math.sqrt(2) * math.log(nid - 1) if nid > 1 else 1.0

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.triplet = TripletLoss(margin=0.3)

        # Learnable uncertainty weights (same as McMotLoss)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id  = nn.Parameter(-1.05 * torch.ones(1))

        # Cache for FGL/DDF losses
        self.fgl_targets = self.fgl_targets_dn = None
        self.own_targets = self.own_targets_dn = None
        self.num_pos = self.num_neg = None

    # ------------------------------------------------------------------
    # Detection losses (ported from ECCriterion)
    # ------------------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx   = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes    = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _      = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits         = outputs['pred_logits']
        target_classes_o   = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes     = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target             = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o     = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score       = target_score_o.unsqueeze(-1) * target

        pred_score  = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        weight      = pred_score.pow(self.gamma) * (1 - target) + target
        loss        = F.binary_cross_entropy_with_logits(src_logits, target_score,
                                                          weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        idx        = self._get_src_permutation_idx(indices)
        src_boxes  = outputs['pred_boxes'][idx]
        tgt_boxes  = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox  = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_giou  = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))

        losses = {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': (loss_giou if boxes_weight is None else loss_giou * boxes_weight).sum() / num_boxes,
        }
        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        losses = {}
        if 'pred_corners' not in outputs:
            return losses

        idx          = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        pred_corners = outputs['pred_corners'][idx].reshape(-1, self.reg_max + 1)
        ref_points   = outputs['ref_points'][idx].detach()

        with torch.no_grad():
            key = 'fgl_targets_dn' if 'is_dn' in outputs else 'fgl_targets'
            if getattr(self, key) is None:
                setattr(self, key, bbox2distance(
                    ref_points, box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max, outputs['reg_scale'], outputs['up']))

        target_corners, weight_right, weight_left = getattr(self, key)
        ious = torch.diag(box_iou(
            box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]),
            box_cxcywh_to_xyxy(target_boxes))[0])
        weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

        losses['loss_fgl'] = self._unimodal_focal(
            pred_corners, target_corners, weight_right, weight_left, weight_targets, num_boxes)
        return losses

    def _unimodal_focal(self, pred, label, w_right, w_left, weight=None, avg_factor=None):
        dis_left  = label.long()
        dis_right = dis_left + 1
        loss = (F.cross_entropy(pred, dis_left,  reduction='none') * w_left.reshape(-1)
              + F.cross_entropy(pred, dis_right, reduction='none') * w_right.reshape(-1))
        if weight is not None:
            loss = loss * weight.float()
        return (loss.sum() / avg_factor) if avg_factor is not None else loss.sum()

    # ------------------------------------------------------------------
    # ReID loss
    # ------------------------------------------------------------------
    def loss_reid(self, outputs, targets, indices, num_boxes):
        """CE + optional Triplet on matched queries, per class."""
        if 'pred_reid' not in outputs:
            return {}

        pred_reid = outputs['pred_reid']   # (B, N_det, reid_dim)
        dev = pred_reid.device
        reid_loss = torch.zeros(1, device=dev, requires_grad=False).squeeze()

        for cls_id, nid in self.nid_dict.items():
            all_emb, all_ids = [], []
            for b_idx, (src_idx, tgt_idx) in enumerate(indices):
                if len(src_idx) == 0:
                    continue
                t = targets[b_idx]
                # Ensure indices are on the same device as target tensors.
                # src_idx/tgt_idx come from the matcher (GPU); track_ids may
                # still be on CPU if the dataloader didn't move it to GPU.
                tgt_idx_dev = tgt_idx.to(dev)
                src_idx_dev = src_idx.to(dev)

                # filter to this class
                cls_mask  = (t['labels'].to(dev)[tgt_idx_dev] == cls_id)
                if cls_mask.sum() == 0:
                    continue
                track_ids = t['track_ids'].to(dev)[tgt_idx_dev[cls_mask]]  # (M,)
                valid     = track_ids >= 0
                if valid.sum() == 0:
                    continue
                emb = pred_reid[b_idx][src_idx_dev[cls_mask][valid]]   # (M, reid_dim)
                emb = F.normalize(emb) * self.emb_scale_dict[cls_id]
                all_emb.append(emb)
                all_ids.append(track_ids[valid])

            if not all_emb:
                continue

            emb_cat = torch.cat(all_emb, dim=0)     # (K, reid_dim)
            ids_cat = torch.cat(all_ids, dim=0)     # (K,)
            pred    = self.classifiers[str(cls_id)](emb_cat)
            n_valid = float(ids_cat.numel())

            ce = self.ce_loss(pred, ids_cat) / n_valid
            reid_loss = reid_loss + ce

            if self.use_triplet and emb_cat.shape[0] >= 2:
                reid_loss = reid_loss + self.triplet(emb_cat, ids_cat) / n_valid

        return {'loss_reid': reid_loss}

    # ------------------------------------------------------------------
    # Helpers for go-indices (union across layers)
    # ------------------------------------------------------------------
    def _get_go_indices(self, indices, indices_aux_list):
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([i1[0], i2[0]]), torch.cat([i1[1], i2[1]]))
                       for i1, i2 in zip(indices, indices_aux)]
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
            fr = torch.tensor(list(col2row.keys()), device=ind.device)
            fc = torch.tensor(list(col2row.values()), device=ind.device)
            results.append((fr.long(), fc.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets = self.fgl_targets_dn = None
        self.own_targets = self.own_targets_dn = None
        self.num_pos = self.num_neg = None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'mal':   self.loss_labels_mal,
            'boxes': self.loss_boxes,
            'local': self.loss_local,
            'reid':  self.loss_reid,
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, outputs, targets):
        outputs_no_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(outputs_no_aux, targets)['indices']
        self._clear_cache()

        # Build union indices across all decoder layers for boxes/local
        indices_aux_list, cached_indices, cached_enc = [], [], []
        if 'aux_outputs' in outputs:
            all_aux = outputs['aux_outputs'] + ([outputs['pre_outputs']] if 'pre_outputs' in outputs else [])
            for aux in all_aux:
                idx = self.matcher(aux, targets)['indices']
                cached_indices.append(idx)
                indices_aux_list.append(idx)
            for aux in outputs.get('enc_aux_outputs', []):
                idx = self.matcher(aux, targets)['indices']
                cached_enc.append(idx)
                indices_aux_list.append(idx)
            indices_go = self._get_go_indices(indices, indices_aux_list)
            n_go = sum(len(x[0]) for x in indices_go)
            n_go_t = torch.as_tensor([n_go], dtype=torch.float,
                                     device=next(iter(outputs.values())).device)
            if _is_dist_available():
                torch.distributed.all_reduce(n_go_t)
            num_boxes_go = max(n_go_t.item() / _get_world_size(), 1.0)
        else:
            indices_go   = indices
            num_boxes_go = 1.0

        num_boxes_t = torch.as_tensor(
            [sum(len(t['labels']) for t in targets)], dtype=torch.float,
            device=next(iter(outputs.values())).device)
        if _is_dist_available():
            torch.distributed.all_reduce(num_boxes_t)
        num_boxes = max(num_boxes_t.item() / _get_world_size(), 1.0)

        losses = {}

        # Main detection losses
        for loss in self.losses:
            if loss == 'reid':
                continue
            use_go     = self.use_uni_set and loss in ('boxes', 'local')
            idx_in     = indices_go if use_go else indices
            nb_in      = num_boxes_go if use_go else num_boxes
            meta       = self._get_meta(loss, outputs, targets, idx_in)
            l_dict     = self.get_loss(loss, outputs, targets, idx_in, nb_in, **meta)
            l_dict     = {k: v * self.weight_dict.get(k, 1.0) for k, v in l_dict.items()}
            losses.update(l_dict)

        # Aux detection losses
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:
                    aux['up'], aux['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss == 'reid':
                        continue
                    use_go = self.use_uni_set and loss in ('boxes', 'local')
                    idx_in = indices_go if use_go else cached_indices[i]
                    nb_in  = num_boxes_go if use_go else num_boxes
                    meta   = self._get_meta(loss, aux, targets, idx_in)
                    l_dict = self.get_loss(loss, aux, targets, idx_in, nb_in, **meta)
                    l_dict = {k: v * self.weight_dict.get(k, 1.0) for k, v in l_dict.items()}
                    losses.update({k + f'_aux_{i}': v for k, v in l_dict.items()})

        # DN losses
        if 'dn_outputs' in outputs:
            dn_meta    = outputs['dn_meta']
            indices_dn = self._get_cdn_matched_indices(dn_meta, targets)
            dn_nb      = num_boxes * dn_meta['dn_num_group']
            dn_nb      = max(dn_nb, 1)
            for i, aux in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:
                    aux['is_dn'] = True
                    aux['up'], aux['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss == 'reid':
                        continue
                    meta   = self._get_meta(loss, aux, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux, targets, indices_dn, dn_nb, **meta)
                    l_dict = {k: v * self.weight_dict.get(k, 1.0) for k, v in l_dict.items()}
                    losses.update({k + f'_dn_{i}': v for k, v in l_dict.items()})

        # ReID loss — only on main output, only on matched detection queries
        if 'reid' in self.losses and self.id_weight > 0:
            reid_dict = self.loss_reid(outputs, targets, indices, num_boxes)
            losses.update({k: v * self.id_weight for k, v in reid_dict.items()})

        # Uncertainty-weighted total (same scheme as McMotLoss)
        det_loss  = sum(v for k, v in losses.items() if 'reid' not in k)
        reid_loss = losses.get('loss_reid', torch.zeros(1, device=self.s_det.device).squeeze())

        total = 0.5 * (torch.exp(-self.s_det) * det_loss
                     + torch.exp(-self.s_id)  * reid_loss
                     + self.s_det + self.s_id)
        losses['loss'] = total
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def _get_meta(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None or loss not in ('boxes',):
            return {}
        idx       = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        iou, _    = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(tgt_boxes))
        return {'boxes_weight': torch.diag(iou)}

    @staticmethod
    def _get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group    = dn_meta['dn_num_group']
        num_gts  = [len(t['labels']) for t in targets]
        device   = targets[0]['labels'].device
        result   = []
        for i, ng in enumerate(num_gts):
            if ng > 0:
                gt_idx = torch.arange(ng, dtype=torch.int64, device=device).tile(dn_num_group)
                result.append((dn_positive_idx[i], gt_idx))
            else:
                result.append((torch.zeros(0, dtype=torch.int64, device=device),
                               torch.zeros(0, dtype=torch.int64, device=device)))
        return result
