"""
DetrMotLoss + DetrMotTrainer — Full-DETR JDE training for DEIMv2JDE.

Loss formula:
    L = L_det + id_weight · L_reid

L_det  : DEIMCriterion weighted sum (loss_bbox×5, loss_giou×2, loss_mal×1 …)
         averaged over decoder/DN/enc layer groups → single-layer scale (~4-5).
L_reid : per-class softmax-CE + Triplet on Hungarian-matched queries (~1-2).
         id_weight=1 (default) gives reid ~25% of total — matches FairMOT convention.

Batch keys required:
    'input'   : (B, 3, H, W)   normalised image tensor
    'targets' : list of B dicts, each with
        'boxes'     : (N, 4) cxcywh [0,1]
        'labels'    : (N,)   class indices (0-indexed)
        'track_ids' : (N,)   identity labels for ReID (−1 = ignore)
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.base_losses import TripletLoss
from lib.utils.det_eval import COCOEvaluator, VISDRONE_CLASSES
from .base_trainer import BaseTrainer


# ── helpers ────────────────────────────────────────────────────────────────────

def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=-1).clip(0, 1)


# ── Loss ───────────────────────────────────────────────────────────────────────

class DetrMotLoss(nn.Module):
    """
    DEIM detection loss + per-query ReID loss for DEIMv2JDE.

    Args:
        opt        : training options (must contain nID_dict, reid_dim, id_weight, tri)
        criterion  : DEIMCriterion instance (built from YAML config)
        matcher    : HungarianMatcher instance shared with criterion
    """

    def __init__(self, opt, criterion: nn.Module, matcher: nn.Module) -> None:
        super().__init__()
        self.opt       = opt
        self.criterion = criterion
        self.matcher   = matcher

        if opt.id_weight > 0:
            self.emb_dim      = opt.reid_dim
            self.nID_dict     = opt.nID_dict     # {cls_id: num_identities}

            self.classifiers  = nn.ModuleDict()
            self.emb_scale    : Dict[int, float] = {}
            for cls_id, nID in self.nID_dict.items():
                self.classifiers[str(cls_id)] = nn.Linear(self.emb_dim, nID)
                self.emb_scale[cls_id] = math.sqrt(2) * math.log(max(nID - 1, 1))

            self.ce_loss  = nn.CrossEntropyLoss(ignore_index=-1)
            self.TriLoss  = TripletLoss()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_matching_indices(
        self,
        outputs: Dict[str, Any],
        targets: List[Dict[str, Any]],
        epoch: int = 0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Re-run the matcher on outputs_without_aux to get (src, tgt) pairs."""
        outputs_no_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        return self.matcher(outputs_no_aux, targets, epoch=epoch)['indices']

    def _reid_loss(
        self,
        pred_reid: torch.Tensor,               # (B, N, reid_dim)
        targets:   List[Dict[str, Any]],        # batch targets with 'track_ids', 'labels'
        indices:   List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Per-class softmax-CE + Triplet on Hungarian-matched queries.

        Only queries matched to a GT box contribute to the ReID loss.
        Queries matched to objects with track_id == −1 are ignored.
        """
        if not hasattr(self, 'classifiers'):
            return pred_reid.new_zeros(())

        total_loss = pred_reid.new_zeros(())
        n_valid    = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            reid_b  = pred_reid[b][src_idx]                    # (M, reid_dim)
            tgt     = targets[b]
            tids    = tgt.get('track_ids', tgt.get('ids')).to(reid_b.device)  # (N_gt,)
            clabels = tgt['labels'].to(reid_b.device)          # (N_gt,)

            matched_tids   = tids[tgt_idx]                     # (M,)
            matched_labels = clabels[tgt_idx]                  # (M,)

            for cls_id in self.nID_dict:
                mask = (matched_labels == cls_id) & (matched_tids >= 0)
                if mask.sum() == 0:
                    continue

                feat = self.emb_scale[cls_id] * F.normalize(reid_b[mask], dim=-1)
                tid  = matched_tids[mask]
                pred = self.classifiers[str(cls_id)](feat)

                if getattr(self.opt, 'tri', False):
                    total_loss = total_loss + self.ce_loss(pred, tid) + self.TriLoss(feat, tid)
                else:
                    total_loss = total_loss + self.ce_loss(pred, tid)

                n_valid += mask.sum().item()

        return total_loss / max(n_valid, 1)

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Tuple:
        opt     = self.opt
        targets = batch.get('targets', [])

        # targets come from DataLoader on CPU; move to the same device as outputs
        device = outputs['pred_logits'].device
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        # ── DETR detection loss ────────────────────────────────────────────────
        epoch   = batch.get('epoch', 0)
        det_losses = self.criterion(outputs, targets, epoch=epoch)
        det_loss   = sum(det_losses.values())

        # Average over decoder layer groups so det_loss sits at single-layer scale
        # (~4-5). The weight_dict factors (loss_bbox×5, loss_giou×2, loss_mal×1 …)
        # are preserved inside the criterion — we only average the repeated layers.
        # This lets id_weight=1 give reid ~25% contribution, matching FairMOT convention.
        _AUX_TAGS = ('_aux_', '_dn_', '_enc_', '_pre')
        n_base   = sum(1 for k in det_losses if not any(t in k for t in _AUX_TAGS))
        n_groups = max(len(det_losses) / max(n_base, 1), 1)
        det_loss_avg = det_loss / n_groups

        # ── ReID loss ──────────────────────────────────────────────────────────
        # Reuse the final-layer indices already computed inside criterion.forward()
        # instead of running the Hungarian matcher a second time.
        if opt.id_weight > 0 and 'pred_reid' in outputs:
            indices   = self.criterion._last_indices
            reid_loss = self._reid_loss(outputs['pred_reid'], targets, indices)
        else:
            reid_loss = outputs['pred_logits'].new_zeros(())

        loss = det_loss_avg + opt.id_weight * reid_loss

        # ── Stats ──────────────────────────────────────────────────────────────
        def _t(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(float(v))

        _zero = loss.new_zeros(())
        loss_stats: Dict[str, Any] = {
            'loss':       loss,
            'det_loss':   _t(det_loss_avg),
            'reid_loss':  _t(reid_loss),
            'loss_cls':   _zero,
            'loss_bbox':  _zero,
            'loss_giou':  _zero,
        }
        # Accumulate per-layer sub-losses across all decoder/DN/encoder layers.
        # 'loss_cls' sums all cls variants; 'loss_bbox'/'loss_giou' also sum all layers.
        _CLS_KEYS = frozenset(('loss_focal', 'loss_vfl', 'loss_mal'))
        _BOX_KEYS = frozenset(('loss_bbox', 'loss_giou'))
        for k, v in det_losses.items():
            # strip layer suffix: _aux_0, _dn_1, _enc_0, _pre
            base_key = k.split('_aux_')[0].split('_dn_')[0].split('_enc_')[0].split('_pre')[0]
            if base_key in _CLS_KEYS:
                loss_stats['loss_cls'] = loss_stats['loss_cls'] + _t(v)
            elif base_key in _BOX_KEYS:
                loss_stats[base_key] = loss_stats[base_key] + _t(v)

        return loss, loss_stats


# ── Trainer ────────────────────────────────────────────────────────────────────

class DetrMotTrainer(BaseTrainer):
    """Trainer for DEIMv2JDE using full DETR detection + per-query ReID."""

    def _get_losses(self, opt):
        # Build DEIM criterion from YAML config (same config used to build the model)
        deim_config = getattr(opt, 'deim_config', '')
        if not deim_config:
            raise ValueError('--deim_config required for deimv2_jde trainer')

        _models = os.path.join(os.path.dirname(__file__), '..', 'models')
        _models = os.path.normpath(_models)
        if _models not in sys.path:
            sys.path.insert(0, _models)

        import engine
        from engine.core import YAMLConfig

        cfg = YAMLConfig(deim_config)
        # Override num_classes to match dataset
        for cls_name in ('DEIMCriterion', 'DEIMTransformer', 'DFINETransformer'):
            if cls_name in cfg.global_cfg:
                cfg.global_cfg[cls_name]['num_classes'] = opt.num_classes

        criterion = cfg.criterion   # DEIMCriterion built from YAML
        matcher   = criterion.matcher

        loss_states = [
            'loss', 'det_loss', 'reid_loss',
            'loss_cls', 'loss_bbox', 'loss_giou',
        ]
        return loss_states, DetrMotLoss(opt, criterion, matcher)

    # ── batch helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _move_targets(targets, device):
        """Move DETR-format target dicts to device."""
        out = []
        for t in targets:
            out.append({k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()})
        return out

    # ── save / debug ───────────────────────────────────────────────────────────

    def save_result(self, output, batch, results) -> None:
        pass  # not used for DETR-format inference

    def debug(self, batch, output, iter_id) -> None:
        pass

    # ── validation ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        epoch: int,
        val_loader,
        logger=None,
        score_thr: float = 0.3,
    ) -> Dict[str, float]:
        """Evaluate mAP50 on validation set using DETR output."""
        opt = self.opt
        mwl = self.model_with_loss
        model = (self.ema.module
                 if self.ema is not None
                 else (mwl.module.model if hasattr(mwl, 'module') else mwl.model))
        model.eval()

        ev = COCOEvaluator(
            num_classes=opt.num_classes,
            class_names=VISDRONE_CLASSES[:opt.num_classes],
        )

        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    if k != 'meta' and k != 'targets':
                        batch[k] = batch[k].to(opt.device, non_blocking=True)

                output = model(batch['input'])

                boxes_t  = output['pred_boxes']   # (B, N, 4) cxcywh [0,1]
                logits_t = output['pred_logits']  # (B, N, C)
                probs    = logits_t.sigmoid()
                scores_t, labels_t = probs.max(dim=-1)

                B = batch['input'].shape[0]
                for b in range(B):
                    scores = scores_t[b].cpu().numpy()
                    labels = labels_t[b].cpu().numpy().astype(np.int64)
                    boxes  = boxes_t[b].cpu().numpy()

                    keep        = scores >= score_thr
                    pred_boxes  = _cxcywh_to_xyxy(boxes[keep]).astype(np.float32)
                    pred_scores = scores[keep].astype(np.float32)
                    pred_labels = labels[keep]

                    gt = batch['targets'][b]
                    gt_boxes_raw = gt['boxes'].cpu().numpy()
                    gt_labels    = (gt['labels'].cpu().numpy().astype(np.int64)
                                   if gt['labels'].ndim == 1
                                   else gt['labels'][:, 0].cpu().numpy().astype(np.int64))
                    if len(gt_boxes_raw) > 0:
                        gt_boxes = _cxcywh_to_xyxy(gt_boxes_raw).astype(np.float32)
                    else:
                        gt_boxes  = np.zeros((0, 4), dtype=np.float32)
                        gt_labels = np.zeros((0,),   dtype=np.int64)

                    ev.update(pred_boxes, pred_scores, pred_labels,
                              gt_boxes,  gt_labels)

        stats = ev.summarize()
        print(f'\n[eval] epoch {epoch:03d}  (DEIMv2JDE)')
        ev.print_summary(stats)

        if logger is not None:
            for k, v in stats.items():
                logger.scalar_summary(f'val_{k}', v, epoch)

        if self.ema is None:
            model.train()
        return stats
