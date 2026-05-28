"""
DeimMotLoss + DeimMotTrainer — McMotLoss-style training for DEIMMotNet.

Loss formula (identical to AMOT's McMotLoss):
    det_loss  = hm_weight * hm_loss + wh_weight * wh_loss + off_weight * off_loss
    reid_loss = Σ_cls [ CE(emb_scale * norm(id_feat), cls_tr_ids)
                         + tri * TripletLoss(id_feat, cls_tr_ids) ] / N_obj
    total     = exp(−s_det) * det_loss + exp(−s_id) * reid_loss + (s_det + s_id)
    total    *= 0.5

s_det and s_id are learnable temperature parameters that balance detection
and ReID losses without manual weight tuning (Kendall et al. 2018).
"""
from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.base_losses import (
    FocalLoss, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, TripletLoss,
)
from lib.models.decode import mot_decode
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
from lib.utils.post_process import ctdet_post_process
from lib.utils.det_eval import COCOEvaluator, VISDRONE_CLASSES
from .base_trainer import BaseTrainer


# ── Loss ───────────────────────────────────────────────────────────────────────

class DeimMotLoss(nn.Module):
    """
    McMotLoss adapted for DEIMMotNet.

    Required batch keys:
        'hm'        : (B, C, H, W)            Gaussian-rendered heatmap GT
        'wh'        : (B, max_obj, 2)          WH at peak locations
        'reg'       : (B, max_obj, 2)          Sub-pixel offset
        'ind'       : (B, max_obj)             Flat spatial index of each peak
        'reg_mask'  : (B, max_obj)             1 for valid objects, 0 for padding
        'cls_id_map': (B, 1, H, W)             Class ID at each heatmap location (−1 = bg)
        'cls_tr_ids': (B, num_classes, H, W)   Track ID at each location (0-indexed, −1 = ignore)
    """

    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        # Heatmap loss
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()

        # Box regression losses
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else RegLoss()
        self.crit_wh  = (
            NormRegL1Loss()   if opt.norm_wh    else
            RegWeightedL1Loss() if opt.cat_spec_wh else
            self.crit_reg
        )

        # ReID losses
        if opt.id_weight > 0:
            self.emb_dim  = opt.reid_dim
            self.nID_dict = opt.nID_dict   # {cls_id: num_identities}

            # Per-class linear classifier for softmax-CE ReID
            self.classifiers = nn.ModuleDict()
            self.emb_scale_dict: Dict[int, float] = {}
            for cls_id, nID in self.nID_dict.items():
                self.classifiers[str(cls_id)] = nn.Linear(self.emb_dim, nID)
                # angular margin scale: sqrt(2) * log(nID − 1)  (matches AMOT)
                self.emb_scale_dict[cls_id] = math.sqrt(2) * math.log(max(nID - 1, 1))

            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.TriLoss = TripletLoss()
            self.s_id    = nn.Parameter(-1.05 * torch.ones(1))

        # Learnable detection loss temperature (init = exp(1.85) ≈ 6.36 scale)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))

    def forward(self, outputs, batch) -> tuple:
        opt = self.opt
        hm_loss = wh_loss = off_loss = reid_loss = 0.0

        for s in range(opt.num_stacks):
            output = outputs[s]
            dev    = output['hm'].device

            # ── Detection ──────────────────────────────────────────────────────
            # Apply sigmoid + numerical clamp (avoids log(0) in FocalLoss)
            hm = _sigmoid(output['hm'])
            hm_loss += self.crit(hm, batch['hm'].to(dev)) / opt.num_stacks

            if opt.wh_weight > 0:
                wh_loss += self.crit_wh(
                    output['wh'],
                    batch['reg_mask'].to(dev),
                    batch['ind'].to(dev),
                    batch['wh'].to(dev),
                ) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(
                    output['reg'],
                    batch['reg_mask'].to(dev),
                    batch['ind'].to(dev),
                    batch['reg'].to(dev),
                ) / opt.num_stacks

            # ── ReID ───────────────────────────────────────────────────────────
            if opt.id_weight > 0:
                cls_id_map = batch['cls_id_map'].to(dev)   # (B, 1, H, W)
                cls_tr_ids = batch['cls_tr_ids'].to(dev)   # (B, num_cls, H, W)

                for cls_id, _nID in self.nID_dict.items():
                    # pixel positions belonging to this class
                    inds = torch.where(cls_id_map == cls_id)   # (B_idx, 0, H_idx, W_idx)
                    if inds[0].shape[0] == 0:
                        continue

                    # Gather id features at object center locations → (N, reid_dim)
                    feat = output['id'][inds[0], :, inds[2], inds[3]]
                    feat = self.emb_scale_dict[cls_id] * F.normalize(feat, dim=1)

                    # Ground-truth track IDs at same locations
                    target_ids = cls_tr_ids[inds[0], cls_id, inds[2], inds[3]]

                    pred = self.classifiers[str(cls_id)](feat)
                    n    = max(float(target_ids.nelement()), 1.0)

                    if opt.tri:
                        reid_loss += (
                            self.ce_loss(pred, target_ids)
                            + self.TriLoss(feat, target_ids)
                        ) / n
                    else:
                        reid_loss += self.ce_loss(pred, target_ids) / n

        # ── Combine with learned temperature scaling ───────────────────────────
        det_loss = (
            opt.hm_weight  * hm_loss
            + opt.wh_weight  * wh_loss
            + opt.off_weight * off_loss
        )

        if opt.id_weight > 0:
            loss = (
                torch.exp(-self.s_det) * det_loss
                + torch.exp(-self.s_id) * reid_loss
                + (self.s_det + self.s_id)
            )
        else:
            loss = torch.exp(-self.s_det) * det_loss + self.s_det

        loss *= 0.5

        def _t(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(float(v))

        loss_stats: Dict[str, Any] = {
            'loss':     loss,
            'hm_loss':  _t(hm_loss),
            'wh_loss':  _t(wh_loss),
            'off_loss': _t(off_loss),
        }
        if opt.id_weight > 0:
            loss_stats['id_loss'] = _t(reid_loss)

        return loss, loss_stats


# ── Trainer ────────────────────────────────────────────────────────────────────

def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1).clip(0, 1)


class DeimMotTrainer(BaseTrainer):
    """Trainer for DEIMMotNet using McMotLoss-style loss."""

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        if opt.id_weight > 0:
            loss_states.append('id_loss')
        return loss_states, DeimMotLoss(opt)

    def save_result(self, output, batch, results) -> None:
        out = output[0] if isinstance(output, (list, tuple)) else output
        reg = out.get('reg') if self.opt.reg_offset else None
        dets = mot_decode(
            heatmap=_sigmoid(out['hm']),
            wh=out['wh'],
            reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh,
            K=self.opt.K,
        )
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(),
            batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            out['hm'].shape[2],
            out['hm'].shape[3],
            out['hm'].shape[1],
        )
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

    def debug(self, batch, output, iter_id) -> None:
        pass

    def evaluate(
        self,
        epoch: int,
        val_loader,
        logger=None,
        score_thr: float = 0.3,
    ) -> Dict[str, float]:
        """Run model over val_loader and return per-class AP@50 + mAP50."""
        opt = self.opt
        mwl = self.model_with_loss
        if self.ema is not None:
            model = self.ema.module
        else:
            model = mwl.module.model if hasattr(mwl, 'module') else mwl.model
        model.eval()

        ev = COCOEvaluator(
            num_classes=opt.num_classes,
            class_names=VISDRONE_CLASSES[:opt.num_classes],
        )

        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to(opt.device, non_blocking=True)

                output = model(batch['input'])
                out    = output[0]

                # Decode detections: (B, K, 6) → [x1, y1, x2, y2, score, cls]
                reg  = out.get('reg') if opt.reg_offset else None
                dets = mot_decode(
                    heatmap=_sigmoid(out['hm']),
                    wh=out['wh'],
                    reg=reg,
                    cat_spec_wh=opt.cat_spec_wh,
                    K=opt.K,
                )
                dets_np = dets.detach().cpu().numpy()  # (B, K, 6)

                B = batch['input'].shape[0]
                for b in range(B):
                    det_b = dets_np[b]                    # (K, 6): x1y1x2y2 in feat coords
                    scores = det_b[:, 4]
                    labels = det_b[:, 5].astype(np.int64)

                    # Rescale from feature map → normalised [0,1]
                    H_feat = out['hm'].shape[2]
                    W_feat = out['hm'].shape[3]
                    boxes  = det_b[:, :4].copy()
                    boxes[:, [0, 2]] /= W_feat
                    boxes[:, [1, 3]] /= H_feat
                    boxes = boxes.clip(0, 1).astype(np.float32)

                    keep = scores >= score_thr
                    pred_boxes  = boxes[keep]
                    pred_scores = scores[keep].astype(np.float32)
                    pred_labels = labels[keep]

                    gt = batch['targets'][b]
                    gt_boxes_raw = gt['boxes'].cpu().numpy()   # cxcywh [0,1]
                    gt_labels    = gt['labels'].cpu().numpy().astype(np.int64)
                    if len(gt_boxes_raw) > 0:
                        gt_boxes = _cxcywh_to_xyxy(gt_boxes_raw).astype(np.float32)
                    else:
                        gt_boxes  = np.zeros((0, 4), dtype=np.float32)
                        gt_labels = np.zeros((0,),   dtype=np.int64)

                    ev.update(pred_boxes, pred_scores, pred_labels,
                              gt_boxes,  gt_labels)

        stats = ev.summarize()
        print(f'\n[eval] epoch {epoch:03d}')
        ev.print_summary(stats)

        if logger is not None:
            for k, v in stats.items():
                logger.scalar_summary(f'val_{k}', v, epoch)

        if self.ema is None:
            model.train()
        return stats
