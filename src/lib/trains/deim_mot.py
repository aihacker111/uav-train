"""
DeimMotLoss + DeimMotTrainer — McMotLoss-style training for DEIMMotNet.

Loss formula — Kendall et al., CVPR 2018, eq. 9 (flat, one σ per sub-loss):
    L = ½ · Σᵢ [ exp(−sᵢ) · Lᵢ + sᵢ ]

    sᵢ = log(σᵢ²) is a learnable scalar; σᵢ is the homoscedastic uncertainty.
    Active sub-losses: hm, wh, off, [iou], [repul], [id].
    Manual hm_weight / wh_weight / … are superseded by exp(−sᵢ).
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
    ciou_loss, VarifocalLoss, LogWHLoss, repulsion_loss,
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
        if opt.mse_loss:
            self.crit = torch.nn.MSELoss()
        elif getattr(opt, 'vfl', False):
            self.crit = VarifocalLoss()
        else:
            self.crit = FocalLoss()
        self.use_vfl = getattr(opt, 'vfl', False) and not opt.mse_loss

        # Box regression losses
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else RegLoss()
        self.use_log_wh = getattr(opt, 'log_wh', False)
        if self.use_log_wh:
            self.crit_wh = LogWHLoss()
        elif opt.norm_wh:
            self.crit_wh = NormRegL1Loss()
        elif opt.cat_spec_wh:
            self.crit_wh = RegWeightedL1Loss()
        else:
            self.crit_wh = self.crit_reg

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

        # Repulsion loss smoothing threshold (0.0 = plain −log(1 − IoU))
        self.repul_sigma = getattr(opt, 'repul_sigma', 0.0)

        # ── Kendall uncertainty parameters ────────────────────────────────────
        # Kendall et al., CVPR 2018 — "Multi-Task Learning Using Uncertainty
        # to Weigh Losses for Scene Geometry and Semantics", eq. 9.
        #
        #   L = ½ · Σᵢ [ exp(−sᵢ) · Lᵢ + sᵢ ]
        #
        # sᵢ = log(σᵢ²) is a learnable scalar per sub-loss.
        # σᵢ is the homoscedastic (task-specific) uncertainty.
        # Effective weight = exp(−sᵢ); regulariser = sᵢ prevents σᵢ → ∞.
        #
        # Init: 0.0 → weight = 1 at the start of training.
        # 'id' is initialised at −1.05 to preserve prior scaling (exp(1.05) ≈ 2.86×).
        _lv_keys = ['hm', 'wh', 'off']
        if opt.id_weight > 0:
            _lv_keys.append('id')
        if getattr(opt, 'iou_weight',   0.0) > 0:
            _lv_keys.append('iou')
        if getattr(opt, 'repul_weight', 0.0) > 0:
            _lv_keys.append('repul')
        self.log_vars = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(-1.05 if k == 'id' else 0.0))
            for k in _lv_keys
        })

    def forward(self, outputs, batch) -> tuple:
        opt = self.opt
        hm_loss = wh_loss = off_loss = reid_loss = iou_loss = repul = 0.0

        for s in range(opt.num_stacks):
            output = outputs[s]
            dev    = output['hm'].device

            # ── Detection ──────────────────────────────────────────────────────
            hm_gt  = batch['hm'].to(dev)

            if self.use_vfl:
                # Build IoU quality map for positive pixels if CIoU is also enabled
                iou_map = None
                if getattr(opt, 'iou_weight', 0.0) > 0 and opt.reg_offset:
                    W_feat   = output['hm'].shape[3]
                    ind_v    = batch['ind'].to(dev)
                    mask_v   = batch['reg_mask'].to(dev).bool()
                    pred_wh_v  = _tranpose_and_gather_feat(output['wh'],  ind_v)
                    pred_reg_v = _tranpose_and_gather_feat(output['reg'], ind_v)
                    cx_int_v = (ind_v % W_feat).float()
                    cy_int_v = (ind_v // W_feat).float()
                    gt_wh_v  = batch['wh'].to(dev)
                    gt_reg_v = batch['reg'].to(dev)

                    def _xyxy(cx, cy, wh):
                        return torch.stack([cx - wh[...,0]*.5, cy - wh[...,1]*.5,
                                            cx + wh[...,0]*.5, cy + wh[...,1]*.5], -1)

                    pred_wh_dec = (torch.exp(pred_wh_v) if self.use_log_wh
                                   else pred_wh_v)
                    pb = _xyxy(cx_int_v + pred_reg_v[...,0], cy_int_v + pred_reg_v[...,1], pred_wh_dec)
                    gb = _xyxy(cx_int_v + gt_reg_v[...,0],  cy_int_v + gt_reg_v[...,1],  gt_wh_v)

                    # Compute per-object IoU for valid objects only
                    B, max_obj = ind_v.shape
                    H_f, W_f = output['hm'].shape[2], W_feat
                    iou_map = output['hm'].new_zeros(B, opt.num_classes, H_f, W_f)
                    for b in range(B):
                        valid_idx = mask_v[b].nonzero(as_tuple=True)[0]
                        if len(valid_idx) == 0:
                            continue
                        pb_v = pb[b][valid_idx]; gb_v = gb[b][valid_idx]
                        ix1 = torch.max(pb_v[:,0], gb_v[:,0]); iy1 = torch.max(pb_v[:,1], gb_v[:,1])
                        ix2 = torch.min(pb_v[:,2], gb_v[:,2]); iy2 = torch.min(pb_v[:,3], gb_v[:,3])
                        inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
                        pw=(pb_v[:,2]-pb_v[:,0]).clamp(0); ph=(pb_v[:,3]-pb_v[:,1]).clamp(0)
                        gw=(gb_v[:,2]-gb_v[:,0]).clamp(0); gh=(gb_v[:,3]-gb_v[:,1]).clamp(0)
                        iou = inter / (pw*ph + gw*gh - inter + 1e-7)
                        # scatter IoU values onto the heatmap at peak positions
                        peak_inds = ind_v[b][valid_idx]
                        # use hm GT to find which class each peak belongs to
                        cls_at_peak = hm_gt[b, :, peak_inds // W_f, peak_inds % W_f].argmax(0)
                        for i, (pi, cls, q) in enumerate(zip(peak_inds, cls_at_peak, iou)):
                            iou_map[b, cls, pi // W_f, pi % W_f] = q.detach()

                hm_loss += self.crit(output['hm'], hm_gt, iou_map) / opt.num_stacks
            else:
                hm = _sigmoid(output['hm'])
                hm_loss += self.crit(hm, hm_gt) / opt.num_stacks

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

            # ── CIoU loss + Repulsion loss (shared box computation) ───────────
            _use_iou   = 'iou'   in self.log_vars
            _use_repul = 'repul' in self.log_vars
            if (_use_iou or _use_repul) and opt.reg_offset:
                W_feat   = output['hm'].shape[3]
                mask_reg = batch['reg_mask'].to(dev).bool()          # (B, max_obj)
                ind      = batch['ind'].to(dev)                      # (B, max_obj)

                pred_wh  = _tranpose_and_gather_feat(output['wh'],  ind)  # (B, max_obj, 2)
                pred_reg = _tranpose_and_gather_feat(output['reg'], ind)  # (B, max_obj, 2)

                cx_int = (ind % W_feat).float()
                cy_int = (ind // W_feat).float()

                gt_wh  = batch['wh'].to(dev)   # (B, max_obj, 2)
                gt_reg = batch['reg'].to(dev)  # (B, max_obj, 2)

                # Apply log→linear decoding if log_wh head was used
                pred_wh_dec = torch.exp(pred_wh) if self.use_log_wh else pred_wh

                def _to_xyxy(cx, cy, wh):
                    return torch.stack([
                        cx - wh[..., 0] * 0.5, cy - wh[..., 1] * 0.5,
                        cx + wh[..., 0] * 0.5, cy + wh[..., 1] * 0.5,
                    ], dim=-1)

                pred_boxes = _to_xyxy(cx_int + pred_reg[..., 0],
                                      cy_int + pred_reg[..., 1], pred_wh_dec)
                gt_boxes   = _to_xyxy(cx_int + gt_reg[..., 0],
                                      cy_int + gt_reg[..., 1],   gt_wh)

                if _use_iou:
                    valid_pred = pred_boxes[mask_reg]
                    valid_gt   = gt_boxes[mask_reg]
                    if valid_pred.shape[0] > 0:
                        iou_loss += ciou_loss(valid_pred, valid_gt) / opt.num_stacks

                if _use_repul:
                    repul += repulsion_loss(
                        pred_boxes, gt_boxes, mask_reg, self.repul_sigma,
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

        # ── Kendall uncertainty weighting (Kendall et al., CVPR 2018, eq. 9) ──
        # L = ½ · Σᵢ [ exp(−sᵢ) · Lᵢ + sᵢ ]
        # Manual hm/wh/off/iou/repul/id weights are superseded by exp(−sᵢ).
        def _kendall(key: str, val) -> torch.Tensor:
            s = self.log_vars[key]
            v = val if isinstance(val, torch.Tensor) else hm_loss.new_zeros(())
            return torch.exp(-s) * v + s

        components = [('hm', hm_loss), ('wh', wh_loss), ('off', off_loss)]
        if 'iou'   in self.log_vars: components.append(('iou',   iou_loss))
        if 'repul' in self.log_vars: components.append(('repul', repul))
        if 'id'    in self.log_vars: components.append(('id',    reid_loss))

        loss = 0.5 * sum(_kendall(k, v) for k, v in components)

        def _t(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(float(v))

        loss_stats: Dict[str, Any] = {
            'loss':     loss,
            'hm_loss':  _t(hm_loss),
            'wh_loss':  _t(wh_loss),
            'off_loss': _t(off_loss),
        }
        if 'iou'   in self.log_vars: loss_stats['iou_loss']   = _t(iou_loss)
        if 'repul' in self.log_vars: loss_stats['repul_loss'] = _t(repul)
        if 'id'    in self.log_vars: loss_stats['id_loss']    = _t(reid_loss)
        # Kendall effective weights — monitor how σᵢ evolves during training
        for k, s in self.log_vars.items():
            loss_stats[f'w_{k}'] = torch.exp(-s).detach()

        return loss, loss_stats


# ── Trainer ────────────────────────────────────────────────────────────────────

def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1).clip(0, 1)


class DeimMotTrainer(BaseTrainer):
    """Trainer for DEIMMotNet using McMotLoss-style loss."""

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        if getattr(opt, 'iou_weight',   0.0) > 0: loss_states.append('iou_loss')
        if getattr(opt, 'repul_weight', 0.0) > 0: loss_states.append('repul_loss')
        if opt.id_weight > 0:                      loss_states.append('id_loss')
        # Kendall effective weights logged every iteration
        loss_states += ['w_hm', 'w_wh', 'w_off']
        if opt.id_weight > 0:                      loss_states.append('w_id')
        if getattr(opt, 'iou_weight',   0.0) > 0: loss_states.append('w_iou')
        if getattr(opt, 'repul_weight', 0.0) > 0: loss_states.append('w_repul')
        return loss_states, DeimMotLoss(opt)

    def save_result(self, output, batch, results) -> None:
        out = output[0] if isinstance(output, (list, tuple)) else output
        reg = out.get('reg') if self.opt.reg_offset else None
        dets, _, _ = mot_decode(
            heatmap=_sigmoid(out['hm']),
            wh=out['wh'],
            reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh,
            K=self.opt.K,
            log_wh=getattr(self.opt, 'log_wh', False),
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
                dets, _, _ = mot_decode(
                    heatmap=_sigmoid(out['hm']),
                    wh=out['wh'],
                    reg=reg,
                    cat_spec_wh=opt.cat_spec_wh,
                    K=opt.K,
                    log_wh=getattr(opt, 'log_wh', False),
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
