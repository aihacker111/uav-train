from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from lib.models.losses.hybrid_loss import HybridLoss
from lib.models.networks.deim_uav.heads import CenterNetOutput, DETROutput
from lib.utils.det_eval import COCOEvaluator, VISDRONE_CLASSES
from .base_trainer import BaseTrainer


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) cxcywh [0,1] → xyxy [0,1]"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1).clip(0, 1)


class HybridTrainer(BaseTrainer):
    """
    Trainer for HybridCenterNetDETR.

    Expects model outputs as:
      {'stage1': CenterNetOutput, 'stage2': DETROutput, 'query_scores': ..., 'query_classes': ...}

    Batch must contain CenterNet-format stage-1 targets and DETR-format 'targets' list.
    ReID is enabled when opt.id_weight > 0 and opt.nID_dict is available.
    """

    def _get_losses(self, opt):
        loss_stats = [
            'loss',
            'loss_s1', 'loss_hm', 'loss_wh', 'loss_reg',
            'loss_s2', 'loss_cls', 'loss_bbox', 'loss_ciou', 'loss_consist',
            'w_s1', 'w_s2',   # effective stage weights (for monitoring curriculum)
        ]

        reid_classifier = None
        if getattr(opt, 'id_weight', 0) > 0 and hasattr(opt, 'nID_dict') and opt.nID_dict:
            total_ids = sum(opt.nID_dict.values())
            reid_dim  = getattr(opt, 'reid_dim', 256)
            reid_classifier = nn.Linear(reid_dim, total_ids)
            loss_stats.append('loss_reid')

        loss = HybridLoss(
            num_classes           = opt.num_classes,
            lambda_wh             = getattr(opt, 'wh_weight',            0.1),
            lambda_reg            = getattr(opt, 'off_weight',            1.0),
            lambda_bbox           = getattr(opt, 'bbox_weight',           2.0),
            lambda_ciou           = getattr(opt, 'giou_weight',           2.0),
            lambda_reid           = getattr(opt, 'id_weight',             1.0),
            lambda_stage1         = getattr(opt, 'stage1_weight',         2.0),
            lambda_stage2         = getattr(opt, 'stage2_weight',         1.0),
            lambda_consist        = getattr(opt, 'consist_weight',        0.02),
            consist_warmup_epochs = getattr(opt, 'consist_warmup_epochs', 5),
            aux_loss              = True,
            reid_classifier       = reid_classifier,
            total_epochs          = getattr(opt, 'num_epochs',            0),
        )
        return loss_stats, loss

    def save_result(self, output: Dict[str, Any], batch: dict, results: dict) -> None:
        stage2: DETROutput = output['stage2']
        img_id = batch['meta']['img_id'].cpu().numpy()[0]

        boxes  = stage2.boxes.detach().cpu()    # (B, K, 4)  cxcywh in [0,1]
        logits = stage2.logits.detach().cpu()   # (B, K, C)
        scores = logits.sigmoid().max(dim=-1)   # (values, indices)

        results[img_id] = {
            'boxes':   boxes[0],
            'scores':  scores.values[0],
            'classes': scores.indices[0],
        }

    def debug(self, batch, output, iter_id) -> None:
        pass

    # ── Evaluation ────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        epoch: int,
        val_loader,
        logger=None,
        score_thr: float = 0.3,
    ) -> Dict[str, float]:
        """
        Run model over val_loader and return per-class AP@50 + mAP50.

        Printed example::
            [eval] epoch 10 | mAP50 0.412 | AP_cls0 0.550 | AP_cls1 0.302 | ...

        Args:
            epoch     : current epoch number (for TensorBoard logging)
            val_loader: DataLoader built with hybrid_collate_fn
            logger    : Logger instance (optional, for TensorBoard)
            score_thr : discard predictions with max-class score below this

        Returns:
            dict of metric names → float values
        """
        opt = self.opt
        mwl = self.model_with_loss
        model = mwl.module.model if hasattr(mwl, 'module') else mwl.model
        model.eval()

        ev = COCOEvaluator(num_classes=opt.num_classes,
                           class_names=VISDRONE_CLASSES[:opt.num_classes])

        with torch.no_grad():
            for batch in val_loader:
                # Move to device (same logic as base_trainer)
                for k in batch:
                    if k == 'meta':
                        pass
                    elif k == 'targets':
                        batch[k] = [
                            {kk: vv.to(opt.device, non_blocking=True)
                             for kk, vv in t.items()}
                            for t in batch[k]
                        ]
                    else:
                        batch[k] = batch[k].to(opt.device, non_blocking=True)

                output   = model(batch['input'])
                stage2: DETROutput = output['stage2']

                # Batch GPU ops before CPU transfer: sigmoid + argmax/max in fp32
                prob_t            = stage2.logits.sigmoid()           # (B, K, C) GPU
                cls_scores_t, cls_labels_t = prob_t.max(dim=-1)      # (B, K) each

                boxes_cpu  = stage2.boxes.cpu()       # (B, K, 4)
                scores_cpu = cls_scores_t.cpu()       # (B, K)
                labels_cpu = cls_labels_t.cpu()       # (B, K)

                B = batch['input'].shape[0]
                for b in range(B):
                    cls_scores = scores_cpu[b].numpy()   # (K,)
                    cls_labels = labels_cpu[b].numpy()   # (K,)

                    keep = cls_scores >= score_thr
                    pred_boxes  = _cxcywh_to_xyxy(boxes_cpu[b].numpy()[keep])
                    pred_scores = cls_scores[keep].astype(np.float32)
                    pred_labels = cls_labels[keep].astype(np.int64)

                    gt = batch['targets'][b]
                    gt_boxes_raw = gt['boxes'].cpu().numpy()    # (N, 4) cxcywh
                    gt_labels    = gt['labels'].cpu().numpy().astype(np.int64)

                    if len(gt_boxes_raw) > 0:
                        gt_boxes = _cxcywh_to_xyxy(gt_boxes_raw)
                    else:
                        gt_boxes  = np.zeros((0, 4), dtype=np.float32)
                        gt_labels = np.zeros((0,),   dtype=np.int64)

                    ev.update(pred_boxes.astype(np.float32), pred_scores, pred_labels,
                              gt_boxes.astype(np.float32),  gt_labels)

        stats = ev.summarize()
        print(f'\n[eval] epoch {epoch:03d}')
        ev.print_summary(stats)

        if logger is not None:
            for k, v in stats.items():
                logger.scalar_summary(f'val_{k}', v, epoch)

        model.train()
        return stats
