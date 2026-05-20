from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from lib.models.losses.hybrid_loss import HybridLoss
from lib.models.networks.hybrid.heads import DETROutput
from lib.utils.det_eval import COCOEvaluator, VISDRONE_CLASSES
from .base_trainer import BaseTrainer


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) cxcywh [0,1] → xyxy [0,1]"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1).clip(0, 1)


class HybridTrainer(BaseTrainer):
    """
    Trainer for HybridDETR.

    Expects model outputs as:
      {'score_map': Tensor, 'stage2': DETROutput,
       'dn_out': DETROutput|None, 'dn_meta': DNMeta|None,
       'query_scores': Tensor, 'query_classes': Tensor}

    Batch must contain DETR-format 'targets' list and CenterNet-format 'hm'
    (used by the scorer loss).  ReID is enabled when opt.id_weight > 0.
    """

    def _get_losses(self, opt):
        loss_stats = [
            'loss',
            'loss_score',
            'loss_s2', 'loss_cls', 'loss_bbox', 'loss_ciou',
            'loss_dn', 'loss_dn_l1', 'loss_dn_cls',
            'tau_query',
        ]

        reid_classifier = None
        if getattr(opt, 'id_weight', 0) > 0 and hasattr(opt, 'nID_dict') and opt.nID_dict:
            total_ids       = sum(opt.nID_dict.values())
            reid_dim        = getattr(opt, 'reid_dim', 256)
            reid_classifier = nn.Linear(reid_dim, total_ids)
            loss_stats.append('loss_reid')

        loss = HybridLoss(
            num_classes    = opt.num_classes,
            lambda_bbox    = getattr(opt, 'bbox_weight',    2.0),
            lambda_ciou    = getattr(opt, 'giou_weight',    2.0),
            lambda_reid    = getattr(opt, 'id_weight',      1.0),
            lambda_dn_l1   = getattr(opt, 'dn_l1_weight',  1.0),
            lambda_dn_cls  = getattr(opt, 'dn_cls_weight',  1.0),
            aux_loss       = True,
            reid_classifier = reid_classifier,
        )
        return loss_stats, loss

    def save_result(self, output: Dict[str, Any], batch: dict, results: dict) -> None:
        stage2: DETROutput = output['stage2']
        img_id = batch['meta']['img_id'].cpu().numpy()[0]

        boxes  = stage2.boxes.detach().cpu()
        logits = stage2.logits.detach().cpu()
        scores = logits.sigmoid().max(dim=-1)

        results[img_id] = {
            'boxes':   boxes[0],
            'scores':  scores.values[0],
            'classes': scores.indices[0],
        }

    def debug(self, batch, output, iter_id) -> None:
        pass

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        epoch: int,
        val_loader,
        logger=None,
        score_thr: float = 0.3,
    ) -> Dict[str, float]:
        opt = self.opt
        mwl = self.model_with_loss
        model = mwl.module.model if hasattr(mwl, 'module') else mwl.model
        model.eval()

        ev = COCOEvaluator(num_classes=opt.num_classes,
                           class_names=VISDRONE_CLASSES[:opt.num_classes])

        with torch.no_grad():
            for batch in val_loader:
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

                prob_t                   = stage2.logits.sigmoid()
                cls_scores_t, cls_labels_t = prob_t.max(dim=-1)

                boxes_cpu  = stage2.boxes.cpu()
                scores_cpu = cls_scores_t.cpu()
                labels_cpu = cls_labels_t.cpu()

                B = batch['input'].shape[0]
                for b in range(B):
                    cls_scores = scores_cpu[b].numpy()
                    cls_labels = labels_cpu[b].numpy()

                    keep = cls_scores >= score_thr
                    pred_boxes  = _cxcywh_to_xyxy(boxes_cpu[b].numpy()[keep])
                    pred_scores = cls_scores[keep].astype(np.float32)
                    pred_labels = cls_labels[keep].astype(np.int64)

                    gt = batch['targets'][b]
                    gt_boxes_raw = gt['boxes'].cpu().numpy()
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
