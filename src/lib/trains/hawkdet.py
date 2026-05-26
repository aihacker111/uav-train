"""HawkDetTrainer — trainer for HawkDet (ECViT + HybridEncoder + 4-scale TOOD T-head DFL)."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch

from lib.models.losses.hawkdet_loss import HawkDetLoss
from lib.utils.det_eval import COCOEvaluator, VISDRONE_CLASSES
from .base_trainer import BaseTrainer


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) cxcywh [0,1] → xyxy [0,1]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1).clip(0, 1)


class HawkDetTrainer(BaseTrainer):
    """Trainer for HawkDet.

    Training model output : {'cls': ..., 'reg': ...}
    Inference model output: {'pred_boxes': (B,N,4), 'pred_scores': (B,N,C)}
    """

    def _get_losses(self, opt):
        loss_stats = ['loss', 'loss_cls', 'loss_dfl', 'loss_giou', 'loss_reid', 'num_pos']

        loss = HawkDetLoss(
            num_classes = opt.num_classes,
            reg_max     = getattr(opt, 'reg_max',     16),
            lambda_cls  = getattr(opt, 'cls_weight',  1.0),
            lambda_dfl  = getattr(opt, 'dfl_weight',  1.5),
            lambda_giou = getattr(opt, 'giou_weight', 2.5),
            tal_topk    = getattr(opt, 'tal_topk',    13),
            tal_alpha   = getattr(opt, 'tal_alpha',   1.0),
            tal_beta    = getattr(opt, 'tal_beta',    6.0),
            qfl_beta    = getattr(opt, 'qfl_beta',    2.0),
            reid_dim    = getattr(opt, 'reid_dim',    0),
            num_ids     = getattr(opt, 'nID',         0),
            lambda_reid = getattr(opt, 'id_weight',   1.0),
        )
        return loss_stats, loss

    def save_result(self, output: Dict[str, Any], batch: dict, results: dict) -> None:
        img_id = batch['meta']['img_id'].cpu().numpy()[0]

        pred_boxes  = output['pred_boxes'].detach().cpu()    # (B, N, 4) cxcywh
        pred_scores = output['pred_scores'].detach().cpu()   # (B, N, C)

        scores, classes = pred_scores[0].max(dim=-1)
        results[img_id] = {
            'boxes':   pred_boxes[0],
            'scores':  scores,
            'classes': classes,
        }

    def debug(self, batch, output, iter_id) -> None:
        pass

    def evaluate(
        self,
        epoch:     int,
        val_loader,
        logger=None,
        score_thr: float | None = None,
    ) -> Dict[str, float]:
        opt = self.opt
        if score_thr is None:
            score_thr = getattr(opt, 'score_thr', 0.25)

        mwl   = self.model_with_loss
        model = mwl.module.model if hasattr(mwl, 'module') else mwl.model
        model.eval()

        ev = COCOEvaluator(
            num_classes  = opt.num_classes,
            class_names  = VISDRONE_CLASSES[:opt.num_classes],
        )

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

                output = model(batch['input'])

                pred_boxes  = output['pred_boxes'].cpu()    # (B, N, 4)
                pred_scores = output['pred_scores'].cpu()   # (B, N, C)
                cls_scores, cls_labels = pred_scores.max(dim=-1)

                B = batch['input'].shape[0]
                for b in range(B):
                    scores_np = cls_scores[b].numpy()
                    labels_np = cls_labels[b].numpy()

                    keep           = scores_np >= score_thr
                    pred_boxes_np  = _cxcywh_to_xyxy(pred_boxes[b].numpy()[keep])
                    pred_scores_np = scores_np[keep].astype(np.float32)
                    pred_labels_np = labels_np[keep].astype(np.int64)

                    gt           = batch['targets'][b]
                    gt_boxes_raw = gt['boxes'].cpu().numpy()
                    gt_labels_np = gt['labels'].cpu().numpy()
                    if gt_labels_np.ndim == 2:
                        gt_labels_np = gt_labels_np[:, 0]
                    gt_labels_np = gt_labels_np.astype(np.int64)

                    if len(gt_boxes_raw) > 0:
                        gt_boxes_np = _cxcywh_to_xyxy(gt_boxes_raw)
                    else:
                        gt_boxes_np  = np.zeros((0, 4), dtype=np.float32)
                        gt_labels_np = np.zeros((0,),   dtype=np.int64)

                    ev.update(
                        pred_boxes_np.astype(np.float32), pred_scores_np,
                        pred_labels_np,
                        gt_boxes_np.astype(np.float32),   gt_labels_np,
                    )

        stats = ev.summarize()
        print(f'\n[HawkDet eval] epoch {epoch:03d}')
        ev.print_summary(stats)

        if logger is not None:
            for k, v in stats.items():
                logger.scalar_summary(f'val_{k}', v, epoch)

        model.train()
        return stats
