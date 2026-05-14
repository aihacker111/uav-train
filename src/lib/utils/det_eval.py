"""
Per-class AP evaluator supporting mAP50 and mAP50:95.

Boxes must be supplied in xyxy absolute-pixel (or normalized) format — the
same coordinate space for predictions and ground truth.

Usage::
    ev = DetectionEvaluator(num_classes=10)
    for frame_id in sorted(pred_dict):
        ev.update(pred_boxes, pred_scores, pred_labels,
                  gt_boxes,   gt_labels)
    stats = ev.summarize()
    # {'mAP50': 0.412, 'mAP50:95': 0.251, 'AP50_cls0': 0.55, ...}
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ── Primitives ────────────────────────────────────────────────────────────────

def _box_iou(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """(M, 4) xyxy × (N, 4) xyxy → (M, N) IoU."""
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-7)


def _coco_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """101-point interpolated AP (COCO-style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = rec >= t
        ap  += (prec[mask].max() if mask.any() else 0.0) / 101
    return float(ap)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class DetectionEvaluator:
    """
    Accumulates per-image prediction / GT pairs and computes:
      - mAP50    : AP at IoU = 0.50
      - mAP50:95 : AP averaged over IoU in [0.50 : 0.05 : 0.95]  (COCO primary)
      - AP50_clsN : per-class AP at IoU = 0.50
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._gts:   List[Tuple[np.ndarray, np.ndarray]]             = []

    def reset(self) -> None:
        self._preds.clear()
        self._gts.clear()

    def update(
        self,
        pred_boxes:  np.ndarray,   # (M, 4) xyxy
        pred_scores: np.ndarray,   # (M,)
        pred_labels: np.ndarray,   # (M,) int
        gt_boxes:    np.ndarray,   # (N, 4) xyxy
        gt_labels:   np.ndarray,   # (N,)   int
    ) -> None:
        self._preds.append((
            pred_boxes.astype(np.float32),
            pred_scores.astype(np.float32),
            pred_labels.astype(np.int64),
        ))
        self._gts.append((
            gt_boxes.astype(np.float32),
            gt_labels.astype(np.int64),
        ))

    # ── Core computation ──────────────────────────────────────────────────────

    def _ap_at_iou(self, iou_thr: float) -> Dict[int, float]:
        """Return per-class AP at a single IoU threshold."""
        aps: Dict[int, float] = {}
        for cls in range(self.num_classes):
            scores_all: List[float] = []
            tp_all:     List[int]   = []
            n_gt = 0

            for (pb, ps, pl), (gb, gl) in zip(self._preds, self._gts):
                gt_m = gl == cls
                gb_c = gb[gt_m]
                n_gt += int(gt_m.sum())

                pr_m = pl == cls
                pb_c = pb[pr_m]
                ps_c = ps[pr_m]

                if len(pb_c) == 0:
                    continue
                if len(gb_c) == 0:
                    scores_all.extend(ps_c.tolist())
                    tp_all.extend([0] * len(pb_c))
                    continue

                iou     = _box_iou(pb_c, gb_c)          # (M, N)
                matched = np.zeros(len(gb_c), dtype=bool)

                for i in np.argsort(-ps_c):
                    j   = int(iou[i].argmax())
                    hit = float(iou[i, j]) >= iou_thr and not matched[j]
                    scores_all.append(float(ps_c[i]))
                    tp_all.append(int(hit))
                    if hit:
                        matched[j] = True

            if n_gt == 0:
                continue

            s      = np.array(scores_all)
            tp_arr = np.array(tp_all)
            if len(s) == 0:
                aps[cls] = 0.0
                continue

            order = np.argsort(-s)
            tp    = tp_arr[order].cumsum()
            fp    = (1 - tp_arr[order]).cumsum()
            rec   = tp / (n_gt + 1e-7)
            prec  = tp / (tp + fp + 1e-7)
            aps[cls] = _coco_ap(rec, prec)

        return aps

    def summarize(self) -> Dict[str, float]:
        """Compute mAP50 and mAP50:95 over all accumulated images."""
        # mAP @ 0.50
        aps50 = self._ap_at_iou(0.50)
        mAP50 = float(np.mean(list(aps50.values()))) if aps50 else 0.0

        # mAP @ 0.50:0.05:0.95
        iou_thrs = np.linspace(0.50, 0.95, 10)
        all_aps  = [self._ap_at_iou(t) for t in iou_thrs]
        all_cls  = set(k for a in all_aps for k in a)
        mAP5095_cls = {
            cls: float(np.mean([a.get(cls, 0.0) for a in all_aps]))
            for cls in all_cls
        }
        mAP5095 = float(np.mean(list(mAP5095_cls.values()))) if mAP5095_cls else 0.0

        result: Dict[str, float] = {
            'mAP50':    mAP50,
            'mAP50:95': mAP5095,
        }
        for cls, v in sorted(aps50.items()):
            result[f'AP50_cls{cls}'] = v
        return result
