"""
COCO-style detection evaluator — no pycocotools dependency.

Follows the exact COCO 2017 evaluation protocol:
  • IoU thresholds  : 0.50 : 0.05 : 0.95  (10 values)
  • Area ranges     : all / small / medium / large
  • Max detections  : 1 / 10 / 100
  • Interpolation   : 101-point recall axis
  • Metrics         : AP, AP50, AP75, AP_S, AP_M, AP_L,
                      AR_1, AR_10, AR_100, AR_S, AR_M, AR_L
  • Per-class AP@50

VisDrone-specific:
  • Annotations with score=0 or category=0/11 are treated as "ignore".
    Predictions that overlap an ignored region (IoU ≥ 0.5) are neither
    TP nor FP — they are simply removed from the PR curve.

Usage::
    ev = COCOEvaluator(num_classes=10)
    for each image:
        ev.update(pred_boxes, pred_scores, pred_labels,
                  gt_boxes,   gt_labels,
                  ignore_boxes=ignore_boxes)      # optional ignore mask
    ev.print_summary()                            # COCO-style table
    stats = ev.get_stats()                        # dict
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ── VisDrone class names (0-indexed) ─────────────────────────────────────────
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor',
]

# COCO area ranges (absolute pixel² — same as pycocotools)
_AREA_RANGES: Dict[str, Tuple[float, float]] = {
    'all':    (0.0,     1e10),
    'small':  (0.0,     32 ** 2),
    'medium': (32 ** 2, 96 ** 2),
    'large':  (96 ** 2, 1e10),
}

_IOU_THRS  = np.linspace(0.50, 0.95, 10, endpoint=True)   # [.50, .55, …, .95]
_MAX_DETS  = [1, 10, 100]
_RECALL_TH = np.linspace(0.0, 1.0, 101, endpoint=True)    # 101-point


# ── Helpers ────────────────────────────────────────────────────────────────────

def _box_iou(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """(M, 4) xyxy × (N, 4) xyxy → (M, N) IoU matrix."""
    a1 = (b1[:, 2] - b1[:, 0]).clip(0) * (b1[:, 3] - b1[:, 1]).clip(0)
    a2 = (b2[:, 2] - b2[:, 0]).clip(0) * (b2[:, 3] - b2[:, 1]).clip(0)
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-7)


def _box_area(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) xyxy → (N,) area."""
    return ((boxes[:, 2] - boxes[:, 0]).clip(0) *
            (boxes[:, 3] - boxes[:, 1]).clip(0))


def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated AP (COCO protocol)."""
    ap = 0.0
    for thr in _RECALL_TH:
        mask = recalls >= thr
        ap += (precisions[mask].max() if mask.any() else 0.0)
    return float(ap / len(_RECALL_TH))


# ── Per-image record ───────────────────────────────────────────────────────────

class _ImageRecord:
    __slots__ = ('pb', 'ps', 'pl', 'pa', 'gb', 'gl', 'ga', 'ib')

    def __init__(
        self,
        pred_boxes:   np.ndarray,   # (M, 4) xyxy
        pred_scores:  np.ndarray,   # (M,)
        pred_labels:  np.ndarray,   # (M,) int
        gt_boxes:     np.ndarray,   # (N, 4) xyxy
        gt_labels:    np.ndarray,   # (N,)   int
        ignore_boxes: Optional[np.ndarray],  # (K, 4) xyxy  — don't-care regions
    ):
        self.pb = pred_boxes.astype(np.float32)
        self.ps = pred_scores.astype(np.float32)
        self.pl = pred_labels.astype(np.int32)
        self.pa = _box_area(self.pb) if len(self.pb) else np.zeros(0, np.float32)

        self.gb = gt_boxes.astype(np.float32)
        self.gl = gt_labels.astype(np.int32)
        self.ga = _box_area(self.gb) if len(self.gb) else np.zeros(0, np.float32)

        self.ib = ignore_boxes.astype(np.float32) if ignore_boxes is not None and len(ignore_boxes) else None


# ── Main Evaluator ─────────────────────────────────────────────────────────────

class COCOEvaluator:
    """
    COCO 2017 detection evaluation protocol.

    Drop-in upgrade from the old DetectionEvaluator:
      - same .update() signature (ignore_boxes is optional)
      - .summarize() returns the same dict keys (mAP50, mAP50:95, AP50_clsN)
        plus additional COCO metrics
      - .print_summary() prints the canonical COCO 12-line table
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes  = num_classes
        self.class_names  = class_names or (
            VISDRONE_CLASSES[:num_classes]
            if num_classes <= len(VISDRONE_CLASSES)
            else [f'cls{i}' for i in range(num_classes)]
        )
        self._records: List[_ImageRecord] = []

    def reset(self) -> None:
        self._records.clear()

    def update(
        self,
        pred_boxes:   np.ndarray,
        pred_scores:  np.ndarray,
        pred_labels:  np.ndarray,
        gt_boxes:     np.ndarray,
        gt_labels:    np.ndarray,
        ignore_boxes: Optional[np.ndarray] = None,
    ) -> None:
        self._records.append(_ImageRecord(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, ignore_boxes,
        ))

    # ── Core: evaluate one class at one IoU thr / area range / max_dets ───────

    def _eval_single(
        self,
        cls:        int,
        iou_thr:    float,
        area_range: Tuple[float, float],
        max_dets:   int,
    ) -> Tuple[float, float]:
        """Returns (AP, AR) for one (class, IoU, area, max_dets) combination."""
        a_min, a_max = area_range
        scores_list: List[float] = []
        tp_list:     List[int]   = []
        n_gt = 0

        for rec in self._records:
            # ── Ground truth for this class + area ──────────────────────────
            gt_m = (rec.gl == cls) & (rec.ga >= a_min) & (rec.ga < a_max)
            gb_c = rec.gb[gt_m]
            n_gt += int(gt_m.sum())

            # ── Predictions for this class, score-sorted, capped at max_dets ─
            pr_m = rec.pl == cls
            pb_c = rec.pb[pr_m]
            ps_c = rec.ps[pr_m]
            pa_c = rec.pa[pr_m]

            if len(pb_c) == 0:
                continue

            order   = np.argsort(-ps_c)
            pb_c    = pb_c[order][:max_dets]
            ps_c    = ps_c[order][:max_dets]
            pa_c    = pa_c[order][:max_dets]

            # ── Remove predictions that fall in ignore regions ───────────────
            # A prediction is "ignored" if it overlaps an ignore box at IoU ≥ 0.5
            # → it is removed entirely from the PR curve (not TP, not FP).
            keep = np.ones(len(pb_c), dtype=bool)
            if rec.ib is not None and len(rec.ib):
                iou_ign = _box_iou(pb_c, rec.ib)   # (M, K_ign)
                keep &= iou_ign.max(axis=1) < 0.5

            pb_c = pb_c[keep]
            ps_c = ps_c[keep]
            pa_c = pa_c[keep]

            if len(pb_c) == 0:
                continue

            # ── Apply area filter to predictions ────────────────────────────
            area_keep = (pa_c >= a_min) & (pa_c < a_max)

            if len(gb_c) == 0:
                # All valid preds are FP
                for i, ak in enumerate(area_keep):
                    if ak:
                        scores_list.append(float(ps_c[i]))
                        tp_list.append(0)
                continue

            # ── Greedy TP/FP assignment ──────────────────────────────────────
            iou_mat = _box_iou(pb_c, gb_c)    # (M, N_gt)
            matched = np.zeros(len(gb_c), dtype=bool)

            for i in range(len(pb_c)):
                if not area_keep[i]:
                    continue
                j   = int(iou_mat[i].argmax())
                hit = (float(iou_mat[i, j]) >= iou_thr) and not matched[j]
                scores_list.append(float(ps_c[i]))
                tp_list.append(int(hit))
                if hit:
                    matched[j] = True

        # ── PR curve ────────────────────────────────────────────────────────
        if n_gt == 0 or len(scores_list) == 0:
            return 0.0, 0.0

        s       = np.array(scores_list)
        tp_arr  = np.array(tp_list)
        order   = np.argsort(-s)
        tp_cum  = tp_arr[order].cumsum()
        fp_cum  = (1 - tp_arr[order]).cumsum()
        rec     = tp_cum / (n_gt + 1e-7)
        prec    = tp_cum / (tp_cum + fp_cum + 1e-7)

        ap = _interpolated_ap(rec, prec)
        ar = float(tp_cum[-1]) / (n_gt + 1e-7) if len(tp_cum) else 0.0
        return ap, ar

    # ── Aggregate metrics ──────────────────────────────────────────────────────

    def _mean_over_classes(self, values: Dict[int, float]) -> float:
        return float(np.mean(list(values.values()))) if values else 0.0

    def summarize(self) -> Dict[str, float]:
        """
        Compute all COCO metrics.

        Returns a flat dict with keys:
          mAP, mAP50, mAP75, mAP_S, mAP_M, mAP_L,
          mAR_1, mAR_10, mAR_100, mAR_S, mAR_M, mAR_L,
          AP50_<class_name> (per-class at IoU=0.50)
        """
        result: Dict[str, float] = {}

        # ── mAP @ [.5:.95] (primary metric) ──────────────────────────────────
        ap_all_thr: Dict[int, List[float]] = {c: [] for c in range(self.num_classes)}
        for thr in _IOU_THRS:
            for cls in range(self.num_classes):
                ap, _ = self._eval_single(cls, thr, _AREA_RANGES['all'], 100)
                ap_all_thr[cls].append(ap)
        map_per_cls = {cls: float(np.mean(v)) for cls, v in ap_all_thr.items() if v}
        result['mAP'] = self._mean_over_classes(map_per_cls)

        # ── AP50, AP75 ────────────────────────────────────────────────────────
        ap50 = {cls: self._eval_single(cls, 0.50, _AREA_RANGES['all'], 100)[0]
                for cls in range(self.num_classes)}
        ap75 = {cls: self._eval_single(cls, 0.75, _AREA_RANGES['all'], 100)[0]
                for cls in range(self.num_classes)}
        result['mAP50'] = self._mean_over_classes(ap50)
        result['mAP75'] = self._mean_over_classes(ap75)
        # legacy key used by hybrid.py logger
        result['mAP50:95'] = result['mAP']

        # ── AP by area ────────────────────────────────────────────────────────
        for area_name in ('small', 'medium', 'large'):
            ap_area = {}
            for cls in range(self.num_classes):
                ap_thr = [
                    self._eval_single(cls, thr, _AREA_RANGES[area_name], 100)[0]
                    for thr in _IOU_THRS
                ]
                ap_area[cls] = float(np.mean(ap_thr))
            result[f'mAP_{area_name[0].upper()}'] = self._mean_over_classes(ap_area)

        # ── AR @ max_dets = 1, 10, 100 (all area) ────────────────────────────
        for md in _MAX_DETS:
            ar_per_cls: Dict[int, List[float]] = {c: [] for c in range(self.num_classes)}
            for thr in _IOU_THRS:
                for cls in range(self.num_classes):
                    _, ar = self._eval_single(cls, thr, _AREA_RANGES['all'], md)
                    ar_per_cls[cls].append(ar)
            ar_mean = {cls: float(np.mean(v)) for cls, v in ar_per_cls.items()}
            result[f'mAR_{md}'] = self._mean_over_classes(ar_mean)

        # ── AR by area @ max_dets=100 ─────────────────────────────────────────
        for area_name in ('small', 'medium', 'large'):
            ar_area: Dict[int, List[float]] = {c: [] for c in range(self.num_classes)}
            for thr in _IOU_THRS:
                for cls in range(self.num_classes):
                    _, ar = self._eval_single(cls, thr, _AREA_RANGES[area_name], 100)
                    ar_area[cls].append(ar)
            ar_mean = {cls: float(np.mean(v)) for cls, v in ar_area.items()}
            result[f'mAR_{area_name[0].upper()}'] = self._mean_over_classes(ar_mean)

        # ── Per-class AP@50 ───────────────────────────────────────────────────
        for cls, v in sorted(ap50.items()):
            name = self.class_names[cls] if cls < len(self.class_names) else f'cls{cls}'
            result[f'AP50_{name}'] = v
            result[f'AP50_cls{cls}'] = v   # legacy key

        return result

    # ── COCO-style output table ────────────────────────────────────────────────

    def print_summary(self, stats: Optional[Dict[str, float]] = None) -> None:
        """Print the canonical 12-line COCO summary table."""
        if stats is None:
            stats = self.summarize()

        lines = [
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 'mAP'),
            ('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', 'mAP50'),
            ('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', 'mAP75'),
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 'mAP_S'),
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 'mAP_M'),
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', 'mAP_L'),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]', 'mAR_1'),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]', 'mAR_10'),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 'mAR_100'),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 'mAR_S'),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 'mAR_M'),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', 'mAR_L'),
        ]
        print()
        for label, key in lines:
            v = stats.get(key, -1.0)
            print(f' {label} = {v:0.3f}')

        # Per-class breakdown
        print()
        print(' Per-class AP @ IoU=0.50:')
        for cls in range(self.num_classes):
            name = self.class_names[cls] if cls < len(self.class_names) else f'cls{cls}'
            v = stats.get(f'AP50_{name}', 0.0)
            print(f'   {name:<20s} {v:0.3f}')
        print()


# ── Backward-compatible alias ──────────────────────────────────────────────────

class DetectionEvaluator(COCOEvaluator):
    """Alias so existing imports keep working without any change."""
    pass
