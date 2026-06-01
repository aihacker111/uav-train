"""
Augmentation helpers (AMOT / JDE-style).
All label ops use (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh unless noted.

Spatial augmentation order follows EdgeCrafter ecdet.yml (no mosaic):
  1. random_photometric_distort  (replaces HSV-only)
  2. random_zoom_out             (scale out)
  3. random_iou_crop             (SSD-style crop, p=0.8)
  4. sanitize_boxes              (clip + drop degenerate boxes)
  [letterbox to network size]
  5. horizontal flip
  6. sanitize_boxes              (after letterbox + flip)
"""

import random
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes, width, height):
    """(N,4) cxcywh normalized → (N,4) xyxy pixel"""
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * width
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * height
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * width
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_cxcywh(boxes, width, height):
    """(N,4) xyxy pixel → (N,4) cxcywh normalized"""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    w  = (boxes[:, 2] - boxes[:, 0]) / width
    h  = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([cx, cy, w, h], axis=1)


def sanitize_boxes(labels, width, height, min_size=2):
    """Clip boxes to image and drop degenerate ones.
    labels: (N, 6) [cls, tid, cx, cy, w, h] normalized.
    """
    if len(labels) == 0:
        return labels
    boxes = cxcywh_to_xyxy(labels[:, 2:6], width, height)
    np.clip(boxes[:, [0, 2]], 0, width,  out=boxes[:, [0, 2]])
    np.clip(boxes[:, [1, 3]], 0, height, out=boxes[:, [1, 3]])
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    if not keep.any():
        return np.zeros((0, labels.shape[1]), dtype=labels.dtype)
    out = labels[keep].copy()
    out[:, 2:6] = xyxy_to_cxcywh(boxes[keep], width, height)
    return out


# ---------------------------------------------------------------------------
# 1. Photometric distortion (EdgeCrafter: RandomPhotometricDistort, p=0.5)
# ---------------------------------------------------------------------------

def random_photometric_distort(img,
                                brightness_delta=32,
                                contrast_range=(0.5, 1.5),
                                saturation_range=(0.5, 1.5),
                                hue_delta=18):
    """Random brightness / contrast / saturation / hue on BGR uint8 image.

    Matches torchvision RandomPhotometricDistort: each sub-operation applied
    independently with p=0.5; contrast is randomly applied before or after
    color-space ops.
    """
    img = img.astype(np.float32)

    # Brightness
    if random.random() < 0.5:
        img += random.uniform(-brightness_delta, brightness_delta)

    # Contrast (randomly placed before or after HSV ops)
    apply_contrast_first = random.random() < 0.5
    if apply_contrast_first and random.random() < 0.5:
        img *= random.uniform(*contrast_range)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Saturation
    if random.random() < 0.5:
        img_hsv[:, :, 1] *= random.uniform(*saturation_range)

    # Hue
    if random.random() < 0.5:
        img_hsv[:, :, 0] += random.uniform(-hue_delta, hue_delta)
        img_hsv[:, :, 0] %= 180.0

    np.clip(img_hsv[:, :, 1], 0, 255, out=img_hsv[:, :, 1])
    np.clip(img_hsv[:, :, 2], 0, 255, out=img_hsv[:, :, 2])
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Contrast (after HSV if not applied first)
    if not apply_contrast_first and random.random() < 0.5:
        img *= random.uniform(*contrast_range)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Random zoom out (EdgeCrafter: RandomZoomOut, fill=0)
# ---------------------------------------------------------------------------

def random_zoom_out(img, labels, max_scale=4.0, fill_value=0, p=0.5):
    """Place the image on a larger canvas (zoom out), then adjust labels.

    labels: (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh.
    Returns: (img_canvas, labels_adjusted) — labels still normalized to new canvas.
    """
    if random.random() > p:
        return img, labels

    h, w = img.shape[:2]
    scale  = random.uniform(1.0, max_scale)
    new_h  = int(h * scale)
    new_w  = int(w * scale)

    canvas = np.full((new_h, new_w, 3), fill_value, dtype=img.dtype)
    top  = random.randint(0, new_h - h)
    left = random.randint(0, new_w - w)
    canvas[top:top + h, left:left + w] = img

    if len(labels) > 0:
        out = labels.copy()
        out[:, 2] = (labels[:, 2] * w + left) / new_w   # cx
        out[:, 3] = (labels[:, 3] * h + top)  / new_h   # cy
        out[:, 4] = labels[:, 4] * w / new_w             # bw
        out[:, 5] = labels[:, 5] * h / new_h             # bh
        labels = out

    return canvas, labels


# ---------------------------------------------------------------------------
# 3. Random IoU crop (EdgeCrafter: RandomIoUCrop, p=0.8, min_scale=0.3)
# ---------------------------------------------------------------------------

_IOU_THRESHOLDS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)

def random_iou_crop(img, labels,
                    min_scale=0.3, max_scale=1.0,
                    min_ar=0.5, max_ar=2.0,
                    trials=40, p=0.8):
    """SSD-style random crop with minimum Jaccard overlap constraint.

    labels: (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh.
    A crop is accepted when at least one GT box has IoU >= sampled threshold.
    Boxes whose centre falls inside the crop are kept and re-normalized.
    """
    if random.random() > p or len(labels) == 0:
        return img, labels

    h, w   = img.shape[:2]
    iou_thr = random.choice(_IOU_THRESHOLDS)
    boxes_px = cxcywh_to_xyxy(labels[:, 2:6], w, h)   # (N, 4) pixel xyxy

    for _ in range(trials):
        scale = random.uniform(min_scale, max_scale)
        ar    = random.uniform(min_ar, max_ar)
        cw    = max(1, min(int(w * scale * (ar ** 0.5)), w))
        ch    = max(1, min(int(h * scale / (ar ** 0.5)), h))

        left = random.randint(0, w - cw)
        top  = random.randint(0, h - ch)

        # Jaccard overlap between crop and each GT box
        ix1 = np.maximum(boxes_px[:, 0], left)
        iy1 = np.maximum(boxes_px[:, 1], top)
        ix2 = np.minimum(boxes_px[:, 2], left + cw)
        iy2 = np.minimum(boxes_px[:, 3], top  + ch)
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        box_area = ((boxes_px[:, 2] - boxes_px[:, 0]) *
                    (boxes_px[:, 3] - boxes_px[:, 1])).clip(min=1e-6)
        iou = inter / box_area

        if iou.max() < iou_thr:
            continue

        # Keep boxes whose centre is inside the crop
        cx_px = (boxes_px[:, 0] + boxes_px[:, 2]) / 2
        cy_px = (boxes_px[:, 1] + boxes_px[:, 3]) / 2
        keep  = ((cx_px >= left) & (cx_px < left + cw) &
                 (cy_px >= top)  & (cy_px < top  + ch))
        if not keep.any():
            continue

        img_crop = img[top:top + ch, left:left + cw]

        new_px = boxes_px[keep].copy()
        new_px[:, [0, 2]] = np.clip(new_px[:, [0, 2]] - left, 0, cw)
        new_px[:, [1, 3]] = np.clip(new_px[:, [1, 3]] - top,  0, ch)

        new_labels = labels[keep].copy()
        new_labels[:, 2:6] = xyxy_to_cxcywh(new_px, cw, ch)

        return img_crop, new_labels

    return img, labels


# ---------------------------------------------------------------------------
# Legacy: HSV-only augmentation (kept for reference, replaced by
# random_photometric_distort in the main pipeline)
# ---------------------------------------------------------------------------

def augment_hsv(img, fraction=0.5):
    """Random S/V scaling on BGR uint8 image (in-place).
    Kept for backward compatibility — prefer random_photometric_distort.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    a = (random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    a = (random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img
