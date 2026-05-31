"""
Augmentation helpers (AMOT / JDE-style).
All label ops use (N, 6) [cls, tid, cx, cy, w, h] normalized cxcywh unless noted.
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
    """Clip boxes to image and drop degenerate ones."""
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
# HSV augmentation (AMOT LoadImagesAndLabels)
# ---------------------------------------------------------------------------

def augment_hsv(img, fraction=0.5):
    """
    Random saturation / value scaling on a BGR uint8 image (in-place).
    Matches AMOT: fraction=0.5 → scale factor in [0.5, 1.5] per channel.
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
