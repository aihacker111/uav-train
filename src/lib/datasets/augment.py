"""
Augmentation ops ported from EdgeCrafter's ecdet.yml pipeline.
All ops work on numpy BGR uint8 images + labels (N,6) [cls,tid,cx,cy,w,h] normalized.
"""

import math
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
    """Clip boxes to image and drop degenerate ones. labels: (N,6) [cls,tid,cx,cy,w,h] norm."""
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
# Photometric distortion  (equiv. torchvision RandomPhotometricDistort)
# ---------------------------------------------------------------------------

def photometric_distort(img, p=0.5):
    """
    Random brightness, contrast, saturation, hue.
    img: numpy BGR uint8 (H, W, 3)  — modified in-place copy
    """
    if random.random() >= p:
        return img

    img = img.astype(np.float32)

    # brightness
    if random.random() < 0.5:
        img += random.uniform(-32, 32)

    contrast_first = random.random() < 0.5
    if contrast_first and random.random() < 0.5:
        img *= random.uniform(0.5, 1.5)

    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if random.random() < 0.5:                             # saturation
        hsv[:, :, 1] *= random.uniform(0.5, 1.5)
    if random.random() < 0.5:                             # hue
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-18, 18)) % 180

    np.clip(hsv[:, :, 0], 0, 179, out=hsv[:, :, 0])
    np.clip(hsv[:, :, 1], 0, 255, out=hsv[:, :, 1])
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if not contrast_first and random.random() < 0.5:      # contrast (after color)
        img *= random.uniform(0.5, 1.5)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Random zoom-out  (equiv. torchvision RandomZoomOut)
# ---------------------------------------------------------------------------

def random_zoom_out(img, labels, fill=114, max_scale=4.0):
    """
    Paste img onto a larger canvas (1× to max_scale×), then return the canvas.
    Labels are adjusted to the canvas coordinate system.
    """
    if random.random() < 0.5:
        return img, labels

    h, w = img.shape[:2]
    scale = random.uniform(1.0, max_scale)
    oh, ow = int(h * scale), int(w * scale)
    top  = random.randint(0, oh - h)
    left = random.randint(0, ow - w)

    canvas = np.full((oh, ow, 3), fill, dtype=img.dtype)
    canvas[top:top + h, left:left + w] = img

    if len(labels) > 0:
        out = labels.copy()
        out[:, 2] = (labels[:, 2] * w + left) / ow
        out[:, 3] = (labels[:, 3] * h + top)  / oh
        out[:, 4] = labels[:, 4] * w / ow
        out[:, 5] = labels[:, 5] * h / oh
        labels = out

    return canvas, labels


# ---------------------------------------------------------------------------
# Random IoU crop  (equiv. torchvision RandomIoUCrop / SSD-style)
# ---------------------------------------------------------------------------

def random_iou_crop(img, labels, p=0.8, min_scale=0.3, trials=40):
    """
    Sample a crop region that has minimum IoU with all ground-truth boxes.
    Returns the cropped image and adjusted labels.
    """
    if random.random() >= p or len(labels) == 0:
        return img, labels

    h, w = img.shape[:2]
    boxes = cxcywh_to_xyxy(labels[:, 2:6], w, h)
    min_iou_opts = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    min_iou = random.choice(min_iou_opts)

    for _ in range(trials):
        scale  = random.uniform(min_scale, 1.0)
        aspect = random.uniform(0.5, 2.0)
        cw = int(w * scale * math.sqrt(aspect))
        ch = int(h * scale / math.sqrt(aspect))
        if cw > w or ch > h or cw < 4 or ch < 4:
            continue
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        x2, y2 = x1 + cw, y1 + ch

        ix1  = np.maximum(boxes[:, 0], x1)
        iy1  = np.maximum(boxes[:, 1], y1)
        ix2  = np.minimum(boxes[:, 2], x2)
        iy2  = np.minimum(boxes[:, 3], y2)
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        area  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-6
        iou   = inter / area

        if iou.min() < min_iou:
            continue

        img = img[y1:y2, x1:x2]
        nh, nw = img.shape[:2]

        shifted = boxes.copy()
        shifted[:, [0, 2]] -= x1
        shifted[:, [1, 3]] -= y1
        out = labels.copy()
        out[:, 2:6] = xyxy_to_cxcywh(shifted, nw, nh)
        return img, sanitize_boxes(out, nw, nh)

    return img, labels


# ---------------------------------------------------------------------------
# Mosaic  (cache-based 4-image mosaic matching EdgeCrafter)
# ---------------------------------------------------------------------------

class MosaicAugmentor:
    """
    Cache-based 4-image mosaic.
    Each tile is letterboxed to `output_size × output_size` and arranged in a 2×2 grid.
    An affine transform (rotation, translate, scale) is applied on the combined canvas.

    img: numpy BGR uint8 (H, W, 3)
    labels: (N, 6) [cls, tid, cx, cy, w, h] normalized
    """

    def __init__(self, output_size=304, max_cached=50, random_pop=True,
                 rotation=10, translation=(0.1, 0.1), scaling=(0.5, 1.5), fill=114):
        self.output_size = output_size
        self.max_cached  = max_cached
        self.random_pop  = random_pop
        self.rotation    = rotation
        self.translation = translation
        self.scaling     = scaling
        self.fill        = fill
        self.cache       = []          # list of (img_s×s, labels_norm)

    # ------------------------------------------------------------------

    def _letterbox_to_tile(self, img, labels):
        """Resize img to output_size×output_size keeping aspect ratio; adjust labels."""
        s = self.output_size
        h, w = img.shape[:2]
        ratio = min(s / h, s / w)
        nw, nh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_left = (s - nw) // 2
        pad_top  = (s - nh) // 2
        tile = np.full((s, s, 3), self.fill, dtype=np.uint8)
        tile[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        if len(labels) > 0:
            out = labels.copy()
            out[:, 2] = (labels[:, 2] * w * ratio + pad_left) / s
            out[:, 3] = (labels[:, 3] * h * ratio + pad_top)  / s
            out[:, 4] = labels[:, 4] * w * ratio / s
            out[:, 5] = labels[:, 5] * h * ratio / s
            labels = out
        return tile, labels

    def _affine(self, img, labels):
        """RandomAffine on the mosaic canvas (matching EdgeCrafter params)."""
        h, w = img.shape[:2]
        angle = random.uniform(-self.rotation, self.rotation)
        scale = random.uniform(*self.scaling)
        tx    = random.uniform(-self.translation[0], self.translation[0]) * w
        ty    = random.uniform(-self.translation[1], self.translation[1]) * h

        R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        R[0, 2] += tx
        R[1, 2] += ty
        img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=(self.fill,) * 3)

        if len(labels) > 0:
            boxes = cxcywh_to_xyxy(labels[:, 2:6], w, h)
            n = len(boxes)
            # transform all 4 corners of each box
            pts = np.ones((n * 4, 3))
            pts[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            M = np.vstack([R, [0, 0, 1]])
            pts = (pts @ M.T)[:, :2].reshape(n, 8)
            xs = pts[:, [0, 2, 4, 6]]
            ys = pts[:, [1, 3, 5, 7]]
            new_boxes = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1)
            out = labels.copy()
            out[:, 2:6] = xyxy_to_cxcywh(new_boxes, w, h)
            labels = sanitize_boxes(out, w, h)

        return img, labels

    # ------------------------------------------------------------------

    def __call__(self, img, labels):
        s = self.output_size
        ncols = labels.shape[1] if len(labels) else 6

        # letterbox current sample → push to cache
        tile, tile_lbl = self._letterbox_to_tile(img, labels)
        self.cache.append((tile.copy(), tile_lbl.copy() if len(tile_lbl) else tile_lbl))
        if len(self.cache) > self.max_cached:
            idx = random.randint(0, len(self.cache) - 2) if self.random_pop else 0
            self.cache.pop(idx)

        # sample 3 others from cache
        idxs  = random.choices(range(len(self.cache)), k=3)
        tiles  = [(tile, tile_lbl)] + [
            (self.cache[i][0].copy(),
             self.cache[i][1].copy() if len(self.cache[i][1]) else self.cache[i][1])
            for i in idxs
        ]

        # build 2×2 canvas
        canvas  = np.full((s * 2, s * 2, 3), self.fill, dtype=np.uint8)
        offsets = [(0, 0), (s, 0), (0, s), (s, s)]   # (x_off, y_off)
        merged  = []

        for (x_off, y_off), (t_img, t_lbl) in zip(offsets, tiles):
            canvas[y_off:y_off + s, x_off:x_off + s] = t_img
            if len(t_lbl) > 0:
                lbl = t_lbl.copy()
                lbl[:, 2] = (lbl[:, 2] * s + x_off) / (2 * s)
                lbl[:, 3] = (lbl[:, 3] * s + y_off) / (2 * s)
                lbl[:, 4] /= 2
                lbl[:, 5] /= 2
                merged.append(lbl)

        all_labels = (np.concatenate(merged) if merged
                      else np.zeros((0, ncols), dtype=np.float32))

        canvas, all_labels = self._affine(canvas, all_labels)
        return canvas, all_labels
