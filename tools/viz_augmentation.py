"""
Visualize the PIL augmentation pipeline on VisDrone samples.

Shows side-by-side: BEFORE (letterbox only) vs AFTER (full PIL pipeline + letterbox).
Box stats (S=small <32px, M=medium 32-96px, L=large >96px) are shown per image.

Usage:
    cd /Users/tinvo0908/Desktop/uav-train
    python tools/viz_augmentation.py
"""
import argparse
import os
import random
import sys
import warnings

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import PIL.Image
from lib.datasets.transforms import build_aerial_mot_transforms

CLS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle"]
CLS_COLORS = [
    (255, 80, 80),
    (255, 165, 0),
    (80, 200, 80),
    (80, 80, 255),
    (200, 80, 200),
    (80, 200, 200),
    (255, 220, 50),
]


def letterbox(img, height=512, width=892, color=(127.5, 127.5, 127.5)):
    shape = img.shape[:2]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) * 0.5
    dh = (height - new_shape[1]) * 0.5
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    h_out, w_out = img.shape[:2]
    h64 = ((h_out + 63) // 64) * 64
    w64 = ((w_out + 63) // 64) * 64
    if h64 != h_out or w64 != w_out:
        img = cv2.copyMakeBorder(img, 0, h64 - h_out, 0, w64 - w_out, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


def draw_boxes(img, labels_xyxy, title=""):
    """labels_xyxy: (N,6) cls,tid,x1,y1,x2,y2"""
    out = img.copy()
    small = medium = large = 0
    for row in labels_xyxy:
        cls_id = int(row[0])
        x1, y1, x2, y2 = int(row[2]), int(row[3]), int(row[4]), int(row[5])
        bw, bh = x2 - x1, y2 - y1
        if max(bw, bh) < 32:
            small += 1
        elif max(bw, bh) < 96:
            medium += 1
        else:
            large += 1
        color = CLS_COLORS[cls_id % len(CLS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        name = CLS_NAMES[cls_id] if cls_id < len(CLS_NAMES) else str(cls_id)
        cv2.putText(out, name, (x1, max(y1 - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
    stats = "S:%d  M:%d  L:%d  total:%d" % (small, medium, large, len(labels_xyxy))
    cv2.putText(out, stats, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 240, 60), 2, cv2.LINE_AA)
    cv2.putText(out, stats, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out, small, medium, large


def process_sample(img_path, label_path, out_w, out_h, pil_tf, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None, None, None

    h0, w0 = img_bgr.shape[:2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if os.path.isfile(label_path):
            labels_0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
        else:
            labels_0 = np.zeros((0, 6), np.float32)

    # BEFORE: letterbox only
    img_lb, rb, dw_b, dh_b = letterbox(img_bgr.copy(), height=out_h, width=out_w)
    if len(labels_0):
        lb = labels_0.copy()
        lbs_before = np.stack([
            lb[:, 0], lb[:, 1],
            rb * w0 * (lb[:, 2] - lb[:, 4] / 2) + dw_b,
            rb * h0 * (lb[:, 3] - lb[:, 5] / 2) + dh_b,
            rb * w0 * (lb[:, 2] + lb[:, 4] / 2) + dw_b,
            rb * h0 * (lb[:, 3] + lb[:, 5] / 2) + dh_b,
        ], axis=1)
    else:
        lbs_before = np.zeros((0, 6), np.float32)

    img_before, s_b, m_b, l_b = draw_boxes(img_lb, lbs_before, "BEFORE  " + os.path.basename(img_path))

    # AFTER: PIL pipeline + letterbox
    img_pil = PIL.Image.fromarray(img_bgr[:, :, ::-1])

    if len(labels_0) > 0:
        boxes_xyxy = np.stack([
            w0 * (labels_0[:, 2] - labels_0[:, 4] / 2),
            h0 * (labels_0[:, 3] - labels_0[:, 5] / 2),
            w0 * (labels_0[:, 2] + labels_0[:, 4] / 2),
            h0 * (labels_0[:, 3] + labels_0[:, 5] / 2),
        ], axis=1).astype(np.float32)
        target = {
            "boxes": torch.tensor(boxes_xyxy),
            "labels": torch.tensor(labels_0[:, :2]),
            "size": torch.tensor([h0, w0]),
        }
    else:
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0, 2), dtype=torch.float32),
            "size": torch.tensor([h0, w0]),
        }

    img_pil_aug, target_aug = pil_tf(img_pil, target)
    img_aug_bgr = cv2.cvtColor(np.array(img_pil_aug), cv2.COLOR_RGB2BGR)
    img_aug_lb, ra, dw_a, dh_a = letterbox(img_aug_bgr, height=out_h, width=out_w)

    if len(target_aug["boxes"]) > 0:
        boxes = target_aug["boxes"].numpy()
        cls_trk = target_aug["labels"].numpy()
        lbs_after = np.stack([
            cls_trk[:, 0], cls_trk[:, 1],
            boxes[:, 0] * ra + dw_a,
            boxes[:, 1] * ra + dh_a,
            boxes[:, 2] * ra + dw_a,
            boxes[:, 3] * ra + dh_a,
        ], axis=1)
    else:
        lbs_after = np.zeros((0, 6), np.float32)

    img_after, s_a, m_a, l_a = draw_boxes(img_aug_lb, lbs_after, "AFTER")
    return img_before, img_after, (s_b, m_b, l_b), (s_a, m_a, l_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="VisDrone2019-7cls")
    ap.add_argument("--list-file", default="VisDrone2019-7cls/VisDrone6cls.train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--input-wh", default="892,512")
    ap.add_argument("--out-dir", default="/tmp/aug_viz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    W, H = map(int, args.input_wh.split(","))
    os.makedirs(args.out_dir, exist_ok=True)

    pil_tf = build_aerial_mot_transforms()
    print("Pipeline:", [type(t).__name__ for t in pil_tf.transforms])

    with open(args.list_file) as fh:
        all_paths = [line.strip() for line in fh if line.strip()]

    sampled = random.sample(all_paths, min(args.n, len(all_paths)))
    print("\nVisualizing %d samples at %dx%d -> %s" % (len(sampled), W, H, args.out_dir))
    print("Left = letterbox only.  Right = PIL pipeline + letterbox.\n")

    total_sb = total_mb = total_lb = 0
    total_sa = total_ma = total_la = 0
    processed = 0

    for i, rel_path in enumerate(sampled):
        img_path = os.path.join(args.data_root, rel_path) if not os.path.isabs(rel_path) else rel_path
        label_path = img_path.replace("images", "labels_with_ids").replace(".jpg", ".txt").replace(".png", ".txt")

        before, after, sb, sa = process_sample(img_path, label_path, W, H, pil_tf, seed=args.seed + i)

        if before is None:
            print("  [skip]", img_path)
            continue

        th = min(before.shape[0], after.shape[0])
        tw = min(before.shape[1], after.shape[1])
        before = cv2.resize(before, (tw, th))
        after = cv2.resize(after, (tw, th))
        side_by_side = np.concatenate([before, after], axis=1)

        out_path = os.path.join(args.out_dir, "sample_%03d.jpg" % i)
        cv2.imwrite(out_path, side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 92])

        total_sb += sb[0]; total_mb += sb[1]; total_lb += sb[2]
        total_sa += sa[0]; total_ma += sa[1]; total_la += sa[2]
        processed += 1
        print("  [%3d/%d] %s  before S/M/L=%s  after S/M/L=%s" % (
            i + 1, len(sampled), os.path.basename(img_path), str(sb), str(sa)))

    print("\n" + "=" * 60)
    print("ANALYSIS over %d samples:" % processed)
    print("  BEFORE  small:%4d  medium:%4d  large:%4d  total:%d" % (
        total_sb, total_mb, total_lb, total_sb + total_mb + total_lb))
    print("  AFTER   small:%4d  medium:%4d  large:%4d  total:%d" % (
        total_sa, total_ma, total_la, total_sa + total_ma + total_la))
    if total_sb > 0:
        change = (total_sa - total_sb) / max(total_sb, 1) * 100
        print("  small count change: %+.1f%%" % change)
    print("\nDone. Open %s/" % args.out_dir)


if __name__ == "__main__":
    main()
