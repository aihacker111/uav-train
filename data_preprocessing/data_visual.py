# encoding=utf-8
"""
Visualize converted AMOT-format labels overlaid on images.

Usage examples:
  # Visualize 20 random frames from train split
  python data_visual.py --data_root /path/to/VisDrone2019-7cls --split train --mode random --n 20

  # Visualize specific sequence (all frames)
  python data_visual.py --data_root /path/to/VisDrone2019-7cls --split train --mode seq --seq uav0000013_00000_v

  # Visualize first N frames of every sequence
  python data_visual.py --data_root /path/to/VisDrone2019-7cls --split train --mode first --n 3
"""

import os
import cv2
import argparse
import random
import numpy as np
from tqdm import tqdm

# ── Class definitions ─────────────────────────────────────────────────────────

CLS_NAMES = {
    0: 'pedestrian',
    1: 'people',
    2: 'car',
    3: 'truck',
    4: 'motorcycle',
    5: 'bicycle',
    6: 'bus',
}

# BGR colors for OpenCV
CLS_BGR = {
    0: (178, 121,  66),   # pedestrian  — blue
    1: ( 43, 142, 242),   # people      — orange
    2: ( 89,  87, 225),   # car         — red
    3: (178, 183, 118),   # truck       — teal
    4: ( 79, 161,  89),   # motorcycle  — green
    5: ( 72, 201, 237),   # bicycle     — yellow
    6: (161, 122, 176),   # bus         — purple
}


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_frame(img, label_path):
    """
    Draw bounding boxes and track IDs onto img (in-place).
    Labels format: cls_id track_id cx cy w h (normalized)
    """
    if not os.path.isfile(label_path):
        return img

    H, W = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue

        cls_id   = int(parts[0])
        track_id = int(parts[1])
        cx, cy, bw, bh = map(float, parts[2:6])

        # Denormalize
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        color = CLS_BGR.get(cls_id, (255, 255, 255))
        label = f'{CLS_NAMES.get(cls_id, str(cls_id))}#{track_id}'

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)

        # Label text
        cv2.putText(img, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def add_legend(img):
    """Add class legend to bottom-left corner."""
    H, W = img.shape[:2]
    pad = 6
    th  = 16
    box_w = 160
    box_h = len(CLS_NAMES) * (th + pad) + pad

    overlay = img.copy()
    cv2.rectangle(overlay, (0, H - box_h), (box_w, H), (30, 30, 30), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    for i, (cls_id, name) in enumerate(CLS_NAMES.items()):
        y = H - box_h + pad + i * (th + pad) + th
        color = CLS_BGR[cls_id]
        cv2.rectangle(img, (pad, y - th + 2), (pad + 12, y + 2), color, -1)
        cv2.putText(img, name, (pad + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    return img


def add_info_bar(img, seq, frame_name, n_objects):
    """Add info bar at top of image."""
    bar = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    info = f'Seq: {seq}  |  Frame: {frame_name}  |  Objects: {n_objects}'
    cv2.putText(bar, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return np.vstack([bar, img])


# ── Collection helpers ────────────────────────────────────────────────────────

def collect_all_frames(images_root, labels_root):
    """Return list of (seq, frame_name, img_path, label_path)."""
    frames = []
    for seq in sorted(os.listdir(images_root)):
        img_dir = os.path.join(images_root, seq)
        lbl_dir = os.path.join(labels_root, seq)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith('.jpg'):
                continue
            frames.append((
                seq,
                fname,
                os.path.join(img_dir, fname),
                os.path.join(lbl_dir, fname.replace('.jpg', '.txt')),
            ))
    return frames


# ── Modes ─────────────────────────────────────────────────────────────────────

def visualize_random(images_root, labels_root, out_dir, n, seed=42):
    frames = collect_all_frames(images_root, labels_root)
    random.seed(seed)
    chosen = random.sample(frames, min(n, len(frames)))

    out_path = os.path.join(out_dir, 'random_frames')
    os.makedirs(out_path, exist_ok=True)

    for seq, fname, img_path, lbl_path in tqdm(chosen, desc='Random frames'):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = draw_frame(img, lbl_path)
        n_obj = sum(1 for _ in open(lbl_path) if _.strip()) if os.path.isfile(lbl_path) else 0
        img = add_info_bar(img, seq, fname, n_obj)
        img = add_legend(img)
        out_name = f'{seq}__{fname}'
        cv2.imwrite(os.path.join(out_path, out_name), img)

    print(f'  Saved {len(chosen)} frames → {out_path}')


def visualize_sequence(images_root, labels_root, out_dir, seq_name):
    img_dir = os.path.join(images_root, seq_name)
    lbl_dir = os.path.join(labels_root, seq_name)

    if not os.path.isdir(img_dir):
        print(f'  [Error] Sequence not found: {seq_name}')
        return

    out_path = os.path.join(out_dir, f'seq_{seq_name}')
    os.makedirs(out_path, exist_ok=True)

    frame_names = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
    for fname in tqdm(frame_names, desc=f'Seq {seq_name}'):
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = draw_frame(img, lbl_path)
        n_obj = sum(1 for _ in open(lbl_path) if _.strip()) if os.path.isfile(lbl_path) else 0
        img = add_info_bar(img, seq_name, fname, n_obj)
        img = add_legend(img)
        cv2.imwrite(os.path.join(out_path, fname), img)

    print(f'  Saved {len(frame_names)} frames → {out_path}')


def visualize_first_n(images_root, labels_root, out_dir, n):
    out_path = os.path.join(out_dir, f'first_{n}_per_seq')
    os.makedirs(out_path, exist_ok=True)

    seqs = sorted(os.listdir(images_root))
    for seq in tqdm(seqs, desc='Sequences'):
        img_dir = os.path.join(images_root, seq)
        lbl_dir = os.path.join(labels_root, seq)
        if not os.path.isdir(img_dir):
            continue

        frame_names = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))[:n]
        for fname in frame_names:
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = draw_frame(img, lbl_path)
            n_obj = sum(1 for _ in open(lbl_path) if _.strip()) if os.path.isfile(lbl_path) else 0
            img = add_info_bar(img, seq, fname, n_obj)
            img = add_legend(img)
            out_name = f'{seq}__{fname}'
            cv2.imwrite(os.path.join(out_path, out_name), img)

    print(f'  Saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/Users/tinvo0908/Desktop/AMOT/VisDrone2019-7cls')
    parser.add_argument('--split',     default='train', choices=['train', 'val'])
    parser.add_argument('--mode',      default='random', choices=['random', 'seq', 'first'],
                        help='random: N random frames | seq: one full sequence | first: first N frames per seq')
    parser.add_argument('--seq',       default=None,    help='Sequence name (mode=seq only)')
    parser.add_argument('--n',         type=int, default=20, help='Number of frames (mode=random/first)')
    parser.add_argument('--out_dir',   default='/Users/tinvo0908/Desktop/AMOT/data_preprocessing/visual_output')
    args = parser.parse_args()

    images_root = os.path.join(args.data_root, args.split, 'images')
    labels_root = os.path.join(args.data_root, args.split, 'labels_with_ids')
    out_dir     = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(images_root):
        print(f'[Error] images not found: {images_root}')
        return

    print(f'Split: {args.split}  |  Mode: {args.mode}')

    if args.mode == 'random':
        visualize_random(images_root, labels_root, out_dir, args.n)
    elif args.mode == 'seq':
        if args.seq is None:
            print('[Error] --seq required for mode=seq')
            return
        visualize_sequence(images_root, labels_root, out_dir, args.seq)
    elif args.mode == 'first':
        visualize_first_n(images_root, labels_root, out_dir, args.n)


if __name__ == '__main__':
    main()
