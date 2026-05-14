# encoding=utf-8
"""
Convert raw VisDrone MOT annotations → AMOT label format with 7-class mapping.

VisDrone 10 classes (1-indexed in annotation file):
  1:pedestrian  2:people  3:bicycle  4:car  5:van
  6:truck  7:tricycle  8:awning-tricycle  9:bus  10:motor

Output 7 classes (0-indexed):
  0:pedestrian  1:people  2:car  3:truck  4:motorcycle  5:bicycle  6:bus

Merge rules:
  pedestrian(0)              → pedestrian(0)   [individual, trackable]
  people(1)                  → people(1)        [crowd / heavily occluded]
  car(3)        + van(4)     → car(2)
  truck(5)                   → truck(3)
  motor(9)                   → motorcycle(4)
  bicycle(2)                 → bicycle(5)
  bus(8)                     → bus(6)
  tricycle(6) + awning-tricycle(7) → DROPPED

Track ID note:
  car+van merge into one class — their original track IDs may collide within
  the same sequence. We use (original_cls_id, original_target_id) as a unique
  key to avoid this collision.
"""

import os
import copy
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm

# ── Class mapping ─────────────────────────────────────────────────────────────

# original cls_id (0-indexed) → new cls_id (0-indexed), None = drop
REMAP = {
    0: 0,     # pedestrian      → pedestrian (0)
    1: 1,     # people          → people (1)
    2: 5,     # bicycle         → bicycle (5)
    3: 2,     # car             → car (2)
    4: 2,     # van             → car (2)
    5: 3,     # truck           → truck (3)
    6: None,  # tricycle        → drop
    7: None,  # awning-tricycle → drop
    8: 6,     # bus             → bus (6)
    9: 4,     # motor           → motorcycle (4)
}

NEW_ID2CLS = {
    0: 'pedestrian',
    1: 'people',
    2: 'car',
    3: 'truck',
    4: 'motorcycle',
    5: 'bicycle',
    6: 'bus',
}

NUM_NEW_CLASSES = len(NEW_ID2CLS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_ignore_regions(img, boxes):
    for box in boxes:
        box = list(map(lambda x: int(x + 0.5), box))
        img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = [0, 0, 0]
    return img


def gen_dot_index_file(data_root, rel_path, out_root, f_name):
    """
    Write a .train / .val index file listing relative image paths.
    Each line: <rel_path>/<seq>/<frame>.jpg
    """
    out_path = os.path.join(out_root, f_name)
    cnt = 0
    with open(out_path, 'w') as f:
        root = os.path.join(data_root, rel_path)
        seqs = sorted(os.listdir(root))
        for seq in seqs:
            img_dir = os.path.join(root, seq)
            for img in sorted(os.listdir(img_dir)):
                if img.endswith('.jpg'):
                    line = os.path.join(rel_path, seq, img)
                    f.write(line + '\n')
                    cnt += 1
    print(f'Index written → {out_path}  ({cnt} images)')


# ── Core conversion ───────────────────────────────────────────────────────────

def gen_track_dataset(src_root, dst_root, viz_root=None):
    """
    Convert one VisDrone MOT split (train or val) into AMOT label format.

    src_root layout:
        src_root/sequences/<seq>/<frame>.jpg
        src_root/annotations/<seq>.txt

    dst_root layout (output):
        dst_root/images/<seq>/<frame>.jpg
        dst_root/labels_with_ids/<seq>/<frame>.txt
    """
    os.makedirs(dst_root, exist_ok=True)
    dst_img_root = os.path.join(dst_root, 'images')
    dst_txt_root = os.path.join(dst_root, 'labels_with_ids')
    os.makedirs(dst_img_root, exist_ok=True)
    os.makedirs(dst_txt_root, exist_ok=True)

    # Global track ID counter per new class (accumulates across sequences)
    global_start_id = defaultdict(int)  # new_cls_id → next available track_id

    frame_cnt = 0
    seq_names = sorted(os.listdir(os.path.join(src_root, 'sequences')))

    for seq in tqdm(seq_names, desc='Sequences'):
        print(f'\nProcessing: {seq}')

        seq_img_dir = os.path.join(src_root, 'sequences', seq)
        seq_ann_path = os.path.join(src_root, 'annotations', seq + '.txt')

        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_ann_path)):
            print(f'  [Warning] missing images or annotation, skipping.')
            continue

        os.makedirs(os.path.join(dst_img_root, seq), exist_ok=True)
        os.makedirs(os.path.join(dst_txt_root, seq), exist_ok=True)

        # ── Parse annotation file ─────────────────────────────────────────────
        with open(seq_ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        n = len(lines)
        seq_arr = np.zeros((n, 10), dtype=np.int32)
        for i, line in enumerate(lines):
            seq_arr[i] = [int(x) for x in line.strip().split(',')]

        ignore_mask = seq_arr[:, 7] == 0
        obj_mask    = (seq_arr[:, 7] > 0) & (seq_arr[:, 7] < 11)

        ignore_labels = seq_arr[ignore_mask]
        obj_labels    = seq_arr[obj_mask]

        # Group by frame
        ignore_by_frame = defaultdict(list)
        obj_by_frame    = defaultdict(list)
        for row in ignore_labels:
            ignore_by_frame[row[0]].append(row[2:6])
        for row in obj_labels:
            obj_by_frame[row[0]].append(row)

        # ── Build per-sequence track ID mapping ───────────────────────────────
        # Key: (orig_cls_id, orig_target_id) → unique within this sequence
        # Value: new sequential track_id (globally unique across sequences)
        #
        # We collect all unique (orig_cls, target_id) pairs first,
        # then assign sequential IDs per new class.

        # Step 1: collect unique (orig_cls, target_id) per new class
        pairs_per_new_cls = defaultdict(set)  # new_cls_id → set of (orig_cls, target_id)
        for row in obj_labels:
            orig_cls  = row[7] - 1  # 0-indexed
            new_cls   = REMAP.get(orig_cls)
            if new_cls is None:
                continue
            target_id = row[1]
            pairs_per_new_cls[new_cls].add((orig_cls, target_id))

        # Step 2: assign sequential IDs within this sequence (globally offset)
        pair2track = {}  # (orig_cls, target_id) → global track_id
        seq_max_new_ids = defaultdict(int)
        for new_cls, pairs in pairs_per_new_cls.items():
            sorted_pairs = sorted(pairs)
            for i, pair in enumerate(sorted_pairs):
                pair2track[pair] = global_start_id[new_cls] + i + 1
            seq_max_new_ids[new_cls] = len(sorted_pairs)

        for new_cls, cnt in seq_max_new_ids.items():
            print(f'  {NEW_ID2CLS[new_cls]:12s}: {cnt} tracks  (global start {global_start_id[new_cls]})')

        # ── Process frames ────────────────────────────────────────────────────
        for fr_id in sorted(obj_by_frame.keys()):
            fr_name = '{:07d}.jpg'.format(fr_id)
            fr_path = os.path.join(seq_img_dir, fr_name)
            if not os.path.isfile(fr_path):
                continue

            img = cv2.imread(fr_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W, _ = img.shape

            # Paint ignore regions on image before saving
            draw_ignore_regions(img, ignore_by_frame[fr_id])

            dst_img_path = os.path.join(dst_img_root, seq, fr_name)
            if not os.path.isfile(dst_img_path):
                cv2.imwrite(dst_img_path, img)

            if viz_root is not None:
                viz_dir = os.path.join(viz_root, seq)
                os.makedirs(viz_dir, exist_ok=True)
                img_viz = copy.deepcopy(img)

            # Generate label lines
            label_lines = []
            for row in obj_by_frame[fr_id]:
                orig_cls  = row[7] - 1
                new_cls   = REMAP.get(orig_cls)
                if new_cls is None:
                    continue

                target_id  = row[1]
                occlusion  = row[9]
                if occlusion > 1:  # drop heavy occlusion (>50%)
                    continue

                track_id = pair2track.get((orig_cls, target_id))
                if track_id is None:
                    continue

                x, y, bw, bh = row[2], row[3], row[4], row[5]

                if viz_root is not None:
                    cv2.rectangle(img_viz, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(img_viz, f'{NEW_ID2CLS[new_cls]}#{track_id}',
                                (x, y + 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (225, 255, 255), 1)

                # Normalize
                cx = (x + bw * 0.5) / W
                cy = (y + bh * 0.5) / H
                nw = bw / W
                nh = bh / H

                label_lines.append(
                    f'{new_cls} {track_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n'
                )

            if viz_root is not None:
                cv2.imwrite(os.path.join(viz_dir, fr_name), img_viz)

            label_path = os.path.join(dst_txt_root, seq, fr_name.replace('.jpg', '.txt'))
            with open(label_path, 'w', encoding='utf-8') as f:
                f.writelines(label_lines)

            frame_cnt += 1

        # Advance global track ID counters
        for new_cls, cnt in seq_max_new_ids.items():
            global_start_id[new_cls] += cnt

    print(f'\nTotal frames processed: {frame_cnt}')
    print('Global track counts per class:')
    for new_cls in range(NUM_NEW_CLASSES):
        print(f'  {NEW_ID2CLS[new_cls]:12s}: {global_start_id[new_cls]} total tracks')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Paths — adjust if needed ──────────────────────────────────────────────
    # Raw VisDrone MOT data (sequences/ + annotations/)
    TRAIN_SRC = '/Users/tinvo0908/Desktop/AMOT/VisDrone2019-MOT-train'
    VAL_SRC   = '/Users/tinvo0908/Desktop/AMOT/VisDrone2019-MOT-val'

    # Output root (converted AMOT format)
    OUT_ROOT  = '/Users/tinvo0908/Desktop/AMOT/VisDrone2019-7cls'

    # Step 1: Convert train
    print('=' * 60)
    print('Converting TRAIN split...')
    print('=' * 60)
    gen_track_dataset(
        src_root=TRAIN_SRC,
        dst_root=os.path.join(OUT_ROOT, 'train'),
        viz_root=None,
    )

    # Step 2: Convert val
    print('=' * 60)
    print('Converting VAL split...')
    print('=' * 60)
    gen_track_dataset(
        src_root=VAL_SRC,
        dst_root=os.path.join(OUT_ROOT, 'val'),
        viz_root=None,
    )

    # Step 3: Generate index files
    gen_dot_index_file(
        data_root=OUT_ROOT,
        rel_path='train/images',
        out_root=OUT_ROOT,
        f_name='VisDrone6cls.train',
    )
    gen_dot_index_file(
        data_root=OUT_ROOT,
        rel_path='val/images',
        out_root=OUT_ROOT,
        f_name='VisDrone6cls.val',
    )

    print('\nDone. Output at:', OUT_ROOT)
    print('Update visdrone.json to point to:', OUT_ROOT)
