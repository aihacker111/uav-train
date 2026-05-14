# encoding=utf-8
"""
Analyze class distribution, box statistics, and track counts
from the converted AMOT-format dataset (output of gen_dataset_7cls.py).

Usage:
    python data_analyze.py --data_root /path/to/VisDrone2019-7cls
                           --splits train val
                           --out_dir  ./analyze_output
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
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

CLS_COLORS = {
    0: '#4e79a7',  # pedestrian  — blue
    1: '#f28e2b',  # people      — orange
    2: '#e15759',  # car         — red
    3: '#76b7b2',  # truck       — teal
    4: '#59a14f',  # motorcycle  — green
    5: '#edc948',  # bicycle     — yellow
    6: '#b07aa1',  # bus         — purple
}

NUM_CLASSES = len(CLS_NAMES)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_labels(labels_root):
    """
    Read all label files under labels_root/<seq>/<frame>.txt.
    Returns:
        annotations: list of (cls_id, track_id, cx, cy, w, h)
        per_frame:   dict seq → list of frame annotation counts
        track_ids:   dict cls_id → set of global track_ids
    """
    annotations = []
    per_frame   = defaultdict(list)
    track_ids   = defaultdict(set)

    seqs = sorted(os.listdir(labels_root))
    for seq in tqdm(seqs, desc=f'  Reading {os.path.basename(labels_root)}'):
        seq_dir = os.path.join(labels_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for fname in sorted(os.listdir(seq_dir)):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(seq_dir, fname)
            count = 0
            with open(fpath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    cls_id   = int(parts[0])
                    track_id = int(parts[1])
                    cx, cy, w, h = map(float, parts[2:6])
                    annotations.append((cls_id, track_id, cx, cy, w, h))
                    track_ids[cls_id].add(track_id)
                    count += 1
            per_frame[seq].append(count)

    return annotations, per_frame, track_ids


def analyze_split(data_root, split):
    labels_root = os.path.join(data_root, split, 'labels_with_ids')
    if not os.path.isdir(labels_root):
        print(f'  [Skip] {labels_root} not found')
        return None

    annotations, per_frame, track_ids = parse_labels(labels_root)

    ann_arr = np.array([(a[0], a[1], a[2], a[3], a[4], a[5])
                        for a in annotations], dtype=np.float32)

    cls_count   = defaultdict(int)
    cls_w       = defaultdict(list)
    cls_h       = defaultdict(list)
    cls_area    = defaultdict(list)
    cls_aspect  = defaultdict(list)

    for cls_id, tid, cx, cy, w, h in annotations:
        cls_count[cls_id]  += 1
        cls_w[cls_id].append(w)
        cls_h[cls_id].append(h)
        cls_area[cls_id].append(w * h)
        cls_aspect[cls_id].append(w / h if h > 0 else 0)

    # Frames stats
    all_frame_counts = []
    total_frames = 0
    for seq, counts in per_frame.items():
        all_frame_counts.extend(counts)
        total_frames += len(counts)

    return {
        'split':          split,
        'total_ann':      len(annotations),
        'total_frames':   total_frames,
        'total_seqs':     len(per_frame),
        'cls_count':      dict(cls_count),
        'cls_w':          dict(cls_w),
        'cls_h':          dict(cls_h),
        'cls_area':       dict(cls_area),
        'cls_aspect':     dict(cls_aspect),
        'track_ids':      {k: len(v) for k, v in track_ids.items()},
        'frame_counts':   all_frame_counts,
        'per_frame':      per_frame,
    }


# ── Text report ───────────────────────────────────────────────────────────────

def print_report(stats):
    s = stats
    print(f'\n{"="*60}')
    print(f'  Split: {s["split"].upper()}')
    print(f'{"="*60}')
    print(f'  Sequences  : {s["total_seqs"]}')
    print(f'  Frames     : {s["total_frames"]}')
    print(f'  Annotations: {s["total_ann"]}')
    print(f'  Avg obj/frame: {np.mean(s["frame_counts"]):.1f}  '
          f'(max {max(s["frame_counts"])}, min {min(s["frame_counts"])})')
    print()
    print(f'  {"Class":<14} {"Count":>8}  {"Tracks":>8}  '
          f'{"W mean":>8}  {"H mean":>8}  {"Area mean":>10}')
    print(f'  {"-"*70}')
    for cls_id in range(NUM_CLASSES):
        cnt  = s['cls_count'].get(cls_id, 0)
        trk  = s['track_ids'].get(cls_id, 0)
        if cnt == 0:
            continue
        w_m  = np.mean(s['cls_w'][cls_id]) * 100
        h_m  = np.mean(s['cls_h'][cls_id]) * 100
        ar_m = np.mean(s['cls_area'][cls_id]) * 100 * 100
        print(f'  {CLS_NAMES[cls_id]:<14} {cnt:>8}  {trk:>8}  '
              f'{w_m:>7.2f}%  {h_m:>7.2f}%  {ar_m:>9.4f}%²')


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_class_distribution(stats_list, out_dir):
    fig, axes = plt.subplots(1, len(stats_list), figsize=(7 * len(stats_list), 5))
    if len(stats_list) == 1:
        axes = [axes]

    for ax, s in zip(axes, stats_list):
        cls_ids = [i for i in range(NUM_CLASSES) if i in s['cls_count']]
        counts  = [s['cls_count'][i] for i in cls_ids]
        names   = [CLS_NAMES[i] for i in cls_ids]
        colors  = [CLS_COLORS[i] for i in cls_ids]

        bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(f'{s["split"].upper()} — Class Distribution', fontsize=13, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Annotation count')
        ax.tick_params(axis='x', rotation=30)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                    f'{cnt:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'class_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_box_size(stats_list, out_dir):
    fig, axes = plt.subplots(2, len(stats_list), figsize=(7 * len(stats_list), 8))
    if len(stats_list) == 1:
        axes = axes.reshape(2, 1)

    for col, s in enumerate(stats_list):
        # Width distribution
        ax_w = axes[0][col]
        for cls_id in range(NUM_CLASSES):
            if cls_id not in s['cls_w']:
                continue
            ws = np.array(s['cls_w'][cls_id]) * 100
            ax_w.hist(ws, bins=50, alpha=0.5, color=CLS_COLORS[cls_id],
                      label=CLS_NAMES[cls_id], density=True)
        ax_w.set_title(f'{s["split"].upper()} — Box Width (%)', fontsize=11)
        ax_w.set_xlabel('Width (% of image width)')
        ax_w.set_ylabel('Density')
        ax_w.legend(fontsize=7)
        ax_w.set_xlim(0, 20)

        # Height distribution
        ax_h = axes[1][col]
        for cls_id in range(NUM_CLASSES):
            if cls_id not in s['cls_h']:
                continue
            hs = np.array(s['cls_h'][cls_id]) * 100
            ax_h.hist(hs, bins=50, alpha=0.5, color=CLS_COLORS[cls_id],
                      label=CLS_NAMES[cls_id], density=True)
        ax_h.set_title(f'{s["split"].upper()} — Box Height (%)', fontsize=11)
        ax_h.set_xlabel('Height (% of image height)')
        ax_h.set_ylabel('Density')
        ax_h.legend(fontsize=7)
        ax_h.set_xlim(0, 20)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'box_size_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_objects_per_frame(stats_list, out_dir):
    fig, axes = plt.subplots(1, len(stats_list), figsize=(7 * len(stats_list), 4))
    if len(stats_list) == 1:
        axes = [axes]

    for ax, s in zip(axes, stats_list):
        counts = s['frame_counts']
        ax.hist(counts, bins=40, color='#4e79a7', edgecolor='white', linewidth=0.5)
        ax.axvline(np.mean(counts), color='red', linestyle='--',
                   label=f'Mean: {np.mean(counts):.1f}')
        ax.set_title(f'{s["split"].upper()} — Objects per Frame', fontsize=11)
        ax.set_xlabel('Object count per frame')
        ax.set_ylabel('Frame count')
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'objects_per_frame.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_track_counts(stats_list, out_dir):
    fig, axes = plt.subplots(1, len(stats_list), figsize=(7 * len(stats_list), 5))
    if len(stats_list) == 1:
        axes = [axes]

    for ax, s in zip(axes, stats_list):
        cls_ids = [i for i in range(NUM_CLASSES) if i in s['track_ids']]
        counts  = [s['track_ids'][i] for i in cls_ids]
        names   = [CLS_NAMES[i] for i in cls_ids]
        colors  = [CLS_COLORS[i] for i in cls_ids]

        bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(f'{s["split"].upper()} — Unique Track IDs per Class', fontsize=11)
        ax.set_xlabel('Class')
        ax.set_ylabel('Unique tracks')
        ax.tick_params(axis='x', rotation=30)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                    f'{cnt:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'track_counts.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/Users/tinvo0908/Desktop/AMOT/VisDrone2019-7cls')
    parser.add_argument('--splits',    nargs='+', default=['train', 'val'])
    parser.add_argument('--out_dir',   default='/Users/tinvo0908/Desktop/AMOT/data_preprocessing/analyze_output')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stats_list = []
    for split in args.splits:
        print(f'\nAnalyzing [{split}]...')
        s = analyze_split(args.data_root, split)
        if s is not None:
            stats_list.append(s)
            print_report(s)

    if not stats_list:
        print('No data found.')
        return

    print('\nGenerating plots...')
    plot_class_distribution(stats_list, args.out_dir)
    plot_box_size(stats_list, args.out_dir)
    plot_objects_per_frame(stats_list, args.out_dir)
    plot_track_counts(stats_list, args.out_dir)

    print(f'\nAll outputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
