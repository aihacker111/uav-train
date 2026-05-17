"""
AMOT Video Demo
---------------
Run detection + tracking on a video file and save annotated output.

Usage:
    python demo_video.py \
        --input  path/to/video.mp4 \
        --output path/to/output.mp4 \
        --model  path/to/model.pth \
        --arch   hybrid_small \
        --conf   0.35 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import _init_paths  # noqa: F401  (sets up lib/ on sys.path)

from lib.models.model import create_model, load_model
from lib.tracker.multitracker import HybridMCJDETracker, MCJDETracker
from lib.opts import opts

# ── class names ───────────────────────────────────────────────────────────────
ID2CLS = {
    0: 'pedestrian',
    1: 'people',
    2: 'car',
    3: 'truck',
    4: 'motorcycle',
    5: 'bicycle',
    6: 'bus',
}

# Class color palette (BGR) — visually distinct per class
_CLS_PALETTE = [
    (255, 128,   0),   # 0 pedestrian  — orange
    (255, 200,   0),   # 1 people      — yellow-orange
    (  0, 160, 255),   # 2 car         — sky-blue
    (180,   0, 220),   # 3 truck       — purple
    (255,  60, 180),   # 4 motorcycle  — pink
    ( 50, 220,  50),   # 5 bicycle     — green
    (220,  40,  40),   # 6 bus         — red
]


# ── visual utilities ──────────────────────────────────────────────────────────

def _track_color(track_id: int, cls_id: int) -> tuple[int, int, int]:
    """
    Blend class base color with a per-ID hue shift so every track has a
    unique but class-recognisable color.
    """
    base = np.array(_CLS_PALETTE[cls_id % len(_CLS_PALETTE)], dtype=np.float32)
    rng  = np.random.default_rng(track_id * 2654435761 & 0xFFFF)
    shift = rng.uniform(-40, 40, size=3)
    color = np.clip(base + shift, 0, 255).astype(int)
    return (int(color[0]), int(color[1]), int(color[2]))


def _draw_box(img, x1, y1, x2, y2, color, thickness=2):
    """Draw a plain bounding box."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_label(img, text, x, y, color, font_scale=0.45, thickness=1):
    """
    Draw a pill-shaped label badge above (x, y).
    Returns the label height so callers can stack labels.
    """
    font      = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 5, 3

    bx1 = x
    by1 = y - th - 2 * pad_y - baseline
    bx2 = x + tw + 2 * pad_x
    by2 = y

    # Filled background with slight transparency
    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    cv2.putText(img, text, (bx1 + pad_x, by2 - pad_y - baseline),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return th + 2 * pad_y


def _draw_track(img, tlwh, track_id, cls_id, score):
    """Draw one tracked object: plain box + label badge."""
    x1 = int(tlwh[0])
    y1 = int(tlwh[1])
    x2 = int(tlwh[0] + tlwh[2])
    y2 = int(tlwh[1] + tlwh[3])

    color  = _track_color(track_id, cls_id)
    cls_nm = ID2CLS.get(cls_id, f'cls{cls_id}')

    # ── bounding box ────────────────────────────────────────────────────────
    _draw_box(img, x1, y1, x2, y2, color, thickness=2)

    # ── label: class name + ID + confidence ─────────────────────────────────
    label  = f'{cls_nm}  #{track_id}  {score:.0%}'
    label_y = max(y1, 20)
    _draw_label(img, label, x1, label_y, color, font_scale=0.42, thickness=1)


def _draw_hud(img, frame_id: int, fps: float, counts: dict[int, int]):
    """
    Top-left heads-up display: frame / FPS / object counts per class.
    """
    h, w = img.shape[:2]
    panel_w  = 220
    panel_h  = 30 + 18 * max(len(counts), 1) + 10
    panel_x  = 10
    panel_y  = 10

    # Semi-transparent dark panel
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    font  = cv2.FONT_HERSHEY_DUPLEX
    small = 0.38
    med   = 0.44

    # Header row
    cv2.putText(img, f'Frame {frame_id:05d}   {fps:.1f} FPS',
                (panel_x + 8, panel_y + 18),
                font, med, (200, 255, 200), 1, cv2.LINE_AA)

    # Per-class counts
    row_y = panel_y + 34
    total = 0
    for cls_id, cnt in sorted(counts.items()):
        if cnt == 0:
            continue
        dot_color = _CLS_PALETTE[cls_id % len(_CLS_PALETTE)]
        cv2.circle(img, (panel_x + 14, row_y - 4), 4, dot_color, cv2.FILLED)
        cv2.putText(img,
                    f'{ID2CLS.get(cls_id, cls_id):<18s} {cnt:>3d}',
                    (panel_x + 24, row_y),
                    font, small, (220, 220, 220), 1, cv2.LINE_AA)
        row_y += 18
        total += cnt

    # Total line
    cv2.putText(img,
                f'{"total":<18s} {total:>3d}',
                (panel_x + 24, row_y + 2),
                font, small, (255, 255, 100), 1, cv2.LINE_AA)


# ── tracker wrapper ───────────────────────────────────────────────────────────

class VideoTracker:
    def __init__(self, opt):
        self.opt     = opt
        self.is_hybrid = 'hybrid' in opt.arch
        self._use_imagenet = 'lwdetr' in opt.arch or 'hybrid' in opt.arch

        if self.is_hybrid:
            self.tracker = HybridMCJDETracker(opt, frame_rate=opt.frame_rate)
        else:
            self.tracker = MCJDETracker(opt, frame_rate=opt.frame_rate)

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """BGR frame → normalised CHW tensor on device."""
        target_h, target_w = self.opt.img_size   # (H, W)
        h0, w0 = frame.shape[:2]
        ratio  = min(target_h / h0, target_w / w0)
        new_h, new_w = int(h0 * ratio), int(w0 * ratio)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas  = np.full((target_h, target_w, 3), 127.5, dtype=np.float32)
        pad_top  = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        img = canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0   # BGR→RGB, HWC→CHW

        if self._use_imagenet:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            img  = (img - mean) / std

        tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.opt.device)
        return tensor

    def update(self, frame: np.ndarray):
        """
        Run one tracking step.
        Returns list of dicts:  {tlwh, track_id, cls_id, score}
        """
        blob   = self.preprocess(frame)
        result = self.tracker.update_tracking(blob, frame)

        tracks = []
        for cls_id, cls_tracks in result.items():
            for trk in cls_tracks:
                tlwh  = trk.curr_tlwh
                tid   = trk.track_id
                score = float(trk.score)

                # Filter: area and confidence threshold.
                # The tracker internally uses low-score detections (0.1→conf_thres)
                # in third-association to keep tracks alive, so trk.score can fall
                # below conf_thres. Re-apply the threshold here to avoid showing
                # those borderline tracks in the output.
                if tlwh[2] * tlwh[3] < self.opt.min_box_area:
                    continue
                if score < self.opt.conf_thres:
                    continue

                tracks.append({
                    'tlwh':     tlwh,
                    'track_id': tid,
                    'cls_id':   int(cls_id),
                    'score':    score,
                })
        return tracks


# ── main ──────────────────────────────────────────────────────────────────────

def run(opt, input_path: str, output_path: str, speed: float = 0.5):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {input_path}')

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    opt.frame_rate = int(src_fps)

    # Output FPS = src_fps × speed  →  video plays slower when speed < 1.0
    out_fps = max(1.0, src_fps * speed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (src_w, src_h))

    tracker  = VideoTracker(opt)
    frame_id = 0
    fps_ema  = src_fps

    print(f'Input : {input_path}  ({src_w}×{src_h} @ {src_fps:.1f} fps, {total_f} frames)')
    print(f'Output FPS: {out_fps:.1f}  (speed={speed}x)')
    print(f'Output: {output_path}')
    print('Processing...')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        t0     = time.perf_counter()
        tracks = tracker.update(frame)
        elapsed = time.perf_counter() - t0
        fps_ema  = 0.9 * fps_ema + 0.1 * (1.0 / max(elapsed, 1e-4))

        # ── annotate ──────────────────────────────────────────────────────
        canvas = frame.copy()
        counts: dict[int, int] = defaultdict(int)

        for trk in tracks:
            _draw_track(canvas, trk['tlwh'], trk['track_id'], trk['cls_id'], trk['score'])
            counts[trk['cls_id']] += 1

        _draw_hud(canvas, frame_id, fps_ema, counts)

        writer.write(canvas)

        if frame_id % 30 == 0:
            pct = 100 * frame_id / max(total_f, 1)
            print(f'  [{frame_id:>6}/{total_f}]  {pct:.1f}%  {fps_ema:.1f} fps  '
                  f'tracks={sum(counts.values())}')

    cap.release()
    writer.release()
    print(f'\nDone — saved to {output_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='AMOT video demo')
    p.add_argument('--input',   required=True,         help='Input video path')
    p.add_argument('--output',  default='output.mp4',  help='Output video path')
    p.add_argument('--model',   required=True,         help='Path to .pth checkpoint')
    p.add_argument('--arch',    default='hybrid_small',
                   help='Model arch: hybrid_tiny | hybrid_small | hybrid_base')
    p.add_argument('--conf',    type=float, default=0.35, help='Detection confidence threshold')
    p.add_argument('--nms',     type=float, default=0.45, help='NMS IoU threshold')
    p.add_argument('--input-wh',type=str,   default='832,512',
                   help='Model input resolution W,H (must match training)')
    p.add_argument('--K',       type=int,   default=200, help='Max objects per frame')
    p.add_argument('--min-box-area', type=float, default=100, help='Min box area (px²)')
    p.add_argument('--track-buffer',  type=int, default=30, help='Lost-track buffer (frames)')
    p.add_argument('--reid-dim', type=int, default=128, help='ReID embedding dimension')
    p.add_argument('--device',  default='cuda:0',
                   help='Device: cuda:0 | cuda:1 | cpu')
    p.add_argument('--num-classes', type=int, default=7, help='Number of object classes')
    p.add_argument('--speed', type=float, default=0.5,
                   help='Output playback speed multiplier. 0.5 = half speed, 1.0 = real-time.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Build a minimal opt compatible with the tracker
    _opt = opts().parse(args=[
        '--task',        'hybrid',
        '--arch',         args.arch,
        '--load_model',   args.model,
        '--conf_thres',   str(args.conf),
        '--nms_thres',    str(args.nms),
        '--input-wh',     args.input_wh,
        '--K',            str(args.K),
        '--min-box-area', str(args.min_box_area),
        '--track_buffer', str(args.track_buffer),
        '--reid_dim',     str(args.reid_dim),
        '--gpus',         args.device.replace('cuda:', '') if 'cuda' in args.device else '-1',
        '--reid_cls_ids', ','.join(str(i) for i in range(args.num_classes)),
        '--num_workers',  '0',
    ])
    _opt.device      = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    _opt.num_classes = args.num_classes
    _opt.nID_dict    = {}
    _opt.img_size    = (int(args.input_wh.split(',')[1]),   # H
                        int(args.input_wh.split(',')[0]))   # W

    # Attributes normally set by update_dataset_info_and_set_heads()
    _opt.heads      = {}    # hybrid model builds its own heads from HybridModelConfig
    _opt.head_conv  = 256
    _opt.mean       = [0.485, 0.456, 0.406]
    _opt.std        = [0.229, 0.224, 0.225]
    _opt.input_h    = int(args.input_wh.split(',')[1])
    _opt.input_w    = int(args.input_wh.split(',')[0])
    _opt.output_h   = _opt.input_h // _opt.down_ratio
    _opt.output_w   = _opt.input_w // _opt.down_ratio
    _opt.input_res  = max(_opt.input_h, _opt.input_w)
    _opt.output_res = max(_opt.output_h, _opt.output_w)

    # Detection / tracking thresholds
    _opt.conf_thres   = args.conf
    _opt.det_thres    = args.conf
    _opt.nms_thres    = args.nms
    _opt.min_box_area = args.min_box_area

    run(_opt, args.input, args.output, speed=args.speed)
