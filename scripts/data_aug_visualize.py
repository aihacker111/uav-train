"""
Visualize every augmentation in transforms.py.

Usage (from the src/ directory):
    python ../scripts/data_aug_visualize.py [--image PATH] [--out aug_viz.jpg]

If --image is not given the script creates a synthetic UAV-like scene with
bounding boxes so no real data is needed to run it.

Output: one large JPEG grid — each row is one augmentation showing
        [original | augmented] side by side with boxes drawn and the
        augmentation name as a header label.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch

# ── make src/lib importable ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from lib.datasets.transforms import (
    RandomHorizontalFlip,
    RandomResize,
    ScaleBiasedCrop,
    RandomColorJitter,
    RandomGrayscale,
    RandomNightMode,
    RandomFog,
    RandomMotionBlur,
    RandomJPEGCompression,
    RandomSensorNoise,
    RandomSunGlare,
    RandomGaussianBlur,
    RandomOcclusionPatch,
    CopyPaste,
    Mosaic,
)


# ── helpers ───────────────────────────────────────────────────────────────────

LABEL_FONT_SIZE = 18
BOX_COLORS = [
    (255,  80,  80), (80, 200,  80), ( 80, 120, 255),
    (255, 200,  60), (200,  80, 255), ( 60, 220, 220),
]


def _try_font(size: int):
    for name in ('DejaVuSans-Bold.ttf', 'Arial.ttf', 'FreeSansBold.ttf'):
        try:
            return PIL.ImageFont.truetype(name, size)
        except OSError:
            pass
    return PIL.ImageFont.load_default()


def _draw_boxes(img: PIL.Image.Image, boxes_xyxy: torch.Tensor, labels: torch.Tensor) -> PIL.Image.Image:
    """Draw xyxy boxes on a copy of img."""
    out  = img.copy()
    draw = PIL.ImageDraw.Draw(out)
    font = _try_font(12)
    for i, (box, lbl) in enumerate(zip(boxes_xyxy, labels)):
        x1, y1, x2, y2 = box.tolist()
        color = BOX_COLORS[int(lbl) % len(BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, y1 + 2), str(int(lbl)), fill=color, font=font)
    return out


def _make_panel(name: str,
                before: PIL.Image.Image,
                after:  PIL.Image.Image,
                boxes_before: torch.Tensor,
                labels_before: torch.Tensor,
                boxes_after: torch.Tensor,
                labels_after: torch.Tensor,
                cell_w: int = 480,
                cell_h: int = 320) -> PIL.Image.Image:
    """Return one side-by-side panel: [label bar | orig | augmented]."""
    HEADER = 32
    panel  = PIL.Image.new('RGB', (cell_w * 2, cell_h + HEADER), (30, 30, 30))
    draw   = PIL.ImageDraw.Draw(panel)
    font   = _try_font(LABEL_FONT_SIZE)

    # Header label
    draw.text((6, 6), name, fill=(240, 240, 60), font=font)

    # Resize both to cell_w × cell_h for display
    b_disp = _draw_boxes(before, boxes_before, labels_before).resize((cell_w, cell_h))
    a_disp = _draw_boxes(after,  boxes_after,  labels_after ).resize((cell_w, cell_h))

    panel.paste(b_disp, (0,      HEADER))
    panel.paste(a_disp, (cell_w, HEADER))

    # Divider
    draw.line([(cell_w, HEADER), (cell_w, HEADER + cell_h)], fill=(200, 200, 200), width=2)
    draw.text((4,         HEADER + 4), 'original',  fill=(200, 200, 200), font=_try_font(11))
    draw.text((cell_w+4,  HEADER + 4), 'augmented', fill=(200, 200, 200), font=_try_font(11))
    return panel


def _make_synthetic_image(w: int = 1280, h: int = 720) -> Tuple[PIL.Image.Image, Dict]:
    """
    Generate a synthetic aerial-scene image with fake vehicles (colored rectangles
    on a grass-like background) so the script works without real data.
    """
    rng = np.random.default_rng(42)

    # Background: noisy green/brown gradient simulating ground
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for c, (base, var) in enumerate([(60, 40), (100, 50), (30, 25)]):
        bg[:, :, c] = np.clip(
            rng.integers(base - var, base + var, (h, w)), 0, 255
        ).astype(np.uint8)

    # Road stripes
    for y_start in range(0, h, 120):
        bg[y_start:y_start+8, :, 0] = 120
        bg[y_start:y_start+8, :, 1] = 110
        bg[y_start:y_start+8, :, 2] = 80
    for x_start in range(0, w, 160):
        bg[:, x_start:x_start+8, 0] = 120
        bg[:, x_start:x_start+8, 1] = 110
        bg[:, x_start:x_start+8, 2] = 80

    img  = PIL.Image.fromarray(bg)
    draw = PIL.ImageDraw.Draw(img)

    boxes, labels = [], []
    vehicle_colors = [
        ((200, 50, 50), 0),    # red car → ped
        ((50, 50, 200), 1),    # blue car → car
        ((200, 200, 50), 2),   # yellow van → van
        ((80, 200, 80), 3),    # green truck → truck
        ((180, 80, 180), 4),   # purple bus → bus
    ]
    n_vehicles = 25
    for _ in range(n_vehicles):
        vw = rng.integers(18, 55)
        vh = rng.integers(10, 32)
        x1 = rng.integers(0, w - vw)
        y1 = rng.integers(0, h - vh)
        x2, y2 = x1 + vw, y1 + vh
        color_rgb, cls_id = vehicle_colors[rng.integers(0, len(vehicle_colors))]
        draw.rectangle([x1, y1, x2, y2], fill=color_rgb, outline=(255, 255, 255))
        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id)

    target = {
        'boxes':  torch.tensor(boxes,  dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'size':   torch.tensor([h, w]),
    }
    return img, target


def _load_real_image(path: str) -> Tuple[PIL.Image.Image, Dict]:
    img = PIL.Image.open(path).convert('RGB')
    w, h = img.size
    # Dummy boxes across the image so we can see box propagation
    boxes = torch.tensor([
        [w * 0.1, h * 0.1, w * 0.25, h * 0.25],
        [w * 0.4, h * 0.3, w * 0.60, h * 0.55],
        [w * 0.7, h * 0.6, w * 0.90, h * 0.85],
        [w * 0.2, h * 0.6, w * 0.40, h * 0.80],
        [w * 0.6, h * 0.1, w * 0.80, h * 0.30],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    target = {'boxes': boxes, 'labels': labels, 'size': torch.tensor([h, w])}
    return img, target


# ── augmentation catalogue ────────────────────────────────────────────────────

def _sample_fn_factory(img, target):
    """Return a sample_fn that always returns the same (img, target) — enough for visualization."""
    def _fn():
        return img, target
    return _fn


def build_aug_list(sample_fn) -> List[Tuple[str, object]]:
    """Return [(name, transform)] with all probabilities forced to 1.0."""
    return [
        ('RandomHorizontalFlip',
         RandomHorizontalFlip(p=1.0)),

        ('ScaleBiasedCrop',
         ScaleBiasedCrop(min_scale=0.35, max_scale=0.70, beta_alpha=2.0, beta_beta=5.0, p=1.0)),

        ('RandomColorJitter',
         RandomColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2, p=1.0)),

        ('RandomGrayscale',
         RandomGrayscale(p=1.0)),

        ('RandomNightMode',
         RandomNightMode(brightness_range=(0.08, 0.20), noise_std_range=(15.0, 30.0), p=1.0)),

        ('RandomFog',
         RandomFog(fog_coeff_range=(0.3, 0.5), fog_color=(210, 215, 220), p=1.0)),

        ('RandomMotionBlur',
         RandomMotionBlur(kernel_size_range=(11, 19), p=1.0)),

        ('RandomJPEGCompression',
         RandomJPEGCompression(quality_range=(20, 40), p=1.0)),

        ('RandomSensorNoise',
         RandomSensorNoise(gaussian_std_range=(20.0, 35.0), poisson_scale=0.15, p=1.0)),

        ('RandomSunGlare',
         RandomSunGlare(intensity_range=(0.5, 0.7), p=1.0)),

        ('RandomOcclusionPatch',
         RandomOcclusionPatch(patch_scale_range=(0.08, 0.22), num_patches=3, p=1.0)),

        ('RandomGaussianBlur',
         RandomGaussianBlur(kernel_size=21, sigma=(3.0, 5.0), p=1.0)),

        ('RandomResize (multi-scale)',
         RandomResize([320, 480, 640], max_size=1333)),

        ('CopyPaste',
         CopyPaste(sample_fn=sample_fn, max_objects=20, paste_prob=1.0, p=1.0, min_area=4.0)),

        ('Mosaic',
         Mosaic(sample_fn=sample_fn, p=1.0)),
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize augmentations from transforms.py')
    parser.add_argument('--image', default=None,
                        help='Path to a real image. Omit to use synthetic scene.')
    parser.add_argument('--out', default='aug_visualization.jpg',
                        help='Output JPEG path (default: aug_visualization.jpg)')
    parser.add_argument('--cell-w', type=int, default=480,
                        help='Width of each image cell in the grid (default: 480)')
    parser.add_argument('--cell-h', type=int, default=320,
                        help='Height of each image cell in the grid (default: 320)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('[aug_visualize] loading image …')
    if args.image:
        img, target = _load_real_image(args.image)
    else:
        print('[aug_visualize] no --image given — using synthetic aerial scene')
        img, target = _make_synthetic_image()

    sample_fn = _sample_fn_factory(img, target)
    aug_list  = build_aug_list(sample_fn)

    panels = []
    for name, transform in aug_list:
        print(f'  applying {name} …')
        try:
            aug_img, aug_tgt = transform(img.copy(), {k: v.clone() if isinstance(v, torch.Tensor) else v
                                                       for k, v in target.items()})
        except Exception as exc:
            print(f'  [WARN] {name} failed: {exc}')
            aug_img, aug_tgt = img.copy(), target

        if not isinstance(aug_img, PIL.Image.Image):
            print(f'  [WARN] {name} returned non-PIL output — skipping')
            continue

        panel = _make_panel(
            name,
            before=img,
            after=aug_img,
            boxes_before=target.get('boxes', torch.zeros((0, 4))),
            labels_before=target.get('labels', torch.zeros(0, dtype=torch.long)),
            boxes_after=aug_tgt.get('boxes', torch.zeros((0, 4))),
            labels_after=aug_tgt.get('labels', torch.zeros(0, dtype=torch.long)),
            cell_w=args.cell_w,
            cell_h=args.cell_h,
        )
        panels.append(panel)

    # Stack all panels vertically
    panel_w = panels[0].width
    panel_h = panels[0].height
    total_h = panel_h * len(panels)

    grid = PIL.Image.new('RGB', (panel_w, total_h), (15, 15, 15))
    for i, panel in enumerate(panels):
        grid.paste(panel, (0, i * panel_h))

    grid.save(args.out, quality=92)
    print(f'\n[aug_visualize] saved → {os.path.abspath(args.out)}')
    print(f'  grid size: {grid.width} × {grid.height} px,  {len(panels)} augmentations')


if __name__ == '__main__':
    main()
