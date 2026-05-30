"""
VisDroneDataset — DEIMv2-registered dataset for VisDrone MOT data.

Label format on disk (labels_with_ids/*.txt):
    cls_id  track_id  cx  cy  w  h   (all normalised [0,1])

What __getitem__ returns (DEIMv2-compatible):
    img     : PIL.Image RGB
    target  : {
        boxes    : BoundingBoxes(N,4) pixel XYXY  ← tv_tensor so transforms handle it
        labels   : (N,2) int64  col-0=cls_id, col-1=global_track_id
                   ↑ 2-column so SanitizeBoundingBoxes/RandomIoUCrop filter
                     both cls and track-id together, keeping them in sync
        image_id : (1,) int64
        orig_size: (2,) int64  (H, W)
    }

After ConvertBoxes(fmt='cxcywh', normalize=True) at the end of the transform
pipeline, boxes become normalised cxcywh [0,1].

In DEIMMotCriterion.forward() the 2-column labels are split:
    cls_ids   = labels[:, 0]   → passed to DEIMCriterion (expects 1D)
    track_ids = labels[:, 1]   → used for ReID loss
"""
from __future__ import annotations

import os
import os.path as osp
import warnings
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from .._misc import convert_to_tv_tensor
from ...core import register


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_image_list(txt_path: str, root: str) -> List[str]:
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return [osp.join(root, p) if not osp.isabs(p) else p for p in lines]


def _label_path(img_path: str) -> str:
    return (img_path
            .replace('images', 'labels_with_ids')
            .replace('.png', '.txt')
            .replace('.jpg', '.txt'))


def _load_labels(lbl_path: str) -> np.ndarray:
    """Return (N, 6) float32: cls_id, track_id, cx, cy, w, h (norm)."""
    if not osp.isfile(lbl_path) or os.path.getsize(lbl_path) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lb = np.loadtxt(lbl_path, dtype=np.float32)
    return lb.reshape(-1, 6)


# ── dataset ───────────────────────────────────────────────────────────────────

@register()
class VisDroneDataset(Dataset):
    """
    DEIMv2-compatible VisDrone MOT dataset.

    data_cfg : path to JSON with keys 'root', 'train', 'val'
               e.g. src/lib/cfg/visdrone.json
    split    : 'train' | 'val'
    transforms: DEIMv2 transform pipeline injected by YAMLConfig
    num_classes: 10 for VisDrone
    with_reid  : if True, compute global track-id offsets (training only)
    """
    __inject__ = ['transforms']

    def __init__(
        self,
        data_cfg: str,
        split: str = 'train',
        transforms=None,
        num_classes: int = 10,
        with_reid: bool = True,
    ) -> None:
        super().__init__()
        cfg   = json.load(open(data_cfg))
        root  = cfg['root']
        paths = cfg.get(split, {})    # {dataset_name: image_list_txt}; empty if split not in JSON

        self.num_classes = num_classes
        self.transforms  = transforms
        self.with_reid   = with_reid
        self.nID_dict: Dict[int, int] = {}

        self._img_files: List[str] = []
        self._lbl_files: List[str] = []
        self._ds_names:  List[str] = []

        if not paths:
            print(f'[VisDroneDataset] WARNING: split={split!r} not in {data_cfg} — empty dataset')
            self._len = 0
            return

        for ds, txt_path in paths.items():
            imgs = _load_image_list(txt_path, root)
            self._img_files.extend(imgs)
            self._lbl_files.extend(_label_path(p) for p in imgs)
            self._ds_names.extend([ds] * len(imgs))

        self._len = len(self._img_files)

        if with_reid:
            self._build_offsets(paths, root)
            print(f'[VisDroneDataset] split={split}  images={self._len}')
            for cls_id, n in self.nID_dict.items():
                print(f'  cls {cls_id}: {n} identities')
        else:
            self.nID_dict: Dict[int, int] = {}
            print(f'[VisDroneDataset] split={split}  images={self._len}  (no ReID)')

    # ── build global track-id offsets ─────────────────────────────────────────

    def _build_offsets(self, paths: dict, root: str) -> None:
        """Per-(dataset, cls_id) cumulative track-id start index."""
        tid_max: Dict[str, Dict[int, int]] = {}
        for ds, txt_path in paths.items():
            imgs    = _load_image_list(txt_path, root)
            max_ids: Dict[int, int] = defaultdict(int)
            for img in imgs:
                lb = _load_labels(_label_path(img))
                for row in lb:
                    c, t = int(row[0]), int(row[1])
                    if 0 <= c < self.num_classes and t > max_ids[c]:
                        max_ids[c] = t
            tid_max[ds] = max_ids

        start: Dict[str, Dict[int, int]] = {}
        last:  Dict[int, int] = defaultdict(int)
        for ds, max_ids in tid_max.items():
            start[ds] = {}
            for c, n in max_ids.items():
                start[ds][c] = last[c]
                last[c] += n

        self._tid_start = start
        self.nID_dict   = {k: int(v) for k, v in last.items()}

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self._img_files[idx]
        lbl_path = self._lbl_files[idx]
        ds       = self._ds_names[idx]

        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        raw = _load_labels(lbl_path)   # (N, 6): cls_id, track_id, cx, cy, w, h  [0,1]

        if len(raw) > 0:
            # ── Convert normalised cxcywh → pixel xyxy for tv_tensor ──────────
            cx, cy, bw, bh = raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5]
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            boxes_np = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            # ── Wrap as BoundingBoxes tv_tensor ──────────────────────────────
            # SanitizeBoundingBoxes, RandomIoUCrop, etc. need this to know the
            # box format and canvas size so they can correctly filter/transform.
            boxes = convert_to_tv_tensor(
                torch.from_numpy(boxes_np),
                key='boxes',
                box_format='xyxy',
                spatial_size=(H, W),
            )

            # ── Compute global track-id offsets ───────────────────────────────
            cls_np = raw[:, 0].astype(np.int64)
            tid_np = raw[:, 1].astype(np.int64)

            if self.with_reid and hasattr(self, '_tid_start'):
                offs = self._tid_start.get(ds, {})
                tid_global = np.where(
                    tid_np > 0,
                    tid_np - 1 + np.array([offs.get(int(c), 0) for c in cls_np]),
                    -1,
                ).astype(np.int64)
            else:
                tid_global = np.full(len(raw), -1, dtype=np.int64)

            # ── 2-column labels: [cls_id | track_id] ─────────────────────────
            # Using a single 2-column tensor guarantees that when transforms
            # filter boxes (e.g. SanitizeBoundingBoxes applies `labels[mask]`),
            # BOTH the class id and the track id are filtered together — no desync.
            labels = torch.stack([
                torch.from_numpy(cls_np),
                torch.from_numpy(tid_global),
            ], dim=1)   # (N, 2) int64

        else:
            boxes  = convert_to_tv_tensor(
                torch.zeros((0, 4), dtype=torch.float32),
                key='boxes', box_format='xyxy', spatial_size=(H, W),
            )
            labels = torch.zeros((0, 2), dtype=torch.int64)

        target = {
            'boxes':     boxes,
            'labels':    labels,                              # (N, 2)  col-0=cls  col-1=track_id
            'image_id':  torch.tensor([idx], dtype=torch.int64),
            'orig_size': torch.tensor([H, W], dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)

        return img, target

    # ── DEIMv2 epoch tracking (required by DataLoader.set_epoch + Compose policy) ─

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return getattr(self, '_epoch', -1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._len

    @property
    def categories(self):
        return [{'id': i, 'name': f'cls{i}'} for i in range(self.num_classes)]
