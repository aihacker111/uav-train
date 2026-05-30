"""
Image/video loaders for tracking inference.

LoadImages — iterate over a directory or file list, one frame at a time.
LoadVideo  — iterate over a video file frame by frame.
letterbox  — pad-and-resize while preserving aspect ratio, snap to multiples of 64.
"""
import glob
import os

import cv2
import numpy as np
import torch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _normalize(img, use_imagenet=True):
    img = img / 255.0
    if use_imagenet:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img


def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    """Pad-and-resize preserving aspect ratio; output dims are multiples of 64."""
    h0, w0 = img.shape[:2]
    ratio = min(float(height) / h0, float(width) / w0)
    new_w = round(w0 * ratio)
    new_h = round(h0 * ratio)
    dw = (width  - new_w) * 0.5
    dh = (height - new_h) * 0.5
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    # Snap to multiples of 64 for ViT patch-embed compatibility
    h_out, w_out = img.shape[:2]
    h64 = ((h_out + 63) // 64) * 64
    w64 = ((w_out + 63) // 64) * 64
    if h64 != h_out or w64 != w_out:
        img = cv2.copyMakeBorder(img, 0, h64 - h_out, 0, w64 - w_out,
                                 cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


class LoadImages:
    """Iterate over images in a directory or a list of paths."""

    def __init__(self, path, img_size=(1088, 608), use_imagenet_norm=True):
        self.use_imagenet_norm = use_imagenet_norm
        self.frame_rate = 10
        self.width  = img_size[0]
        self.height = img_size[1]

        if isinstance(path, str):
            if os.path.isdir(path):
                exts = {'.jpg', '.jpeg', '.png', '.tif'}
                self.files = sorted(
                    f for f in glob.glob(f'{path}/*.*')
                    if os.path.splitext(f)[1].lower() in exts
                )
            elif os.path.isfile(path):
                self.files = [path]
            else:
                raise FileNotFoundError(f'[LoadImages] not found: {path}')
        elif isinstance(path, list):
            self.files = path
        else:
            self.files = []

        self.nF    = len(self.files)
        self.count = 0
        assert self.nF > 0, f'No images found in {path}'

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration

        img_path = self.files[self.count]
        img0 = cv2.imread(img_path)
        assert img0 is not None, f'Failed to load {img_path}'

        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = _normalize(img, self.use_imagenet_norm)
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]
        img0 = cv2.imread(img_path)
        assert img0 is not None, f'Failed to load {img_path}'
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = _normalize(img, self.use_imagenet_norm)
        return img_path, img, img0

    def __len__(self):
        return self.nF


class LoadVideo:
    """Iterate over frames of a video file."""

    def __init__(self, path, img_size=(1088, 608), use_imagenet_norm=True):
        self.use_imagenet_norm = use_imagenet_norm
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = img_size[0]
        self.height = img_size[1]
        self.count  = 0
        print(f'Video: {self.vn} frames @ {self.frame_rate} fps')

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.vn:
            raise StopIteration

        ret, img0 = self.cap.read()
        assert img0 is not None, f'Failed to load frame {self.count}'

        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = _normalize(img, self.use_imagenet_norm)
        return self.count, img, img0

    def __len__(self):
        return self.vn
