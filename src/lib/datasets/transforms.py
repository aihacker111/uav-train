"""
Transforms and data augmentation for image + bbox (PIL-based).
Ported from RF-DETR/LW-DETR transforms — rfdetr dependencies removed.

Added augmentations:
  ScaleBiasedCrop — zoom-in biased random crop for small-object detection
  CopyPaste       — paste objects from a donor image to increase object density
  Mosaic          — 4-image 2×2 grid (needs dataset reference via sample_fn)
"""
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
from numbers import Number

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as TF


def _box_xyxy_to_cxcywh(boxes):
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return TF.interpolate(input, size=size, scale_factor=scale_factor,
                          mode=mode, align_corners=align_corners)


# ── primitive ops ────────────────────────────────────────────────────────────

def crop(image: PIL.Image.Image, target: Dict, region: Tuple) -> Tuple:
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    target["size"] = torch.tensor([h, w])
    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)
        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target


def hflip(image: PIL.Image.Image, target: Dict) -> Tuple:
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    return flipped_image, target


def resize(image: PIL.Image.Image, target: Optional[Dict], size, max_size=None) -> Tuple:
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_orig = float(min(w, h))
            max_orig = float(max(w, h))
            if max_orig / min_orig * size > max_size:
                size = int(round(max_size * min_orig / max_orig))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            return (int(size * h / w), size)
        return (size, int(size * w / h))

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"]
        target["boxes"] = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

    if "area" in target:
        target["area"] = target["area"] * (ratio_width * ratio_height)

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image: PIL.Image.Image, target: Optional[Dict], padding: Tuple) -> Tuple:
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


# ── transform classes ────────────────────────────────────────────────────────

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class SquareResize:
    def __init__(self, sizes: List[int]):
        self.sizes = sizes

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        rescaled_img = F.resize(img, (size, size))
        w, h = rescaled_img.size
        if target is None:
            return rescaled_img, None
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_img.size, img.size))
        ratio_width, ratio_height = ratios
        target = target.copy()
        if "boxes" in target:
            target["boxes"] = target["boxes"] * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        if "area" in target:
            target["area"] = target["area"] * (ratio_width * ratio_height)
        target["size"] = torch.tensor([h, w])
        if "masks" in target:
            target['masks'] = interpolate(
                target['masks'][:, None].float(), (h, w), mode="nearest")[:, 0] > 0.5
        return rescaled_img, target


class RandomPad:
    def __init__(self, max_pad: int):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class Pad:
    def __init__(self, size=None, size_divisor=32, pad_mode=0, offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        if not isinstance(size, (int, Sequence)):
            raise TypeError("size must be int or Sequence, got {}".format(type(size)))
        if isinstance(size, int):
            size = [size, size]
        assert pad_mode in [-1, 0, 1, 2]
        if pad_mode == -1:
            assert offsets, 'offsets required when pad_mode=-1'
        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32) * np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def __call__(self, im, target):
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert im_h <= h and im_w <= w
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            return im.astype(np.float32), target

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]
        im = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            target["size"] = torch.tensor([h, w])
            return im, target
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = np.asarray(target["boxes"])
            target["boxes"] = torch.from_numpy(self.apply_bbox(boxes, offsets))
            target["size"] = torch.tensor([h, w])
        return im, target


class RandomExpand:
    """Randomly expand the canvas (simulates drone altitude variation)."""
    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5, 127.5, 127.5)):
        assert ratio > 1.01
        self.ratio = ratio
        self.prob = prob
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        self.fill_value = tuple(fill_value)

    def __call__(self, img, target):
        if np.random.uniform(0., 1.) < self.prob:
            return img, target
        height, width = img.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not (h > height and w > width):
            return img, target
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        pad_op = Pad(size=[h, w], pad_mode=-1, offsets=[x, y], fill_value=self.fill_value)
        return pad_op(img, target)


class PILtoNdArray:
    def __call__(self, img, target):
        return np.asarray(img), target


class NdArraytoPIL:
    def __call__(self, img, target):
        return F.to_pil_image(img.astype('uint8')), target


class RandomSelect:
    """Randomly choose between two transform pipelines with probability p."""
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class RandomColorJitter:
    """Color jitter for appearance variation (illumination, time-of-day)."""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.8):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img: PIL.Image.Image, target):
        if random.random() < self.p:
            img = self.jitter(img)
        return img, target


class RandomGaussianBlur:
    """Gaussian blur to simulate motion blur and low-resolution sensors."""
    def __init__(self, kernel_size: int = 11, sigma=(0.1, 2.0), p: float = 0.3):
        self.blur = T.GaussianBlur(kernel_size, sigma)
        self.p = p

    def __call__(self, img: PIL.Image.Image, target):
        if random.random() < self.p:
            img = self.blur(img)
        return img, target


class RandomGrayscale:
    """Randomly convert to grayscale to simulate night/IR imagery."""
    def __init__(self, p: float = 0.15):
        self.p = p

    def __call__(self, img: PIL.Image.Image, target):
        if random.random() < self.p:
            img = F.to_grayscale(img, num_output_channels=3)
        return img, target


class ScaleBiasedCrop:
    """
    Random crop biased toward smaller crop scales (zoom-in) to make small
    objects appear larger in the training crop.

    Uses a Beta(α, β) distribution for the scale:  α < β  biases toward
    smaller values (more zoom-in).  Default Beta(2, 5) concentrates mass
    around scale ≈ 0.28, which converts a 10-px VisDrone object to ~36px —
    within the range CenterNet can detect reliably.

    After cropping the image is NOT resized — the letterbox step in jde.py
    will pad it to the final training resolution.  Boxes that fall outside
    the crop are discarded (same as RandomSizeCrop).
    """
    def __init__(
        self,
        min_scale:  float = 0.25,
        max_scale:  float = 0.85,
        beta_alpha: float = 2.0,
        beta_beta:  float = 5.0,
        p:          float = 0.5,
    ) -> None:
        self.min_scale  = min_scale
        self.max_scale  = max_scale
        self.beta_alpha = beta_alpha
        self.beta_beta  = beta_beta
        self.p          = p

    def __call__(self, img: PIL.Image.Image, target):
        if random.random() >= self.p:
            return img, target

        w, h = img.size
        scale = float(np.random.beta(self.beta_alpha, self.beta_beta))
        scale = self.min_scale + scale * (self.max_scale - self.min_scale)

        crop_w = max(32, int(w * scale))
        crop_h = max(32, int(h * scale))
        region = T.RandomCrop.get_params(img, [crop_h, crop_w])
        return crop(img, target, region)


class CopyPaste:
    """
    Copy-Paste augmentation (Ghiasi et al. 2021) for dense small-object datasets.

    Randomly pastes objects from a donor (image, target) pair onto the current
    image.  Requires a `sample_fn` callable that returns a random (PIL.Image,
    target_dict) pair from the dataset — typically `dataset[random_idx]`.

    Only objects whose pasted bounding box has area ≥ min_area (in the
    destination image) are kept.  Pasted boxes are appended to the existing
    target boxes/labels/ids, giving the model more objects to learn from per
    sample without changing the scene geometry.

    NOTE: sample_fn is called lazily inside __call__, so this transform can
    be constructed before the dataset is fully loaded.  Pass it a lambda:
        CopyPaste(sample_fn=lambda: dataset[random.randint(0, len(dataset)-1)])
    """

    def __init__(
        self,
        sample_fn:      Callable,         # () → (PIL.Image, target_dict)
        max_objects:    int   = 20,        # max number of objects to paste
        paste_prob:     float = 0.5,       # probability of pasting each object
        p:              float = 0.5,       # probability of applying at all
        min_area:       float = 4.0,       # discard pasted boxes smaller than this (px²)
        alpha:          float = 1.0,       # blend alpha (1.0 = hard paste)
    ) -> None:
        self.sample_fn   = sample_fn
        self.max_objects = max_objects
        self.paste_prob  = paste_prob
        self.p           = p
        self.min_area    = min_area
        self.alpha       = alpha

    def __call__(
        self,
        img:    PIL.Image.Image,
        target: Dict,
    ) -> Tuple[PIL.Image.Image, Dict]:
        if random.random() >= self.p:
            return img, target

        donor_img, donor_tgt = self.sample_fn()

        if 'boxes' not in donor_tgt or len(donor_tgt['boxes']) == 0:
            return img, target

        dst_w, dst_h = img.size
        src_w, src_h = donor_img.size

        # Resize donor to destination size so boxes are in the same coord space
        if (src_w, src_h) != (dst_w, dst_h):
            donor_img, donor_tgt = resize(donor_img, donor_tgt, (dst_h, dst_w))

        donor_boxes  = donor_tgt['boxes']    # (N, 4) xyxy
        donor_labels = donor_tgt['labels']

        # Sample objects to paste
        n  = min(len(donor_boxes), self.max_objects)
        idx = [i for i in range(len(donor_boxes)) if random.random() < self.paste_prob]
        idx = idx[:n]
        if not idx:
            return img, target

        dst_img = img.copy()
        paste_boxes, paste_labels = [], []

        for i in idx:
            x1, y1, x2, y2 = donor_boxes[i].int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(dst_w, x2), min(dst_h, y2)
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            # Crop the object patch from donor
            patch = donor_img.crop((x1, y1, x2, y2))

            # Random paste location (keep fully inside destination)
            pw, ph = patch.size
            max_px = max(0, dst_w - pw)
            max_py = max(0, dst_h - ph)
            px = random.randint(0, max_px)
            py = random.randint(0, max_py)

            area = pw * ph
            if area < self.min_area:
                continue

            dst_img.paste(patch, (px, py))

            paste_boxes.append(torch.tensor([px, py, px + pw, py + ph], dtype=torch.float32))
            paste_labels.append(donor_labels[i])

        if not paste_boxes:
            return img, target

        target = target.copy()
        new_boxes  = torch.stack(paste_boxes)                # (M, 4)
        new_labels = torch.stack(paste_labels)               # (M,)

        if 'boxes' in target and len(target['boxes']) > 0:
            target['boxes']  = torch.cat([target['boxes'],  new_boxes],  dim=0)
            target['labels'] = torch.cat([target['labels'], new_labels], dim=0)
            if 'ids' in target:
                # Pasted objects get id=-1 (no valid track id)
                target['ids'] = torch.cat([
                    target['ids'],
                    torch.full((len(new_boxes),), -1, dtype=target['ids'].dtype),
                ])
        else:
            target['boxes']  = new_boxes
            target['labels'] = new_labels

        return dst_img, target


class Mosaic:
    """
    4-image 2×2 Mosaic augmentation (YOLOv5 style) for dense small-object training.

    Combines 4 images into a single 2×2 grid at the target resolution.
    Each quadrant is filled with a randomly cropped region of one image.
    All bounding boxes are adjusted to the new coordinate system.

    Like CopyPaste, requires a `sample_fn` callable that returns a random
    (PIL.Image, target_dict) pair from the dataset.

    The output image has the same (H, W) as the input.
    """

    def __init__(
        self,
        sample_fn: Callable,    # () → (PIL.Image, target_dict)
        p:         float = 0.5,
    ) -> None:
        self.sample_fn = sample_fn
        self.p         = p

    def _place(
        self,
        canvas:  PIL.Image.Image,
        img:     PIL.Image.Image,
        target:  Dict,
        x_off:   int,
        y_off:   int,
        cell_w:  int,
        cell_h:  int,
    ) -> Dict:
        """Resize img to cell size, paste onto canvas, return adjusted boxes."""
        src_w, src_h = img.size
        scale_x = cell_w / max(src_w, 1)
        scale_y = cell_h / max(src_h, 1)

        rimg, rtgt = resize(img, target, (cell_h, cell_w))
        canvas.paste(rimg, (x_off, y_off))

        new_boxes = []
        if 'boxes' in rtgt and len(rtgt['boxes']) > 0:
            b = rtgt['boxes'].clone()
            b[:, [0, 2]] += x_off
            b[:, [1, 3]] += y_off
            new_boxes = b
        return rtgt, new_boxes

    def __call__(
        self,
        img:    PIL.Image.Image,
        target: Dict,
    ) -> Tuple[PIL.Image.Image, Dict]:
        if random.random() >= self.p:
            return img, target

        W, H = img.size
        cw, ch = W // 2, H // 2

        canvas = PIL.Image.new('RGB', (W, H))

        samples = [(img, target)] + [self.sample_fn() for _ in range(3)]
        placements = [
            (0,  0),           # top-left
            (cw, 0),           # top-right
            (0,  ch),          # bottom-left
            (cw, ch),          # bottom-right
        ]

        all_boxes, all_labels, all_ids = [], [], []
        has_ids = 'ids' in target

        for (simg, stgt), (ox, oy) in zip(samples, placements):
            rtgt, boxes = self._place(canvas, simg, stgt, ox, oy, cw, ch)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_labels.append(rtgt['labels'])
                if has_ids and 'ids' in rtgt:
                    all_ids.append(rtgt['ids'])
                elif has_ids:
                    all_ids.append(torch.full((len(boxes),), -1, dtype=torch.long))

        new_target = target.copy()
        if all_boxes:
            new_target['boxes']  = torch.cat(all_boxes)
            new_target['labels'] = torch.cat(all_labels)
            if has_ids and all_ids:
                new_target['ids'] = torch.cat(all_ids)
        new_target['size'] = torch.tensor([H, W])

        return canvas, new_target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = _box_xyxy_to_cxcywh(target["boxes"])
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n    {0}".format(t)
        format_string += "\n)"
        return format_string


# ── recommended pipeline for aerial MOT (VisDrone) ──────────────────────────

def build_aerial_mot_transforms(sample_fn=None):
    """
    PIL-based pre-letterbox augmentation for aerial MOT (VisDrone 1920×1080).

    Design constraints:
    - ScaleBiasedCrop: zoom-in biased crop to make small objects larger.
      Applied with p=0.4 before multi-scale resize so objects aren't shrunk
      further.  min_scale=0.35 ensures at least 35% of the image is kept.
    - CopyPaste / Mosaic (optional): require `sample_fn` from the dataset.
      Pass sample_fn=lambda: dataset[random.randint(0, len(dataset)-1)].
    - RandomResize only: simulate different drone altitudes.
    - Horizontal flip only: vertical flip rarely seen in drone footage.
    - ColorJitter / GaussianBlur / Grayscale: lighting / sensor variation.
    - Letterbox (in jde.py) handles the final 608×1088 reshape.
    """
    transforms = [
        RandomHorizontalFlip(p=0.5),
        ScaleBiasedCrop(min_scale=0.35, max_scale=0.85, beta_alpha=2.0, beta_beta=5.0, p=0.4),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.8),
        RandomGaussianBlur(kernel_size=5, sigma=(0.1, 1.5), p=0.3),
        RandomGrayscale(p=0.15),
        RandomResize([576, 608, 640, 672, 704], max_size=1333),
    ]

    if sample_fn is not None:
        # Insert CopyPaste before the final resize (works on PIL images)
        transforms.insert(1, CopyPaste(sample_fn=sample_fn, max_objects=15, p=0.4))

    return Compose(transforms)


def build_mosaic_transforms(sample_fn):
    """
    Mosaic-first pipeline for aggressive small-object augmentation.

    Use this as an alternative to build_aerial_mot_transforms when training
    from scratch or when small-object recall is low.  Mosaic is applied first
    (it needs the raw PIL image before any spatial transforms).
    """
    return Compose([
        Mosaic(sample_fn=sample_fn, p=0.5),
        RandomHorizontalFlip(p=0.5),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.8),
        RandomGaussianBlur(kernel_size=5, sigma=(0.1, 1.5), p=0.3),
        RandomGrayscale(p=0.15),
        RandomResize([576, 608, 640, 672, 704], max_size=1333),
    ])
