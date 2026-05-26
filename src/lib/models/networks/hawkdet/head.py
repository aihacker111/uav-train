"""
THead — TOOD Task-aligned Head with DFL regression for HawkDet.

One instance per feature scale (S4 / S8 / S16 / S32).

Architecture per scale:
  shared_tower : num_convs × (Conv2d-BN-SiLU)
  cls_attn     : 1×1 Conv → sigmoid — multiplicative spatial reweight for cls
  reg_attn     : 1×1 Conv → sigmoid — multiplicative spatial reweight for reg
  cls_pred     : 3×3 Conv → num_classes          (raw logits)
  reg_pred     : 3×3 Conv → 4×(reg_max+1)        (DFL distribution logits)

References:
  TOOD — Task-aligned One-stage Object Detection, Feng et al. ICCV 2021
  GFL  — Generalized Focal Loss, Li et al. NeurIPS 2020
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ── Shared conv block ─────────────────────────────────────────────────────────

def _conv_bn_silu(in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, 1, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


# ── THead ─────────────────────────────────────────────────────────────────────

class THead(nn.Module):
    """TOOD Task-aligned Head with DFL regression.

    Args:
        in_ch       : input feature channels (= encoder hidden_dim).
        num_classes : number of foreground classes.
        reg_max     : DFL bin count − 1 (output has reg_max+1 bins per side).
        num_convs   : depth of shared tower.
        feat_ch     : hidden channels inside the tower.
    """

    def __init__(
        self,
        in_ch:       int,
        num_classes: int,
        reg_max:     int = 16,
        num_convs:   int = 4,
        feat_ch:     int = 256,
        reid_dim:    int = 0,
    ) -> None:
        super().__init__()
        self.reg_max     = reg_max
        self.num_classes = num_classes
        self.reid_dim    = reid_dim

        tower = [_conv_bn_silu(in_ch, feat_ch)]
        for _ in range(num_convs - 1):
            tower.append(_conv_bn_silu(feat_ch, feat_ch))
        self.tower = nn.Sequential(*tower)

        # Task-Aligned Predictors: 1×1 attention gate per task (TOOD §3.2)
        self.cls_attn = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.reg_attn = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)

        self.cls_pred  = nn.Conv2d(feat_ch, num_classes,        3, 1, 1)
        self.reg_pred  = nn.Conv2d(feat_ch, 4 * (reg_max + 1), 3, 1, 1)
        # ReID branch: 1×1 on raw tower features (no task-attn — identity-level semantics)
        self.reid_pred = nn.Conv2d(feat_ch, reid_dim, 1) if reid_dim > 0 else None

        self._init_weights()

    def _init_weights(self) -> None:
        # Prior: ~1% foreground → logit ≈ log(0.01/0.99) ≈ −4.6
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.zeros_(self.reg_pred.bias)

    def forward(self, x: Tensor):
        """
        Args:
            x: (B, in_ch, H, W)
        Returns:
            cls  : (B, num_classes,    H, W)  raw logits
            reg  : (B, 4*(reg_max+1), H, W)  DFL distribution logits
            reid : (B, reid_dim,       H, W)  unnormalised embeddings, or None
        """
        feat     = self.tower(x)
        cls_feat = feat * self.cls_attn(feat).sigmoid()
        reg_feat = feat * self.reg_attn(feat).sigmoid()
        reid = self.reid_pred(feat) if self.reid_pred is not None else None
        return self.cls_pred(cls_feat), self.reg_pred(reg_feat), reid


# ── Anchor / decode utilities ─────────────────────────────────────────────────

def make_anchors(
    feat_h: int,
    feat_w: int,
    stride: int,
    device: torch.device,
) -> Tensor:
    """Generate anchor points (grid cell centres) for one feature level.

    Returns:
        (H*W, 2) — (x, y) in input-image pixel space.
    """
    sy = (torch.arange(feat_h, device=device).float() + 0.5) * stride
    sx = (torch.arange(feat_w, device=device).float() + 0.5) * stride
    gy, gx = torch.meshgrid(sy, sx, indexing='ij')
    return torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (H*W, 2)


def dfl_decode(reg_pred: Tensor, reg_max: int) -> Tensor:
    """Decode DFL logits → expected (l, t, r, b) distances.

    Applies softmax over the reg_max+1 bins and takes the expectation.

    Args:
        reg_pred: (..., 4*(reg_max+1))
    Returns:
        (..., 4)
    """
    *leading, last = reg_pred.shape
    dist = reg_pred.reshape(*leading, 4, reg_max + 1).softmax(dim=-1)
    proj = torch.arange(reg_max + 1, dtype=dist.dtype, device=dist.device)
    return (dist * proj).sum(dim=-1)  # (..., 4)


def ltrb_to_cxcywh(
    distance: Tensor,
    anchors:  Tensor,
    stride:   int | Tensor,
    input_hw: tuple,
) -> Tensor:
    """Convert decoded ltrb bin distances → normalised cxcywh boxes.

    Args:
        distance : (..., N, 4) — distances in bin space (before × stride).
        anchors  : (N, 2)      — pixel-space anchor points (x, y).
        stride   : scalar or (N,) tensor — multiply bins → pixels.
        input_hw : (H, W) of input image.
    Returns:
        (..., N, 4) cxcywh in [0, 1].
    """
    H, W   = input_hw
    dist_px = distance * stride  # convert bins → pixels
    ax, ay  = anchors[..., 0], anchors[..., 1]

    x1 = (ax - dist_px[..., 0]) / W
    y1 = (ay - dist_px[..., 1]) / H
    x2 = (ax + dist_px[..., 2]) / W
    y2 = (ay + dist_px[..., 3]) / H

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = (x2 - x1).clamp(min=0)
    h  = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1).clamp(0, 1)
