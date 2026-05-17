"""
MultiScaleNeck: aggregates all ViT output features → DETR memory + finest feature.

Changes vs. original:
  - proj0 replaced by LWDetrProjector (C2f block matching backbone.0.projector.*)
  - Takes ALL vit_features (concatenated) instead of only the last one
  - forward() returns (MultiScaleNeckOutput, finest_feat) in one pass (no double projection)
  - CenterNetUpsampleNeck upsamples finest stride-16 → stride-4 for small-object detection

Level layout for DETR memory:
  Level 0 (finest): projector(concat vit_features)  → H/16 × W/16
  Level k (extra) : stride-2 conv(level k-1)         → H/(16·2^k)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import NeckConfig


# ── Sinusoidal 2-D positional encoding ───────────────────────────────────────

def _build_2d_sincos_pos(H: int, W: int, dim: int, device: torch.device) -> Tensor:
    assert dim % 4 == 0
    half = dim // 2
    y = torch.arange(H, dtype=torch.float32, device=device)
    x = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    freq = 10000 ** (2 * torch.arange(half // 2, dtype=torch.float32, device=device) / half)
    pos_x = grid_x.reshape(-1, 1) / freq
    pos_y = grid_y.reshape(-1, 1) / freq
    enc = torch.cat([pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()], dim=-1)
    return enc.unsqueeze(0)


# ── LW-DETR Projector (matches backbone.0.projector.* checkpoint keys) ───────

class _ConvBN(nn.Module):
    """Conv2d + BatchNorm2d (no bias in conv), matching cv*.conv / cv*.bn keys."""
    def __init__(self, c_in: int, c_out: int, k: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class _C2fBottleneck(nn.Module):
    """Bottleneck block: cv1(3×3) → cv2(3×3) + residual (matches m.N.cv1/cv2 keys)."""
    def __init__(self, c: int) -> None:
        super().__init__()
        self.cv1 = _ConvBN(c, c, 3)
        self.cv2 = _ConvBN(c, c, 3)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x))


class _C2fBlock(nn.Module):
    """
    C2f feature aggregation (ultralytics style) matching backbone.0.projector.stages.0.0.

    cv1: Conv1×1(c_in → c_out) — splits output into two halves
    m.N: n bottleneck blocks applied sequentially on second half
    cv2: Conv1×1((2+n)×c_out//2 → c_out) — fuses all branches
    """
    def __init__(self, c_in: int, c_out: int, n: int = 3) -> None:
        super().__init__()
        c_ = c_out // 2
        self.cv1 = _ConvBN(c_in,           c_out,         1)
        self.cv2 = _ConvBN((2 + n) * c_,   c_out,         1)
        self.m   = nn.ModuleList([_C2fBottleneck(c_) for _ in range(n)])

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, 1))   # [c_, c_]
        y.extend(m(y[-1]) for m in self.m)  # append each bottleneck output
        return self.cv2(torch.cat(y, 1))


class LWDetrProjector(nn.Module):
    """
    Feature projector matching backbone.0.projector.* in LW-DETR checkpoints.

    Structure:
        stages.0.0 : _C2fBlock(c_in, 256, n=3)
        stages.0.1 : nn.LayerNorm(256)

    Input : concatenated ViT stage features (B, n_feats × embed_dim, H/16, W/16)
    Output: (B, 256, H/16, W/16)
    """
    def __init__(self, c_in: int, c_out: int = 256) -> None:
        super().__init__()
        self.stages = nn.ModuleList([
            nn.ModuleList([
                _C2fBlock(c_in, c_out, n=3),
                nn.LayerNorm(c_out),
            ])
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.stages[0][0](x)                                   # C2f
        x = self.stages[0][1](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        return x


# ── Stride-4 upsample neck for CenterNet head ─────────────────────────────────

class CenterNetUpsampleNeck(nn.Module):
    """
    4× upsample from ViT stride-16 to stride-4 for the CenterNet head.

    Small objects (20-50px) are only ~1-3px at stride 16, making CenterNet
    heatmap learning near-impossible.  Upsampling to stride 4 gives 5-12px,
    providing enough spatial resolution for Gaussian peak supervision.

    Two 2× ConvTranspose2d steps keep channels at hidden_dim throughout so the
    DETR decoder memory (stride-16) stays completely unchanged.
    """
    def __init__(self, ch: int) -> None:
        super().__init__()
        # kernel_size=2, padding=0 gives identical 2× upsample as k=4,p=1
        # but 4× fewer MACs per ConvTranspose step — critical at stride-4 resolution
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2, padding=0, bias=False),
            nn.GroupNorm(32, ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2, padding=0, bias=False),
            nn.GroupNorm(32, ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """(B, ch, H/16, W/16) → (B, ch, H/4, W/4)"""
        return self.up(x)


# ── MultiScaleNeckOutput ──────────────────────────────────────────────────────

class MultiScaleNeckOutput:
    __slots__ = ('memory', 'spatial_shapes', 'level_start_idx', 'pos_embed', 'valid_ratios')

    def __init__(self, memory, spatial_shapes, level_start_idx, pos_embed, valid_ratios):
        self.memory          = memory
        self.spatial_shapes  = spatial_shapes
        self.level_start_idx = level_start_idx
        self.pos_embed       = pos_embed
        self.valid_ratios    = valid_ratios


# ── MultiScaleNeck ────────────────────────────────────────────────────────────

class MultiScaleNeck(nn.Module):
    """
    Aggregates ALL ViT output features via LWDetrProjector → DETR memory + finest feature.

    Changes from v1:
      - Takes concatenated vit_features (all stages) instead of only the last one.
      - proj0 = LWDetrProjector matching backbone.0.projector.* checkpoint keys.
      - forward() returns (MultiScaleNeckOutput, finest_feat_s16) in a single pass.

    Args:
        in_ch : n_vit_features × embed_dim  (e.g. 3×192=576 for tiny, 4×192=768 for small)
        cfg   : NeckConfig
    """

    def __init__(self, in_ch: int, cfg: NeckConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        self.hidden_dim        = D
        self.num_output_levels = cfg.num_output_levels

        self.proj0 = LWDetrProjector(in_ch, D)

        self.extra_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(D, D, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, D),
            )
            for _ in range(cfg.num_output_levels - 1)
        ])

    def _flatten_and_embed(
        self, feats: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        parts, pos_parts, shapes = [], [], []
        for feat in feats:
            B, D, H, W = feat.shape
            flat = feat.flatten(2).permute(0, 2, 1)
            pos  = _build_2d_sincos_pos(H, W, D, feat.device)
            parts.append(flat)
            pos_parts.append(pos.expand(B, -1, -1))
            shapes.append((H, W))
        memory  = torch.cat(parts,     dim=1)
        pos_emb = torch.cat(pos_parts, dim=1)
        spatial_shapes  = torch.as_tensor(shapes, dtype=torch.long, device=memory.device)
        level_start_idx = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ])
        return memory, spatial_shapes, level_start_idx, pos_emb

    def _valid_ratios(self, B: int, L: int, device: torch.device) -> Tensor:
        return torch.ones(B, L, 2, dtype=torch.float32, device=device)

    def forward(
        self, vit_features: List[Tensor]
    ) -> Tuple[MultiScaleNeckOutput, Tensor]:
        """
        Args:
            vit_features : list of (B, embed_dim, H/16, W/16) from all ViT output blocks
        Returns:
            (MultiScaleNeckOutput, finest_s16)
            finest_s16 : (B, hidden_dim, H/16, W/16) — projected stride-16 feature
        """
        # Concatenate all ViT stage features along the channel axis
        cat_feat = torch.cat(vit_features, dim=1)   # (B, n×embed_dim, H/16, W/16)
        finest   = self.proj0(cat_feat)             # (B, D, H/16, W/16)

        levels = [finest]
        for extra in self.extra_projs:
            levels.append(extra(levels[-1]))

        memory, spatial_shapes, level_start_idx, pos_embed = self._flatten_and_embed(levels)
        valid_ratios = self._valid_ratios(finest.shape[0], len(levels), finest.device)

        neck_out = MultiScaleNeckOutput(
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_idx=level_start_idx,
            pos_embed=pos_embed,
            valid_ratios=valid_ratios,
        )
        return neck_out, finest
