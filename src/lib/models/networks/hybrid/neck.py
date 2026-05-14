"""
MultiScaleNeck: projects ViT features to the flattened multi-scale memory
required by MSDeformAttn cross-attention.

Level layout:
  Level 0 (finest): proj(vit_features[-1])              → H/16 × W/16
  Level 1 (extra) : stride-2 conv(level 0)              → H/32 × W/32
  Level k (extra) : stride-2 conv(level k-1)            → H/(16·2^k) × W/(16·2^k)

Default is num_output_levels=1, which uses only the H/16 scale to match the
pretrained LW-DETR checkpoints (trained with --projector_scale P4).
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
    """Return (1, H*W, dim) sinusoidal positional encoding."""
    assert dim % 4 == 0, "hidden_dim must be divisible by 4 for 2-D sincos"
    half = dim // 2

    y = torch.arange(H, dtype=torch.float32, device=device)
    x = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')   # (H, W)

    freq = torch.arange(half // 2, dtype=torch.float32, device=device)
    freq = 10000 ** (2 * freq / half)                       # (half//2,)

    pos_x = grid_x.reshape(-1, 1) / freq                   # (H*W, half//2)
    pos_y = grid_y.reshape(-1, 1) / freq

    enc = torch.cat([
        pos_x.sin(), pos_x.cos(),
        pos_y.sin(), pos_y.cos(),
    ], dim=-1)                                              # (H*W, dim)
    return enc.unsqueeze(0)                                 # (1, H*W, dim)


# ── MultiScaleNeck ────────────────────────────────────────────────────────────

class MultiScaleNeckOutput:
    """Plain struct returned by MultiScaleNeck.forward."""
    __slots__ = ('memory', 'spatial_shapes', 'level_start_idx', 'pos_embed', 'valid_ratios')

    def __init__(
        self,
        memory: Tensor,
        spatial_shapes: Tensor,
        level_start_idx: Tensor,
        pos_embed: Tensor,
        valid_ratios: Tensor,
    ) -> None:
        self.memory          = memory
        self.spatial_shapes  = spatial_shapes
        self.level_start_idx = level_start_idx
        self.pos_embed       = pos_embed
        self.valid_ratios    = valid_ratios


class MultiScaleNeck(nn.Module):
    """
    Projects the finest ViT feature (vit_features[-1]) to hidden_dim, then
    optionally generates extra coarser levels via stride-2 convolutions.

    Args:
        in_ch : embed_dim of the ViT backbone (e.g. 192 for tiny/small)
        cfg   : NeckConfig — controls hidden_dim and num_output_levels
    """

    def __init__(self, in_ch: int, cfg: NeckConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        self.hidden_dim        = D
        self.num_output_levels = cfg.num_output_levels

        # Project finest backbone feature → hidden_dim at H/16
        self.proj0 = nn.Sequential(
            nn.Conv2d(in_ch, D, kernel_size=1, bias=False),
            nn.GroupNorm(32, D),
        )

        # Stride-2 conv for each extra spatial level
        self.extra_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(D, D, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, D),
            )
            for _ in range(cfg.num_output_levels - 1)
        ])

    # ── helpers ───────────────────────────────────────────────────────────────

    def _flatten_and_embed(
        self, feats: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        parts, pos_parts, shapes = [], [], []
        for feat in feats:
            B, D, H, W = feat.shape
            flat = feat.flatten(2).permute(0, 2, 1)            # (B, H*W, D)
            pos  = _build_2d_sincos_pos(H, W, D, feat.device)  # (1, H*W, D)
            parts.append(flat)
            pos_parts.append(pos.expand(B, -1, -1))
            shapes.append((H, W))

        memory  = torch.cat(parts,     dim=1)                   # (B, ΣHW, D)
        pos_emb = torch.cat(pos_parts, dim=1)

        spatial_shapes  = torch.as_tensor(shapes, dtype=torch.long, device=memory.device)
        level_start_idx = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ])
        return memory, spatial_shapes, level_start_idx, pos_emb

    def _valid_ratios(self, B: int, L: int, device: torch.device) -> Tensor:
        return torch.ones(B, L, 2, dtype=torch.float32, device=device)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, vit_features: List[Tensor]) -> MultiScaleNeckOutput:
        """
        Args:
            vit_features : list of (B, embed_dim, H/16, W/16) from ViT blocks
                           (only vit_features[-1] is used as the finest feature)
        Returns:
            MultiScaleNeckOutput with num_output_levels spatial levels
        """
        finest = self.proj0(vit_features[-1])   # (B, D, H/16, W/16)
        levels = [finest]
        for extra in self.extra_projs:
            levels.append(extra(levels[-1]))    # (B, D, H/32, ...) stride-2

        memory, spatial_shapes, level_start_idx, pos_embed = self._flatten_and_embed(levels)
        valid_ratios = self._valid_ratios(finest.shape[0], len(levels), finest.device)

        return MultiScaleNeckOutput(
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_idx=level_start_idx,
            pos_embed=pos_embed,
            valid_ratios=valid_ratios,
        )

    # ── finest feature for CenterNet head ─────────────────────────────────────

    def finest_feature(self, vit_features: List[Tensor]) -> Tensor:
        """
        Returns the projected finest-scale feature (B, hidden_dim, H/16, W/16)
        for the CenterNet head.  Re-applies proj0 — acceptable since it is only
        called once per forward pass in HybridCenterNetDETR.
        """
        return self.proj0(vit_features[-1])
