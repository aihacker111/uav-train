# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Adapted for AMOT: removed fairscale/util.box_ops dependencies.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, Mlp, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, Mlp, trunc_normal_


def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0),
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_cae=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_cae = use_cae
        if use_cae:
            # CAE style: no bias in linear, separate learnable q/v biases
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        if self.use_cae:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            attn.masked_fill_(mask.reshape(B, 1, 1, N).expand_as(attn), float('-inf'))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 window=False, use_cae=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_cae=use_cae)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.window = window
        self.use_cae = use_cae
        if use_cae:
            init_values = 0.1
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x, mask=None):
        B, HW, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if not self.window:
            x = x.reshape(B // 16, 16 * HW, C)
            if mask is not None:
                mask = mask.reshape(B // 16, 16 * HW)

        if self.use_cae:
            x = self.gamma_1 * self.attn(x, mask)
        else:
            x = self.attn(x, mask)

        if not self.window:
            x = x.reshape(B, HW, C)
            if mask is not None:
                mask = mask.reshape(B, HW)

        x = shortcut + self.drop_path(x)
        if self.use_cae:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """
    ViT backbone from LW-DETR (ViTDet style).
    Input image H and W must both be divisible by 64
    (patch_size=16, then H/16 and W/16 must be divisible by 4 for windowed attention).
    Output: list of (B, C, H/16, W/16) tensors at selected layer indexes.
    """

    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 use_abs_pos=True, window_block_indexes=(), pretrain_img_size=224,
                 pretrain_use_cls_token=True, out_feature_indexes=None, use_cae=False):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) ** 2
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,
                window=(i in window_block_indexes),
                use_cae=use_cae,
            )
            self.blocks.append(block)

        self.window_block_indexes = window_block_indexes
        out_feature_indexes = [ind if ind >= 0 else ind + depth for ind in out_feature_indexes]
        out_feature_indexes = [ind for ind in range(depth) if ind in out_feature_indexes]
        self._out_features = [True if i in out_feature_indexes else False for i in range(depth)]
        self._out_feature_channels = [embed_dim] * len(out_feature_indexes)
        assert self._out_features[-1] is True

        self.embed_dim = embed_dim

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, H/16, W/16, C)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token,
                                 (x.shape[1], x.shape[2]))

        B, H, W, C = x.shape
        assert (H % 4 == 0) and (W % 4 == 0), \
            f"H={H}, W={W} after patch embed must both be divisible by 4. " \
            f"Use input image size divisible by 64."
        h, w = H // 4, W // 4

        # Reshape into 4×4 sub-windows for efficient attention
        x = x.reshape(B, 4, h, 4, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B * 16, h * w, C)

        out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x, mask=None)
            if self._out_features[idx]:
                out.append(x.reshape(B, 4, 4, h, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W))
        return out
