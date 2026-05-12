from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lwdetr_vit import ViT


# ── Shared head builder ──────────────────────────────────────────────────────

def _make_head(in_ch, out_ch, head_conv):
    if head_conv > 0:
        return nn.Sequential(
            nn.Conv2d(in_ch, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, out_ch, kernel_size=1, bias=True),
        )
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)


def _init_heads(heads_dict):
    for head, module in heads_dict.items():
        last = list(module.children())[-1] if isinstance(module, nn.Sequential) else module
        if hasattr(last, 'bias') and last.bias is not None:
            nn.init.constant_(last.bias, -4.6 if head == 'hm' else 0.0)


# ── Decoder: N × (B, C, H/16, W/16) → (B, out_ch, H/4, W/4) ───────────────

def _up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LWDeTrDecoder(nn.Module):
    """
    Fuse N feature maps at H/16 resolution, then upsample 4× to H/4.
    Each feature map is (B, embed_dim, H/16, W/16).
    """

    def __init__(self, embed_dim, out_ch=64, num_feats=3):
        super().__init__()
        mid_ch = max(out_ch * 2, 128)
        # project each feature to mid_ch, then fuse by addition
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_feats)
        ])
        # two 2× upsample blocks: H/16 → H/8 → H/4
        self.up1 = _up_block(mid_ch, mid_ch)
        self.up2 = _up_block(mid_ch, out_ch)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        out = None
        for i, feat in enumerate(feats):
            p = self.proj[i](feat)
            out = p if out is None else out + p
        return self.up2(self.up1(out))


# ── Full model ───────────────────────────────────────────────────────────────

class LWDeTrCenterNet(nn.Module):
    """
    LW-DETR ViT backbone + FPN-style decoder + CenterNet heads.
    Compatible with AMOT's reid_motion() — output is a dense feature map at H/4.
    """

    def __init__(self, backbone: ViT, decoder: LWDeTrDecoder, heads: dict, head_conv: int):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.heads = list(heads.keys())

        out_ch = decoder.up2[-3].out_channels  # output channels of last ConvTranspose
        for head, num_out in heads.items():
            self.__setattr__(head, _make_head(out_ch, num_out, head_conv))

        _init_heads({h: self.__getattr__(h) for h in self.heads})

    def forward(self, x):
        feats = self.backbone(x)           # list of (B, C, H/16, W/16)
        feat = self.decoder(feats)         # (B, out_ch, H/4, W/4)
        out = {h: self.__getattr__(h)(feat) for h in self.heads}
        return [out]


# ── Factory functions ────────────────────────────────────────────────────────

def _build(embed_dim, depth, window_block_indexes, out_feature_indexes,
           heads, head_conv, num_layers):
    # num_heads: tiny/small use 12 (head_dim=16), base uses 12 (head_dim=32)
    num_heads = 12
    backbone = ViT(
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos=True,
        window_block_indexes=window_block_indexes,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature_indexes=out_feature_indexes,
        use_cae=True,
    )
    num_feats = len(out_feature_indexes)
    decoder = LWDeTrDecoder(embed_dim=embed_dim, out_ch=64, num_feats=num_feats)
    return LWDeTrCenterNet(backbone, decoder, heads, head_conv)


def get_lwdetr_net(num_layers=0, heads=None, head_conv=256):
    """
    num_layers selects variant:
      0 or unset → tiny  (embed=192, depth=6)
      1           → small (embed=384, depth=10)
      2           → base  (embed=768, depth=12)
    """
    if heads is None:
        heads = {'hm': 1, 'wh': 2, 'reg': 2, 'id': 128}

    configs = {
        # (embed_dim, depth, window_block_indexes, out_feature_indexes)
        # Matches LW-DETR official checkpoints exactly:
        0: (192,  6, [0, 2, 4],          [1, 3, 5]),        # lwdetr_tiny
        1: (192, 10, [0, 1, 3, 6, 7, 9], [2, 4, 5, 9]),     # lwdetr_small
        2: (384, 12, [0, 1, 3, 4, 6, 7, 9, 10], [2, 5, 8, 11]),  # lwdetr_base
    }
    key = num_layers if num_layers in configs else 0
    embed_dim, depth, win_idx, out_idx = configs[key]
    return _build(embed_dim, depth, win_idx, out_idx, heads, head_conv, num_layers)
