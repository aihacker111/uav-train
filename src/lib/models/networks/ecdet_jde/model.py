"""
EdgeCrafterJDE — ECViT + HybridEncoder + CenterNet JDE heads.

Drop-in backbone replacement for AMOT's DLA-34 inside MCJDETracker.

Forward output: [{'hm': Tensor, 'wh': Tensor, 'reg': Tensor, 'id': Tensor}]
  — single-element list so tracker calls model.forward(x)[-1]

hm  : raw logits (B, num_classes, H/4, W/4)  — tracker calls .sigmoid_() itself
wh  : (B, 2, H/4, W/4)  pixel-scale width/height
reg : (B, 2, H/4, W/4)  sub-pixel center offset
id  : (B, reid_dim, H/4, W/4)  raw ReID embeddings (tracker L2-normalises)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _head(in_ch: int, out_ch: int, head_conv: int) -> nn.Module:
    if head_conv > 0:
        return nn.Sequential(
            nn.Conv2d(in_ch, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, out_ch, 1, bias=True),
        )
    return nn.Conv2d(in_ch, out_ch, 1, bias=True)


class EdgeCrafterJDE(nn.Module):
    """ECViT backbone + HybridEncoder + CenterNet JDE heads for AMOT-style tracking."""

    def __init__(
        self,
        ecdet:       nn.Module,   # ECDet(backbone, encoder) built from YAMLConfig
        num_classes: int,
        hidden_dim:  int = 192,   # HybridEncoder S4 output channels
        head_conv:   int = 64,
        reid_dim:    int = 128,
    ) -> None:
        super().__init__()
        self.backbone = ecdet.backbone
        self.encoder  = ecdet.encoder

        self.hm_head  = _head(hidden_dim, num_classes, head_conv)
        self.wh_head  = _head(hidden_dim, 2,           head_conv)
        self.reg_head = _head(hidden_dim, 2,           head_conv)
        self.id_head  = _head(hidden_dim, reid_dim,    head_conv) if reid_dim > 0 else None

        # CenterNet bias init: prior probability 0.01 → -log((1-0.01)/0.01) ≈ -4.595
        last_hm = self.hm_head[-1] if isinstance(self.hm_head, nn.Sequential) else self.hm_head
        nn.init.constant_(last_hm.bias, -4.595)

    def forward(self, x: Tensor, targets=None):
        backbone_feats = self.backbone(x)              # [S8, S16, S32]
        feats          = self.encoder(backbone_feats)  # [S4, S8, S16, S32]
        s4             = feats[0]                      # (B, hidden_dim, H/4, W/4)

        out = {
            'hm':  self.hm_head(s4),    # raw logits; MCJDETracker calls .sigmoid_()
            'wh':  self.wh_head(s4),
            'reg': self.reg_head(s4),
        }
        if self.id_head is not None:
            out['id'] = self.id_head(s4)
        else:
            out['id'] = torch.zeros(
                s4.shape[0], 1, s4.shape[2], s4.shape[3], device=s4.device
            )

        return [out]  # list — tracker accesses model.forward(x)[-1]

    def load_pretrained(self, path: str) -> None:
        """Load backbone + encoder weights from an ECDet / HawkDet checkpoint."""
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        if isinstance(state, dict) and any(k.startswith('module.') for k in state):
            state = {k[7:]: v for k, v in state.items()}

        current = self.state_dict()
        ok: dict = {}
        for k, v in state.items():
            if k.startswith('ecdet.backbone.'):
                target = k.replace('ecdet.backbone.', 'backbone.', 1)
            elif k.startswith('ecdet.encoder.'):
                target = k.replace('ecdet.encoder.', 'encoder.', 1)
            elif k.startswith('backbone.') or k.startswith('encoder.'):
                target = k
            else:
                continue
            if target in current and v.shape == current[target].shape:
                ok[target] = v

        self.load_state_dict(ok, strict=False)
        print(f'[EdgeCrafterJDE] pretrained: {len(ok)} tensors loaded from {path}')
