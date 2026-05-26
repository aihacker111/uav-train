"""
HawkDet — ECViT backbone + HybridEncoder + 4-scale TOOD T-head with DFL.

Architecture:
    image
      └─ ECViT backbone  → [S8, S16, S32]
             └─ HybridEncoder → [S4(256), S8(256), S16(256), S32(256)]
                    └─ THead × 4  (one per scale, shared architecture)
                           └─ cls (C) + reg (4×(R+1)) [+ reid (D)]  per location

Training output  : {'cls': List[(B,C,Hi,Wi)], 'reg': List[(B,4*(R+1),Hi,Wi)],
                    'reid': List[(B,D,Hi,Wi)] or None}
Inference output : {'pred_boxes': (B,N,4) cxcywh [0,1],
                    'pred_scores': (B,N,C) sigmoid,
                    'reid': (B,N,D) L2-normalised or None}

The decoder (ECTransformer) is fully removed.  All prediction is dense and
anchor-free: no query limit, no Hungarian matching at inference.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor

from .head import THead, make_anchors, dfl_decode, ltrb_to_cxcywh


class HawkDet(nn.Module):
    """HawkDet detection model."""

    strides: List[int] = [4, 8, 16, 32]

    def __init__(
        self,
        ecdet:       nn.Module,   # ECDet(backbone, encoder, decoder) — decoder ignored
        num_classes: int,
        hidden_dim:  int  = 256,
        reg_max:     int  = 16,
        num_convs:   int  = 2,
        feat_ch:     int  = 128,
        reid_dim:    int  = 0,
    ) -> None:
        super().__init__()
        # Keep only backbone + encoder from ECDet
        self.backbone    = ecdet.backbone
        self.encoder     = ecdet.encoder
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.reid_dim    = reid_dim

        # One T-head per scale (S4, S8, S16, S32)
        self.heads = nn.ModuleList([
            THead(hidden_dim, num_classes, reg_max, num_convs,
                  feat_ch=feat_ch, reid_dim=reid_dim)
            for _ in range(len(self.strides))
        ])

        # Fixed projection for DFL expectation (shared, not learned)
        self.register_buffer(
            'proj',
            torch.arange(reg_max + 1, dtype=torch.float32),
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x:       Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        backbone_feats = self.backbone(x)              # [S8, S16, S32]
        feats          = self.encoder(backbone_feats)  # [S4, S8, S16, S32]

        cls_list, reg_list, reid_list = [], [], []
        for feat, head in zip(feats, self.heads):
            cls, reg, reid = head(feat)
            cls_list.append(cls)
            reg_list.append(reg)
            if reid is not None:
                reid_list.append(reid)

        if self.training:
            return {
                'cls':  cls_list,
                'reg':  reg_list,
                'reid': reid_list if reid_list else None,
            }

        return self._decode(cls_list, reg_list, reid_list or None, x.shape[-2:])

    # ── Inference decode ─────────────────────────────────────────────────────

    def _decode(
        self,
        cls_list:  List[Tensor],
        reg_list:  List[Tensor],
        reid_list: Optional[List[Tensor]],
        input_hw:  tuple,
    ) -> Dict[str, Any]:
        all_boxes, all_scores, all_reid = [], [], []
        B = cls_list[0].shape[0]

        for i, (cls, reg) in enumerate(zip(cls_list, reg_list)):
            _, C, H, W = cls.shape
            stride = self.strides[i]

            anchors  = make_anchors(H, W, stride, cls.device)
            reg_flat = reg.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            distance = dfl_decode(reg_flat, self.reg_max)

            all_boxes.append(ltrb_to_cxcywh(distance, anchors, stride, input_hw))
            all_scores.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, C).sigmoid())

            if reid_list is not None:
                reid_flat = reid_list[i].permute(0, 2, 3, 1).reshape(B, H * W, -1)
                all_reid.append(reid_flat)

        out = {
            'pred_boxes':  torch.cat(all_boxes,  dim=1),  # (B, N, 4)
            'pred_scores': torch.cat(all_scores, dim=1),  # (B, N, C)
        }
        if all_reid:
            reid_cat = torch.cat(all_reid, dim=1)          # (B, N, D)
            out['reid'] = torch.nn.functional.normalize(reid_cat, dim=-1)
        else:
            out['reid'] = None
        return out

    # ── Weight loading ───────────────────────────────────────────────────────

    def load_pretrained(self, path: str) -> None:
        """Load backbone + encoder weights from an ECDet checkpoint.

        Keys in the checkpoint that start with 'ecdet.backbone.' or
        'ecdet.encoder.' are remapped.  Decoder keys and shape-mismatched
        keys are silently skipped — detection heads always start from scratch.
        """
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict) and any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}

        current = self.state_dict()
        compatible: dict   = {}
        skipped_shape: list = []
        skipped_extra: list = []

        for k, v in state.items():
            # Remap ecdet.backbone.* → backbone.*, ecdet.encoder.* → encoder.*
            if k.startswith('ecdet.backbone.'):
                target = k.replace('ecdet.backbone.', 'backbone.', 1)
            elif k.startswith('ecdet.encoder.'):
                target = k.replace('ecdet.encoder.', 'encoder.', 1)
            elif k.startswith('backbone.') or k.startswith('encoder.'):
                target = k
            else:
                skipped_extra.append(k)
                continue

            if target not in current:
                skipped_extra.append(k)
                continue
            if v.shape != current[target].shape:
                skipped_shape.append((k, tuple(v.shape), tuple(current[target].shape)))
                continue
            compatible[target] = v

        self.load_state_dict(compatible, strict=False)
        print(
            f'[HawkDet] pretrained: {len(compatible)} tensors loaded '
            f'({len(skipped_shape)} shape-mismatch, {len(skipped_extra)} skipped)\n'
            f'  from: {path}'
        )

    def deploy(self) -> 'HawkDet':
        self.eval()
        if hasattr(self.backbone, 'deploy'):
            self.backbone.deploy()
        return self
