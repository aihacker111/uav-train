"""
DEIMMotNet: DEIMv2 backbone/encoder + CenterNet detection + ReID heads.

Replaces the DETR decoder entirely, producing the same output format as
the original AMOT (DLA-34) model so the full AMOT tracking pipeline is
reused without modification:

    [{'hm': (B,C,H,W), 'wh': (B,2,H,W), 'reg': (B,2,H,W), 'id': (B,D,H,W)}]

Benefits vs HybridDEIM:
  - No query limit (num_queries cap removed — all objects decoded)
  - Spatial ReID map → reid_motion tracking in MCJDETracker works correctly
  - Faster inference (no decoder cross-attention)
  - McMotLoss (AMOT) directly applicable
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional


def _make_head(in_ch: int, out_ch: int, head_conv: int) -> nn.Sequential:
    """3×3 conv + ReLU + 1×1 conv prediction head."""
    return nn.Sequential(
        nn.Conv2d(in_ch, head_conv, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(head_conv, out_ch, kernel_size=1, bias=True),
    )


class DEIMMotNet(nn.Module):
    """
    DEIM backbone + HybridEncoder + CenterNet/ReID heads.

    The decoder from the original DEIM model is ignored at inference time.
    Pretrained backbone/encoder weights are loaded via load_pretrained().

    Args:
        deim       : Full DEIM model (backbone + encoder + decoder).
                     Only backbone and encoder are used in forward().
        num_classes: Number of object categories.
        hidden_dim : Encoder output channels — must match HybridEncoder.hidden_dim
                     in the YAML config (typically 256).
        head_conv  : Intermediate channels for all conv heads (64 is a good default).
        reid_dim   : ReID embedding dimension.
    """

    def __init__(
        self,
        deim: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        head_conv: int = 64,
        reid_dim: int = 128,
    ) -> None:
        super().__init__()
        self.deim = deim

        # S8 → S4 upsample
        # self.cn_upsample = nn.Sequential(
        #     nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False),
        #     nn.GroupNorm(32, hidden_dim),
        #     nn.ReLU(inplace=True),
        # )
        self.cn_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False)
        )

        # Prediction heads — all output raw logits (sigmoid applied in the loss)
        self.hm_head  = _make_head(hidden_dim, num_classes, head_conv)
        self.wh_head  = _make_head(hidden_dim, 2,           head_conv)
        self.reg_head = _make_head(hidden_dim, 2,           head_conv)
        self.id_head  = _make_head(hidden_dim, reid_dim,    head_conv)

        # CenterNet-style heatmap bias init: prior ≈ 0.01  →  logit ≈ −4.6
        nn.init.constant_(self.hm_head[-1].bias, -4.595)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Tensor]]:
        """
        Args:
            x      : (B, 3, H, W) input image (ImageNet-normalised).
            targets: ignored — kept for API compatibility with HybridDEIM.

        Returns:
            List of one dict (num_stacks = 1):
              'hm'  (B, C, Hf, Wf)       raw heatmap logits
              'wh'  (B, 2, Hf, Wf)       width/height
              'reg' (B, 2, Hf, Wf)       sub-pixel offset
              'id'  (B, reid_dim, Hf, Wf) reid embedding map
            where Hf = H/4, Wf = W/4  (stride 4, same as AMOT).
        """
        feats = self.deim.backbone(x)
        feats = self.deim.encoder(feats)   # list of [S8, S16, S32] feature maps

        s4 = self.cn_upsample(feats[0])    # (B, hidden_dim, H/4, W/4)

        return [{
            'hm':  self.hm_head(s4),
            'wh':  self.wh_head(s4),
            'reg': self.reg_head(s4),
            'id':  self.id_head(s4),
        }]

    # ── Weight utilities ───────────────────────────────────────────────────────

    def load_pretrained(self, path: str) -> None:
        """Load DEIM backbone+encoder pretrained weights (decoder keys are skipped)."""
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        missing, unexpected = self.deim.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(missing)
        print(f'[DEIMMotNet] pretrained: {n_loaded}/{len(state)} tensors loaded from {path}')
        if missing:
            print(f'  missing ({len(missing)}): {missing[:4]}{"…" if len(missing) > 4 else ""}')
        if unexpected:
            print(f'  unexpected ({len(unexpected)}): {unexpected[:4]}{"…" if len(unexpected) > 4 else ""}')
