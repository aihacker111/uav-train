"""
DEIMMotNet: DEIMv2 backbone/encoder + CenterNet detection + ReID heads.

Output format (compatible with AMOT tracking pipeline):
    [{'hm': (B,C,H,W), 'wh': (B,2,H,W), 'reg': (B,2,H,W), 'id': (B,D,H,W)}]

Feature pyramid
---------------
HybridEncoder produces three outputs [enc_S8, enc_S16, enc_S32] (each
hidden_dim channels).  All three are fused via a lightweight FPN top-down
path so no encoder compute is wasted:

    enc_S32 ─ proj_s32 ──────────────────────────────► p32
    enc_S16 ─ proj_s16 ─ + nearest_up(p32, ×2) ──────► p16
    enc_S8  ─ proj_s8  ─ + nearest_up(p16, ×2) ──────► p8
                               bilinear ×2
                                    ↓
                          + lateral_s4(sta.stem hook)   ← 16ch, ~3 K params
                               DilatedContext
                                    ↓
                                 heads  (H/4 × W/4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional


class DilatedContext(nn.Module):
    """
    Multi-scale depthwise-dilated context block for dense / cluttered regions.

    Two parallel depthwise convolutions (d1, d2) widen the receptive field on
    the S4 feature map without changing resolution or channel count.  A
    pointwise conv merges the branches; a residual skip preserves semantics.

    On S4 (stride-4 map, default dilations=2,4):
        d=2 → 5×5 kernel  → ~20×20 original-image pixels
        d=4 → 9×9 kernel  → ~36×36 original-image pixels

    Extra params: 2×(C×9) depthwise + 2C×C pointwise  ≈ 135 K for C=256
    """

    def __init__(self, channels: int, dilations: tuple = (2, 4)) -> None:
        super().__init__()
        d1, d2 = dilations
        self.dw1 = nn.Conv2d(channels, channels, 3,
                             padding=d1, dilation=d1, groups=channels, bias=False)
        self.dw2 = nn.Conv2d(channels, channels, 3,
                             padding=d2, dilation=d2, groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn  = nn.BatchNorm2d(channels)
        nn.init.kaiming_normal_(self.pw.weight, mode='fan_out')

    def forward(self, x: Tensor) -> Tensor:
        ctx = torch.cat([self.dw1(x), self.dw2(x)], dim=1)   # (B, 2C, H, W)
        return x + self.bn(F.relu(self.pw(ctx), inplace=True))


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

    Args:
        deim       : Full DEIM model (backbone + encoder + decoder).
        num_classes: Number of object categories.
        hidden_dim : Encoder output channels (HybridEncoder.hidden_dim).
        head_conv  : Intermediate channels for prediction heads.
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

        # ── S4 lateral via hook on sta.stem ───────────────────────────────────
        self._s4_cache: Optional[Tensor] = None
        self._hook_handle = None

        s4_ch = self._register_s4_hook(deim)

        # proj_enc: 1×1 to stabilise enc_S8 before bilinear upsample
        self.proj_enc = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        nn.init.kaiming_normal_(self.proj_enc.weight, mode='fan_out')

        if s4_ch is not None:
            self.lateral_s4 = nn.Conv2d(s4_ch, hidden_dim, 1, bias=False)
            nn.init.kaiming_normal_(self.lateral_s4.weight, mode='fan_out')
            print(f'[DEIMMotNet] STA S4 lateral: {s4_ch}ch → {hidden_dim}ch  '
                  f'({s4_ch * hidden_dim:,} params)')
        else:
            self.lateral_s4 = None
            print('[DEIMMotNet] STA not found — using bilinear upsample only')

        # ── Dilated context (dense-scene receptive field widening) ────────────
        self.dilated_ctx = DilatedContext(hidden_dim)

        # ── Prediction heads ──────────────────────────────────────────────────
        self.hm_head  = _make_head(hidden_dim, num_classes, head_conv)
        self.wh_head  = _make_head(hidden_dim, 2,           head_conv)
        self.reg_head = _make_head(hidden_dim, 2,           head_conv)
        self.id_head  = _make_head(hidden_dim, reid_dim,    head_conv)

        self._init_head_weights(log_wh=False)  # caller sets via load_pretrained or opt

    # ── Hook setup ─────────────────────────────────────────────────────────────

    def _register_s4_hook(self, deim: nn.Module) -> Optional[int]:
        """
        Register a forward hook on deim.backbone.sta.stem.
        Returns the S4 channel count, or None if STA is not present.
        """
        backbone = deim.backbone
        if not (hasattr(backbone, 'sta') and hasattr(backbone.sta, 'stem')):
            return None

        def _hook(module, inp, out):
            self._s4_cache = out

        self._hook_handle = backbone.sta.stem.register_forward_hook(_hook)

        # Probe channel count with a dummy forward
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                backbone(dummy)
            s4_ch = self._s4_cache.shape[1]
            self._s4_cache = None
            return s4_ch
        except Exception as e:
            print(f'[DEIMMotNet] S4 probe failed: {e} — disabling lateral')
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None
            return None

    # ── Weight init ────────────────────────────────────────────────────────────

    def _init_head_weights(self, log_wh: bool = False) -> None:
        for head in (self.hm_head, self.wh_head, self.reg_head, self.id_head):
            nn.init.kaiming_normal_(head[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(head[0].bias, 0.0)
            nn.init.constant_(head[2].bias, 0.0)
        # heatmap prior: sigmoid(−4.595) ≈ 0.01
        nn.init.constant_(self.hm_head[2].bias, -4.595)
        # log_wh: init bias to log(typical object size in feature-map coords)
        # VisDrone cars ~50×30px at stride-4 → log(12.5)≈2.5, log(7.5)≈2.0
        # Avoids cold-start where exp(0)=1px boxes dominate early training
        if log_wh:
            nn.init.constant_(self.wh_head[2].bias[0], 2.5)  # log width
            nn.init.constant_(self.wh_head[2].bias[1], 2.0)  # log height

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Tensor]]:
        """
        Returns list of one dict:
          'hm'  (B, C, H/4, W/4)        raw heatmap logits
          'wh'  (B, 2, H/4, W/4)        width / height
          'reg' (B, 2, H/4, W/4)        sub-pixel offset
          'id'  (B, reid_dim, H/4, W/4) reid embedding map
        """
        # Reset cache; hook fills it during backbone forward
        self._s4_cache = None

        feats     = self.deim.backbone(x)          # [S8, S16, S32] unchanged
        enc_feats = self.deim.encoder(feats)        # [enc_S8, enc_S16, enc_S32]

        # Bilinear upsample enc_S8 → S4 resolution (no learnable kernel)
        s4 = F.interpolate(
            self.proj_enc(enc_feats[0]),
            scale_factor=2, mode='bilinear', align_corners=False,
        )

        # Fuse with STA S4: fine-grained spatial details captured by hook
        if self.lateral_s4 is not None and self._s4_cache is not None:
            s4 = s4 + self.lateral_s4(self._s4_cache)

        # Widen receptive field for dense / closely-packed objects
        s4 = self.dilated_ctx(s4)

        return [{
            'hm':  self.hm_head(s4),
            'wh':  self.wh_head(s4),
            'reg': self.reg_head(s4),
            'id':  self.id_head(s4),
        }]

    # ── Weight loading ─────────────────────────────────────────────────────────

    def load_pretrained(self, path: str) -> None:
        """Load DEIM backbone+encoder pretrained weights (decoder keys skipped)."""
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
