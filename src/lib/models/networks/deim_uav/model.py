"""
HybridDEIM: DEIMv2 backbone/encoder/decoder + CenterNet auxiliary head.

Architecture:
    image
      ├─ deim.backbone → [f1(S8), f2(S16), f3(S32)]
      │       └─ deim.encoder → same scales, richer features
      │               └─ deim.decoder → pred_boxes, pred_logits
      └─ cn_upsample(f1.detach()) → S4 feature
                └─ CenterNetHead → hm, wh, reg
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, List, Optional

from .heads import CenterNetHead, CenterNetOutput, DETROutput
from .config import CenterNetHeadConfig


class HybridDEIM(nn.Module):
    def __init__(
        self,
        deim: nn.Module,
        num_classes: int,
        hidden_dim: int = 192,
        head_conv: int  = 32,
    ) -> None:
        super().__init__()
        self.deim = deim

        self.cn_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cn_head = CenterNetHead(hidden_dim, CenterNetHeadConfig(head_conv=head_conv, num_classes=num_classes))

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        feats    = self.deim.backbone(x)
        feats    = self.deim.encoder(feats)

        # Branch 1 — CenterNet (detached: gradients don't flow back to encoder)
        cn_out   = self.cn_head(self.cn_upsample(feats[0].detach()))

        # Branch 2 — DEIMv2 decoder (gradients flow to backbone + encoder)
        deim_out = self.deim.decoder(feats, targets)

        return {'stage1': cn_out, 'stage2': self._wrap_deim(deim_out)}

    def _wrap_deim(self, deim_out: Dict[str, Any]) -> DETROutput:
        pred_boxes  = deim_out['pred_boxes']
        pred_logits = deim_out['pred_logits']
        aux = deim_out.get('aux_outputs') or []
        if aux:
            boxes_all  = torch.stack([a['pred_boxes']  for a in aux] + [pred_boxes])
            logits_all = torch.stack([a['pred_logits'] for a in aux] + [pred_logits])
        else:
            boxes_all  = pred_boxes.unsqueeze(0)
            logits_all = pred_logits.unsqueeze(0)
        return DETROutput(
            boxes      = pred_boxes,
            logits     = pred_logits,
            reid       = None,
            boxes_all  = boxes_all,
            logits_all = logits_all,
        )

    def load_pretrained(self, path: str) -> None:
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        missing, unexpected = self.deim.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(missing)
        print(f'[HybridDEIM] pretrained: {n_loaded}/{len(state)} tensors loaded from {path}')
        if missing:
            print(f'  missing    ({len(missing)}): {missing[:4]}{"…" if len(missing) > 4 else ""}')
        if unexpected:
            print(f'  unexpected ({len(unexpected)}): {unexpected[:4]}{"…" if len(unexpected) > 4 else ""}')

    def deploy(self) -> 'HybridDEIM':
        self.eval()
        if hasattr(self.deim, 'deploy'):
            self.deim.deploy()
        return self
