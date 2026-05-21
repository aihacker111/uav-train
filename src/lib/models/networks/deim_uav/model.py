"""
HybridDEIM: DEIMv2 backbone/encoder/decoder + CenterNet auxiliary head.

Architecture:
    image
      ├─ deim.backbone → [f1(S8), f2(S16), f3(S32)]
      │       └─ deim.encoder → same scales, richer features
      │               └─ deim.decoder(feats_guided) → pred_boxes, pred_logits
      └─ cn_upsample(f1.detach()) → S4 feature
                └─ CenterNetHead → hm, wh, reg
                        └─ spatial prior → modulates f1 → feats_guided
                                (S2 loss trains cn_head via this path)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        reid_dim: int   = 0,
    ) -> None:
        super().__init__()
        self.deim = deim

        self.cn_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cn_head = CenterNetHead(hidden_dim, CenterNetHeadConfig(head_conv=head_conv, num_classes=num_classes))

        self.reid_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        ) if reid_dim > 0 else None

        # Learnable scale for heatmap spatial prior: stored as raw logit, applied via
        # sigmoid so the effective scale is always in (0, 1). logit(0.1) ≈ -2.197.
        self.heatmap_scale = nn.Parameter(torch.tensor(-2.197))

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        feats    = self.deim.backbone(x)
        feats    = self.deim.encoder(feats)

        # Branch 1 — CenterNet on detached features: S1 loss only trains cn_head/cn_upsample,
        # never the encoder. Gradient from S2 will reach cn_head via the spatial prior below.
        cn_out = self.cn_head(self.cn_upsample(feats[0].detach()))

        # Spatial prior: sum heatmap over classes → (B,1,H_hm,W_hm).
        # Detached: prior is a fixed spatial signal per forward pass.
        # Keeping gradient here would let S2 loss train cn_head via a second path,
        # conflicting with S1 focal loss and causing heatmap oscillation → noisy prior.
        prior = cn_out.hm.sum(dim=1, keepdim=True).detach()
        prior = prior / prior.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)

        # Resize from stride-4 heatmap → stride-8 encoder feature spatial size
        prior_s8 = F.interpolate(prior, size=feats[0].shape[2:],
                                  mode='bilinear', align_corners=False)

        # Amplify encoder features at high-confidence object regions.
        # sigmoid(heatmap_scale) ∈ (0,1) — bounded by construction, never explodes.
        # Starts at sigmoid(-2.197) ≈ 0.1 and grows as heatmap becomes more reliable.
        scale = torch.sigmoid(self.heatmap_scale)
        feats_guided = list(feats)
        feats_guided[0] = feats[0] * (1.0 + scale * prior_s8)

        # Branch 2 — DEIMv2 decoder sees spatially-guided encoder features.
        # Normalize targets: labels from dataset are (N, 2) = [cls_id, track_id],
        # DEIM denoising expects (N,) class indices only.
        deim_targets = None
        if targets is not None:
            deim_targets = [
                {
                    'boxes':  t['boxes'].to(x.device),
                    'labels': (t['labels'][:, 0].long() if t['labels'].ndim == 2
                               else t['labels'].long()).to(x.device),
                }
                for t in targets
            ]
        deim_out = self.deim.decoder(feats_guided, deim_targets)

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

        reid = None
        if self.reid_mlp is not None and 'hs' in deim_out:
            reid = F.normalize(self.reid_mlp(deim_out['hs']), dim=-1)

        return DETROutput(
            boxes      = pred_boxes,
            logits     = pred_logits,
            reid       = reid,
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
