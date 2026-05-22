"""
HybridDEIM: DEIMv2 backbone/encoder/decoder + CenterNet auxiliary head.

Architecture:
    image
      ├─ deim.backbone → [f1(S8), f2(S16), f3(S32)]
      │       └─ deim.encoder → same scales, richer features
      │
      ├─ cn_upsample(f1.detach()) → S4 feature
      │       └─ CenterNetHead → hm, wh, reg
      │               ↓
      │         top-K heatmap peaks  ←── object proposals
      │               ↓
      └─ deim.decoder(feats, ref_points=peaks) → refined pred_boxes, pred_logits

Design rationale:
  CenterNet acts as a lightweight proposal generator: its heatmap peaks at S4
  resolution are converted to reference points that seed the DETR decoder's
  cross-attention, replacing the generic encoder-anchor selection.

  Gradient flow:
    - Stage-1 loss  → cn_head, cn_upsample only  (feats[0] is detached)
    - Stage-2 loss  → full encoder + decoder      (feats not detached)
    - Heatmap peaks are detached before reference-point conversion, so
      Stage-2 does NOT backprop through the peak extraction step — the
      two branches have clean, non-conflicting gradient paths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, List, Optional

from .heads import CenterNetHead, CenterNetOutput
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
        self.cn_head = CenterNetHead(
            hidden_dim,
            CenterNetHeadConfig(head_conv=head_conv, num_classes=num_classes),
        )

        self.reid_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        ) if reid_dim > 0 else None

    # ── Heatmap → reference points ─────────────────────────────────────────────

    def _heatmap_to_ref_points(self, hm: Tensor, cn_wh: Tensor, k: int) -> Tensor:
        """
        Extract top-K peaks from the CenterNet heatmap and convert to DETR
        reference points in logit (inverse-sigmoid) space.

        Args:
            hm    : (B, C, H, W) — sigmoid heatmap, values in (0, 1)
            cn_wh : (B, 2, H, W) — predicted wh in heatmap pixel space (detached)
            k     : number of reference points (= num_queries)

        Returns:
            (B, k, 4) — cxcywh reference points in logit space.
        """
        B, C, H, W = hm.shape

        # Max-score across classes → pick K highest-scoring pixels globally
        score_map = hm.max(dim=1).values          # (B, H, W)
        flat      = score_map.reshape(B, H * W)   # (B, H*W)
        _, topk_idx = flat.topk(k, dim=-1)        # (B, k)

        # Flat index → normalised centre coordinates
        y_idx = (topk_idx // W).float()
        x_idx = (topk_idx  % W).float()
        cx = (x_idx + 0.5) / W                   # (B, k)
        cy = (y_idx + 0.5) / H                   # (B, k)

        # Gather cn_wh at peak positions and normalize to [0, 1].
        # cn_wh is in heatmap pixel space (same grid as hm), so dividing by
        # W / H gives the fraction of the image — identical to how cx/cy are built.
        # Clamp keeps values sensible when the wh head is still noisy early in
        # training: min=0.01 ≈ 13px at 1280px, max=0.5 = half the image.
        wh_flat  = cn_wh.reshape(B, 2, H * W)                        # (B, 2, H*W)
        idx_exp  = topk_idx.unsqueeze(1).expand(B, 2, k)             # (B, 2, k)
        wh_peaks = wh_flat.gather(2, idx_exp)                        # (B, 2, k)
        w_norm   = (wh_peaks[:, 0, :] / W).clamp(0.01, 0.5)         # (B, k)
        h_norm   = (wh_peaks[:, 1, :] / H).clamp(0.01, 0.5)         # (B, k)

        # cxcywh → clamp away from [0,1] boundaries → logit space
        boxes = torch.stack([cx, cy, w_norm, h_norm], dim=-1).clamp(1e-4, 1.0 - 1e-4)
        return torch.log(boxes / (1.0 - boxes))   # (B, k, 4)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # ── Shared backbone + encoder ──────────────────────────────────────────
        feats = self.deim.backbone(x)
        feats = self.deim.encoder(feats)

        # ── Branch 1: CenterNet on detached S8 feature ────────────────────────
        # .detach() ensures Stage-1 loss only trains cn_head + cn_upsample.
        # Stage-2 loss trains the encoder via its own clean gradient path.
        cn_feat = self.cn_upsample(feats[0].detach())   # S4 feature map
        cn_out  = self.cn_head(cn_feat)

        # ── Heatmap peaks → DETR reference points ─────────────────────────────
        # Detach heatmap so Stage-2 loss does not backprop through peak extraction.
        num_queries = self.deim.decoder.num_queries
        heatmap_ref = self._heatmap_to_ref_points(
            cn_out.hm.detach(), cn_out.wh.detach(), k=num_queries,
        )   # (B, num_queries, 4) in logit space

        # ── Branch 2: DETR decoder seeded with heatmap proposals ──────────────
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

        deim_out = self.deim.decoder(feats, deim_targets, heatmap_ref_points=heatmap_ref)

        return {'stage1': cn_out, 'stage2': self._wrap_deim(deim_out)}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _wrap_deim(self, deim_out: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            'pred_boxes':  deim_out['pred_boxes'],
            'pred_logits': deim_out['pred_logits'],
        }
        for key in ('pred_corners', 'ref_points', 'up', 'reg_scale',
                    'aux_outputs', 'enc_aux_outputs', 'enc_meta',
                    'pre_outputs', 'dn_outputs', 'dn_pre_outputs', 'dn_meta'):
            if key in deim_out:
                out[key] = deim_out[key]
        if self.reid_mlp is not None and 'hs' in deim_out:
            out['reid'] = F.normalize(self.reid_mlp(deim_out['hs']), dim=-1)
        return out

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