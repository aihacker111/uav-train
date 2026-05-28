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
        default_wh: float = 0.05,   # default half-size for heatmap-derived proposals
    ) -> None:
        super().__init__()
        self.deim = deim
        self.default_wh = default_wh

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

    def _heatmap_to_ref_points(self, hm: Tensor, wh_map: Tensor, k: int) -> Tensor:
        """
        Extract top-K peaks from the CenterNet heatmap and convert to DETR
        reference points in logit (inverse-sigmoid) space.

        Args:
            hm     : (B, C, H, W) — sigmoid heatmap, values in (0, 1)
            wh_map : (B, 2, H, W) — predicted WH in heatmap-pixel scale (cn_head output)
            k      : number of reference points to produce (= num_queries)

        Returns:
            (B, k, 4) — cxcywh reference points in logit space, same format
            as DEIMTransformer's enc_topk_bbox_unact so they can be dropped in
            as a direct replacement.
        """
        B, C, H, W = hm.shape

        score_map = hm.max(dim=1).values          # (B, H, W)
        flat      = score_map.reshape(B, H * W)   # (B, H*W)

        _, topk_idx = flat.topk(k, dim=-1)        # (B, k)

        y_idx = (topk_idx // W).float()
        x_idx = (topk_idx  % W).float()

        cx = (x_idx + 0.5) / W
        cy = (y_idx + 0.5) / H

        # Gather predicted WH at peak locations and normalise to [0, 1].
        # wh_map is in heatmap-pixel scale; dividing by (W, H) gives the same
        # normalised ratio as the input image (output_pixels / output_size ==
        # input_pixels / input_size since stride cancels).
        flat_w = wh_map[:, 0].reshape(B, H * W)  # (B, H*W)
        flat_h = wh_map[:, 1].reshape(B, H * W)
        pred_w = flat_w.gather(1, topk_idx).float() / W   # (B, k)
        pred_h = flat_h.gather(1, topk_idx).float() / H

        # cxcywh in [0, 1] — clamp away from boundaries before logit
        boxes = torch.stack([cx, cy, pred_w, pred_h], dim=-1).clamp(1e-4, 1.0 - 1e-4)

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
        # Detach both hm and wh so Stage-2 loss does not backprop through here.
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