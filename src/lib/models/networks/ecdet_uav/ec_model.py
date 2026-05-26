"""
HybridECDet: ECDet backbone/encoder/decoder with S4 as a native encoder level.

Architecture:
    image
      └─ ecdet.backbone → [f1_raw(S8,C_b), f2_raw(S16,C_b), f3_raw(S32,C_b)]
              └─ ecdet.encoder → [S4(D), S8(D), S16(D), S32(D)]
                      (S4 = upsample of S8 post-PAN; D = hidden_dim)
                      └─ ecdet.decoder → refined pred_boxes, pred_logits

Gradient flow:
  All loss → decoder → encoder → backbone (single gradient path, no detach).

  enc_score_head runs on the full 4-level memory (L = H4W4 + H8W8 + H16W16 + H32W32).
  _select_topk naturally picks S4 positions for small objects → queries initialised
  at dense S4 locations → decoder cross-attends all 4 levels via deformable attention.

Coordinate system:
  - DETR boxes GT: normalised cxcywh in [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, List, Optional

from .heads import CenterNetHead, CenterNetOutput, DETROutput, extract_peaks_as_ref_points
from .config import CenterNetHeadConfig


class HybridECDet(nn.Module):
    def __init__(
        self,
        ecdet: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        reid_dim: int   = 0,
        heatmap_head_conv: int = 64,
    ) -> None:
        super().__init__()
        self.ecdet = ecdet

        self.reid_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        ) if reid_dim > 0 else None

        # CenterNet heatmap head on S4 feature (stride-4, hidden_dim channels).
        # Its peaks are used as dynamic reference points for ECTransformer queries
        # instead of the fixed encoder top-K selection.
        hm_cfg = CenterNetHeadConfig(head_conv=heatmap_head_conv, num_classes=num_classes)
        self.heatmap_head = CenterNetHead(in_ch=hidden_dim, cfg=hm_cfg)

        # num_queries is needed during forward to cap topK from heatmap
        self._num_queries: int = ecdet.decoder.num_queries

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        backbone_feats = self.ecdet.backbone(x)              # [S8, S16, S32]
        feats          = self.ecdet.encoder(backbone_feats)  # [S4, S8, S16, S32]

        # Run CenterNet heatmap head on S4 feature.
        hm_out: CenterNetOutput = self.heatmap_head(feats[0])

        # Extract top-K peaks → reference points for ECTransformer queries.
        # topk_idx is the flat S4 position index (used to gather memory content).
        heatmap_ref_points, heatmap_topk_idx = self._peaks_from_heatmap(hm_out)

        ec_targets = self._format_targets(targets, x.device)
        ec_out = self.ecdet.decoder(
            feats, ec_targets,
            heatmap_ref_points=heatmap_ref_points,
            heatmap_topk_idx=heatmap_topk_idx,
        )

        detr_out = self._wrap_ec(ec_out)
        detr_out.heatmap_out = hm_out
        return {'stage1': hm_out, 'stage2': detr_out}

    @torch.no_grad()
    def _peaks_from_heatmap(self, hm_out: CenterNetOutput):
        """
        Extract top-K peaks from heatmap for ECTransformer query init.

        Both ref_points and topk_idx are derived from the SAME topk call so
        their ordering is guaranteed to match — the i-th ref_point corresponds
        to the i-th index in topk_idx.

        Returns:
            ref_points: (B, K, 4) inverse-sigmoid — passed as heatmap_ref_points.
            topk_idx:   (B, K) flat S4 indices    — used to gather memory content.
        """
        hm, wh, reg = hm_out.hm, hm_out.wh, hm_out.reg
        B, C, H, W = hm.shape
        num_q = self._num_queries

        # NMS: suppress non-maxima in 3×3 neighbourhood
        hm_max = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        hm_nms = (hm_max == hm).float() * hm

        scores, _ = hm_nms.max(dim=1)          # (B, H, W) class-agnostic
        scores_flat = scores.flatten(1)         # (B, H*W)

        K = min(num_q, H * W)
        # sorted=True ensures ref_points[i] ↔ topk_idx[i] everywhere
        _, topk_idx = scores_flat.topk(K, dim=1, sorted=True)   # (B, K)

        topk_y = (topk_idx // W).float()
        topk_x = (topk_idx  % W).float()

        def _gather2(feat2d):
            flat = feat2d.permute(0, 2, 3, 1).reshape(B, H * W, 2)
            return flat.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))

        topk_reg = _gather2(reg)   # (B, K, 2)
        topk_wh  = _gather2(wh)    # (B, K, 2)

        cx = ((topk_x + topk_reg[..., 0]) / W).clamp(1e-4, 1 - 1e-4)
        cy = ((topk_y + topk_reg[..., 1]) / H).clamp(1e-4, 1 - 1e-4)
        bw = (topk_wh[..., 0] / W).clamp(1e-4, 1.0)
        bh = (topk_wh[..., 1] / H).clamp(1e-4, 1.0)
        ref = torch.stack([cx, cy, bw, bh], dim=-1)  # (B, K, 4)

        # Pad to exactly num_queries if feature map is smaller than num_queries
        if K < num_q:
            ref      = torch.cat([ref,      ref[:, :1, :].expand(-1, num_q - K, -1)], dim=1)
            topk_idx = torch.cat([topk_idx, topk_idx[:, :1].expand(-1, num_q - K)],   dim=1)

        ref_points = torch.log(ref / (1.0 - ref + 1e-6))   # inverse sigmoid
        return ref_points, topk_idx

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _format_targets(
        self,
        targets: Optional[List[Dict[str, Any]]],
        device: torch.device,
    ) -> Optional[List[Dict[str, Any]]]:
        if targets is None:
            return None
        return [
            {
                'boxes':  t['boxes'].to(device),
                'labels': (t['labels'][:, 0].long() if t['labels'].ndim == 2
                           else t['labels'].long()).to(device),
            }
            for t in targets
        ]

    def _wrap_ec(self, ec_out: Dict[str, Any]) -> DETROutput:
        pred_boxes  = ec_out['pred_boxes']
        pred_logits = ec_out['pred_logits']
        aux = ec_out.get('aux_outputs', [])

        if aux:
            boxes_all  = torch.stack([a['pred_boxes']  for a in aux] + [pred_boxes])
            logits_all = torch.stack([a['pred_logits'] for a in aux] + [pred_logits])
        else:
            boxes_all  = pred_boxes.unsqueeze(0)
            logits_all = pred_logits.unsqueeze(0)

        reid = None
        if self.reid_mlp is not None and 'hs' in ec_out:
            reid = F.normalize(self.reid_mlp(ec_out['hs']), dim=-1)

        return DETROutput(
            boxes           = pred_boxes,
            logits          = pred_logits,
            reid            = reid,
            boxes_all       = boxes_all,
            logits_all      = logits_all,
            dn_outputs      = ec_out.get('dn_outputs',      None),
            dn_meta         = ec_out.get('dn_meta',         None),
            enc_aux_outputs = ec_out.get('enc_aux_outputs', None),
            heatmap_out     = None,  # populated by HybridECDet.forward() after wrapping
        )

    def load_pretrained(self, path: str) -> None:
        """Load pretrained ECDet weights into self.ecdet with full mismatch handling.

        Handles three categories of incompatibility:
        1. Class-count heads: shape differs when num_classes changes → skipped.
        2. Anchor buffers: shape depends on eval_spatial_size → skipped, then
           re-generated from the model's own eval_spatial_size.
        3. Unexpected keys (extra keys in checkpoint not in current model): ignored.
        4. Missing keys (new modules like s4_upsample): kept at random init.
        """
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict) and any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}

        model_state = self.ecdet.state_dict()

        compatible: dict   = {}
        skipped_shape: list = []
        skipped_extra: list = []

        for k, v in state.items():
            if k not in model_state:
                skipped_extra.append(k)
                continue
            if v.shape != model_state[k].shape:
                skipped_shape.append(
                    (k, tuple(v.shape), tuple(model_state[k].shape))
                )
                continue
            compatible[k] = v

        missing, _ = self.ecdet.load_state_dict(compatible, strict=False)

        n_ckpt  = len(state)
        n_ok    = len(compatible)
        n_shape = len(skipped_shape)
        n_extra = len(skipped_extra)

        print(
            f'[HybridECDet] pretrained loaded: {n_ok}/{n_ckpt} tensors '
            f'({n_shape} shape-mismatch skipped, {n_extra} unexpected skipped)'
            f'\n  from: {path}'
        )
        if skipped_shape:
            print(f'  shape-mismatch ({n_shape}) — kept random init:')
            for k, cs, ms in skipped_shape:
                print(f'    {k}: ckpt {cs} vs model {ms}')
        if skipped_extra:
            print(f'  unexpected in ckpt ({n_extra}): {skipped_extra[:4]}'
                  f'{"..." if n_extra > 4 else ""}')

        dec = self.ecdet.decoder
        if (
            hasattr(dec, '_generate_anchors')
            and getattr(dec, 'eval_spatial_size', None) is not None
            and any(k in ('decoder.anchors', 'decoder.valid_mask')
                    for k, _, _ in skipped_shape)
        ):
            anchors, valid_mask = dec._generate_anchors()
            dec.register_buffer('anchors',    anchors)
            dec.register_buffer('valid_mask', valid_mask)
            h, w = dec.eval_spatial_size
            n_anchors = anchors.shape[1]
            print(
                f'  anchors re-generated for eval_spatial_size={[h, w]}: '
                f'{n_anchors} positions'
            )

    def deploy(self) -> 'HybridECDet':
        self.eval()
        if hasattr(self.ecdet, 'deploy'):
            self.ecdet.deploy()
        return self
