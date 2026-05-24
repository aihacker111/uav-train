"""
HybridECDet: ECDet-S backbone/encoder/decoder + CenterNet auxiliary head.

Architecture:
    image
      ├─ ecdet.backbone → [f1_raw(S8,C_b), f2_raw(S16,C_b), f3_raw(S32,C_b)]
      │       └─ ecdet.encoder(backbone_feats) → [f1(S8,D), f2(S16,D), f3(S32,D)]
      │                                           (D = hidden_dim, e.g. 256)
      │
      ├─ cn_backbone_proj(f1_raw) → cn_upsample → CenterNetHead → hm, wh, reg  [Stage-1]
      │       top-K heatmap peaks + predicted wh  ←── object proposals
      │               ↓
      └─ ecdet.decoder(encoder_feats, ref_points=peaks) → refined pred_boxes, pred_logits

Gradient flow:
  Stage-1 loss  → cn_head → cn_upsample → cn_backbone_proj → backbone
  Stage-2 loss  → decoder → encoder → backbone
  Both stages share the backbone: it receives gradient from CenterNet (spatial
  precision) and DETR (semantic discrimination) simultaneously, learning features
  useful for both dense and sparse detection.
  The encoder is updated exclusively by Stage-2 → free to specialise for DETR.

  Heatmap peaks are detached before ref-point conversion so Stage-2 does NOT
  backprop through peak extraction.

Coordinate system:
  - CenterNet wh/reg GT: S4 pixel units (output_h × output_w)
  - DETR boxes GT: normalised cxcywh in [0, 1]
  - Heatmap peaks → ref_points cx/cy: (x_idx + 0.5) / W_s4  ∈ (0, 1)
  - Predicted wh → ref_points w/h:   wh_s4_pixels / W_s4    ∈ (0, 1)
    (wh in S4 pixel units; dividing by W_s4 gives the same [0,1] normalisation
     as cx because stride-4 cancels: w_px_input / W_input = w_s4 / W_s4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, List, Optional

from .heads import CenterNetHead, CenterNetOutput, DETROutput
from .config import CenterNetHeadConfig


class HybridECDet(nn.Module):
    def __init__(
        self,
        ecdet: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        backbone_s8_channels: int = 192,
        cn_dim: int     = 128,
        head_conv: int  = 16,
        reid_dim: int   = 0,
        default_wh: float = 0.05,
    ) -> None:
        super().__init__()
        self.ecdet = ecdet
        self.default_wh = default_wh

        # Project backbone S8 channels (e.g. 192) → cn_dim.
        # cn_dim is intentionally smaller than hidden_dim: the CenterNet branch is
        # an auxiliary head and does not need the full encoder feature width.
        # Keeping cn_dim small (default 64) reduces params and S4-resolution FLOPs
        # by ~18× compared to using hidden_dim=256 throughout.
        self.cn_backbone_proj = nn.Sequential(
            nn.Conv2d(backbone_s8_channels, cn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cn_dim),
            nn.ReLU(inplace=True),
        )

        # S8 → S4 upsample for CenterNet branch
        self.cn_upsample = nn.Sequential(
            nn.ConvTranspose2d(cn_dim, cn_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(cn_dim),
            nn.ReLU(inplace=True),
        )
        self.cn_head = CenterNetHead(
            cn_dim,
            CenterNetHeadConfig(head_conv=head_conv, num_classes=num_classes),
        )

        self.reid_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        ) if reid_dim > 0 else None

    # ── Heatmap → reference points ──────────────────────────────────────────────

    def _heatmap_to_ref_points(
        self, hm: Tensor, wh_map: Optional[Tensor], reg_map: Optional[Tensor], k: int
    ) -> Tensor:
        """
        Convert top-K CenterNet heatmap peaks to ECTransformer reference points
        in logit (inverse-sigmoid) space.

        Args:
            hm      : (B, C, H, W) — sigmoid heatmap at S4 resolution, values in (0, 1)
            wh_map  : (B, 2, H, W) — Stage-1 predicted wh in S4 pixel units, or None
                      for fallback to default_wh.  Must be detached before calling.
            reg_map : (B, 2, H, W) — sub-pixel centre offset predicted by Stage-1 reg
                      head.  GT convention: reg = actual_S4_pos - floor(actual_S4_pos),
                      so reg ∈ [0, 1).  Adding reg_x to x_idx gives the precise S4
                      position without the +0.5 grid-centre assumption.  None falls
                      back to the grid-centre approximation (+0.5).
            k       : number of reference points (= num_queries, typically 300)

        Returns:
            (B, k, 4) — cxcywh reference points in logit space.
            cx/cy: (x_s4 + reg_x) / W_s4  ∈ (0, 1) — sub-pixel corrected centre
            w/h  : wh_s4_pixels / W_s4    ∈ (0, 1) — same normalisation as cx/cy
        """
        B, C, H, W = hm.shape

        # Max across classes per pixel → global top-K (avoids one class monopolising all K slots)
        score_map = hm.max(dim=1).values          # (B, H, W)
        flat      = score_map.reshape(B, H * W)   # (B, H*W)
        _, topk_idx = flat.topk(k, dim=-1)        # (B, k)

        y_idx = (topk_idx // W).float()
        x_idx = (topk_idx  % W).float()

        if reg_map is not None:
            # GT reg = actual_S4_pos - floor(actual_S4_pos) ∈ [0, 1).
            # x_idx is the floor → x_idx + reg_x = precise S4 position.
            # Clamp to valid grid range before normalising.
            flat_rx = reg_map[:, 0].reshape(B, H * W)
            flat_ry = reg_map[:, 1].reshape(B, H * W)
            reg_x   = flat_rx.gather(1, topk_idx)   # (B, k)
            reg_y   = flat_ry.gather(1, topk_idx)
            cx = (x_idx + reg_x).clamp(0, W - 1) / W
            cy = (y_idx + reg_y).clamp(0, H - 1) / H
        else:
            # Fallback: assume object centre is at grid-cell centre (+0.5).
            cx = (x_idx + 0.5) / W
            cy = (y_idx + 0.5) / H

        if wh_map is not None:
            # Use Stage-1 predicted wh (S4 pixel units) normalised to [0, 1].
            # w_norm = w_s4 / W_s4 — identical convention to cx above.
            # Clamp to [0.02, 0.5]: avoids degenerate zero-size boxes at epoch-0
            # and prevents outliers from pulling decoder attention off target.
            flat_w = wh_map[:, 0].reshape(B, H * W)   # (B, H*W)
            flat_h = wh_map[:, 1].reshape(B, H * W)
            pred_w = (flat_w.gather(1, topk_idx) / W).clamp(0.02, 0.5)
            pred_h = (flat_h.gather(1, topk_idx) / H).clamp(0.02, 0.5)
        else:
            pred_w = torch.full_like(cx, self.default_wh)
            pred_h = torch.full_like(cx, self.default_wh)

        boxes = torch.stack([cx, cy, pred_w, pred_h], dim=-1).clamp(1e-4, 1.0 - 1e-4)
        return torch.log(boxes / (1.0 - boxes))   # (B, k, 4) logit space

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # ── Shared backbone (gradient from BOTH Stage-1 and Stage-2) ────────────
        backbone_feats = self.ecdet.backbone(x)   # [S8(C_b), S16(C_b), S32(C_b)]

        # ── Encoder — Stage-2 exclusive (DETR-semantic features) ────────────────
        # Encoder gradient comes only from Stage-2 loss → free to specialise for
        # sparse DETR detection without Stage-1 interference.
        feats = self.ecdet.encoder(backbone_feats)   # [S8(D), S16(D), S32(D)]

        # ── Stage-1: CenterNet on pre-encoder backbone S8 features ──────────────
        # backbone_feats[0] carries fine-grained spatial detail that the encoder
        # would otherwise wash out with DETR-oriented semantics.
        # No detach: Stage-1 loss flows backbone_feats[0] → backbone, complementing
        # Stage-2's semantic gradient with a spatial-precision signal.
        cn_feat = self.cn_upsample(self.cn_backbone_proj(backbone_feats[0]))
        cn_out  = self.cn_head(cn_feat)

        # ── Heatmap peaks → reference points ────────────────────────────────────
        # Detach hm, wh, reg: Stage-2 loss must NOT backprop through peak extraction
        # or Stage-1 predictions — the two loss paths remain cleanly separated.
        # reg_map corrects cx/cy from grid-centre to true sub-pixel object centre,
        # fixing the asymmetric box fitting ("lẹm/to ra") caused by centre offset.
        num_queries = self.ecdet.decoder.num_queries
        heatmap_ref = self._heatmap_to_ref_points(
            cn_out.hm.detach(),
            cn_out.wh.detach(),
            cn_out.reg.detach(),
            k=num_queries,
        )   # (B, num_queries, 4) logit-space cxcywh

        # ── Stage-2: ECTransformer on encoder features, seeded with heatmap refs ─
        ec_targets = self._format_targets(targets, x.device)
        ec_out = self.ecdet.decoder(
            feats,
            ec_targets,
            heatmap_ref_points=heatmap_ref,
        )

        return {'stage1': cn_out, 'stage2': self._wrap_ec(ec_out)}

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
        """Map ECTransformer output dict → DETROutput (compatible with HybridLoss)."""
        pred_boxes  = ec_out['pred_boxes']    # (B, K, 4) cxcywh [0,1]
        pred_logits = ec_out['pred_logits']   # (B, K, C)
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
            boxes      = pred_boxes,
            logits     = pred_logits,
            reid       = reid,
            boxes_all  = boxes_all,
            logits_all = logits_all,
            dn_outputs = ec_out.get('dn_outputs', None),
            dn_meta    = ec_out.get('dn_meta',    None),
        )

    def load_pretrained(self, path: str) -> None:
        """Load pretrained ECDet weights into self.ecdet with full mismatch handling.

        Handles three categories of incompatibility between a COCO/Objects365
        pretrained checkpoint and a fine-tuned model:

        1. Class-count heads (enc_score_head, dec_score_head, denoising_class_embed):
           Shape differs when num_classes changes (e.g. 80 → 10). These layers are
           *skipped* so they keep their random init and are trained from scratch.

        2. Anchor buffers (anchors, valid_mask):
           Shape depends on eval_spatial_size. Pretrained: 640×640 → 8400 positions.
           Current model may use a different input size (e.g. 512×832 → 8736).
           Buffers are *skipped* during load, then immediately re-generated from the
           model's own eval_spatial_size so inference/training remains consistent.

        3. Unexpected keys (extra keys in checkpoint not in current model): ignored.
        """
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict) and any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}

        model_state = self.ecdet.state_dict()

        # ── Classify every checkpoint key ─────────────────────────────────────
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

        # ── Load only the compatible subset ───────────────────────────────────
        # strict=False: the keys absent from `compatible` (skipped_shape / new
        # model layers) are reported as "missing" — that is expected and fine.
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

        # ── Re-register anchor buffers with current eval_spatial_size ─────────
        # anchors / valid_mask are derived from eval_spatial_size; if they were
        # skipped above (different input resolution), we must regenerate them so
        # the decoder uses the correct grid for the current image size.
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