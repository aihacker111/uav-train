"""
DEIMv2JDE — Full DETR tracking model with grid-based queries and per-query ReID.

Architecture
------------
Input
  │
  ▼  [PRETRAINED]
DINOv3STAs Backbone → [f_S8, f_S16, f_S32]
  │
  ▼  [PRETRAINED]
HybridEncoder → [enc_S8, enc_S16, enc_S32]  (spatial feature maps, hidden_dim ch each)
  │
  ├── GridQueryGen (NEW)
  │     S16 features (40×40 for 640 input) + S32 features (20×20)
  │     → content queries  (B, N_grid, C)   — projected encoder tokens
  │     → ref_points_logit (B, N_grid, 4)   — grid center cxcywh in logit space
  │
  ▼  [PRETRAINED]
DEIMTransformer.decoder (grid-query injection mode)
  → hs: (B, N_grid, hidden_dim)  last-layer decoder hidden states
  │
  ├── dec_bbox_head → pred_boxes  (B, N_grid, 4)  cxcywh [0,1]
  ├── dec_score_head → pred_logits (B, N_grid, C)  raw logits
  └── reid_mlp (NEW) → pred_reid  (B, N_grid, reid_dim)  L2-normalised

Query count:
  N_grid = H/16 × W/16  +  H/32 × W/32
  For 640×640: 40×40 + 20×20 = 1600 + 400 = 2000 queries
  For 800×608: 50×38 + 25×19 = 1900+475 = 2375 queries
  → Scales naturally with input resolution; no fixed cap.

Tracking interface (inference):
  pred_boxes, pred_logits, pred_reid  →  MCJDETracker  (Kalman + ReID cosine)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple


# ── Grid query generator ───────────────────────────────────────────────────────

class GridQueryGen(nn.Module):
    """
    Build content queries and reference-point boxes from encoder feature maps.

    For each requested stride level the H/s × W/s spatial positions form a
    uniform grid.  Each grid cell produces:
      - One content query  : projected encoder token at that position
      - One reference box  : grid-centre cxcywh in logit (inverse-sigmoid) space,
                             with scale-appropriate default WH

    Args:
        hidden_dim    : encoder + decoder hidden dimension
        grid_strides  : encoder output strides to use (default: [16, 32])
        default_wh_s16: default box half-size for stride-16 level (normalised)
    """

    def __init__(
        self,
        hidden_dim: int,
        grid_strides: Tuple[int, ...] = (16, 32),
        default_wh_s16: float = 0.08,
    ) -> None:
        super().__init__()
        self.grid_strides    = grid_strides
        self.default_wh_s16  = default_wh_s16

        # One linear proj per stride level
        self.proj = nn.ModuleDict({
            str(s): nn.Linear(hidden_dim, hidden_dim, bias=False)
            for s in grid_strides
        })
        for p in self.proj.values():
            nn.init.xavier_uniform_(p.weight)

    @staticmethod
    def _logit(x: Tensor) -> Tensor:
        x = x.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(x / (1.0 - x))

    def forward(
        self,
        enc_feats: List[Tensor],
        stride_map: Optional[Dict[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            enc_feats : list of (B, C, H_i, W_i) encoder spatial feature maps,
                        ordered [S8, S16, S32] (indices 0, 1, 2).
            stride_map: optional override mapping stride → enc_feats index.
                        Defaults to {8:0, 16:1, 32:2}.

        Returns:
            content   : (B, N_grid, C)  projected content queries
            ref_pts   : (B, N_grid, 4)  cxcywh reference boxes in logit space
        """
        if stride_map is None:
            stride_map = {8: 0, 16: 1, 32: 2}

        all_content: List[Tensor] = []
        all_ref: List[Tensor]    = []

        for stride in self.grid_strides:
            feat = enc_feats[stride_map[stride]]   # (B, C, H, W)
            B, C, H, W = feat.shape

            # ── content queries ─────────────────────────────────────────────
            tokens  = feat.flatten(2).permute(0, 2, 1)       # (B, H*W, C)
            content = self.proj[str(stride)](tokens)           # (B, H*W, C)

            # ── reference boxes ─────────────────────────────────────────────
            ys = (torch.arange(H, device=feat.device, dtype=torch.float32) + 0.5) / H
            xs = (torch.arange(W, device=feat.device, dtype=torch.float32) + 0.5) / W
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')    # (H, W)

            # WH scales with stride: S16→0.08, S32→0.16
            wh_val = self.default_wh_s16 * (stride / 16.0)
            wh = torch.full((H * W, 2), wh_val, device=feat.device)

            cxcywh = torch.cat([
                gx.flatten().unsqueeze(-1),
                gy.flatten().unsqueeze(-1),
                wh,
            ], dim=-1)                                         # (H*W, 4)
            ref_pts = self._logit(cxcywh)                      # logit space
            ref_pts = ref_pts.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 4)

            all_content.append(content)
            all_ref.append(ref_pts)

        return torch.cat(all_content, dim=1), torch.cat(all_ref, dim=1)


# ── DEIMv2JDE ─────────────────────────────────────────────────────────────────

class DEIMv2JDE(nn.Module):
    """
    Full DEIMv2 DETR model with grid-based queries and per-query ReID head.

    Args:
        deim       : full DEIM(backbone, encoder, decoder) object
        num_classes: number of object categories
        hidden_dim : encoder/decoder hidden dimension
        reid_dim   : ReID embedding dimension
        grid_strides: encoder stride levels to use for grid queries (default (16,32))
    """

    def __init__(
        self,
        deim: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        reid_dim: int = 128,
        grid_strides: Tuple[int, ...] = (16, 32),
    ) -> None:
        super().__init__()
        self.deim = deim

        self.grid_qgen = GridQueryGen(
            hidden_dim=hidden_dim,
            grid_strides=grid_strides,
        )

        # ReID head: 2-layer MLP on last-layer decoder tokens
        self.reid_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, reid_dim),
        )
        nn.init.xavier_uniform_(self.reid_mlp[0].weight)
        nn.init.xavier_uniform_(self.reid_mlp[2].weight)
        nn.init.zeros_(self.reid_mlp[0].bias)
        nn.init.zeros_(self.reid_mlp[2].bias)

        self._num_classes = num_classes

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_targets(
        targets: Optional[List[Dict[str, Any]]],
        device: torch.device,
    ) -> Optional[List[Dict[str, Tensor]]]:
        """Convert trainer batch targets to DEIM decoder format."""
        if targets is None:
            return None
        out = []
        for t in targets:
            d: Dict[str, Tensor] = {
                'boxes':  t['boxes'].to(device),
            }
            lbl = t['labels']
            d['labels'] = (lbl[:, 0].long() if lbl.ndim == 2 else lbl.long()).to(device)
            out.append(d)
        return out

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            x       : (B, 3, H, W) input images
            targets : list of GT dicts with 'boxes' (cxcywh [0,1]) and 'labels'

        Returns dict:
            Training:
              pred_boxes, pred_logits, pred_reid, hs,
              aux_outputs, enc_aux_outputs, pre_outputs, enc_meta,
              (optionally dn_outputs, dn_pre_outputs, dn_meta)
            Inference:
              pred_boxes, pred_logits, pred_reid, hs
        """
        # ── backbone + encoder ─────────────────────────────────────────────────
        feats     = self.deim.backbone(x)
        enc_feats = self.deim.encoder(feats)
        # enc_feats: list of (B, hidden_dim, H/s, W/s) spatial tensors

        # ── grid queries ───────────────────────────────────────────────────────
        content, ref_pts = self.grid_qgen(enc_feats)
        # content : (B, N_grid, hidden_dim)
        # ref_pts : (B, N_grid, 4)  logit-space cxcywh

        # ── decoder (grid-query injection mode) ────────────────────────────────
        deim_targets = self._format_targets(targets, x.device)
        out = self.deim.decoder(
            enc_feats,
            deim_targets,
            heatmap_ref_points=ref_pts,
            query_content=content,
        )

        # ── ReID head ─────────────────────────────────────────────────────────
        # hs is (B, N_grid, hidden_dim): last-layer decoder states (after DN split)
        hs = out['hs']
        out['pred_reid'] = F.normalize(self.reid_mlp(hs), dim=-1)

        return out

    # ── weight loading ─────────────────────────────────────────────────────────

    def load_pretrained(self, path: str) -> None:
        """Load deimv2_dinov3_s pretrained weights into backbone+encoder+decoder."""
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('ema', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        missing, unexpected = self.deim.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(missing)
        print(f'[DEIMv2JDE] pretrained: {n_loaded}/{len(state)} tensors loaded from {path}')
        if missing:
            print(f'  missing ({len(missing)}): {missing[:5]}{"…" if len(missing) > 5 else ""}')
        if unexpected:
            print(f'  unexpected ({len(unexpected)}): {unexpected[:5]}{"…" if len(unexpected) > 5 else ""}')

    def deploy(self) -> 'DEIMv2JDE':
        self.eval()
        if hasattr(self.deim, 'deploy'):
            self.deim.deploy()
        return self
