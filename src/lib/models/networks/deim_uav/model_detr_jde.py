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

import math
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
      - One content query : projected encoder token at that position
      - One reference box : grid-centre CX/CY (fixed) + learned WH (per cell,
                            predicted from encoder features via wh_head)

    WH is NOT hardcoded.  wh_head is a lightweight Linear that predicts
    log(w) and log(h) per token.  Its bias is initialised so exp(bias) equals a
    sensible starting point (0.025 for S16, 0.05 for S32), after which
    gradient from the detection loss (Hungarian GIoU + bbox) refines it toward
    the dataset's actual object-size distribution.

    Gradient path:
        L_det → pre_bboxes → ref_points_unact → _logit(wh) → wh_head weights

    Args:
        hidden_dim  : encoder + decoder hidden dimension
        grid_strides: encoder output strides to use (default: [16, 32])
        init_wh_s16 : initial WH for S16 in normalised coords (default 0.025 ≈ 16px on 640px);
                      S32 is initialised to init_wh_s16 × 2
    """

    def __init__(
        self,
        hidden_dim: int,
        grid_strides: Tuple[int, ...] = (16, 32),
        init_wh_s16: float = 0.025,
    ) -> None:
        super().__init__()
        self.grid_strides = grid_strides

        # Content query projection — one Linear per stride
        self.proj = nn.ModuleDict({
            str(s): nn.Linear(hidden_dim, hidden_dim, bias=False)
            for s in grid_strides
        })
        for p in self.proj.values():
            nn.init.xavier_uniform_(p.weight)

        # WH prediction head — one Linear per stride (equivalent to 1×1 Conv2d
        # but operates on gathered tokens directly, avoiding full-map computation).
        # Weight=0 so initial prediction equals exactly exp(bias).
        self.wh_head = nn.ModuleDict({
            str(s): nn.Linear(hidden_dim, 2, bias=True)
            for s in grid_strides
        })
        for s, head in self.wh_head.items():
            init_wh = init_wh_s16 * int(s) / 16.0      # S16→0.025, S32→0.05
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(init_wh))

    @staticmethod
    def _logit(x: Tensor) -> Tensor:
        x = x.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(x / (1.0 - x))

    def forward_selected(
        self,
        enc_feats: List[Tensor],
        s16_idx: Tensor,
        stride_map: Optional[Dict[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Efficient selective forward: compute queries ONLY for chosen positions.

        For S16 strides: content and WH are computed at the K selected positions
        only — no intermediate (B, N_s16, C) tensor is ever materialised.
        For other strides (S32): all cells are computed as usual (small count).

        Memory comparison vs forward() then gather:
            forward()        : proj(B, 1600, C) + gather → (B, K, C)
            forward_selected : gather tokens → proj(B, K, C)   [K < 1600]

        Args:
            enc_feats : list of (B, C, H_i, W_i) encoder feature maps [S8, S16, S32]
            s16_idx   : (B, K) flat indices into the S16 feature map (row-major)
            stride_map: optional stride→enc_feats-index mapping

        Returns:
            content : (B, K + N_s32, C)
            ref_pts : (B, K + N_s32, 4)  logit-space cxcywh
        """
        if stride_map is None:
            stride_map = {8: 0, 16: 1, 32: 2}

        all_content: List[Tensor] = []
        all_ref:     List[Tensor] = []

        for stride in self.grid_strides:
            feat = enc_feats[stride_map[stride]]   # (B, C, H, W)
            B, C, H, W = feat.shape

            ys = (torch.arange(H, device=feat.device, dtype=torch.float32) + 0.5) / H
            xs = (torch.arange(W, device=feat.device, dtype=torch.float32) + 0.5) / W
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)

            if stride == 16:
                # ── S16: efficient path — only K selected positions ────────────
                cx_all = gx.flatten()                              # (H*W,)
                cy_all = gy.flatten()                              # (H*W,)
                cx = cx_all[s16_idx].unsqueeze(-1)                 # (B, K, 1)
                cy = cy_all[s16_idx].unsqueeze(-1)                 # (B, K, 1)

                # Gather K encoder tokens → proj (content) + wh_head (WH)
                feat_flat  = feat.flatten(2)                               # (B, C, H*W)
                idx        = s16_idx.unsqueeze(1).expand(-1, C, -1)       # (B, C, K)
                tokens_sel = feat_flat.gather(2, idx).permute(0, 2, 1)    # (B, K, C)
                content    = self.proj[str(stride)](tokens_sel)            # (B, K, C)
                wh         = self.wh_head[str(stride)](tokens_sel)\
                                 .exp().clamp(0.004, 0.6)                  # (B, K, 2)

            else:
                # ── S32 (and any other stride): full path — small cell count ───
                tokens  = feat.flatten(2).permute(0, 2, 1)                # (B, H*W, C)
                content = self.proj[str(stride)](tokens)                   # (B, H*W, C)
                wh      = self.wh_head[str(stride)](tokens)\
                              .exp().clamp(0.004, 0.6)                     # (B, H*W, 2)
                cx = gx.flatten().view(1, -1, 1).expand(B, -1, 1)
                cy = gy.flatten().view(1, -1, 1).expand(B, -1, 1)

            cxcywh  = torch.cat([cx, cy, wh], dim=-1)
            ref_pts = self._logit(cxcywh)

            all_content.append(content)
            all_ref.append(ref_pts)

        return torch.cat(all_content, dim=1), torch.cat(all_ref, dim=1)


# ── Objectness head ───────────────────────────────────────────────────────────

class ObjectnessHead(nn.Module):
    """
    Lightweight 2-conv head predicting per-cell objectness on S16 encoder features.
    Output is sigmoid-normalised → soft score in (0, 1).
    Bias initialised to sigmoid(−4.595) ≈ 0.01 so the model starts near the
    true prior (~1% of cells contain objects in UAV scenes).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        mid = max(hidden_dim // 2, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, mid, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, mid), mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
        )
        nn.init.kaiming_normal_(self.conv[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv[3].bias, -4.595)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x).sigmoid()   # (B, 1, H, W)


# ── Gaussian label generator ───────────────────────────────────────────────────

def gaussian_objectness_labels(
    targets: List[Dict[str, Any]],
    H: int,
    W: int,
    sigma: float = 1.5,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Build size-normalised Gaussian soft labels for the objectness head.

    For each GT box (cx, cy, w, h) in normalised [0,1] coords, the label at
    cell (i,j) is:

        label[i,j] = max over GT of  exp(−d_norm² / 2σ²)

    where d_norm = dist(cell_centre, GT_centre) / (sqrt(w·h) / 2).

    Scale-invariant: a 5-px object and a 100-px object both produce peak=1.0
    at their nearest cell with identical spatial fall-off shape.

    Returns (B, 1, H, W) float32 in [0, 1].
    """
    B      = len(targets)
    device = device or (targets[0]['boxes'].device if B > 0 else torch.device('cpu'))
    labels = torch.zeros(B, 1, H, W, device=device)

    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H   # (H,)
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W   # (W,)
    cy_g, cx_g = torch.meshgrid(ys, xs, indexing='ij')                      # (H, W)

    # minimum radius: 2 pixels in normalised coords so very tiny objects still
    # activate at least their nearest cell
    min_radius = 2.0 / max(H, W)

    for b, t in enumerate(targets):
        boxes = t['boxes'].to(device)   # (N, 4) cxcywh [0,1]
        if boxes.shape[0] == 0:
            continue
        cx, cy = boxes[:, 0], boxes[:, 1]
        w,  h  = boxes[:, 2], boxes[:, 3]
        radius = (w * h).sqrt() / 2.0                 # (N,)
        radius = radius.clamp(min=min_radius)

        # (N, H, W) — vectorised over all GT boxes
        dx     = cx_g.unsqueeze(0) - cx.view(-1, 1, 1)
        dy     = cy_g.unsqueeze(0) - cy.view(-1, 1, 1)
        d_norm = (dx**2 + dy**2).sqrt() / radius.view(-1, 1, 1)
        gauss  = (-d_norm**2 / (2.0 * sigma**2)).exp()   # (N, H, W)

        labels[b, 0] = gauss.max(dim=0).values

    return labels


# ── DEIMv2JDE ─────────────────────────────────────────────────────────────────

class DEIMv2JDE(nn.Module):
    """
    Full DEIMv2 DETR model with grid-based queries, per-query ReID, and
    objectness-guided query selection.

    Args:
        deim        : full DEIM(backbone, encoder, decoder) object
        num_classes : number of object categories
        hidden_dim  : encoder/decoder hidden dimension
        reid_dim    : ReID embedding dimension
        grid_strides      : encoder stride levels for grid queries (default (16, 32))
min_queries       : minimum total query count guaranteed at inference (default 500)
        train_k_headroom  : K = max_GT_per_image × headroom at training (default 2.5)
        min_train_k       : minimum S16 queries at training even for sparse frames (default 200)
    """

    def __init__(
        self,
        deim: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        reid_dim: int = 128,
        grid_strides: Tuple[int, ...] = (16, 32),
        min_queries: int = 500,
        train_k_headroom: float = 2.5,
        min_train_k: int = 200,
    ) -> None:
        super().__init__()
        self.deim             = deim
        self.min_queries      = min_queries
        self.train_k_headroom = train_k_headroom
        self.min_train_k      = min_train_k

        self.grid_qgen = GridQueryGen(
            hidden_dim=hidden_dim,
            grid_strides=grid_strides,
        )

        # Objectness head on S16 encoder features (index 1 in enc_feats)
        self.obj_head = ObjectnessHead(hidden_dim)

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

    def _adaptive_train_k(
        self,
        targets: Optional[List[Dict[str, Any]]],
        N_s16: int,
    ) -> int:
        """
        Compute per-batch S16 query count from GT object density.

        K = max_GT_objects_in_batch × train_k_headroom
        clamped to [min_train_k, N_s16].

        Using the batch maximum (not average) ensures the densest image in the
        batch always has enough queries for full recall.  Sparse images in the
        same batch get "extra" queries — these are simply unmatched background
        context, which the decoder tolerates well.

        Examples (headroom=2.5, min_train_k=200, N_s16=1600):
          50  GT objects  →  K = 125  → clamped to 200  (sparse frame)
          200 GT objects  →  K = 500                     (typical VisDrone)
          500 GT objects  →  K = 1250                    (dense frame)
          700 GT objects  →  K = 1600 (capped at N_s16)  (very dense)
        """
        if not targets:
            return self.min_train_k
        max_gt = max(t['boxes'].shape[0] for t in targets)
        k = int(max_gt * self.train_k_headroom)
        return max(self.min_train_k, min(N_s16, k))

    # ── unified query builder ──────────────────────────────────────────────────

    def _make_queries(
        self,
        enc_feats: List[Tensor],
        obj_scores: Tensor,
        targets: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute adaptive K, select top-K S16 indices by objectness score,
        then call grid_qgen.forward_selected() — queries are computed ONLY
        for the K chosen positions, not for all 1600 then filtered.

        K logic:
          Training  (targets given): K = max_GT_in_batch × headroom
          Inference (targets None) : K = count(score > 0.05)
          Both are clamped to [min_s16, N_s16].
        """
        N_s16   = enc_feats[1].shape[2] * enc_feats[1].shape[3]
        N_s32   = (enc_feats[2].shape[2] * enc_feats[2].shape[3]
                   if 32 in self.grid_qgen.grid_strides else 0)
        min_s16 = max(1, self.min_queries - N_s32)

        scores_flat = obj_scores.flatten(2).squeeze(1)   # (B, N_s16)

        if targets is not None:
            # Training: density-adaptive fixed budget
            K = self._adaptive_train_k(targets, N_s16)
        else:
            # Inference: score-threshold adaptive
            above = int((scores_flat > 0.05).sum(dim=1).max().item())
            K = max(min_s16, min(N_s16, above))

        s16_idx = scores_flat.topk(K, dim=-1).indices   # (B, K)
        return self.grid_qgen.forward_selected(enc_feats, s16_idx)

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
              pred_boxes, pred_logits, pred_reid, obj_scores, hs,
              aux_outputs, enc_aux_outputs, pre_outputs, enc_meta,
              (optionally dn_outputs, dn_pre_outputs, dn_meta)
            Inference:
              pred_boxes, pred_logits, pred_reid, obj_scores, hs
        """
        feats     = self.deim.backbone(x)
        enc_feats = self.deim.encoder(feats)

        # Objectness scores — always computed (used by loss + query selection)
        obj_scores = self.obj_head(enc_feats[1])   # (B, 1, H/16, W/16)

        # Select top-K positions and generate queries only there (no waste)
        content, ref_pts = self._make_queries(
            enc_feats, obj_scores,
            targets if self.training else None,
        )

        deim_targets = self._format_targets(targets, x.device)
        out = self.deim.decoder(
            enc_feats,
            deim_targets,
            heatmap_ref_points=ref_pts,
            query_content=content,
        )

        hs = out['hs']
        out['pred_reid']  = F.normalize(self.reid_mlp(hs), dim=-1)
        out['obj_scores'] = obj_scores

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
