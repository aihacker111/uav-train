"""
HybridDETR — main model composing all pipeline components.

Forward pass:
  image (B, 3, H, W)
    ↓  ViT backbone
  vit_features: list of (B, embed_dim, H/16, W/16)
    ↓  MultiScaleNeck
  MultiScaleNeckOutput   +   finest_s16 (B, D, H/16, W/16)
    ↓  TokenScorer
  ScorerOutput  (score_map: B×1×H/8×W/8,  feat_s8: B×D×H/8×W/8)
    ↓  QueryGenerator  [score_map + feat_s8]
  detect_bundle  (K dynamic queries)
    ↓  [training only] DNQueryGenerator → dn_bundle
  concat [detect_bundle ‖ dn_bundle] + attention mask
    ↓  HybridDecoder  [all_queries + neck memory]
  DecoderOutput  (hs, refs_logit)  split → detect / dn slices
    ↓  DETRHead  (shared weights)
  DETROutput  (boxes, logits, reid, …)

Returns dict:
  {
      'score_map':     (B, 1, H/8, W/8)   — objectness logit (post-sigmoid)
      'stage2':        DETROutput,
      'dn_out':        DETROutput | None,  — DN reconstruction output (train only)
      'dn_meta':       DNMeta | None,
      'query_scores':  (B, K),
      'query_classes': (B, K),
      'tau_query':     scalar tensor,
  }
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as grad_ckpt
from torch import Tensor

from .config import HybridModelConfig, ViTConfig
from .neck import MultiScaleNeck
from .heads import TokenScorer, DETRHead
from .query_gen import QueryGenerator, QueryBundle
from .dn_gen import DNQueryGenerator, DNMeta
from .decoder import HybridDecoder
from ..lwdetr_vit import ViT


# ── Backbone factory ───────────────────────────────────────────────────────────

def _build_vit(cfg: ViTConfig) -> ViT:
    return ViT(
        img_size               = 1024,
        patch_size             = 16,
        embed_dim              = cfg.embed_dim,
        depth                  = cfg.depth,
        num_heads              = 12,
        mlp_ratio              = 4.0,
        qkv_bias               = True,
        drop_path_rate         = 0.1,
        norm_layer             = partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos            = True,
        window_block_indexes   = cfg.window_block_indexes,
        pretrain_img_size      = 224,
        pretrain_use_cls_token = True,
        out_feature_indexes    = cfg.out_feature_indexes,
        use_cae                = True,
    )


# ── Main model ─────────────────────────────────────────────────────────────────

class HybridDETR(nn.Module):
    """
    Two-stage detector/tracker for AMOT on UAV imagery.

    Stage 1 — TokenScorer (stride-8):
      Lightweight objectness scorer with multi-scale s16+s8 score fusion.
      Rescues tiny objects (<10px) that would be invisible at stride-16 alone.

    Stage 2 — DETR decoder:
      Refines top-K scored locations through L layers of MSDeformAttn
      cross-attention over ViT features, producing accurate boxes, class
      logits, and ReID embeddings.

    DN training:
      num_dn_groups noisy copies of GT boxes/labels are appended to detect
      queries as additional decoder input.  A structured attention mask
      isolates DN groups from detect queries and from each other.
      DN reconstruction loss provides direct supervision without matching.
    """

    def __init__(self, cfg: HybridModelConfig) -> None:
        super().__init__()
        vit_cfg  = cfg.vit
        neck_cfg = cfg.neck
        dec_cfg  = cfg.decoder

        dec_cfg.num_feature_levels = neck_cfg.num_output_levels

        self.backbone     = _build_vit(vit_cfg)
        neck_in_ch        = vit_cfg.embed_dim * vit_cfg.num_feature_levels
        self.neck         = MultiScaleNeck(neck_in_ch, neck_cfg)
        self.token_scorer = TokenScorer(neck_cfg.hidden_dim, cfg.scorer)
        self.query_gen    = QueryGenerator(neck_cfg.hidden_dim, cfg.query_gen)
        self.decoder      = HybridDecoder(dec_cfg)
        self.detr_head    = DETRHead(cfg.detr, bbox_reparam=dec_cfg.bbox_reparam)

        self.dn_gen = DNQueryGenerator(
            hidden_dim  = neck_cfg.hidden_dim,
            num_classes = cfg.detr.num_classes,
            cfg         = cfg.dn,
        )

        self.grad_checkpoint = cfg.grad_checkpoint

    def forward(
        self,
        x:       Tensor,
        targets: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        # ── ViT backbone ──────────────────────────────────────────────────────
        if self.grad_checkpoint and self.training:
            vit_features: List[Tensor] = grad_ckpt.checkpoint(
                self.backbone, x, use_reentrant=False,
            )
        else:
            vit_features: List[Tensor] = self.backbone(x)

        # ── Multi-scale neck ──────────────────────────────────────────────────
        neck_out, finest_s16 = self.neck(vit_features)

        # ── TokenScorer (stride-8, multi-scale fusion) ────────────────────────
        scorer_out = self.token_scorer(finest_s16)
        # score_map: (B, 1, H/8, W/8)  feat_s8: (B, D, H/8, W/8)

        # ── Query generation: score peaks → DETR seeds ───────────────────────
        # feat_s8 is detached so Stage-2 gradients don't corrupt backbone features;
        # score_map is NOT detached so Gumbel-STE carries Stage-2 gradient back
        # to the scorer through the selection weights.
        detect_bundle = self.query_gen(
            scorer_out.score_map, scorer_out.feat_s8.detach(),
        )
        K_detect = detect_bundle.content.shape[1]

        # ── DN training: append noisy GT queries ─────────────────────────────
        dn_bundle: Optional[QueryBundle] = None
        dn_meta:   Optional[DNMeta]      = None
        attn_mask: Optional[Tensor]      = None

        if self.training and targets is not None:
            dn_bundle, dn_meta = self.dn_gen(targets, x.device)

        if dn_bundle is not None and dn_meta is not None:
            K_dn = dn_meta.dn_num_queries

            # Concatenate detect + DN queries along the query dimension
            all_bundle = QueryBundle(
                ref_points = torch.cat([detect_bundle.ref_points,
                                        dn_bundle.ref_points], dim=1),
                content    = torch.cat([detect_bundle.content,
                                        dn_bundle.content],    dim=1),
                scores     = torch.cat([detect_bundle.scores,
                                        dn_bundle.scores],     dim=1),
                classes    = torch.cat([detect_bundle.classes,
                                        dn_bundle.classes],    dim=1),
            )

            max_gt = K_dn // max(dn_meta.dn_num_groups, 1)
            attn_mask = DNQueryGenerator.build_attn_mask(
                K_detect       = K_detect,
                K_dn           = K_dn,
                dn_num_groups  = dn_meta.dn_num_groups,
                max_gt         = max_gt,
                device         = x.device,
            )
        else:
            all_bundle = detect_bundle
            K_dn       = 0

        # ── Decoder ───────────────────────────────────────────────────────────
        dec_out = self.decoder(all_bundle, neck_out, attn_mask=attn_mask)

        # ── Split detect / DN hidden states ───────────────────────────────────
        if K_dn > 0:
            hs_det   = dec_out.hs[:, :, :K_detect, :]
            refs_det = dec_out.refs_logit[:, :, :K_detect, :]
            hs_dn    = dec_out.hs[:, :, K_detect:, :]
            refs_dn  = dec_out.refs_logit[:, :, K_detect:, :]
            dn_out_detr  = self.detr_head(hs_dn, refs_dn)   # shared head weights
        else:
            hs_det   = dec_out.hs
            refs_det = dec_out.refs_logit
            dn_out_detr = None

        detr_out = self.detr_head(hs_det, refs_det)

        return {
            'score_map':     scorer_out.score_map,           # (B, 1, H/8, W/8)
            'stage2':        detr_out,
            'dn_out':        dn_out_detr,                    # DETROutput or None
            'dn_meta':       dn_meta,                        # DNMeta or None
            'query_scores':  detect_bundle.scores,           # (B, K)
            'query_classes': detect_bundle.classes,          # (B, K)
            'tau_query':     torch.tensor(self.query_gen._tau, device=x.device),
        }

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Propagate epoch to QueryGenerator for τ annealing."""
        self.query_gen.set_tau(epoch, total_epochs)

    @staticmethod
    def _compatible_state(src: dict, model_state: dict) -> tuple[dict, list]:
        ok, skipped = {}, []
        for k, v in src.items():
            if k in model_state and model_state[k].shape == v.shape:
                ok[k] = v
            else:
                skipped.append(k)
        return ok, skipped

    def load_pretrained(self, path: str) -> None:
        """
        Load LW-DETR COCO pretrained weights into backbone and decoder.
        Backbone and neck projector transfer completely; decoder transfers
        shape-matched weights only.
        """
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt)

        # ── Backbone ──────────────────────────────────────────────────────────
        _bb_prefix   = 'backbone.0.encoder.'
        backbone_src = {k[len(_bb_prefix):]: v
                        for k, v in state.items() if k.startswith(_bb_prefix)}
        miss1, _ = self.backbone.load_state_dict(backbone_src, strict=False)
        print(f'[load_pretrained] backbone: {len(backbone_src)} tensors'
              + (f', {len(miss1)} missing' if miss1 else ''))

        # ── Neck projector ────────────────────────────────────────────────────
        _proj_prefix = 'backbone.0.projector.'
        proj_src = {k[len(_proj_prefix):]: v
                    for k, v in state.items() if k.startswith(_proj_prefix)}
        if proj_src:
            proj_model = self.neck.proj0.state_dict()
            ok_proj, skip_proj = self._compatible_state(proj_src, proj_model)
            self.neck.proj0.load_state_dict(ok_proj, strict=False)
            print(f'[load_pretrained] neck.proj0: {len(ok_proj)} tensors transferred'
                  + (f', {len(skip_proj)} shape-mismatched (skipped)' if skip_proj else ''))

        # ── Decoder ───────────────────────────────────────────────────────────
        decoder_src = {k[len('transformer.decoder.'):]: v
                       for k, v in state.items() if k.startswith('transformer.decoder.')}
        model_dec = self.decoder.transformer_decoder.state_dict()
        ok, skipped = self._compatible_state(decoder_src, model_dec)
        self.decoder.transformer_decoder.load_state_dict(ok, strict=False)
        print(f'[load_pretrained] decoder: {len(ok)} tensors transferred'
              + (f', {len(skipped)} shape-mismatched (skipped)' if skipped else ''))
        print(f'[load_pretrained] done — {path}')


# ── Factory ────────────────────────────────────────────────────────────────────

def build_hybrid_model(cfg: HybridModelConfig) -> HybridDETR:
    model = HybridDETR(cfg)
    if cfg.pretrained_path:
        model.load_pretrained(cfg.pretrained_path)
    return model


# Backward-compat alias used by existing checkpoints / evaluation scripts
HybridCenterNetDETR = HybridDETR
