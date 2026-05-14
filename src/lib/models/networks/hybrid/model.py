"""
HybridCenterNetDETR — main model composing all pipeline components.

Forward pass:
  image (B, 3, H, W)
    ↓  ViT backbone
  vit_features: list of (B, embed_dim, H/16, W/16)
    ↓  MultiScaleNeck
  MultiScaleNeckOutput   +   finest (B, hidden_dim, H/16, W/16)
    ↓  CenterNetHead
  CenterNetOutput  (hm, wh, reg)
    ↓  QueryGenerator  [hm + finest]
  QueryBundle  (K dynamic queries)
    ↓  HybridDecoder  [queries + neck memory]
  DecoderOutput  (hs, refs_logit)
    ↓  DETRHead
  DETROutput  (boxes, logits, reid, …)

Returns dict:
  {
      'stage1':        CenterNetOutput,
      'stage2':        DETROutput,
      'query_scores':  (B, K),  — heatmap confidence of each query
      'query_classes': (B, K),  — class predicted at each heatmap peak
  }
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch import Tensor

from .config import HybridModelConfig, ViTConfig
from .neck import MultiScaleNeck
from .heads import CenterNetHead, DETRHead
from .query_gen import QueryGenerator
from .decoder import HybridDecoder
from ..lwdetr_vit import ViT


# ── Backbone factory ───────────────────────────────────────────────────────────

def _build_vit(cfg: ViTConfig) -> ViT:
    """Instantiate ViT matching LW-DETR checkpoint architecture."""
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

class HybridCenterNetDETR(nn.Module):
    """
    Two-stage detector/tracker backbone for AMOT.

    Stage 1  — CenterNet: produces an unlimited-resolution heatmap and
               selects the top-K peaks as dynamic query seeds.

    Stage 2  — DETR decoder: refines each query through L layers of
               MSDeformAttn cross-attention over multi-scale ViT features,
               producing accurate boxes, class logits, and ReID embeddings.
    """

    def __init__(self, cfg: HybridModelConfig) -> None:
        super().__init__()
        vit_cfg  = cfg.vit
        neck_cfg = cfg.neck
        dec_cfg  = cfg.decoder

        # Propagate neck output level count → MSDeformAttn n_levels in decoder
        dec_cfg.num_feature_levels = neck_cfg.num_output_levels

        self.backbone = _build_vit(vit_cfg)
        # Neck takes only the finest backbone feature (embed_dim channels)
        self.neck     = MultiScaleNeck(vit_cfg.embed_dim, neck_cfg)

        self.centernet_head = CenterNetHead(neck_cfg.hidden_dim, cfg.centernet)
        self.query_gen      = QueryGenerator(neck_cfg.hidden_dim, cfg.query_gen)
        self.decoder        = HybridDecoder(dec_cfg)
        self.detr_head      = DETRHead(cfg.detr, bbox_reparam=dec_cfg.bbox_reparam)

    def forward(self, x: Tensor) -> Dict[str, Any]:
        # ── ViT backbone ──────────────────────────────────────────────────────
        vit_features: List[Tensor] = self.backbone(x)

        # ── Multi-scale neck ──────────────────────────────────────────────────
        neck_out = self.neck(vit_features)
        # Finest-scale (highest resolution) feature for the CenterNet head
        finest   = self.neck.finest_feature(vit_features)   # (B, D, H/16, W/16)

        # ── Stage 1: dense heatmap detection ─────────────────────────────────
        cn_out = self.centernet_head(finest)   # CenterNetOutput

        # ── Query generation: heatmap peaks → DETR seeds ─────────────────────
        queries = self.query_gen(cn_out.hm, finest)   # QueryBundle(B, K, ·)

        # ── Stage 2: DETR decoder refinement ─────────────────────────────────
        dec_out  = self.decoder(queries, neck_out)
        detr_out = self.detr_head(dec_out.hs, dec_out.refs_logit)

        return {
            'stage1':        cn_out,
            'stage2':        detr_out,
            'query_scores':  queries.scores,    # (B, K) heatmap confidence
            'query_classes': queries.classes,   # (B, K) stage-1 class index
        }

    @staticmethod
    def _compatible_state(src: dict, model_state: dict) -> tuple[dict, list]:
        """Return (compatible_src, skipped_keys) filtering shape-mismatched entries."""
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

        Backbone (backbone.encoder.*) transfers completely.

        Decoder (transformer.decoder.*) transfers only when shapes match — layers
        with mismatched shapes (e.g. if num_feature_levels or dim_feedforward
        differ from the checkpoint) are silently skipped so training can proceed
        from a mix of pretrained and fresh weights.

        Task-specific heads (class_embed, bbox_embed, reid_mlp) are always
        trained from scratch (different num_classes and task).
        """
        ckpt  = torch.load(path, map_location='cpu')
        state = ckpt.get('model', ckpt)

        # ── Backbone ──────────────────────────────────────────────────────────
        # Checkpoint key format: 'backbone.0.encoder.<param>'
        _bb_prefix = 'backbone.0.encoder.'
        backbone_src = {
            k[len(_bb_prefix):]: v
            for k, v in state.items() if k.startswith(_bb_prefix)
        }
        miss1, _ = self.backbone.load_state_dict(backbone_src, strict=False)
        print(f'[load_pretrained] backbone: {len(backbone_src)} tensors'
              + (f', {len(miss1)} missing' if miss1 else ''))

        # ── Decoder ───────────────────────────────────────────────────────────
        decoder_src = {
            k[len('transformer.decoder.'):]: v
            for k, v in state.items() if k.startswith('transformer.decoder.')
        }
        model_dec = self.decoder.transformer_decoder.state_dict()
        ok, skipped = self._compatible_state(decoder_src, model_dec)

        _, unexp = self.decoder.transformer_decoder.load_state_dict(ok, strict=False)
        print(f'[load_pretrained] decoder : {len(ok)} tensors transferred'
              + (f', {len(skipped)} shape-mismatched (skipped)' if skipped else ''))
        if skipped:
            print(f'  skipped: {skipped[:4]}{"..." if len(skipped) > 4 else ""}')

        print(f'[load_pretrained] done — {path}')


# ── Factory ────────────────────────────────────────────────────────────────────

def build_hybrid_model(cfg: HybridModelConfig) -> HybridCenterNetDETR:
    """Create the hybrid model and optionally load pretrained weights."""
    model = HybridCenterNetDETR(cfg)
    if cfg.pretrained_path:
        model.load_pretrained(cfg.pretrained_path)
    return model
