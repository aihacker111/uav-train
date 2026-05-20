"""
HybridDecoder: wraps LW-DETR's TransformerDecoder to accept
dynamic queries from QueryGenerator instead of fixed learnable embeddings.

Pipeline:
  QueryBundle(content, ref_points)   ─┐
  MultiScaleNeckOutput(memory, ...)  ─┼─→ TransformerDecoder → DecoderOutput
                                      ┘
  DecoderOutput:
    hs         : (num_layers, B, K, D)   — hidden states from each decoder layer
    refs_logit : (num_layers, B, K, 4)   — iteratively refined ref points (unsigmoid)

The decoder's bbox_embed MLP is used for per-layer reference-point refinement
(guides WHERE to attend), while DETRHead.box_mlp makes the final box prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .config import DecoderConfig
from .query_gen import QueryBundle
from .neck import MultiScaleNeckOutput
from ..deform_attn import TransformerDecoder, TransformerDecoderLayer, MLP


# ── Output struct ──────────────────────────────────────────────────────────────

@dataclass
class DecoderOutput:
    hs:         Tensor   # (num_layers, B, K, D)  — per-layer decoder hidden states
    refs_logit: Tensor   # (num_layers, B, K, 4)  — per-layer refined ref points (unsigmoid)


# ── HybridDecoder ──────────────────────────────────────────────────────────────

class HybridDecoder(nn.Module):
    """
    Wraps LW-DETR's TransformerDecoder with dynamic query support.

    Differences from vanilla LW-DETR Transformer:
      - Queries come from QueryGenerator (heatmap peaks), not fixed embeddings.
      - No encoder stage — memory is provided by MultiScaleNeck directly.
      - bbox_embed guides per-layer reference refinement; final prediction is
        handled by DETRHead (separate parameters for clean modularity).
    """

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim

        decoder_layer = TransformerDecoderLayer(
            d_model            = D,
            sa_nhead           = cfg.sa_nheads,
            ca_nhead           = cfg.ca_nheads,
            dim_feedforward    = cfg.dim_feedforward,
            dropout            = cfg.dropout,
            activation         = 'relu',
            normalize_before   = False,
            group_detr         = 1,
            num_feature_levels = cfg.num_feature_levels,
            dec_n_points       = cfg.num_points,
            skip_self_attn     = False,
        )

        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers           = cfg.num_layers,
            norm                 = nn.LayerNorm(D),
            return_intermediate  = True,
            d_model              = D,
            lite_refpoint_refine = False,
            bbox_reparam         = cfg.bbox_reparam,
        )

        # Shared bbox_embed: used INSIDE the decoder for per-layer ref refinement.
        # Predicts deltas that update the reference points at each decoder layer,
        # guiding cross-attention to progressively better object positions.
        self.bbox_embed = MLP(D, D, 4, 3)
        self.transformer_decoder.bbox_embed = self.bbox_embed

        self._init_weights()

    def _init_weights(self) -> None:
        # Only initialise bbox_embed (new module); leave transformer_decoder
        # parameters at their construction defaults so pretrained weights can
        # be loaded cleanly via load_pretrained() without overwriting them.
        nn.init.zeros_(self.bbox_embed.layers[-1].weight)
        nn.init.zeros_(self.bbox_embed.layers[-1].bias)

    def forward(
        self,
        queries:   QueryBundle,
        neck:      MultiScaleNeckOutput,
        attn_mask: 'torch.Tensor | None' = None,
    ) -> DecoderOutput:
        """
        Args:
            queries   : QueryBundle — content (B, K, D), ref_points (B, K, 4) unsigmoid
            neck      : MultiScaleNeckOutput — memory, pos_embed, spatial_shapes, …
            attn_mask : optional bool (Q, Q) self-attention mask for DN training.
                        True = blocked.  Shape (K_total, K_total) applied to all
                        batch elements and heads uniformly.
        Returns:
            DecoderOutput with hs and refs_logit, both (num_layers, B, K, ·)
        """
        hs, refs_logit = self.transformer_decoder(
            tgt                 = queries.content,
            memory              = neck.memory,
            pos                 = neck.pos_embed,
            refpoints_unsigmoid = queries.ref_points,
            level_start_index   = neck.level_start_idx,
            spatial_shapes      = neck.spatial_shapes,
            valid_ratios        = neck.valid_ratios,
            tgt_mask            = attn_mask,
        )
        return DecoderOutput(hs=hs, refs_logit=refs_logit)
