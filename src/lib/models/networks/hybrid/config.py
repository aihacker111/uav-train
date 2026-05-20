from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ── ViT variant table ─────────────────────────────────────────────────────────
_VIT_VARIANTS = {
    #        embed  depth  window_blocks            out_feat_indexes     num_feat_levels
    'tiny':  (192,  6,  [0, 2, 4],                [1, 3, 5]),          # 3 levels
    'small': (192, 10,  [0, 1, 3, 6, 7, 9],       [2, 4, 5, 9]),       # 4 levels
    'base':  (768, 12,  [0, 1, 3, 4, 6, 7, 9, 10],[2, 5, 8, 11]),      # 4 levels
}


@dataclass
class ViTConfig:
    variant: str = 'small'          # 'tiny' | 'small' | 'base'

    @property
    def embed_dim(self) -> int:
        return _VIT_VARIANTS[self.variant][0]

    @property
    def depth(self) -> int:
        return _VIT_VARIANTS[self.variant][1]

    @property
    def window_block_indexes(self) -> List[int]:
        return _VIT_VARIANTS[self.variant][2]

    @property
    def out_feature_indexes(self) -> List[int]:
        return _VIT_VARIANTS[self.variant][3]

    @property
    def num_feature_levels(self) -> int:
        return len(self.out_feature_indexes)


@dataclass
class NeckConfig:
    hidden_dim: int        = 256
    num_output_levels: int = 1
    top_down_fusion: bool  = False


@dataclass
class DecoderConfig:
    hidden_dim: int         = 256
    num_layers: int         = 3
    sa_nheads: int          = 8
    ca_nheads: int          = 16
    dim_feedforward: int    = 2048
    dropout: float          = 0.0
    num_feature_levels: int = 1
    num_points: int         = 2
    bbox_reparam: bool      = False


@dataclass
class TokenScorerConfig:
    """Lightweight objectness scorer at stride-8.

    use_multiscale_fusion: fuse an s16-branch score signal into the s8 score map.
    This rescues tiny objects (<10px) that are only ~1 cell at stride-16:
      logit_fused = logit_s8 + bilinear_upsample(logit_s16)  → sigmoid
    """
    head_conv: int              = 64
    use_multiscale_fusion: bool = True


@dataclass
class DETRHeadConfig:
    """Stage-2 per-query prediction heads."""
    hidden_dim: int  = 256
    num_classes: int = 7
    reid_dim: int    = 256


@dataclass
class QueryGenConfig:
    top_k: int             = 200
    score_threshold: float = 0.01
    use_gumbel: bool       = True
    tau_start: float       = 1.0
    tau_end: float         = 0.1
    use_spatial_partition: bool  = False
    sp_grid_rows: int            = 4
    sp_grid_cols: int            = 4
    sp_queries_per_region: int   = 50
    sp_overlap_ratio: float      = 0.25
    sp_global_queries: int       = 32


@dataclass
class DNConfig:
    """Denoising (DN) training configuration.

    During training, num_dn_groups noisy copies of each GT box/label are appended
    to the detect queries and processed by the decoder in parallel.  A structured
    attention mask prevents detect↔DN and cross-group cross-attention.  The DN
    reconstruction loss (L1 + focal) provides direct supervision without matching.

    num_dn_groups       : number of independently-noised GT copies per image.
    dn_label_noise_ratio: fraction of DN queries whose class label is randomised.
    dn_box_noise_scale  : noise magnitude applied to normalised box coordinates.
    max_dn_queries      : per-image cap on total DN queries (memory safety).
    """
    num_dn_groups:          int   = 5
    dn_label_noise_ratio:   float = 0.5
    dn_box_noise_scale:     float = 0.4
    max_dn_queries:         int   = 500


@dataclass
class HybridModelConfig:
    vit:        ViTConfig          = field(default_factory=lambda: ViTConfig('small'))
    neck:       NeckConfig         = field(default_factory=NeckConfig)
    decoder:    DecoderConfig      = field(default_factory=DecoderConfig)
    scorer:     TokenScorerConfig  = field(default_factory=TokenScorerConfig)
    detr:       DETRHeadConfig     = field(default_factory=DETRHeadConfig)
    query_gen:  QueryGenConfig     = field(default_factory=QueryGenConfig)
    dn:         DNConfig           = field(default_factory=DNConfig)
    pretrained_path: str           = ''
    grad_checkpoint: bool          = False
