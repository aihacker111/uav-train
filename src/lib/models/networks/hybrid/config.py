from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ── ViT variant table ─────────────────────────────────────────────────────────
# Matches LWDETR_*_60e_coco.pth checkpoints exactly.
# Both tiny and small use the vit_tiny encoder (embed_dim=192);
# they differ only in depth and attention window layout.

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
    """
    Projects ViT features to uniform hidden_dim for the decoder.

    num_output_levels controls how many spatial scales the neck emits to the
    decoder.  Set to 1 to match the pretrained LW-DETR checkpoints (P4 only,
    single H/16 × W/16 scale).  Set to >1 to get a feature pyramid (each extra
    level is produced by a stride-2 conv, giving H/32, H/64, …).
    """
    hidden_dim: int       = 256
    num_output_levels: int = 1    # 1 = pretrained-compatible (P4 single scale)


@dataclass
class DecoderConfig:
    # All values verified against LWDETR_tiny/small_60e_coco.pth weight shapes:
    #   attention_weights  (32, 256) = ca_nheads(16) × n_levels(1) × n_points(2)
    #   sampling_offsets   (64, 256) = ca_nheads(16) × n_levels(1) × n_points(2) × 2
    #   linear1           (2048, 256) → dim_feedforward=2048
    # num_feature_levels is propagated at build time from NeckConfig.num_output_levels.
    hidden_dim: int         = 256
    num_layers: int         = 6     # 3 from pretrained; layers 4-6 use fresh init
    sa_nheads: int          = 8
    ca_nheads: int          = 16    # verified from checkpoint
    dim_feedforward: int    = 2048  # verified from checkpoint (was wrongly 1024)
    dropout: float          = 0.0
    num_feature_levels: int = 1     # set at build time from neck.num_output_levels
    num_points: int         = 2     # verified from checkpoint
    bbox_reparam: bool      = False # False: logit refs + additive delta (QueryGen compatible)


@dataclass
class CenterNetHeadConfig:
    """Stage-1 dense detection head."""
    head_conv: int   = 64
    num_classes: int = 7


@dataclass
class DETRHeadConfig:
    """Stage-2 per-query prediction heads."""
    hidden_dim: int  = 256
    num_classes: int = 7
    reid_dim: int    = 256


@dataclass
class QueryGenConfig:
    top_k: int             = 500    # candidates extracted from heatmap
    nms_kernel: int        = 3      # local-max NMS window
    score_threshold: float = 0.01   # discard near-zero peaks


@dataclass
class HybridModelConfig:
    vit: ViTConfig               = field(default_factory=lambda: ViTConfig('small'))
    neck: NeckConfig             = field(default_factory=NeckConfig)
    decoder: DecoderConfig       = field(default_factory=DecoderConfig)
    centernet: CenterNetHeadConfig = field(default_factory=CenterNetHeadConfig)
    detr: DETRHeadConfig         = field(default_factory=DETRHeadConfig)
    query_gen: QueryGenConfig    = field(default_factory=QueryGenConfig)
    pretrained_path: str         = ''
