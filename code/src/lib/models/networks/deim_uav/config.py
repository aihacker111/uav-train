from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CenterNetHeadConfig:
    """Stage-1 dense detection head config."""
    head_conv: int   = 32
    num_classes: int = 7


@dataclass
class DETRHeadConfig:
    """Stage-2 per-query prediction head config."""
    hidden_dim: int  = 256
    num_classes: int = 7
    reid_dim: int    = 256
