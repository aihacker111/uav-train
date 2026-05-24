from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CenterNetHeadConfig:
    """Stage-1 dense detection head config."""
    head_conv: int   = 32
    num_classes: int = 7
