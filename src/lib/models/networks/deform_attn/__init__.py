"""
Self-contained Multi-Scale Deformable Attention package.

Before training, compile the CUDA extension once:
    cd src/lib/models/networks/deform_attn/ops
    python setup.py build install
    # or: bash make.sh
"""

from .transformer import TransformerDecoder, TransformerDecoderLayer, MLP
from .ops.modules import MSDeformAttn

__all__ = ['TransformerDecoder', 'TransformerDecoderLayer', 'MLP', 'MSDeformAttn']
