"""
Hybrid CenterNet-DETR architecture for multi-object tracking.

Public API
----------
build_hybrid_model(cfg)       — instantiate the full model
HybridModelConfig             — top-level configuration dataclass
ViTConfig, NeckConfig, ...    — per-component configuration dataclasses

Output types (returned by HybridCenterNetDETR.forward)
------------------------------------------------------
CenterNetOutput   — stage-1 dense heatmap predictions
DETROutput        — stage-2 per-query refined predictions
QueryBundle       — intermediate query representation
DecoderOutput     — raw decoder hidden states + reference points

Import notes
------------
model.py and decoder.py depend on LW-DETR's CUDA MSDeformAttn ops and are
loaded lazily (via __getattr__) so that importing configs/heads/neck does not
require compiled CUDA extensions.
"""

# Eagerly importable — no CUDA extension dependency
from .config import (
    HybridModelConfig,
    ViTConfig,
    NeckConfig,
    DecoderConfig,
    CenterNetHeadConfig,
    DETRHeadConfig,
    QueryGenConfig,
)
from .heads import CenterNetOutput, DETROutput
from .query_gen import QueryBundle
from .neck import MultiScaleNeckOutput

__all__ = [
    # Model factory (lazy — requires compiled CUDA ops)
    'build_hybrid_model',
    'HybridCenterNetDETR',
    # Configs
    'HybridModelConfig',
    'ViTConfig',
    'NeckConfig',
    'DecoderConfig',
    'CenterNetHeadConfig',
    'DETRHeadConfig',
    'QueryGenConfig',
    # Output types
    'CenterNetOutput',
    'DETROutput',
    'QueryBundle',
    'DecoderOutput',
    'MultiScaleNeckOutput',
]

_LAZY = {
    'HybridCenterNetDETR': ('.model',   'HybridCenterNetDETR'),
    'build_hybrid_model':  ('.model',   'build_hybrid_model'),
    'DecoderOutput':       ('.decoder', 'DecoderOutput'),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_rel, attr = _LAZY[name]
        import importlib
        mod = importlib.import_module(module_rel, package=__name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
