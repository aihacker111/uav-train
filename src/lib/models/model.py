from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .networks.lwdetr_centernet import get_lwdetr_net
# hybrid imports are deferred inside create_model to avoid loading CUDA ops at import time


# ── Model factory ──────────────────────────────────────────────────────────────

_SIZE_MAP = {'tiny': 0, 'small': 1, 'base': 2}

_MODEL_REGISTRY = {
    'lwdetr': get_lwdetr_net,
}


def create_model(arch: str, heads: dict, head_conv: int) -> nn.Module:
    """
    Instantiate a model by architecture string.

    Supported formats:
      'lwdetr_small'   → get_lwdetr_net(num_layers=1, ...)
      'hybrid_small'   → build_hybrid_model with ViTConfig('small')
    """
    if '_' in arch:
        prefix, suffix = arch.split('_', 1)
    else:
        prefix, suffix = arch, ''

    if prefix == 'hybrid':
        from .networks.hybrid import build_hybrid_model, HybridModelConfig
        variant = suffix if suffix in ('tiny', 'small', 'base') else 'small'
        cfg = HybridModelConfig()
        cfg.vit.variant = variant
        return build_hybrid_model(cfg)

    num_layers = _SIZE_MAP.get(suffix, 0)
    get_model  = _MODEL_REGISTRY[prefix]
    return get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)


# ── Weight loading ─────────────────────────────────────────────────────────────

def load_pretrained_backbone(model: nn.Module, ckpt_path: str) -> nn.Module:
    """
    Load LW-DETR COCO pretrained encoder weights into model.backbone.

    Expects checkpoint format: {'model': {'backbone.0.encoder.*': tensor, ...}}
    Only the ViT encoder weights are transferred; projector and transformer
    heads are intentionally skipped (different task / num_classes).
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    src        = checkpoint.get('model', checkpoint)
    prefix     = 'backbone.0.encoder.'
    state      = {k[len(prefix):]: v for k, v in src.items() if k.startswith(prefix)}

    missing, unexpected = model.backbone.load_state_dict(state, strict=True)
    print(f'[load_pretrained_backbone] {ckpt_path}')
    print(f'  transferred {len(state)} encoder tensors')
    if missing:
        print(f'  missing     ({len(missing)}): {missing[:3]}...')
    if unexpected:
        print(f'  unexpected  ({len(unexpected)}): {unexpected[:3]}...')
    return model


def load_model(
    model:     nn.Module,
    path:      str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    resume:    bool = False,
    lr:        Optional[float] = None,
    lr_step:   Optional[list]  = None,
):
    """
    Load a full AMOT checkpoint (model weights + optional optimizer state).

    Shape mismatches are handled gracefully: mismatched tensors are replaced
    with the model's current random weights so training can continue.
    """
    checkpoint  = torch.load(path, map_location='cpu')
    if 'epoch' in checkpoint:
        print(f'[load_model] {path}  (epoch {checkpoint["epoch"]})')

    src         = checkpoint.get('state_dict', checkpoint)
    state_dict  = {
        (k[7:] if k.startswith('module') and not k.startswith('module_list') else k): v
        for k, v in src.items()
    }

    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            print(f'  [skip] {k}: ckpt {v.shape} ≠ model {model_state[k].shape}')
            state_dict[k] = model_state[k]

    # Fill any keys present in the model but absent from the checkpoint
    for k in model_state:
        if k not in state_dict:
            state_dict[k] = model_state[k]

    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr    = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for pg in optimizer.param_groups:
                pg['lr'] = start_lr
            print(f'  resumed optimizer  lr={start_lr}')
        else:
            print('  [warn] no optimizer state in checkpoint')

    if optimizer is not None:
        return model, optimizer, checkpoint.get('epoch', 0)
    return model


def save_model(path: str, epoch: int, model: nn.Module, optimizer=None) -> None:
    """Save model (and optionally optimizer) state to disk."""
    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    data = {'epoch': epoch, 'state_dict': state_dict}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
