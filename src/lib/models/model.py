from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn


def create_model(arch: str, heads: dict, head_conv: int,
                 reid_dim: int = 256, num_classes: int = 7,
                 opt=None) -> nn.Module:
    """
    Instantiate a model by architecture string.

    For hybrid architectures, pass --ecdet_config pointing to an ECDet YAML
    (e.g. lib/models/configs/ecdet_s_uav.yml).  The factory:
      1. Loads the YAML via EdgeCrafter's YAMLConfig / registry system.
      2. Overrides num_classes in ECTransformer to match the dataset.
      3. Wraps the ECDet model in HybridECDet (pure DETR flow with 4-level encoder).
    """
    if 'hybrid' not in arch:
        raise NotImplementedError(
            f"create_model: arch={arch!r} is not registered. "
            "Only 'hybrid*' architectures are supported."
        )

    opt = (heads.get('__opt__') if isinstance(heads, dict) else None) or opt
    ecdet_config = getattr(opt, 'ecdet_config', '') if opt else ''
    if not ecdet_config:
        raise ValueError(
            "--ecdet_config is required for hybrid architectures. "
            "Example: --ecdet_config lib/models/configs/ecdet_s_uav.yml"
        )

    # Add src/lib/models/ to sys.path so `engine` package (copied from EdgeCrafter)
    # is importable and its @register() decorators fire.
    _models_dir = os.path.dirname(os.path.abspath(__file__))
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)

    from engine.core import YAMLConfig

    cfg = YAMLConfig(ecdet_config)

    # Override ECTransformer num_classes to match the dataset (COCO=80, VisDrone=10)
    if 'ECTransformer' in cfg.yaml_cfg:
        cfg.yaml_cfg['ECTransformer']['num_classes'] = num_classes

    ecdet_model = cfg.model   # ECDet(backbone=ECViT, encoder=HybridEncoder, decoder=ECTransformer)

    encoder_cfg = cfg.yaml_cfg.get('HybridEncoder', {})
    hidden_dim  = encoder_cfg.get('hidden_dim', 256)

    from lib.models.networks.ecdet_uav.ec_model import HybridECDet
    return HybridECDet(
        ecdet=ecdet_model,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        reid_dim=reid_dim,
    )


# ── Weight loading ──────────────────────────────────────────────────────────────

def load_model(
    model:      nn.Module,
    path:       str,
    optimizer:  Optional[torch.optim.Optimizer] = None,
    resume:     bool  = False,
    lr:         Optional[float] = None,
    lr_step:    Optional[list]  = None,
    use_cosine: bool  = False,
    loss:       Optional[nn.Module] = None,
):
    """
    Load a full AMOT checkpoint (model weights + optional optimizer state).

    Shape mismatches are handled gracefully: mismatched tensors are replaced
    with the model's current random weights so training can continue.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if 'epoch' in checkpoint:
        print(f'[load_model] {path}  (epoch {checkpoint["epoch"]})')

    src        = checkpoint.get('state_dict', checkpoint)
    state_dict = {
        (k[7:] if k.startswith('module') and not k.startswith('module_list') else k): v
        for k, v in src.items()
    }

    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            print(f'  [skip] {k}: ckpt {v.shape} != model {model_state[k].shape}')
            state_dict[k] = model_state[k]

    for k in model_state:
        if k not in state_dict:
            state_dict[k] = model_state[k]

    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            pre_lrs = [pg['lr'] for pg in optimizer.param_groups]
            ref_lr  = pre_lrs[-1]

            ckpt_opt = checkpoint['optimizer']

            n_cur  = len(optimizer.param_groups)
            n_ckpt = len(ckpt_opt.get('param_groups', []))
            if n_cur != n_ckpt:
                print(f'  [warn] checkpoint has {n_ckpt} param groups, '
                      f'current optimizer has {n_cur} -- keeping first {n_cur}')
                kept_groups    = ckpt_opt['param_groups'][:n_cur]
                kept_param_ids = {pid for g in kept_groups for pid in g['params']}
                ckpt_opt = {
                    'state':        {k: v for k, v in ckpt_opt['state'].items()
                                     if k in kept_param_ids},
                    'param_groups': kept_groups,
                }

            optimizer.load_state_dict(ckpt_opt)
            start_epoch = checkpoint['epoch']

            target_ref_lr = lr
            if not use_cosine:
                for step in lr_step or []:
                    if start_epoch >= step:
                        target_ref_lr *= 0.1

            for pg, pre_lr in zip(optimizer.param_groups, pre_lrs):
                pg['lr'] = (target_ref_lr * pre_lr / ref_lr) if ref_lr > 0 else target_ref_lr

            lrs_str = ', '.join(f'{pg["lr"]:.2e}' for pg in optimizer.param_groups)
            print(f'  resumed optimizer  epoch={start_epoch}  lrs=[{lrs_str}]'
                  f'  (cosine={use_cosine})')
        else:
            print('  [warn] no optimizer state in checkpoint -- starting optimizer fresh')

    if loss is not None and 'loss_state' in checkpoint:
        loss.load_state_dict(checkpoint['loss_state'])
        print('  restored loss state')
    elif loss is not None and resume:
        print('  [warn] no loss_state in checkpoint -- loss starts fresh')

    if optimizer is not None:
        return model, optimizer, checkpoint.get('epoch', 0)
    return model


def save_model(
    path:      str,
    epoch:     int,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss:      Optional[nn.Module]             = None,
) -> None:
    """Save model (and optionally optimizer + loss) state to disk."""
    unwrapped = getattr(model, 'module', model)
    data = {
        'epoch':      epoch,
        'state_dict': unwrapped.state_dict(),
    }
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if loss is not None and len(list(loss.parameters())) > 0:
        data['loss_state'] = loss.state_dict()
    torch.save(data, path)
