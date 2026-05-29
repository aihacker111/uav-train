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

    Supported architectures:
      'hybrid*'   — HybridDEIM: DEIMv2 backbone/encoder + DETR decoder + CenterNet aux head.
                    Pass --deim_config pointing to a DEIM-UAV YAML.
      'deim_mot*' — DEIMMotNet: DEIMv2 backbone/encoder + CenterNet + ReID heads only.
                    No DETR decoder. Same output format as AMOT (DLA-34).
                    Pass --deim_config pointing to the same YAML.

    opt can be supplied via heads['__opt__'] (train.py convention) or the opt= kwarg.
    """
    if 'hybrid' not in arch and 'deim_mot' not in arch:
        raise NotImplementedError(
            f"create_model: arch={arch!r} is not registered. "
            "Supported: 'hybrid*' or 'deim_mot*'."
        )

    opt = (heads.get('__opt__') if isinstance(heads, dict) else None) or opt
    deim_config = getattr(opt, 'deim_config', '') if opt else ''
    if not deim_config:
        raise ValueError(
            "--deim_config is required for DEIM-based architectures. "
            "Example: --deim_config src/lib/models/configs/deim-uav/deimv2_hgnetv2_s_coco.yml"
        )

    # Ensure src/lib/models/ is on sys.path so `import engine` resolves correctly
    _models = os.path.dirname(os.path.abspath(__file__))
    if _models not in sys.path:
        sys.path.insert(0, _models)

    import engine  # triggers all @register() decorators
    from engine.core import YAMLConfig

    cfg = YAMLConfig(deim_config)

    # Override decoder num_classes to match the dataset (COCO=80, VisDrone=7)
    for cls_name in ('DEIMTransformer', 'DFINETransformer', 'RTDETRTransformerv2'):
        if cls_name in cfg.global_cfg:
            cfg.global_cfg[cls_name]['num_classes'] = num_classes

    deim_model = cfg.model   # DEIM(backbone, encoder, decoder) fully built

    # Extract encoder hidden_dim for the CenterNet upsample head
    enc_key = next(
        (k for k in ('HybridEncoder', 'LiteEncoder') if k in cfg.yaml_cfg), None
    )
    hidden_dim = cfg.yaml_cfg[enc_key].get('hidden_dim', 256) if enc_key else 256

    if 'deim_mot' in arch:
        from lib.models.networks.deim_uav.model_mot import DEIMMotNet
        _log_wh = getattr(opt, 'log_wh', False) if opt else False
        net = DEIMMotNet(
            deim=deim_model,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            head_conv=head_conv,
            reid_dim=reid_dim,
        )
        net._init_head_weights(log_wh=_log_wh)
        return net

    from lib.models.networks.deim_uav.model import HybridDEIM
    return HybridDEIM(
        deim=deim_model,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        head_conv=head_conv,
        reid_dim=reid_dim,
    )


# ── Weight loading ─────────────────────────────────────────────────────────────

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
            print(f'  [skip] {k}: ckpt {v.shape} ≠ model {model_state[k].shape}')
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
                      f'current optimizer has {n_cur} — keeping first {n_cur}')
                kept_groups    = ckpt_opt['param_groups'][:n_cur]
                kept_param_ids = {pid for g in kept_groups for pid in g['params']}
                ckpt_opt = {
                    'state':        {k: v for k, v in ckpt_opt['state'].items()
                                     if k in kept_param_ids},
                    'param_groups': kept_groups,
                }

            optimizer.load_state_dict(ckpt_opt)
            start_epoch = checkpoint['epoch']

            if use_cosine:
                target_ref_lr = lr
            else:
                target_ref_lr = lr
                for step in lr_step or []:
                    if start_epoch >= step:
                        target_ref_lr *= 0.1

            for pg, pre_lr in zip(optimizer.param_groups, pre_lrs):
                pg['lr'] = (target_ref_lr * pre_lr / ref_lr) if ref_lr > 0 else target_ref_lr

            lrs_str = ', '.join(f'{pg["lr"]:.2e}' for pg in optimizer.param_groups)
            print(f'  resumed optimizer  epoch={start_epoch}  lrs=[{lrs_str}]'
                  f'  (cosine={use_cosine})')
        else:
            print('  [warn] no optimizer state in checkpoint — starting optimizer fresh')

    if loss is not None and 'loss_state' in checkpoint:
        loss.load_state_dict(checkpoint['loss_state'])
        print('  restored loss state')
    elif loss is not None and resume:
        print('  [warn] no loss_state in checkpoint — loss starts fresh')

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
