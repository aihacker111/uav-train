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


def create_model(arch: str, heads: dict, head_conv: int, reid_dim: int = 256) -> nn.Module:
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
        cfg.detr.reid_dim = reid_dim
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
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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

    When resume=True the optimizer state is restored.  LR is reset to `lr`
    with step-decay applied — UNLESS use_cosine=True, in which case the LR
    is left at `lr` without any step-decay so the cosine scheduler can
    compute the correct value from its own last_epoch.
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

    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            # Capture per-group LRs from the fresh optimizer BEFORE loading the
            # checkpoint.  These reflect the intended backbone/heads ratio
            # (e.g. backbone_lr_scale) set at optimizer creation time and are
            # used below to restore the ratio after LR reset.
            pre_lrs = [pg['lr'] for pg in optimizer.param_groups]
            ref_lr  = pre_lrs[-1]          # last group = heads = opt.lr (unscaled)

            ckpt_opt = checkpoint['optimizer']

            # ── Param-group count mismatch ────────────────────────────────────
            # The checkpoint was saved AFTER BaseTrainer.__init__ added the loss
            # param group (e.g. ReID classifier), so it has n_cur+1 groups while
            # the freshly created optimizer still has n_cur groups.  Truncate the
            # checkpoint to match so load_state_dict does not raise ValueError.
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

            # ── Target LR for the reference (heads) group ─────────────────────
            # Cosine: do NOT apply step-decay — the scheduler recomputes the
            # correct value from last_epoch; applying decay here would
            # double-decay and give a far-too-small LR.
            if use_cosine:
                target_ref_lr = lr
            else:
                target_ref_lr = lr
                for step in lr_step or []:
                    if start_epoch >= step:
                        target_ref_lr *= 0.1

            # ── Restore per-group LRs, preserving backbone/heads ratio ────────
            # Example: backbone group gets target_ref_lr * backbone_lr_scale
            #          instead of being reset to target_ref_lr like the heads.
            for pg, pre_lr in zip(optimizer.param_groups, pre_lrs):
                pg['lr'] = (target_ref_lr * pre_lr / ref_lr) if ref_lr > 0 else target_ref_lr

            lrs_str = ', '.join(f'{pg["lr"]:.2e}' for pg in optimizer.param_groups)
            print(f'  resumed optimizer  epoch={start_epoch}  lrs=[{lrs_str}]'
                  f'  (cosine={use_cosine})')
        else:
            print('  [warn] no optimizer state in checkpoint — starting optimizer fresh')

    # Restore learnable loss parameters (e.g. ReID classifier in HybridLoss).
    # These are NOT part of model.state_dict() so they must be saved/loaded separately.
    if loss is not None and 'loss_state' in checkpoint:
        loss.load_state_dict(checkpoint['loss_state'])
        print('  restored loss state (ReID classifier weights)')
    elif loss is not None and resume:
        print('  [warn] no loss_state in checkpoint — ReID classifier starts fresh')

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
    """Save model (and optionally optimizer + loss) state to disk.

    Unwraps DataParallel / DistributedDataParallel / custom _DataParallel so
    the saved state_dict always has clean keys (no 'module.' prefix).

    `loss` should be the trainer's loss module (e.g. HybridLoss) so that
    learnable parameters inside it (e.g. ReID classifier) are persisted and
    can be restored on resume — they are NOT part of model.state_dict().
    """
    unwrapped  = getattr(model, 'module', model)
    data = {
        'epoch':      epoch,
        'state_dict': unwrapped.state_dict(),
    }
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if loss is not None and len(list(loss.parameters())) > 0:
        data['loss_state'] = loss.state_dict()
    torch.save(data, path)
