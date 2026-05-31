import torch
import torch.nn as nn

from .ecdet_jde.model import build_ecdet_jde, ECDetJDE


def create_model(arch: str, opt) -> ECDetJDE:
    """Create ECDetJDE model. `arch` must start with 'ecdet_jde'."""
    assert arch.startswith('ecdet_jde'), \
        f"Unknown arch '{arch}'. Only 'ecdet_jde' is supported."
    return build_ecdet_jde(opt)


def load_model(model: nn.Module,
               model_path: str,
               optimizer=None,
               resume: bool = False,
               lr: float = None,
               lr_step: list = None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'epoch' in checkpoint:
        print(f'Loaded {model_path}, epoch {checkpoint["epoch"]}')

    state_dict_ = checkpoint.get('state_dict', checkpoint)
    state_dict  = {k[7:] if k.startswith('module') and not k.startswith('module_list')
                   else k: v for k, v in state_dict_.items()}

    model_sd = model.state_dict()
    for k in list(state_dict):
        if k in model_sd and state_dict[k].shape != model_sd[k].shape:
            print(f'Skip {k}: ckpt {state_dict[k].shape} vs model {model_sd[k].shape}')
            state_dict[k] = model_sd[k]
    for k in model_sd:
        if k not in state_dict:
            state_dict[k] = model_sd[k]
    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in (lr_step or []):
                if start_epoch >= step:
                    start_lr *= 0.1
            for pg in optimizer.param_groups:
                pg['lr'] = start_lr
            print(f'Resumed optimizer with lr {start_lr}')
        else:
            print('No optimizer state in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    return model


def load_model_pretrain(model: nn.Module, model_path: str,
                        optimizer=None, resume=False, lr=None,
                        lr_step=None, freeze_params=False):
    result = load_model(model, model_path, optimizer, resume, lr, lr_step)
    if freeze_params and isinstance(result, tuple):
        m = result[0]
    else:
        m = result if not isinstance(result, tuple) else result[0]

    if freeze_params:
        checkpoint = torch.load(model_path, map_location='cpu')
        ckpt_sd = checkpoint.get('state_dict', checkpoint)
        ckpt_keys = set(ckpt_sd.keys())
        for name, param in m.named_parameters():
            if name in ckpt_keys:
                param.requires_grad = False
        print('Parameters loaded from checkpoint have been frozen.')

    return result


def save_model(path: str, epoch: int, model: nn.Module, optimizer=None):
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    data = {'epoch': epoch, 'state_dict': sd}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
