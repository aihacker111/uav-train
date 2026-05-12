from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .networks.lwdetr_centernet import get_lwdetr_net

_model_factory = {
    'lwdetr': get_lwdetr_net,
}

def create_model(arch, heads, head_conv):
    if '_' in arch:
        prefix = arch[:arch.find('_')]
        suffix = arch[arch.find('_') + 1:]
        try:
            num_layers = int(suffix)
        except ValueError:
            # e.g. 'lwdetr_tiny' → num_layers=0, 'lwdetr_small' → 1, 'lwdetr_base' → 2
            _size_map = {'tiny': 0, 'small': 1, 'base': 2}
            num_layers = _size_map.get(suffix, 0)
        arch = prefix
    else:
        num_layers = 0

    get_model = _model_factory[arch]
    return get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)


def load_lwdetr_backbone(model, ckpt_path):
    """
    Load LW-DETR COCO pretrained encoder weights into model.backbone.
    Checkpoint format: {'model': {'backbone.0.encoder.*': tensor, ...}}
    Only encoder weights are loaded; projector and transformer are discarded.
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    src = checkpoint.get('model', checkpoint)

    prefix = 'backbone.0.encoder.'
    state = {k[len(prefix):]: v for k, v in src.items() if k.startswith(prefix)}

    missing, unexpected = model.backbone.load_state_dict(state, strict=True)
    print(f'[load_lwdetr_backbone] {ckpt_path}')
    print(f'  loaded {len(state)} encoder tensors')
    if missing:
        print(f'  missing  ({len(missing)}): {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  unexpected ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
    return model


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if 'epoch' in checkpoint:
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    state_dict_ = checkpoint.get('state_dict', checkpoint)
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f'Skip {k}: required {model_state_dict[k].shape}, loaded {state_dict[k].shape}')
                state_dict[k] = model_state_dict[k]
    for k in model_state_dict:
        if k not in state_dict:
            state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    return model


def save_model(path, epoch, model, optimizer=None):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
