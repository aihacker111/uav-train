from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import json

import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import torch.utils.data
from torchvision.transforms import transforms as T

from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory
from lib.optim import FlatCosineLRScheduler


def build_optimizer(model, opt):
    """
    AdamW with 3 param groups matching EdgeCrafter's ecdet.yml optimizer config:
      - backbone (non-norm/bn/bias) : lr * 0.05,  weight_decay = default
      - backbone (norm/bn/bias)     : lr * 0.05,  weight_decay = 0
      - non-backbone (norm/bn/bias) : lr,          weight_decay = 0
      - everything else             : lr,          weight_decay = default
    """
    base_lr      = opt.lr
    backbone_lr  = base_lr * 0.05   # 1/20, same ratio as EdgeCrafter (0.000025 / 0.0005)
    weight_decay = opt.weight_decay

    patterns = [
        (r'^(?=.*backbone)(?!.*(?:norm|bn|bias)).*$', {'lr': backbone_lr}),
        (r'^(?=.*backbone)(?=.*(?:norm|bn|bias)).*$', {'lr': backbone_lr, 'weight_decay': 0.}),
        (r'^(?!.*backbone)(?=.*(?:norm|bn|bias)).*$', {'weight_decay': 0.}),
    ]

    param_groups = []
    visited = []
    for pattern, extras in patterns:
        params = {
            k: v for k, v in model.named_parameters()
            if v.requires_grad and re.search(pattern, k)
        }
        if params:
            param_groups.append({'params': list(params.values()), **extras})
            visited.extend(params.keys())

    remaining = {
        k: v for k, v in model.named_parameters()
        if v.requires_grad and k not in visited
    }
    if remaining:
        param_groups.append({'params': list(remaining.values())})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    return optimizer


def build_lr_scheduler(optimizer, opt, iter_per_epoch):
    """
    FlatCosineLRScheduler if --lr_scheduler=flatcosine, else None (step-decay handled in epoch loop).
    """
    if opt.lr_scheduler != 'flatcosine':
        return None

    flat_epochs    = opt.flat_epoch if opt.flat_epoch >= 0 else max(1, opt.num_epochs // 5)
    no_aug_epochs  = opt.no_aug_epochs
    warmup_iter    = min(opt.warmup_iter, 3 * iter_per_epoch)

    print(
        f'FlatCosineLRScheduler: lr={opt.lr}, lr_gamma={opt.lr_gamma}, '
        f'warmup_iter={warmup_iter}, flat_epochs={flat_epochs}, no_aug_epochs={no_aug_epochs}'
    )
    return FlatCosineLRScheduler(
        optimizer,
        lr_gamma      = opt.lr_gamma,
        iter_per_epoch= iter_per_epoch,
        total_epochs  = opt.num_epochs,
        warmup_iter   = warmup_iter,
        flat_epochs   = flat_epochs,
        no_aug_epochs = no_aug_epochs,
    )


def run(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)

    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root   = data_config['root']
    print("Dataset root: %s" % dataset_root)
    f.close()

    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt=opt,
                      root=dataset_root,
                      paths=trainset_paths,
                      img_size=opt.input_wh,
                      augment=True,
                      transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    print("opt.gpus_str: ", opt.gpus_str)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt)

    optimizer = build_optimizer(model, opt)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    lr_scheduler = build_lr_scheduler(optimizer, opt, iter_per_epoch=len(train_loader))

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
                       epoch, model, optimizer)
        logger.write('\n')

        # step-decay fallback (only when not using FlatCosine)
        if lr_scheduler is None and epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    print("opt.gpus: ", opt.gpus)
    print('epoch:', opt.num_epochs)
    run(opt)
