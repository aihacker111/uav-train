from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
from lib.datasets.transforms import build_aerial_mot_transforms


def build_transforms(use_imagenet_norm, augment):
    ops = [T.ToTensor()]
    if use_imagenet_norm:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if augment:
        ops.append(T.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0))
    return T.Compose(ops)


def run(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)

    data_config = json.load(open(opt.data_cfg))
    dataset_root   = data_config['root']
    trainset_paths = data_config['train']
    print('Dataset root:', dataset_root)

    use_imagenet_norm = 'lwdetr' in opt.arch or 'hybrid' in opt.arch
    pil_transform = build_aerial_mot_transforms()

    # ── Collate function (hybrid task needs variable-length DETR targets) ────────
    collate_fn = None
    if opt.task == 'hybrid':
        from lib.datasets.dataset.jde import hybrid_collate_fn
        collate_fn = hybrid_collate_fn

    # ── Train dataset ────────────────────────────────────────────────────────────
    train_dataset = Dataset(
        opt=opt,
        root=dataset_root,
        paths=trainset_paths,
        img_size=opt.input_wh,
        augment=True,
        transforms=build_transforms(use_imagenet_norm, augment=True),
        pil_transform=pil_transform,
    )
    opt = opts().update_dataset_info_and_set_heads(opt, train_dataset)
    print('opt:\n', opt)
    logger = Logger(opt)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ── Validation dataset (optional) ────────────────────────────────────────────
    val_loader = None
    val_interval = getattr(opt, 'val_intervals', 5)
    if 'val' in data_config and data_config['val']:
        val_dataset = Dataset(
            opt=opt,
            root=dataset_root,
            paths=data_config['val'],
            img_size=opt.input_wh,
            augment=False,
            transforms=build_transforms(use_imagenet_norm, augment=False),
            pil_transform=None,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=max(1, opt.batch_size // 2),
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        print(f'Val dataset: {len(val_dataset)} images, eval every {val_interval} epochs')

    # ── Model ────────────────────────────────────────────────────────────────────
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # Split optimizer: backbone (small LR) + heads (full LR)
    _use_split_opt = ('lwdetr' in opt.arch or 'hybrid' in opt.arch) and hasattr(model, 'backbone')
    if _use_split_opt:
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.parameters() if id(p) in backbone_ids],
             'lr': opt.lr * opt.backbone_lr_scale, 'weight_decay': 0.05},
            {'params': [p for p in model.parameters() if id(p) not in backbone_ids],
             'lr': opt.lr, 'weight_decay': 0.0},
        ])
        print(f'Split AdamW: backbone lr={opt.lr * opt.backbone_lr_scale:.2e}, heads lr={opt.lr:.2e}')
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # Load pretrained weights
    if opt.backbone_weights:
        if 'hybrid' in opt.arch and hasattr(model, 'load_pretrained'):
            model.load_pretrained(opt.backbone_weights)
        elif hasattr(model, 'backbone'):
            from lib.models.model import load_pretrained_backbone
            model = load_pretrained_backbone(model, opt.backbone_weights)

    # Resume from checkpoint
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    # ── Trainer ──────────────────────────────────────────────────────────────────
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    # ── Training loop ────────────────────────────────────────────────────────────
    print('Starting training...')

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        # ── Train ────────────────────────────────────────────────────────────────
        log_train, _ = trainer.train(epoch, train_loader)
        logger.write(f'epoch {epoch:03d} | train |')
        for k, v in log_train.items():
            logger.scalar_summary(f'train_{k}', v, epoch)
            logger.write(f' {k} {v:.4f} |')
        logger.write('\n')

        # ── Periodic validation / mAP evaluation ────────────────────────────────
        if val_loader is not None and epoch % val_interval == 0:
            if hasattr(trainer, 'evaluate'):
                val_stats = trainer.evaluate(epoch, val_loader, logger=logger)
                logger.write(f'epoch {epoch:03d} | val   |')
                for k, v in val_stats.items():
                    logger.write(f' {k} {v:.4f} |')
                logger.write('\n')

        # ── Always save last checkpoint ──────────────────────────────────────────
        save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)

        # ── Periodic checkpoint (every 5 epochs or at lr_step) ───────────────────
        if epoch in opt.lr_step or epoch % 5 == 0:
            save_model(os.path.join(opt.save_dir, f'model_{epoch}.pth'), epoch, model, optimizer)

        # ── LR decay at scheduled steps ──────────────────────────────────────────
        if epoch in opt.lr_step:
            decay = 0.1 ** (opt.lr_step.index(epoch) + 1)
            for pg in optimizer.param_groups:
                pg['lr'] *= decay
            print(f'Epoch {epoch}: LR × {decay}')

        logger.write('\n')

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    print('gpus:', opt.gpus)
    print('epochs:', opt.num_epochs)
    run(opt)
