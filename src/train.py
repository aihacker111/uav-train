from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import json
import random

import numpy as np
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import torch.distributed as dist
import torch.utils.data
from torchvision.transforms import transforms as T

from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory
from lib.datasets.transforms import build_aerial_mot_transforms, disable_mosaic


def build_transforms(use_imagenet_norm, augment):
    ops = [T.ToTensor()]
    if use_imagenet_norm:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if augment:
        ops.append(T.RandomErasing(p=0.3, scale=(0.05, 0.20), ratio=(0.3, 3.3), value=0))
    return T.Compose(ops)


def build_cosine_scheduler(optimizer, warmup_iters: int, total_iters: int,
                           min_lr_ratio: float = 0.01, last_epoch: int = -1):
    """
    Linear warmup → cosine decay LR scheduler (per-iteration).

    Phase 1 [0, warmup_iters):  lr scales linearly from 0 → base_lr
    Phase 2 [warmup_iters, T):  cosine decay from base_lr → min_lr_ratio * base_lr

    ViT backbones are sensitive to large LR at initialisation; warmup prevents
    weight collapse in the first few hundred iterations.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return max(step, 1) / max(warmup_iters, 1)
        progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def scale_lr_for_multigpu(optimizer, opt) -> float:
    """Scale every param-group LR based on effective batch size vs. reference.

    Effective batch size = opt.batch_size * num_gpus.
    The caller should invoke this ONCE, right after the optimizer is built and
    BEFORE any checkpoint is loaded (load_model will restore the saved LRs).

    Returns the scalar factor that was applied (>1 means scale-up).
    """
    method = getattr(opt, 'lr_scale', 'linear')
    # Use DDP world_size when distributed, otherwise fall back to len(opt.gpus)
    num_gpus = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else len(opt.gpus)
    if method == 'none' or num_gpus <= 1:
        return 1.0

    effective_bs  = opt.batch_size * num_gpus
    base_bs       = max(getattr(opt, 'base_batch_size', opt.batch_size), 1)
    ratio         = effective_bs / base_bs

    if method == 'sqrt':
        factor = math.sqrt(ratio)
    else:  # 'linear'
        factor = ratio

    for pg in optimizer.param_groups:
        pg['lr'] *= factor

    lrs = [f'{pg["lr"]:.2e}' for pg in optimizer.param_groups]
    print(f'[LR scale] method={method}, gpus={num_gpus}, '
          f'effective_bs={effective_bs}, base_bs={base_bs}, '
          f'factor={factor:.4f}  ->  lrs={lrs}')
    return factor


def run(opt):
    # ── Distributed setup ─────────────────────────────────────────────────────
    # torchrun sets LOCAL_RANK env var; old torch.distributed.launch only passes
    # --local-rank as a CLI arg (stored in opt.local_rank, default -1).
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank < 0:                    # env var not set — try CLI arg fallback
        local_rank = opt.local_rank
    is_distributed = local_rank >= 0
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank       = 0
        world_size = 1
        local_rank = 0

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    if rank == 0:
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
    # Wire CopyPaste with dataset sample_fn now that the dataset exists.
    # pil_transform was built without sample_fn above; rebuild it here so
    # CopyPaste gets a callable that draws raw PIL images from the training set.
    def _sample_fn():
        return train_dataset._raw_pil_sample(random.randint(0, len(train_dataset) - 1))
    train_dataset.pil_transform = build_aerial_mot_transforms(sample_fn=_sample_fn)

    opt = opts().update_dataset_info_and_set_heads(opt, train_dataset)
    if rank == 0:
        print('opt:\n', opt)
    logger = Logger(opt) if rank == 0 else None

    # ── Sampler: DistributedSampler for DDP, optional repeat-factor for single-GPU
    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if getattr(opt, 'use_repeat_sampling', False) and rank == 0:
            print('WARNING: repeat_factor sampling is not supported in DDP mode; using DistributedSampler')
    elif getattr(opt, 'use_repeat_sampling', False) and hasattr(train_dataset, '_compute_repeat_factors'):
        thresh = getattr(opt, 'repeat_thresh', 0.001)
        rf = train_dataset._compute_repeat_factors(thresh)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=rf, num_samples=len(rf), replacement=True)
        print(f'RepeatFactorSampler: thresh={thresh}, '
              f'min_rf={min(rf):.2f}, max_rf={max(rf):.2f}, '
              f'mean_rf={sum(rf)/len(rf):.2f}')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(opt.num_workers > 0),  # prevent worker respawn stalls
        prefetch_factor=2 if opt.num_workers > 0 else None,
    )
    print(f'[rank {rank}] dataset={len(train_dataset)} imgs, '
          f'sampler={type(train_sampler).__name__}, '
          f'batch_size={opt.batch_size}, steps/epoch={len(train_loader)}')

    # ── Validation dataset (optional, only on rank 0) ───────────────────────────
    val_loader = None
    val_interval = getattr(opt, 'val_intervals', 5)
    if rank == 0 and 'val' in data_config and data_config['val']:
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
    # torchrun assigns GPUs via LOCAL_RANK; setting CUDA_VISIBLE_DEVICES would conflict.
    if not is_distributed:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print('Creating model...')
    # Pass opt via sentinel key so hybrid create_model can wire grad_checkpoint
    # and top_down_fusion without changing the public create_model signature.
    _heads = dict(opt.heads, **{'__opt__': opt}) if opt.task == 'hybrid' else opt.heads
    model = create_model(opt.arch, _heads, opt.head_conv,
                         reid_dim=getattr(opt, 'reid_dim', 256),
                         num_classes=opt.num_classes)

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

    # Load pretrained weights (before any LR scaling or checkpoint resume)
    if opt.backbone_weights:
        if 'hybrid' in opt.arch and hasattr(model, 'load_pretrained'):
            model.load_pretrained(opt.backbone_weights)
        elif hasattr(model, 'backbone'):
            from lib.models.model import load_pretrained_backbone
            model = load_pretrained_backbone(model, opt.backbone_weights)

    # ── Trainer (created BEFORE load_model so loss params are in optimizer) ──────
    # Creating the Trainer first ensures the loss param group (e.g. ReID classifier)
    # is already registered in the optimizer before load_model restores state —
    # this fixes the N vs N+1 param group mismatch on resume.
    use_cosine  = getattr(opt, 'cosine_lr', False)
    Trainer     = train_factory[opt.task]
    trainer     = Trainer(opt=opt, model=model, optimizer=optimizer)

    # Resume from checkpoint — pass use_cosine so load_model doesn't double-decay,
    # and pass trainer.loss so ReID classifier weights are also restored.
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step,
            use_cosine=use_cosine, loss=trainer.loss)

    # Scale LR for multi-GPU AFTER checkpoint is loaded so scaling isn't
    # overwritten by the optimizer state restore inside load_model.
    scale_lr_for_multigpu(optimizer, opt)

    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    # ── Cosine LR scheduler with linear warmup ───────────────────────────────────
    use_cosine = getattr(opt, 'cosine_lr', False)
    scheduler  = None
    if use_cosine:
        iters_per_epoch = len(train_loader)
        total_iters     = iters_per_epoch * opt.num_epochs
        warmup_iters    = getattr(opt, 'warmup_iters', min(1000, iters_per_epoch))
        min_lr_ratio    = getattr(opt, 'min_lr_ratio', 0.01)
        resumed_step    = start_epoch * iters_per_epoch

        # LambdaLR requires 'initial_lr' in every param group when last_epoch > -1.
        # Param groups added after the scheduler was last saved (e.g. the loss/ReID
        # group added by BaseTrainer.__init__) won't have this key — set it now.
        for pg in optimizer.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']

        scheduler = build_cosine_scheduler(
            optimizer, warmup_iters, total_iters, min_lr_ratio,
            last_epoch=resumed_step - 1 if resumed_step > 0 else -1,
        )
        print(f'Cosine LR: warmup={warmup_iters} iters, '
              f'total={total_iters} iters, min_lr_ratio={min_lr_ratio}, '
              f'resumed_step={resumed_step}')

    # ── Training loop ────────────────────────────────────────────────────────────
    if rank == 0:
        print('Starting training...')
    _global_step       = start_epoch * len(train_loader)
    _freeze_epochs     = getattr(opt, 'freeze_backbone_epochs', 0)
    _backbone_frozen   = False

    # Apply backbone freeze if we are resuming mid-freeze
    if _freeze_epochs > 0 and start_epoch < _freeze_epochs and hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        _backbone_frozen = True
        print(f'Backbone frozen (resuming mid-freeze, unfreeze at epoch {_freeze_epochs + 1})')

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        # Sync epoch across workers so DistributedSampler shuffles consistently
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # ── Backbone freeze / unfreeze ────────────────────────────────────────────
        if _freeze_epochs > 0 and hasattr(model, 'backbone'):
            if epoch == 1 and not _backbone_frozen:
                for p in model.backbone.parameters():
                    p.requires_grad_(False)
                _backbone_frozen = True
                print(f'Epoch {epoch}: backbone frozen for {_freeze_epochs} epoch(s)')
            elif epoch == _freeze_epochs + 1 and _backbone_frozen:
                for p in model.backbone.parameters():
                    p.requires_grad_(True)
                _backbone_frozen = False
                print(f'Epoch {epoch}: backbone unfrozen')

        # ── Notify loss of current epoch (consistency loss warmup ramp) ──────────
        if hasattr(trainer.loss, 'set_epoch'):
            trainer.loss.set_epoch(epoch)

        # ── Close-mosaic: disable Mosaic/Perspective/MixUp in last N epochs ─────
        _close_mosaic = getattr(opt, 'close_mosaic_epochs', 10)
        if _close_mosaic > 0 and epoch == opt.num_epochs - _close_mosaic + 1:
            train_dataset.pil_transform = disable_mosaic(train_dataset.pil_transform)
            remaining = [type(t).__name__ for t in train_dataset.pil_transform.transforms]
            print(f'Epoch {epoch}: close-mosaic — pipeline now: {remaining}')

        # ── Train ────────────────────────────────────────────────────────────────
        # Inject scheduler so base_trainer can call .step() after each batch
        if scheduler is not None:
            trainer.scheduler = scheduler
        log_train, _ = trainer.train(epoch, train_loader)
        _global_step += len(train_loader)

        if rank == 0:
            logger.write(f'epoch {epoch:03d} | train |')
            for k, v in log_train.items():
                logger.scalar_summary(f'train_{k}', v, epoch)
                logger.write(f' {k} {v:.4f} |')
            cur_lr = optimizer.param_groups[-1]['lr']
            logger.write(f' lr {cur_lr:.2e} |')
            logger.write('\n')

        # ── Periodic validation / mAP evaluation (rank 0 only) ──────────────────
        if rank == 0 and val_loader is not None and epoch % val_interval == 0:
            if hasattr(trainer, 'evaluate'):
                val_stats = trainer.evaluate(epoch, val_loader, logger=logger)
                logger.write(f'epoch {epoch:03d} | val   |')
                for k, v in val_stats.items():
                    logger.write(f' {k} {v:.4f} |')
                logger.write('\n')

        # ── Checkpointing (rank 0 only) ──────────────────────────────────────────
        if rank == 0:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer, loss=trainer.loss)
            if epoch in opt.lr_step or epoch % 5 == 0:
                save_model(os.path.join(opt.save_dir, f'model_{epoch}.pth'),
                           epoch, model, optimizer, loss=trainer.loss)

        # ── Step-decay (only when cosine scheduler is NOT used) ──────────────────
        if not use_cosine and epoch in opt.lr_step:
            decay = 0.1 ** (opt.lr_step.index(epoch) + 1)
            for pg in optimizer.param_groups:
                pg['lr'] *= decay
            if rank == 0:
                print(f'Epoch {epoch}: LR × {decay}')

        if rank == 0:
            logger.write('\n')

    if rank == 0:
        logger.close()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    opt = opts().parse()
    print('gpus:', opt.gpus)
    print('epochs:', opt.num_epochs)
    run(opt)
