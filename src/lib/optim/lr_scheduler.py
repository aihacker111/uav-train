"""
Ported from EdgeCrafter/ecdetseg/engine/optim/lr_scheduler.py
FlatCosineLRScheduler: warmup → flat → cosine decay → min_lr (per-iteration).
"""

import math
from functools import partial


def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter,
                         current_iter, init_lr, min_lr):
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2 if warmup_iter > 0 else init_lr
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(
            math.pi * (current_iter - flat_iter) / (total_iter - flat_iter - no_aug_iter)
        ))
        return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler:
    """
    Per-iteration LR scheduler matching EdgeCrafter's ecdet.yml schedule.

    Args:
        optimizer:       AdamW optimizer instance (already has base LRs set).
        lr_gamma:        min_lr = base_lr * lr_gamma  (e.g. 0.5).
        iter_per_epoch:  len(train_dataloader).
        total_epochs:    total training epochs.
        warmup_iter:     number of warm-up iterations (e.g. 2000).
        flat_epochs:     epochs to stay at peak LR before cosine decay starts.
        no_aug_epochs:   trailing epochs at min_lr (after augmentation stops).
    """

    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs,
                 warmup_iter, flat_epochs, no_aug_epochs):
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.min_lrs  = [base_lr * lr_gamma for base_lr in self.base_lrs]

        total_iter  = int(iter_per_epoch * total_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        flat_iter   = int(iter_per_epoch * flat_epochs)

        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter,
                               flat_iter, no_aug_iter)

    def step(self, current_iter, optimizer):
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i])
        return optimizer
