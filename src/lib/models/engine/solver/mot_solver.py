"""
MotSolver — DetSolver extended for JDE multi-object tracking.

Changes over DetSolver:
  1. train()  — after BaseSolver.train(), builds ReID classifiers from
                dataset.nID_dict and registers them in the optimizer.
  2. fit()    — saves best checkpoint by training loss (not val mAP),
                since VisDrone may not have a validation split.
                If a val split IS present it also runs COCO eval.
"""
from __future__ import annotations

import json
import time
import datetime

import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from .det_engine import train_one_epoch, evaluate
from ._solver import BaseSolver
from .det_solver import DetSolver
from ..core import register


@register()
class MotSolver(DetSolver):
    """DetSolver variant for MOT/JDE training with ReID."""

    # ── train() ───────────────────────────────────────────────────────────────

    def train(self):
        """BaseSolver.train() variant — val_dataloader is optional for MOT."""
        # Replicate BaseSolver.train() but guard val_dataloader construction,
        # since VisDrone may not have a validation split in visdrone.json.
        self._setup()
        self.optimizer            = self.cfg.optimizer
        self.lr_scheduler         = self.cfg.lr_scheduler
        self.lr_warmup_scheduler  = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader,
            shuffle=self.cfg.train_dataloader.shuffle,
        )

        # val_dataloader is optional — only build if declared in YAML
        val_dl = self.cfg.val_dataloader   # None when 'val_dataloader' not in YAML
        if val_dl is not None:
            self.val_dataloader = dist_utils.warp_loader(val_dl, shuffle=val_dl.shuffle)
        else:
            self.val_dataloader = None

        self.evaluator = self.cfg.evaluator   # None when 'evaluator' not in YAML

        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

        # ── Build ReID classifiers from dataset nID_dict ──────────────────────
        criterion = getattr(self, 'criterion', None)
        if criterion is None:
            return

        # Unwrap DDP-wrapped dataloader if needed
        dataset = (getattr(self.train_dataloader, 'dataset', None)
                   or getattr(getattr(self.train_dataloader, 'dataset', None), 'dataset', None))

        nID_dict = getattr(dataset, 'nID_dict', None)
        if not nID_dict:
            print('[MotSolver] dataset has no nID_dict — ReID classifiers not built')
            return

        if hasattr(criterion, 'build_classifiers'):
            criterion.build_classifiers(nID_dict, device=self.device)
            print(f'[MotSolver] ReID classifiers: {len(nID_dict)} classes, '
                  f'{sum(nID_dict.values())} total identities')

            cls_params = list(criterion.classifiers.parameters())
            if cls_params:
                head_lr = self.optimizer.param_groups[-1]['lr']
                self.optimizer.add_param_group({
                    'params':       cls_params,
                    'lr':           head_lr,
                    'weight_decay': 1e-4,
                })
                print(f'[MotSolver] added {len(cls_params)} classifier tensors '
                      f'to optimizer (lr={head_lr:.2e})')

    # ── fit() ─────────────────────────────────────────────────────────────────

    def fit(self):
        """Training loop.

        Saves checkpoints by best training loss (not val mAP), since
        VisDrone may not have a validation split.  If a val split exists
        and an evaluator is configured, COCO mAP is also logged.
        """
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print('-' * 42 + 'Start MOT training' + '-' * 42)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print(f'     ## Using Self-defined Scheduler-{args.lrsheduler} ##')
            # FlatCosineLRScheduler reads group["initial_lr"] which PyTorch only
            # sets when a standard lr_scheduler is used first. Set it manually.
            for pg in self.optimizer.param_groups:
                if 'initial_lr' not in pg:
                    pg['initial_lr'] = pg['lr']
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer, args.lr_gamma, iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        has_val = (self.val_dataloader is not None
                   and len(self.val_dataloader.dataset) > 0)
        best_loss  = float('inf')
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # ── EMA restart at stop_epoch ─────────────────────────────────────
            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                best_ckpt = str(self.output_dir / 'best.pth')
                if (self.output_dir / 'best.pth').exists():
                    self.load_resume_state(best_ckpt)
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            # ── Train one epoch ───────────────────────────────────────────────
            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                grad_accum=getattr(args, 'grad_accum', 1),
            )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            # ── Save last checkpoint every epoch ──────────────────────────────
            if self.output_dir:
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')
                if (epoch + 1) % args.checkpoint_freq == 0:
                    dist_utils.save_on_master(
                        self.state_dict(), self.output_dir / f'checkpoint{epoch:04}.pth')

            # ── Save best checkpoint by training loss ─────────────────────────
            cur_loss = train_stats.get('loss', float('inf'))
            if cur_loss < best_loss and self.output_dir:
                best_loss = cur_loss
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
                print(f'[epoch {epoch}] best checkpoint updated (loss={best_loss:.4f})')

            # ── Optional COCO val eval (only when val split exists) ───────────
            test_stats: dict = {}
            if has_val and self.evaluator is not None:
                module = self.ema.module if self.ema else self.model
                test_stats, _ = evaluate(
                    module, self.criterion, self.postprocessor,
                    self.val_dataloader, self.evaluator, self.device,
                )

            # ── Log ───────────────────────────────────────────────────────────
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}':  v for k, v in test_stats.items()},
                'epoch': epoch,
            }
            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / 'log.txt').open('a') as f:
                    f.write(json.dumps(log_stats) + '\n')

            if self.writer and dist_utils.is_main_process():
                for k, v in train_stats.items():
                    self.writer.add_scalar(f'Train/{k}', v, epoch)
                self.writer.add_scalar('Train/best_loss', best_loss, epoch)

        total_time = time.time() - start_time
        print('Training time {}'.format(datetime.timedelta(seconds=int(total_time))))
        print(f'Best training loss: {best_loss:.4f}  → saved to {self.output_dir}/best.pth')

    # ── val() ─────────────────────────────────────────────────────────────────

    def val(self):
        """Evaluation-only pass — used with --test-only."""
        self.eval()
        if self.val_dataloader is None or len(self.val_dataloader.dataset) == 0:
            print('[MotSolver] No val split — skipping COCO eval. '
                  'Use track_AMOT.py for tracking metrics.')
            return
        module = self.ema.module if self.ema else self.model
        evaluate(module, self.criterion, self.postprocessor,
                 self.val_dataloader, self.evaluator, self.device)
