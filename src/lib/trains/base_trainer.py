from __future__ import annotations

import os
import time
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from progress.bar import Bar

from lib.utils.utils import AverageMeter
from lib.models.data_parallel import DataParallel


class ModelWithLoss(nn.Module):
    """Wraps a model and its loss function for unified forward/backward."""

    def __init__(self, model: nn.Module, loss: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.loss  = loss

    def forward(self, batch: Dict[str, Any]):
        outputs    = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs=outputs, batch=batch)
        return outputs, loss, loss_stats


class BaseTrainer:
    def __init__(self, opt, model: nn.Module, optimizer=None) -> None:
        self.opt        = opt
        self.optimizer  = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss       = ModelWithLoss(model, self.loss)
        self.scheduler  = None   # set externally by train.py when cosine LR is used

        # AMP scaler — only active when --use_amp is set and CUDA is available.
        # GradScaler tracks the loss scale and skips optimizer steps when inf/nan
        # gradients appear (e.g. from fp16 overflow), then recovers automatically.
        use_amp = getattr(opt, 'use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if use_amp else None
        if use_amp:
            print('[AMP] Mixed-precision training enabled (fp16 forward + fp32 weights)')

        # Register any learnable parameters inside the loss (e.g. ReID classifier).
        # Explicitly mirror the last param group's LR so these params receive the
        # same (already multi-GPU-scaled) learning rate as the model heads.
        #
        # When resuming, the checkpoint's optimizer.state_dict() already contains
        # this extra param group (it was saved with it).  load_model restores all
        # groups so we must NOT add it again — that would create a duplicate group
        # and misalign the optimizer state.  We detect a resume by comparing the
        # current number of param groups against what we expect before adding.
        loss_params = list(self.loss.parameters())
        if loss_params:
            already_registered = any(
                any(p is lp for p in pg['params'] for lp in loss_params)
                for pg in self.optimizer.param_groups
            )
            if not already_registered:
                heads_lr = self.optimizer.param_groups[-1]['lr']
                self.optimizer.add_param_group({'params': loss_params, 'lr': heads_lr})

    def set_device(self, gpus, chunk_sizes, device) -> None:
        is_ddp = dist.is_available() and dist.is_initialized()
        if is_ddp:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.model_with_loss = self.model_with_loss.to(device)
            self.model_with_loss = torch.nn.parallel.DistributedDataParallel(
                self.model_with_loss,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=getattr(self.opt, 'find_unused_parameters', False),
            )
        elif len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss,
                device_ids=list(range(len(gpus))),
                chunk_sizes=chunk_sizes,
            ).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase: str, epoch: int, data_loader) -> tuple:
        model_with_loss = self.model_with_loss

        if phase == 'train':
            model_with_loss.train()
        else:
            is_ddp = dist.is_available() and dist.is_initialized()
            if is_ddp or len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt             = self.opt
        grad_accum      = getattr(opt, 'grad_accum', 1)
        results         = {}
        data_time       = AverageMeter()
        batch_time      = AverageMeter()
        avg_loss_stats  = {l: AverageMeter() for l in self.loss_stats}
        num_iters       = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar             = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        if phase == 'train':
            self.optimizer.zero_grad()  # clear any stale grads before epoch starts

        # Smoothed batch time for stable ETA (exponential moving average, α=0.05)
        _ema_batch_s    = None
        _EMA_ALPHA      = 0.05
        end             = time.time()

        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_iters:
                break

            data_time.update(time.time() - end)

            for k in batch:
                if k in ('meta', 'targets'):
                    # meta:    CPU-only metadata, not needed on GPU
                    # targets: list-of-dict DETR annotations; scatter_gather moves
                    #          each chunk's tensors directly to the assigned GPU,
                    #          so pre-moving everything to the primary device here
                    #          would cause a wasteful GPU0→GPU_i cross-device hop.
                    pass
                else:
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            amp_enabled = self.scaler is not None
            with autocast(enabled=amp_enabled):
                outputs, loss, loss_stats = model_with_loss(batch)

            loss = loss.mean()
            if phase == 'train':
                # Gradient accumulation with AMP support.
                # scaler.scale() multiplies the loss by the current loss-scale factor
                # so gradients stay in fp32 representable range even in fp16 forward.
                scaled_loss = self.scaler.scale(loss / grad_accum) if amp_enabled \
                              else (loss / grad_accum)
                scaled_loss.backward()

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) >= num_iters:
                    grad_clip = getattr(opt, 'grad_clip', 0.0)
                    if amp_enabled:
                        # unscale_ restores true gradients before clipping/stepping
                        self.scaler.unscale_(self.optimizer)
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(
                                (p for pg in self.optimizer.param_groups for p in pg['params']),
                                grad_clip,
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(
                                (p for pg in self.optimizer.param_groups for p in pg['params']),
                                grad_clip,
                            )
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

            elapsed = time.time() - end
            batch_time.update(elapsed)
            end = time.time()

            # Smoothed ETA using EMA of batch time (avoids wild spikes from I/O stalls)
            _ema_batch_s = elapsed if _ema_batch_s is None \
                else _EMA_ALPHA * elapsed + (1 - _EMA_ALPHA) * _ema_batch_s
            _remaining   = (num_iters - batch_idx - 1) * _ema_batch_s
            _eta_m, _eta_s = divmod(int(_remaining), 60)
            _eta_h, _eta_m = divmod(_eta_m, 60)
            _eta_str     = f'{_eta_h:d}:{_eta_m:02d}:{_eta_s:02d}' if _eta_h \
                           else f'{_eta_m:d}:{_eta_s:02d}'

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta} '.format(
                epoch, batch_idx, num_iters, phase=phase,
                total=bar.elapsed_td, eta=_eta_str)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            if not opt.hide_data_time:
                Bar.suffix += '|Data {dt.val:.3f}s({dt.avg:.3f}s) |Net {bt.avg:.3f}s'.format(
                    dt=data_time, bt=batch_time)

            if opt.print_iter > 0:
                if batch_idx % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(outputs, batch, results)

            del outputs, loss, loss_stats, batch

        bar.finish()
        ret          = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time']  = bar.elapsed_td.total_seconds() / 60.0
        return ret, results

    # ── Abstract interface ─────────────────────────────────────────────────────

    def _get_losses(self, opt):
        raise NotImplementedError

    def debug(self, batch, output, iter_id) -> None:
        raise NotImplementedError

    def save_result(self, output, batch, results) -> None:
        raise NotImplementedError

    # ── Convenience wrappers ───────────────────────────────────────────────────

    def train(self, epoch: int, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def val(self, epoch: int, data_loader):
        return self.run_epoch('val', epoch, data_loader)
