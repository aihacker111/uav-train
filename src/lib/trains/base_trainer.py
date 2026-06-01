from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import torch
from progress.bar import Bar
from lib.utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model.forward(batch['pre_input'], batch['input'])
        loss, loss_stats = self.loss.forward(outputs=outputs, batch=batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None, **kwargs):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)

        if hasattr(self, '_build_model_with_loss'):
            self.model_with_loss = self._build_model_with_loss(model, self.loss)
        else:
            self.model_with_loss = ModleWithLoss(model, self.loss)

        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        dev_ids = [i for i in range(len(gpus))]
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss,
                device_ids=dev_ids,
                chunk_sizes=chunk_sizes,
            ).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader, scheduler=None):
        model_with_loss = self.model_with_loss

        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt           = self.opt
        accum         = max(1, getattr(opt, 'grad_accum', 1))
        clip_max_norm = getattr(opt, 'clip_max_norm', 0.1)
        use_amp       = getattr(opt, 'use_amp', False) and phase == 'train'

        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters

        # Progress bar tracks optimizer steps, not raw batches.
        # num_steps = how many times the optimizer actually fires.
        num_steps = math.ceil(num_iters / accum) if phase == 'train' else num_iters

        data_time      = AverageMeter()
        batch_time     = AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        results        = {}

        # Per-step accumulation buffer: collect raw loss values across
        # `accum` batches, then average once before updating avg_loss_stats.
        accum_buf = {l: [] for l in self.loss_stats}
        step_id   = 0

        # Grad-accum label for bar (only shown when accum > 1)
        ga_tag = f' [GA×{accum}]' if accum > 1 and phase == 'train' else ''

        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_steps)
        end = time.time()

        if phase == 'train':
            self.optimizer.zero_grad()

        for batch_i, batch in enumerate(data_loader):
            if batch_i >= num_iters:
                break

            data_time.update(time.time() - end)

            for k in batch:
                if k == 'meta':
                    continue
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            # ----------------------------------------------------------------
            # Forward
            # ----------------------------------------------------------------
            with torch.amp.autocast('cuda', enabled=use_amp):
                output, loss, loss_stats = model_with_loss.forward(batch)

            loss = loss.mean()

            # Collect raw per-batch loss values for display averaging
            for l in self.loss_stats:
                if l in loss_stats:
                    accum_buf[l].append(loss_stats[l].mean().item())

            # ----------------------------------------------------------------
            # Backward — scale by 1/accum so effective grad = mean over accum batches
            # ----------------------------------------------------------------
            if phase == 'train':
                scaler.scale(loss / accum).backward()

            # ----------------------------------------------------------------
            # Optimizer step — fires every `accum` batches (or on last batch)
            # ----------------------------------------------------------------
            is_last_batch  = (batch_i + 1 >= num_iters)
            is_update_step = phase == 'train' and (
                (batch_i + 1) % accum == 0 or is_last_batch
            )

            if is_update_step:
                if clip_max_norm > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model_with_loss.parameters(), clip_max_norm)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                step_id += 1

            batch_time.update(time.time() - end)
            end = time.time()

            # ----------------------------------------------------------------
            # Progress bar — update only on optimizer step (train) or every
            # batch (val), so the bar reflects real work units.
            # ----------------------------------------------------------------
            should_display = is_update_step or phase != 'train'

            if should_display:
                # Average the accumulated batch losses for this optimizer step
                for l in self.loss_stats:
                    if accum_buf[l]:
                        step_val = sum(accum_buf[l]) / len(accum_buf[l])
                        avg_loss_stats[l].update(step_val)
                        accum_buf[l] = []

                # Display position: step_id for train, batch index for val
                disp_cur   = step_id   if phase == 'train' else batch_i + 1
                disp_total = num_steps if phase == 'train' else num_iters

                Bar.suffix = (
                    '{phase}: [{epoch}][{cur}/{total}]{ga}'
                    ' |Tot: {elapsed:} |ETA: {eta:} '
                ).format(
                    phase=phase, epoch=epoch,
                    cur=disp_cur, total=disp_total,
                    ga=ga_tag,
                    elapsed=bar.elapsed_td, eta=bar.eta_td,
                )

                for l in avg_loss_stats:
                    Bar.suffix += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

                if not opt.hide_data_time:
                    Bar.suffix += '|Data {:.3f}s |Net {:.3f}s'.format(
                        data_time.avg, batch_time.avg)

                if opt.print_iter > 0:
                    if (disp_cur - 1) % opt.print_iter == 0:
                        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
                else:
                    bar.next()

            if opt.test:
                self.save_result(output, batch, results)

            del output, loss, loss_stats, batch

        bar.finish()

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.0
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader, scheduler=None):
        return self.run_epoch('train', epoch, data_loader, scheduler=scheduler)
