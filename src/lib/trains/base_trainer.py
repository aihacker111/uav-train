from __future__ import annotations

import time
from typing import Dict, Any

import torch
import torch.nn as nn
from progress.bar import Bar

from lib.utils.utils import AverageMeter


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

        # Register any learnable parameters inside the loss (e.g. ReID classifier)
        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device) -> None:
        if len(gpus) > 1:
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
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt             = self.opt
        results         = {}
        data_time       = AverageMeter()
        batch_time      = AverageMeter()
        avg_loss_stats  = {l: AverageMeter() for l in self.loss_stats}
        num_iters       = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar             = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end             = time.time()

        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_iters:
                break

            data_time.update(time.time() - end)

            for k in batch:
                if k == 'meta':
                    pass
                elif k == 'targets':
                    # list of per-image dicts with variable-size tensors
                    batch[k] = [
                        {kk: vv.to(device=opt.device, non_blocking=True)
                         for kk, vv in t.items()}
                        for t in batch[k]
                    ]
                else:
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            outputs, loss, loss_stats = model_with_loss(batch)

            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, batch_idx, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

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
