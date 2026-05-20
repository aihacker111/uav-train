import torch
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply

from .scatter_gather import scatter_kwargs


class _DataParallel(Module):
    r"""Custom DataParallel that uses our fixed scatter_gather.

    Differences from torch.nn.DataParallel:
    - scatter: correctly chunks list-of-dict targets (DETR annotations) to each
      GPU rather than splitting tensors inside each per-image dict.
    - gather:  handles (model_out, loss, loss_stats) tuples whose model_out may
      contain dataclass objects (ScorerOutput, DETROutput) that PyTorch's
      built-in gather cannot recurse into.
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super().__init__()

        if not torch.cuda.is_available():
            self.module      = module
            self.device_ids  = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim           = dim
        self.module        = module
        self.device_ids    = device_ids
        self.chunk_sizes   = chunk_sizes
        self.output_device = output_device

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs  = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids,
                              dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        """Gather (model_out, loss, loss_stats) tuples from all GPU replicas.

        PyTorch's built-in gather recurses into dicts and hits dataclass objects
        (ScorerOutput, DETROutput) which are neither tensors, plain dicts, nor
        namedtuples — causing TypeError.

        Strategy:
        - loss scalar  : stack → (n_gpus,) on output_device; caller does .mean()
        - loss_stats   : stack each key → (n_gpus,) on output_device
        - model_out    : take GPU-0's copy (not used in training backward pass)
        """
        if (outputs
                and isinstance(outputs[0], (tuple, list))
                and len(outputs[0]) == 3
                and isinstance(outputs[0][1], torch.Tensor)
                and isinstance(outputs[0][2], dict)):

            model_outs = [o[0] for o in outputs]
            losses     = torch.stack([o[1].to(output_device) for o in outputs])
            stats_keys = outputs[0][2].keys()
            gathered_stats = {
                k: torch.stack([o[2][k].to(output_device) for o in outputs])
                for k in stats_keys
            }
            return model_outs[0], losses, gathered_stats

        # Fallback: standard PyTorch gather (works for plain tensor/dict outputs)
        return gather(outputs, output_device, dim=self.dim)


def DataParallel(module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
    """Factory that always returns _DataParallel for multi-GPU, so that our
    fixed scatter (list-of-dict chunking + device placement) and custom gather
    (dataclass-safe) are always active.

    Single-GPU path delegates to torch.nn.DataParallel (no scatter needed).
    """
    if device_ids is not None and len(device_ids) <= 1:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    return _DataParallel(module, device_ids, output_device, dim, chunk_sizes)
