import torch
from torch.nn.parallel._functions import Scatter


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    r"""
    Scatter tensors and other objects across GPUs.

    - torch.Tensor          : split along `dim` via Scatter (respects chunk_sizes).
    - list of dict          : chunk list by image count AND move each dict's tensors
                              directly to the assigned GPU — avoids a wasteful
                              GPU0→GPU_i cross-device hop that happens when the
                              caller pre-moves all targets to the primary device.
    - list of other types   : recurse element-wise then zip across GPUs.
    - tuple                 : recurse element-wise then zip across GPUs.
    - dict                  : recurse over (k, v) items, rebuild per-GPU dicts.
    - primitive / other     : broadcast one reference to every GPU.
    """
    n_gpus = len(target_gpus)

    def _list_sizes(n):
        """How many items each GPU gets, proportional to chunk_sizes."""
        if chunk_sizes is not None:
            total  = sum(chunk_sizes)
            sizes  = [round(n * cs / total) for cs in chunk_sizes]
            sizes[-1] = n - sum(sizes[:-1])   # absorb rounding remainder
            return sizes
        base, rem = divmod(n, n_gpus)
        return [base + (1 if i < rem else 0) for i in range(n_gpus)]

    def scatter_map(obj):
        # ── Tensor: use PyTorch's native Scatter which respects chunk_sizes ──────
        if torch.is_tensor(obj):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)

        # ── List ──────────────────────────────────────────────────────────────────
        if isinstance(obj, list):
            if not obj:
                return [[] for _ in target_gpus]

            # List-of-dict (per-sample annotations like DETR targets):
            #   Chunk the *list* and move each chunk's tensors to the assigned GPU.
            #   Splitting tensors *inside* each dict is incorrect — it would give
            #   every GPU all N dicts with object-count tensors halved.
            if isinstance(obj[0], dict):
                sizes   = _list_sizes(len(obj))
                chunks  = []
                start   = 0
                for gpu_idx, sz in enumerate(sizes):
                    dev   = target_gpus[gpu_idx]
                    chunk = obj[start:start + sz]
                    chunks.append([
                        {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in d.items()}
                        for d in chunk
                    ])
                    start += sz
                return chunks

            # Generic list: recurse element-wise and zip across GPUs.
            return list(map(list, zip(*map(scatter_map, obj))))

        # ── Tuple ─────────────────────────────────────────────────────────────────
        if isinstance(obj, tuple):
            if not obj:
                return [() for _ in target_gpus]
            return list(zip(*map(scatter_map, obj)))

        # ── Dict ──────────────────────────────────────────────────────────────────
        if isinstance(obj, dict):
            if not obj:
                return [{} for _ in target_gpus]
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))

        # ── Primitive / non-tensor: broadcast unchanged ───────────────────────────
        return [obj for _ in target_gpus]

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    return tuple(inputs), tuple(kwargs)
