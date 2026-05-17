import torch
from torch.nn.parallel._functions import Scatter


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    r"""
    Scatter tensors and other objects across GPUs.

    - torch.Tensor: split along `dim` using Scatter (respects chunk_sizes).
    - list of dict (per-sample annotations): chunk the list by slicing so
      each GPU gets the right number of complete per-image target dicts.
      Recursing into each dict and splitting its tensors is wrong — it
      would give every GPU all N dicts with object tensors halved.
    - list / tuple of other types: recurse element-wise then zip.
    - dict: recurse over items then reconstruct per-GPU dicts.
    - primitive / other: broadcast (one reference per GPU).
    """
    n_gpus = len(target_gpus)

    def _chunk_sizes_list(n):
        """Split n items across GPUs respecting chunk_sizes if given."""
        if chunk_sizes is not None:
            # chunk_sizes are image counts; scale proportionally for lists
            total = sum(chunk_sizes)
            sizes, start = [], 0
            for cs in chunk_sizes:
                sz = round(n * cs / total)
                sizes.append(sz)
                start += sz
            # fix rounding drift on last GPU
            sizes[-1] = n - sum(sizes[:-1])
            return sizes
        base, rem = divmod(n, n_gpus)
        return [base + (1 if i < rem else 0) for i in range(n_gpus)]

    def scatter_map(obj):
        if torch.is_tensor(obj):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)

        if isinstance(obj, list):
            if not obj:
                return [[] for _ in target_gpus]
            # Per-sample annotation lists (list-of-dict): chunk at list level.
            # Each GPU must receive complete target dicts for its images;
            # splitting tensor contents inside each dict is incorrect.
            if isinstance(obj[0], dict):
                sizes  = _chunk_sizes_list(len(obj))
                chunks, start = [], 0
                for sz in sizes:
                    chunks.append(obj[start:start + sz])
                    start += sz
                return chunks
            # List of tensors or other — recurse element-wise then zip
            return list(map(list, zip(*map(scatter_map, obj))))

        if isinstance(obj, tuple):
            if not obj:
                return [() for _ in target_gpus]
            return list(zip(*map(scatter_map, obj)))

        if isinstance(obj, dict):
            if not obj:
                return [{} for _ in target_gpus]
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))

        # Primitive / non-tensor: broadcast unchanged to every GPU
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
