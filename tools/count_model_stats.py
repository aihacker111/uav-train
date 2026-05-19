"""
Count model parameters, GFLOPs / MACs, GPU memory, and inference speed (FPS / latency).

Usage:
    python tools/count_model_stats.py                          # default: lwdetr_tiny
    python tools/count_model_stats.py --arch hybrid_small
    python tools/count_model_stats.py --arch hybrid_small --half          # fp16 speed
    python tools/count_model_stats.py --arch hybrid_small --batch-size 4  # batch throughput
    python tools/count_model_stats.py --arch lwdetr_small --ckpt path/to/ckpt.pth
    python tools/count_model_stats.py --arch hybrid_tiny --warmup 50 --runs 200

Requires: pip install thop
"""

import sys
import os
import argparse
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from lib.models.model import create_model

_DIVIDER  = '─' * 60
_DIVIDER2 = '┄' * 60

_DEFAULT_CKPTS = {
    'lwdetr_tiny':  '../lwdetr_coco_pretrained/LWDETR_tiny_60e_coco.pth',
    'lwdetr_small': '../lwdetr_coco_pretrained/LWDETR_small_60e_coco.pth',
}
_LWDETR_HEADS = {'hm': 7, 'wh': 2, 'reg': 2, 'id': 256}


# ── Parameter counting ────────────────────────────────────────────────────────

def _module_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def print_param_table(model: nn.Module, arch: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"\n{_DIVIDER}")
    print(f"  PARAMETERS  —  {arch}")
    print(_DIVIDER2)
    print(f"  {'Total':<24}: {total / 1e6:>8.2f} M")
    print(f"  {'Trainable':<24}: {trainable / 1e6:>8.2f} M")
    if frozen > 0:
        print(f"  {'Frozen':<24}: {frozen / 1e6:>8.2f} M")

    components = [
        # hybrid
        ('backbone',       'Backbone (ViT)'),
        ('neck',           'Multi-scale Neck'),
        ('cn_upsample',    'CenterNet Upsample'),
        ('centernet_head', 'CenterNet Head'),
        ('query_gen',      'Query Generator'),
        ('decoder',        'DETR Decoder'),
        ('detr_head',      'DETR Head'),
        # lwdetr
        ('head',           'Detection Head'),
        ('transformer',    'Transformer'),
    ]
    has_any = False
    for attr, label in components:
        if hasattr(model, attr):
            n = _module_params(getattr(model, attr))
            pct = 100 * n / total
            if not has_any:
                print(_DIVIDER2)
                has_any = True
            print(f"  {'└─ ' + label:<24}: {n / 1e6:>8.2f} M  ({pct:5.1f}%)")


# ── GFLOPs via thop ───────────────────────────────────────────────────────────

def _try_thop(model: nn.Module, dummy: torch.Tensor) -> tuple[float, str]:
    """
    Returns (gflops, status_message).
    thop often fails on custom CUDA ops (e.g. MSDeformAttn); we catch and report.
    """
    try:
        from thop import profile
        m = copy.deepcopy(model).eval().cpu()
        d = dummy.cpu()
        macs, _ = profile(m, inputs=(d,), verbose=False)
        del m
        return macs * 2 / 1e9, 'ok'
    except ImportError:
        return 0.0, 'thop not installed — pip install thop'
    except Exception as e:
        short = str(e)[:80]
        return 0.0, f'thop failed ({short})'


def print_flops(model: nn.Module, dummy: torch.Tensor) -> None:
    print(_DIVIDER2)
    gflops, status = _try_thop(model, dummy)
    if status == 'ok':
        print(f"  {'GFLOPs (FP32)':<24}: {gflops:>8.2f} G")
    else:
        print(f"  GFLOPs               : [{status}]")


# ── GPU memory ────────────────────────────────────────────────────────────────

def measure_gpu_memory(
    model: nn.Module,
    dummy: torch.Tensor,
    device: torch.device,
    use_half: bool,
) -> tuple[float, float]:
    """
    Returns (peak_fwd_MB, peak_fwd_half_MB).
    Runs a single forward pass and measures peak CUDA memory.
    """
    if device.type != 'cuda':
        return 0.0, 0.0

    model = model.to(device)
    dummy = dummy.to(device)

    def _measure(m, d):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        with torch.no_grad():
            _ = m(d)
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB

    fp32_mb = _measure(model, dummy)
    if use_half:
        model_h = copy.deepcopy(model).half()
        dummy_h = dummy.half()
        fp16_mb = _measure(model_h, dummy_h)
        del model_h
    else:
        fp16_mb = 0.0

    return fp32_mb, fp16_mb


def print_memory(fp32_mb: float, fp16_mb: float, use_half: bool) -> None:
    if fp32_mb == 0.0:
        return
    print(_DIVIDER2)
    print(f"  {'GPU Mem (FP32, fwd)':<24}: {fp32_mb:>8.1f} MB")
    if use_half and fp16_mb > 0:
        print(f"  {'GPU Mem (FP16, fwd)':<24}: {fp16_mb:>8.1f} MB  ({100*fp16_mb/fp32_mb:.0f}% of fp32)")


# ── Inference speed ───────────────────────────────────────────────────────────

def print_speed(
    model:    nn.Module,
    dummy:    torch.Tensor,
    warmup:   int,
    runs:     int,
    device:   torch.device,
    use_half: bool,
    batch_size: int,
) -> None:
    model = copy.deepcopy(model).to(device).eval()
    dummy = dummy.to(device)
    if use_half:
        model = model.half()
        dummy = dummy.half()

    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()

        if use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(runs):
                _ = model(dummy)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt)
        else:
            t0 = time.perf_counter()
            for _ in range(runs):
                _ = model(dummy)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ms_per_batch = elapsed_ms / runs
    fps          = 1000.0 * batch_size / ms_per_batch

    print(_DIVIDER2)
    dtype_str = 'FP16' if use_half else 'FP32'
    print(f"  {'Device':<24}: {device}  [{dtype_str}]")
    print(f"  {'Batch size':<24}: {batch_size}")
    print(f"  {'Warmup / Runs':<24}: {warmup} / {runs}")
    print(f"  {'Latency / batch':<24}: {ms_per_batch:>8.1f} ms")
    print(f"  {'Latency / image':<24}: {ms_per_batch / batch_size:>8.1f} ms")
    print(f"  {'Throughput':<24}: {fps:>8.1f} FPS")


# ── Output shape summary ──────────────────────────────────────────────────────

def print_output_shapes(outputs, arch: str, input_h: int, input_w: int) -> None:
    print(_DIVIDER2)
    print(f"  {'Input':<24}: {input_h}×{input_w}")

    if arch.startswith('hybrid'):
        s1 = outputs['stage1']
        s2 = outputs['stage2']
        print(f"  {'Stage-1 heatmap':<24}: {tuple(s1.hm.shape)}")
        print(f"  {'Stage-1 wh':<24}: {tuple(s1.wh.shape)}")
        print(f"  {'Stage-2 boxes':<24}: {tuple(s2.boxes.shape)}")
        print(f"  {'Stage-2 logits':<24}: {tuple(s2.logits.shape)}")
        print(f"  {'Stage-2 reid':<24}: {tuple(s2.reid.shape)}")
        print(f"  {'Query scores':<24}: {tuple(outputs['query_scores'].shape)}")
    else:
        last = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        for k, v in last.items():
            if hasattr(v, 'shape'):
                print(f"  {k:<24}: {tuple(v.shape)}")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Model parameter / GFLOPs / speed profiler')
    p.add_argument('--arch', default='lwdetr_tiny',
                   help='lwdetr_tiny | lwdetr_small | lwdetr_base | '
                        'hybrid_tiny | hybrid_small | hybrid_base')
    p.add_argument('--ckpt', default='',
                   help='Checkpoint path (.pth). Leave empty for built-in defaults.')
    p.add_argument('--no-ckpt', action='store_true',
                   help='Skip checkpoint loading entirely (random weights).')
    p.add_argument('--input-h', type=int, default=704,
                   help='Input height (divisible by 64). Default: 704')
    p.add_argument('--input-w', type=int, default=1280,
                   help='Input width  (divisible by 64). Default: 1280')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size for throughput measurement. Default: 1')
    p.add_argument('--half', action='store_true',
                   help='Measure FP16 (half-precision) speed and memory.')
    p.add_argument('--warmup', type=int, default=20,
                   help='Warmup iterations before timing. Default: 20')
    p.add_argument('--runs', type=int, default=100,
                   help='Timed iterations for FPS measurement. Default: 100')
    p.add_argument('--device', default='',
                   help='"cuda", "cpu", or "" for auto-detect.')
    p.add_argument('--no-speed', action='store_true',
                   help='Skip speed measurement (params + GFLOPs only).')
    p.add_argument('--num-output-levels', type=int, default=1,
                   help='Feature pyramid levels for hybrid neck (1=single-scale, 2=P4+P5, 3=P4+P5+P6).')
    p.add_argument('--top-down-fusion', action='store_true',
                   help='Enable FPN top-down fusion (requires --num-output-levels > 1).')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    arch       = args.arch
    input_h    = args.input_h
    input_w    = args.input_w
    batch_size = args.batch_size
    use_half   = args.half

    is_hybrid = arch.startswith('hybrid')
    # For hybrid, pass profiling flags via __opt__ sentinel so create_model
    # can wire num_output_levels and top_down_fusion into NeckConfig.
    if is_hybrid:
        class _FakeOpt:
            num_output_levels = args.num_output_levels
            top_down_fusion   = args.top_down_fusion
            grad_checkpoint   = False
        heads = {'__opt__': _FakeOpt()}
    else:
        heads = _LWDETR_HEADS

    print(f"\n{_DIVIDER}")
    cfg_note = f'levels={args.num_output_levels}' + (' +FPN' if args.top_down_fusion else '')
    print(f"  Building: {arch}  ({input_h}×{input_w}  bs={batch_size}  {cfg_note})")
    model = create_model(arch, heads, head_conv=256)
    model.eval()

    # ── Pretrained weights ────────────────────────────────────────────────────
    if not args.no_ckpt:
        ckpt_path = args.ckpt or _DEFAULT_CKPTS.get(arch, '')
        if ckpt_path:
            ckpt_abs = os.path.join(os.path.dirname(__file__), ckpt_path)
            if os.path.isfile(ckpt_abs):
                print(f"  Loading: {ckpt_abs}")
                if is_hybrid and hasattr(model, 'load_pretrained'):
                    model.load_pretrained(ckpt_abs)
                else:
                    from lib.models.model import load_pretrained_backbone
                    model = load_pretrained_backbone(model, ckpt_abs)
            else:
                print(f"  [warn] checkpoint not found: {ckpt_abs}")

    dummy = torch.zeros(batch_size, 3, input_h, input_w)

    # ── Params ────────────────────────────────────────────────────────────────
    print_param_table(model, arch)

    # ── GFLOPs (always single-image for comparability) ────────────────────────
    dummy_single = torch.zeros(1, 3, input_h, input_w)
    print_flops(model, dummy_single)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ── GPU memory ────────────────────────────────────────────────────────────
    fp32_mb, fp16_mb = measure_gpu_memory(model, dummy, device, use_half)
    print_memory(fp32_mb, fp16_mb, use_half)

    # ── Speed ─────────────────────────────────────────────────────────────────
    if not args.no_speed:
        print_speed(model, dummy, args.warmup, args.runs, device, use_half, batch_size)
        if use_half:
            print(f"  {'[also FP32 speed]'}")
            print_speed(model, dummy, args.warmup, args.runs, device, False, batch_size)

    # ── Output shapes ─────────────────────────────────────────────────────────
    model_cpu = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        outputs = model_cpu(dummy_single)
    print_output_shapes(outputs, arch, input_h, input_w)
    print(f"{_DIVIDER}\n")


if __name__ == '__main__':
    main()
