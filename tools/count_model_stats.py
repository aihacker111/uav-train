"""
Count model parameters, GFLOPs / MACs, and inference speed (FPS / latency).

Usage:
    python tools/count_model_stats.py                       # default: lwdetr_tiny
    python tools/count_model_stats.py --arch hybrid_small
    python tools/count_model_stats.py --arch lwdetr_small --ckpt path/to/ckpt.pth
    python tools/count_model_stats.py --arch hybrid_tiny --warmup 50 --runs 200

Requires: pip install thop
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from lib.models.model import create_model

_DIVIDER = '─' * 52

# Default pretrained checkpoint paths (relative to tools/)
_DEFAULT_CKPTS = {
    'lwdetr_tiny':  '../lwdetr_coco_pretrained/LWDETR_tiny_60e_coco.pth',
    'lwdetr_small': '../lwdetr_coco_pretrained/LWDETR_small_60e_coco.pth',
}

_LWDETR_HEADS = {'hm': 7, 'wh': 2, 'reg': 2, 'id': 256}


# ── Parameter counting ────────────────────────────────────────────────────────

def _module_params(module):
    return sum(p.numel() for p in module.parameters())


def print_param_table(model, arch: str) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{_DIVIDER}")
    print(f"  Architecture     : {arch}")
    print(f"  Total params     : {total / 1e6:.2f} M")
    print(f"  Trainable params : {trainable / 1e6:.2f} M")

    # Per-component breakdown
    components = [
        ('backbone',       'Backbone (ViT)'),
        ('neck',           'Multi-scale Neck'),
        ('centernet_head', 'CenterNet Head'),
        ('cn_upsample',    'CN Upsample Neck'),
        ('query_gen',      'Query Generator'),
        ('decoder',        'DETR Decoder'),
        ('detr_head',      'DETR Head'),
        # lwdetr components
        ('head',           'Detection Head'),
        ('transformer',    'Transformer'),
    ]
    for attr, label in components:
        if hasattr(model, attr):
            n = _module_params(getattr(model, attr))
            print(f"  └─ {label:<20}: {n / 1e6:.2f} M")

    return total


# ── GFLOPs via thop ───────────────────────────────────────────────────────────

def print_flops(model, dummy: torch.Tensor) -> None:
    try:
        import copy
        from thop import profile
        model_copy = copy.deepcopy(model).eval()
        macs, _ = profile(model_copy, inputs=(dummy,), verbose=False)
        del model_copy
        print(f"  MACs             : {macs / 1e9:.2f} G")
        print(f"  GFLOPs (×2 MACs) : {macs * 2 / 1e9:.2f} G")
    except ImportError:
        print("  GFLOPs           : [thop not installed — run: pip install thop]")
    except Exception as e:
        print(f"  GFLOPs           : [thop failed — {e}]")


# ── Inference speed ───────────────────────────────────────────────────────────

def print_speed(model, dummy: torch.Tensor, warmup: int, runs: int, device: torch.device) -> None:
    model = model.to(device)
    dummy = dummy.to(device)
    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        # Warmup — fills CUDA JIT caches and avoids cold-start noise
        for _ in range(warmup):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()

        # Timed runs
        if use_cuda:
            # CUDA events give sub-millisecond accuracy without Python overhead
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(runs):
                _ = model(dummy)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt)  # total ms for all runs
        else:
            t0 = time.perf_counter()
            for _ in range(runs):
                _ = model(dummy)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ms_per_frame = elapsed_ms / runs
    fps          = 1000.0 / ms_per_frame
    print(f"  Device           : {device}")
    print(f"  Warmup / Runs    : {warmup} / {runs}")
    print(f"  Latency          : {ms_per_frame:.1f} ms/frame")
    print(f"  Throughput       : {fps:.1f} FPS")


# ── Output shape summary ──────────────────────────────────────────────────────

def print_output_shapes(outputs, arch: str) -> None:
    print(f"\n  Input shape      : {INPUT_H}×{INPUT_W}")

    if arch.startswith('hybrid'):
        # outputs is a dict: {'stage1': CenterNetOutput, 'stage2': DETROutput, ...}
        stage1 = outputs['stage1']
        stage2 = outputs['stage2']
        print(f"  Stage-1  hm      : {tuple(stage1.hm.shape)}")
        print(f"  Stage-1  wh      : {tuple(stage1.wh.shape)}")
        print(f"  Stage-1  reg     : {tuple(stage1.reg.shape)}")
        print(f"  Stage-2  boxes   : {tuple(stage2.boxes.shape)}")
        print(f"  Stage-2  logits  : {tuple(stage2.logits.shape)}")
        print(f"  Stage-2  reid    : {tuple(stage2.reid.shape)}")
        print(f"  Query scores     : {tuple(outputs['query_scores'].shape)}")
    else:
        # lwdetr: outputs is a list of dicts
        last = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        for k, v in last.items():
            if hasattr(v, 'shape'):
                print(f"  Output [{k}]       : {tuple(v.shape)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Count model parameters and GFLOPs')
    parser.add_argument('--arch', default='lwdetr_tiny',
                        help='Architecture string: lwdetr_tiny | lwdetr_small | lwdetr_base | '
                             'hybrid_tiny | hybrid_small | hybrid_base')
    parser.add_argument('--ckpt', default='',
                        help='Path to pretrained checkpoint (.pth). '
                             'Leave empty to use defaults for lwdetr variants.')
    parser.add_argument('--input-h', type=int, default=704,
                        help='Input height (must be divisible by 64)')
    parser.add_argument('--input-w', type=int, default=1280,
                        help='Input width (must be divisible by 64)')
    parser.add_argument('--no-ckpt', action='store_true',
                        help='Skip pretrained weight loading entirely')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Number of warmup iterations before timing')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of timed forward passes for FPS measurement')
    parser.add_argument('--device', default='',
                        help='Device for speed test: "cuda", "cpu", or "" for auto-detect')
    return parser.parse_args()


args   = parse_args()
ARCH   = args.arch
INPUT_H = args.input_h
INPUT_W = args.input_w


def main():
    is_hybrid = ARCH.startswith('hybrid')
    heads     = {} if is_hybrid else _LWDETR_HEADS

    print(f"Building: {ARCH}  ({INPUT_H}×{INPUT_W})")
    model = create_model(ARCH, heads, head_conv=256)
    model.eval()

    # ── Pretrained weights ───────────────────────────────────────────────────
    if not args.no_ckpt:
        ckpt_path = args.ckpt or _DEFAULT_CKPTS.get(ARCH, '')
        if ckpt_path:
            ckpt_abs = os.path.join(os.path.dirname(__file__), ckpt_path)
            if os.path.isfile(ckpt_abs):
                if is_hybrid and hasattr(model, 'load_pretrained'):
                    model.load_pretrained(ckpt_abs)
                else:
                    from lib.models.model import load_pretrained_backbone
                    model = load_pretrained_backbone(model, ckpt_abs)
            else:
                print(f"  [warn] checkpoint not found: {ckpt_abs}")

    # ── Parameter table ──────────────────────────────────────────────────────
    print_param_table(model, ARCH)

    # ── GFLOPs ───────────────────────────────────────────────────────────────
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
    print_flops(model, dummy)

    # ── Inference speed ───────────────────────────────────────────────────────
    if args.device:
        speed_device = torch.device(args.device)
    elif torch.cuda.is_available():
        speed_device = torch.device('cuda')
    else:
        speed_device = torch.device('cpu')
    print_speed(model, dummy, warmup=args.warmup, runs=args.runs, device=speed_device)

    # ── Forward pass + output shapes ─────────────────────────────────────────
    print()
    model_cpu = model.cpu()
    with torch.no_grad():
        outputs = model_cpu(dummy)
    print_output_shapes(outputs, ARCH)
    print(f"{_DIVIDER}\n")


if __name__ == '__main__':
    main()
