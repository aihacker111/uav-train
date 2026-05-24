"""
Measure parameters, GFLOPs, GPU memory, and FPS for HybridECDet.

Usage (run from src/):
    python ../tools/count_model_stats.py --ecdet_config lib/models/configs/ecdet_s_uav.yml
    python ../tools/count_model_stats.py --ecdet_config lib/models/configs/ecdet_s_uav.yml --half --batch-size 4 --no-speed

Requires: pip install thop  (for GFLOPs; optional)
"""
import sys, os, argparse, time, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from lib.models.model import create_model

COMPONENTS = [
    ('ecdet.backbone',   'Backbone (ECViT)'),
    ('ecdet.encoder',    'Encoder (HybridEncoder)'),
    ('ecdet.decoder',    'Decoder (ECTransformer)'),
    ('cn_backbone_proj', 'CenterNet Proj'),
    ('cn_upsample',      'CenterNet Upsample'),
    ('cn_head',          'CenterNet Head'),
]

_LINE = '─' * 60
_DASH = '┄' * 60


def build_model(args):
    class Opt:
        ecdet_config = args.ecdet_config
    return create_model(
        arch='hybrid_ecdet',
        heads={'__opt__': Opt()},
        head_conv=args.head_conv,
        num_classes=args.num_classes,
    )


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_gflops(model, dummy):
    try:
        from thop import profile
        macs, _ = profile(copy.deepcopy(model).eval().cpu(), inputs=(dummy.cpu(),), verbose=False)
        return macs * 2 / 1e9, 'ok'
    except ImportError:
        return 0.0, 'thop not installed'
    except Exception as e:
        return 0.0, f'thop failed: {str(e)[:60]}'


def measure_fps(model, dummy, device, warmup, runs, half):
    m = copy.deepcopy(model).to(device).eval()
    d = dummy.to(device)
    if half:
        m, d = m.half(), d.half()
    with torch.no_grad():
        for _ in range(warmup):
            m(d)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(runs):
                m(d)
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1) / runs
        else:
            s = time.perf_counter()
            for _ in range(runs):
                m(d)
            ms = (time.perf_counter() - s) * 1000 / runs
    return ms, dummy.shape[0] * 1000 / ms


def measure_gpu_mem(model, dummy, device):
    if device.type != 'cuda':
        return 0.0
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(dummy.to(device))
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ecdet_config',     required=True, help='Path to ECDet YAML config')
    p.add_argument('--num_classes',      type=int, default=10)
    p.add_argument('--head_conv',        type=int, default=32)
    p.add_argument('--input-h',          type=int, default=512)
    p.add_argument('--input-w',          type=int, default=832)
    p.add_argument('--batch-size',       type=int, default=1)
    p.add_argument('--warmup',           type=int, default=20)
    p.add_argument('--runs',             type=int, default=100)
    p.add_argument('--half',             action='store_true')
    p.add_argument('--no-speed',         action='store_true')
    p.add_argument('--device',           default='')
    p.add_argument('--backbone_weights', default='', help='Pretrained .pth to check missing keys')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f'\n{_LINE}')
    print(f'  Model  : HybridECDet (ECViT + CenterNet head)')
    print(f'  Config : {args.ecdet_config}')
    print(f'  Device : {device}  |  Input: {args.input_h}×{args.input_w}  bs={args.batch_size}')
    print(_LINE)

    model = build_model(args).eval()

    if args.backbone_weights:
        print(f'  Loading pretrained : {args.backbone_weights}')
        model.load_pretrained(args.backbone_weights)
        print(_LINE)

    dummy        = torch.zeros(args.batch_size, 3, args.input_h, args.input_w)
    dummy_single = torch.zeros(1,               3, args.input_h, args.input_w)

    # ── Parameters ──────────────────────────────────────────────────────────────
    total, trainable = count_params(model)
    print(f"  {'Total params':<26}: {total / 1e6:.2f} M")
    print(f"  {'Trainable params':<26}: {trainable / 1e6:.2f} M")
    for attr, label in COMPONENTS:
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            n = sum(p.numel() for p in obj.parameters())
            print(f"  {'  └─ ' + label:<26}: {n / 1e6:.2f} M  ({100 * n / total:.1f}%)")
        except AttributeError:
            pass

    # ── GFLOPs & GPU memory ──────────────────────────────────────────────────────
    print(_DASH)
    gflops, status = try_gflops(model, dummy_single)
    if status == 'ok':
        print(f"  {'GFLOPs (bs=1)':<26}: {gflops:.2f} G")
    else:
        print(f"  GFLOPs                   : [{status}]")

    mb = measure_gpu_mem(model, dummy, device)
    if mb:
        print(f"  {'GPU Mem FP32 (fwd)':<26}: {mb:.0f} MB")

    # ── Speed ────────────────────────────────────────────────────────────────────
    if not args.no_speed:
        print(_DASH)
        for half in ([True, False] if args.half else [False]):
            dtype = 'FP16' if half else 'FP32'
            ms, fps = measure_fps(model, dummy, device, args.warmup, args.runs, half)
            print(f"  {dtype}  latency/img: {ms / args.batch_size:.1f} ms   FPS: {fps:.1f}")

    # ── Output shapes ────────────────────────────────────────────────────────────
    print(_DASH)
    with torch.no_grad():
        out = model(dummy_single)
    s1, s2 = out['stage1'], out['stage2']
    print(f"  stage1 (CenterNet)     hm: {tuple(s1.hm.shape)}  wh: {tuple(s1.wh.shape)}")
    print(f"  stage2 (ECTransformer) boxes: {tuple(s2.boxes.shape)}  logits: {tuple(s2.logits.shape)}")
    print(f'{_LINE}\n')


if __name__ == '__main__':
    main()
