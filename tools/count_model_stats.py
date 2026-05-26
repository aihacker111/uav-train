"""
Measure parameters, GFLOPs, GPU memory, and FPS for HawkDet / HybridECDet.

Usage (run from repo root):
    python tools/count_model_stats.py --arch hawkdet_s --ecdet_config src/lib/models/configs/ecdet_s_uav.yml
    python tools/count_model_stats.py --arch hawkdet_s --ecdet_config src/lib/models/configs/ecdet_s_uav.yml --half --batch-size 4
    python tools/count_model_stats.py --arch hybrid_ecdet --ecdet_config src/lib/models/configs/ecdet_s_uav.yml

Requires: pip install thop  (for GFLOPs; optional)
"""
import sys, os, argparse, time, copy, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from lib.models.model import create_model

_LINE = '─' * 62
_DASH = '┄' * 62

# ── Per-arch component table ──────────────────────────────────────────────────

COMPONENTS_HAWKDET = [
    ('backbone', 'Backbone (ECViT)'),
    ('encoder',  'Encoder (HybridEncoder)'),
    ('heads',    'Detection heads (4× THead)'),
]

COMPONENTS_HYBRID = [
    ('ecdet.backbone',        'Backbone (ECViT)'),
    ('ecdet.encoder',         'Encoder (HybridEncoder)'),
    ('ecdet.decoder',         'Decoder (ECTransformer)'),
    ('reid_mlp',              'ReID MLP'),
]


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(args):
    class Opt:
        ecdet_config = args.ecdet_config
        reid_dim     = args.reid_dim
        reg_max      = args.reg_max
        num_convs    = args.num_convs
        head_feat_ch = args.head_feat_ch

    if 'hawkdet' in args.arch:
        return create_model(
            arch        = args.arch,
            heads       = {'__opt__': Opt()},
            head_conv   = 0,
            num_classes = args.num_classes,
            opt         = Opt(),
        )
    else:
        return create_model(
            arch        = 'hybrid_ecdet',
            heads       = {'__opt__': Opt()},
            head_conv   = 0,
            num_classes = args.num_classes,
        )


# ── Stats helpers ─────────────────────────────────────────────────────────────

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
        return 0.0, 'thop not installed — pip install thop'
    except Exception as e:
        return 0.0, f'thop failed: {str(e)[:80]}'


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


# ── Arch-specific display ─────────────────────────────────────────────────────

def print_components(model, components, total):
    for attr, label in components:
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            n = sum(p.numel() for p in obj.parameters())
            print(f"    {label:<34}: {n/1e6:>6.2f} M  ({100*n/total:>4.1f}%)")
        except AttributeError:
            pass


def print_output_shapes_hawkdet(model, dummy_single):
    with torch.no_grad():
        out = model(dummy_single)
    print(f"    pred_boxes  : {tuple(out['pred_boxes'].shape)}")
    print(f"    pred_scores : {tuple(out['pred_scores'].shape)}")
    if out.get('reid') is not None:
        print(f"    reid        : {tuple(out['reid'].shape)}")
    else:
        print(f"    reid        : None  (reid_dim=0)")

    H, W = dummy_single.shape[-2:]
    strides = model.strides
    total_anchors = sum(math.ceil(H/s) * math.ceil(W/s) for s in strides)
    parts = [f"S{s}:{math.ceil(H/s)*math.ceil(W/s)}" for s in strides]
    print(f"    anchors     : {total_anchors:,}  ({' + '.join(parts)})")


def print_output_shapes_hybrid(model, dummy_single):
    with torch.no_grad():
        out = model(dummy_single)
    s2 = out['stage2']
    print(f"    boxes  : {tuple(s2.boxes.shape)}")
    print(f"    logits : {tuple(s2.logits.shape)}")
    enc = s2.enc_aux_outputs
    if enc:
        print(f"    enc_aux: logits {tuple(enc[0]['pred_logits'].shape)}  "
              f"boxes {tuple(enc[0]['pred_boxes'].shape)}")


def print_encoder_layout(model, h, w, is_hawkdet):
    try:
        enc     = model.encoder if is_hawkdet else model.ecdet.encoder
        strides = getattr(enc, 'out_strides', [4, 8, 16, 32])
    except AttributeError:
        strides = [4, 8, 16, 32]

    shapes    = [(math.ceil(h / s), math.ceil(w / s)) for s in strides]
    total_pos = sum(r * c for r, c in shapes)
    print(f"    Feature positions L = {total_pos:,}")
    for s, (r, c) in zip(strides, shapes):
        print(f"      S{s:<3}  {r}×{c} = {r*c:,}")


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch',             default='hawkdet_s',
                   help='hawkdet_s/m/l/x  or  hybrid_ecdet')
    p.add_argument('--ecdet_config',     required=True)
    p.add_argument('--num_classes',      type=int, default=10)
    p.add_argument('--reid_dim',         type=int, default=0,
                   help='ReID embedding dim (0 = no ReID branch)')
    p.add_argument('--reg_max',          type=int, default=16)
    p.add_argument('--num_convs',        type=int, default=2)
    p.add_argument('--head_feat_ch',     type=int, default=128,
                   help='THead hidden channels (128 for S/M, 256 for L/X)')
    p.add_argument('--input-h',          type=int, default=512)
    p.add_argument('--input-w',          type=int, default=832)
    p.add_argument('--batch-size',       type=int, default=1)
    p.add_argument('--warmup',           type=int, default=20)
    p.add_argument('--runs',             type=int, default=100)
    p.add_argument('--half',             action='store_true')
    p.add_argument('--no-speed',         action='store_true')
    p.add_argument('--device',           default='')
    p.add_argument('--backbone_weights', default='')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    is_hawkdet = 'hawkdet' in args.arch
    device     = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    components = COMPONENTS_HAWKDET if is_hawkdet else COMPONENTS_HYBRID
    arch_label = f'HawkDet ({args.arch})' if is_hawkdet else 'HybridECDet'

    print(f'\n{_LINE}')
    print(f'  Model  : {arch_label}')
    print(f'  Config : {args.ecdet_config}')
    print(f'  Device : {device}  |  Input: {args.input_h}×{args.input_w}  bs={args.batch_size}')
    if is_hawkdet:
        print(f'  reid_dim={args.reid_dim}  reg_max={args.reg_max}  num_convs={args.num_convs}')
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
    print(f"  Parameters")
    print(f"    {'Total':<34}: {total/1e6:>7.2f} M")
    print(f"    {'Trainable':<34}: {trainable/1e6:>7.2f} M")
    print_components(model, components, total)

    # ── Encoder layout ───────────────────────────────────────────────────────────
    print(_DASH)
    print_encoder_layout(model, args.input_h, args.input_w, is_hawkdet)

    # ── GFLOPs ───────────────────────────────────────────────────────────────────
    print(_DASH)
    gflops, status = try_gflops(model, dummy_single)
    if status == 'ok':
        print(f"  {'GFLOPs (bs=1)':<38}: {gflops:.2f} G")
    else:
        print(f"  GFLOPs : [{status}]")

    # ── GPU memory ───────────────────────────────────────────────────────────────
    mb = measure_gpu_mem(model, dummy, device)
    if mb:
        print(f"  {'GPU Mem FP32 (bs=' + str(args.batch_size) + ')':<38}: {mb:.0f} MB")

    # ── Speed ────────────────────────────────────────────────────────────────────
    if not args.no_speed:
        print(_DASH)
        for half in ([True, False] if args.half else [False]):
            dtype = 'FP16' if half else 'FP32'
            ms, fps = measure_fps(model, dummy, device, args.warmup, args.runs, half)
            print(f"  {dtype}  latency/img : {ms/args.batch_size:.1f} ms   "
                  f"FPS (bs={args.batch_size}) : {fps:.1f}")

    # ── Output shapes ────────────────────────────────────────────────────────────
    print(_DASH)
    print(f"  Output shapes")
    if is_hawkdet:
        print_output_shapes_hawkdet(model, dummy_single)
    else:
        print_output_shapes_hybrid(model, dummy_single)

    print(f'{_LINE}\n')


if __name__ == '__main__':
    main()
