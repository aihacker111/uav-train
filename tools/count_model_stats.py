"""
Measure parameters, GFLOPs, GPU memory, and FPS for HybridDEIM (DEIMv2 + CenterNet head).

Usage (run from repo root):
    cd src/
    python ../tools/count_model_stats.py --deim_config lib/models/configs/deim-uav/deimv2_hgnetv2_s_coco.yml
    python ../tools/count_model_stats.py --deim_config lib/models/configs/deim-uav/deimv2_hgnetv2_n_coco.yml --no-speed
    python ../tools/count_model_stats.py --deim_config ... --half --batch-size 4

Requires: pip install thop  (for GFLOPs; optional)
"""
import sys, os, argparse, time, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

# Build model inline so we can disable pretrained backbone loading (random weights
# are fine for parameter / GFLOPs / speed measurement).
def _build_model(args):
    _models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'lib', 'models')
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)

    import engine  # registers all @register() decorators
    from engine.core import YAMLConfig
    from lib.models.networks.deim_uav.model import HybridDEIM

    cfg = YAMLConfig(args.deim_config)

    # Override num_classes
    for cls_name in ('DEIMTransformer', 'DFINETransformer', 'RTDETRTransformerv2'):
        if cls_name in cfg.global_cfg:
            cfg.global_cfg[cls_name]['num_classes'] = args.num_classes

    # Disable pretrained backbone — no weights needed for stats
    for key, val in cfg.global_cfg.items():
        if isinstance(val, dict) and any(x in key for x in ('HGNet', 'ResNet', 'DINO', 'ViT', 'VIT')):
            val['pretrained'] = False

    deim_model = cfg.model
    enc_key    = next((k for k in ('HybridEncoder', 'LiteEncoder') if k in cfg.yaml_cfg), None)
    hidden_dim = cfg.yaml_cfg[enc_key].get('hidden_dim', 256) if enc_key else 256

    return HybridDEIM(deim=deim_model, num_classes=args.num_classes,
                      hidden_dim=hidden_dim, head_conv=32)

_LINE = '─' * 60
_DASH = '┄' * 60

COMPONENTS = [
    ('deim.backbone', 'Backbone (DEIM-UAV)'),
    ('deim.encoder',  'Encoder (DEIM-UAV)'),
    ('deim.decoder',  'Decoder (DEIM-UAV)'),
    ('cn_upsample',   'CenterNet Upsample'),
    ('cn_head',       'CenterNet Head'),
]


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
    p.add_argument('--deim_config', required=True)
    p.add_argument('--num_classes', type=int, default=7)
    p.add_argument('--input-h',    type=int, default=704)
    p.add_argument('--input-w',    type=int, default=1280)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup',     type=int, default=20)
    p.add_argument('--runs',       type=int, default=100)
    p.add_argument('--half',       action='store_true')
    p.add_argument('--no-speed',        action='store_true')
    p.add_argument('--device',          default='')
    p.add_argument('--backbone_weights', default='', help='Path to DEIMv2 pretrained .pth to check missing keys')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"\n{_LINE}")
    print(f"  Model  : DEIM-UAV (HybridDEIM: DEIM-UAV backbone + CenterNet head)")
    print(f"  Config : {args.deim_config}")
    print(f"  Device : {device}  |  Input: {args.input_h}×{args.input_w}  bs={args.batch_size}")
    print(_LINE)

    model  = _build_model(args)
    model.eval()

    # Allow arbitrary input resolution: encoder + decoder cache pos_embed / anchors /
    # valid_mask for eval_spatial_size (set in YAML, usually 640×640). Setting to None
    # forces dynamic computation from actual H×W — required for non-default resolutions.
    for submod in (getattr(model.deim, 'encoder', None), getattr(model.deim, 'decoder', None)):
        if submod is not None and hasattr(submod, 'eval_spatial_size'):
            submod.eval_spatial_size = None

    if args.backbone_weights:
        print(f"  Loading pretrained : {args.backbone_weights}")
        model.load_pretrained(args.backbone_weights)
        print(_LINE)
    dummy        = torch.zeros(args.batch_size, 3, args.input_h, args.input_w)
    dummy_single = torch.zeros(1,               3, args.input_h, args.input_w)

    # ── Parameters ──────────────────────────────────────────────────────────────
    total, trainable = count_params(model)
    print(f"  {'Total params':<24}: {total / 1e6:.2f} M")
    print(f"  {'Trainable params':<24}: {trainable / 1e6:.2f} M")
    for attr, label in COMPONENTS:
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            n = sum(p.numel() for p in obj.parameters())
            print(f"  {'  └─ ' + label:<24}: {n / 1e6:.2f} M  ({100 * n / total:.1f}%)")
        except AttributeError:
            pass

    # ── GFLOPs & GPU memory ──────────────────────────────────────────────────────
    print(_DASH)
    gflops, status = try_gflops(model, dummy_single)
    if status == 'ok':
        print(f"  {'GFLOPs (bs=1)':<24}: {gflops:.2f} G")
    else:
        print(f"  GFLOPs                 : [{status}]")

    mb = measure_gpu_mem(model, dummy, device)
    if mb:
        print(f"  {'GPU Mem FP32 (fwd)':<24}: {mb:.0f} MB")

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
    print(f"  stage1 (CenterNet)  hm: {tuple(s1.hm.shape)}  wh: {tuple(s1.wh.shape)}")
    print(f"  stage2 (DEIM-UAV) boxes: {tuple(s2.boxes.shape)}  logits: {tuple(s2.logits.shape)}")
    print(f"{_LINE}\n")


if __name__ == '__main__':
    main()
