"""
Measure parameters, GFLOPs, GPU memory, and FPS for DEIMv2-based models.

Supported architectures:
    hybrid   — HybridDEIM: DEIMv2 backbone/encoder/decoder + CenterNet aux head
    deim_mot — DEIMMotNet: DEIMv2 backbone/encoder + CenterNet + ReID heads (no decoder)

Usage (run from repo root):
    cd src/
    python ../tools/count_model_stats.py --arch deim_mot --deim_config lib/models/configs/deim-uav/deimv2_hgnetv2_s_coco.yml
    python ../tools/count_model_stats.py --arch hybrid   --deim_config lib/models/configs/deim-uav/deimv2_hgnetv2_s_coco.yml
    python ../tools/count_model_stats.py --arch deim_mot --deim_config ... --no-speed
    python ../tools/count_model_stats.py --arch deim_mot --deim_config ... --half --batch-size 4

Requires: pip install thop  (for GFLOPs; optional)
"""
import sys, os, argparse, time, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

_LINE = '─' * 60
_DASH = '┄' * 60

COMPONENTS_HYBRID = [
    ('deim.backbone', 'Backbone (DEIM-UAV)'),
    ('deim.encoder',  'Encoder (DEIM-UAV)'),
    ('deim.decoder',  'Decoder (DEIM-UAV)'),
    ('cn_upsample',   'CenterNet Upsample'),
    ('cn_head',       'CenterNet Head'),
]

COMPONENTS_DEIM_MOT = [
    ('deim.backbone', 'Backbone (DINOv3STAs)'),
    ('deim.encoder',  'Encoder (HybridEncoder)'),
    ('proj_enc',      'Enc proj 1×1'),
    ('lateral_s4',    'S4 lateral 1×1'),
    ('hm_head',       'Heatmap Head'),
    ('wh_head',       'WH Head'),
    ('reg_head',      'Offset Head'),
    ('id_head',       'ReID Head'),
]


def _build_deim_base(args):
    """Build and return (deim_model, hidden_dim) with pretrained backbone disabled."""
    _models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'lib', 'models')
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)

    import engine  # registers all @register() decorators
    from engine.core import YAMLConfig

    cfg = YAMLConfig(args.deim_config)

    for cls_name in ('DEIMTransformer', 'DFINETransformer', 'RTDETRTransformerv2'):
        if cls_name in cfg.global_cfg:
            cfg.global_cfg[cls_name]['num_classes'] = args.num_classes

    # Disable pretrained backbone — random weights are fine for stats
    for key, val in cfg.global_cfg.items():
        if isinstance(val, dict) and any(x in key for x in ('HGNet', 'ResNet', 'DINO', 'ViT', 'VIT')):
            val['pretrained'] = False

    deim_model = cfg.model
    enc_key    = next((k for k in ('HybridEncoder', 'LiteEncoder') if k in cfg.yaml_cfg), None)
    hidden_dim = cfg.yaml_cfg[enc_key].get('hidden_dim', 256) if enc_key else 256
    return deim_model, hidden_dim


def _build_model(args):
    deim_model, hidden_dim = _build_deim_base(args)

    if args.arch == 'deim_mot':
        from lib.models.networks.deim_uav.model_mot import DEIMMotNet
        return DEIMMotNet(
            deim=deim_model,
            num_classes=args.num_classes,
            hidden_dim=hidden_dim,
            head_conv=args.head_conv,
            reid_dim=args.reid_dim,
        )
    else:
        from lib.models.networks.deim_uav.model import HybridDEIM
        return HybridDEIM(
            deim=deim_model,
            num_classes=args.num_classes,
            hidden_dim=hidden_dim,
            head_conv=args.head_conv,
        )


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _fix_s4_hook(m):
    """Re-register the STA S4 forward hook lost during deepcopy.

    PyTorch hooks are not transferred by deepcopy, so _s4_cache stays None
    on the copy and lateral_s4 is never applied.  This re-wires the hook on
    the copied backbone's sta.stem.
    """
    if not (hasattr(m, '_s4_cache') and hasattr(m, 'lateral_s4') and m.lateral_s4 is not None):
        return
    try:
        stem = m.deim.backbone.sta.stem
    except AttributeError:
        return
    if hasattr(m, '_hook_handle') and m._hook_handle is not None:
        try:
            m._hook_handle.remove()
        except Exception:
            pass
    def _hook(module, inp, out):
        m._s4_cache = out
    m._hook_handle = stem.register_forward_hook(_hook)


def try_gflops(model, dummy):
    try:
        from thop import profile
        m_copy = copy.deepcopy(model).eval().cpu()
        _fix_s4_hook(m_copy)
        macs, _ = profile(m_copy, inputs=(dummy.cpu(),), verbose=False)
        return macs * 2 / 1e9, 'ok'
    except ImportError:
        return 0.0, 'thop not installed'
    except Exception as e:
        return 0.0, f'thop failed: {str(e)[:60]}'


def measure_fps(model, dummy, device, warmup, runs, half):
    m = copy.deepcopy(model).to(device).eval()
    _fix_s4_hook(m)
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
    p.add_argument('--arch',        default='deim_mot',
                   choices=['deim_mot', 'hybrid'],
                   help='deim_mot = DEIMMotNet (no decoder); hybrid = HybridDEIM')
    p.add_argument('--deim_config', required=True)
    p.add_argument('--num_classes', type=int, default=7)
    p.add_argument('--reid_dim',    type=int, default=128,
                   help='ReID embedding dim (deim_mot only)')
    p.add_argument('--head_conv',   type=int, default=64,
                   help='Intermediate channels in prediction heads')
    p.add_argument('--input-h',     type=int, default=512)
    p.add_argument('--input-w',     type=int, default=896)
    p.add_argument('--batch-size',  type=int, default=1)
    p.add_argument('--warmup',      type=int, default=20)
    p.add_argument('--runs',        type=int, default=100)
    p.add_argument('--half',        action='store_true')
    p.add_argument('--no-speed',    action='store_true')
    p.add_argument('--device',      default='')
    p.add_argument('--backbone_weights', default='',
                   help='Path to DEIMv2 pretrained .pth (loads via model.load_pretrained)')
    return p.parse_args()


def _print_output_shapes(model, dummy_single, arch):
    with torch.no_grad():
        out = model(dummy_single)

    if arch == 'deim_mot':
        o = out[0]
        print(f"  hm  : {tuple(o['hm'].shape)}  (B, num_classes, H/4, W/4)")
        print(f"  wh  : {tuple(o['wh'].shape)}")
        print(f"  reg : {tuple(o['reg'].shape)}")
        print(f"  id  : {tuple(o['id'].shape)}  (B, reid_dim, H/4, W/4)")
    else:
        s1, s2 = out['stage1'], out['stage2']
        print(f"  stage1 (CenterNet)  hm: {tuple(s1.hm.shape)}  wh: {tuple(s1.wh.shape)}")
        print(f"  stage2 (DEIM-UAV) boxes: {tuple(s2.boxes.shape)}  logits: {tuple(s2.logits.shape)}")


def main():
    args   = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    arch_label = {
        'deim_mot': 'DEIMMotNet (backbone + encoder + CenterNet/ReID, no decoder)',
        'hybrid':   'HybridDEIM (backbone + encoder + decoder + CenterNet aux)',
    }[args.arch]

    print(f"\n{_LINE}")
    print(f"  Model  : {arch_label}")
    print(f"  Config : {args.deim_config}")
    print(f"  Device : {device}  |  Input: {args.input_h}×{args.input_w}  bs={args.batch_size}")
    if args.arch == 'deim_mot':
        print(f"  reid_dim={args.reid_dim}  head_conv={args.head_conv}  num_classes={args.num_classes}")
    print(_LINE)

    model = _build_model(args)
    model.eval()

    if args.arch == 'deim_mot':
        s4_on = getattr(model, 'lateral_s4', None) is not None
        print(f"  STA S4 lateral       : {'active' if s4_on else 'disabled (STA not found)'}")

    # Allow arbitrary resolution — clear cached spatial sizes
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
    print(f"  {'Total params':<26}: {total / 1e6:.2f} M")
    print(f"  {'Trainable params':<26}: {trainable / 1e6:.2f} M")

    components = COMPONENTS_DEIM_MOT if args.arch == 'deim_mot' else COMPONENTS_HYBRID
    for attr, label in components:
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
    _print_output_shapes(model, dummy_single, args.arch)
    print(f"{_LINE}\n")


if __name__ == '__main__':
    main()
