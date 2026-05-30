"""
Measure parameters, GFLOPs, GPU memory, and FPS for DEIMv2-based models.

Supported architectures:
    hybrid      — HybridDEIM: DEIMv2 backbone/encoder/decoder + CenterNet aux head
    deim_mot    — DEIMMotNet: DEIMv2 backbone/encoder + CenterNet + ReID heads (no decoder)
    deimv2_jde  — DEIMv2JDE:  full DETR + objectness-guided grid queries + per-query ReID

Usage (run from repo root):
    cd src/
    python ../tools/count_model_stats.py --arch deim_mot   --deim_config lib/models/configs/deim-uav/deimv2_hgnetv2_s_coco.yml
    python ../tools/count_model_stats.py --arch hybrid     --deim_config lib/models/configs/deim-uav/deimv2_hgnetv2_s_coco.yml
    python ../tools/count_model_stats.py --arch deimv2_jde --deim_config lib/models/configs/deim-uav/deimv2_dinov3_s_coco.yml
    python ../tools/count_model_stats.py --arch deimv2_jde --deim_config ... --grid-strides 16 32 --no-speed
    python ../tools/count_model_stats.py --arch deimv2_jde --deim_config ... --half --batch-size 4

Requires: pip install thop  (for GFLOPs; optional)
"""
import sys, os, argparse, time, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

_LINE = '─' * 68
_DASH = '┄' * 68


# ── Component tables ───────────────────────────────────────────────────────────

COMPONENTS_HYBRID = [
    ('deim.backbone', 'Backbone (DINOv3STAs)'),
    ('deim.encoder',  'Encoder  (HybridEncoder)'),
    ('deim.decoder',  'Decoder  (DEIMTransformer)'),
    ('cn_upsample',   'CenterNet Upsample'),
    ('cn_head',       'CenterNet Head'),
]

COMPONENTS_DEIM_MOT = [
    ('deim.backbone', 'Backbone (DINOv3STAs)'),
    ('deim.encoder',  'Encoder  (HybridEncoder)'),
    ('proj_s32',      'FPN proj S32 1×1'),
    ('proj_s16',      'FPN proj S16 1×1'),
    ('proj_s8',       'FPN proj S8  1×1'),
    ('lateral_s4',    'S4 lateral 1×1'),
    ('dilated_ctx',   'DilatedContext'),
    ('hm_head',       'Heatmap Head'),
    ('wh_head',       'WH Head'),
    ('reg_head',      'Offset Head'),
    ('id_head',       'ReID Head'),
]

COMPONENTS_DEIMV2_JDE = [
    ('deim.backbone',    'Backbone (DINOv3STAs)'),
    ('deim.encoder',     'Encoder  (HybridEncoder)'),
    ('deim.decoder',     'Decoder  (DEIMTransformer)'),
    ('obj_head',         'ObjectnessHead  [NEW]'),
    ('grid_qgen',        'GridQueryGen    [NEW]'),
    ('grid_qgen.proj',   '  └─ content proj (Linear/stride)'),
    ('grid_qgen.wh_head','  └─ WH head (Conv1×1/stride) [NEW]'),
    ('reid_mlp',         'ReID MLP        [NEW]'),
]


# ── Model builder ──────────────────────────────────────────────────────────────

def _build_deim_base(args):
    _models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'lib', 'models')
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)

    import engine
    from engine.core import YAMLConfig

    cfg = YAMLConfig(args.deim_config)

    for cls_name in ('DEIMTransformer', 'DFINETransformer', 'RTDETRTransformerv2'):
        if cls_name in cfg.global_cfg:
            cfg.global_cfg[cls_name]['num_classes'] = args.num_classes

    for key, val in cfg.global_cfg.items():
        if isinstance(val, dict) and any(x in key for x in ('HGNet', 'ResNet', 'DINO', 'ViT', 'VIT')):
            val['pretrained'] = False

    deim_model = cfg.model
    enc_key    = next((k for k in ('HybridEncoder', 'LiteEncoder') if k in cfg.yaml_cfg), None)
    hidden_dim = cfg.yaml_cfg[enc_key].get('hidden_dim', 256) if enc_key else 256
    return deim_model, hidden_dim


def _build_model(args):
    deim_model, hidden_dim = _build_deim_base(args)

    if args.arch == 'deimv2_jde':
        from lib.models.networks.deim_uav.model_detr_jde import DEIMv2JDE
        return DEIMv2JDE(
            deim             = deim_model,
            num_classes      = args.num_classes,
            hidden_dim       = hidden_dim,
            reid_dim         = args.reid_dim,
            grid_strides     = tuple(args.grid_strides),
            min_queries      = args.min_queries,
            train_k_headroom = args.train_k_headroom,
            min_train_k      = args.min_train_k,
        )

    if args.arch == 'deim_mot':
        from lib.models.networks.deim_uav.model_mot import DEIMMotNet
        return DEIMMotNet(
            deim        = deim_model,
            num_classes = args.num_classes,
            hidden_dim  = hidden_dim,
            head_conv   = args.head_conv,
            reid_dim    = args.reid_dim,
        )

    from lib.models.networks.deim_uav.model import HybridDEIM
    return HybridDEIM(
        deim        = deim_model,
        num_classes = args.num_classes,
        hidden_dim  = hidden_dim,
        head_conv   = args.head_conv,
    )


# ── Measurement helpers ────────────────────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _fix_s4_hook(m):
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
        return 0.0, f'thop failed: {str(e)[:80]}'


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


# ── Output shape reporter ──────────────────────────────────────────────────────

def _print_output_shapes(model, dummy_single, args):
    """Run one inference forward and print output tensor shapes."""
    model.eval()
    with torch.no_grad():
        out = model(dummy_single)

    arch = args.arch
    H, W = args.input_h, args.input_w

    if arch == 'deim_mot':
        o = out[0]
        print(f"  hm  : {tuple(o['hm'].shape)}   (B, num_classes, H/4, W/4)")
        print(f"  wh  : {tuple(o['wh'].shape)}")
        print(f"  reg : {tuple(o['reg'].shape)}")
        print(f"  id  : {tuple(o['id'].shape)}   (B, reid_dim, H/4, W/4)")

    elif arch == 'deimv2_jde':
        boxes      = out['pred_boxes']    # (B, N, 4)
        logits     = out['pred_logits']   # (B, N, C)
        reid       = out['pred_reid']     # (B, N, reid_dim)
        obj_scores = out.get('obj_scores')

        N = boxes.shape[1]

        # Max possible queries (all grid cells, no objectness filtering)
        max_n_s16 = (H // 16) * (W // 16)
        max_n_s32 = (H // 32) * (W // 32) if 32 in args.grid_strides else 0
        max_n     = max_n_s16 + max_n_s32

        # Adaptive range at inference: min_queries → max_n
        min_q = args.min_queries

        print(f"  pred_boxes    : {tuple(boxes.shape)}   cxcywh [0,1]")
        print(f"  pred_logits   : {tuple(logits.shape)}")
        print(f"  pred_reid     : {tuple(reid.shape)}   L2-normalised")
        if obj_scores is not None:
            print(f"  obj_scores    : {tuple(obj_scores.shape)}   sigmoid [0,1]")

        print(f"")
        print(f"  N (this run)  : {N}  queries fed to decoder")
        print(f"  N range       : {min_q} – {max_n}  (adaptive, objectness-guided)")

        parts = []
        for s in args.grid_strides:
            nh, nw = H // s, W // s
            parts.append(f"S{s} {nh}×{nw}={nh*nw}")
        print(f"  Grid capacity : {' + '.join(parts)} = {max_n} total")
        print(f"  Inference K   : top-K S16 by obj_score (count > 0.05) + all S32")
        print(f"  Training K    : max_GT_in_batch × {args.train_k_headroom:.1f}  "
              f"(min {args.min_train_k}, max {max_n_s16} S16)")

    else:  # hybrid
        s1, s2 = out['stage1'], out['stage2']
        print(f"  stage1 (CenterNet)  hm: {tuple(s1.hm.shape)}  wh: {tuple(s1.wh.shape)}")
        print(f"  stage2 (DEIM-UAV) boxes: {tuple(s2.boxes.shape)}  logits: {tuple(s2.logits.shape)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch',        default='deim_mot',
                   choices=['deim_mot', 'hybrid', 'deimv2_jde'])
    p.add_argument('--deim_config', required=True)
    p.add_argument('--num_classes', type=int,   default=7)
    p.add_argument('--reid_dim',    type=int,   default=128)
    p.add_argument('--head_conv',   type=int,   default=64)

    # deimv2_jde query config
    p.add_argument('--grid-strides',      type=int,   nargs='+', default=[16, 32])
    p.add_argument('--min-queries',       type=int,   default=500,
                   help='Min total queries at inference (default 500)')
    p.add_argument('--train-k-headroom',  type=float, default=2.5,
                   help='K = max_GT × headroom at training (default 2.5)')
    p.add_argument('--min-train-k',       type=int,   default=200,
                   help='Min S16 queries at training (default 200)')

    p.add_argument('--input-h',     type=int,   default=640)
    p.add_argument('--input-w',     type=int,   default=640)
    p.add_argument('--batch-size',  type=int,   default=1)
    p.add_argument('--warmup',      type=int,   default=20)
    p.add_argument('--runs',        type=int,   default=100)
    p.add_argument('--half',        action='store_true')
    p.add_argument('--no-speed',    action='store_true')
    p.add_argument('--device',      default='')
    p.add_argument('--backbone_weights', default='')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    arch_label = {
        'deim_mot':   'DEIMMotNet      (backbone + encoder + CenterNet/ReID)',
        'hybrid':     'HybridDEIM      (backbone + encoder + decoder + CenterNet aux)',
        'deimv2_jde': 'DEIMv2JDE       (backbone + encoder + decoder + obj-guided grid queries + ReID)',
    }[args.arch]

    H, W = args.input_h, args.input_w

    print(f"\n{_LINE}")
    print(f"  Model  : {arch_label}")
    print(f"  Config : {args.deim_config}")
    print(f"  Device : {device}  |  Input: {H}×{W}  bs={args.batch_size}")

    if args.arch in ('deim_mot', 'deimv2_jde'):
        print(f"  reid_dim={args.reid_dim}  num_classes={args.num_classes}")

    if args.arch == 'deimv2_jde':
        parts = []
        total_q = 0
        for s in args.grid_strides:
            nh, nw = H // s, W // s
            parts.append(f"S{s}({nh}×{nw}={nh*nw})")
            total_q += nh * nw
        print(f"  grid_strides={args.grid_strides}  →  max N_grid = {' + '.join(parts)} = {total_q}")
        print(f"  train_k_headroom={args.train_k_headroom}  min_train_k={args.min_train_k}")
        print(f"  min_queries(inference)={args.min_queries}")

    if args.arch == 'deim_mot':
        print(f"  head_conv={args.head_conv}")

    print(_LINE)

    model = _build_model(args)
    model.eval()

    if args.arch == 'deim_mot':
        s4_on  = getattr(model, 'lateral_s4', None) is not None
        fpn_on = all(hasattr(model, k) for k in ('proj_s32', 'proj_s16', 'proj_s8'))
        print(f"  FPN top-down (S32→S4) : {'active' if fpn_on else 'disabled'}")
        print(f"  STA S4 lateral        : {'active' if s4_on else 'disabled (STA not found)'}")

    # Clear cached spatial sizes so arbitrary resolutions work
    for submod in (getattr(model.deim, 'encoder', None), getattr(model.deim, 'decoder', None)):
        if submod is not None and hasattr(submod, 'eval_spatial_size'):
            submod.eval_spatial_size = None

    if args.backbone_weights:
        print(f"  Loading pretrained : {args.backbone_weights}")
        model.load_pretrained(args.backbone_weights)
    print(_LINE)

    dummy        = torch.zeros(args.batch_size, 3, H, W)
    dummy_single = torch.zeros(1,               3, H, W)

    # ── Parameters ────────────────────────────────────────────────────────────
    total, trainable = count_params(model)
    frozen           = total - trainable

    print(f"  {'Total params':<32}: {total     / 1e6:.3f} M")
    print(f"  {'Trainable params':<32}: {trainable / 1e6:.3f} M")
    print(f"  {'Frozen params':<32}: {frozen    / 1e6:.3f} M")
    print(_DASH)

    components = (COMPONENTS_DEIMV2_JDE if args.arch == 'deimv2_jde'
                  else COMPONENTS_DEIM_MOT if args.arch == 'deim_mot'
                  else COMPONENTS_HYBRID)

    for attr, label in components:
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            n   = sum(p.numel() for p in obj.parameters())
            tr  = sum(p.numel() for p in obj.parameters() if p.requires_grad)
            frz = '❄' if tr == 0 else ''
            print(f"  {label:<38}: {n / 1e6:>7.3f} M  ({100 * n / total:>4.1f}%) {frz}")
        except AttributeError:
            pass

    # ── Pretrained vs new params (deimv2_jde) ─────────────────────────────────
    if args.arch == 'deimv2_jde':
        print(_DASH)
        n_deim     = sum(p.numel() for p in model.deim.parameters())
        n_obj_head = sum(p.numel() for p in model.obj_head.parameters())
        n_qgen     = sum(p.numel() for p in model.grid_qgen.parameters())
        n_reid     = sum(p.numel() for p in model.reid_mlp.parameters())
        n_new      = n_obj_head + n_qgen + n_reid

        print(f"  {'Pretrained  (deim.*)':<32}: {n_deim     / 1e6:.3f} M  ({100 * n_deim  / total:.1f}%)")
        print(f"  {'New — ObjectnessHead':<32}: {n_obj_head / 1e6:.3f} M  ({100 * n_obj_head / total:.1f}%)")
        print(f"  {'New — GridQueryGen':<32}: {n_qgen     / 1e6:.3f} M  ({100 * n_qgen   / total:.1f}%)")
        print(f"  {'  └─ content proj':<32}: {sum(p.numel() for p in model.grid_qgen.proj.parameters()) / 1e6:.3f} M")
        print(f"  {'  └─ WH head (learned)':<32}: {sum(p.numel() for p in model.grid_qgen.wh_head.parameters()) / 1e6:.3f} M")
        print(f"  {'New — ReID MLP':<32}: {n_reid     / 1e6:.3f} M  ({100 * n_reid   / total:.1f}%)")
        print(f"  {'New total':<32}: {n_new      / 1e6:.3f} M  ({100 * n_new    / total:.1f}%)")

    # ── GFLOPs & GPU memory ───────────────────────────────────────────────────
    print(_DASH)
    gflops, status = try_gflops(model, dummy_single)
    if status == 'ok':
        print(f"  {'GFLOPs (bs=1)':<32}: {gflops:.2f} G")
    else:
        print(f"  GFLOPs                           : [{status}]")

    mb = measure_gpu_mem(model, dummy, device)
    if mb:
        print(f"  {'GPU Mem FP32 peak (fwd)':<32}: {mb:.0f} MB")

    # ── Speed ─────────────────────────────────────────────────────────────────
    if not args.no_speed:
        print(_DASH)
        for half in ([True, False] if args.half else [False]):
            dtype = 'FP16' if half else 'FP32'
            ms, fps = measure_fps(model, dummy, device, args.warmup, args.runs, half)
            lat = ms / args.batch_size
            print(f"  {dtype}  latency/img : {lat:.1f} ms   FPS : {fps:.1f}   "
                  f"(bs={args.batch_size}, {args.runs} runs)")

    # ── Output shapes ─────────────────────────────────────────────────────────
    print(_DASH)
    print("  Output shapes (inference, bs=1):")
    _print_output_shapes(model, dummy_single, args)
    print(f"{_LINE}\n")


if __name__ == '__main__':
    main()
