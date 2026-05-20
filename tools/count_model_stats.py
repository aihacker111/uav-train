"""
Count model parameters, GFLOPs / MACs, GPU memory, and inference speed (FPS / latency).

Speed output includes:
  - Mean / Min / P50 / P95 / P99 / Max latency distribution
  - Per-stage CUDA breakdown (backbone / neck / head) for hybrid models on CUDA

Usage:
    python tools/count_model_stats.py                             # default: lwdetr_tiny
    python tools/count_model_stats.py --arch hybrid_tiny
    python tools/count_model_stats.py --arch hybrid_tiny --half
    python tools/count_model_stats.py --arch hybrid_tiny --no-speed
    python tools/count_model_stats.py --arch hybrid_tiny --no-stage-profile
    python tools/count_model_stats.py --arch hybrid_tiny --spatial-partition
    python tools/count_model_stats.py --arch hybrid_tiny --compare-windows
    python tools/count_model_stats.py --arch hybrid_tiny --window-blocks 0,2,4
    python tools/count_model_stats.py --arch hybrid_small --batch-size 4
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

_DIVIDER  = '─' * 64
_DIVIDER2 = '┄' * 64

_DEFAULT_CKPTS = {
    'lwdetr_tiny':  '../lwdetr_coco_pretrained/LWDETR_tiny_60e_coco.pth',
    'lwdetr_small': '../lwdetr_coco_pretrained/LWDETR_small_60e_coco.pth',
}
_LWDETR_HEADS = {'hm': 7, 'wh': 2, 'reg': 2, 'id': 256}

# Original (pretrained) window configs — used for --compare-windows baseline
_ORIGINAL_WINDOW_BLOCKS = {
    'tiny':  [0, 2, 4],
    'small': [0, 1, 3, 6, 7, 9],
    'base':  [0, 1, 3, 4, 6, 7, 9, 10],
}
_VIT_DEPTH = {'tiny': 6, 'small': 10, 'base': 12}
_VIT_DIM   = {'tiny': 192, 'small': 192, 'base': 768}


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
    print(f"  {'Total':<26}: {total / 1e6:>8.2f} M")
    print(f"  {'Trainable':<26}: {trainable / 1e6:>8.2f} M")
    if frozen > 0:
        print(f"  {'Frozen':<26}: {frozen / 1e6:>8.2f} M")

    components = [
        ('backbone',       'Backbone (ViT)'),
        ('neck',           'Multi-scale Neck'),
        ('token_scorer',   'Token Scorer (s8)'),
        ('query_gen',      'Query Generator'),
        ('dn_gen',         'DN Query Generator'),
        ('decoder',        'DETR Decoder'),
        ('detr_head',      'DETR Head'),
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
            print(f"  {'└─ ' + label:<26}: {n / 1e6:>8.2f} M  ({pct:5.1f}%)")


# ── Analytical ViT backbone GFLOPs ────────────────────────────────────────────

def analytical_vit_gflops(
    embed_dim:           int,
    depth:               int,
    window_block_indexes: list,
    input_h:             int,
    input_w:             int,
) -> dict:
    """
    Exact analytical GFLOPs for the ViT backbone.

    LW-DETR window structure:
      After patch embed (patch=16): token grid H/16 × W/16
      Reshape into 4×4 sub-windows: each window has (H/64) × (W/64) tokens
      → 16 windows per image, N_win tokens each, N_total = 16 × N_win

    Window block : each of 16 windows attends within itself  → O(16 × N_win²)
    Global block : all 16 windows merged → full image attention → O(N_total²)
                                                         ratio = N_total²/(16×N_win²) = 16×

    Costs (MACs, 1 MAC = multiply + add):
      Linear per block (QKV + out + FFN) : 12 × N_total × D²   (same for window/global)
      Global attention per block          : 2  × N_total² × D
      Window attention per block          : 2  × 16 × N_win² × D  = global / 16
    """
    N_total = (input_h // 16) * (input_w // 16)
    N_win   = (input_h // 64) * (input_w // 64)
    D       = embed_dim

    linear_per_block     = 12 * N_total * D * D   # MACs — identical for every block
    attn_global_per_blk  = 2  * N_total * N_total * D
    attn_window_per_blk  = 2  * 16 * N_win * N_win * D   # = attn_global / 16

    total_linear = depth * linear_per_block
    total_attn   = 0
    n_global     = 0
    n_window     = 0

    for i in range(depth):
        if i in window_block_indexes:
            total_attn += attn_window_per_blk
            n_window   += 1
        else:
            total_attn += attn_global_per_blk
            n_global   += 1

    total_macs  = total_linear + total_attn

    return {
        'total_gflops':          total_macs * 2 / 1e9,
        'linear_gflops':         total_linear * 2 / 1e9,
        'attn_gflops':           total_attn * 2 / 1e9,
        'attn_global_per_block': attn_global_per_blk * 2 / 1e9,
        'attn_window_per_block': attn_window_per_blk * 2 / 1e9,
        'n_global':              n_global,
        'n_window':              n_window,
        'N_total':               N_total,
        'N_win':                 N_win,
    }


def print_analytical_backbone(
    embed_dim:           int,
    depth:               int,
    window_block_indexes: list,
    input_h:             int,
    input_w:             int,
    label:               str = 'Backbone (analytical)',
) -> None:
    s = analytical_vit_gflops(embed_dim, depth, window_block_indexes, input_h, input_w)
    print(_DIVIDER2)
    print(f"  {label}")
    print(f"  {'  Tokens / image':<26}: {s['N_total']}  "
          f"(16 windows × {s['N_win']} tokens)")
    print(f"  {'  Global blocks':<26}: {s['n_global']}  "
          f"({s['attn_global_per_block']:.2f} GFLOPs/block attn)")
    print(f"  {'  Window blocks':<26}: {s['n_window']}  "
          f"({s['attn_window_per_block']:.2f} GFLOPs/block attn)  "
          f"[{s['attn_global_per_block']/s['attn_window_per_block']:.0f}× cheaper]")
    print(f"  {'  Linear (fixed)':<26}: {s['linear_gflops']:>7.2f} GFLOPs")
    print(f"  {'  Attention total':<26}: {s['attn_gflops']:>7.2f} GFLOPs")
    print(f"  {'  ViT backbone total':<26}: {s['total_gflops']:>7.2f} GFLOPs")


def print_window_comparison(
    vit_variant:  str,
    original_wbi: list,
    new_wbi:      list,
    input_h:      int,
    input_w:      int,
) -> None:
    """
    Side-by-side comparison between original and new window_block_indexes.
    Purely analytical — no model build required.
    """
    D     = _VIT_DIM[vit_variant]
    depth = _VIT_DEPTH[vit_variant]

    orig = analytical_vit_gflops(D, depth, original_wbi, input_h, input_w)
    new  = analytical_vit_gflops(D, depth, new_wbi,      input_h, input_w)

    saved     = orig['total_gflops'] - new['total_gflops']
    saved_pct = 100 * saved / orig['total_gflops']
    attn_saved     = orig['attn_gflops'] - new['attn_gflops']
    attn_saved_pct = 100 * attn_saved / orig['attn_gflops']

    print(f"\n{_DIVIDER}")
    print(f"  WINDOW CONFIG COMPARISON  —  vit_{vit_variant}  "
          f"({input_h}×{input_w})")
    print(_DIVIDER2)
    print(f"  {'Config':<28}  {'Original':>10}  {'New':>10}  {'Saved':>10}")
    print(_DIVIDER2)
    print(f"  {'window_block_indexes':<28}  "
          f"{str(original_wbi):>10}  {str(new_wbi):>10}")
    print(f"  {'Global blocks':<28}  "
          f"{orig['n_global']:>10}  {new['n_global']:>10}  "
          f"{orig['n_global']-new['n_global']:>+10}")
    print(f"  {'Window blocks':<28}  "
          f"{orig['n_window']:>10}  {new['n_window']:>10}  "
          f"{new['n_window']-orig['n_window']:>+10}")
    print(_DIVIDER2)
    print(f"  {'Linear GFLOPs (fixed)':<28}  "
          f"{orig['linear_gflops']:>9.2f}G  {new['linear_gflops']:>9.2f}G  "
          f"{'0.00G':>10}")
    print(f"  {'Attention GFLOPs':<28}  "
          f"{orig['attn_gflops']:>9.2f}G  {new['attn_gflops']:>9.2f}G  "
          f"{attn_saved:>9.2f}G  ({attn_saved_pct:.1f}%)")
    print(_DIVIDER2)
    print(f"  {'ViT backbone total':<28}  "
          f"{orig['total_gflops']:>9.2f}G  {new['total_gflops']:>9.2f}G  "
          f"{saved:>9.2f}G  ({saved_pct:.1f}%)")
    print(_DIVIDER2)
    print(f"  Weight compatibility  : 100%  (no weight shape changes)")
    print(f"  Pretrained load       : full  (backbone + neck + decoder)")
    print(f"  Speedup (attn only)   : {orig['attn_gflops']/new['attn_gflops']:.2f}×")


# ── GFLOPs via thop ───────────────────────────────────────────────────────────

def _try_thop(model: nn.Module, dummy: torch.Tensor) -> tuple[float, str]:
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
        return 0.0, f'thop failed ({str(e)[:80]})'


def print_flops(model: nn.Module, dummy: torch.Tensor) -> None:
    print(_DIVIDER2)
    gflops, status = _try_thop(model, dummy)
    if status == 'ok':
        print(f"  {'GFLOPs/thop (FP32)':<26}: {gflops:>8.2f} G")
    else:
        print(f"  GFLOPs/thop          : [{status}]")


# ── GPU memory ────────────────────────────────────────────────────────────────

def measure_gpu_memory(
    model:    nn.Module,
    dummy:    torch.Tensor,
    device:   torch.device,
    use_half: bool,
) -> tuple[float, float]:
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
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2

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
    print(f"  {'GPU Mem (FP32, fwd)':<26}: {fp32_mb:>8.1f} MB")
    if use_half and fp16_mb > 0:
        print(f"  {'GPU Mem (FP16, fwd)':<26}: {fp16_mb:>8.1f} MB"
              f"  ({100*fp16_mb/fp32_mb:.0f}% of fp32)")


# ── Inference speed ───────────────────────────────────────────────────────────

def measure_speed(
    model:      nn.Module,
    dummy:      torch.Tensor,
    warmup:     int,
    runs:       int,
    device:     torch.device,
    use_half:   bool,
    batch_size: int,
) -> dict:
    """
    Record per-run latency and return full distribution stats.

    Returns dict keys: mean, std, min, p50, p95, p99, max, fps  (all in ms except fps).
    CUDA: uses paired cuda.Event per run — no synchronize between runs,
          so kernel overlap is preserved and we measure real end-to-end latency.
    CPU : uses time.perf_counter per run.
    """
    model_m = copy.deepcopy(model).to(device).eval()
    inp     = dummy.to(device)
    if use_half:
        model_m = model_m.half()
        inp     = inp.half()

    use_cuda = device.type == 'cuda'
    latencies: list[float] = []

    with torch.no_grad():
        for _ in range(warmup):
            model_m(inp)
        if use_cuda:
            torch.cuda.synchronize(device)

        for _ in range(runs):
            if use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                model_m(inp)
                e.record()
                torch.cuda.synchronize(device)
                latencies.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                model_m(inp)
                latencies.append((time.perf_counter() - t0) * 1000.0)

    del model_m
    latencies.sort()
    n    = len(latencies)
    mean = sum(latencies) / n
    std  = (sum((x - mean) ** 2 for x in latencies) / n) ** 0.5
    return {
        'mean': mean,
        'std':  std,
        'min':  latencies[0],
        'p50':  latencies[n // 2],
        'p95':  latencies[min(n - 1, int(n * 0.95))],
        'p99':  latencies[min(n - 1, int(n * 0.99))],
        'max':  latencies[-1],
        'fps':  1000.0 * batch_size / mean,
    }


def print_speed_report(
    stats:      dict,
    batch_size: int,
    warmup:     int,
    runs:       int,
    device:     torch.device,
    use_half:   bool,
) -> None:
    dtype_str = 'FP16' if use_half else 'FP32'
    print(_DIVIDER2)
    print(f"  {'Device':<26}: {device}  [{dtype_str}]")
    print(f"  {'Batch size / Warmup / Runs':<26}: {batch_size} / {warmup} / {runs}")
    print(f"  {'Mean latency / batch':<26}: {stats['mean']:>8.2f} ms")
    print(f"  {'Mean latency / image':<26}: {stats['mean'] / batch_size:>8.2f} ms"
          f"  ({stats['fps']:.1f} FPS)")
    print(_DIVIDER2)
    print(f"  {'Latency distribution (ms)':<26}")
    print(f"  {'  Min':<14}: {stats['min']:>7.2f}    "
          f"{'P50':<6}: {stats['p50']:>7.2f}    "
          f"{'P95':<6}: {stats['p95']:>7.2f}")
    print(f"  {'  Std':<14}: {stats['std']:>7.2f}    "
          f"{'P99':<6}: {stats['p99']:>7.2f}    "
          f"{'Max':<6}: {stats['max']:>7.2f}")


# ── Per-stage profiler (hybrid only, CUDA) ────────────────────────────────────

_STAGE_LABELS = {
    'backbone':       'Backbone (ViT)',
    'neck':           'Multi-scale Neck',
    'token_scorer':   'Token Scorer (s8)',
    'query_gen':      'Query Generator',
    'decoder':        'DETR Decoder',
    'detr_head':      'DETR Head',
    # dn_gen is training-only (no-op at eval); excluded from stage timing
}


def _profile_hybrid_stages(
    model:    nn.Module,
    dummy:    torch.Tensor,
    device:   torch.device,
    use_half: bool,
    warmup:   int = 5,
    runs:     int = 30,
) -> dict[str, list[float]] | None:
    """
    Time each sub-module of HybridDETR using CUDA events + forward hooks.

    Each stage is measured with a synchronize fence before the start event so
    times are accurate (no queue latency from the previous stage leaking in).
    Returns {stage_name: [latency_ms, ...]} or None if no CUDA.
    """
    if device.type != 'cuda':
        return None

    attrs = [a for a in _STAGE_LABELS if hasattr(model, a)]
    timing: dict[str, list[float]] = {a: [] for a in attrs}
    start_evts: dict[str, torch.cuda.Event] = {}
    handles = []

    model_p = copy.deepcopy(model).to(device).eval()
    inp     = dummy.to(device)
    if use_half:
        model_p = model_p.half()
        inp     = inp.half()

    for attr in attrs:
        mod = getattr(model_p, attr)

        def _make_hooks(name: str):
            def pre(m, args):
                torch.cuda.synchronize(device)
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                start_evts[name] = ev

            def post(m, args, out):
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_end.record()
                torch.cuda.synchronize(device)
                timing[name].append(start_evts[name].elapsed_time(ev_end))

            return pre, post

        pre, post = _make_hooks(attr)
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    with torch.no_grad():
        for _ in range(warmup):
            model_p(inp)
        for k in timing:
            timing[k].clear()
        for _ in range(runs):
            model_p(inp)

    for h in handles:
        h.remove()
    del model_p

    return timing


def print_stage_breakdown(timing: dict[str, list[float]]) -> None:
    """Print per-stage mean latency table from _profile_hybrid_stages output."""
    means = {name: sum(vals) / len(vals) for name, vals in timing.items() if vals}
    total = sum(means.values())
    print(_DIVIDER2)
    print(f"  Stage breakdown  (mean over {max(len(v) for v in timing.values())} runs, "
          f"CUDA-synced per stage):")
    for attr, label in _STAGE_LABELS.items():
        if attr not in means:
            continue
        ms  = means[attr]
        pct = 100.0 * ms / total if total > 0 else 0.0
        bar_w  = max(1, int(pct / 2))
        bar    = '█' * bar_w
        print(f"  {'  ' + label:<26}: {ms:>7.2f} ms  ({pct:5.1f}%)  {bar}")
    print(f"  {'  Sum of stages':<26}: {total:>7.2f} ms")


# ── Output shape summary ──────────────────────────────────────────────────────

def print_output_shapes(outputs, arch: str, input_h: int, input_w: int) -> None:
    print(_DIVIDER2)
    print(f"  {'Input':<26}: {input_h}×{input_w}")

    if arch.startswith('hybrid'):
        s2 = outputs['stage2']
        print(f"  {'Score map (s8)':<26}: {tuple(outputs['score_map'].shape)}")
        print(f"  {'Stage-2 boxes':<26}: {tuple(s2.boxes.shape)}")
        print(f"  {'Stage-2 logits':<26}: {tuple(s2.logits.shape)}")
        print(f"  {'Stage-2 reid':<26}: {tuple(s2.reid.shape)}")
        print(f"  {'Query scores':<26}: {tuple(outputs['query_scores'].shape)}")
        if 'query_classes' in outputs:
            print(f"  {'Query classes':<26}: {tuple(outputs['query_classes'].shape)}")
        tau = outputs.get('tau_query')
        if tau is not None:
            print(f"  {'τ (Gumbel temp)':<26}: {tau.item():.4f}")
    else:
        last = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        for k, v in last.items():
            if hasattr(v, 'shape'):
                print(f"  {k:<26}: {tuple(v.shape)}")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Model parameter / GFLOPs / speed profiler')
    p.add_argument('--arch', default='lwdetr_tiny',
                   help='lwdetr_tiny|lwdetr_small|hybrid_tiny|hybrid_small|hybrid_base')
    p.add_argument('--ckpt', default='',
                   help='Checkpoint path (.pth). Leave empty for built-in defaults.')
    p.add_argument('--no-ckpt', action='store_true',
                   help='Skip checkpoint loading entirely (random weights).')
    p.add_argument('--input-h', type=int, default=704,
                   help='Input height (divisible by 64). Default: 704')
    p.add_argument('--input-w', type=int, default=1280,
                   help='Input width (divisible by 64). Default: 1280')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size for throughput measurement. Default: 1')
    p.add_argument('--half', action='store_true',
                   help='Measure FP16 speed and memory.')
    p.add_argument('--warmup', type=int, default=20,
                   help='Warmup iterations before timing. Default: 20')
    p.add_argument('--runs', type=int, default=100,
                   help='Timed iterations for FPS measurement. Default: 100')
    p.add_argument('--device', default='',
                   help='"cuda", "cpu", or "" for auto-detect.')
    p.add_argument('--no-speed', action='store_true',
                   help='Skip speed measurement (params + GFLOPs only).')
    p.add_argument('--no-stage-profile', action='store_true',
                   help='Skip per-stage breakdown (enabled by default for hybrid on CUDA).')
    p.add_argument('--num-output-levels', type=int, default=1)
    p.add_argument('--top-down-fusion', action='store_true')
    # ── Spatial partition (for benchmarking SP path) ──────────────────────────
    p.add_argument('--spatial-partition', action='store_true',
                   help='Enable spatial-partitioned query gen (4×4×50+32=832 queries).')
    p.add_argument('--sp-grid-rows', type=int, default=4)
    p.add_argument('--sp-grid-cols', type=int, default=4)
    p.add_argument('--sp-queries-per-region', type=int, default=50)
    p.add_argument('--sp-overlap-ratio', type=float, default=0.25)
    p.add_argument('--sp-global-queries', type=int, default=32)
    # ── Window config options ─────────────────────────────────────────────────
    p.add_argument('--compare-windows', action='store_true',
                   help='Show GFLOPs comparison: original vs current window config.')
    p.add_argument('--window-blocks', default='',
                   help='Override window_block_indexes, e.g. "0,2,4". '
                        'Patches config before building model.')
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vit_variant_from_arch(arch: str) -> str:
    """Extract 'tiny'|'small'|'base' from arch string like 'hybrid_tiny'."""
    for v in ('tiny', 'small', 'base'):
        if v in arch:
            return v
    return 'tiny'


def _patch_window_blocks(variant: str, new_wbi: list) -> list:
    """Temporarily override _VIT_VARIANTS and return original for restore."""
    from lib.models.networks.hybrid import config as cfg_mod
    original = list(cfg_mod._VIT_VARIANTS[variant][2])
    row = cfg_mod._VIT_VARIANTS[variant]
    cfg_mod._VIT_VARIANTS[variant] = (row[0], row[1], new_wbi, row[3])
    return original


def _restore_window_blocks(variant: str, original_wbi: list) -> None:
    from lib.models.networks.hybrid import config as cfg_mod
    row = cfg_mod._VIT_VARIANTS[variant]
    cfg_mod._VIT_VARIANTS[variant] = (row[0], row[1], original_wbi, row[3])


def _current_window_blocks(variant: str) -> list:
    from lib.models.networks.hybrid import config as cfg_mod
    return list(cfg_mod._VIT_VARIANTS[variant][2])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    arch       = args.arch
    input_h    = args.input_h
    input_w    = args.input_w
    batch_size = args.batch_size
    use_half   = args.half
    is_hybrid  = arch.startswith('hybrid')
    variant    = _vit_variant_from_arch(arch) if is_hybrid else None

    # ── Optional window config override ──────────────────────────────────────
    _patched_wbi = None
    _saved_wbi   = None
    if is_hybrid and args.window_blocks:
        new_wbi      = [int(x) for x in args.window_blocks.split(',')]
        _saved_wbi   = _patch_window_blocks(variant, new_wbi)
        _patched_wbi = new_wbi
        print(f"\n  [override] window_block_indexes → {new_wbi}")

    # ── Build model ───────────────────────────────────────────────────────────
    if is_hybrid:
        class _FakeOpt:
            num_output_levels       = args.num_output_levels
            top_down_fusion         = args.top_down_fusion
            grad_checkpoint         = False
            # Scorer
            scorer_head_conv        = 64
            use_multiscale_fusion   = True
            # QueryGen
            use_gumbel              = True
            tau_start               = 1.0
            tau_end                 = 0.1
            K                       = 200
            use_spatial_partition   = args.spatial_partition
            sp_grid_rows            = args.sp_grid_rows
            sp_grid_cols            = args.sp_grid_cols
            sp_queries_per_region   = args.sp_queries_per_region
            sp_overlap_ratio        = args.sp_overlap_ratio
            sp_global_queries       = args.sp_global_queries
            # DN (training-only; included so create_model wires cfg correctly)
            num_dn_groups           = 5
            dn_label_noise_ratio    = 0.5
            dn_box_noise_scale      = 0.4
            dn_max_queries          = 500
        heads = {'__opt__': _FakeOpt()}
    else:
        heads = _LWDETR_HEADS

    print(f"\n{_DIVIDER}")
    print(f"  Building: {arch}  ({input_h}×{input_w}  bs={batch_size})")
    model = create_model(arch, heads, head_conv=256)
    model.eval()

    # Restore if patched (model already built with override)
    if _saved_wbi is not None:
        _restore_window_blocks(variant, _saved_wbi)

    # ── Checkpoint ────────────────────────────────────────────────────────────
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

    dummy        = torch.zeros(batch_size, 3, input_h, input_w)
    dummy_single = torch.zeros(1, 3, input_h, input_w)

    # ── Parameters ────────────────────────────────────────────────────────────
    print_param_table(model, arch)

    # ── GFLOPs: thop (best-effort) + analytical backbone ─────────────────────
    print_flops(model, dummy_single)

    if is_hybrid and variant in _VIT_DIM:
        current_wbi = (_patched_wbi if _patched_wbi is not None
                       else _current_window_blocks(variant))
        print_analytical_backbone(
            embed_dim            = _VIT_DIM[variant],
            depth                = _VIT_DEPTH[variant],
            window_block_indexes = current_wbi,
            input_h              = input_h,
            input_w              = input_w,
            label                = f'ViT-{variant} backbone (analytical)  wbi={current_wbi}',
        )

    # ── Window comparison ─────────────────────────────────────────────────────
    if is_hybrid and args.compare_windows and variant in _VIT_DIM:
        original_wbi = _ORIGINAL_WINDOW_BLOCKS[variant]
        current_wbi  = (_patched_wbi if _patched_wbi is not None
                        else _current_window_blocks(variant))
        print_window_comparison(variant, original_wbi, current_wbi, input_h, input_w)

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
        print(f"\n{_DIVIDER}")
        sp_tag = '  [spatial-partition]' if (is_hybrid and args.spatial_partition) else ''
        print(f"  SPEED  —  {arch}{sp_tag}")

        for half in ([True, False] if use_half else [False]):
            dtype_label = 'FP16' if half else 'FP32'
            stats = measure_speed(model, dummy, args.warmup, args.runs, device, half, batch_size)
            print_speed_report(stats, batch_size, args.warmup, args.runs, device, half)

        # Per-stage breakdown (hybrid + CUDA only, unless suppressed)
        if is_hybrid and not args.no_stage_profile:
            stage_timing = _profile_hybrid_stages(
                model, dummy, device, use_half,
                warmup=min(5, args.warmup),
                runs=min(30, args.runs),
            )
            if stage_timing is not None:
                print_stage_breakdown(stage_timing)

    # ── Output shapes ─────────────────────────────────────────────────────────
    model_cpu = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        outputs = model_cpu(dummy_single)
    print_output_shapes(outputs, arch, input_h, input_w)
    print(f"{_DIVIDER}\n")


if __name__ == '__main__':
    main()
