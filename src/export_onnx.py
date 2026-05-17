"""
Export HybridCenterNetDETR to ONNX.

Usage:
    cd src
    python export_onnx.py \
        --checkpoint /path/to/model_last.pth \
        --arch       hybrid_small \
        --output     ../exported/amot_hybrid.onnx \
        --input-h    608 \
        --input-w    1088 \
        --batch-size 1 \
        --opset      17 \
        [--fp16]     \
        [--simplify] \
        [--verify]

Outputs
-------
  <output>.onnx          — main ONNX model
  <output>.simplified.onnx  — (optional) onnx-simplifier result

ONNX inputs
-----------
  images : float32  (B, 3, H, W)  — ImageNet-normalised

ONNX outputs
------------
  hm           : float32  (B, C, H/4, W/4)   — sigmoid heatmap  (Stage-1)
  wh           : float32  (B, 2, H/4, W/4)   — width/height     (Stage-1)
  reg          : float32  (B, 2, H/4, W/4)   — sub-pixel offset (Stage-1)
  boxes        : float32  (B, K, 4)           — cxcywh [0,1]    (Stage-2 final layer)
  logits       : float32  (B, K, C)           — raw class logits (Stage-2 final layer)
  reid         : float32  (B, K, reid_dim)    — L2-normed embeds (Stage-2)
  query_scores : float32  (B, K)              — heatmap confidence of each query
  query_classes: int64    (B, K)              — stage-1 class index of each query
"""
from __future__ import annotations

import argparse
import os
import sys

# ── make sure src/ is on the path ─────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX wrapper — returns plain tensors instead of dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class HybridONNXWrapper(nn.Module):
    """
    Thin wrapper around HybridCenterNetDETR that:
      1. Calls model.forward() normally.
      2. Unwraps all dataclass outputs into plain Tensors.
      3. Drops auxiliary tensors only needed for training loss (boxes_all,
         logits_all — per-decoder-layer outputs used for auxiliary DETR losses).

    This makes the graph straightforward for ONNX serialisation.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor):
        out = self.model(images)

        s1 = out['stage1']      # CenterNetOutput
        s2 = out['stage2']      # DETROutput

        return (
            s1.hm,                   # (B, C, H/4, W/4)
            s1.wh,                   # (B, 2, H/4, W/4)
            s1.reg,                  # (B, 2, H/4, W/4)
            s2.boxes,                # (B, K, 4)
            s2.logits,               # (B, K, C)
            s2.reid,                 # (B, K, reid_dim)
            out['query_scores'],     # (B, K)
            out['query_classes'],    # (B, K)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _enable_export_mode(model: nn.Module) -> int:
    """
    Call .export() on every MSDeformAttn layer so they use the pure-PyTorch
    reference kernel instead of the custom CUDA extension.

    The pure-PyTorch path uses F.grid_sample + standard ops — fully ONNX-
    exportable with opset >= 16.
    """
    from lib.models.networks.deform_attn.ops.modules.ms_deform_attn import MSDeformAttn

    count = 0
    for m in model.modules():
        if isinstance(m, MSDeformAttn):
            m.export()
            count += 1
    return count


def _build_dummy(batch: int, h: int, w: int, device: torch.device, fp16: bool) -> Tensor:
    # Use randn, NOT zeros.
    # With all-zeros input the heatmap is ~0.01 everywhere (sigmoid of -4.595 bias),
    # so torch.topk has many ties that PyTorch and OnnxRuntime break differently.
    # That produces different query_classes → different content features → totally
    # different Stage-2 outputs even though the ONNX graph is correct.
    # A random input creates a non-degenerate heatmap with clear topk winners.
    dtype = torch.float16 if fp16 else torch.float32
    x = torch.randn(batch, 3, h, w, dtype=torch.float32, device=device)
    # ImageNet normalisation so values are in a realistic range for the ViT
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x * std + mean).clamp(0, 1)
    x = (x - mean) / std
    return x.to(dtype)


def _dynamic_axes(batch_dynamic: bool) -> dict:
    """
    Return dynamic_axes dict for torch.onnx.export.
    The spatial dimensions (H, W) are STATIC — fixing them gives better ONNX
    graph optimisation (constant-folds pos encodings, spatial_shapes, etc.).
    To support multiple resolutions export once per resolution.
    """
    axes: dict = {}
    if batch_dynamic:
        axes['images'] = {0: 'batch'}
        axes['hm']           = {0: 'batch'}
        axes['wh']           = {0: 'batch'}
        axes['reg']          = {0: 'batch'}
        axes['boxes']        = {0: 'batch'}
        axes['logits']       = {0: 'batch'}
        axes['reid']         = {0: 'batch'}
        axes['query_scores'] = {0: 'batch'}
        axes['query_classes']= {0: 'batch'}
    return axes


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════

def export(args) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'[export] device={device}  arch={args.arch}  '
          f'input={args.batch_size}×3×{args.input_h}×{args.input_w}  '
          f'opset={args.opset}  fp16={args.fp16}')

    # ── 1. Build model ──────────────────────────────────────────────────────────
    print('[export] building model …')
    from lib.models.model import create_model, load_model

    model = create_model(args.arch, heads={}, head_conv=-1,
                         reid_dim=getattr(args, 'reid_dim', 256))

    if args.checkpoint:
        print(f'[export] loading checkpoint: {args.checkpoint}')
        model = load_model(model, args.checkpoint)   # returns model (no optimizer)

    # ── 2. Enable ONNX-compatible deformable attention ──────────────────────────
    n_deform = _enable_export_mode(model)
    print(f'[export] set {n_deform} MSDeformAttn layers to export mode (pure-PyTorch kernel)')

    # ── 3. Wrap + eval + optional fp16 ─────────────────────────────────────────
    wrapper = HybridONNXWrapper(model).to(device).eval()
    if args.fp16:
        wrapper = wrapper.half()
        print('[export] model cast to float16')

    # ── 4. Dummy forward pass to validate ──────────────────────────────────────
    dummy = _build_dummy(args.batch_size, args.input_h, args.input_w, device, args.fp16)
    print('[export] running dummy forward … ', end='', flush=True)
    with torch.no_grad():
        outs = wrapper(dummy)
    print('ok')
    names = ['hm', 'wh', 'reg', 'boxes', 'logits', 'reid', 'query_scores', 'query_classes']
    for name, t in zip(names, outs):
        print(f'         {name:15s} {tuple(t.shape)}  dtype={t.dtype}')

    # ── 5. Export ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f'[export] exporting to {args.output} …')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            args.output,
            input_names  = ['images'],
            output_names = names,
            dynamic_axes = _dynamic_axes(batch_dynamic=not args.static_batch),
            opset_version        = args.opset,
            do_constant_folding  = True,
            export_params        = True,
            verbose              = False,
        )
    size_mb = os.path.getsize(args.output) / 1e6
    print(f'[export] saved → {args.output}  ({size_mb:.1f} MB)')

    # ── 6. Optional: onnx-simplifier ───────────────────────────────────────────
    simplified_path = args.output.replace('.onnx', '.simplified.onnx')
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify as onnxsim_simplify
            print('[simplify] running onnxsim …', end='', flush=True)
            model_onnx = onnx.load(args.output)
            model_sim, ok = onnxsim_simplify(model_onnx)
            if ok:
                onnx.save(model_sim, simplified_path)
                size_mb2 = os.path.getsize(simplified_path) / 1e6
                print(f' ok → {simplified_path}  ({size_mb2:.1f} MB)')
            else:
                print(' FAILED (keeping original)')
        except ImportError:
            print('[simplify] onnxsim not installed — skip  (pip install onnxsim)')

    # ── 7. Optional: onnxruntime verification ──────────────────────────────────
    if args.verify:
        _verify(args.output, dummy.cpu().numpy(), outs, args)

    return args.output


# ═══════════════════════════════════════════════════════════════════════════════
# Verify
# ═══════════════════════════════════════════════════════════════════════════════

def _verify(onnx_path: str, np_input, torch_outs, args):
    print('[verify] checking with onnxruntime …')
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print('[verify] onnxruntime not installed — skip  (pip install onnxruntime)')
        return

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if torch.cuda.is_available() and not args.cpu else ['CPUExecutionProvider']

    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_outs = sess.run(None, {'images': np_input.astype('float32')})

    # Thresholds: integer outputs (query_classes) must match exactly;
    # float32 outputs allow small fp rounding; fp16 exports allow more slack.
    float_tol = 5e-3 if args.fp16 else 1e-3
    # integer outputs: exact match required (diff == 0)
    int_outputs = {'query_classes'}

    names = ['hm', 'wh', 'reg', 'boxes', 'logits', 'reid', 'query_scores', 'query_classes']
    all_ok = True
    for name, ort_t, pt_t in zip(names, ort_outs, torch_outs):
        pt_np  = pt_t.cpu().float().numpy()
        ort_f  = ort_t.astype('float32')
        diff   = np.abs(ort_f - pt_np).max()
        mean_d = np.abs(ort_f - pt_np).mean()

        if name in int_outputs:
            ok = diff == 0.0
        else:
            ok = diff < float_tol

        status = '✓' if ok else '✗'
        print(f'  {status}  {name:15s}  max_diff={diff:.2e}  mean_diff={mean_d:.2e}')
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print('[verify] all outputs match  ✓')
    else:
        print('[verify] WARNING: outputs differ.')
        print('         Possible causes:')
        print('           • query_classes diff > 0  → topk tie-breaking (use --randn dummy, not zeros)')
        print('           • boxes/logits/reid diff   → follows from wrong query_classes (same root cause)')
        print('           • fp16 large diff          → try exporting in float32 first')
        print('           • opset mismatch           → try --opset 16 or 17')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Export HybridCenterNetDETR to ONNX')

    p.add_argument('--checkpoint', type=str, default='',
                   help='Path to .pth checkpoint (leave empty to export random weights)')
    p.add_argument('--arch', type=str, default='hybrid_small',
                   choices=['hybrid_tiny', 'hybrid_small', 'hybrid_base'],
                   help='Model variant')
    p.add_argument('--output', type=str, default='../exported/amot_hybrid.onnx',
                   help='Output .onnx path')
    p.add_argument('--input-h', type=int, default=512,
                   help='Input height (must be divisible by 16)')
    p.add_argument('--input-w', type=int, default=832,
                   help='Input width (must be divisible by 16)')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size for the dummy input')
    p.add_argument('--opset', type=int, default=17,
                   help='ONNX opset version (17 recommended)')
    p.add_argument('--reid-dim', type=int, default=256,
                   help='ReID embedding dimension (must match training)')
    p.add_argument('--fp16', action='store_true',
                   help='Export in float16 (TensorRT-friendly)')
    p.add_argument('--simplify', action='store_true',
                   help='Run onnx-simplifier after export (pip install onnxsim)')
    p.add_argument('--verify', action='store_true',
                   help='Verify outputs with onnxruntime (pip install onnxruntime)')
    p.add_argument('--static-batch', action='store_true',
                   help='Fix batch dim as static (smaller graph, faster on some runtimes)')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU even if CUDA is available')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Validate input dimensions
    assert args.input_h % 16 == 0, f'--input-h must be divisible by 16, got {args.input_h}'
    assert args.input_w % 16 == 0, f'--input-w must be divisible by 16, got {args.input_w}'

    export(args)
