"""
Export HybridECDet to ONNX using the same opts as training.

Pass the identical flags you used for training (minus dataset / logging ones).
The script builds the model exactly like train.py, then wraps it for ONNX.

Example — mirrors your training command:
    python src/export_onnx.py \
        --load_model    exp/hybrid/hybrid_ecdet_s_visdrone/model_best.pth \
        --task          hybrid \
        --arch          hybrid_ecdet \
        --ecdet_config  src/lib/models/configs/ecdet_s_uav.yml \
        --num_classes   10 \
        --head_conv     32 \
        --reid_dim      0 \
        --input_w       832 \
        --input_h       512 \
        --output        model.onnx \
        --opset         17

Add --dynamic for dynamic batch size.
Add --gpu to run export on GPU (default: cpu).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _init_paths  # noqa: F401

import torch
import torch.nn as nn
from lib.models.model import create_model, load_model


# ── ONNX wrapper ───────────────────────────────────────────────────────────────

class ExportWrapper(nn.Module):
    """
    Forward returns plain tensors (no dicts / dataclasses) so ONNX can trace it.

    Outputs:
        boxes  (B, Q, 4)        Stage-2 cxcywh boxes, normalised [0, 1]
        scores (B, Q, C)        Stage-2 class scores (after sigmoid)
        hm     (B, C, H/4, W/4) Stage-1 CenterNet heatmap (after sigmoid)
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out    = self.model(x)
        boxes  = out['stage2'].boxes
        scores = out['stage2'].logits.sigmoid()
        hm     = out['stage1'].hm
        return boxes, scores, hm


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Export HybridECDet to ONNX')

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument('--load_model',  required=True,
                   help='Path to .pth checkpoint')
    p.add_argument('--output',      default='model.onnx',
                   help='Output .onnx path  (default: model.onnx)')

    # ── model — copy from your training command ────────────────────────────────
    p.add_argument('--task',         default='hybrid')
    p.add_argument('--arch',         default='hybrid_ecdet')
    p.add_argument('--ecdet_config', default='src/lib/models/configs/ecdet_s_uav.yml',
                   help='ECDet YAML config — same as training')
    p.add_argument('--num_classes',  type=int, default=10)
    p.add_argument('--head_conv',    type=int, default=32)
    p.add_argument('--reid_dim',     type=int, default=0,
                   help='0 = disable ReID head for export (recommended)')

    # ── input ─────────────────────────────────────────────────────────────────
    p.add_argument('--input_w',  type=int, default=832)
    p.add_argument('--input_h',  type=int, default=512)

    # ── export settings ───────────────────────────────────────────────────────
    p.add_argument('--opset',   type=int,        default=17)
    p.add_argument('--dynamic', action='store_true',
                   help='Export with dynamic batch-size axis')
    p.add_argument('--gpu',     action='store_true',
                   help='Run export on GPU (default: CPU)')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # ── Build opt namespace — same fields create_model / load_model need ───────
    # Mirrors what train.py does: opts().parse() → update_dataset_info_and_set_heads
    # We skip the dataset step and set the fields manually.
    import types
    opt              = types.SimpleNamespace()
    opt.task         = args.task
    opt.arch         = args.arch
    opt.ecdet_config = args.ecdet_config
    opt.head_conv    = args.head_conv
    opt.reid_dim     = args.reid_dim
    opt.num_classes  = args.num_classes
    # heads dict — same structure as train.py sets it
    opt.heads = {
        'hm': args.num_classes,
        'wh': 2,
        'id': args.reid_dim,
    }

    # ── Create model (same call as train.py line 197-200) ─────────────────────
    _heads = dict(opt.heads, **{'__opt__': opt})   # inject opt so create_model reads ecdet_config
    print(f'[build]  arch={opt.arch}  ecdet_config={opt.ecdet_config}')
    print(f'         num_classes={opt.num_classes}  head_conv={opt.head_conv}  reid_dim={opt.reid_dim}')

    model = create_model(opt.arch, _heads, opt.head_conv,
                         reid_dim=opt.reid_dim,
                         num_classes=opt.num_classes)
    model = load_model(model, args.load_model)
    model = model.to(device).eval()

    wrapper = ExportWrapper(model)

    # ── Dummy input ────────────────────────────────────────────────────────────
    dummy = torch.zeros(1, 3, args.input_h, args.input_w, device=device)

    print(f'[input]  shape={tuple(dummy.shape)}  device={device}')

    # ── Forward trace (verify before export) ──────────────────────────────────
    with torch.no_grad():
        boxes, scores, hm = wrapper(dummy)
    print(f'[trace]  boxes={tuple(boxes.shape)}  scores={tuple(scores.shape)}  hm={tuple(hm.shape)}')

    # ── ONNX export ───────────────────────────────────────────────────────────
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'image':  {0: 'batch'},
            'boxes':  {0: 'batch'},
            'scores': {0: 'batch'},
            'hm':     {0: 'batch'},
        }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        opset_version       = args.opset,
        input_names         = ['image'],
        output_names        = ['boxes', 'scores', 'hm'],
        dynamic_axes        = dynamic_axes,
        do_constant_folding = True,
    )

    size_mb = os.path.getsize(args.output) / 1e6
    print(f'[export] {args.output}  ({size_mb:.1f} MB)  opset={args.opset}  dynamic={args.dynamic}')

    # ── Verify with ONNXRuntime (optional) ────────────────────────────────────
    try:
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu else ['CPUExecutionProvider']
        sess = ort.InferenceSession(args.output, providers=providers)
        feed = {sess.get_inputs()[0].name: dummy.cpu().numpy()}
        outs = sess.run(None, feed)
        print(f'[onnxrt] OK  outputs: {[o.shape for o in outs]}')
    except ImportError:
        print('[onnxrt] skipped — pip install onnxruntime to verify')


if __name__ == '__main__':
    main()
