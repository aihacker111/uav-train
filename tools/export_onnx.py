"""
Export DEIMMotNet to ONNX.

Usage (run from repo root):
    cd src/
    python ../tools/export_onnx.py \
        --deim_config lib/models/configs/deim-uav/deimv2_dinov3_s_coco.yml \
        --weights     ../exp/deim_mot/deim_mot_dinov3_s/model_best.pth \
        --output      ../deim_mot_dinov3_s.onnx \
        --input-h 704 --input-w 1280

Notes:
    - Hook-based S4 lateral works with ONNX tracing because the tracer follows
      actual tensor data flow: stem output → _s4_cache → lateral_s4.
    - Use --dynamic-batch to export with dynamic batch dimension.
    - Use --log-wh if the model was trained with --log_wh (decodes wh inside ONNX).
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ONNX-safe interpolate wrapper ──────────────────────────────────────────────
# torch.onnx.export traces F.interpolate with scale_factor correctly on opset 11+
# but recompute_scale_factor must be False to avoid shape-inference issues.
_orig_interpolate = F.interpolate

def _onnx_safe_interpolate(input, size=None, scale_factor=None, mode='nearest',
                           align_corners=None, recompute_scale_factor=None, antialias=False):
    return _orig_interpolate(input, size=size, scale_factor=scale_factor,
                             mode=mode, align_corners=align_corners,
                             recompute_scale_factor=False)


def _build_model(args):
    _models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'lib', 'models')
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)

    import engine
    from engine.core import YAMLConfig
    from lib.models.networks.deim_uav.model_mot import DEIMMotNet

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

    model = DEIMMotNet(
        deim=deim_model,
        num_classes=args.num_classes,
        hidden_dim=hidden_dim,
        head_conv=args.head_conv,
        reid_dim=args.reid_dim,
    )
    model._init_head_weights(log_wh=args.log_wh)
    return model


class DEIMMotONNX(nn.Module):
    """
    Thin wrapper around DEIMMotNet that:
      1. Returns flat tensors (no list-of-dict) for ONNX compatibility.
      2. Optionally decodes log-space WH inside the graph.
      3. Applies sigmoid to heatmap inside the graph.
    """
    def __init__(self, model: nn.Module, log_wh: bool = False) -> None:
        super().__init__()
        self.model   = model
        self.log_wh  = log_wh

    def forward(self, x):
        out   = self.model(x)[0]       # unwrap list-of-dict
        hm    = out['hm'].sigmoid()    # (B, C, H/4, W/4)
        wh    = out['wh']              # (B, 2, H/4, W/4)
        reg   = out['reg']             # (B, 2, H/4, W/4)
        id_   = out['id']              # (B, reid_dim, H/4, W/4)

        if self.log_wh:
            wh = torch.exp(wh)         # decode log-space inside ONNX

        return hm, wh, reg, id_


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--deim_config', required=True)
    p.add_argument('--weights',     default='',   help='Path to .pth checkpoint')
    p.add_argument('--output',      default='deim_mot.onnx')
    p.add_argument('--num_classes', type=int, default=7)
    p.add_argument('--reid_dim',    type=int, default=128)
    p.add_argument('--head_conv',   type=int, default=64)
    p.add_argument('--input-h',     type=int, default=704)
    p.add_argument('--input-w',     type=int, default=1280)
    p.add_argument('--opset',       type=int, default=17)
    p.add_argument('--dynamic-batch', action='store_true',
                   help='Export with dynamic batch dimension.')
    p.add_argument('--log-wh',      action='store_true',
                   help='Model trained with --log_wh: decode exp() inside ONNX.')
    p.add_argument('--simplify',    action='store_true',
                   help='Run onnxsim after export (pip install onnxsim).')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'\n[export] config  : {args.deim_config}')
    print(f'[export] weights : {args.weights or "none (random)"}')
    print(f'[export] output  : {args.output}')
    print(f'[export] input   : {args.input_h}×{args.input_w}')
    print(f'[export] opset   : {args.opset}  log_wh={args.log_wh}  dynamic_batch={args.dynamic_batch}')

    # ── Build model ──────────────────────────────────────────────────────────────
    model = _build_model(args)

    if args.weights:
        ckpt  = torch.load(args.weights, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[export] loaded {len(state)-len(missing)}/{len(state)} tensors')
        if missing:
            print(f'  missing   : {missing[:4]}{"…" if len(missing)>4 else ""}')
        if unexpected:
            print(f'  unexpected: {unexpected[:4]}{"…" if len(unexpected)>4 else ""}')

    # Clear cached spatial sizes before tracing
    for submod in (getattr(model.deim, 'encoder', None), getattr(model.deim, 'decoder', None)):
        if submod is not None and hasattr(submod, 'eval_spatial_size'):
            submod.eval_spatial_size = None

    model.eval()

    wrapper = DEIMMotONNX(model, log_wh=args.log_wh)
    wrapper.eval()

    dummy = torch.zeros(1, 3, args.input_h, args.input_w)

    # ── Patch F.interpolate for ONNX safety ──────────────────────────────────────
    F.interpolate = _onnx_safe_interpolate

    # ── Dynamic axes ─────────────────────────────────────────────────────────────
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch'},
            'hm':    {0: 'batch'},
            'wh':    {0: 'batch'},
            'reg':   {0: 'batch'},
            'id':    {0: 'batch'},
        }

    # ── Export ───────────────────────────────────────────────────────────────────
    print('[export] tracing model ...')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            opset_version=args.opset,
            input_names=['image'],
            output_names=['hm', 'wh', 'reg', 'id'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    F.interpolate = _orig_interpolate  # restore

    print(f'[export] saved → {args.output}')

    # ── Verify ───────────────────────────────────────────────────────────────────
    try:
        import onnx
        m = onnx.load(args.output)
        onnx.checker.check_model(m)
        print('[export] onnx check: OK')
        print(f'[export] graph inputs : {[i.name for i in m.graph.input]}')
        print(f'[export] graph outputs: {[o.name for o in m.graph.output]}')
    except ImportError:
        print('[export] onnx not installed — skipping check (pip install onnx)')

    # ── Simplify ─────────────────────────────────────────────────────────────────
    if args.simplify:
        try:
            import onnxsim
            print('[export] running onnxsim ...')
            m_simp, ok = onnxsim.simplify(args.output)
            if ok:
                import onnx
                onnx.save(m_simp, args.output)
                print(f'[export] simplified → {args.output}')
            else:
                print('[export] onnxsim: simplification failed, keeping original')
        except ImportError:
            print('[export] onnxsim not installed (pip install onnxsim)')

    print()


if __name__ == '__main__':
    main()
