"""
Save a random-weight DEIMMotNet checkpoint then export it to ONNX.
Useful for testing the export pipeline without a real trained checkpoint.

Usage (run from repo root):
    cd src/
    python ../tools/save_dummy_and_export.py \
        --deim_config lib/models/configs/deim-uav/deimv2_dinov3_s_coco.yml \
        --output      ../dummy_deim_mot.onnx
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F

_LINE = '─' * 60


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--deim_config', required=True)
    p.add_argument('--output',      default='../dummy_deim_mot.onnx')
    p.add_argument('--num_classes', type=int, default=7)
    p.add_argument('--reid_dim',    type=int, default=128)
    p.add_argument('--head_conv',   type=int, default=64)
    p.add_argument('--input-h',     type=int, default=704)
    p.add_argument('--input-w',     type=int, default=1280)
    p.add_argument('--opset',       type=int, default=17)
    p.add_argument('--log-wh',      action='store_true')
    p.add_argument('--simplify',    action='store_true')
    return p.parse_args()


def build_model(args):
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

    # Disable pretrained download — random weights only
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


# ── ONNX wrapper ───────────────────────────────────────────────────────────────

class DEIMMotONNX(torch.nn.Module):
    def __init__(self, model, log_wh=False):
        super().__init__()
        self.model  = model
        self.log_wh = log_wh

    def forward(self, x):
        out = self.model(x)[0]
        hm  = out['hm'].sigmoid()
        wh  = torch.exp(out['wh']) if self.log_wh else out['wh']
        reg = out['reg']
        id_ = out['id']
        return hm, wh, reg, id_


# ── interpolate patch for ONNX ─────────────────────────────────────────────────

_orig_interpolate = F.interpolate

def _onnx_interpolate(input, size=None, scale_factor=None, mode='nearest',
                      align_corners=None, recompute_scale_factor=None, antialias=False):
    return _orig_interpolate(input, size=size, scale_factor=scale_factor,
                             mode=mode, align_corners=align_corners,
                             recompute_scale_factor=False)


def main():
    args = parse_args()

    pth_path = args.output.replace('.onnx', '_dummy.pth')

    print(_LINE)
    print(f'  config    : {args.deim_config}')
    print(f'  input     : {args.input_h}×{args.input_w}')
    print(f'  num_cls   : {args.num_classes}  reid_dim={args.reid_dim}  log_wh={args.log_wh}')
    print(f'  dummy pth : {pth_path}')
    print(f'  onnx out  : {args.output}')
    print(_LINE)

    # ── Step 1: build & save dummy checkpoint ─────────────────────────────────
    print('\n[1/3] Building model with random weights ...')
    model = build_model(args)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(pth_path)), exist_ok=True)
    torch.save({'model': model.state_dict(), 'epoch': 0}, pth_path)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'      {n_params:.1f} M params  →  saved to {pth_path}')

    # ── Step 2: verify dummy forward ──────────────────────────────────────────
    print('\n[2/3] Verifying forward pass ...')
    dummy = torch.zeros(1, 3, args.input_h, args.input_w)
    with torch.no_grad():
        out = model(dummy)[0]
    print(f'      hm  : {tuple(out["hm"].shape)}')
    print(f'      wh  : {tuple(out["wh"].shape)}')
    print(f'      reg : {tuple(out["reg"].shape)}')
    print(f'      id  : {tuple(out["id"].shape)}')

    # ── Step 3: export to ONNX ────────────────────────────────────────────────
    print('\n[3/3] Exporting to ONNX ...')

    for submod in (getattr(model.deim, 'encoder', None), getattr(model.deim, 'decoder', None)):
        if submod is not None and hasattr(submod, 'eval_spatial_size'):
            submod.eval_spatial_size = None

    wrapper = DEIMMotONNX(model, log_wh=args.log_wh)
    wrapper.eval()

    F.interpolate = _onnx_interpolate
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            opset_version=args.opset,
            input_names=['image'],
            output_names=['hm', 'wh', 'reg', 'id'],
            do_constant_folding=True,
            verbose=False,
        )

    F.interpolate = _orig_interpolate
    print(f'      saved → {args.output}')

    # ── Verify ONNX ───────────────────────────────────────────────────────────
    try:
        import onnx
        m = onnx.load(args.output)
        onnx.checker.check_model(m)
        print('      onnx check : OK')
        size_mb = os.path.getsize(args.output) / 1024 ** 2
        print(f'      file size  : {size_mb:.1f} MB')
    except ImportError:
        print('      [skip] pip install onnx to verify')

    # ── Simplify ──────────────────────────────────────────────────────────────
    if args.simplify:
        try:
            import onnxsim, onnx
            m_sim, ok = onnxsim.simplify(args.output)
            if ok:
                onnx.save(m_sim, args.output)
                print(f'      onnxsim    : OK → {args.output}')
            else:
                print('      onnxsim    : failed, keeping original')
        except ImportError:
            print('      [skip] pip install onnxsim to simplify')

    print(f'\n{_LINE}')
    print(f'  Done.')
    print(f'  dummy pth : {pth_path}')
    print(f'  onnx      : {args.output}')
    print(_LINE)


if __name__ == '__main__':
    main()
