"""
Count total parameters and GFLOPs of ECDetJDE.

Architecture constants are read directly from model.py (_ECVIT_CONFIGS,
_ECDET_*) — no need to duplicate them here.

Usage:
    python tools/count_model.py
    python tools/count_model.py --ecvit_name ecvits
    python tools/count_model.py --ecvit_name ecvits --input_h 608 --input_w 1088
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import torch
from thop import profile, clever_format
from torchinfo import summary as torchinfo_summary

from lib.models.ecdet_jde.ecvit import ViTAdapter
from lib.models.ecdet_jde.hybrid_encoder import HybridEncoder
from lib.models.ecdet_jde.decoder import ECTransformer
from lib.models.ecdet_jde.model import (
    ECDetJDE,
    _ECVIT_CONFIGS,
    _ECDET_NHEAD,
    _ECDET_NUM_QUERIES,
    _ECDET_NUM_LAYERS,
    _ECDET_REG_MAX,
    _ECDET_NUM_POINTS,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Count ECDetJDE params and GFLOPs')
    parser.add_argument('--ecvit_name', default='ecvitt',
                        choices=list(_ECVIT_CONFIGS.keys()),
                        help='ECViT backbone variant')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of object classes (VisDrone=10)')
    parser.add_argument('--reid_dim',   type=int, default=128,
                        help='ReID embedding dimension')
    parser.add_argument('--input_h',    type=int, default=608)
    parser.add_argument('--input_w',    type=int, default=1088)
    parser.add_argument('--device',     default='cpu', help='cpu or cuda:0')
    return parser.parse_args()


def build_model(args) -> ECDetJDE:
    vcfg = _ECVIT_CONFIGS[args.ecvit_name]

    hidden_dim = vcfg['proj_dim'] if vcfg['proj_dim'] else vcfg['embed_dim']

    backbone = ViTAdapter(
        name               = args.ecvit_name,
        weights_path       = None,
        skip_load_backbone = True,
    )

    encoder = HybridEncoder(
        in_channels     = [hidden_dim] * 3,
        feat_strides    = [8, 16, 32],
        hidden_dim      = hidden_dim,
        use_encoder_idx = [2],
        nhead           = _ECDET_NHEAD,
        dim_feedforward = vcfg['enc_dim_ff'],
        expansion       = vcfg['expansion'],
        depth_mult      = vcfg['depth_mult'],
    )

    decoder = ECTransformer(
        num_classes       = args.num_classes,
        hidden_dim        = hidden_dim,
        num_queries       = _ECDET_NUM_QUERIES,
        feat_channels     = [hidden_dim] * 3,
        feat_strides      = [8, 16, 32],
        num_levels        = 3,
        num_points        = _ECDET_NUM_POINTS,
        nhead             = _ECDET_NHEAD,
        num_layers        = _ECDET_NUM_LAYERS,
        dim_feedforward   = vcfg['dec_dim_ff'],
        activation        = 'silu',
        num_denoising     = 0,
        eval_spatial_size = (args.input_h, args.input_w),
        eval_idx          = -1,
        aux_loss          = False,
        reg_max           = _ECDET_REG_MAX,
        reid_dim          = args.reid_dim,
        mask_downsample_ratio = None,
    )

    return ECDetJDE(backbone, encoder, decoder)


def count_parameters(model: ECDetJDE):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    parts = {
        'backbone (ECViT)':        model.backbone,
        'encoder (HybridEncoder)': model.encoder,
        'decoder (ECTransformer)': model.decoder,
        '  └─ reid_head':          model.decoder.reid_head,
    }

    print('\n' + '─' * 56)
    print(f'  {"Component":<32} {"Params":>10}  {"M":>6}')
    print('─' * 56)
    for name, mod in parts.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f'  {name:<32} {n:>10,}  {n/1e6:>5.2f}M')
    print('─' * 56)
    print(f'  {"Total":<32} {total:>10,}  {total/1e6:>5.2f}M')
    print(f'  {"  Trainable":<32} {trainable:>10,}  {trainable/1e6:>5.2f}M')
    if frozen:
        print(f'  {"  Frozen":<32} {frozen:>10,}  {frozen/1e6:>5.2f}M')
    print('─' * 56)
    return total, trainable


def count_flops(model: ECDetJDE, args):
    device = torch.device(args.device)
    model  = model.to(device).eval()
    x      = torch.zeros(1, 3, args.input_h, args.input_w, device=device)

    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)

    macs_fmt, params_fmt = clever_format([macs, params], '%.3f')

    print('\n' + '─' * 56)
    print(f'  Input resolution : {args.input_h} × {args.input_w}')
    print(f'  MACs             : {macs_fmt}   ({macs/1e9:.2f} G)')
    print(f'  GFLOPs (2×MACs)  : {macs*2/1e9:.2f} G')
    print(f'  Params (thop)    : {params_fmt}')
    print('─' * 56)
    return macs


def main():
    args  = parse_args()
    vcfg  = _ECVIT_CONFIGS[args.ecvit_name]

    model = build_model(args)
    model.eval()

    hidden_dim = vcfg['proj_dim'] if vcfg['proj_dim'] else vcfg['embed_dim']

    print(f'\n{"="*56}')
    print(f'  ECDetJDE  —  {args.ecvit_name.upper()} backbone')
    print(f'  embed_dim={vcfg["embed_dim"]}  hidden_dim={hidden_dim}')
    print(f'  nhead(enc/dec)={_ECDET_NHEAD}  layers={_ECDET_NUM_LAYERS}  queries={_ECDET_NUM_QUERIES}')
    print(f'  Classes={args.num_classes}  ReID dim={args.reid_dim}  reg_max={_ECDET_REG_MAX}')
    print(f'{"="*56}')

    count_parameters(model)

    try:
        count_flops(model, args)
    except Exception as e:
        print(f'\n[thop] GFLOPs count failed: {e}')
        print('  → Try running on CPU with a smaller input.')

    print('\n' + '─' * 56)
    print('  torchinfo layer summary (top-level):')
    print('─' * 56)
    try:
        torchinfo_summary(
            model,
            input_size=(1, 3, args.input_h, args.input_w),
            device      = args.device,
            depth       = 2,
            col_names   = ('input_size', 'output_size', 'num_params', 'mult_adds'),
            verbose     = 0,
        )
    except Exception as e:
        print(f'  torchinfo failed: {e}')


if __name__ == '__main__':
    main()
