"""
Count total parameters and GFLOPs of ECDetJDE.

Usage:
    python tools/count_model.py
    python tools/count_model.py --ecvit_name ecvits --input_h 608 --input_w 1088
    python tools/count_model.py --ecdet_pretrained /path/to/ckpt.pth
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
from lib.models.ecdet_jde.model import ECDetJDE


def parse_args():
    parser = argparse.ArgumentParser(description='Count ECDetJDE params and GFLOPs')
    parser.add_argument('--ecvit_name',    default='ecvitt',
                        choices=['ecvitt', 'ecvittplus', 'ecvits', 'ecvitsplus'],
                        help='ECViT backbone variant')
    parser.add_argument('--num_classes',   type=int, default=10,
                        help='number of object classes (VisDrone=10)')
    parser.add_argument('--num_queries',   type=int, default=300)
    parser.add_argument('--reid_dim',      type=int, default=128)
    parser.add_argument('--hidden_dim',    type=int, default=192)
    parser.add_argument('--num_layers',    type=int, default=4)
    parser.add_argument('--nhead',         type=int, default=3)
    parser.add_argument('--dim_ff',        type=int, default=512)
    parser.add_argument('--input_h',       type=int, default=608)
    parser.add_argument('--input_w',       type=int, default=1088)
    parser.add_argument('--device',        default='cpu',
                        help='cpu or cuda:0')
    return parser.parse_args()


def build_model(args) -> ECDetJDE:
    backbone = ViTAdapter(
        name         = args.ecvit_name,
        weights_path = None,          # skip weight loading for counting
        embed_dim    = args.hidden_dim,
        num_heads    = args.nhead,
        num_levels   = 3,
        skip_load_backbone = True,
    )

    encoder = HybridEncoder(
        in_channels     = [args.hidden_dim] * 3,
        feat_strides    = [8, 16, 32],
        hidden_dim      = args.hidden_dim,
        use_encoder_idx = [1],
        dim_feedforward = args.dim_ff,
        expansion       = 0.34,
        depth_mult      = 0.5,
    )

    decoder = ECTransformer(
        num_classes       = args.num_classes,
        hidden_dim        = args.hidden_dim,
        num_queries       = args.num_queries,
        feat_channels     = [args.hidden_dim] * 3,
        feat_strides      = [8, 16, 32],
        num_levels        = 3,
        num_points        = [4, 4, 4],
        nhead             = args.nhead,
        num_layers        = args.num_layers,
        dim_feedforward   = args.dim_ff,
        activation        = 'silu',
        num_denoising     = 0,         # disable DN for clean FLOP count
        eval_spatial_size = (args.input_h, args.input_w),
        eval_idx          = -1,
        aux_loss          = False,
        reg_max           = 32,
        reid_dim          = args.reid_dim,
        mask_downsample_ratio = None,
    )

    return ECDetJDE(backbone, encoder, decoder)


def count_parameters(model: torch.nn.Module):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen  = total - trainable

    # Per-component breakdown
    parts = {
        'backbone (ECViT)':  model.backbone,
        'encoder (HybridEncoder)': model.encoder,
        'decoder (ECTransformer)': model.decoder,
        '  └─ reid_head':    model.decoder.reid_head,
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


def count_flops(model: torch.nn.Module, args):
    device = torch.device(args.device)
    model  = model.to(device).eval()
    x      = torch.zeros(1, 3, args.input_h, args.input_w, device=device)

    # thop needs the model in eval mode with num_denoising=0
    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)

    macs_fmt, params_fmt = clever_format([macs, params], '%.3f')

    print('\n' + '─' * 56)
    print(f'  Input resolution : {args.input_h} × {args.input_w}')
    print(f'  MACs (≈GFLOPs×½) : {macs_fmt}   ({macs/1e9:.2f} G)')
    print(f'  GFLOPs (2×MACs)  : {macs*2/1e9:.2f} G')
    print(f'  Params (thop)    : {params_fmt}')
    print('─' * 56)
    return macs


def main():
    args  = parse_args()
    model = build_model(args)
    model.eval()

    print(f'\n{"="*56}')
    print(f'  ECDetJDE  —  {args.ecvit_name.upper()} backbone')
    print(f'  Classes: {args.num_classes}  |  Queries: {args.num_queries}  |  ReID dim: {args.reid_dim}')
    print(f'{"="*56}')

    # 1) Parameter count
    count_parameters(model)

    # 2) GFLOPs via thop
    try:
        count_flops(model, args)
    except Exception as e:
        print(f'\n[thop] GFLOPs count failed: {e}')
        print('  → Try running on CPU with a smaller input.')

    # 3) torchinfo summary (optional, verbose)
    print('\n' + '─' * 56)
    print('  torchinfo layer summary (top-level):')
    print('─' * 56)
    try:
        torchinfo_summary(
            model,
            input_size=(1, 3, args.input_h, args.input_w),
            device=args.device,
            depth=2,
            col_names=('input_size', 'output_size', 'num_params', 'mult_adds'),
            verbose=0,
        )
    except Exception as e:
        print(f'  torchinfo failed: {e}')


if __name__ == '__main__':
    main()
