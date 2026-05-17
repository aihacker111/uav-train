"""
Save a randomly-initialised HybridCenterNetDETR checkpoint.
Useful for testing the training pipeline and ONNX export without real weights.

Usage:
    cd src
    python save_dummy_model.py                          # hybrid_small, saved to ../models/
    python save_dummy_model.py --arch hybrid_tiny --output ../models/dummy_tiny.pth
"""
import argparse
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
from lib.models.model import create_model, save_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch',     default='hybrid_small',
                   choices=['hybrid_tiny', 'hybrid_small', 'hybrid_base'])
    p.add_argument('--output',   default='')
    p.add_argument('--reid-dim', type=int, default=256)
    p.add_argument('--epoch',    type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    out = args.output or f'../models/dummy_{args.arch}.pth'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print(f'[dummy] building {args.arch} …')
    model = create_model(args.arch, heads={}, head_conv=-1, reid_dim=args.reid_dim)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[dummy] {n_params / 1e6:.1f} M parameters')

    save_model(out, epoch=args.epoch, model=model)
    size_mb = os.path.getsize(out) / 1e6
    print(f'[dummy] saved → {out}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
