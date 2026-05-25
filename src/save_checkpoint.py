"""
Save / load a HybridECDet checkpoint.

Usage — save:
    python src/save_checkpoint.py \
        --load_model exp/hybrid/hybrid_ecdet_s_visdrone/model_last.pth \
        --save_path  exp/hybrid/hybrid_ecdet_s_visdrone/model_export.pth

Usage — verify (load back and print keys):
    python src/save_checkpoint.py --verify \
        --load_model exp/hybrid/hybrid_ecdet_s_visdrone/model_export.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _init_paths  # noqa: F401 — sets up lib path

import torch
from lib.models.model import create_model, load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--load_model', required=True, help='Path to source .pth checkpoint')
    p.add_argument('--save_path',  default='',    help='Output path (used when not --verify)')
    p.add_argument('--verify',     action='store_true', help='Just load and print checkpoint keys')
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.load_model, map_location='cpu')
    print(f'[load]  {args.load_model}')
    print(f'  keys : {list(ckpt.keys())}')
    if 'epoch' in ckpt:
        print(f'  epoch: {ckpt["epoch"]}')
    if 'state_dict' in ckpt:
        n = len(ckpt['state_dict'])
        print(f'  params: {n} tensors')

    if args.verify:
        print('[verify] checkpoint looks OK')
        return

    if not args.save_path:
        raise ValueError('Provide --save_path when not using --verify')

    # Save a clean weights-only checkpoint (no optimizer / loss state)
    clean = {'state_dict': ckpt['state_dict'], 'epoch': ckpt.get('epoch', 0)}
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save(clean, args.save_path)
    print(f'[save]  {args.save_path}  ({os.path.getsize(args.save_path)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
