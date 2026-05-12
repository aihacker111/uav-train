"""
Count model parameters and compute GFLOPs / MACs.
Usage: python tools/count_model_stats.py
Requires: pip install thop
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from lib.models.model import create_model

# ── config ──────────────────────────────────────────────────────────────
ARCH      = 'lwdetr_tiny'    # lwdetr_tiny | lwdetr_small | lwdetr_base
HEADS     = {'hm': 10, 'wh': 2, 'reg': 2, 'id': 128}
HEAD_CONV = 256
INPUT_H   = 640   # must be divisible by 64 for LW-DETR
INPUT_W   = 1088
CKPT_MAP  = {
    'lwdetr_tiny':  '../lwdetr_coco_pretrained/LWDETR_tiny_60e_coco.pth',
    'lwdetr_small': '../lwdetr_coco_pretrained/LWDETR_small_60e_coco.pth',
}
# ────────────────────────────────────────────────────────────────────────


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print(f"Building model: {ARCH}")
    model = create_model(ARCH, HEADS, HEAD_CONV)

    ckpt_path = CKPT_MAP.get(ARCH, '')
    if ckpt_path:
        ckpt_abs = os.path.join(os.path.dirname(__file__), ckpt_path)
        if os.path.isfile(ckpt_abs):
            from lib.models.model import load_lwdetr_backbone
            model = load_lwdetr_backbone(model, ckpt_abs)
        else:
            print(f"  [warn] checkpoint not found: {ckpt_abs}")

    model.eval()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)

    # ── Parameters ──────────────────────────────────────────────────────
    total_params, trainable_params = count_params(model)
    print(f"\n{'─'*45}")
    print(f"  Total params     : {total_params / 1e6:.2f} M")
    print(f"  Trainable params : {trainable_params / 1e6:.2f} M")

    if hasattr(model, 'backbone'):
        bb = sum(p.numel() for p in model.backbone.parameters())
        print(f"  └─ Backbone      : {bb / 1e6:.2f} M")
    if hasattr(model, 'decoder'):
        dec = sum(p.numel() for p in model.decoder.parameters())
        print(f"  └─ Decoder       : {dec / 1e6:.2f} M")

    # ── GFLOPs / MACs via thop ───────────────────────────────────────────
    try:
        import copy
        from thop import profile
        model_copy = copy.deepcopy(model).eval()
        macs, _ = profile(model_copy, inputs=(dummy,), verbose=False)
        del model_copy
        print(f"  MACs             : {macs / 1e9:.2f} G")
        print(f"  GFLOPs           : {macs * 2 / 1e9:.2f} G")
    except ImportError:
        print("  [thop not installed] run: pip install thop")

    print(f"{'─'*45}\n")
    print(f"  Input  : {tuple(dummy.shape)}  ({INPUT_H}×{INPUT_W})")
    out = model(dummy)
    for k, v in out[-1].items():
        print(f"  Output [{k}] : {tuple(v.shape)}")
    print(f"{'─'*45}\n")


if __name__ == '__main__':
    main()
