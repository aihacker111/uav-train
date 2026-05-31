"""
ECDetJDE: ECViT + HybridEncoder + ECTransformer(+ReID head) for MCMOT.

Architecture:
  ViTAdapter (ECViT backbone, pretrained EdgeCrafter COCO)
      ↓ 3-level features (stride 8/16/32)
  HybridEncoder (FPN+PAN, pretrained)
      ↓ enriched multi-scale features
  ECTransformer (pretrained, + reid_head added on top)
      ↓ N queries
      ├─ pred_logits  (B, N, num_classes)
      ├─ pred_boxes   (B, N, 4)  cxcywh norm
      └─ pred_reid    (B, N, reid_dim)

The forward contract mirrors CenterNet's list-of-dicts pattern:
  training:   returns raw ECTransformer dict  (for ECDetJDECriterion)
  inference:  returns same dict               (for ECDetJDEPostProcessor)
"""

import torch
import torch.nn as nn

from .ecvit        import ViTAdapter
from .hybrid_encoder import HybridEncoder
from .decoder      import ECTransformer


class ECDetJDE(nn.Module):
    def __init__(self,
                 backbone: ViTAdapter,
                 encoder:  HybridEncoder,
                 decoder:  ECTransformer,
                 ):
        super().__init__()
        self.backbone = backbone
        self.encoder  = encoder
        self.decoder  = decoder

    def forward(self, x, targets=None):
        feats  = self.backbone(x)         # list of 3 feature maps
        feats  = self.encoder(feats)      # list of enriched feature maps
        out    = self.decoder(feats, targets)
        return out

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ecdet_jde(opt) -> ECDetJDE:
    """
    Build ECDetJDE from opts.

    Required opt fields:
        opt.ecvit_name          : str  — 'ecvitt' | 'ecvits' | 'ecvittplus' | 'ecvitsplus'
        opt.ecvit_weights       : str  — path to ECViT backbone weights (.pth)
        opt.ecdet_pretrained    : str  — path to full ECDet COCO checkpoint (loads backbone+encoder+decoder)
        opt.num_classes         : int  — number of object classes
        opt.reid_dim            : int  — ReID embedding dimension (default 128)
        opt.num_queries         : int  — number of DETR queries (default 300)
        opt.eval_spatial_size   : list — [H, W] for anchor pre-generation (e.g. [608, 1088])
    """
    num_classes  = opt.num_classes
    reid_dim     = getattr(opt, 'reid_dim',          128)
    num_queries  = getattr(opt, 'num_queries',        300)
    hidden_dim   = getattr(opt, 'ecdet_hidden_dim',  192)
    num_layers   = getattr(opt, 'ecdet_num_layers',    4)
    eval_size    = getattr(opt, 'eval_spatial_size',  None)
    nhead        = getattr(opt, 'ecdet_nhead',          3)
    dim_ff       = getattr(opt, 'ecdet_dim_ff',       512)

    # --- Backbone ---
    backbone = ViTAdapter(
        name         = getattr(opt, 'ecvit_name', 'ecvitt'),
        weights_path = getattr(opt, 'ecvit_weights', None),
        embed_dim    = hidden_dim,
        num_heads    = nhead,
        num_levels   = 3,
    )

    # --- Encoder ---
    # expansion=0.67 → c4=64 (matches ECDet-S checkpoint)
    # fuse_op='sum'  → c1=hidden_dim=192 (matches ECDet-S checkpoint, not cat which gives 384)
    encoder = HybridEncoder(
        in_channels     = [hidden_dim] * 3,
        feat_strides    = [8, 16, 32],
        hidden_dim      = hidden_dim,
        use_encoder_idx = [1],
        dim_feedforward = dim_ff,
        expansion       = 0.67,
        depth_mult      = 0.67,
        fuse_op         = 'sum',
    )

    # --- Decoder (ECTransformer with ReID head) ---
    decoder = ECTransformer(
        num_classes       = num_classes,
        hidden_dim        = hidden_dim,
        num_queries       = num_queries,
        feat_channels     = [hidden_dim] * 3,
        feat_strides      = [8, 16, 32],
        num_levels        = 3,
        num_points        = [4, 4, 4],
        nhead             = nhead,
        num_layers        = num_layers,
        dim_feedforward   = dim_ff,
        activation        = 'silu',
        num_denoising     = 100,
        eval_spatial_size = tuple(eval_size) if eval_size else None,
        eval_idx          = -1,
        aux_loss          = True,
        reg_max           = 32,
        reg_scale         = 4.0,
        reid_dim          = reid_dim,
        mask_downsample_ratio = None,      # no segmentation
    )

    model = ECDetJDE(backbone, encoder, decoder)

    # --- Load pretrained ECDet checkpoint ---
    ecdet_ckpt = getattr(opt, 'ecdet_pretrained', '')
    if ecdet_ckpt:
        _load_ecdet_pretrained(model, ecdet_ckpt)

    return model


def _load_ecdet_pretrained(model: ECDetJDE, ckpt_path: str):
    """
    Load ECDet COCO checkpoint into ECDetJDE.
    - backbone, encoder, decoder weights are loaded (strict=False)
    - reid_head is skipped (new, not in checkpoint)
    """
    print(f'[ECDetJDE] Loading ECDet pretrained from: {ckpt_path}')
    state = torch.load(ckpt_path, map_location='cpu')

    # ECDet checkpoint may be wrapped under 'model' or 'state_dict'
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']

    # Strip 'module.' prefix from DataParallel checkpoints
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    reid_keys = [k for k in missing if 'reid_head' in k]
    other_missing = [k for k in missing if 'reid_head' not in k]

    print(f'[ECDetJDE] Pretrained loaded — '
          f'reid_head params (new, expected missing): {len(reid_keys)} | '
          f'other missing: {len(other_missing)} | '
          f'unexpected: {len(unexpected)}')
    if other_missing:
        print(f'[ECDetJDE] Warning — unexpected missing keys: {other_missing[:10]}')
