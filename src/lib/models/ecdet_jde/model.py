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



# Per-variant architecture constants read directly from EdgeCrafter configs.
# Sources:
#   ecdetseg/configs/ecdet/ecdet.yml   (base: nhead, num_points, csp_type, fuse_op, use_encoder_idx)
#   ecdetseg/configs/ecdet/ecdet_s.yml (ecvitt)
#   ecdetseg/configs/ecdet/ecdet_m.yml (ecvittplus)
#   ecdetseg/configs/ecdet/ecdet_l.yml (ecvits)
#   ecdetseg/configs/ecdet/ecdet_x.yml (ecvitsplus)
#
# Columns:
#   embed_dim   – ViTAdapter internal dimension
#   num_heads   – ViTAdapter attention heads  (variant-specific, NOT shared with encoder/decoder)
#   proj_dim    – ViTAdapter output projection; None → output dim == embed_dim
#   expansion   – HybridEncoder CSP expansion ratio
#   depth_mult  – HybridEncoder CSP depth multiplier
#   enc_dim_ff  – HybridEncoder TransformerEncoderLayer FFN dim
#   dec_dim_ff  – ECTransformer decoder FFN dim
_ECVIT_CONFIGS = {
    'ecvitt':     dict(embed_dim=192, num_heads=3, proj_dim=None, expansion=0.34, depth_mult=0.67, enc_dim_ff=512,  dec_dim_ff=512),
    'ecvittplus': dict(embed_dim=256, num_heads=4, proj_dim=None, expansion=0.75, depth_mult=0.67, enc_dim_ff=512,  dec_dim_ff=1024),
    'ecvits':     dict(embed_dim=384, num_heads=6, proj_dim=256,  expansion=0.75, depth_mult=1.0,  enc_dim_ff=1024, dec_dim_ff=1024),
    'ecvitsplus': dict(embed_dim=384, num_heads=6, proj_dim=256,  expansion=1.5,  depth_mult=1.0,  enc_dim_ff=2048, dec_dim_ff=2048),
}

# Constants shared across all ECDet variants (from base ecdet.yml)
_ECDET_NHEAD         = 8      # HybridEncoder & ECTransformer attention heads
_ECDET_NUM_QUERIES   = 300    # ECTransformer object queries
_ECDET_NUM_LAYERS    = 4      # ECTransformer decoder layers
_ECDET_NUM_DENOISING = 100    # DN-DETR denoising groups
_ECDET_REG_MAX       = 32     # DFL distribution bins
_ECDET_REG_SCALE     = 4.0    # DFL regression scale
_ECDET_NUM_POINTS    = [3, 6, 3]   # MSDeformAttn sampling points per scale


def build_ecdet_jde(opt) -> ECDetJDE:
    """
    Build ECDetJDE from opts.

    Required opt fields:
        opt.ecvit_name        : str   — variant: 'ecvitt' | 'ecvittplus' | 'ecvits' | 'ecvitsplus'
        opt.ecvit_weights     : str   — path to ECViT backbone weights (.pth)
        opt.ecdet_pretrained  : str   — path to full ECDet COCO checkpoint
        opt.num_classes       : int   — number of object classes
        opt.reid_dim          : int   — ReID embedding dimension
        opt.eval_spatial_size : list  — [H, W] for anchor pre-generation (e.g. [608, 1088])
    """
    ecvit_name  = getattr(opt, 'ecvit_name', 'ecvitt')
    num_classes = opt.num_classes
    reid_dim    = getattr(opt, 'reid_dim',         128)
    eval_size   = getattr(opt, 'eval_spatial_size', None)

    vcfg       = _ECVIT_CONFIGS.get(ecvit_name, _ECVIT_CONFIGS['ecvitt'])
    embed_dim  = vcfg['embed_dim']
    num_heads  = vcfg['num_heads']    # ViT backbone heads — must match pre-trained checkpoint
    proj_dim   = vcfg['proj_dim']     # None → no proj, feature dim stays embed_dim
    expansion  = vcfg['expansion']
    depth_mult = vcfg['depth_mult']
    enc_dim_ff = vcfg['enc_dim_ff']   # HybridEncoder FFN
    dec_dim_ff = vcfg['dec_dim_ff']   # ECTransformer FFN
    hidden_dim = proj_dim if proj_dim is not None else embed_dim  # encoder/decoder width

    # --- Backbone ---
    backbone = ViTAdapter(
        name         = ecvit_name,
        weights_path = getattr(opt, 'ecvit_weights', None),
        embed_dim    = embed_dim,
        num_heads    = num_heads,
        proj_dim     = proj_dim,
        num_levels   = 3,
    )

    # --- Encoder ---
    encoder = HybridEncoder(
        in_channels     = [hidden_dim] * 3,
        feat_strides    = [8, 16, 32],
        hidden_dim      = hidden_dim,
        nhead           = _ECDET_NHEAD,
        use_encoder_idx = [2],          # SA on stride-32 only (base ecdet.yml)
        num_encoder_layers = 1,
        dim_feedforward = enc_dim_ff,
        expansion       = expansion,
        depth_mult      = depth_mult,
        fuse_op         = 'sum',        # base ecdet.yml: fuse_op: sum
        csp_type        = 'csp2',       # base ecdet.yml: csp_type: csp2
    )

    # --- Decoder ---
    decoder = ECTransformer(
        num_classes       = num_classes,
        hidden_dim        = hidden_dim,
        num_queries       = _ECDET_NUM_QUERIES,
        feat_channels     = [hidden_dim] * 3,
        feat_strides      = [8, 16, 32],
        num_levels        = 3,
        num_points        = _ECDET_NUM_POINTS,
        nhead             = _ECDET_NHEAD,
        num_layers        = _ECDET_NUM_LAYERS,
        dim_feedforward   = dec_dim_ff,
        activation        = 'silu',
        num_denoising     = _ECDET_NUM_DENOISING,
        label_noise_ratio = 0.5,
        box_noise_scale   = 1.0,
        eval_spatial_size = tuple(eval_size) if eval_size else None,
        eval_idx          = -1,
        aux_loss          = True,
        reg_max           = _ECDET_REG_MAX,
        reg_scale         = _ECDET_REG_SCALE,
        reid_dim          = reid_dim,
        mask_downsample_ratio = None,
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
