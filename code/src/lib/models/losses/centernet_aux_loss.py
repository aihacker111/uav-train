"""
CenterNetAuxLoss: lightweight auxiliary loss for the CenterNet head in HybridDEIM.

Expected target keys in the batch dict (CenterNet format — identical to Stage-1
targets used by HybridLoss, so the same dataloader produces both):

    hm       : (B, C, H/4, W/4)  Gaussian-rendered ground-truth heatmap
    wh       : (B, max_obj, 2)   width/height in pixels at each GT peak location
    reg      : (B, max_obj, 2)   sub-pixel centre offset at each GT peak
    ind      : (B, max_obj)      flat spatial index (row*W + col) of each GT peak
    reg_mask : (B, max_obj)      1 = valid object, 0 = padding

Total loss:
    L = L_focal(hm) + λ_wh * L_SmoothL1(wh) + λ_reg * L_SmoothL1(reg)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hybrid_loss import centernet_focal_loss, _gather_at_ind
from ..networks.deim_uav.heads import CenterNetOutput


class CenterNetAuxLoss(nn.Module):
    """
    Auxiliary CenterNet loss used alongside DEIMCriterion in HybridDEIM training.

    Args:
        wh_weight  : Weight for the width/height SmoothL1 term (default 0.1,
                     kept small because DEIM's box head already handles size).
        reg_weight : Weight for the sub-pixel offset SmoothL1 term (default 1.0).
    """

    def __init__(
        self,
        wh_weight:  float = 0.1,
        reg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.wh_weight  = wh_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        cn_out:  CenterNetOutput,
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Args:
            cn_out  : CenterNetOutput from HybridDEIM.cn_head.
            targets : Batch dict with keys 'hm', 'wh', 'reg', 'ind', 'reg_mask'.
        Returns:
            Dict with per-term losses and 'cn_total'.
        """
        gt_hm    = targets['hm']                          # (B, C, H, W)
        gt_wh    = targets['wh']                          # (B, max_obj, 2)
        gt_reg   = targets['reg']                         # (B, max_obj, 2)
        ind      = targets['ind']                         # (B, max_obj)
        mask     = targets['reg_mask'].bool()             # (B, max_obj)

        # ── Focal loss on full heatmap ─────────────────────────────────────────
        l_hm = centernet_focal_loss(cn_out.hm, gt_hm)

        # ── Regression losses at GT peak locations only ───────────────────────
        # _gather_at_ind: (B, C, H, W) + (B, max_obj) → (B, max_obj, C)
        pred_wh  = _gather_at_ind(cn_out.wh,  ind)   # (B, max_obj, 2)
        pred_reg = _gather_at_ind(cn_out.reg, ind)   # (B, max_obj, 2)

        n_pos = mask.sum()
        if n_pos > 0:
            l_wh  = F.smooth_l1_loss(pred_wh[mask],  gt_wh[mask],  beta=1.0)
            l_reg = F.smooth_l1_loss(pred_reg[mask], gt_reg[mask], beta=0.5)
        else:
            # No valid objects in this batch — return zero but keep the graph alive
            l_wh  = (pred_wh  * 0).sum()
            l_reg = (pred_reg * 0).sum()

        total = l_hm + self.wh_weight * l_wh + self.reg_weight * l_reg

        return {
            'cn_hm':    l_hm,
            'cn_wh':    l_wh,
            'cn_reg':   l_reg,
            'cn_total': total,
        }
