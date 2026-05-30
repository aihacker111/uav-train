"""
DEIMMotCriterion — DEIMCriterion extended with per-query ReID loss.

Registered as 'DEIMMotCriterion' in the DEIMv2 YAML config system.

Loss:
    L_total = L_det + id_weight * L_reid
    L_det   = DEIMCriterion losses (mal, boxes, local, fgl, ddf, ...)
    L_reid  = per-class softmax-CE + optional Triplet on Hungarian-matched queries
"""
import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deim_criterion import DEIMCriterion
from ..core import register


class TripletLoss(nn.Module):
    """Online hard-mining Triplet loss."""
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = inputs.size(0)
        if n < 2:
            return inputs.new_zeros(())
        dist = torch.cdist(inputs, inputs, p=2)
        mask_pos = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        mask_neg = ~mask_pos

        dist_ap = (dist * mask_pos.float()).max(dim=1)[0]
        dist_an = (dist + (~mask_neg).float() * dist.max()).min(dim=1)[0]

        y = torch.ones_like(dist_ap)
        return self.ranking_loss(dist_an, dist_ap, y)


@register()
class DEIMMotCriterion(DEIMCriterion):
    """
    DEIMCriterion + per-query ReID loss for JDE-style multi-object tracking.

    Extra YAML parameters:
        id_weight  : float  — ReID loss weight (default 1.0)
        reid_dim   : int    — embedding dimension (must match DEIMTransformer reid_dim)
        nID_dict   : dict   — {cls_id: num_identities}, set by MotSolver before fit()
        use_triplet: bool   — also compute triplet loss (default False)
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        id_weight: float = 1.0,
        reid_dim: int = 128,
        use_triplet: bool = False,
    ) -> None:
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
            mal_alpha=mal_alpha,
            use_uni_set=use_uni_set,
        )
        self.id_weight   = id_weight
        self.reid_dim    = reid_dim
        self.use_triplet = use_triplet
        self._last_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        # nID_dict is set externally by MotSolver.fit() after the dataset is loaded.
        # classifiers + emb_scale are built lazily on first call to _build_classifiers().
        self.nID_dict: Dict[int, int] = {}
        self.classifiers: Optional[nn.ModuleDict] = None
        self.emb_scale:   Dict[int, float] = {}
        self.ce_loss      = nn.CrossEntropyLoss(ignore_index=-1)
        self.triplet_loss = TripletLoss() if use_triplet else None

    def build_classifiers(self, nID_dict: Dict[int, int], device=None) -> None:
        """Build per-class linear classifiers from nID_dict. Called by MotSolver."""
        self.nID_dict    = nID_dict
        self.classifiers = nn.ModuleDict()
        self.emb_scale   = {}
        for cls_id, nID in nID_dict.items():
            self.classifiers[str(cls_id)] = nn.Linear(self.reid_dim, nID)
            self.emb_scale[cls_id] = math.sqrt(2) * math.log(max(nID - 1, 1))
        if device is not None:
            self.classifiers = self.classifiers.to(device)

    # ── ReID loss ─────────────────────────────────────────────────────────────

    def _reid_loss(
        self,
        pred_reid: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if self.classifiers is None or not self.nID_dict:
            return pred_reid.new_zeros(())

        total_loss = pred_reid.new_zeros(())
        n_valid    = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            reid_b  = pred_reid[b][src_idx]                           # (M, reid_dim)
            tgt     = targets[b]
            tids    = tgt['track_ids'].to(reid_b.device)              # (N_gt,)  1D
            clabels = tgt['labels'].to(reid_b.device)                 # (N_gt,)  1D cls_ids

            matched_tids   = tids[tgt_idx]
            matched_labels = clabels[tgt_idx]

            for cls_id in self.nID_dict:
                mask = (matched_labels == cls_id) & (matched_tids >= 0)
                if mask.sum() == 0:
                    continue
                scale = self.emb_scale.get(cls_id, 1.0)
                feat  = scale * F.normalize(reid_b[mask], dim=-1)
                tid   = matched_tids[mask]
                pred  = self.classifiers[str(cls_id)](feat)
                total_loss = total_loss + self.ce_loss(pred, tid)
                if self.triplet_loss is not None:
                    total_loss = total_loss + self.triplet_loss(feat, tid)
                n_valid += mask.sum().item()

        return total_loss / max(n_valid, 1)

    # ── Override forward ──────────────────────────────────────────────────────

    def forward(self, outputs, targets, epoch=0, **kwargs):
        """Compute detection losses + optional ReID loss.

        Targets from VisDroneDataset use 2-column labels: (N, 2) where
            col-0 = cls_id   (expected by DEIMCriterion as 1D)
            col-1 = track_id (used here for ReID loss)

        We split them so DEIMCriterion (and the matcher) get 1D cls_ids,
        while we keep track_ids for the ReID head.
        """
        # ── Split 2-column labels into cls_ids + track_ids ────────────────────
        targets_det  = []   # for DEIMCriterion: labels = 1D cls_id
        track_ids_list = [] # for ReID loss

        for t in targets:
            lbl = t['labels']
            if isinstance(lbl, torch.Tensor) and lbl.ndim == 2 and lbl.shape[-1] == 2:
                cls_ids   = lbl[:, 0].long()
                track_ids = lbl[:, 1].long()
            else:
                cls_ids   = lbl.long() if isinstance(lbl, torch.Tensor) else torch.as_tensor(lbl).long()
                track_ids = t.get('track_ids', torch.full_like(cls_ids, -1))

            targets_det.append({**t, 'labels': cls_ids})
            track_ids_list.append(track_ids)

        # ── DEIMCriterion detection losses (mal, boxes, local, fgl, ddf) ─────
        det_losses = super().forward(outputs, targets_det, epoch=epoch, **kwargs)

        # ── Hungarian matching indices for the final decoder layer ────────────
        # Run the matcher once more (parent already ran it, but doesn't expose it).
        # Cheap: matcher is called at most once per forward anyway.
        self._last_indices = self.matcher(
            {k: v for k, v in outputs.items() if 'aux' not in k},
            targets_det, epoch=epoch,
        )['indices']

        # ── ReID loss ─────────────────────────────────────────────────────────
        if self.id_weight > 0 and 'pred_reid' in outputs and self.classifiers is not None:
            # Attach track_ids back for _reid_loss
            targets_reid = [
                {**t_det, 'track_ids': tids}
                for t_det, tids in zip(targets_det, track_ids_list)
            ]
            reid_loss = self._reid_loss(outputs['pred_reid'], targets_reid, self._last_indices)
            det_losses['loss_reid'] = self.id_weight * reid_loss

        return det_losses
