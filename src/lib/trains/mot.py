"""
ECDetJDE training step — EdgeCrafter detection loss + ReID loss.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from lib.models.ecdet_jde import ECDetJDECriterion, HungarianMatcher
from .base_trainer import BaseTrainer


def _build_criterion(opt) -> ECDetJDECriterion:
    matcher = HungarianMatcher(
        weight_dict={'cost_class': 2.0, 'cost_bbox': 5.0, 'cost_giou': 2.0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    # Weight dict from EdgeCrafter ecdet.yml (base config)
    # RF-DETR style: focal + L1 + GIoU (no MAL, no FGL/DDF)
    weight_dict = {
        'loss_cls':  2.0,   # focal classification
        'loss_bbox': 5.0,   # L1 box regression
        'loss_giou': 2.0,   # GIoU box regression
        'loss_rep':  1.0,   # Repulsion Loss (Wang et al. CVPR 2018)
    }
    return ECDetJDECriterion(
        matcher            = matcher,
        num_classes        = opt.num_classes,
        nid_dict           = opt.nID_dict,
        reid_dim           = getattr(opt, 'reid_dim', 128),
        weight_dict        = weight_dict,
        losses             = ('focal', 'boxes', 'rep'),
        gamma              = 2.0,   # focal γ (standard RetinaNet value)
        alpha              = 0.25,  # focal α (standard RetinaNet value)
        boxes_weight_format= None,   # RF-DETR style: no IoU gating on L1/GIoU
        use_uni_set        = True,
        id_weight          = getattr(opt, 'id_weight', 1.0),
        use_triplet        = getattr(opt, 'tri', False),
    )


class ECDetJDEWithLoss(nn.Module):
    """Wraps model + criterion for DataParallel."""

    def __init__(self, model, criterion):
        super().__init__()
        self.model     = model
        self.criterion = criterion

    def forward(self, batch):
        B = batch['input'].shape[0]
        targets = []
        for i in range(B):
            n = int(batch['detr_num_objs'][i].item())
            valid_labels = batch['detr_labels'][i, :n]
            valid_boxes  = batch['detr_boxes'][i, :n]
            valid_tids   = batch['detr_track_ids'][i, :n]
            keep = valid_labels >= 0
            targets.append({
                'labels':    valid_labels[keep],
                'boxes':     valid_boxes[keep],
                'track_ids': valid_tids[keep],
            })

        outputs   = self.model(batch['input'], targets)
        loss_dict = self.criterion(outputs, targets)
        return outputs, loss_dict['loss'], loss_dict


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None, **kwargs):
        super().__init__(opt, model, optimizer=optimizer, **kwargs)

    def _get_losses(self, opt):
        loss_states = ['loss_main', 'loss_cls', 'loss_bbox', 'loss_giou', 'loss_rep', 'loss_reid']
        criterion = _build_criterion(opt)
        return loss_states, criterion

    def _build_model_with_loss(self, model, loss):
        return ECDetJDEWithLoss(model, loss)

    def save_result(self, output, batch, results):
        pass
