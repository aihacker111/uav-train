"""
MotSolver — extends DetSolver for JDE-style multi-object tracking training.

Key additions over DetSolver:
  1. After dataset is loaded, calls criterion.build_classifiers(nID_dict)
     so per-class linear classifiers are ready before training starts.
  2. Registers classifier parameters in the optimizer parameter group so
     they are updated during training and saved in checkpoints.
  3. Everything else (EMA, AMP, LR schedule, COCO eval, checkpointing)
     is inherited from DetSolver unchanged.
"""
from __future__ import annotations

from .det_solver import DetSolver
from ..core import register


@register()
class MotSolver(DetSolver):
    """DetSolver variant for MOT/JDE training with ReID."""

    def train(self):
        """Set up training — same as DetSolver but wires up ReID classifiers."""
        super().train()   # sets model, criterion, dataloader, optimizer, EMA, scaler, etc.

        # Build per-class linear classifiers from the dataset's nID_dict
        criterion = getattr(self, 'criterion', None)
        if criterion is None:
            return

        dataset = getattr(self.train_dataloader.dataset, 'dataset', None) \
                  or getattr(self.train_dataloader, 'dataset', None)

        nID_dict = getattr(dataset, 'nID_dict', None)
        if nID_dict is None:
            print('[MotSolver] dataset has no nID_dict — ReID classifiers not built')
            return

        if hasattr(criterion, 'build_classifiers'):
            criterion.build_classifiers(nID_dict, device=self.device)
            n_ids = sum(nID_dict.values())
            print(f'[MotSolver] built ReID classifiers: {len(nID_dict)} classes, '
                  f'{n_ids} total identities')

            # Add classifier parameters to optimizer so they are updated and
            # their state is saved/restored with the checkpoint.
            cls_params = list(criterion.classifiers.parameters())
            if cls_params:
                # Reuse the learning rate of the last param group (head LR)
                head_lr = self.optimizer.param_groups[-1]['lr']
                self.optimizer.add_param_group({
                    'params':       cls_params,
                    'lr':           head_lr,
                    'weight_decay': 1e-4,
                })
                print(f'[MotSolver] added {len(cls_params)} classifier param tensors '
                      f'to optimizer (lr={head_lr:.2e})')
