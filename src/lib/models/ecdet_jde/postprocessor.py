"""
ECDetJDE PostProcessor.
Converts raw ECTransformer output → per-class detection dicts compatible
with MCJDETracker (same interface that mot_decode used to produce).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ECDetJDEPostProcessor(nn.Module):
    """
    Converts ECTransformer output to tracking-ready detections.

    Output per image (dict):
      {cls_id: np.ndarray (N, 6)}  where columns = [x1, y1, x2, y2, score, cls_id]

    And reid embeddings per class:
      {cls_id: np.ndarray (N, reid_dim)}
    """

    def __init__(self, num_classes: int, conf_thres: float = 0.3,
                 num_top_queries: int = 300):
        super().__init__()
        self.num_classes   = num_classes
        self.conf_thres    = conf_thres
        self.num_top_queries = num_top_queries

    def forward(self, outputs: dict, orig_hw: tuple):
        """
        Args:
            outputs: ECTransformer output dict with keys:
                'pred_logits': (1, N, num_classes)
                'pred_boxes':  (1, N, 4)  cxcywh normalized [0,1]
                'pred_reid':   (1, N, reid_dim)
            orig_hw: (H, W) of the original image

        Returns:
            dets_dict:  {cls_id: Tensor(M, 6)}  xyxy + score + cls in image coords
            reid_dict:  {cls_id: Tensor(M, reid_dim)}  L2-normalized embeddings
        """
        logits = outputs['pred_logits'][0]  # (N, num_classes)
        boxes  = outputs['pred_boxes'][0]   # (N, 4) cxcywh norm
        reid   = outputs['pred_reid'][0]    # (N, reid_dim)

        H, W = orig_hw
        scores_all = logits.sigmoid()       # (N, num_classes)

        # Convert boxes cxcywh→xyxy in image coords
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - bw * 0.5) * W
        y1 = (cy - bh * 0.5) * H
        x2 = (cx + bw * 0.5) * W
        y2 = (cy + bh * 0.5) * H
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, 4)

        # L2-normalize ReID embeddings
        reid_norm = F.normalize(reid, dim=-1)  # (N, reid_dim)

        dets_dict = {}
        reid_dict = {}
        for cls_id in range(self.num_classes):
            cls_scores = scores_all[:, cls_id]          # (N,)
            keep       = cls_scores > self.conf_thres
            if keep.sum() == 0:
                dets_dict[cls_id] = torch.zeros((0, 6), device=logits.device)
                reid_dict[cls_id] = torch.zeros((0, reid.shape[-1]), device=logits.device)
                continue

            cls_boxes  = boxes_xyxy[keep]               # (M, 4)
            cls_scores_f = cls_scores[keep]             # (M,)
            cls_reid   = reid_norm[keep]                # (M, reid_dim)
            cls_id_col = torch.full((cls_scores_f.shape[0], 1), cls_id,
                                    dtype=cls_scores_f.dtype, device=logits.device)

            dets_dict[cls_id] = torch.cat([cls_boxes, cls_scores_f.unsqueeze(-1), cls_id_col], dim=-1)
            reid_dict[cls_id] = cls_reid

        return dets_dict, reid_dict
