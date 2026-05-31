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
                 num_top_queries: int = 300, nms_thres: float = 0.45):
        super().__init__()
        self.num_classes     = num_classes
        self.conf_thres      = conf_thres
        self.num_top_queries = num_top_queries
        self.nms_thres       = nms_thres

    def forward(self, outputs: dict, orig_hw: tuple, net_hw: tuple = None):
        """
        Args:
            outputs: ECTransformer output dict with keys:
                'pred_logits': (1, N, num_classes)
                'pred_boxes':  (1, N, 4)  cxcywh normalized [0,1] in letterbox space
                'pred_reid':   (1, N, reid_dim)
            orig_hw: (H, W) of the original image before letterboxing
            net_hw:  (H, W) of the network input (letterboxed). When provided the
                     inverse letterbox transform is applied so boxes land on the
                     original image correctly.  If None, a direct scale is used
                     (only correct when orig aspect ratio == net aspect ratio).

        Returns:
            dets_dict:  {cls_id: Tensor(M, 6)}  xyxy + score + cls in image coords
            reid_dict:  {cls_id: Tensor(M, reid_dim)}  L2-normalized embeddings
        """
        logits = outputs['pred_logits'][0]  # (N, num_classes)
        boxes  = outputs['pred_boxes'][0]   # (N, 4) cxcywh norm
        reid   = outputs['pred_reid'][0]    # (N, reid_dim)

        orig_h, orig_w = orig_hw
        scores_all = logits.sigmoid()       # (N, num_classes)

        # Convert boxes cxcywh (letterbox-normalized) → xyxy in original image coords
        cx, cy, bw, bh = boxes.unbind(-1)

        if net_hw is not None:
            # Inverse letterbox: remove padding then scale to original size
            net_h, net_w = net_hw
            ratio = min(net_h / orig_h, net_w / orig_w)
            new_w = round(orig_w * ratio)
            new_h = round(orig_h * ratio)
            dw = (net_w - new_w) * 0.5  # x padding in pixels
            dh = (net_h - new_h) * 0.5  # y padding in pixels

            # Step 1: pixel coords in letterbox image
            cx_px = cx * net_w
            cy_px = cy * net_h
            bw_px = bw * net_w
            bh_px = bh * net_h
            # Step 2: remove padding and scale back
            cx_orig = (cx_px - dw) / ratio
            cy_orig = (cy_px - dh) / ratio
            bw_orig = bw_px / ratio
            bh_orig = bh_px / ratio
        else:
            # Fallback: direct scale (assumes no aspect-ratio mismatch)
            cx_orig = cx * orig_w
            cy_orig = cy * orig_h
            bw_orig = bw * orig_w
            bh_orig = bh * orig_h

        x1 = (cx_orig - bw_orig * 0.5).clamp(min=0, max=orig_w)
        y1 = (cy_orig - bh_orig * 0.5).clamp(min=0, max=orig_h)
        x2 = (cx_orig + bw_orig * 0.5).clamp(min=0, max=orig_w)
        y2 = (cy_orig + bh_orig * 0.5).clamp(min=0, max=orig_h)
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

            cls_boxes    = boxes_xyxy[keep]             # (M, 4)
            cls_scores_f = cls_scores[keep]             # (M,)
            cls_reid     = reid_norm[keep]              # (M, reid_dim)

            # NMS per class — removes redundant overlapping predictions
            nms_keep = torchvision.ops.nms(cls_boxes, cls_scores_f, self.nms_thres)
            cls_boxes    = cls_boxes[nms_keep]
            cls_scores_f = cls_scores_f[nms_keep]
            cls_reid     = cls_reid[nms_keep]

            cls_id_col = torch.full((cls_scores_f.shape[0], 1), cls_id,
                                    dtype=cls_scores_f.dtype, device=logits.device)

            dets_dict[cls_id] = torch.cat([cls_boxes, cls_scores_f.unsqueeze(-1), cls_id_col], dim=-1)
            reid_dict[cls_id] = cls_reid

        return dets_dict, reid_dict
