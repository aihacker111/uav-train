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

    Output per image (dicts):
      high_dets_dict  {cls_id: Tensor(N, 6)}  score >= conf_thres
      low_dets_dict   {cls_id: Tensor(M, 6)}  low_conf_thres <= score < conf_thres
      reid_dict       {cls_id: Tensor(N, D)}  L2-normed embeddings for high-conf dets
    """

    def __init__(self, num_classes: int, conf_thres: float = 0.3,
                 low_conf_thres: float = 0.0,
                 num_top_queries: int = 300, nms_thres: float = 0.45):
        super().__init__()
        self.num_classes     = num_classes
        self.conf_thres      = conf_thres
        self.low_conf_thres  = low_conf_thres
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
            high_dets_dict: {cls_id: Tensor(M, 6)}  score >= conf_thres
            low_dets_dict:  {cls_id: Tensor(K, 6)}  low_conf_thres <= score < conf_thres
            reid_dict:      {cls_id: Tensor(M, D)}  L2-normalized embeddings for high-conf dets
        """
        logits = outputs['pred_logits'][0]  # (N, num_classes)
        boxes  = outputs['pred_boxes'][0]   # (N, 4) cxcywh norm
        reid   = outputs['pred_reid'][0]    # (N, reid_dim)

        orig_h, orig_w = orig_hw
        scores_all = logits.sigmoid()       # (N, num_classes)
        dev = logits.device
        reid_dim = reid.shape[-1]

        # Convert boxes cxcywh (letterbox-normalized) → xyxy in original image coords
        cx, cy, bw, bh = boxes.unbind(-1)

        if net_hw is not None:
            net_h, net_w = net_hw
            ratio = min(net_h / orig_h, net_w / orig_w)
            new_w = round(orig_w * ratio)
            new_h = round(orig_h * ratio)
            dw = (net_w - new_w) * 0.5
            dh = (net_h - new_h) * 0.5
            cx_px = cx * net_w
            cy_px = cy * net_h
            bw_px = bw * net_w
            bh_px = bh * net_h
            cx_orig = (cx_px - dw) / ratio
            cy_orig = (cy_px - dh) / ratio
            bw_orig = bw_px / ratio
            bh_orig = bh_px / ratio
        else:
            cx_orig = cx * orig_w
            cy_orig = cy * orig_h
            bw_orig = bw * orig_w
            bh_orig = bh * orig_h

        x1 = (cx_orig - bw_orig * 0.5).clamp(min=0, max=orig_w)
        y1 = (cy_orig - bh_orig * 0.5).clamp(min=0, max=orig_h)
        x2 = (cx_orig + bw_orig * 0.5).clamp(min=0, max=orig_w)
        y2 = (cy_orig + bh_orig * 0.5).clamp(min=0, max=orig_h)
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, 4)

        reid_norm = F.normalize(reid, dim=-1)  # (N, reid_dim)

        high_dets_dict = {}
        low_dets_dict  = {}
        reid_dict      = {}

        for cls_id in range(self.num_classes):
            cls_scores = scores_all[:, cls_id]  # (N,)

            # ── High-confidence detections ──────────────────────────────────────
            high_keep = cls_scores >= self.conf_thres
            if high_keep.sum() == 0:
                high_dets_dict[cls_id] = torch.zeros((0, 6), device=dev)
                reid_dict[cls_id]      = torch.zeros((0, reid_dim), device=dev)
            else:
                h_boxes  = boxes_xyxy[high_keep]
                h_scores = cls_scores[high_keep]
                h_reid   = reid_norm[high_keep]
                nms_idx  = torchvision.ops.nms(h_boxes, h_scores, self.nms_thres)
                h_boxes, h_scores, h_reid = h_boxes[nms_idx], h_scores[nms_idx], h_reid[nms_idx]
                cls_col = torch.full((h_scores.shape[0], 1), cls_id,
                                     dtype=h_scores.dtype, device=dev)
                high_dets_dict[cls_id] = torch.cat([h_boxes, h_scores.unsqueeze(-1), cls_col], dim=-1)
                reid_dict[cls_id]      = h_reid

            # ── Low-confidence detections (second-pass ByteTrack pool) ──────────
            if self.low_conf_thres > 0:
                low_keep = (cls_scores >= self.low_conf_thres) & (cls_scores < self.conf_thres)
                if low_keep.sum() == 0:
                    low_dets_dict[cls_id] = torch.zeros((0, 6), device=dev)
                else:
                    l_boxes  = boxes_xyxy[low_keep]
                    l_scores = cls_scores[low_keep]
                    nms_idx  = torchvision.ops.nms(l_boxes, l_scores, self.nms_thres)
                    l_boxes, l_scores = l_boxes[nms_idx], l_scores[nms_idx]
                    cls_col = torch.full((l_scores.shape[0], 1), cls_id,
                                         dtype=l_scores.dtype, device=dev)
                    low_dets_dict[cls_id] = torch.cat([l_boxes, l_scores.unsqueeze(-1), cls_col], dim=-1)
            else:
                low_dets_dict[cls_id] = torch.zeros((0, 6), device=dev)

        return high_dets_dict, low_dets_dict, reid_dict
