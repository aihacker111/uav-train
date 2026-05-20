"""
DNQueryGenerator: denoising training queries from GT boxes / labels.

During each training forward pass:
  1.  For each image, num_dn_groups independently-noised copies of each GT box
      and label are created.
  2.  Box noise: Gaussian noise scaled by dn_box_noise_scale, clipped to [0,1].
  3.  Label noise: with probability dn_label_noise_ratio each label is replaced
      by a uniformly random class (including the true class — no forced mismatch).
  4.  Content queries are derived from a learnable label embedding table.
  5.  Reference points are the noised box coordinates in logit (unsigmoid) space.
  6.  All images in the batch are padded to max_gt × num_dn_groups DN queries.

The returned DNMeta carries the GT assignments needed by the loss (no matching).

Attention mask (build_attn_mask):
  Prevents the decoder self-attention from leaking information across boundaries:
    • detect queries  ←→  DN queries : fully blocked
    • DN group g      ←→  DN group g': blocked for g ≠ g'
  Within a detect group or within a single DN group, attention is unrestricted.
  Shape: (K_detect + K_dn, K_detect + K_dn)  dtype=torch.bool  (True=blocked).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import DNConfig
from .query_gen import QueryBundle

_EPS = 1e-6


# ── DN meta ───────────────────────────────────────────────────────────────────

@dataclass
class DNMeta:
    dn_num_queries: int           # total DN query slots per image (max_gt * G, padded)
    dn_num_groups:  int           # effective number of DN groups used
    gt_labels:      List[Tensor]  # per-image original GT labels (before noise)
    gt_boxes:       List[Tensor]  # per-image original GT boxes cxcywh [0,1]
    batch_sizes:    Tensor        # (B,) — number of valid GTs per image


# ── DNQueryGenerator ──────────────────────────────────────────────────────────

class DNQueryGenerator(nn.Module):
    """
    Creates denoising queries for DN-DETR style training.

    Parameters
    ----------
    hidden_dim : int
        Matches the decoder hidden dimension D.
    num_classes : int
        Number of foreground classes.  The embedding table has num_classes + 1
        rows (+1 for the "noised / unknown" slot, unused at inference).
    cfg : DNConfig
        Noise and capacity hyper-parameters.
    """

    def __init__(self, hidden_dim: int, num_classes: int, cfg: DNConfig) -> None:
        super().__init__()
        self.num_dn_groups        = cfg.num_dn_groups
        self.label_noise_ratio    = cfg.dn_label_noise_ratio
        self.box_noise_scale      = cfg.dn_box_noise_scale
        self.max_dn_queries       = cfg.max_dn_queries
        self.num_classes          = num_classes

        # One embedding per class + one "noise" slot
        self.label_embed = nn.Embedding(num_classes + 1, hidden_dim)
        nn.init.normal_(self.label_embed.weight, std=0.02)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid_to_logit(x: Tensor) -> Tensor:
        x = x.clamp(_EPS, 1.0 - _EPS)
        return torch.log(x / (1.0 - x))

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        targets: List[dict],
        device:  torch.device,
    ) -> tuple[Optional[QueryBundle], Optional[DNMeta]]:
        """
        Parameters
        ----------
        targets : list of dicts with 'boxes' (N,4) cxcywh and 'labels' (N,) long
        device  : target device for all output tensors

        Returns
        -------
        (QueryBundle, DNMeta)  or  (None, None) when no GT exists in the batch.
        """
        B           = len(targets)
        batch_sizes = [len(t['boxes']) for t in targets]
        max_gt      = max(batch_sizes)

        if max_gt == 0:
            return None, None

        # Cap to memory limit: reduce groups before reducing per-image queries
        G      = min(self.num_dn_groups,
                     self.max_dn_queries // max(max_gt, 1))
        G      = max(G, 1)
        K_dn   = max_gt * G          # per-image DN slot count (padded to max_gt)

        all_content, all_refs = [], []
        all_scores,  all_cls  = [], []

        for b in range(B):
            n_gt   = batch_sizes[b]
            boxes  = targets[b]['boxes'].to(device, dtype=torch.float32)   # (n_gt, 4)
            labels = targets[b]['labels'].to(device)                        # (n_gt,)

            if n_gt == 0:
                # Fully-padded image — use zero content / zero refs (not used in loss)
                all_content.append(torch.zeros(K_dn, self.label_embed.embedding_dim,
                                               device=device))
                all_refs.append(torch.zeros(K_dn, 4, device=device))
                all_scores.append(torch.zeros(K_dn, device=device))
                all_cls.append(torch.zeros(K_dn, dtype=torch.long, device=device))
                continue

            # Repeat each GT for G groups  →  (n_gt * G,)
            boxes_rep  = boxes.repeat(G, 1)     # (n_gt*G, 4)
            labels_rep = labels.repeat(G)       # (n_gt*G,)

            # ── Label noise ────────────────────────────────────────────────
            if self.label_noise_ratio > 0.0:
                noise_mask     = torch.rand(len(labels_rep), device=device) < self.label_noise_ratio
                random_labels  = torch.randint(0, self.num_classes,
                                               (int(noise_mask.sum().item()),),
                                               device=device)
                labels_rep     = labels_rep.clone()
                labels_rep[noise_mask] = random_labels

            # ── Box noise ──────────────────────────────────────────────────
            # cx/cy: additive Gaussian scaled by dn_box_noise_scale / 2
            # w/h  : multiplicative Gaussian scaled by dn_box_noise_scale × box_dim
            noise      = torch.randn_like(boxes_rep)
            cxcy_noise = noise[:, :2] * (self.box_noise_scale * 0.5)
            wh_noise   = noise[:, 2:] * self.box_noise_scale * boxes_rep[:, 2:].detach()
            noised_boxes = (boxes_rep + torch.cat([cxcy_noise, wh_noise], dim=-1)).clamp(_EPS, 1.0 - _EPS)

            # ── Content from label embeddings ──────────────────────────────
            content_rep = self.label_embed(labels_rep)   # (n_gt*G, D)

            # ── Reference points in logit space ────────────────────────────
            ref_logit_rep = self._sigmoid_to_logit(noised_boxes)  # (n_gt*G, 4)

            # ── Pad to K_dn = max_gt * G ───────────────────────────────────
            K_actual = n_gt * G
            if K_actual < K_dn:
                pad = K_dn - K_actual
                content_rep   = F.pad(content_rep,   (0, 0, 0, pad))
                ref_logit_rep = F.pad(ref_logit_rep, (0, 0, 0, pad))
                scores_b = torch.cat([torch.ones(K_actual, device=device),
                                      torch.zeros(pad, device=device)])
                cls_b    = torch.cat([labels_rep,
                                      torch.zeros(pad, dtype=torch.long, device=device)])
            else:
                scores_b = torch.ones(K_actual, device=device)
                cls_b    = labels_rep

            all_content.append(content_rep)
            all_refs.append(ref_logit_rep)
            all_scores.append(scores_b)
            all_cls.append(cls_b)

        dn_bundle = QueryBundle(
            ref_points = torch.stack(all_refs,    dim=0),   # (B, K_dn, 4)
            content    = torch.stack(all_content, dim=0),   # (B, K_dn, D)
            scores     = torch.stack(all_scores,  dim=0),   # (B, K_dn)
            classes    = torch.stack(all_cls,     dim=0),   # (B, K_dn)
        )

        dn_meta = DNMeta(
            dn_num_queries = K_dn,
            dn_num_groups  = G,
            gt_labels      = [t['labels'].to(device) for t in targets],
            gt_boxes       = [t['boxes'].to(device, dtype=torch.float32) for t in targets],
            batch_sizes    = torch.tensor(batch_sizes, device=device),
        )

        return dn_bundle, dn_meta

    # ── attention mask ────────────────────────────────────────────────────────

    @staticmethod
    def build_attn_mask(
        K_detect:      int,
        K_dn:          int,
        dn_num_groups: int,
        max_gt:        int,
        device:        torch.device,
    ) -> Tensor:
        """
        Build a bool attention mask for the decoder self-attention.

        True  = this (query, key) pair is blocked (ignored).
        False = attention is allowed.

        Layout:
          [0 : K_detect]         — detect queries
          [K_detect : K_detect + max_gt * g]  — DN group g

        Blocked connections:
          detect ↔ DN (all groups)
          DN group g ↔ DN group g' (g ≠ g')
        """
        K_total = K_detect + K_dn
        mask    = torch.zeros(K_total, K_total, dtype=torch.bool, device=device)

        if K_dn == 0:
            return mask

        # detect ↔ DN: fully blocked in both directions
        mask[:K_detect, K_detect:] = True
        mask[K_detect:, :K_detect] = True

        # DN cross-group: blocked
        for g in range(dn_num_groups):
            g_s = K_detect + g * max_gt
            g_e = K_detect + (g + 1) * max_gt
            for g2 in range(dn_num_groups):
                if g2 == g:
                    continue
                g2_s = K_detect + g2 * max_gt
                g2_e = K_detect + (g2 + 1) * max_gt
                mask[g_s:g_e, g2_s:g2_e] = True

        return mask
