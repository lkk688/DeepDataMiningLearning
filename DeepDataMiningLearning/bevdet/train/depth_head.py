"""
DepthHead - BEVDepth-style depth distribution prediction head.

Used as auxiliary supervision during training only; dropped at inference
(no extra parameters or compute in deployed checkpoints).

Pipeline
--------
1. Hook captures camera FPN features (P3 level) inside MultiTaskBEVFusion.
2. DepthHead predicts a per-pixel discrete depth distribution
       d_logits  : [B*Nc, D_bins, Hf, Wf]
3. build_depth_gt() projects raw LiDAR points into each camera's feature
   grid using lidar2img and img_aug_matrix to produce
       d_gt      : [B*Nc, Hf, Wf]   (long, ignore=-1)
4. Loss = CrossEntropy(d_logits, d_gt, ignore_index=-1)

Why it helps
------------
The camera branch of LSS-style architectures (including our CA-LSS) lifts
2D features to BEV using either implicit or explicit depth. BEVDepth showed
that adding explicit depth supervision dramatically tightens the lifted
features, since LiDAR provides essentially-free depth labels.

This module is independent of mmdet3d (uses only torch + numpy).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DepthHead
# ---------------------------------------------------------------------------

class DepthHead(nn.Module):
    """
    Args:
        in_channels:      camera FPN P3 channel count (256 in our config).
        hidden_channels:  small hidden dim of the head.
        d_bins:           number of discrete depth bins.
        image_size:       (H_img, W_img) of camera input.
        feature_size:     (Hf, Wf) of P3 feature map.
        dbound:           (d_min, d_max, d_step) — controls discretization.
                          d_bins = int((d_max - d_min) / d_step).
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        d_bins: Optional[int] = None,
        image_size: Tuple[int, int] = (256, 704),
        feature_size: Tuple[int, int] = (32, 88),
        dbound: Tuple[float, float, float] = (1.0, 60.0, 1.0),
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.feature_size = tuple(feature_size)
        self.dbound = tuple(dbound)

        if d_bins is None:
            d_min, d_max, d_step = self.dbound
            d_bins = int((d_max - d_min) / d_step)
        self.d_bins = int(d_bins)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.d_bins, kernel_size=1),
        )

    def forward(self, cam_feats: torch.Tensor) -> torch.Tensor:
        """
        cam_feats: [B, Nc, C, Hf, Wf] OR [B*Nc, C, Hf, Wf]
        Returns:   [B*Nc, D_bins, Hf, Wf]
        """
        if cam_feats.dim() == 5:
            B, Nc, C, Hf, Wf = cam_feats.shape
            cam_feats = cam_feats.reshape(B * Nc, C, Hf, Wf)
        elif cam_feats.dim() != 4:
            raise ValueError(f"DepthHead expects 4D or 5D input, got {cam_feats.shape}")
        return self.head(cam_feats.contiguous())

    def loss(
        self,
        depth_logits: torch.Tensor,  # [B*Nc, D_bins, Hf, Wf]
        depth_gt: torch.Tensor,      # [B*Nc, Hf, Wf]   long, ignore=-1
    ) -> Dict[str, torch.Tensor]:
        if depth_logits.shape[-2:] != depth_gt.shape[-2:]:
            raise ValueError(
                f"DepthHead.loss spatial mismatch: logits {depth_logits.shape} vs gt {depth_gt.shape}"
            )
        loss = F.cross_entropy(depth_logits, depth_gt, ignore_index=-1)
        # Coverage = fraction of cells with a valid label
        coverage = (depth_gt != -1).float().mean()
        return {
            "loss_depth": loss,
            "depth_coverage": coverage.detach(),
        }


# ---------------------------------------------------------------------------
# GT builder: project LiDAR points → per-camera feature-grid depth labels
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_depth_gt(
    points: List[torch.Tensor],
    lidar2img: torch.Tensor,
    img_aug_matrix: Optional[torch.Tensor],
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    dbound: Tuple[float, float, float],
    d_bins: Optional[int] = None,
) -> torch.Tensor:
    """
    Construct depth bin labels by projecting raw LiDAR points into each
    camera's feature grid.  For each feature cell we keep the MIN depth
    of the projected points (closest hit); cells without any projected
    point are marked as -1 (ignored by cross-entropy).

    Args
    ----
    points:         B-list of [N_i, >=3] LiDAR points in LiDAR frame
                    (only first 3 cols are used).
    lidar2img:      [B, Nc, 4, 4]  LiDAR → image (pixel) projection.
    img_aug_matrix: [B, Nc, 3, 3] or [B, Nc, 4, 4] or None — image
                    augmentation matrix applied AFTER lidar2img.
    image_size:     (H_img, W_img).
    feature_size:   (Hf, Wf).
    dbound:         (d_min, d_max, d_step).
    d_bins:         override #bins; default int((d_max-d_min)/d_step).

    Returns
    -------
    depth_gt: [B*Nc, Hf, Wf]  long, ignore=-1.
    """
    B, Nc, _, _ = lidar2img.shape
    Hf, Wf = feature_size
    H_img, W_img = image_size
    d_min, d_max, d_step_native = dbound
    if d_bins is None:
        d_bins = int((d_max - d_min) / d_step_native)
    # Effective discretization step: ensures the full [d_min, d_max] range
    # maps uniformly into d_bins, regardless of the view_transform's native step.
    d_step_eff = (d_max - d_min) / float(d_bins)

    sx = float(W_img) / float(Wf)  # pixels per feature-step (W)
    sy = float(H_img) / float(Hf)  # pixels per feature-step (H)

    device = lidar2img.device
    proj_dtype = torch.float32  # projection in fp32 for numerical safety

    depth_gt_full = torch.full((B, Nc, Hf, Wf), -1, dtype=torch.long, device=device)

    for b in range(B):
        pts = points[b]
        if pts is None or pts.numel() == 0:
            continue
        xyz = pts[:, :3].to(proj_dtype)
        N = xyz.shape[0]
        ones = torch.ones((N, 1), dtype=proj_dtype, device=device)
        homo = torch.cat([xyz, ones], dim=-1)  # [N, 4]

        for c in range(Nc):
            M = lidar2img[b, c].to(proj_dtype)
            if img_aug_matrix is not None:
                aug = img_aug_matrix[b, c].to(proj_dtype)
                if aug.shape == (3, 3):
                    aug4 = torch.eye(4, dtype=proj_dtype, device=device)
                    aug4[:3, :3] = aug
                    aug = aug4
                M = aug @ M

            proj = (M @ homo.T).T  # [N, 4]
            depth = proj[:, 2]
            in_front = depth > 0.5  # 0.5 m margin to avoid jitter near 0
            if in_front.sum() == 0:
                continue

            uv = proj[in_front, :2] / depth[in_front].unsqueeze(-1).clamp(min=1e-6)
            d = depth[in_front]

            u_feat = uv[:, 0] / sx
            v_feat = uv[:, 1] / sy

            valid = (
                (u_feat >= 0) & (u_feat < Wf)
                & (v_feat >= 0) & (v_feat < Hf)
                & (d >= d_min) & (d < d_max)
            )
            if valid.sum() == 0:
                continue

            u_feat = u_feat[valid].long()
            v_feat = v_feat[valid].long()
            d = d[valid]

            # Per-cell min depth via scatter_reduce. Match dtypes explicitly:
            # `d` is float32 (proj_dtype), so cell_depth must be float32 too.
            flat_idx = v_feat * Wf + u_feat
            cell_depth = torch.full(
                (Hf * Wf,), float("inf"), device=device, dtype=d.dtype
            )
            cell_depth.scatter_reduce_(0, flat_idx, d, reduce="amin", include_self=True)

            cell_depth = cell_depth.view(Hf, Wf)
            valid_cells = torch.isfinite(cell_depth)
            d_bin = ((cell_depth - d_min) / d_step_eff).long().clamp(0, d_bins - 1)

            cam_target = depth_gt_full[b, c]
            cam_target[valid_cells] = d_bin[valid_cells]

    # DENSIFY (env DEPTH_DENSIFY=1): nearest-fill the -1 cells so depth supervision
    # is dense (sparse LiDAR coverage is the bottleneck for a lift-splat detector).
    # Fast nearest-fill (min-bin = closest depth, matching per-cell min-depth);
    # SLIC-superpixel fill is the quality upgrade if this helps.
    import os as _os
    if _os.environ.get("DEPTH_DENSIFY", "").strip() in ("1", "true", "True"):
        depth_gt_full = _densify_bins(depth_gt_full)
    return depth_gt_full.view(B * Nc, Hf, Wf)


def _densify_bins(gt, iters=24):
    """gt [B,Nc,Hf,Wf] long, -1=empty -> nearest-fill empties with min-bin neighbor."""
    import torch.nn.functional as _F
    B, Nc, Hf, Wf = gt.shape
    g = gt.float().view(B * Nc, 1, Hf, Wf)
    valid = (g >= 0).float()
    big = torch.where(valid > 0, g, torch.full_like(g, 1e4))
    for _ in range(iters):
        empty = valid < 0.5
        if not bool(empty.any()):
            break
        nbr = -_F.max_pool2d(-big, 3, stride=1, padding=1)          # min bin in 3x3
        nbr_has = _F.max_pool2d(valid, 3, stride=1, padding=1) > 0.5
        fill = empty & nbr_has
        g = torch.where(fill, nbr, g)
        big = torch.where(fill, nbr, big)
        valid = torch.where(fill, torch.ones_like(valid), valid)
    out = torch.where(valid > 0.5, g.round().long(),
                      torch.full_like(g, -1).long())
    return out.view(B, Nc, Hf, Wf)
