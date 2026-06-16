"""
Temporal-LiDAR support for B7-B (multi-frame on the LiDAR side).

V2 (2026-05-10): refactored as a *subclass* of BEVFusionCA. Earlier version
wrapped BEVFusionCA, which caused two real training bugs:

  • temporal_proj was lazy-built on first forward — AFTER the optimizer was
    constructed — so it never landed in any optimizer param group and
    never received updates (stayed exactly at identity-init for 3 epochs).
  • TemporalBEVFusionCA.loss didn't call img_aux_head, so the aux BEV
    heatmap loss was silently dropped.

Inheritance fixes both: loss / hook plumbing / aux head are inherited from
BEVFusionCA unchanged; we only override extract_feat to inject the
temporal LiDAR path. temporal_proj is built eagerly in __init__ so it's
visible to the optimizer.

Additional fix in V2: past branch runs the LiDAR encoder in eval mode
(BEVDet4D-style) so BatchNorm running stats only update from the current
distribution. Earlier dual-pass with both branches in train mode corrupted
BN running stats — the dominant cause of the −0.14 NDS regression in V1.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS

from .bevfusion_ca import BEVFusionCA


def _split_points_by_time(
    points: List[torch.Tensor],
    time_threshold: float,
    time_dim: int = 4,
):
    """Split each per-sample point cloud into (now, past) by time column."""
    now_list, past_list = [], []
    for p in points:
        if p.numel() == 0 or p.shape[1] <= time_dim:
            now_list.append(p)
            past_list.append(p)
            continue
        t = p[:, time_dim].abs()
        now_mask = t <= time_threshold
        past_mask = ~now_mask
        if now_mask.sum() == 0 or past_mask.sum() == 0:
            now_list.append(p)
            past_list.append(p)
        else:
            now_list.append(p[now_mask])
            past_list.append(p[past_mask])
    return now_list, past_list


@MODELS.register_module()
class TemporalBEVFusionCA(BEVFusionCA):
    """
    BEVFusionCA + past-LiDAR-sweep branch (BEVDet4D-style).

    Inherits all of BEVFusionCA's loss / predict / hook plumbing and only
    overrides `extract_feat` to inject a second LiDAR-encoder pass on the
    past portion of the aggregated multi-sweep cloud.

    Args:
        time_threshold: float
            Points with |time| ≤ τ → "now" branch, > τ → "past" branch.
            Default 0.1 s = roughly the latest 2 sweeps at nuScenes' 20 Hz.
        **bevfusion_ca_kwargs: forwarded to BEVFusionCA.
    """

    def __init__(self, time_threshold: float = 0.1, **bevfusion_ca_kwargs):
        super().__init__(**bevfusion_ca_kwargs)
        self.time_threshold = float(time_threshold)

        # Eager temporal_proj so it's visible to the optimizer.
        # ConvFuser is configured with in_channels=[camera_ch, lidar_ch];
        # we want the lidar channel count for the 2C → C projection.
        fl_in = getattr(self.fusion_layer, 'in_channels', None)
        if isinstance(fl_in, (list, tuple)) and len(fl_in) >= 2:
            lidar_C = int(fl_in[-1])
        else:
            lidar_C = 256                                   # fallback
        proj = nn.Conv2d(2 * lidar_C, lidar_C, kernel_size=1, bias=False)
        with torch.no_grad():
            # Identity-init on the "now" half so the temporal block starts as
            # a no-op and gradually learns to integrate past info.
            proj.weight.zero_()
            eye = torch.eye(lidar_C).view(lidar_C, lidar_C, 1, 1)
            proj.weight[:, :lidar_C].copy_(eye)
        self.temporal_proj = proj
        print(f"[TemporalBEVFusionCA] eager temporal_proj: "
              f"Conv2d({2*lidar_C}->{lidar_C}, identity-init on the 'now' half)")

    # ------------------------------------------------------------------
    # The only override: extract_feat injects a past-branch LiDAR pass.
    # ------------------------------------------------------------------

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        """
        Mirror BEVFusionCA.extract_feat but split the LiDAR cloud and run
        the LiDAR encoder twice (now with grad, past with no_grad in
        eval-mode BN). Concat & project back to the original lidar-BEV
        channel count, then proceed with fusion / pts_backbone / pts_neck.
        """
        import numpy as np
        from copy import deepcopy

        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features: List[torch.Tensor] = []

        # ---------- IMAGE BRANCH (identical to parent) ----------
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))
            lidar2image       = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.asarray(camera_intrinsics))
            camera2lidar      = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix    = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix  = imgs.new_tensor(np.asarray(lidar_aug_matrix))

            img_feature = self.extract_img_feat(
                imgs, deepcopy(points),
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, batch_input_metas,
            )
            features.append(img_feature)

        # ---------- LIDAR BRANCH (TEMPORAL) ----------
        pts_feature = self._extract_temporal_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        # ---------- LATE FUSION + post-fusion backbone+neck ----------
        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    # ------------------------------------------------------------------
    # Temporal LiDAR feature extraction with frozen-BN past branch
    # ------------------------------------------------------------------

    def _extract_temporal_pts_feat(self, batch_inputs_dict: Dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        points_now, points_past = _split_points_by_time(points, self.time_threshold)

        # CURRENT branch: train mode, with grad
        bev_now = self.extract_pts_feat({**batch_inputs_dict, 'points': points_now})

        # PAST branch: encoder modules in eval() mode so BN uses (and does NOT
        # update) running stats. This avoids the dual-distribution running-stat
        # corruption that destroyed B7 v1 (NDS 0.5463 vs B5 0.6848).
        enc_modules: List[nn.Module] = []
        for attr in ('pts_voxel_encoder', 'pts_middle_encoder'):
            mod = getattr(self, attr, None)
            if mod is not None:
                enc_modules.append(mod)

        saved_training = [m.training for m in enc_modules]
        try:
            for m in enc_modules:
                m.eval()
            with torch.no_grad():
                bev_past = self.extract_pts_feat(
                    {**batch_inputs_dict, 'points': points_past}
                )
        finally:
            for m, was_train in zip(enc_modules, saved_training):
                m.train(was_train)
        bev_past = bev_past.detach()

        # Concat + project back to bev_now's channel count.
        combined = torch.cat([bev_now, bev_past], dim=1)
        return self.temporal_proj(combined)
