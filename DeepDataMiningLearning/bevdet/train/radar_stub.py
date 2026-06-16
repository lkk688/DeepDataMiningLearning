"""
RadarPillarEncoder – Phase 2 stub for radar-camera-LiDAR fusion.

Status: STUB – interface is defined and documented; forward() raises
        NotImplementedError.  Implement when ready for Phase 2.

nuScenes Radar Data Format
--------------------------
Each radar sweep provides N points with 18 attributes:
    [x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp,
     is_quality_valid, ambig_state, x_rms, y_rms, invalid_states,
     pdh0, vx_rms, vy_rms]

Useful dims for a pillar encoder: [x, y, z, rcs, vx_comp, vy_comp] = 6 dims.
nuScenes has 5 radar sensors (front, front-left, front-right, back-left, back-right).

Integration point (how to wire into MultiTaskBEVFusion)
--------------------------------------------------------
The radar BEV tensor [B, R_channels, H, W] is concatenated to the LiDAR BEV
[B, 256, H, W] BEFORE the ConvFuser.  The ConvFuser's in_channels must include
the radar channels:

    # In config (updated for radar):
    fusion_layer = dict(
        type='ConvFuser',
        in_channels=[camera_ch, lidar_ch + radar_ch],  # e.g. [160, 256+64]
        out_channels=256,
    )

    # In MultitaskBEVFusion.loss():
    if self.radar_encoder:
        radar_bev = self.radar_encoder(batch_inputs['radar_points'], batch_metas)
        # radar_bev: [B, R_ch, H, W]
        # Concatenate to LiDAR BEV before fusion_layer
        # (requires hooking into extract_feat or modifying BEVFusionCA)

Recommended Phase 2 approach
-----------------------------
1. Add radar point loading to the mmdet3d data pipeline:
       dict(type='LoadRadarPointsFromMultiSweeps', sweeps_num=6, use_dim=[0,1,2,5,8,9])
2. Implement RadarPillarEncoder (PointPillars-style, lightweight).
3. In MultiTaskBEVFusion, hook into SECOND's BEV output to concatenate radar BEV.
4. Update ConvFuser in_channels to include radar channels.

See: CenterFusion, CRN, RCBEVDet for reference implementations.
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn


class RadarPillarEncoder(nn.Module):
    """
    Lightweight pillar-based radar feature extractor.

    Input:  radar points [N, input_dim] per batch (variable length)
    Output: radar BEV feature map [B, out_channels, bev_h, bev_w]

    Args:
        input_dim:    number of features per radar point (default 6: x,y,z,rcs,vx,vy)
        out_channels: output BEV channels (default 64; added to LiDAR BEV)
        pc_range:     [x_min,y_min,z_min,x_max,y_max,z_max]
        voxel_size:   [dx, dy, dz]  (should match LiDAR voxel xy for BEV alignment)
        max_points_per_pillar: cap for pillar pooling
        max_pillars:  max active pillars per sample
    """

    # Expected interface contract (Phase 2 implementer must match this)
    # forward(radar_points: List[Tensor[N,D]], batch_metas) -> Tensor[B, out_channels, H, W]

    def __init__(
        self,
        input_dim: int = 6,
        out_channels: int = 64,
        pc_range: List[float] = None,
        voxel_size: List[float] = None,
        max_points_per_pillar: int = 10,
        max_pillars: int = 10000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_channels = out_channels
        self.pc_range = pc_range or [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = voxel_size or [0.075, 0.075, 8.0]  # coarse Z for radar
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        # BEV spatial dimensions (same as LiDAR BEV for direct concatenation)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        dx, dy, _ = self.voxel_size
        self.bev_h = int(round((x_max - x_min) / dx))   # 1440
        self.bev_w = int(round((y_max - y_min) / dy))   # 1440
        # After SECOND 8× downsampling: 180 × 180 → set voxel_size to match

        # Pillar feature network (PointNet-style MLP)
        # Placeholder – implement in Phase 2
        self.pfn = nn.Sequential(
            nn.Linear(input_dim + 3, 64),   # +3 for pillar center offset
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Scatter BEV conv
        self.bev_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        radar_points: List[torch.Tensor],   # list of [N_i, input_dim] per sample
        batch_metas: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Returns: [B, out_channels, bev_h, bev_w]

        NOT IMPLEMENTED – Phase 2.
        The return value below is a zeros tensor so the training pipeline
        can be tested end-to-end without real radar features.
        """
        # TODO Phase 2: implement proper pillar voxelization and scatter
        raise NotImplementedError(
            "RadarPillarEncoder.forward() is not yet implemented. "
            "This is a Phase 2 feature. Set use_radar=False in MultiTaskBEVFusion "
            "to train without radar."
        )

        # Reference implementation skeleton (fill in for Phase 2):
        #
        # B = len(radar_points)
        # bev_feats = torch.zeros(B, self.out_channels, self.bev_h, self.bev_w,
        #                         device=radar_points[0].device)
        # for b, pts in enumerate(radar_points):
        #     if pts.shape[0] == 0:
        #         continue
        #     # 1. Compute pillar indices (ix, iy)
        #     # 2. Clamp to valid range
        #     # 3. Compute offset features (x-pillar_center, y-pillar_center, z)
        #     # 4. Max-pool per pillar via scatter_max or loop
        #     # 5. PFN forward
        #     # 6. Scatter to BEV canvas
        # return self.bev_conv(bev_feats)


class RadarNullEncoder(nn.Module):
    """
    Drop-in replacement for RadarPillarEncoder that outputs zeros.
    Useful for ablation: enables radar data path in config without actual encoding.
    """

    def __init__(self, out_channels: int = 64, bev_h: int = 180, bev_w: int = 180):
        super().__init__()
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, radar_points, batch_metas=None) -> torch.Tensor:
        B = len(radar_points)
        dev = radar_points[0].device if radar_points else torch.device("cpu")
        return torch.zeros(B, self.out_channels, self.bev_h, self.bev_w, device=dev)
