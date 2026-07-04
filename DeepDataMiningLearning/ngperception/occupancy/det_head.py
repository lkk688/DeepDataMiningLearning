"""
ngperception.occupancy.det_head
===============================

**M3 — one encoder, two heads.** A 3-D object-detection head that consumes the occupancy
module's **fused camera+LiDAR voxel volume** (`models/lss_occ.py`'s `vox`, shape (B,C,X,Y,Z)),
so occupancy and detection share a single front-end. The head collapses Z into the channel dim
→ a BEV feature map → a 2-D backbone → the anchor head (reused from `detection/`), producing
3-D boxes in the *same* grid the occupancy uses (x,y∈[-40,40], z∈[-1,5.4]).

Because it hangs off the occupancy encoder, the **camera / LiDAR / fusion ablation is free** —
it's whatever `LSSOccupancy` was built with (`lidar_fusion` / `lidar_only`).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..detection.pointpillars import AnchorHead, make_bev_backbone


class VoxelDetHead(nn.Module):
    """(B,C,X,Y,Z) fused voxel volume -> 3-D boxes. Z-collapse + BEV backbone + anchor head."""

    def __init__(self, in_channels, nz, pc_range, num_classes=1, det_channels=64,
                 backbone="res", anchor_sizes=((4.6, 1.97, 1.74),), anchor_bottom=-1.0, **head_kw):
        super().__init__()
        self.z_collapse = nn.Sequential(
            nn.Conv2d(in_channels * nz, det_channels, 3, padding=1),
            nn.BatchNorm2d(det_channels), nn.ReLU(inplace=True))
        self.backbone = make_bev_backbone(backbone, det_channels)
        self.head = AnchorHead(self.backbone.num_bev_features, pc_range, num_classes=num_classes,
                               anchor_sizes=anchor_sizes, anchor_bottom=anchor_bottom, **head_kw)

    def forward(self, vox):
        """vox (B,C,X,Y,Z). Collapse Z -> BEV (B, C*Z, Y, X) so H=Y,W=X aligns with anchor gen."""
        B, C, X, Y, Z = vox.shape
        bev = vox.permute(0, 1, 4, 3, 2).reshape(B, C * Z, Y, X)
        return self.head(self.backbone(self.z_collapse(bev)))

    def get_loss(self, pred, gt_list):
        return self.head.get_loss(pred, gt_list)

    def predict(self, pred, **kw):
        return self.head.predict(pred, **kw)
