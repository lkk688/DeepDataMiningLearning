"""
BEVOccHead – 3D occupancy prediction head.

Design
------
Input:  fused BEV features  [B, in_channels, H, W]  (typically [B, 256, 180, 180])
Output: occupancy logits    [B, num_classes, Z, H, W]

The BEV plane is lifted to 3-D by a learned per-voxel deconvolution along Z,
then classified with a small 3-D conv stack.

Loss
----
  loss_occ_ce      : cross-entropy with optional per-class weights
  loss_occ_lovasz  : Lovász-softmax (highly effective for imbalanced voxels)

Dataset compatibility
---------------------
  nuScenes + Occ3D-nuScenes : num_classes=17 (16 semantic + 1 free)
  Binary (free / occupied)  : num_classes=2

  GT format: integer label tensor [B, Z, H, W]; -1 → ignore.

LiDAR pseudo-occupancy (no Occ3D dataset required)
---------------------------------------------------
  If you do not have Occ3D labels, build a binary occupancy GT from
  the raw LiDAR sweep: voxelize points and mark occupied=1 / free=0.
  Set num_classes=2 and pass pseudo_gt=True to the data pipeline.

No mmdet3d dependency.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import lovasz_softmax


# ---------------------------------------------------------------------------
# Occ3D-nuScenes class metadata (for reference when building class weights)
# ---------------------------------------------------------------------------
OCC3D_NUSCENES_CLASSES = [
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain",
    "manmade", "vegetation",
    # index 17 = free space (no label needed if empty voxels are marked as free class)
]

# Recommended weights: upweight rare objects, downweight driveable/free
OCC3D_CLASS_WEIGHTS_17: List[float] = [
    1.0,  # others
    2.0,  # barrier
    3.0,  # bicycle
    2.0,  # bus
    2.0,  # car
    3.0,  # construction_vehicle
    3.0,  # motorcycle
    3.0,  # pedestrian
    3.0,  # traffic_cone
    2.0,  # trailer
    2.0,  # truck
    1.0,  # driveable_surface
    1.0,  # other_flat
    1.0,  # sidewalk
    1.0,  # terrain
    1.0,  # manmade
    1.0,  # vegetation
]


# ---------------------------------------------------------------------------
# BEVOccHead
# ---------------------------------------------------------------------------

class BEVOccHead(nn.Module):
    """
    Args:
        in_channels:      channels from ConvFuser  (default 256)
        num_classes:      17 for Occ3D, 2 for binary LiDAR pseudo-occ
        bev_h, bev_w:     spatial size of input BEV grid (default 180 × 180)
        num_z:            number of Z slices in output voxel grid
                          16 → 0.5 m/slice over [-5, 3] m = 8 m range
        z_range:          [z_min, z_max]  (informational; not used in forward)
        hidden_channels:  internal channel width
        lovasz_weight:    weight for Lovász term (0 to disable)
        class_weights:    list of per-class CE weights (None → uniform)
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 17,
        bev_h: int = 180,
        bev_w: int = 180,
        num_z: int = 16,
        z_range: List[float] = None,
        hidden_channels: int = 128,
        lovasz_weight: float = 1.0,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_z = num_z
        self.z_range = z_range or [-5.0, 3.0]
        self.lovasz_weight = lovasz_weight

        # Step 1: compress BEV channels and simultaneously expand into Z slices.
        # [B, in_channels, H, W] → [B, hidden * num_z, H, W]
        self.bev_to_voxel = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * num_z, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * num_z),
            nn.ReLU(inplace=True),
        )

        # Step 2: lightweight 3-D refinement + classification.
        # [B, hidden, Z, H, W] → [B, num_classes, Z, H, W]
        mid = hidden_channels // 2
        self.voxel_cls = nn.Sequential(
            nn.Conv3d(hidden_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, num_classes, kernel_size=1),
        )

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_feat: [B, C, H, W]
        Returns:
            occ_logits: [B, num_classes, Z, H, W]
        """
        B, _C, H, W = bev_feat.shape
        x = self.bev_to_voxel(bev_feat)               # [B, hidden*Z, H, W]
        x = x.view(B, -1, self.num_z, H, W)           # [B, hidden, Z, H, W]
        return self.voxel_cls(x)                        # [B, num_classes, Z, H, W]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        occ_logits: torch.Tensor,   # [B, C, Z, H, W]
        occ_gt: torch.Tensor,        # [B, Z, H, W]  int64; -1 = ignore
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with 'loss_occ_ce' and optionally 'loss_occ_lovasz'.
        """
        B, C, Z, H, W = occ_logits.shape

        # Flatten spatial dims: [B*Z*H*W, C] and [B*Z*H*W]
        logits_flat = occ_logits.permute(0, 2, 3, 4, 1).reshape(-1, C)
        gt_flat = occ_gt.reshape(-1).long()

        losses: Dict[str, torch.Tensor] = {}

        # Cross-entropy
        ce = F.cross_entropy(
            logits_flat,
            gt_flat,
            weight=self.class_weights,
            ignore_index=-1,
        )
        losses["loss_occ_ce"] = ce

        # Lovász-softmax
        if self.lovasz_weight > 0.0:
            lov = lovasz_softmax(
                occ_logits,
                occ_gt,
                ignore_index=-1,
                classes="present",
            )
            losses["loss_occ_lovasz"] = self.lovasz_weight * lov

        return losses

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices [B, Z, H, W]."""
        logits = self.forward(bev_feat)
        return logits.argmax(dim=1)


# ---------------------------------------------------------------------------
# LiDAR pseudo-occupancy builder (no Occ3D labels needed)
# ---------------------------------------------------------------------------

def build_pseudo_occ_gt(
    points: torch.Tensor,   # [N, 3+] in LiDAR frame (x, y, z, ...)
    pc_range: List[float],  # [x_min, y_min, z_min, x_max, y_max, z_max]
    bev_h: int,
    bev_w: int,
    num_z: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Voxelizes raw LiDAR points into a binary [Z, H, W] occupancy grid.
    Occupied = 1, Free = 0, Out-of-range = -1 (ignored in loss).

    Intended for use as a cheap occupancy target when Occ3D labels
    are not available.  num_classes should be 2 when using this GT.

    Returns: [Z, H, W]  int64
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    xy_step_h = (x_max - x_min) / bev_h
    xy_step_w = (y_max - y_min) / bev_w
    z_step = (z_max - z_min) / num_z

    pts = points[:, :3].to(device)
    # Filter to range
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] < x_max)
        & (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
        & (pts[:, 2] >= z_min) & (pts[:, 2] < z_max)
    )
    pts = pts[mask]

    grid = torch.zeros(num_z, bev_h, bev_w, dtype=torch.int64, device=device)
    if pts.shape[0] == 0:
        return grid

    ix = ((pts[:, 0] - x_min) / xy_step_h).long().clamp(0, bev_h - 1)
    iy = ((pts[:, 1] - y_min) / xy_step_w).long().clamp(0, bev_w - 1)
    iz = ((pts[:, 2] - z_min) / z_step).long().clamp(0, num_z - 1)
    grid[iz, ix, iy] = 1

    return grid
