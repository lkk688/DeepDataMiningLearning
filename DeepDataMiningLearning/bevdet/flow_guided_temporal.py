"""
Flow-Guided Temporal Attention (B10).

Why
---
B8 V2 / B9 used a *per-cell* cross-frame attention over K time tokens with no
prior on where to attend across time. With objects moving up to ~40 BEV cells
between t=0 and t=1.5 s (0.6 m / cell at 0.075 voxel × 8× downsample, and
nuScenes vehicles up to ~16 m/s), the attention has to discover a large
spatial correspondence purely from gradient. With our 3-epoch fine-tune
budget that just doesn't happen — B5R, B8 V2 and B9 all sit within ±0.005
NDS of each other.

This module addresses both ends of that problem:

  1. **Dense flow head** predicts a per-cell 2D BEV velocity v_pred(h, w) ∈
     ℝ² in m/s. Supervised directly by GT velocity on cells inside annotated
     object boxes — we already have these GTs from nuScenes.
  2. **Flow-guided warp** uses v_pred to pre-align each past token to the
     current frame before attention runs: past slice at lag τ_k gets sampled
     at (h - v_y·τ_k/cell, w - v_x·τ_k/cell). After the warp, an attention
     query at cell (h, w) sees the SAME object across all K tokens, and
     only needs to handle residual error.

The aux flow loss converges in the first few hundred iters (it's a direct
regression to GT, not a correspondence-discovery problem). Once flow is
roughly right, the temporal attention block becomes useful immediately —
which is what we need at the 3-6 epoch budget.

Pipeline (one forward pass on the LiDAR branch)
----------------------------------------------
::

    bev_main   = encode(full multi-sweep cloud)     # [B, C, H, W]
    bev_past_k = encode(past slice k)               # k=1..K-1, frozen-BN, no_grad
    v_pred     = flow_head(bev_main)                # [B, 2, H, W]

    for k = 1..K-1:
        past_aligned_k = grid_sample(bev_past_k, base_grid - v_pred·τ_k/cell)

    stack: [bev_main] + [past_aligned_1, ..., past_aligned_{K-1}]
    + temporal PE  → cross-frame attention → residual

    fused = bev_main + α · residual

Aux loss
--------
::

    v_gt = paint(per-object velocity, BEV cells inside box footprint)
    mask = paint(1, BEV cells inside box footprint)
    loss_flow_aux = SmoothL1(v_pred[mask>0], v_gt[mask>0])

This module is a strict superset of `multitoken_temporal_lidar.py`:
the warp degenerates to identity when v_pred = 0, and at α=0 the LiDAR
branch output equals `bev_main` bit-for-bit — same warm-start invariant
as B8 V2.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .bevfusion_ca import BEVFusionCA
from .multitoken_temporal_lidar import _split_points_into_past_slices


# ---------------------------------------------------------------------------
# GT velocity painter — rasterizes object boxes onto the BEV grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def paint_bev_velocity_gt(
    batch_data_samples,
    bev_h: int,
    bev_w: int,
    pc_range: Tuple[float, float, float, float],
    cell_size: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize each annotated object's box footprint onto a (B, 2, H, W)
    velocity field and a (B, 1, H, W) mask of where supervision is valid.

    pc_range: (xmin, ymin, xmax, ymax) in meters, matching the LiDAR BEV.
    cell_size: meters per BEV cell (== voxel_xy * downsample).

    Velocity components are read from the LAST 2 columns of bboxes_3d.tensor
    (the standard 9-d nuScenes box layout: [x, y, z, w, l, h, yaw, vx, vy]).
    After mmdet3d's GlobalRotScaleTrans + RandomFlip3D augmentations, both
    the box centers AND the velocity components are rotated/flipped together
    (mmdet3d treats the trailing 2 dims as a 2D velocity), so post-aug
    boxes-in-BEV-frame match post-aug LiDAR points.
    """
    B = len(batch_data_samples)
    v_gt = torch.zeros((B, 2, bev_h, bev_w), device=device, dtype=torch.float32)
    mask = torch.zeros((B, 1, bev_h, bev_w), device=device, dtype=torch.float32)
    xmin, ymin = pc_range[0], pc_range[1]

    for b, ds in enumerate(batch_data_samples):
        gi = getattr(ds, 'gt_instances_3d', None)
        if gi is None or not hasattr(gi, 'bboxes_3d') or gi.bboxes_3d is None:
            continue
        boxes = gi.bboxes_3d.tensor                                 # [N, 9]
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        cx, cy = boxes[:, 0], boxes[:, 1]
        # mmdet3d LiDARInstance3DBoxes 9-d layout: [x,y,z,w,l,h,yaw,vx,vy].
        # `w` is the box dim along the y-axis (lateral), `l` is along the
        # x-axis (longitudinal). With yaw rotation we honor footprint shape.
        dx, dy = boxes[:, 4], boxes[:, 3]   # l (x-extent), w (y-extent)
        yaw = boxes[:, 6]
        vx, vy = boxes[:, 7], boxes[:, 8]

        for i in range(boxes.shape[0]):
            li, wi = float(dx[i].item()), float(dy[i].item())
            if li <= 0 or wi <= 0:
                continue
            yi = float(yaw[i].item())
            cxi, cyi = float(cx[i].item()), float(cy[i].item())
            # coarse-grid sample of points inside the box, in box-local frame
            nx = max(2, int(np.ceil(li / cell_size)) + 1)
            ny = max(2, int(np.ceil(wi / cell_size)) + 1)
            xs = torch.linspace(-li / 2, li / 2, nx, device=device)
            ys = torch.linspace(-wi / 2, wi / 2, ny, device=device)
            xv, yv = torch.meshgrid(xs, ys, indexing='xy')
            cos_y, sin_y = float(np.cos(yi)), float(np.sin(yi))
            xr = cos_y * xv - sin_y * yv + cxi
            yr = sin_y * xv + cos_y * yv + cyi
            # world → cell indices
            cols = torch.floor((xr - xmin) / cell_size).long().view(-1)
            rows = torch.floor((yr - ymin) / cell_size).long().view(-1)
            in_grid = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
            if not in_grid.any():
                continue
            rr, cc = rows[in_grid], cols[in_grid]
            v_gt[b, 0, rr, cc] = float(vx[i].item())
            v_gt[b, 1, rr, cc] = float(vy[i].item())
            mask[b, 0, rr, cc] = 1.0

    return v_gt, mask


# ---------------------------------------------------------------------------
# Warp helper
# ---------------------------------------------------------------------------

def _warp_past_bev(past_bev: torch.Tensor,
                   v_pred: torch.Tensor,
                   time_lag: float,
                   cell_size: float) -> torch.Tensor:
    """
    Shift past_bev so the object at v_pred·τ in the past is moved forward
    to its current frame position.

    For an output cell (h, w), sample past_bev at
        (h - v_y·τ/cell, w - v_x·τ/cell)

    past_bev : [B, C, H, W]
    v_pred   : [B, 2, H, W]   m/s,  channel 0 = vx,  channel 1 = vy
    time_lag : seconds (scalar, positive == older)
    cell_size: meters per BEV cell
    """
    B, _, H, W = past_bev.shape
    device = past_bev.device

    y = torch.arange(H, device=device, dtype=v_pred.dtype).view(1, H, 1).expand(B, H, W)
    x = torch.arange(W, device=device, dtype=v_pred.dtype).view(1, 1, W).expand(B, H, W)

    delta_x = v_pred[:, 0] * (time_lag / cell_size)   # [B, H, W]
    delta_y = v_pred[:, 1] * (time_lag / cell_size)
    src_x = x - delta_x
    src_y = y - delta_y

    # Normalize to [-1, 1] for grid_sample (align_corners=True is the
    # convention compatible with pixel-index math above).
    src_x_n = 2.0 * src_x / max(W - 1, 1) - 1.0
    src_y_n = 2.0 * src_y / max(H - 1, 1) - 1.0
    grid = torch.stack([src_x_n, src_y_n], dim=-1)    # [B, H, W, 2]
    return F.grid_sample(
        past_bev, grid,
        mode='bilinear', padding_mode='zeros', align_corners=True,
    )


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------

class FlowGuidedTemporalAttention(nn.Module):
    """
    Per-cell cross-frame attention over flow-aligned K tokens.

    Forward returns ``delta`` (the temporal residual) and ``v_pred`` (the
    predicted BEV velocity, exposed so the detector can compute the aux
    flow loss). The output projection is zero-initialized so ``delta = 0``
    at iteration 0 — preserves the warm-start invariant from B8 V2 / B9.
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 flow_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Tiny conv stack for BEV velocity prediction (3 layers).
        self.flow_head = nn.Sequential(
            nn.Conv2d(channels, flow_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, flow_hidden),
            nn.GELU(),
            nn.Conv2d(flow_hidden, flow_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, flow_hidden),
            nn.GELU(),
            nn.Conv2d(flow_hidden, 2, kernel_size=1, bias=True),
        )
        # Zero-init the final conv so v_pred starts at 0 (= no warp at init).
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.out_proj.weight)        # identity-residual init

    def forward(self,
                bev_main: torch.Tensor,
                bev_past_list: List[torch.Tensor],
                time_lags: List[float],
                cell_size: float,
                temporal_pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        bev_main      : [B, C, H, W]  — current full-cloud BEV (with grad)
        bev_past_list : list of K-1 tensors each [B, C, H, W] (no_grad upstream)
        time_lags     : list of K floats; time_lags[0]=0, time_lags[k]=τ_k
        cell_size     : meters per BEV cell
        temporal_pe   : [K, C] learnable temporal positional embedding

        Returns:
            delta : [B, C, H, W]  — temporal residual to add to bev_main
            v_pred: [B, 2, H, W]  — predicted BEV velocity (m/s)
        """
        B, C, H, W = bev_main.shape
        assert C == self.channels
        K = 1 + len(bev_past_list)
        assert len(time_lags) == K

        # 1) predict flow from the current full-cloud BEV
        v_pred = self.flow_head(bev_main.float()).to(bev_main.dtype)   # [B,2,H,W]

        # 2) flow-warp each past token
        warped: List[torch.Tensor] = [bev_main]
        for k, past_bev in enumerate(bev_past_list, start=1):
            warped.append(_warp_past_bev(past_bev, v_pred, time_lags[k], cell_size))

        # 3) build the temporal stack + PE
        bev_stack = torch.stack(warped, dim=1)                          # [B, K, C, H, W]
        pe = temporal_pe.view(1, K, C, 1, 1).to(bev_stack.dtype)
        bev_stack = bev_stack + pe

        # 4) per-cell K-token attention (queries from token 0, K/V over all K)
        x = bev_stack.permute(0, 3, 4, 1, 2).reshape(B * H * W, K, C)
        q = self.q_proj(x[:, 0:1])                                      # [N, 1, C]
        k = self.k_proj(x)                                              # [N, K, C]
        v = self.v_proj(x)                                              # [N, K, C]

        def split(t: torch.Tensor) -> torch.Tensor:
            N, T, _ = t.shape
            return t.view(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = split(q)                                                    # [N, h, 1, d]
        k = split(k)                                                    # [N, h, K, d]
        v = split(v)                                                    # [N, h, K, d]
        attn = (q @ k.transpose(-2, -1)) * self.scale                   # [N, h, 1, K]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                                  # [N, h, 1, d]
        out = out.permute(0, 2, 1, 3).contiguous().view(B * H * W, 1, C)
        out = self.out_proj(out)                                        # zero-init → 0
        delta = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return delta, v_pred


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@MODELS.register_module()
class FlowGuidedTemporalBEVFusion(BEVFusionCA):
    """
    Subclass of BEVFusionCA. Replaces the LiDAR branch with:
        bev_main  = encode(full multi-sweep cloud)        # train mode, grad
        v_flow    = flow_head(bev_main)                   # predicted velocity
        bev_k     = encode(past slice k)                  # eval-BN, no_grad
        past_aligned_k = grid_sample(bev_k, current - v_flow·τ_k)
        delta = attn(token 0 = bev_main, tokens 1..K-1 = past_aligned_k)
        fused = bev_main + delta              # delta=0 at init (out_proj=0)
        v_temporal = temporal_aux_head(fused) # NEW: trains the temporal
                                              # block end-to-end (B10c)

    Adds TWO aux losses in ``loss()``:
      * ``loss_flow_aux``       — supervises ``flow_head`` (used internally
                                  for the warp) against GT velocity painted
                                  onto each annotated object's BEV cells.
      * ``loss_temporal_aux``   — supervises a SECOND velocity prediction
                                  read off the *post-temporal* fused BEV
                                  (``bev_main + delta``). This is the
                                  gradient path that actually trains the
                                  temporal-block weights end-to-end
                                  (``out_proj``, ``q/k/v_proj``,
                                  ``temporal_pe``, past-slice encoders).
                                  Without it, the bbox-head gradient is
                                  too dilute to move ``out_proj`` from
                                  zero-init in 3 epochs.

    Dead-pathway fix (B10c): the old design ALSO multiplied the residual
    by a learnable ``temporal_alpha`` initialized to 0. Combined with the
    zero-init ``out_proj.weight``, this created a gradient deadlock:
    ``scaled = alpha · delta = 0 · 0 = 0`` and ``d(scaled)/d(delta) =
    alpha = 0``, so neither parameter ever received a gradient. B10c
    drops ``temporal_alpha`` entirely; ``out_proj.weight = 0`` alone is
    sufficient for the warm-start invariant *and* lets gradient flow.

    Args:
        num_buckets: K (default 4). 1 main + K-1 past slices.
        window_seconds: total time span the multi-sweep cloud covers (s).
        num_attn_heads: heads inside the K-token attention.
        flow_loss_weight: weight on the SmoothL1 flow-aux loss.
        temporal_aux_loss_weight: weight on the SmoothL1 temporal-aux
            loss (supervises post-temporal velocity prediction).
        cell_size: meters per BEV cell at the LiDAR-encoder output
            resolution (0.075 voxel × 8× downsample = 0.6 m for the
            voxel0075 config).
        pc_range: (xmin, ymin, xmax, ymax) used for the GT painter.
        **bevfusion_ca_kwargs: forwarded to BEVFusionCA.
    """

    def __init__(self,
                 num_buckets: int = 4,
                 window_seconds: float = 1.5,
                 num_attn_heads: int = 4,
                 attn_dropout: float = 0.0,
                 flow_loss_weight: float = 0.5,
                 temporal_aux_loss_weight: float = 0.5,
                 cell_size: float = 0.6,
                 pc_range: Tuple[float, float, float, float] = (-54.0, -54.0, 54.0, 54.0),
                 **bevfusion_ca_kwargs):
        super().__init__(**bevfusion_ca_kwargs)
        assert num_buckets >= 2
        self.num_buckets = int(num_buckets)
        self.window_seconds = float(window_seconds)
        self.flow_loss_weight = float(flow_loss_weight)
        self.temporal_aux_loss_weight = float(temporal_aux_loss_weight)
        self.cell_size = float(cell_size)
        self.pc_range = tuple(pc_range)

        fl_in = getattr(self.fusion_layer, 'in_channels', None)
        lidar_C = int(fl_in[-1]) if isinstance(fl_in, (list, tuple)) and len(fl_in) >= 2 else 256
        self.lidar_channels = lidar_C

        # Learnable per-bucket temporal PE [K, C].
        self.temporal_pe = nn.Parameter(0.02 * torch.randn(self.num_buckets, lidar_C))

        self.temporal_attn = FlowGuidedTemporalAttention(
            channels=lidar_C, num_heads=num_attn_heads, dropout=attn_dropout,
        )

        # NOTE (B10c): no more ``temporal_alpha`` parameter. ``out_proj``
        # being zero-init is sufficient to make ``delta = 0`` at iter 0,
        # which preserves the warm-start invariant without creating the
        # dead-pathway double-zero trap that silently neutered B8/B9/B10b.

        # Temporal aux head: predicts per-cell velocity from the
        # post-temporal (fused) BEV. The gradient from its loss flows
        # back through this head INTO the temporal block (out_proj,
        # q/k/v_proj, temporal_pe, past-slice encoders), giving them a
        # strong, direct supervision signal. Without this, only the
        # dilute bbox-head gradient trains the temporal block —
        # empirically not enough to move out_proj off zero in 3 epochs.
        #
        # NOTE: the final conv is **NOT** zero-initialized. Zero-init
        # would block gradient from flowing backward through it to the
        # temporal block — defeating the whole point of having this
        # head. Instead we use a small-variance kaiming init so the
        # initial v_temporal magnitudes are modest (~m/s scale, not
        # 100 m/s) but the gradient chain stays connected.
        self.temporal_aux_head = nn.Sequential(
            nn.Conv2d(lidar_C, lidar_C // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, lidar_C // 2),
            nn.GELU(),
            nn.Conv2d(lidar_C // 2, 2, kernel_size=1, bias=True),
        )
        # Small-variance init on the final conv: keeps v_temporal at
        # ~O(1) m/s scale at iter 0 (loss is then dominated by GT, not
        # blown up by random predictions) while preserving gradient flow.
        nn.init.normal_(self.temporal_aux_head[-1].weight, std=0.01)
        nn.init.zeros_(self.temporal_aux_head[-1].bias)

        # Time-lag for each token: midpoint of its time interval.
        tau = self.window_seconds
        K = self.num_buckets
        self.time_lags: List[float] = [0.0] + [
            ((k + 0.5) * tau / K) for k in range(1, K)
        ]

        # Scratch slots set during forward, read by loss().
        self._cached_v_pred_flow: Optional[torch.Tensor] = None
        self._cached_v_pred_temporal: Optional[torch.Tensor] = None

        print(
            f'[FlowGuidedTemporalBEVFusion] K={K} buckets, window={tau:.2f}s, '
            f'channels={lidar_C}, heads={num_attn_heads}, '
            f'cell_size={self.cell_size:.2f}m, '
            f'flow_loss_weight={self.flow_loss_weight:.2f}, '
            f'temporal_aux_loss_weight={self.temporal_aux_loss_weight:.2f}, '
            f'lags={[f"{l:.3f}" for l in self.time_lags]}, '
            f'temporal_alpha=DROPPED (was dead-pathway bug)'
        )

    # ------------------------------------------------------------------
    # extract_feat: identical to MultiTokenTemporalBEVFusion except we
    # call our flow-guided _extract_temporal_pts_feat below.
    # ------------------------------------------------------------------

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features: List[torch.Tensor] = []
        img_feature = None

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
            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.asarray(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

            img_feature = self.extract_img_feat(
                imgs, deepcopy(points),
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, batch_input_metas,
            )

        pts_feature = self._extract_temporal_pts_feat(batch_inputs_dict)

        # Optional modality dropout / forced single-modality (no-op unless env set).
        # Reuses BEVFusionCA._modality_mask (inherited).
        if img_feature is not None:
            img_feature, pts_feature = self._modality_mask(img_feature, pts_feature)
            features.append(img_feature)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        self._last_bev = x  # stash final BEV for LC->C distillation (see loss())
        return x

    # ------------------------------------------------------------------
    # K-bucket flow-guided LiDAR feature extraction.
    # ------------------------------------------------------------------

    def _extract_temporal_pts_feat(self, batch_inputs_dict: Dict) -> torch.Tensor:
        points_full = batch_inputs_dict['points']

        # token 0: full-cloud BEV (warm-start-equivalent main path)
        bev_main = self.extract_pts_feat(batch_inputs_dict)

        # tokens 1..K-1: past slices (frozen-BN, no_grad)
        past_slices = _split_points_into_past_slices(
            points_full,
            num_past=self.num_buckets - 1,
            window_seconds=self.window_seconds,
        )
        enc_modules: List[nn.Module] = []
        for attr in ('pts_voxel_encoder', 'pts_middle_encoder'):
            mod = getattr(self, attr, None)
            if mod is not None:
                enc_modules.append(mod)

        saved_training = [m.training for m in enc_modules]
        bev_probes: List[torch.Tensor] = []
        try:
            for m in enc_modules:
                m.eval()
            with torch.no_grad():
                for k in range(self.num_buckets - 1):
                    bev_k = self.extract_pts_feat(
                        {**batch_inputs_dict, 'points': past_slices[k]}
                    )
                    bev_probes.append(bev_k.detach())
        finally:
            for m, was_train in zip(enc_modules, saved_training):
                m.train(was_train)

        # Flow-guided attention. Cache v_pred_flow for loss() to consume.
        delta, v_pred_flow = self.temporal_attn(
            bev_main=bev_main,
            bev_past_list=bev_probes,
            time_lags=self.time_lags,
            cell_size=self.cell_size,
            temporal_pe=self.temporal_pe,
        )
        self._cached_v_pred_flow = v_pred_flow

        # Fused BEV: bev_main + delta. ``out_proj.weight = 0`` makes
        # delta = 0 at iter 0, so fused == bev_main and the warm-start
        # invariant holds. Gradient flows through out_proj normally.
        fused = bev_main + delta.to(bev_main.dtype)

        # Temporal aux head: predict per-cell velocity from the fused
        # (temporal-aware) BEV. Its loss provides a strong direct
        # supervision signal for the temporal block.
        v_pred_temporal = self.temporal_aux_head(fused.float()).to(bev_main.dtype)
        self._cached_v_pred_temporal = v_pred_temporal

        return fused

    # ------------------------------------------------------------------
    # loss(): inherit BEVFusionCA's full loss path, then add the aux
    # flow loss. We piggyback on the parent's hook plumbing so we don't
    # have to redo aux-bev-heatmap / voxel-painting setup.
    # ------------------------------------------------------------------

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        self._cached_v_pred_flow = None
        self._cached_v_pred_temporal = None
        losses = super().loss(batch_inputs_dict, batch_data_samples, **kwargs)

        v_pred_flow = self._cached_v_pred_flow
        v_pred_temporal = self._cached_v_pred_temporal
        self._cached_v_pred_flow = None
        self._cached_v_pred_temporal = None
        if v_pred_flow is None:
            return losses

        # Paint GT velocity once and reuse for both aux losses.
        B, _, H, W = v_pred_flow.shape
        v_gt, mask = paint_bev_velocity_gt(
            batch_data_samples,
            bev_h=H, bev_w=W,
            pc_range=self.pc_range,
            cell_size=self.cell_size,
            device=v_pred_flow.device,
        )

        def _smooth_l1_masked(v_pred):
            # Float32 + mask-weighted SmoothL1 (Huber β=1.0)
            vp = v_pred.float() if v_pred.dtype != torch.float32 else v_pred
            m = mask.expand_as(vp)
            n_pos = m.sum().clamp(min=1.0)
            diff = (vp - v_gt) * m
            abs_d = diff.abs()
            sl1 = torch.where(abs_d < 1.0, 0.5 * diff * diff, abs_d - 0.5)
            return sl1.sum() / n_pos

        losses['loss_flow_aux'] = self.flow_loss_weight * _smooth_l1_masked(v_pred_flow)
        if v_pred_temporal is not None:
            losses['loss_temporal_aux'] = (
                self.temporal_aux_loss_weight * _smooth_l1_masked(v_pred_temporal)
            )
        return losses
