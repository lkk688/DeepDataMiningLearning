"""
QueryOccHead - Query-based occupancy decoder (Tesla-AI-Day-2022-inspired).

Replaces a dense voxel-logits head with:
  1) a small UNet that compresses the fused BEV into a "world tensor"
  2) a position-encoded MLP that takes (sampled world feature, sinusoidal pos
     encoding of (x,y,z)) and outputs occupancy logits per query point.

Decouples grid resolution from training memory:
  • Train on a random subsample of voxels per batch (e.g., 25%) → 4-14×
    activation-memory reduction vs dense BEVOccHead.
  • Inference at any resolution: query at every voxel center for full Occ3D,
    or sparsely from object proposals for embedded deployment.
  • Number of classes K only affects the final linear layer; memory is
    independent of K. Enables Occ3D-nuScenes (K=17) at the same memory cost
    as binary pseudo-occupancy (K=2).

This module is independent of mmdet3d (uses only torch + numpy).
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small UNet for the "world tensor" encoder
# ---------------------------------------------------------------------------

class _SmallBEVUNet(nn.Module):
    """
    Lightweight 2-level UNet on the BEV plane.
    Compresses [B, in_ch, H, W] → [B, out_ch, H, W] preserving spatial size,
    with one down + one up + skip connection so each cell has multi-scale
    context that the MLP can sample at any (x, y).
    """

    def __init__(self, in_ch: int = 256, mid_ch: int = 128, out_ch: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )
        # Down: stride 2
        self.down = nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, bias=False)
        self.enc2 = nn.Sequential(
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )
        # Up
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Decode with skip from enc1
        self.dec = nn.Sequential(
            nn.Conv2d(mid_ch + mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)            # [B, mid_ch, H, W]
        d  = self.down(e1)           # [B, mid_ch, H/2, W/2]
        e2 = self.enc2(d)            # [B, mid_ch, H/2, W/2]
        u  = self.up(e2)             # [B, mid_ch, H, W]
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        out = self.dec(torch.cat([u, e1], dim=1))  # [B, out_ch, H, W]
        return out


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding for 3D query points
# ---------------------------------------------------------------------------

def _sin_pos_encode(xyz: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """
    NeRF-style multi-frequency sinusoidal encoding.

    Args
    ----
    xyz       : [..., 3]     normalized to [-1, 1]
    num_freqs : F            number of octaves

    Returns
    -------
    pe        : [..., 3 * 2 * F]  (sin and cos at F frequencies for x, y, z)
    """
    if num_freqs <= 0:
        return xyz
    freqs = (2.0 ** torch.arange(num_freqs, device=xyz.device, dtype=xyz.dtype)) * math.pi
    # [..., 3, 1] * [F]  ->  [..., 3, F]
    a = xyz.unsqueeze(-1) * freqs
    sin_part = a.sin()
    cos_part = a.cos()
    # interleave: out[..., 3*2F]
    pe = torch.cat([sin_part, cos_part], dim=-1)  # [..., 3, 2F]
    return pe.flatten(-2)                         # [..., 3 * 2F]


# ---------------------------------------------------------------------------
# Manual bilinear sampling at flat (xy) queries with batch index
# ---------------------------------------------------------------------------

def _flat_bilinear_sample(
    feature_map: torch.Tensor,   # [B, C, H, W]
    xy_norm: torch.Tensor,       # [N_total, 2]   in [-1, 1]
    batch_idx: torch.Tensor,     # [N_total]      long, in [0, B)
) -> torch.Tensor:
    """
    Batched, ragged-friendly bilinear sampling.

    For each row i, sample feature_map[batch_idx[i], :, ., .] at the (x, y)
    given by xy_norm[i]. Returns a [N_total, C] tensor.
    """
    B, C, H, W = feature_map.shape

    # [-1, 1] → [0, W-1] / [0, H-1]
    x = (xy_norm[:, 0] + 1.0) * 0.5 * (W - 1)
    y = (xy_norm[:, 1] + 1.0) * 0.5 * (H - 1)

    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    x0 = x0.clamp(min=0)
    y0 = y0.clamp(min=0)

    wx = (x - x0.float()).unsqueeze(-1)
    wy = (y - y0.float()).unsqueeze(-1)

    # Gather four corners.
    # feature_map indexed by [batch_idx, :, y, x] gives [N_total, C] per corner.
    f00 = feature_map[batch_idx, :, y0, x0]
    f01 = feature_map[batch_idx, :, y0, x1]
    f10 = feature_map[batch_idx, :, y1, x0]
    f11 = feature_map[batch_idx, :, y1, x1]

    out = (
        f00 * (1 - wx) * (1 - wy)
        + f01 *      wx  * (1 - wy)
        + f10 * (1 - wx) *      wy
        + f11 *      wx  *      wy
    )
    return out  # [N_total, C]


# ---------------------------------------------------------------------------
# QueryOccHead
# ---------------------------------------------------------------------------

class QueryOccHead(nn.Module):
    """
    Args
    ----
    in_channels         : int    fused BEV channels (typically 256).
    world_channels      : int    world-tensor channel count (typically 64).
    mlp_hidden          : int    MLP hidden size.
    num_classes         : int    K (e.g., 2 for binary, 17 for Occ3D-nuScenes).
    pos_freqs           : int    sinusoidal encoding frequencies (default 6).
    grid_size           : (Z, H, W)  voxel grid for GT (default (16, 180, 180)).
    train_max_queries   : int    cap on queries per batch sample at train time.
                                  ~50k gives ~4× memory reduction vs full grid.
    class_weights       : Optional[Tensor]  per-class CE weight (length K).
    """

    def __init__(
        self,
        in_channels:        int = 256,
        world_channels:     int = 64,
        mlp_hidden:         int = 128,
        num_classes:        int = 2,
        pos_freqs:          int = 6,
        grid_size:          Tuple[int, int, int] = (16, 180, 180),
        train_max_queries:  int = 50_000,
        class_weights:      Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.world_channels = world_channels
        self.num_classes = num_classes
        self.pos_freqs = pos_freqs
        self.grid_z, self.grid_h, self.grid_w = grid_size
        self.train_max_queries = int(train_max_queries)

        self.unet = _SmallBEVUNet(in_ch=in_channels, mid_ch=128, out_ch=world_channels)

        pe_dim = 3 * 2 * pos_freqs if pos_freqs > 0 else 3
        self.mlp = nn.Sequential(
            nn.Linear(world_channels + pe_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_classes),
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights, persistent=False)
        else:
            self.class_weights = None

        # Cache normalized voxel-center grid (built lazily on first call).
        self.register_buffer("_voxel_grid_cached", torch.empty(0), persistent=False)

    # -------------------------------------------------------------------
    # Voxel-grid query construction
    # -------------------------------------------------------------------

    def _build_voxel_query_grid(self, device, dtype) -> torch.Tensor:
        """
        Returns [Z * H * W, 3] voxel centers in normalized [-1, 1]^3.
        Order matches `occ_gt.view(-1)` where occ_gt has shape [Z, H, W].
        """
        Z, H, W = self.grid_z, self.grid_h, self.grid_w
        # Centers, then normalize to [-1, 1]
        zs = (torch.arange(Z, device=device, dtype=dtype) + 0.5) / Z * 2 - 1
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H * 2 - 1
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W * 2 - 1
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [Z*H*W, 3]
        return grid

    # -------------------------------------------------------------------
    # Forward at arbitrary query points (used by inference & loss)
    # -------------------------------------------------------------------

    def _decode(
        self,
        world: torch.Tensor,     # [B, C_w, H, W]
        query_xyz: torch.Tensor, # [N_total, 3]
        batch_idx: torch.Tensor, # [N_total]
    ) -> torch.Tensor:
        """
        Decode logits at flat query points. Returns [N_total, num_classes].
        """
        # Sample world tensor at (x, y) of each query
        sampled = _flat_bilinear_sample(world, query_xyz[:, :2], batch_idx)
        # Position-encode (x, y, z)
        pe = _sin_pos_encode(query_xyz, self.pos_freqs)
        feat = torch.cat([sampled, pe], dim=-1)            # [N_total, C_w + pe_dim]
        return self.mlp(feat)                              # [N_total, K]

    def forward(
        self,
        fused_bev: torch.Tensor,        # [B, in_channels, H, W]
        query_xyz: Optional[torch.Tensor] = None,  # [B, N, 3] or None for full-grid
    ) -> torch.Tensor:
        """
        Inference-style forward. If `query_xyz` is None, queries every voxel center.
        Returns:
          • [B, N, K]  if query_xyz is provided
          • [B, K, Z, H, W]  if query_xyz is None (dense full-grid)
        """
        B = fused_bev.shape[0]
        world = self.unet(fused_bev)                              # [B, C_w, H, W]
        device, dtype = fused_bev.device, fused_bev.dtype

        if query_xyz is None:
            voxel_grid = self._build_voxel_query_grid(device, dtype)        # [Z*H*W, 3]
            N = voxel_grid.shape[0]
            # Replicate across batch and flatten
            flat_q  = voxel_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)
            flat_b  = torch.arange(B, device=device).repeat_interleave(N)
            logits  = self._decode(world, flat_q, flat_b)                    # [B*N, K]
            return logits.view(B, self.grid_z, self.grid_h, self.grid_w, self.num_classes
                              ).permute(0, 4, 1, 2, 3).contiguous()
        else:
            assert query_xyz.dim() == 3 and query_xyz.shape[-1] == 3
            B_q, N, _ = query_xyz.shape
            assert B_q == B
            flat_q = query_xyz.reshape(-1, 3)
            flat_b = torch.arange(B, device=device).repeat_interleave(N)
            logits = self._decode(world, flat_q, flat_b)                     # [B*N, K]
            return logits.view(B, N, self.num_classes)

    # -------------------------------------------------------------------
    # Training-time loss with sub-sampling
    # -------------------------------------------------------------------

    def loss(
        self,
        fused_bev: torch.Tensor,   # [B, C_in, H, W]
        occ_gt:    torch.Tensor,   # [B, Z, H, W]  long, ignore=-1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CE loss on a random subset of voxels per batch sample.

        Subsampling is what makes Q-Occ memory-efficient: instead of forwarding
        every voxel through the MLP, we randomly select up to `train_max_queries`
        voxels per sample, only forward those, and compute CE on them.
        """
        assert occ_gt.dim() == 4, f"occ_gt must be [B, Z, H, W], got {occ_gt.shape}"
        B, Z, H, W = occ_gt.shape
        assert (Z, H, W) == (self.grid_z, self.grid_h, self.grid_w), (
            f"grid mismatch: head expects {(self.grid_z, self.grid_h, self.grid_w)} "
            f"but occ_gt is {(Z, H, W)}"
        )
        device = fused_bev.device
        dtype  = fused_bev.dtype

        # Compute world tensor once per batch
        world = self.unet(fused_bev)                                 # [B, C_w, H, W]

        # Build voxel query grid (cached once per device)
        voxel_grid = self._build_voxel_query_grid(device, dtype)     # [Z*H*W, 3]
        N = voxel_grid.shape[0]
        targets_flat = occ_gt.reshape(B, -1)                          # [B, N]

        # Per-batch valid (non-ignore) and optionally subsampled indices
        flat_q_list  = []
        flat_b_list  = []
        flat_t_list  = []
        for b in range(B):
            valid_idx = (targets_flat[b] >= 0).nonzero(as_tuple=True)[0]   # [N_valid]
            if self.training and self.train_max_queries > 0 and \
                    valid_idx.numel() > self.train_max_queries:
                perm = torch.randperm(valid_idx.numel(), device=device)[: self.train_max_queries]
                valid_idx = valid_idx[perm]
            flat_q_list.append(voxel_grid.index_select(0, valid_idx))      # [n_b, 3]
            flat_t_list.append(targets_flat[b].index_select(0, valid_idx)) # [n_b]
            flat_b_list.append(
                torch.full((valid_idx.numel(),), b, dtype=torch.long, device=device)
            )

        flat_q = torch.cat(flat_q_list, dim=0)   # [N_total, 3]
        flat_t = torch.cat(flat_t_list, dim=0)   # [N_total]
        flat_b = torch.cat(flat_b_list, dim=0)   # [N_total]

        if flat_q.numel() == 0:
            zero = fused_bev.new_zeros(()).requires_grad_(True)
            return {"loss_occ_query": zero, "occ_query_count": zero.detach()}

        logits = self._decode(world, flat_q, flat_b)                  # [N_total, K]

        ce = F.cross_entropy(
            logits, flat_t,
            weight=self.class_weights,
            ignore_index=-1,
        )
        return {
            "loss_occ_query":   ce,
            "occ_query_count":  torch.tensor(flat_q.shape[0], device=device, dtype=torch.float32),
        }
