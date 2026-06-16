"""
Multi-token temporal LiDAR (Stage 1 of the temporal redesign).

Motivation
----------
B7-B (dual-pass: now/past split) gave NDS 0.6647 vs B5's 0.6848 — −0.02
even after we fixed the V1 bugs (lazy temporal_proj / missing aux head /
BN running-stat corruption). Conclusion: a single coarse past pass over
a frozen 1×1 conv is not enough temporal signal to overcome the noise
introduced by the dual encoding.

This module replaces that with the three components the user asked for:

  1. **Multi-token temporal axis (K=4).** Split the existing aggregated
     multi-sweep cloud (~500 ms) into K time buckets so the model sees
     a discrete time axis rather than a binary now/past split.
  2. **Temporal positional encoding.** A learnable per-bucket embedding
     so attention can tell tokens apart in time.
  3. **Cross-frame attention.** Per-BEV-cell attention: query = current
     bucket, K/V = all K buckets at the same (h,w). Light-weight
     (sequence length = K = 4) and directly analogous to BEVFormer's
     temporal self-attention without paying for deformable sampling.

Stage 1 scope (this file)
-------------------------
- LiDAR side only.
- Uses the **existing** LoadPointsFromMultiSweeps cloud — no new data
  loader work. Multi-sweep points are already per-point ego-compensated
  to the current frame, so no extra warp is needed in Stage 1.
- Output channel count matches the original LiDAR-BEV channel, so the
  rest of the pipeline (ConvFuser → pts_backbone → pts_neck → head) is
  untouched.

Stage 2 (separate file) will add past-keyframe loading with an explicit
ego-warp on the BEV grid, extending the temporal window from ~500 ms to
~1.5 s. Stage 3 (deferred) extends temporal to the camera side.

Implementation guardrails (carried over from B7 V2's bug post-mortem)
---------------------------------------------------------------------
- The "main" LiDAR branch encodes the **full multi-sweep cloud** —
  exactly what the warm-start checkpoint was trained on. K-1 additional
  past-time-slice encodings are computed as attention "probes" only.
  Crucially, at α=0 (init) the LiDAR-branch output equals the warm-start
  BEV bit-for-bit. (B8 V1 made the mistake of using a 0-125ms slice as
  the main path; the head saw only ~2 sweeps worth of points at α=0 and
  could not recover in 3 epochs → NDS 0.6382, ↓0.05 vs warm-start.)
- The probe passes (k > 0) run with the encoder modules in ``eval()``
  mode + ``torch.no_grad()`` so BN running stats do NOT update from
  past-only point distributions (BN-corruption was the dominant cause
  of the −0.14 NDS regression in temporal_lidar V1).
- All new submodules (temporal PE, cross-frame attention, residual
  scaler) are built **eagerly in __init__** so they're visible to the
  optimizer that is constructed after the model.
- Output projection is **zero-initialized** so the temporal residual is
  literally zero at iteration 0.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .bevfusion_ca import BEVFusionCA


# ---------------------------------------------------------------------------
# Time-bucketing helper — past-only slices
# ---------------------------------------------------------------------------

def _split_points_into_past_slices(
    points: List[torch.Tensor],
    num_past: int,
    window_seconds: float,
    time_dim: int = 4,
) -> List[List[torch.Tensor]]:
    """
    Split each per-sample multi-sweep cloud into ``num_past`` past-only
    time slices that cover the back portion of the multi-sweep window.

    The full multi-sweep cloud spans |t| ∈ [0, window_seconds]. We treat
    the first 1/(num_past+1) of the window as "current" (token 0 in the
    detector is the full cloud, NOT a slice — see the detector code) and
    split the remaining window into ``num_past`` equal slices:

        slice k=0:  |t| ∈ [(1/(N+1)) * τ, (2/(N+1)) * τ)
        slice k=1:  |t| ∈ [(2/(N+1)) * τ, (3/(N+1)) * τ)
        ...
        slice k=N-1:|t| ∈ [(N/(N+1)) * τ, τ]              (closed)

    where N = num_past, τ = window_seconds. For nuScenes (τ=0.5s,
    N=3) the slices are roughly [125-250ms, 250-375ms, 375-500ms].

    nuScenes LoadPointsFromMultiSweeps stores per-point time in column 4
    as a signed offset; |time| is the point's age in seconds.

    Returns a list of length ``num_past``; ``slices[k]`` is a list (one
    tensor per sample). Samples with an empty slice fall back to a 50-
    point random subsample of their full cloud so the encoder sees a
    non-empty input (its BEV will not match the slice's time but the
    attention block can learn to weight that token down).
    """
    assert num_past >= 1
    N = int(num_past)
    tau = float(window_seconds)
    slices: List[List[torch.Tensor]] = [[] for _ in range(N)]

    cut_lo = [(k + 1) / (N + 1) * tau for k in range(N)]
    cut_hi = [(k + 2) / (N + 1) * tau for k in range(N)]

    for p in points:
        if p.numel() == 0 or p.shape[1] <= time_dim:
            for k in range(N):
                slices[k].append(p)
            continue

        t = p[:, time_dim].abs()
        slice_pts = []
        for k in range(N):
            lo, hi = cut_lo[k], cut_hi[k]
            if k == N - 1:
                mask = (t >= lo) & (t <= hi + 1e-6)
            else:
                mask = (t >= lo) & (t < hi)
            slice_pts.append(p[mask])

        # Empty-slice fallback: use a small random subsample of the full
        # cloud so spconv has something to digest. The resulting BEV will
        # be approximately a scaled copy of the main BEV — the attention
        # block can learn to ignore it.
        fallback = None
        for k in range(N):
            if slice_pts[k].numel() == 0:
                if fallback is None:
                    n = min(p.shape[0], 256)
                    idx = torch.randperm(p.shape[0], device=p.device)[:n]
                    fallback = p[idx]
                slices[k].append(fallback)
            else:
                slices[k].append(slice_pts[k])

    return slices


# ---------------------------------------------------------------------------
# Per-BEV-cell temporal cross-frame attention
# ---------------------------------------------------------------------------

class TemporalCrossFrameAttention(nn.Module):
    """
    Per-position multi-head attention across K temporal tokens.

    For each BEV cell (b, h, w):
      Q = bev_0[b, :, h, w]                  (current bucket only)
      K, V = stack(bev_0..bev_{K-1})[b, :, :, h, w]   (all K buckets)
    Output of shape [B, C, H, W], added residually to bev_0.

    Cheap by construction: the attention sequence length is K (~4), not
    H*W. Total memory is O(B*H*W*K*C), the same order as feeding a tiny
    sequence to nn.MultiheadAttention.

    Output projection is **zero-initialized** so on the first forward
    the residual is exactly bev_0 — important when warm-starting from a
    non-temporal checkpoint.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels={channels} must be divisible by num_heads={num_heads}"
        )
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.out_proj.weight)  # identity-residual init

    def forward(self, bev_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_stack: [B, K, C, H, W]  — temporal stack with PE already
                added. bev_stack[:, 0] is the current bucket.
        Returns:
            delta: [B, C, H, W]  — residual to add to bev_0.
        """
        B, K, C, H, W = bev_stack.shape
        assert C == self.channels

        # Reshape so each BEV cell is an independent K-token sequence.
        # [B, K, C, H, W] → [B, H, W, K, C] → [B*H*W, K, C]
        x = bev_stack.permute(0, 3, 4, 1, 2).reshape(B * H * W, K, C)

        # Q from current bucket only (index 0); K, V from all K.
        q = self.q_proj(x[:, 0:1])         # [N, 1, C]
        k = self.k_proj(x)                  # [N, K, C]
        v = self.v_proj(x)                  # [N, K, C]

        # Multi-head split: [N, T, C] → [N, num_heads, T, head_dim]
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            N, T, _ = t.shape
            return t.view(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = split_heads(q)                  # [N, h, 1, d]
        k = split_heads(k)                  # [N, h, K, d]
        v = split_heads(v)                  # [N, h, K, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale       # [N, h, 1, K]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                       # [N, h, 1, d]

        # Merge heads: [N, h, 1, d] → [N, 1, C]
        out = out.permute(0, 2, 1, 3).contiguous().view(B * H * W, 1, C)
        out = self.out_proj(out)                            # zero-init → 0

        # [B*H*W, 1, C] → [B, C, H, W]
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@MODELS.register_module()
class MultiTokenTemporalBEVFusion(BEVFusionCA):
    """
    BEVFusionCA + K-token temporal LiDAR attention (Stage 1).

    Args:
        num_buckets: K, number of temporal buckets across the multi-sweep
            window. K=4 covers ~125 ms each at the default 500 ms window.
        window_seconds: total |time| span the multi-sweep loader covers.
            nuScenes LoadPointsFromMultiSweeps(sweeps_num=10) ≈ 0.5 s.
        num_attn_heads: heads inside the cross-frame attention block.
        attn_dropout: attention dropout. Default 0.0 (clean comparison).
        **bevfusion_ca_kwargs: forwarded to BEVFusionCA.
    """

    def __init__(
        self,
        num_buckets: int = 4,
        window_seconds: float = 0.5,
        num_attn_heads: int = 4,
        attn_dropout: float = 0.0,
        **bevfusion_ca_kwargs,
    ):
        super().__init__(**bevfusion_ca_kwargs)
        assert num_buckets >= 2, "num_buckets must be >= 2"
        self.num_buckets = int(num_buckets)
        self.window_seconds = float(window_seconds)

        # Resolve the LiDAR-BEV channel count. ConvFuser is configured
        # with in_channels=[camera_ch, lidar_ch].
        fl_in = getattr(self.fusion_layer, "in_channels", None)
        if isinstance(fl_in, (list, tuple)) and len(fl_in) >= 2:
            lidar_C = int(fl_in[-1])
        else:
            lidar_C = 256  # fallback
        self.lidar_channels = lidar_C

        # Learnable per-bucket temporal positional encoding [K, C].
        # Small init so the encoder's BEV magnitudes dominate at start.
        self.temporal_pe = nn.Parameter(0.02 * torch.randn(self.num_buckets, lidar_C))

        # Cross-frame attention block. ``out_proj.weight = 0`` makes
        # ``delta = 0`` at iter 0 → fused = bev_main (warm-start preserved).
        # NOTE (B10c bug fix): we previously ALSO multiplied delta by a
        # learnable ``temporal_alpha`` init at 0. Combined with out_proj=0
        # this was a dead-pathway double-zero trap (alpha and out_proj
        # both received zero gradient and never moved). Dropped here so
        # gradient flows normally through out_proj.
        self.temporal_attn = TemporalCrossFrameAttention(
            channels=lidar_C,
            num_heads=num_attn_heads,
            dropout=attn_dropout,
        )

        print(
            f"[MultiTokenTemporalBEVFusion] K={self.num_buckets} buckets, "
            f"window={self.window_seconds:.3f}s, channels={lidar_C}, "
            f"heads={num_attn_heads}, residual_init=0 (out_proj only; "
            f"temporal_alpha dropped — see B10c)"
        )

    # ------------------------------------------------------------------
    # extract_feat override: multi-pass LiDAR encoder + temporal attn.
    # ------------------------------------------------------------------

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        imgs = batch_inputs_dict.get("imgs", None)
        points = batch_inputs_dict.get("points", None)
        features: List[torch.Tensor] = []

        # ---- IMAGE BRANCH (identical to parent) ----
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta["lidar2img"])
                camera_intrinsics.append(meta["cam2img"])
                camera2lidar.append(meta["cam2lidar"])
                img_aug_matrix.append(meta.get("img_aug_matrix", np.eye(4)))
                lidar_aug_matrix.append(meta.get("lidar_aug_matrix", np.eye(4)))
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
            features.append(img_feature)

        # ---- LIDAR BRANCH (K-token temporal) ----
        pts_feature = self._extract_temporal_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        # ---- LATE FUSION + post-fusion backbone ----
        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    # ------------------------------------------------------------------
    # K-bucket LiDAR feature extraction
    # ------------------------------------------------------------------

    def _extract_temporal_pts_feat(self, batch_inputs_dict: Dict) -> torch.Tensor:
        """
        Token layout (K = self.num_buckets):
            token 0       = encode(FULL multi-sweep cloud)   [main path, with grad]
            token 1..K-1  = encode(past slice k-1)           [probe, no_grad+eval-BN]

        At iteration 0 (α=0, out_proj=0) the LiDAR-branch output is
        exactly token 0, identical to what BEVFusionCA / B5 produces.
        This is the warm-start invariant that B8 V1 broke.
        """
        points_full = batch_inputs_dict["points"]

        # ---- Token 0: full-cloud BEV (warm-start-equivalent main path) ----
        bev_main = self.extract_pts_feat(batch_inputs_dict)

        # ---- Tokens 1..K-1: past-slice probes (frozen-BN, no_grad) ----
        past_slices = _split_points_into_past_slices(
            points_full,
            num_past=self.num_buckets - 1,
            window_seconds=self.window_seconds,
        )

        enc_modules: List[nn.Module] = []
        for attr in ("pts_voxel_encoder", "pts_middle_encoder"):
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
                        {**batch_inputs_dict, "points": past_slices[k]}
                    )
                    bev_probes.append(bev_k.detach())
        finally:
            for m, was_train in zip(enc_modules, saved_training):
                m.train(was_train)

        # Stack tokens: [B, K, C, H, W]. Same (C, H, W) across tokens
        # because the same voxelizer + middle_encoder produced them.
        bev_stack = torch.stack([bev_main] + bev_probes, dim=1)

        # Add temporal positional encoding (broadcast over B, H, W).
        # [K, C] → [1, K, C, 1, 1]
        pe = self.temporal_pe.view(1, self.num_buckets, self.lidar_channels, 1, 1)
        bev_stack = bev_stack + pe.to(bev_stack.dtype)

        # Cross-frame attention → residual on top of bev_main. delta
        # starts at zero (out_proj.weight = 0) so fused = bev_main at
        # iter 0 (warm-start preserved). After backward, out_proj.weight
        # gets a real gradient signal from the bbox-head loss.
        delta = self.temporal_attn(bev_stack)             # [B, C, H, W]
        return bev_main + delta.to(bev_main.dtype)
