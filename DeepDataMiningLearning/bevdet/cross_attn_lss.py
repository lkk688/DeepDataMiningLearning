# projects/cross_attn_lss.py
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.dist import get_dist_info
#x:[B,Nc,C,Hf,Wf] → bev:[B,out_channels,Hy,Hx]

@MODELS.register_module()
class CrossAttnLSSTransform(BaseModule):
    """
    A drop-in replacement of DepthLSSTransform using sparse cross-attention.
    Input / Output keep the same as LSS (single-scale feature):
        - x: Tensor [B, Nc, C, Hf, Wf]
        - returns: BEV features [B, out_channels, Hy, Hx]

    Required runtime kwargs (kept with LSS usage):
        - lidar2img: [B, Nc, 4, 4]
        - (optional) img_aug_matrix: [B, Nc, 3, 3]  # resize/crop aug

    Key ideas:
      * For each BEV cell center (x, y), sample K height anchors (z) → 3D points
      * Project to each camera with (aug @ lidar2img), sample image features by grid_sample
      * Do lightweight attention across {camera × z} samples to fuse as the BEV token
      * Map to out_channels via 1×1 conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 2,
        # cross-attn hyper-params
        num_z: Optional[int] = None,      # if None, derive from dbound; else override
        use_cam_embed: bool = True,
        attn_chunk: int = 4096,           # chunk BEV queries to save memory
        init_cfg: Optional[dict] = None,
        debug: bool  = False
    ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size      # [Himg, Wimg]
        self.feature_size = feature_size  # [Hf, Wf] (from neck)
        self.downsample = downsample
        self.debug = debug
        self._debug_printed = debug   # first-time header
        self._debug_chunk_printed = debug    # first chunk prints

        # BEV grid spec
        self.xbound = xbound  # [xmin, xmax, dx]
        self.ybound = ybound
        self.zbound = zbound  # [zmin, zmax, #bins or step? here仅用于中心高度范围]
        self.dbound = dbound  # [dmin, dmax, dstep] in original LSS; here as reference

        # number of height anchors for sampling per BEV cell
        if num_z is not None:
            self.num_z = int(num_z)
        else:
            # fall back: use small K to save memory; or derive from dbound step
            dmin, dmax, dstep = dbound
            derived = max(1, int((dmax - dmin) / max(dstep, 1.0)))
            self.num_z = min(4, derived)  # default 4 anchors (memory-friendly)

        self.use_cam_embed = use_cam_embed
        self.attn_chunk = attn_chunk

        # simple BEV 2D positional encoding → C
        self.pos_embed = nn.Sequential(
            nn.Linear(2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )

        # optional camera embedding (max 12 cams; sufficient for nuScenes)
        self.max_cams = 12
        if use_cam_embed:
            self.cam_embed = nn.Embedding(self.max_cams, in_channels)

        # final projection to out_channels
        self.out_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # scale for dot-product attention
        self.attn_scale = 1.0 / math.sqrt(in_channels)

        # buffers for cached BEV grid & z-anchors (built lazily on device)
        self.register_buffer("_bev_xy_cached", None, persistent=False)
        self.register_buffer("_z_anchors_cached", None, persistent=False)

    # ---------- helpers ----------
    def _build_bev_grid(self, device):
        """
        Build BEV grid centers with *effective* step = (dx * downsample, dy * downsample),
        so the output BEV map spatial size matches the LSS behavior when `downsample > 1`.
        This makes our CrossAttn output (Hy_out, Hx_out) consistent with the LiDAR branch.
        """
        xmin, xmax, dx = self.xbound
        ymin, ymax, dy = self.ybound

        # Effective step after downsample (e.g., dx=0.3, downsample=2 -> 0.6)
        dx_eff = dx * float(self.downsample)
        dy_eff = dy * float(self.downsample)

        # Create evenly-spaced grid centers. We use "center of cell" convention:
        # start = min + step/2, then step by step until < max.
        xs = torch.arange(xmin + dx_eff / 2, xmax, dx_eff, device=device)
        ys = torch.arange(ymin + dy_eff / 2, ymax, dy_eff, device=device)

        # Mesh order: (Hy, Hx) => y first then x (typical in BEV implementations)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [Hy, Hx]
        bev_xy = torch.stack([xx, yy], dim=-1)          # [Hy, Hx, 2]
        return bev_xy

    def _build_z_anchors(self, device):
        zmin, zmax, _ = self.zbound
        K = self.num_z
        if K == 1:
            zs = torch.tensor([(zmin + zmax) * 0.5], device=device)
        else:
            zs = torch.linspace(zmin + (zmax - zmin) / (2 * K),
                                zmax - (zmax - zmin) / (2 * K),
                                steps=K, device=device)
        return zs  # [K]

    @staticmethod
    def _aug4x4_from_3x3(aug_3x3: torch.Tensor) -> torch.Tensor:
        """Make a 4x4 from 3x3 image aug matrix (pad bottom-right with 1)."""
        B, Nc, _, _ = aug_3x3.shape
        aug4 = torch.zeros((B, Nc, 4, 4), device=aug_3x3.device, dtype=aug_3x3.dtype)
        aug4[..., :3, :3] = aug_3x3
        aug4[..., 3, 3] = 1.0
        return aug4

    # ---------- main forward ----------
    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [B, Nc, C, Hf, Wf]  (single-scale feature after neck)
        Kwargs (expected):
            lidar2img: [B, Nc, 4, 4]
            img_aug_matrix (optional): [B, Nc, 3, 3]
        """
        rank, _ = get_dist_info()
        # -------------------- sanity checks & basic shapes --------------------
        assert x.dim() == 5, f'Expect x as [B, Nc, C, Hf, Wf], got {tuple(x.shape)}'
        B, Nc, C, Hf, Wf = x.shape
        assert C == self.in_channels, f'in_channels mismatch: {C} vs {self.in_channels}'

        lidar2img = kwargs.get('lidar2img', None)
        if lidar2img is None:
            # Robustness: also search positional args for a [B, Nc, 4, 4] tensor.
            for a in args:
                if isinstance(a, torch.Tensor) and a.dim() == 4 and a.shape[-2:] == (4, 4):
                    lidar2img = a
                    break
        assert lidar2img is not None, 'lidar2img is required (shape [B, Nc, 4, 4])'

        img_aug = kwargs.get('img_aug_matrix', None)  # [B, Nc, 3, 3] if provided

        device = x.device
        dtype_x = x.dtype

        # Lazily build & cache BEV grid XY and Z anchors on the right device.
        if self._bev_xy_cached is None or self._bev_xy_cached.device != device:
            self._bev_xy_cached = self._build_bev_grid(device)  # [Hy, Hx, 2]
        if self._z_anchors_cached is None or self._z_anchors_cached.device != device:
            self._z_anchors_cached = self._build_z_anchors(device)  # [K]

        Hy, Hx, _ = self._bev_xy_cached.shape
        Q = Hy * Hx                          # number of BEV queries
        K = int(self._z_anchors_cached.shape[0])  # height anchors per BEV

        # First-time debug header.
        if self.debug and (rank == 0) and not hasattr(self, "_debug_printed") or not self._debug_printed:
            try:
                print("\n[CrossAttnLSSTransform] ===== DEBUG SHAPES =====")
                print(f"x:              {tuple(x.shape)}  dtype={x.dtype} device={x.device}")
                print(f"lidar2img:      {tuple(lidar2img.shape)}")
                print(f"img_aug_matrix: {None if img_aug is None else tuple(img_aug.shape)}")
                print(f"feature_size:   Hf={Hf}, Wf={Wf}")
                print(f"image_size:     Himg={self.image_size[0]}, Wimg={self.image_size[1]}")
                print(f"BEV grid:       Hy={Hy}, Hx={Hx}, Q=Hy*Hx={Q}")
                print(f"num_z (K):      {K}")
                print(f"attn_chunk:     {self.attn_chunk}")
                xmin, xmax, dx = self.xbound
                ymin, ymax, dy = self.ybound
                print(f"effective step:  dx_eff={dx * self.downsample}, dy_eff={dy * self.downsample}")
                print(f"BEV out size:   Hy={Hy}, Hx={Hx}  (should match LiDAR BEV, e.g., 180x180)")
                print("================================================\n")
            finally:
                self._debug_printed = True

        # Precompose projection matrix: (img_aug @ lidar2img) if aug is provided.
        # Result M: [B, Nc, 4, 4] in the same dtype as lidar2img.
        M = lidar2img
        if img_aug is not None:
            M = self._aug4x4_from_3x3(img_aug) @ M
        # We'll do matmul in M.dtype for numerical stability.
        proj_dtype = M.dtype

        # Prepare the output token buffer: [B, Q, C]
        out_tokens = torch.empty((B, Q, C), device=device, dtype=dtype_x)

        # Convenience: downscale factors to map image pixels -> feature map coordinates.
        Himg, Wimg = self.image_size
        sx = float(Wimg) / float(Wf)   # width downscale (pixels per feature step)
        sy = float(Himg) / float(Hf)   # height downscale

        # ------------- process queries in chunks to limit peak memory -------------
        # We also build position embeddings per-chunk (no need to keep [B, Q, C] alive).
        for i in range(0, Q, self.attn_chunk):
            j = min(Q, i + self.attn_chunk)
            q_len = j - i  # number of queries in this chunk

            # ---- (1) Build BEV positions for this chunk & its query embeddings ----
            # Select BEV centers XY for indices [i:j].
            bev_xy_chunk = self._bev_xy_cached.view(Q, 2)[i:j]     # [q_len, 2]
            # Normalize XY to [-1, 1] to feed the MLP position encoder.
            xmin, xmax, dx = self.xbound
            ymin, ymax, dy = self.ybound
            norm_x = (bev_xy_chunk[..., 0] - xmin) / max(xmax - xmin, 1e-6) * 2 - 1
            norm_y = (bev_xy_chunk[..., 1] - ymin) / max(ymax - ymin, 1e-6) * 2 - 1
            bev_pos = torch.stack([norm_x, norm_y], dim=-1)        # [q_len, 2]
            bev_pos = bev_pos.view(1, q_len, 2).expand(B, -1, -1)  # [B, q_len, 2]
            q_embed = self.pos_embed(bev_pos)                      # [B, q_len, C]

            # ---- (2) Build 3D anchor points for this chunk (q_len × K) ----
            # pts3d: [q_len, K, 3] in LiDAR frame.
            xy = bev_xy_chunk.view(q_len, 1, 2).expand(-1, K, -1)       # [q_len, K, 2]
            z = self._z_anchors_cached.view(1, K, 1).expand(q_len, -1, 1)  # [q_len, K, 1]
            pts3d = torch.cat([xy, z], dim=-1)                           # [q_len, K, 3]

            # Homogeneous coords: [q_len*K, 4], then transpose to [4, q_len*K].
            ones = torch.ones((q_len * K, 1), device=device, dtype=proj_dtype)
            pts_homo = torch.cat([pts3d.view(-1, 3).to(proj_dtype), ones], dim=-1)  # [qK, 4]
            pts_homo_T = pts_homo.t().contiguous().view(1, 1, 4, q_len * K)         # [1,1,4,qK]

            # ---- (3) Project to image plane using batched matmul ----
            # proj = M @ pts_homo_T => [B, Nc, 4, q_len*K]
            proj = torch.matmul(M, pts_homo_T)
            proj_z = proj[..., 2, :]                                   # [B, Nc, qK]
            # Valid if point is in front of the camera.
            valid = (proj_z > 0)

            # Pixel coordinates (u_img, v_img) in original image scale.
            uv_img = proj[..., :2, :] / proj_z.unsqueeze(-2).clamp(min=1e-6)  # [B, Nc, 2, qK]

            # ---- (4) Convert to feature map coordinates & normalize to [-1, 1] ----
            # First move the "2" to the last dim -> [B, Nc, qK, 2]
            uv_img = uv_img.permute(0, 1, 3, 2).contiguous()
            # Map from pixels to feature coordinates by (u_feat = u_img / sx, v_feat = v_img / sy)
            u_feat = (uv_img[..., 0] / sx)  # [B, Nc, qK]
            v_feat = (uv_img[..., 1] / sy)  # [B, Nc, qK]
            # Normalize for grid_sample on feature map (Hf, Wf)
            u_norm = (u_feat / max(Wf - 1, 1)) * 2 - 1
            v_norm = (v_feat / max(Hf - 1, 1)) * 2 - 1
            uv_norm = torch.stack([u_norm, v_norm], dim=-1)             # [B, Nc, qK, 2]

            # ---- (5) Build grid & sample features in chunks’ size ----
            # grid: [B*Nc, qK, 1, 2], x_: [B*Nc, C, Hf, Wf]
            grid = uv_norm.reshape(B * Nc, q_len * K, 1, 2).to(dtype_x).contiguous()
            x_ = x.contiguous().view(B * Nc, C, Hf, Wf)

            sampled = F.grid_sample(
                x_, grid, mode='bilinear', align_corners=False, padding_mode='zeros'
            )  # [B*Nc, C, qK, 1]

            # Reshape sampled to [B, q_len, Nc, K, C]
            sampled = sampled.view(B, Nc, C, q_len * K, 1).squeeze(-1)      # [B, Nc, C, qK]
            sampled = sampled.view(B, Nc, C, q_len, K).permute(0, 3, 1, 4, 2).contiguous()
            # -> [B, q_len, Nc, K, C]

            # Valid mask aligned as [B, q_len, Nc, K]
            valid_mask = valid.view(B, Nc, q_len, K).permute(0, 2, 1, 3).contiguous()

            # Optional per-camera embedding
            # Optional per-camera embedding (broadcast-add on Nc dimension)
            if self.use_cam_embed:
                # cam_ids: [1, Nc] with values 0..Nc-1, wrapped by max_cams
                cam_ids = torch.arange(Nc, device=device).view(1, Nc) % self.max_cams
                # (1, Nc, C) -> reshape to (1, 1, Nc, 1, C) so that it aligns with sampled[B, q_len, Nc, K, C]
                cam_embed = self.cam_embed(cam_ids).view(1, 1, Nc, 1, C).to(sampled.dtype)
                # sanity checks (won't run in production if you like, just for first chunk)
                if i == 0 and self.debug:
                    print("[CrossAttnLSSTransform][CHUNK-0] cam_embed:", tuple(cam_embed.shape)) #(1,1,6,1,256)
                    print("[CrossAttnLSSTransform][CHUNK-0] sampled  :", tuple(sampled.shape)) #(4,2048,6,2,256)
                    # sampled dims:   [B, q_len, Nc, K, C]
                    # cam_embed dims: [1, 1,     Nc, 1, C]
                    assert sampled.shape[2] == Nc and sampled.shape[-1] == cam_embed.shape[-1]
                sampled = sampled + cam_embed  # broadcast on [B, q_len, K]

            # ---- (6) Debug prints for the FIRST chunk only ----
            if self.debug and (rank == 0) and (i == 0):
                print("[CrossAttnLSSTransform][CHUNK-0] shapes:")
                print(f"  bev_xy_chunk: {tuple(bev_xy_chunk.shape)}")        # [q_len, 2]
                print(f"  pts3d:        {tuple(pts3d.shape)}")               # [q_len, K, 3]
                print(f"  pts_homo_T:   {tuple(pts_homo_T.shape)}")          # [1,1,4,qK]
                print(f"  proj:         {tuple(proj.shape)}")                # [B, Nc, 4, qK]
                print(f"  uv_norm:      {tuple(uv_norm.shape)}")             # [B, Nc, qK, 2]
                print(f"  grid:         {tuple(grid.shape)}")                # [B*Nc, qK, 1, 2]
                print(f"  sampled:      {tuple(sampled.shape)}")             # [B, q_len, Nc, K, C]")
                print(f"  valid_mask:   {tuple(valid_mask.shape)}")          # [B, q_len, Nc, K]")

            # ---- (7) Cross-attention over {Nc × K} for this chunk ----
            # q_embed: [B, q_len, C]; sampled: [B, q_len, Nc, K, C]
            q_slice = q_embed
            v_slice = sampled
            m_slice = valid_mask

            # Dot-product attention with scaling. Shape: [B, q_len, Nc, K]
            q_ = q_slice.unsqueeze(2).unsqueeze(3)            # [B, q_len, 1, 1, C]
            scores = (q_ * v_slice).sum(dim=-1) * self.attn_scale

            # Mask invalid samples; use a large negative value instead of -inf
            # so that softmax doesn't produce NaNs when all are invalid.
            scores = scores.masked_fill(~m_slice, -1e9)
            attn = torch.softmax(scores.view(B, q_len, Nc * K), dim=-1).view(B, q_len, Nc, K, 1)

            fused = (attn * v_slice).sum(dim=(2, 3))          # [B, q_len, C]

            # If a position has zero valid samples across all {Nc,K}, force zeros.
            none_valid = (~m_slice).all(dim=(2, 3))           # [B, q_len]
            if none_valid.any():
                fused[none_valid] = 0.0

            # Write back into the output buffer
            out_tokens[:, i:j, :] = fused

        # ---- (8) Tokens -> BEV map, then project to out_channels ----
        bev = out_tokens.transpose(1, 2).contiguous().view(B, C, Hy, Hx)  # [B, C, Hy, Hx]
        bev = self.out_proj(bev)  # [B, out_channels, Hy, Hx]
        return bev