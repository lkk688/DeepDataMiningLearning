# projects/bevdet/cross_attn_lss.py
import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.dist import get_dist_info

# x: Tensor [B,Nc,C,Hf,Wf] OR list([B,Nc,C_l,H_l,W_l]) → bev: [B,out_channels,Hy,Hx]

@MODELS.register_module()
class CrossAttnLSSTransform(BaseModule):
    """
    Cross-Attention view transform (drop-in for DepthLSSTransform).

    Key ideas
    ---------
    • Avoid building a D×H×W depth volume. For each BEV cell center (x,y),
      sample K shallow depths → project into each camera → grid_sample features.
    • Fuse {camera × depth} samples with BEV-query × (image-token) dot-product
      attention. This gives per-cell, per-camera adaptivity with memory that
      scales roughly with the #queries (Hy×Hx), not with D×H×W.
    • Process BEV queries in CHUNKS to bound peak memory; 'attn_chunk' controls
      the trade-off between throughput and memory.

    Multiscale
    ----------
    • If x is a list [x_base, x_extra1, ...], each 5D, we 1×1-project extras to
      in_channels, resize to the base spatial size, and fuse with either:
        - "gated_add": fused = base + σ(α_l)*proj(resize(extra_l)) + ...
        - "sum":       fused = base + proj(resize(extra_l)) + ...
      This preserves the VT interface (single tensor after fusion).

    Outputs
    -------
    • BEV feature map [B, out_channels, Hy, Hx], where Hy/Hx follow x/y bounds
      using the *effective* step (dx*downsample, dy*downsample).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],        # raw image H,W (e.g., 256,704)
        feature_size: Tuple[int, int],      # base feature Hf,Wf (e.g., 32,88 for P3)
        xbound: Tuple[float, float, float], # [xmin, xmax, dx]
        ybound: Tuple[float, float, float], # [ymin, ymax, dy]
        zbound: Tuple[float, float, float], # [zmin, zmax, _] (only range matters here)
        dbound: Tuple[float, float, float], # [dmin, dmax, dstep] (used to derive K)
        downsample: int = 2,
        # Attention sampling hyper-params
        num_z: Optional[int] = None,        # if None, derive small K from dbound
        use_cam_embed: bool = True,         # add per-camera learned bias in token space
        attn_chunk: int = 4096,             # process BEV queries in chunks to cap memory
        # Multiscale options (optional)
        ms_extra_in_channels: Optional[List[int]] = None,  # e.g., [256] if using P4
        ms_fuse_mode: str = "gated_add",    # {"gated_add","sum"}
        ms_align_corners: bool = False,
        init_cfg: Optional[dict] = None,
        debug: bool  = False
    ):
        super().__init__(init_cfg)

        # --- core dims/params ---
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.image_size    = image_size
        self.feature_size  = feature_size
        self.downsample    = downsample
        self.debug         = bool(debug)
        self._debug_printed = False

        # BEV plane & ranges
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        # K shallow depths per BEV query (small by design)
        if num_z is not None:
            self.num_z = int(num_z)
        else:
            dmin, dmax, dstep = dbound
            derived = max(1, int((dmax - dmin) / max(dstep, 1.0)))
            self.num_z = min(4, derived)  # default tiny K (memory-friendly)

        self.use_cam_embed = use_cam_embed
        self.attn_chunk    = int(attn_chunk)

        # --- learnable query pos-embed (2D → C) ---
        self.pos_embed = nn.Sequential(
            nn.Linear(2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )

        # Optional per-camera embedding (NuScenes uses 6 cams)
        self.max_cams = 12
        if self.use_cam_embed:
            self.cam_embed = nn.Embedding(self.max_cams, in_channels)

        # Map fused BEV tokens → desired channel width
        self.out_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Scale for dot-product attention
        self.attn_scale = 1.0 / math.sqrt(in_channels)

        # Cache for BEV XY centers and shallow depth anchors (device-local)
        self.register_buffer("_bev_xy_cached", None, persistent=False)
        self.register_buffer("_z_anchors_cached", None, persistent=False)

        # --- Multiscale adapters ---
        self.ms_fuse_mode = ms_fuse_mode
        self.ms_align_corners = bool(ms_align_corners)
        self.expect_multiscale = ms_extra_in_channels is not None and len(ms_extra_in_channels) > 0

        if self.expect_multiscale:
            # 1×1 proj for each extra level → in_channels
            self.ms_proj = nn.ModuleList([
                nn.Conv2d(c_in, self.in_channels, kernel_size=1, bias=False)
                for c_in in ms_extra_in_channels
            ])
            # Learnable gate α for each extra level (if gated_add)
            self.ms_alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in ms_extra_in_channels])
        else:
            self.ms_proj  = nn.ModuleList()
            self.ms_alpha = nn.ParameterList()

    # ----- helpers: BEV grid & z anchors -----
    def _build_bev_grid(self, device):
        """Centers in metric space with effective step (dx*downsample, dy*downsample)."""
        xmin, xmax, dx = self.xbound
        ymin, ymax, dy = self.ybound
        dx_eff = dx * float(self.downsample)
        dy_eff = dy * float(self.downsample)
        xs = torch.arange(xmin + dx_eff / 2, xmax, dx_eff, device=device)
        ys = torch.arange(ymin + dy_eff / 2, ymax, dy_eff, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([xx, yy], dim=-1)  # [Hy, Hx, 2]

    def _build_z_anchors(self, device):
        """Small set of equally spaced shallow depths (z) in LiDAR frame."""
        zmin, zmax, _ = self.zbound
        K = self.num_z
        if K == 1:
            return torch.tensor([(zmin + zmax) * 0.5], device=device)
        return torch.linspace(zmin + (zmax - zmin)/(2*K),
                             zmax - (zmax - zmin)/(2*K),
                             steps=K, device=device)

    @staticmethod
    def _aug4x4_from_3x3(aug_3x3: torch.Tensor) -> torch.Tensor:
        """Pad a 3×3 image-aug matrix to homogeneous 4×4 for joint projection."""
        B, Nc, _, _ = aug_3x3.shape
        aug4 = torch.zeros((B, Nc, 4, 4), device=aug_3x3.device, dtype=aug_3x3.dtype)
        aug4[..., 3, 3] = 1.0
        aug4[..., :3, :3] = aug_3x3
        return aug4

    # ----- multiscale fuse in image-space -----
    def _fuse_multiscale(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: list of [B,Nc,C_l,H_l,W_l]; feats[0] is base and MUST have C_l==in_channels.
        Returns fused base-level tensor [B,Nc,in_channels,Hb,Wb].
        """
        assert isinstance(feats, (list, tuple)) and len(feats) >= 1
        base = feats[0]
        if not isinstance(base, torch.Tensor) or base.ndim != 5:
            raise ValueError("feats[0] must be 5D [B,Nc,C,H,W].")
        B, Nc, Cb, Hb, Wb = base.shape
        if Cb != self.in_channels:
            raise AssertionError(f"Base channels={Cb} but in_channels={self.in_channels}.")

        fused = base
        if len(feats) == 1:
            return fused

        if self.expect_multiscale and (len(feats) - 1) != len(self.ms_proj):
            raise ValueError("Mismatch: #extra feats vs ms_extra_in_channels in config.")

        proj_idx = 0
        for extra in feats[1:]:
            if extra is None:
                proj_idx += 1
                continue
            if extra.ndim != 5:
                raise ValueError(f"Extra level must be 5D; got {extra.shape}.")
            _, _, Cx, Hx, Wx = extra.shape

            if self.expect_multiscale:
                proj = self.ms_proj[proj_idx]
                gate = None if self.ms_fuse_mode == "sum" else torch.sigmoid(self.ms_alpha[proj_idx])
                proj_idx += 1
            else:
                if Cx != self.in_channels:
                    raise AssertionError("Extra channels must equal in_channels without ms_proj.")
                proj = nn.Identity()
                gate = None

            extra_bnc = extra.flatten(0, 1)                  # [B*Nc,Cx,Hx,Wx]
            extra_bnc = proj(extra_bnc)                      # [B*Nc,C,Hx,Wx]
            extra_up  = F.interpolate(
                extra_bnc, size=(Hb, Wb), mode='bilinear', align_corners=self.ms_align_corners
            ).view(B, Nc, self.in_channels, Hb, Wb)

            fused = fused + (extra_up if gate is None else gate * extra_up)

        return fused

    # ----- forward -----
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs):
        """
        Args
        ----
        x: 5D tensor [B,Nc,C,Hf,Wf] OR list of such tensors (multiscale).
        Kwargs (expected):
          • lidar2img:     [B,Nc,4,4]
          • img_aug_matrix [B,Nc,3,3] (optional)
        """
        rank, _ = get_dist_info()

        # 1) Normalize input: accept single-scale or multiscale, always produce
        #    a *single* fused base tensor [B,Nc,in_channels,Hb,Wb].
        if isinstance(x, (list, tuple)):
            x = self._fuse_multiscale(list(x))
        assert isinstance(x, torch.Tensor) and x.dim() == 5, f'Expect [B,Nc,C,Hf,Wf], got {type(x)} / {getattr(x,"shape",None)}'

        B, Nc, C, Hf, Wf = x.shape
        assert C == self.in_channels, f'in_channels mismatch: {C} vs {self.in_channels}'

        # Projection inputs
        lidar2img = kwargs.get('lidar2img', None)
        if lidar2img is None:
            for a in args:
                if isinstance(a, torch.Tensor) and a.dim() == 4 and a.shape[-2:] == (4, 4):
                    lidar2img = a
                    break
        assert lidar2img is not None, 'lidar2img is required (shape [B,Nc,4,4]).'
        img_aug = kwargs.get('img_aug_matrix', None)  # [B,Nc,3,3] or None

        device = x.device
        dtype_x = x.dtype

        # 2) Cache BEV grid (XY centers) and shallow z anchors on this device.
        if self._bev_xy_cached is None or self._bev_xy_cached.device != device:
            self._bev_xy_cached = self._build_bev_grid(device)  # [Hy,Hx,2]
        if self._z_anchors_cached is None or self._z_anchors_cached.device != device:
            self._z_anchors_cached = self._build_z_anchors(device)  # [K]

        Hy, Hx, _ = self._bev_xy_cached.shape
        Q = Hy * Hx
        K = int(self._z_anchors_cached.shape[0])

        # First-chunk debug
        if self.debug and (rank == 0) and not self._debug_printed:
            try:
                print("\n[CrossAttnLSSTransform] ===== DEBUG =====")
                print(f"x:              {tuple(x.shape)} dtype={x.dtype} device={x.device}")
                print(f"lidar2img:      {tuple(lidar2img.shape)}")
                print(f"img_aug_matrix: {None if img_aug is None else tuple(img_aug.shape)}")
                print(f"feature_size:   Hf={Hf}, Wf={Wf}, image_size={self.image_size}")
                print(f"BEV grid:       Hy={Hy}, Hx={Hx}, Q={Q}, K={K}, attn_chunk={self.attn_chunk}")
                xmin, xmax, dx = self.xbound; ymin, ymax, dy = self.ybound
                print(f"eff step: dx={dx*self.downsample}, dy={dy*self.downsample}")
                print("=========================================\n")
            finally:
                self._debug_printed = True

        # 3) Pre-compose projection matrix: (img_aug @ lidar2img) if aug is provided.
        M = lidar2img
        if img_aug is not None:
            M = self._aug4x4_from_3x3(img_aug) @ M
        proj_dtype = M.dtype

        # Output token buffer [B, Q, C]
        out_tokens = torch.empty((B, Q, C), device=device, dtype=dtype_x)

        # Map image pixels → feature coordinates
        Himg, Wimg = self.image_size
        sx = float(Wimg) / float(Wf)   # pixels per feature step (W)
        sy = float(Himg) / float(Hf)   # pixels per feature step (H)

        # 4) Process BEV queries in chunks to bound peak memory.
        for i in range(0, Q, self.attn_chunk):
            j = min(Q, i + self.attn_chunk)
            q_len = j - i

            # (a) Query pos-embeds for this chunk (normalize XY to [-1,1])
            bev_xy_chunk = self._bev_xy_cached.view(Q, 2)[i:j]          # [q_len,2]
            xmin, xmax, dx = self.xbound; ymin, ymax, dy = self.ybound
            nx = (bev_xy_chunk[..., 0] - xmin) / max(xmax - xmin, 1e-6) * 2 - 1
            ny = (bev_xy_chunk[..., 1] - ymin) / max(ymax - ymin, 1e-6) * 2 - 1
            bev_pos = torch.stack([nx, ny], dim=-1).view(1, q_len, 2).expand(B, -1, -1)  # [B,q_len,2]
            q_embed = self.pos_embed(bev_pos)                                            # [B,q_len,C]

            # (b) Build shallow 3D anchors (q_len×K) in LiDAR frame, homogeneous
            xy = bev_xy_chunk.view(q_len, 1, 2).expand(-1, K, -1)
            z  = self._z_anchors_cached.view(1, K, 1).expand(q_len, -1, 1)
            pts3d = torch.cat([xy, z], dim=-1)                                           # [q_len,K,3]
            ones = torch.ones((q_len * K, 1), device=device, dtype=proj_dtype)
            pts_homo = torch.cat([pts3d.view(-1, 3).to(proj_dtype), ones], dim=-1)       # [qK,4]
            pts_homo_T = pts_homo.t().contiguous().view(1, 1, 4, q_len * K)              # [1,1,4,qK]

            # (c) Project anchors to each camera
            proj = torch.matmul(M, pts_homo_T)                 # [B,Nc,4,qK]
            proj_z = proj[..., 2, :]                           # [B,Nc,qK]
            valid = (proj_z > 0)                               # front of camera
            uv_img = proj[..., :2, :] / proj_z.unsqueeze(-2).clamp(min=1e-6)  # [B,Nc,2,qK]

            # (d) Convert to feature coords (normalize to [-1,1] for grid_sample)
            uv_img = uv_img.permute(0, 1, 3, 2).contiguous()   # [B,Nc,qK,2]
            u_feat = (uv_img[..., 0] / sx); v_feat = (uv_img[..., 1] / sy)
            u_norm = (u_feat / max(Wf - 1, 1)) * 2 - 1
            v_norm = (v_feat / max(Hf - 1, 1)) * 2 - 1
            grid = torch.stack([u_norm, v_norm], dim=-1)       # [B,Nc,qK,2]

            # (e) Sample image features at those coords
            grid = grid.reshape(B * Nc, q_len * K, 1, 2).to(dtype_x).contiguous()
            x_   = x.contiguous().view(B * Nc, C, Hf, Wf)
            sampled = F.grid_sample(x_, grid, mode='bilinear',
                                    align_corners=False, padding_mode='zeros')  # [B*Nc,C,qK,1]
            sampled = sampled.view(B, Nc, C, q_len * K, 1).squeeze(-1)          # [B,Nc,C,qK]
            sampled = sampled.view(B, Nc, C, q_len, K).permute(0, 3, 1, 4, 2).contiguous()
            # -> [B, q_len, Nc, K, C]
            valid_mask = valid.view(B, Nc, q_len, K).permute(0, 2, 1, 3).contiguous()

            # (f) Optional camera embedding (adds a learned bias per camera)
            if self.use_cam_embed:
                cam_ids = torch.arange(Nc, device=device).view(1, Nc) % self.max_cams
                cam_embed = self.cam_embed(cam_ids).view(1, 1, Nc, 1, C).to(sampled.dtype)
                sampled = sampled + cam_embed

            # (g) Cross-attention: scores = <q, v>, masked softmax over Nc×K, then sum
            q_ = q_embed.unsqueeze(2).unsqueeze(3)                 # [B,q_len,1,1,C]
            scores = (q_ * sampled).sum(dim=-1) * self.attn_scale  # [B,q_len,Nc,K]
            scores = scores.masked_fill(~valid_mask, -1e9)
            attn = torch.softmax(scores.view(B, q_len, Nc * K), dim=-1).view(B, q_len, Nc, K, 1)
            fused = (attn * sampled).sum(dim=(2, 3))               # [B,q_len,C]
            none_valid = (~valid_mask).all(dim=(2, 3))             # [B,q_len]
            if none_valid.any():
                fused[none_valid] = 0.0

            out_tokens[:, i:j, :] = fused

        # 5) Tokens → BEV map → 1×1 to out_channels
        bev = out_tokens.transpose(1, 2).contiguous().view(B, C, Hy, Hx)  # [B,C,Hy,Hx]
        bev = self.out_proj(bev)                                          # [B,outC,Hy,Hx]
        return bev