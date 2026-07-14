"""
ngperception.occupancy.models.lss_occ
=====================================

A **depth-supervised Lift-Splat-Shoot (LSS) occupancy** network — pure PyTorch, no mmcv,
no custom CUDA. This is the controllable, in-house alternative to the mmdet3d SOTA nets
(§2.3) and the self-supervised GaussianOcc (§2.5). The design follows BEVDepth's key
insight: **supervise the lift's depth distribution with LiDAR**, which is what lifts
camera occupancy from ~24 to ~40 mIoU.

Pipeline (per surround frame of N=6 cameras):

    image ──ResNet──► feat (C+D channels) ──split──► context C  +  depth-dist D
                                                          │             │
       Lift:  outer product  feat ⊗ depth  → a frustum of weighted features
       Splat: scatter frustum points into the (X,Y,Z) voxel grid (pure-torch)
       Decode: 3D CNN → per-voxel occupancy logits (18 classes)

Losses (built in train_lss.py): occupancy CE vs Occ3D GT + a depth CE on the predicted
depth distribution vs LiDAR-projected GT (the "depth-supervised" part).

Grid (Occ3D-nuScenes): x,y ∈ [-40,40], z ∈ [-1,5.4], 0.4 m → (200,200,16).
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# geometry config
# --------------------------------------------------------------------------- #
XBOUND = (-40.0, 40.0, 0.4)
YBOUND = (-40.0, 40.0, 0.4)
ZBOUND = (-1.0, 5.4, 0.4)
DBOUND = (2.0, 58.0, 0.5)          # depth bins for the lift (112 bins)


def _gridcfg(b):
    lo, hi, step = b
    n = int(round((hi - lo) / step))
    return lo, hi, step, n


class CamEncoder(nn.Module):
    """Image encoder + a DepthNet head producing (depth-dist | context).

    backbone="resnet18" (trainable, stride 16) or "dinov2" (frozen DINOv2-small, patch 14).
    A frozen DINOv2 gives much stronger features; only the DepthNet + decoder then train."""

    def __init__(self, depth_bins: int, ctx_channels: int = 64, backbone: str = "resnet18",
                 upsample: int = 1):
        super().__init__()
        self.backbone = backbone
        self.upsample = upsample        # >1 = supervise/lift at a finer feature resolution
        self.D = depth_bins
        self.C = ctx_channels
        if backbone == "resnet18":
            from torchvision.models import resnet18
            net = resnet18(weights=None)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3)   # stride 16, 256 ch
            feat_dim = 256
        elif backbone in ("dinov2", "dinov2_base"):
            from transformers import AutoModel
            name = "facebook/dinov2-base" if backbone == "dinov2_base" else "facebook/dinov2-small"
            self.dino = AutoModel.from_pretrained(name)
            for p in self.dino.parameters():
                p.requires_grad = False
            feat_dim = 768 if backbone == "dinov2_base" else 384
        elif backbone == "vggt":
            feat_dim = 2048        # frozen VGGT patch tokens, fed from cache (no in-module backbone)
        else:
            raise ValueError(backbone)
        self.depthnet = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, depth_bins + ctx_channels, kernel_size=1))

    def _features(self, x):
        if self.backbone == "resnet18":
            return self.stem(x)                                 # (B*N,256,H/16,W/16)
        b, _, H, W = x.shape
        with torch.no_grad():
            tok = self.dino(x).last_hidden_state[:, 1:]         # drop CLS -> (B*N, h*w, 384)
        h, w = H // 14, W // 14
        return tok.transpose(1, 2).reshape(b, -1, h, w)         # (B*N,384,h,w)

    def forward(self, x):
        return self.forward_feat(self._features(x))

    def forward_feat(self, feat):
        """DepthNet head on precomputed backbone features (B*N, feat_dim, h, w) -> (ctx, depth).
        Used by the `vggt` backbone, whose 2048-d patch tokens come from a cache (not run in-loop)."""
        if self.upsample > 1:                  # finer feature/supervision resolution
            feat = F.interpolate(feat, scale_factor=self.upsample, mode="bilinear",
                                 align_corners=False)
        x = self.depthnet(feat)                # (B*N, D+C, h, w)
        depth = x[:, : self.D].softmax(dim=1)  # categorical depth distribution
        ctx = x[:, self.D :]                   # context features (not yet lifted)
        return ctx, depth                      # (B*N,C,h,w), (B*N,D,h,w)


class VoxelDecoder(nn.Module):
    """3D CNN: pooled context volume (C,X,Y,Z) -> occupancy logits (n_cls,X,Y,Z).

    `n_layers`/`hidden` set the depth/width. Defaults (2,64) match the original ResNet/
    DINOv2-small runs so their checkpoints still load; deeper (e.g. 4,96) for scaling."""

    def __init__(self, in_c: int, n_classes: int = 18, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers, c = [], in_c
        for _ in range(n_layers):
            layers += [nn.Conv3d(c, hidden, 3, padding=1), nn.BatchNorm3d(hidden),
                       nn.ReLU(inplace=True)]
            c = hidden
        layers += [nn.Conv3d(hidden, n_classes, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSSOccupancy(nn.Module):
    """Depth-supervised LSS occupancy network."""

    def __init__(self, image_hw: Tuple[int, int] = None,
                 downsample: int = None, ctx_channels: int = 64, n_classes: int = 18,
                 backbone: str = "resnet18", decoder_hidden: int = 64, decoder_layers: int = 2,
                 feat_upsample: int = 1, refine_iters: int = 1,
                 lidar_fusion: bool = False, lidar_raw: int = 3, lidar_channels: int = 32,
                 lidar_only: bool = False, det_classes: int = 0, det_anchor_sizes=None,
                 det_anchor_bottom=None, det_head_type: str = "anchor",
                 vggt_depth: bool = False):
        super().__init__()
        self.backbone = backbone
        # VGGT-depth lift (ablation #2): blend a frozen-VGGT metric-depth prior into the learned
        # depth distribution. VGGT depth is up-to-scale, so a learned scalar recovers metric scale.
        self.vggt_depth = vggt_depth
        self.feat_upsample = feat_upsample
        # >1 turns on the iterative render-and-refine lift: decode occupancy, sample it back
        # along each camera ray (first-hit transmittance), sharpen the depth dist, re-lift.
        self.refine_iters = refine_iters
        # LiDAR fusion: voxelize the point cloud into the SAME occ grid and concat its
        # embedded features into the camera volume before the decoder (LiDAR as an input, not
        # just depth supervision). Camera-only path is unchanged when False.
        self.lidar_fusion = lidar_fusion
        # ablation: zero the camera lifted volume so the decoder sees LiDAR only (same params).
        self.lidar_only = lidar_only
        _dino = backbone.startswith("dinov2") or backbone == "vggt"  # patch-14 grids at 252x700
        base_ds = 14 if _dino else 16
        # upsampling the features by U makes the effective patch/stride U× finer
        self.downsample = downsample or (base_ds // feat_upsample)
        self.image_hw = image_hw or ((252, 700) if _dino else (256, 704))
        self.C = ctx_channels
        self.n_classes = n_classes
        self.xb, self.yb, self.zb = XBOUND, YBOUND, ZBOUND
        _, _, _, self.nx = _gridcfg(XBOUND)
        _, _, _, self.ny = _gridcfg(YBOUND)
        _, _, _, self.nz = _gridcfg(ZBOUND)
        self.dlo, self.dhi, self.dstep, self.D = _gridcfg(DBOUND)

        self.encoder = CamEncoder(self.D, ctx_channels, backbone=backbone, upsample=feat_upsample)
        dec_in = ctx_channels
        if lidar_fusion:                                 # small 3D CNN over voxelized LiDAR
            self.lidar_branch = nn.Sequential(
                nn.Conv3d(lidar_raw, lidar_channels, 3, padding=1), nn.BatchNorm3d(lidar_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(lidar_channels, lidar_channels, 3, padding=1), nn.BatchNorm3d(lidar_channels),
                nn.ReLU(inplace=True))
            dec_in = ctx_channels + lidar_channels
        self.decoder = VoxelDecoder(dec_in, n_classes,
                                    hidden=decoder_hidden, n_layers=decoder_layers)
        # M3: optional detection head on the SAME fused voxel volume (one encoder, two heads).
        self.det_head = None
        if det_classes > 0:
            det_pcr = [XBOUND[0], YBOUND[0], ZBOUND[0], XBOUND[1], YBOUND[1], ZBOUND[1]]
            if det_head_type == "center":            # arm D: anchor-free CenterPoint head
                from ..det_head import VoxelCenterHead
                self.det_head = VoxelCenterHead(dec_in, self.nz, det_pcr, num_classes=det_classes,
                                                voxel_size=(XBOUND[2], YBOUND[2]),
                                                nx=self.nx, ny=self.ny)
            else:
                from ..det_head import VoxelDetHead
                self.det_head = VoxelDetHead(dec_in, self.nz, det_pcr, num_classes=det_classes,
                                             anchor_sizes=det_anchor_sizes or ((4.6, 1.97, 1.74),),
                                             anchor_bottom=(det_anchor_bottom if det_anchor_bottom is not None
                                                            else -1.0))
        self.free_idx = n_classes - 1                    # Occ3D free/empty class = 17
        if refine_iters > 1:                             # learnable strength of the feedback prior
            self.refine_alpha = nn.Parameter(torch.tensor(1.0))
        if vggt_depth:                                   # ablation #2: VGGT metric-depth prior
            import math
            self.vggt_log_scale = nn.Parameter(torch.tensor(math.log(18.8)))  # up-to-scale -> metric
            self.vggt_blend = nn.Parameter(torch.tensor(2.0))                 # prior strength (log blend)
        self.register_buffer("frustum", self._create_frustum(), persistent=False)
        # grid lower-corner and voxel size, for pooling
        self.register_buffer("bx", torch.tensor([XBOUND[0], YBOUND[0], ZBOUND[0]]), persistent=False)
        self.register_buffer("dx", torch.tensor([XBOUND[2], YBOUND[2], ZBOUND[2]]), persistent=False)
        self.register_buffer("nxyz", torch.tensor([self.nx, self.ny, self.nz]), persistent=False)

    # ---- geometry ------------------------------------------------------- #
    def _create_frustum(self):
        H, W = self.image_hw
        fH, fW = H // self.downsample, W // self.downsample
        ds = torch.arange(self.dlo, self.dhi, self.dstep).view(-1, 1, 1).expand(-1, fH, fW)
        D = ds.shape[0]
        xs = torch.linspace(0, W - 1, fW).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, H - 1, fH).view(1, fH, 1).expand(D, fH, fW)
        return torch.stack((xs, ys, ds), dim=-1)        # (D,fH,fW,3) image (u,v,depth)

    def get_geometry(self, rots, trans, intrins):
        """frustum image points -> ego 3D points. rots/trans: cam->ego; intrins: K."""
        B, N = rots.shape[:2]
        pts = self.frustum.to(rots.device).view(1, 1, self.D, *self.frustum.shape[1:3], 3)
        # (u,v,d) -> (u*d, v*d, d), then K^-1, then cam->ego
        pts = torch.cat([pts[..., :2] * pts[..., 2:3], pts[..., 2:3]], dim=-1)
        comb = rots.matmul(torch.inverse(intrins))      # (B,N,3,3)
        pts = comb.view(B, N, 1, 1, 1, 3, 3).matmul(pts.unsqueeze(-1)).squeeze(-1)
        pts = pts + trans.view(B, N, 1, 1, 1, 3)
        return pts                                       # (B,N,D,fH,fW,3) ego

    def voxel_pool(self, geom, feat):
        """Scatter frustum features into the voxel grid (pure-torch scatter_add).

        geom: (B,N,D,fH,fW,3) ego xyz; feat: (B,N,C,D,fH,fW). -> (B,C,nx,ny,nz)."""
        B, N, C, D, fH, fW = feat.shape
        feat = feat.permute(0, 1, 3, 4, 5, 2).reshape(B, -1, C)         # (B, M, C)
        # voxel indices
        idx = ((geom - self.bx.to(geom)) / self.dx.to(geom)).long().reshape(B, -1, 3)  # (B,M,3)
        nx, ny, nz = self.nxyz.tolist()
        keep = ((idx[..., 0] >= 0) & (idx[..., 0] < nx) &
                (idx[..., 1] >= 0) & (idx[..., 1] < ny) &
                (idx[..., 2] >= 0) & (idx[..., 2] < nz))
        out = feat.new_zeros(B, nx * ny * nz, C)
        flat = idx[..., 0] * (ny * nz) + idx[..., 1] * nz + idx[..., 2]  # (B,M)
        for b in range(B):                         # per-batch scatter (B is tiny)
            k = keep[b]
            out[b].index_add_(0, flat[b][k].clamp(0, nx * ny * nz - 1), feat[b][k])
        return out.view(B, nx, ny, nz, C).permute(0, 4, 1, 2, 3).contiguous()  # (B,C,nx,ny,nz)

    def _norm_grid(self, geom):
        """ego xyz (B,N,D,h,w,3) -> grid_sample coords in [-1,1] for an (nx,ny,nz) volume.
        grid_sample on a (B,1,nx,ny,nz) input reads the last grid dim as W(=nz)→z, H(=ny)→y,
        D_in(=nx)→x, so we stack (z,y,x)."""
        idx = (geom - self.bx.to(geom)) / self.dx.to(geom)          # float voxel index (x,y,z)
        n = self.nxyz.to(geom).float()
        norm = idx / (n - 1) * 2 - 1                                # align_corners=True mapping
        return torch.stack([norm[..., 2], norm[..., 1], norm[..., 0]], dim=-1)

    def _refine_depth(self, occ, depth, geom):
        """Render the decoded occupancy back along each ray and sharpen the depth dist.
        occ: (B,ncls,nx,ny,nz); depth: (B,N,D,h,w); geom: (B,N,D,h,w,3) ego. -> new depth."""
        B, N, D, h, w = depth.shape
        occupied = (1.0 - occ.softmax(1)[:, self.free_idx]).unsqueeze(1)   # (B,1,nx,ny,nz)
        vol = occupied.unsqueeze(1).expand(B, N, 1, self.nx, self.ny, self.nz).reshape(B * N, 1, self.nx, self.ny, self.nz)
        grid = self._norm_grid(geom).reshape(B * N, D, h, w, 3)
        occ_along = F.grid_sample(vol, grid, align_corners=True, padding_mode="zeros")
        occ_along = occ_along.view(B, N, D, h, w).clamp(0, 1)              # occupied prob per depth bin
        # first-hit: prob the ray first meets a surface at bin d  = occ_d * Π_{d'<d}(1-occ_d')
        one_minus = (1.0 - occ_along).clamp(min=1e-4)
        trans = torch.cat([torch.ones_like(one_minus[:, :, :1]),
                           torch.cumprod(one_minus, dim=2)[:, :, :-1]], dim=2)
        first_hit = occ_along * trans
        logit = torch.log(depth + 1e-6) + self.refine_alpha * torch.log(first_hit + 1e-6)
        return logit.softmax(dim=2)

    def forward(self, imgs, rots, trans, intrins, lidar_vox=None, drop_camera=False, drop_lidar=False,
                vggt_depth=None, vggt_feat=None):
        """imgs: (B,N,3,H,W); rots,trans: (B,N,3,3),(B,N,3) cam->ego; intrins: (B,N,3,3).
        lidar_vox: (B,lidar_raw,nx,ny,nz) voxelized LiDAR (fusion mode) or None.
        drop_camera/drop_lidar: per-call **modality dropout** — zero that branch's volume while
        keeping the decoder's channel layout, so ONE fusion model handles camera-only / LiDAR-only
        / fusion (modality-robust training + inference).
        Returns (occ_final, depth_init, aux) with aux={'occ':[...],'depth':[...]} for deep supervision."""
        B, N = imgs.shape[:2]
        if self.backbone == "vggt" and vggt_feat is not None:   # cached frozen-VGGT patch features
            ctx, depth = self.encoder.forward_feat(vggt_feat.flatten(0, 1).float())
        else:
            ctx, depth = self.encoder(imgs.flatten(0, 1))   # (B*N,C,h,w), (B*N,D,h,w)
        C, h, w = ctx.shape[1], ctx.shape[2], ctx.shape[3]
        D = depth.shape[1]
        ctx = ctx.view(B, N, C, h, w)
        depth = depth.view(B, N, D, h, w)
        if self.vggt_depth and vggt_depth is not None:      # blend frozen-VGGT metric-depth prior
            scale = torch.exp(self.vggt_log_scale)
            dm = (vggt_depth.to(depth) * scale).clamp(self.dlo, self.dhi - 1e-3)   # (B,N,h,w) metric
            centers = (torch.arange(D, device=depth.device) * self.dstep + self.dlo).view(1, 1, D, 1, 1)
            prior = torch.exp(-((dm.unsqueeze(2) - centers) / (2.0 * self.dstep)) ** 2)  # (B,N,D,h,w)
            prior = prior / prior.sum(2, keepdim=True).clamp_min(1e-6)
            depth = (depth.clamp_min(1e-6).log()
                     + self.vggt_blend * prior.clamp_min(1e-6).log()).softmax(2)
        geom = self.get_geometry(rots, trans, intrins)      # (B,N,D,h,w,3)
        lid = self.lidar_branch(lidar_vox) if (self.lidar_fusion and lidar_vox is not None) else None
        if lid is not None and drop_lidar:                  # modality dropout: hide LiDAR
            lid = torch.zeros_like(lid)
        occ_all, depth_all = [], []
        cur = depth
        for it in range(self.refine_iters):
            lifted = cur.unsqueeze(2) * ctx.unsqueeze(3)     # ctx ⊗ depth -> (B,N,C,D,h,w)
            vox = self.voxel_pool(geom, lifted)              # (B,C,nx,ny,nz)
            if self.lidar_only or drop_camera:               # hide the camera input
                vox = torch.zeros_like(vox)
            if lid is not None:
                vox = torch.cat([vox, lid], dim=1)           # fuse LiDAR features
            occ = self.decoder(vox)
            occ_all.append(occ); depth_all.append(cur)
            if it < self.refine_iters - 1:
                cur = self._refine_depth(occ, cur, geom)
        aux = {"occ": occ_all, "depth": depth_all}
        if self.det_head is not None:                    # M3: detection off the shared fused vox
            aux["det"] = self.det_head(vox)
        return occ_all[-1], depth_all[0], aux


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.occupancy.models.lss_occ
# Forward-pass smoke test on random inputs; checks output grid shape.
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = LSSOccupancy(image_hw=(256, 704)).to(dev)
    p = sum(x.numel() for x in m.parameters()) / 1e6
    B, N = 1, 6
    imgs = torch.randn(B, N, 3, 256, 704, device=dev)
    rots = torch.eye(3, device=dev).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    trans = torch.randn(B, N, 3, device=dev)
    K = torch.tensor([[500., 0, 352], [0, 500., 128], [0, 0, 1]], device=dev)
    intr = K.view(1, 1, 3, 3).repeat(B, N, 1, 1)
    with torch.no_grad():
        occ, depth, aux = m(imgs, rots, trans, intr)
    print(f"params={p:.1f}M  occ={tuple(occ.shape)}  depth={tuple(depth.shape)}")
    assert occ.shape == (B, 18, 200, 200, 16), occ.shape
    print("OK: occupancy grid (B,18,200,200,16)")
    m2 = LSSOccupancy(image_hw=(256, 704), refine_iters=2).to(dev)
    with torch.no_grad():
        occ2, d2, aux2 = m2(imgs, rots, trans, intr)
    assert occ2.shape == (B, 18, 200, 200, 16) and len(aux2["occ"]) == 2
    print(f"OK: refine_iters=2 -> {len(aux2['occ'])} occ stages, alpha={m2.refine_alpha.item():.2f}")
