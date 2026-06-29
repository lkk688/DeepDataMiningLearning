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

    def __init__(self, depth_bins: int, ctx_channels: int = 64, backbone: str = "resnet18"):
        super().__init__()
        self.backbone = backbone
        self.D = depth_bins
        self.C = ctx_channels
        if backbone == "resnet18":
            from torchvision.models import resnet18
            net = resnet18(weights=None)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3)   # stride 16, 256 ch
            feat_dim = 256
        elif backbone == "dinov2":
            from transformers import AutoModel
            self.dino = AutoModel.from_pretrained("facebook/dinov2-small")
            for p in self.dino.parameters():
                p.requires_grad = False
            feat_dim = 384
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
        x = self.depthnet(self._features(x))   # (B*N, D+C, h, w)
        depth = x[:, : self.D].softmax(dim=1)  # categorical depth distribution
        ctx = x[:, self.D :]
        # lift: outer product context ⊗ depth -> (B*N, C, D, h, w)
        feat = depth.unsqueeze(1) * ctx.unsqueeze(2)
        return feat, depth


class VoxelDecoder(nn.Module):
    """Small 3D CNN: pooled context volume (C,X,Y,Z) -> occupancy logits (n_cls,X,Y,Z)."""

    def __init__(self, in_c: int, n_classes: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, n_classes, 1))

    def forward(self, x):
        return self.net(x)


class LSSOccupancy(nn.Module):
    """Depth-supervised LSS occupancy network."""

    def __init__(self, image_hw: Tuple[int, int] = None,
                 downsample: int = None, ctx_channels: int = 64, n_classes: int = 18,
                 backbone: str = "resnet18"):
        super().__init__()
        self.backbone = backbone
        self.downsample = downsample or (14 if backbone == "dinov2" else 16)
        self.image_hw = image_hw or ((252, 700) if backbone == "dinov2" else (256, 704))
        self.C = ctx_channels
        self.n_classes = n_classes
        self.xb, self.yb, self.zb = XBOUND, YBOUND, ZBOUND
        _, _, _, self.nx = _gridcfg(XBOUND)
        _, _, _, self.ny = _gridcfg(YBOUND)
        _, _, _, self.nz = _gridcfg(ZBOUND)
        self.dlo, self.dhi, self.dstep, self.D = _gridcfg(DBOUND)

        self.encoder = CamEncoder(self.D, ctx_channels, backbone=backbone)
        self.decoder = VoxelDecoder(ctx_channels, n_classes)
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

    def forward(self, imgs, rots, trans, intrins):
        """imgs: (B,N,3,H,W); rots,trans: (B,N,3,3),(B,N,3) cam->ego; intrins: (B,N,3,3).
        Returns occ logits (B,n_classes,nx,ny,nz) and depth dist (B,N,D,fH,fW)."""
        B, N = imgs.shape[:2]
        feat, depth = self.encoder(imgs.flatten(0, 1))      # (B*N,C,D,h,w), (B*N,D,h,w)
        C, D, h, w = feat.shape[1:]
        feat = feat.view(B, N, C, D, h, w)
        geom = self.get_geometry(rots, trans, intrins)      # (B,N,D,h,w,3)
        vox = self.voxel_pool(geom, feat)                   # (B,C,nx,ny,nz)
        occ = self.decoder(vox)                             # (B,n_classes,nx,ny,nz)
        return occ, depth.view(B, N, D, h, w)


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
        occ, depth = m(imgs, rots, trans, intr)
    print(f"params={p:.1f}M  occ={tuple(occ.shape)}  depth={tuple(depth.shape)}")
    assert occ.shape == (B, 18, 200, 200, 16), occ.shape
    print("OK: occupancy grid (B,18,200,200,16)")
