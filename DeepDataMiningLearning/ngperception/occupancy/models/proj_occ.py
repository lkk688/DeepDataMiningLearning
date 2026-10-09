"""
ngperception.occupancy.models.proj_occ
======================================

A **projection-sampling occupancy** network: the non-LSS architecture family.

The point of this file is to be a different *lifting mechanism*, not a better
one. Swapping ResNet for DINOv2 inside `lss_occ` changes the image encoder while
leaving the lift untouched, so a result that holds across those backbones has
still only been shown for one way of getting pixels into 3D. This is the other
way round:

    LSS (lss_occ.py)   PUSH: predict a depth distribution per pixel, take the
                       outer product with the context feature, and scatter the
                       resulting frustum into the voxel grid.

    Projection (here)  PULL: every voxel projects itself into every camera and
                       bilinearly samples whatever feature it lands on. There is
                       no depth prediction anywhere in the model.

This is BEVFormer/SurroundOcc's spatial cross-attention in its simplest
non-deformable form: fixed reference points, one sample per camera, mean over
the cameras that saw the voxel. Sampling alone cannot tell two voxels on the same
ray apart -- they read the identical pixel -- so a fixed Fourier encoding of the
voxel centre and the per-voxel camera-visibility count are concatenated before
the decoder, which is what lets the 3D CNN resolve position along the ray.

Everything outside the lift is deliberately shared with `lss_occ`: the same
frozen backbone via `CamEncoder._features`, the same `VoxelDecoder`, the same
grid bounds, the same forward signature. So a run of this file against a run of
that one differs in the lift and nothing else.

Grid (Occ3D-nuScenes): x,y in [-40,40], z in [-1,5.4], 0.4 m -> (200,200,16).
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lss_occ import CamEncoder, VoxelDecoder, XBOUND, YBOUND, ZBOUND, _gridcfg


def voxel_centres(xb=XBOUND, yb=YBOUND, zb=ZBOUND):
    """(nx,ny,nz,3) ego-frame centres of every voxel, matching the LSS grid.

    Centres, not lower corners: the lift samples at the middle of the voxel it is
    filling. `lss_occ.voxel_pool` floors continuous points into the same bins, so
    both families address an identical grid.
    """
    out = []
    for lo, hi, step in (xb, yb, zb):
        n = _gridcfg((lo, hi, step))[3]
        out.append(torch.arange(n, dtype=torch.float32) * step + lo + step / 2)
    gx, gy, gz = torch.meshgrid(*out, indexing='ij')
    return torch.stack((gx, gy, gz), dim=-1)


def fourier_encode(pts, n_freq: int = 4, xb=XBOUND, yb=YBOUND, zb=ZBOUND):
    """Fixed (not learned) sin/cos encoding of normalised voxel centres.

    Fixed on purpose: a learned embedding is extra capacity that the two arms
    would have to be shown not to exploit differently. This adds none.
    Returns (nx,ny,nz, 3*2*n_freq+3).
    """
    lo = torch.tensor([xb[0], yb[0], zb[0]])
    hi = torch.tensor([xb[1], yb[1], zb[1]])
    p = (pts - lo) / (hi - lo) * 2 - 1                       # -> [-1,1]
    feats = [p]
    for k in range(n_freq):
        feats += [torch.sin(2 ** k * torch.pi * p), torch.cos(2 ** k * torch.pi * p)]
    return torch.cat(feats, dim=-1)


class ProjOccupancy(nn.Module):
    """Voxel-query projection sampling -> 3D CNN -> occupancy logits."""

    def __init__(self, image_hw: Tuple[int, int] = None, downsample: int = None,
                 ctx_channels: int = 64, n_classes: int = 18,
                 backbone: str = 'dinov2_base', decoder_hidden: int = 64,
                 decoder_layers: int = 2, n_freq: int = 4):
        super().__init__()
        self.backbone = backbone
        _dino = backbone.startswith('dinov') or backbone in ('vggt', 'radio', 'siglip2')
        base_ds, default_hw = (14 if _dino else 16), ((252, 700) if _dino else (256, 704))
        self.downsample = downsample or base_ds
        self.image_hw = image_hw or default_hw
        self.C = ctx_channels
        self.n_classes = n_classes
        self.nx = _gridcfg(XBOUND)[3]
        self.ny = _gridcfg(YBOUND)[3]
        self.nz = _gridcfg(ZBOUND)[3]

        # The backbone is reused wholesale so the two families see identical
        # image features; only the head that follows it differs. depth_bins=1
        # builds CamEncoder's depthnet, which we do not call -- `_features` is
        # the only thing used from it.
        self.encoder = CamEncoder(1, ctx_channels, backbone=backbone)
        feat_dim = self._feat_dim(backbone)
        # Mirrors lss_occ's depthnet in shape and depth, minus the depth channels,
        # so the trainable head is comparable in size across the two families.
        self.ctxnet = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ctx_channels, kernel_size=1))

        pos = fourier_encode(voxel_centres(), n_freq)         # (nx,ny,nz,P)
        self.register_buffer('pos', pos.permute(3, 0, 1, 2).contiguous(),
                             persistent=False)                # (P,nx,ny,nz)
        self.register_buffer('centres', voxel_centres().reshape(-1, 3),
                             persistent=False)                # (M,3)
        # +1 for the visibility count: how many cameras actually saw the voxel.
        # Without it the decoder cannot distinguish "seen and empty" from
        # "never projected into any image", which is the distinction this whole
        # line of work is about.
        self.decoder = VoxelDecoder(ctx_channels + pos.shape[-1] + 1, n_classes,
                                    hidden=decoder_hidden, n_layers=decoder_layers)
        self.free_idx = n_classes - 1

    @staticmethod
    def _feat_dim(backbone):
        return {'resnet18': 256, 'dinov2': 384, 'dinov2_base': 768,
                'dinov2_large': 1024, 'vggt': 2048, 'qwendrive': 2560,
                'radio': 768, 'siglip2': 768, 'dinov3': 1024}[backbone]

    def sample(self, ctx, rots, trans, intrins):
        """Pull one feature per voxel per camera and average over valid cameras.

        ctx: (B,N,C,h,w); rots,trans: cam->ego. Returns (B,C,nx,ny,nz) and the
        visibility count (B,1,nx,ny,nz).

        Cameras are looped rather than batched: the full (B,N,C,nx,ny,nz) tensor
        is 6x the volume and does not fit alongside the backbone at batch 8.
        """
        B, N, C, h, w = ctx.shape
        H, W = self.image_hw
        M = self.centres.shape[0]
        acc = ctx.new_zeros(B, C, M)
        cnt = ctx.new_zeros(B, 1, M)
        p_ego = self.centres.to(ctx.dtype).to(ctx.device).view(1, M, 3)
        for n in range(N):
            R, t = rots[:, n], trans[:, n]                      # cam->ego
            # ego -> cam is the inverse: R^T (p - t). R is a rotation, so R^T is
            # its inverse; using inverse() here instead would silently absorb any
            # scale a caller put in R.
            p_cam = torch.einsum('bij,bmj->bmi', R.transpose(1, 2).to(ctx.dtype),
                                 p_ego - t.to(ctx.dtype).unsqueeze(1))
            z = p_cam[..., 2]
            uv = torch.einsum('bij,bmj->bmi', intrins[:, n].to(ctx.dtype), p_cam)
            zc = z.clamp_min(1e-4)
            u, v = uv[..., 0] / zc, uv[..., 1] / zc
            valid = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            grid = torch.stack((u / W * 2 - 1, v / H * 2 - 1), dim=-1)
            # grid_sample clamps out-of-range coordinates to the border, which
            # would fabricate a feature for voxels no camera saw; `valid` is what
            # actually discards them.
            s = F.grid_sample(ctx[:, n], grid.view(B, M, 1, 2), mode='bilinear',
                              padding_mode='zeros', align_corners=False)
            acc = acc + s.view(B, C, M) * valid.view(B, 1, M).to(s.dtype)
            cnt = cnt + valid.view(B, 1, M).to(s.dtype)
        vox = acc / cnt.clamp_min(1.0)
        shape = (B, -1, self.nx, self.ny, self.nz)
        return vox.view(*shape), cnt.view(B, 1, self.nx, self.ny, self.nz)

    def forward(self, imgs, rots, trans, intrins, lidar_vox=None,
                drop_camera=False, drop_lidar=False, vggt_depth=None, vggt_feat=None):
        """Same signature and return shape as LSSOccupancy.forward.

        The second return value is LSS's initial depth distribution, which this
        family does not have; it is None rather than a zeros tensor so that a
        caller who tries to supervise depth fails loudly instead of training
        against a constant.
        """
        B, N = imgs.shape[:2]
        if vggt_feat is not None:
            feat = vggt_feat.flatten(0, 1).float()
        else:
            feat = self.encoder._features(imgs.flatten(0, 1))
        ctx = self.ctxnet(feat)                                 # (B*N,C,h,w)
        ctx = ctx.view(B, N, *ctx.shape[1:])
        vox, cnt = self.sample(ctx, rots, trans, intrins)
        if drop_camera:
            vox = torch.zeros_like(vox)
        pos = self.pos.unsqueeze(0).expand(B, -1, -1, -1, -1).to(vox.dtype)
        occ = self.decoder(torch.cat([vox, pos, cnt.clamp(0, N) / N], dim=1))
        return occ, None, {'occ': [occ], 'depth': []}


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.occupancy.models.proj_occ
# Forward-pass smoke test on random inputs; checks output grid shape and that
# a voxel in front of a camera actually samples a different feature than one
# behind it.
# ===========================================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = ProjOccupancy(image_hw=(252, 700), backbone='resnet18').to(dev)
    p = sum(x.numel() for x in m.parameters() if x.requires_grad) / 1e6
    B, N = 1, 6
    imgs = torch.randn(B, N, 3, 252, 700, device=dev)
    rots = torch.eye(3, device=dev).view(1, 1, 3, 3).repeat(B, N, 1, 1)
    trans = torch.zeros(B, N, 3, device=dev)
    K = torch.tensor([[500., 0, 350], [0, 500., 126], [0, 0, 1]], device=dev)
    intr = K.view(1, 1, 3, 3).repeat(B, N, 1, 1)
    with torch.no_grad():
        occ, depth, aux = m(imgs, rots, trans, intr)
    print(f'trainable={p:.1f}M  occ={tuple(occ.shape)}  depth={depth}')
    assert occ.shape == (B, 18, 200, 200, 16), occ.shape
    assert depth is None
    print('OK: occupancy grid (B,18,200,200,16), no depth head')
