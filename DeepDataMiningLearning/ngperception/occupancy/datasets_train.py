"""
ngperception.occupancy.datasets_train
=====================================

Training dataset for the LSS occupancy net: each nuScenes keyframe ->
  * 6 surround images (resized to the network input, ImageNet-normalized)
  * per-camera intrinsics K (adjusted for the resize) and cam->ego extrinsics
  * Occ3D GT (semantics 200x200x16 + mask_camera)
  * per-camera **LiDAR-projected depth** binned to the lift's depth bins — the signal
    that makes this *depth-supervised* (BEVDepth).

Sample tokens come from the extracted Occ3D `gts/` tree; calibration/images/LiDAR are
resolved via the nuScenes devkit.
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import Occ3DNuScenesDataset
from .geom import CAMS
from .models.lss_occ import DBOUND

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _depth_to_bins(depth, dlo, dstep, nbins):
    """sparse metric depth (H,W) -> bin index (H,W) int64; -1 where invalid."""
    out = np.full(depth.shape, -1, np.int64)
    m = (depth > dlo) & (depth < dlo + dstep * nbins)
    out[m] = ((depth[m] - dlo) / dstep).astype(np.int64)
    return out


class NuScenesOccTrainDataset(Dataset):
    def __init__(self, gts_root: str, nusc, image_hw=(256, 704), downsample=16,
                 scenes=None, max_samples: Optional[int] = None, stride: int = 1,
                 depth_source: str = "lidar"):
        from pyquaternion import Quaternion
        from .geom import PC_RANGE, VOXEL_SIZE, GRID_SIZE
        self.Q = Quaternion
        self.occ = Occ3DNuScenesDataset(gts_root, scenes=scenes,
                                        max_samples=max_samples, stride=stride)
        self.nusc = nusc
        self.H, self.W = image_hw
        self.fH, self.fW = self.H // downsample, self.W // downsample
        self.dlo, self.dstep = DBOUND[0], DBOUND[2]
        self.D = int(round((DBOUND[1] - DBOUND[0]) / DBOUND[2]))
        # "lidar" = project LiDAR sweep (sparse); "occ" = render depth from the dense Occ3D GT
        self.depth_source = depth_source
        self._vorig = np.asarray(PC_RANGE[:3], np.float32)        # grid lower corner
        self._vs = float(VOXEL_SIZE)
        self._gsz = np.asarray(GRID_SIZE, np.int64)

    def _occ_rendered_depth(self, sem, K, R_c2e, t_c2e, ow, oh):
        """Render dense depth from the Occ3D GT: project occupied voxel centres into the
        camera, keep the nearest (z-buffer) per feature cell. Dense + GT-consistent."""
        ii, jj, kk = np.where(sem != 17)                         # occupied voxels
        centers = self._vorig + self._vs * (np.stack([ii, jj, kk], 1) + 0.5)  # ego (N,3)
        # project the 8 voxel corners (not just the centre) so each voxel fills its footprint
        offs = self._vs * np.array([[a, b, c] for a in (-.5, .5) for b in (-.5, .5)
                                    for c in (-.5, .5)], np.float32)             # (8,3)
        centers = (centers[:, None, :] + offs[None, :, :]).reshape(-1, 3)        # (N*8,3)
        cam = (centers - t_c2e) @ R_c2e                          # ego -> cam
        front = cam[:, 2] > 0.1
        cam = cam[front]
        Kf = K.copy(); Kf[0] *= self.fW / ow; Kf[1] *= self.fH / oh  # K at feature resolution
        uvw = cam @ Kf.T
        u = (uvw[:, 0] / uvw[:, 2]).astype(int)
        v = (uvw[:, 1] / uvw[:, 2]).astype(int)
        d = cam[:, 2]
        inb = (u >= 0) & (u < self.fW) & (v >= 0) & (v < self.fH)
        u, v, d = u[inb], v[inb], d[inb]
        dm = np.zeros((self.fH, self.fW), np.float32)
        order = np.argsort(-d)                                   # far first -> near overwrites (z-buffer min)
        dm[v[order], u[order]] = d[order]
        return _depth_to_bins(dm, self.dlo, self.dstep, self.D)

    def __len__(self):
        return len(self.occ)

    def _lidar_depth(self, token, cam_sd_token, K, R_c2e, t_c2e, ow, oh, sx, sy):
        """Project LIDAR_TOP into a camera -> sparse depth at the *resized* image size."""
        from PIL import Image  # noqa
        s = self.nusc.get("sample", token)
        lsd = self.nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        pts = np.fromfile(os.path.join(self.nusc.dataroot, lsd["filename"]),
                          dtype=np.float32).reshape(-1, 5)[:, :3]
        lcs = self.nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
        Rl = self.Q(lcs["rotation"]).rotation_matrix
        tl = np.array(lcs["translation"])
        ego = pts @ Rl.T + tl                                  # lidar -> ego
        cam = (ego - t_c2e) @ R_c2e                            # ego -> cam (R_c2e^T == @R_c2e on rows)
        front = cam[:, 2] > 0.5
        cam = cam[front]
        uvw = cam @ K.T
        u = (uvw[:, 0] / uvw[:, 2]) * sx
        v = (uvw[:, 1] / uvw[:, 2]) * sy
        d = cam[:, 2]
        inb = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        u, v, d = u[inb].astype(int), v[inb].astype(int), d[inb]
        dm = np.zeros((self.H, self.W), np.float32)
        order = np.argsort(-d)
        dm[v[order], u[order]] = d[order]
        # downsample to feature resolution by max-pooling sparse depth
        dm = dm.reshape(self.fH, self.H // self.fH, self.fW, self.W // self.fW).max(axis=(1, 3))
        return _depth_to_bins(dm, self.dlo, self.dstep, self.D)

    def __getitem__(self, i):
        from PIL import Image
        s = self.occ[i]
        sample = self.nusc.get("sample", s.sample_token)
        imgs, Ks, Rs, ts, depths = [], [], [], [], []
        for cam in CAMS:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            img = Image.open(os.path.join(self.nusc.dataroot, sd["filename"])).convert("RGB")
            ow, oh = img.size
            sx, sy = self.W / ow, self.H / oh
            img_r = img.resize((self.W, self.H))
            arr = (np.asarray(img_r, np.float32) / 255.0 - _MEAN) / _STD
            imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
            K = np.array(cs["camera_intrinsic"], np.float32)
            Ks.append(torch.from_numpy(np.diag([sx, sy, 1]).astype(np.float32) @ K))  # resize-adjusted
            R = self.Q(cs["rotation"]).rotation_matrix.astype(np.float32)
            t = np.array(cs["translation"], np.float32)
            Rs.append(torch.from_numpy(R)); ts.append(torch.from_numpy(t))
            if self.depth_source == "occ":
                dbin = self._occ_rendered_depth(s.semantics, K, R, t, ow, oh)
            else:
                dbin = self._lidar_depth(s.sample_token, sd["token"], K, R, t, ow, oh, sx, sy)
            depths.append(torch.from_numpy(dbin))
        return {
            "imgs": torch.stack(imgs), "intrins": torch.stack(Ks),
            "rots": torch.stack(Rs), "trans": torch.stack(ts),
            "depth_gt": torch.stack(depths),                       # (N,fH,fW) bin idx (-1 invalid)
            "semantics": torch.from_numpy(s.semantics.astype(np.int64)),
            "mask_camera": torch.from_numpy(s.mask_camera.astype(bool)),
        }


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.occupancy.datasets_train --gts <gts> --nusc <root>
# Loads one sample and prints tensor shapes + LiDAR-depth coverage.
# ===========================================================================
if __name__ == "__main__":
    import argparse
    from nuscenes import NuScenes
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", default="/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval")
    a = ap.parse_args()
    nusc = NuScenes(version="v1.0-trainval", dataroot=a.nusc, verbose=False)
    ds = NuScenesOccTrainDataset(a.gts, nusc, max_samples=2)
    b = ds[0]
    for k, v in b.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print(f"  LiDAR-depth valid pixels: {(b['depth_gt'] >= 0).sum().item()}")
    print(f"  occupied GT voxels: {(b['semantics'] != 17).sum().item()}")
