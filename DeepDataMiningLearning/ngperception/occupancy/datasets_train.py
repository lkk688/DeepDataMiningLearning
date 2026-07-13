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
                 depth_source: str = "lidar", lidar_sweeps: int = 1, lidar_cache=None,
                 lidar_fusion: bool = False, det_boxes: bool = False, det_class_map=None):
        from pyquaternion import Quaternion
        from .geom import PC_RANGE, VOXEL_SIZE, GRID_SIZE
        self.Q = Quaternion
        self.det_boxes = det_boxes                   # M3: also emit 3D boxes (ego frame) for detection
        # None -> car-only (label 0, back-compat). A {category_prefix: idx} map -> multi-class.
        self.det_class_map = det_class_map
        self.occ = Occ3DNuScenesDataset(gts_root, scenes=scenes,
                                        max_samples=max_samples, stride=stride)
        self.nusc = nusc
        self.H, self.W = image_hw
        self.fH, self.fW = self.H // downsample, self.W // downsample
        self.dlo, self.dstep = DBOUND[0], DBOUND[2]
        self.D = int(round((DBOUND[1] - DBOUND[0]) / DBOUND[2]))
        # "lidar" = project LiDAR sweep; "occ" = render from Occ3D GT; "combined" = both;
        # "lidar_multi" = aggregated multi-sweep LiDAR (+region map, for the loss ablation)
        self.depth_source = depth_source
        self.lidar_sweeps = lidar_sweeps
        self.lidar_fusion = lidar_fusion             # also emit a voxelized-LiDAR volume (fusion input)
        self.lidar_cache = lidar_cache               # dir to cache aggregated multi-sweep points
        if lidar_cache:
            os.makedirs(lidar_cache, exist_ok=True)
        self._vorig = np.asarray(PC_RANGE[:3], np.float32)        # grid lower corner
        self._vs = float(VOXEL_SIZE)
        self._gsz = np.asarray(GRID_SIZE, np.int64)

    def _occ_rendered_depth(self, sem, K, R_c2e, t_c2e, ow, oh):
        """Render dense depth from the Occ3D GT: z-buffer occupied voxel corners into the
        camera. Returns (depth-bins, region-class) per feature cell — the class of the
        nearest surface (road/object/background), used to weight the loss; -1 = empty."""
        ii, jj, kk = np.where(sem != 17)                         # occupied voxels
        cls = sem[ii, jj, kk]                                    # class per voxel
        centers = self._vorig + self._vs * (np.stack([ii, jj, kk], 1) + 0.5)  # ego (N,3)
        # project the 8 voxel corners (not just the centre) so each voxel fills its footprint
        offs = self._vs * np.array([[a, b, c] for a in (-.5, .5) for b in (-.5, .5)
                                    for c in (-.5, .5)], np.float32)             # (8,3)
        centers = (centers[:, None, :] + offs[None, :, :]).reshape(-1, 3)        # (N*8,3)
        cls8 = np.repeat(cls, 8)
        cam = (centers - t_c2e) @ R_c2e                          # ego -> cam
        front = cam[:, 2] > 0.1
        cam, cls8 = cam[front], cls8[front]
        Kf = K.copy(); Kf[0] *= self.fW / ow; Kf[1] *= self.fH / oh  # K at feature resolution
        uvw = cam @ Kf.T
        u = (uvw[:, 0] / uvw[:, 2]).astype(int)
        v = (uvw[:, 1] / uvw[:, 2]).astype(int)
        d = cam[:, 2]
        inb = (u >= 0) & (u < self.fW) & (v >= 0) & (v < self.fH)
        u, v, d, cls8 = u[inb], v[inb], d[inb], cls8[inb]
        dm = np.zeros((self.fH, self.fW), np.float32)
        cm = np.full((self.fH, self.fW), -1, np.int64)
        order = np.argsort(-d)                                   # far first -> near overwrites (z-buffer)
        dm[v[order], u[order]] = d[order]
        cm[v[order], u[order]] = cls8[order]
        return _depth_to_bins(dm, self.dlo, self.dstep, self.D), cm

    def __len__(self):
        return len(self.occ)

    def _load_lidar_pts(self, token):
        """Points in the keyframe LIDAR_TOP frame — a single sweep, or `lidar_sweeps`
        motion-compensated sweeps aggregated (denser real depth). Cached per token (the 6
        cameras of a sample share one cloud)."""
        if getattr(self, "_lcache_tok", None) == token:
            return self._lcache_pts
        s = self.nusc.get("sample", token)
        if self.lidar_sweeps <= 1:
            pts = np.fromfile(os.path.join(self.nusc.dataroot,
                              self.nusc.get("sample_data", s["data"]["LIDAR_TOP"])["filename"]),
                              dtype=np.float32).reshape(-1, 5)[:, :3]
        else:
            cf = (os.path.join(self.lidar_cache, f"{token}_sw{self.lidar_sweeps}.npy")
                  if self.lidar_cache else None)
            if cf and os.path.exists(cf):
                pts = np.load(cf)
            else:
                from nuscenes.utils.data_classes import LidarPointCloud
                pc, _ = LidarPointCloud.from_file_multisweep(
                    self.nusc, s, "LIDAR_TOP", "LIDAR_TOP", nsweeps=self.lidar_sweeps)
                pts = pc.points[:3].T.astype(np.float32)
                if cf:
                    np.save(cf, pts)
        self._lcache_tok, self._lcache_pts = token, pts
        return pts

    def _lidar_voxel(self, token):
        """Voxelize the LiDAR cloud into the occ grid -> (3,nx,ny,nz) per-voxel features:
        [occupancy, log(1+count), mean height-residual]. This is the LiDAR *input* for fusion
        (an occupancy/geometry prior the camera lift lacks), separate from depth supervision."""
        pts = self._load_lidar_pts(token)                       # LIDAR_TOP frame (M,3)
        s = self.nusc.get("sample", token)
        lsd = self.nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        lcs = self.nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
        Rl = self.Q(lcs["rotation"]).rotation_matrix
        tl = np.array(lcs["translation"])
        ego = pts @ Rl.T + tl                                   # lidar -> ego
        gx, gy, gz = self._gsz.tolist()
        idx = np.floor((ego - self._vorig) / self._vs).astype(np.int64)
        m = ((idx[:, 0] >= 0) & (idx[:, 0] < gx) & (idx[:, 1] >= 0) & (idx[:, 1] < gy) &
             (idx[:, 2] >= 0) & (idx[:, 2] < gz))
        idx, ez = idx[m], ego[m, 2]
        count = np.zeros((gx, gy, gz), np.float32)
        np.add.at(count, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        zc = self._vorig[2] + (idx[:, 2] + 0.5) * self._vs      # voxel z-centre
        zres = np.zeros((gx, gy, gz), np.float32)
        np.add.at(zres, (idx[:, 0], idx[:, 1], idx[:, 2]), (ez - zc) / self._vs)
        occ = (count > 0).astype(np.float32)
        mean_zres = np.where(count > 0, zres / np.maximum(count, 1.0), 0.0)
        vol = np.stack([occ, np.log1p(count), mean_zres], 0)    # (3,nx,ny,nz)
        return torch.from_numpy(vol)

    def _det_class_of(self, name):
        """label idx for a box category name; None to skip. Car-only unless det_class_map is set."""
        if self.det_class_map is None:
            return 0 if name.startswith("vehicle.car") else None
        for pref, idx in self.det_class_map.items():
            if name.startswith(pref):
                return idx
        return None

    def _det_boxes(self, token):
        """3D boxes for a sample, transformed LiDAR->**EGO** (the occ voxel-grid frame).
        Returns (M,8) [x,y,z,dx,dy,dz,heading,label], filtered to the grid's x,y range.
        Car-only (label 0) by default; multi-class if `det_class_map` was given."""
        s = self.nusc.get("sample", token)
        lsd = self.nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        _, boxes, _ = self.nusc.get_sample_data(lsd["token"])   # boxes in LIDAR frame
        lcs = self.nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
        Rl = self.Q(lcs["rotation"]); tl = np.array(lcs["translation"])
        lo = self._vorig; hi = lo + self._gsz.astype(np.float32) * self._vs   # ego grid bounds
        out = []
        for b in boxes:
            cls = self._det_class_of(b.name)
            if cls is None:
                continue
            b.rotate(Rl); b.translate(tl)                       # LiDAR -> ego (devkit Box methods)
            x, y, z = b.center; w, l, h = b.wlh
            if not (lo[0] <= x < hi[0] and lo[1] <= y < hi[1]):
                continue
            v = b.orientation.rotation_matrix[:, 0]
            out.append([x, y, z, l, w, h, float(np.arctan2(v[1], v[0])), float(cls)])
        return torch.from_numpy(np.array(out, np.float32).reshape(-1, 8))

    def _lidar_depth(self, token, cam_sd_token, K, R_c2e, t_c2e, ow, oh, sx, sy):
        """Project LiDAR into a camera -> depth at the *resized* image size, binned."""
        s = self.nusc.get("sample", token)
        pts = self._load_lidar_pts(token)
        lsd = self.nusc.get("sample_data", s["data"]["LIDAR_TOP"])
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
        occ_depths, occ_regions = [], []
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
                dbin, _ = self._occ_rendered_depth(s.semantics, K, R, t, ow, oh)
                depths.append(torch.from_numpy(dbin))
            elif self.depth_source == "lidar_multi":
                depths.append(torch.from_numpy(
                    self._lidar_depth(s.sample_token, sd["token"], K, R, t, ow, oh, sx, sy)))
                _, oreg = self._occ_rendered_depth(s.semantics, K, R, t, ow, oh)
                occ_regions.append(torch.from_numpy(oreg))         # region for weighting
            elif self.depth_source == "combined":
                depths.append(torch.from_numpy(
                    self._lidar_depth(s.sample_token, sd["token"], K, R, t, ow, oh, sx, sy)))
                odbin, oreg = self._occ_rendered_depth(s.semantics, K, R, t, ow, oh)
                occ_depths.append(torch.from_numpy(odbin)); occ_regions.append(torch.from_numpy(oreg))
            else:
                depths.append(torch.from_numpy(
                    self._lidar_depth(s.sample_token, sd["token"], K, R, t, ow, oh, sx, sy)))
        out = {
            "imgs": torch.stack(imgs), "intrins": torch.stack(Ks),
            "rots": torch.stack(Rs), "trans": torch.stack(ts),
            "depth_gt": torch.stack(depths),                       # (N,fH,fW) bin idx (-1 invalid)
            "semantics": torch.from_numpy(s.semantics.astype(np.int64)),
            "mask_camera": torch.from_numpy(s.mask_camera.astype(bool)),
        }
        if self.depth_source == "combined":
            out["occ_depth"] = torch.stack(occ_depths)             # (N,fH,fW) bins
            out["occ_region"] = torch.stack(occ_regions)           # (N,fH,fW) class (-1 empty)
        if self.depth_source == "lidar_multi":
            out["occ_region"] = torch.stack(occ_regions)           # region for loss weighting
        if self.lidar_fusion:
            out["lidar_vox"] = self._lidar_voxel(s.sample_token)   # (3,nx,ny,nz) fusion input
        if self.det_boxes:
            out["det_gt"] = self._det_boxes(s.sample_token)        # (M,8) ego-frame boxes
        out["sample_idx"] = torch.tensor(i)                        # -> ds.occ.items[i] for token lookup
        return out


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
