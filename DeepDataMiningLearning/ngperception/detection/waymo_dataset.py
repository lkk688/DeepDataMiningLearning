"""
ngperception.detection.waymo_dataset
=====================================

A basic **Waymo (KITTI-format) 3D detection** loader in our unified format. The Waymo Open
Dataset is commonly exported to a KITTI-style layout (velodyne `.bin` + `label_all` +
multi-camera `calib`); this loader reads that layout, reusing the KITTI camera→LiDAR box
transform ([`kitti_dataset.py`](kitti_dataset.py)) with the **front camera** calib
(`Tr_velo_to_cam_0`). Differences vs KITTI: Waymo is **360°** (points span ~±75 m) and cars
are a touch larger (~4.8×2.1×1.7 m).

**Data note:** only a **1-frame sample** is staged locally
(`2D3DFusion/data/waymokittisample`), enough to verify the loader; full training needs the real
Waymo Open Dataset exported to this layout (an H100-scale job).
"""
from __future__ import annotations
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .kitti_dataset import KittiCalib, camera_to_lidar_boxes


class WaymoCalib(KittiCalib):
    """Waymo-KITTI calib: front-camera Tr_velo_to_cam_0 (+ R0_rect=I)."""

    def __init__(self, path):
        rows = {}
        for line in open(path):
            if ":" in line:
                k, v = line.split(":", 1)
                rows[k.strip()] = np.array(v.split(), np.float32)
        self.R0 = rows["R0_rect"].reshape(3, 3)
        self.V2C = rows["Tr_velo_to_cam_0"].reshape(3, 4)
        self.P2 = rows.get("P0", np.zeros(12, np.float32)).reshape(3, 4)


class WaymoKittiDataset(Dataset):
    def __init__(self, root="/home/lkk688/Developer/2D3DFusion/data/waymokittisample",
                 pc_range=(-75, -75, -2, 75, 75, 4), classes=("Car",), max_frames=None):
        self.root = root
        self.pcr = np.array(pc_range, np.float32)
        self.classes = classes
        ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(root, "velodyne", "*.bin")))
        self.ids = ids[:max_frames] if max_frames else ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        pts = np.fromfile(os.path.join(self.root, "velodyne", fid + ".bin"), np.float32).reshape(-1, 4)
        m = np.all((pts[:, :3] >= self.pcr[:3]) & (pts[:, :3] < self.pcr[3:]), axis=1)
        pts = pts[m]
        cam = []
        lp = os.path.join(self.root, "label_all", fid + ".txt")
        for line in open(lp):
            f = line.split()
            if f[0] not in self.classes:
                continue
            h, w, l = float(f[8]), float(f[9]), float(f[10])
            x, y, z, ry = float(f[11]), float(f[12]), float(f[13]), float(f[14])
            cam.append([x, y, z, l, h, w, ry])
        if cam:
            calib = WaymoCalib(os.path.join(self.root, "calib", fid + ".txt"))
            gt = camera_to_lidar_boxes(np.array(cam, np.float32), calib)
            keep = np.all((gt[:, :2] >= self.pcr[:2]) & (gt[:, :2] < self.pcr[3:5]), axis=1)
            gt = gt[keep]
        else:
            gt = np.zeros((0, 7), np.float32)
        gt = np.concatenate([gt, np.zeros((len(gt), 1), np.float32)], axis=1)   # label 0 = Car
        return {"points": torch.from_numpy(pts), "gt": torch.from_numpy(gt), "id": fid}


# =========================================================================== #
# sanity:  python -m ...detection.waymo_dataset
# =========================================================================== #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lkk688/Developer/2D3DFusion/data/waymokittisample")
    a = ap.parse_args()
    ds = WaymoKittiDataset(a.root)
    print(f"{len(ds)} frames")
    b = ds[0]
    print(f"frame {b['id']}: points={tuple(b['points'].shape)}  cars={tuple(b['gt'].shape)}")
    if len(b["gt"]):
        print(f"  car0 [x,y,z,dx,dy,dz,head]: {[round(float(v),2) for v in b['gt'][0][:7]]}")
        print(f"  mean size dxyz={b['gt'][:,3:6].numpy().mean(0).round(2)} (Waymo car ~4.8,2.1,1.7)")
