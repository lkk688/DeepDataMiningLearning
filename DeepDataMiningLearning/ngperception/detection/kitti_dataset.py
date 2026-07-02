"""
ngperception.detection.kitti_dataset
=====================================

A lean, self-contained **KITTI 3D object-detection** loader for the pure-torch PointPillars
(Car, single class to start). No mmcv, no OpenPCDet framework — just numpy. The one subtle
piece, harvested exactly from `2D3DFusion/mydetector3d` (OpenPCDet lineage), is the
**camera→LiDAR box transform**: KITTI labels are 3-D boxes in the *rectified camera* frame,
so we apply `rect_to_lidar` (via R0_rect + Tr_velo_to_cam) and convert
`[x,y,z,l,h,w,ry]_cam → [x,y,z,dx,dy,dz,heading]_lidar` (bottom-centre → centre, heading
`-(ry+π/2)`).

Layout expected (extracted KITTI object dataset):
    <root>/training/{velodyne,label_2,calib}/<id>.{bin,txt,txt}
    <root>/ImageSets/{train,val}.txt
"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class KittiCalib:
    def __init__(self, path):
        rows = {}
        for line in open(path):
            if ":" in line:
                k, v = line.split(":", 1)
                rows[k.strip()] = np.array(v.split(), np.float32)
        self.P2 = rows["P2"].reshape(3, 4)
        self.R0 = rows["R0_rect"].reshape(3, 3)
        self.V2C = rows["Tr_velo_to_cam"].reshape(3, 4)

    def rect_to_lidar(self, pts):
        """(N,3) rect-camera xyz -> (N,3) LiDAR xyz."""
        hom = np.hstack([pts, np.ones((len(pts), 1), np.float32)])
        R0 = np.eye(4, dtype=np.float32); R0[:3, :3] = self.R0
        V2C = np.eye(4, dtype=np.float32); V2C[:3, :] = self.V2C
        return (hom @ np.linalg.inv(R0 @ V2C).T)[:, :3]


def camera_to_lidar_boxes(cam_boxes, calib):
    """(N,7) [x,y,z,l,h,w,ry]_cam -> (N,7) [x,y,z,dx,dy,dz,heading]_lidar (centre-based)."""
    xyz, l, h, w, r = cam_boxes[:, :3], cam_boxes[:, 3:4], cam_boxes[:, 4:5], cam_boxes[:, 5:6], cam_boxes[:, 6:7]
    xyz_l = calib.rect_to_lidar(xyz)
    xyz_l[:, 2] += h[:, 0] / 2                                      # bottom-centre -> box centre
    return np.concatenate([xyz_l, l, w, h, -(r + np.pi / 2)], axis=1)


def _parse_label(path, classes=("Car",)):
    """-> (M,7) camera boxes [x,y,z,l,h,w,ry] for the requested classes."""
    out = []
    if not os.path.exists(path):
        return np.zeros((0, 7), np.float32)
    for line in open(path):
        f = line.split()
        if f[0] not in classes:
            continue
        h, w, l = float(f[8]), float(f[9]), float(f[10])
        x, y, z, ry = float(f[11]), float(f[12]), float(f[13]), float(f[14])
        out.append([x, y, z, l, h, w, ry])
    return np.array(out, np.float32).reshape(-1, 7)


class KittiCarDataset(Dataset):
    def __init__(self, root, split="train", pc_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 max_frames=None):
        self.train_dir = os.path.join(root, "training")
        ids = open(os.path.join(root, "ImageSets", split + ".txt")).read().split()
        self.ids = ids[:max_frames] if max_frames else ids
        self.pcr = np.array(pc_range, np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        pts = np.fromfile(os.path.join(self.train_dir, "velodyne", fid + ".bin"),
                          np.float32).reshape(-1, 4)
        m = np.all((pts[:, :3] >= self.pcr[:3]) & (pts[:, :3] < self.pcr[3:]), axis=1)
        pts = pts[m]
        calib = KittiCalib(os.path.join(self.train_dir, "calib", fid + ".txt"))
        cam = _parse_label(os.path.join(self.train_dir, "label_2", fid + ".txt"))
        if len(cam):
            gt = camera_to_lidar_boxes(cam, calib)                  # (M,7) lidar
            keep = np.all((gt[:, :2] >= self.pcr[:2]) & (gt[:, :2] < self.pcr[3:5]), axis=1)
            gt = gt[keep]
        else:
            gt = np.zeros((0, 7), np.float32)
        gt = np.concatenate([gt, np.zeros((len(gt), 1), np.float32)], axis=1)   # label col = 0 (Car)
        return {"points": torch.from_numpy(pts), "gt": torch.from_numpy(gt), "id": fid}


def collate(batch):
    return {"points": [b["points"] for b in batch],
            "gt": [b["gt"] for b in batch],
            "id": [b["id"] for b in batch]}


# =========================================================================== #
# sanity: load one frame, print point/box stats
#   python -m ...detection.kitti_dataset --root /mnt/e/Shared/Dataset/Kitti
# =========================================================================== #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/Shared/Dataset/Kitti")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    ds = KittiCarDataset(a.root, split=a.split, max_frames=10)
    print(f"{len(ds)} frames (split={a.split})")
    tot = 0
    for i in range(len(ds)):
        s = ds[i]
        tot += len(s["gt"])
    b = ds[0]
    print(f"frame {b['id']}: points={tuple(b['points'].shape)}  gt_cars={tuple(b['gt'].shape)}")
    if len(b["gt"]):
        g = b["gt"][0]
        print(f"  first Car [x,y,z,dx,dy,dz,head,lbl]: {[round(float(v),2) for v in g]}")
        print(f"  Car size sanity (dx,dy,dz ~ 3.9,1.6,1.5): {[round(float(v),2) for v in g[3:6]]}")
    print(f"  total Cars in first 10 frames: {tot}")
