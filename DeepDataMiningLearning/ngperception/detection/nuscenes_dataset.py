"""
ngperception.detection.nuscenes_dataset
========================================

A basic **nuScenes 3D detection** loader in our unified format (points + gt `(M,8)`
`[x,y,z,dx,dy,dz,heading,label]` in the LIDAR_TOP frame). Uses the nuScenes **devkit** directly
(no OpenPCDet infos): `nusc.get_sample_data(lidar_token)` already returns the sample's
annotation boxes transformed into the LiDAR sensor frame, so the only conversion is
nuScenes `wlh` + orientation-quaternion → our `[dx=l, dy=w, dz=h, heading=yaw]`.

Differences from KITTI to keep in mind: nuScenes is **360°** (range ~[-50,50] both axes), the
lidar has 5 features (x,y,z,intensity,ring — we take the first 4), and cars are a bit larger
(~4.6×1.9×1.7 m). Configure the detector's `pc_range` / `anchor_sizes` accordingly.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# nuScenes category prefix -> detection class index (the standard 10-class detection set).
# Prefix match, so e.g. "vehicle.bus.rigid" -> bus, "human.pedestrian.adult" -> pedestrian.
NUSC_10CLASS = {
    "vehicle.car": 0, "vehicle.truck": 1, "vehicle.construction": 2, "vehicle.bus": 3,
    "vehicle.trailer": 4, "movable_object.barrier": 5, "vehicle.motorcycle": 6,
    "vehicle.bicycle": 7, "human.pedestrian": 8, "movable_object.trafficcone": 9,
}
NUSC_CLASSES = {"vehicle.car": 0}                       # default: car-only (back-compat)


def quaternion_yaw(q):
    """Yaw of a nuScenes box orientation quaternion (rotate x-axis, arctan2)."""
    from pyquaternion import Quaternion
    v = np.dot(Quaternion(q).rotation_matrix, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(v[1], v[0]))


class NuScenesCarDataset(Dataset):
    def __init__(self, dataroot="/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval",
                 version="v1.0-trainval", split="train",
                 pc_range=(-50, -50, -5, 50, 50, 3), class_map=None, max_frames=None, nusc=None,
                 sweeps=1, lidar_cache=None):
        import os
        from nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
        self.nusc = nusc or NuScenes(version=version, dataroot=dataroot, verbose=False)   # reuse devkit
        self.class_map = class_map or NUSC_CLASSES
        self.pcr = np.array(pc_range, np.float32)
        self.sweeps = sweeps                        # 10-sweep aggregation = standard nuScenes density
        self.lidar_cache = lidar_cache
        if lidar_cache:
            os.makedirs(lidar_cache, exist_ok=True)
        want = set(create_splits_scenes()[split])
        toks = []
        for s in self.nusc.sample:
            scene = self.nusc.get("scene", s["scene_token"])["name"]
            if scene in want:
                toks.append(s["token"])
        self.tokens = toks[:max_frames] if max_frames else toks

    def __len__(self):
        return len(self.tokens)

    def _class_of(self, name):
        for pref, idx in self.class_map.items():
            if name.startswith(pref):
                return idx
        return None

    def _load_points(self, token, sample):
        """Single sweep, or `sweeps` motion-compensated sweeps aggregated (LIDAR_TOP frame).
        Multi-sweep is cached per token (aggregation is slow) — denser = distant cars keep points."""
        import os
        if self.sweeps <= 1:
            path = self.nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"])
            return np.fromfile(path, np.float32).reshape(-1, 5)[:, :4]
        cf = os.path.join(self.lidar_cache, f"{token}_sw{self.sweeps}.npy") if self.lidar_cache else None
        if cf and os.path.exists(cf):
            return np.load(cf)
        from nuscenes.utils.data_classes import LidarPointCloud
        pc, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample, "LIDAR_TOP", "LIDAR_TOP",
                                                     nsweeps=self.sweeps)
        pts = pc.points[:4].T.astype(np.float32)                            # (M,4) x,y,z,intensity
        if cf:
            np.save(cf, pts)
        return pts

    def __getitem__(self, i):
        sample = self.nusc.get("sample", self.tokens[i])
        _, boxes, _ = self.nusc.get_sample_data(sample["data"]["LIDAR_TOP"])   # boxes in LiDAR frame
        pts = self._load_points(self.tokens[i], sample)                    # (M,4) x,y,z,intensity
        m = np.all((pts[:, :3] >= self.pcr[:3]) & (pts[:, :3] < self.pcr[3:]), axis=1)
        pts = pts[m]
        gt = []
        for b in boxes:
            cls = self._class_of(b.name)
            if cls is None:
                continue
            x, y, z = b.center
            w, l, h = b.wlh                                                 # nuScenes order: w,l,h
            if not (self.pcr[0] <= x < self.pcr[3] and self.pcr[1] <= y < self.pcr[4]):
                continue
            gt.append([x, y, z, l, w, h, quaternion_yaw(b.orientation), cls])
        gt = np.array(gt, np.float32).reshape(-1, 8)
        return {"points": torch.from_numpy(pts), "gt": torch.from_numpy(gt), "id": self.tokens[i]}


# =========================================================================== #
# sanity: load one frame, check car sizes
#   python -m ...detection.nuscenes_dataset --root <nuscenes>
# =========================================================================== #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/Shared/Dataset/NuScenes/v1.0-trainval")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    ds = NuScenesCarDataset(a.root, split=a.split, max_frames=20)
    print(f"{len(ds)} samples (split={a.split}, first 20 scanned)")
    tot = 0
    for i in range(len(ds)):
        s = ds[i]; tot += len(s["gt"])
        if len(s["gt"]) and i < 3:
            g = s["gt"][0]
            print(f"  {s['id'][:8]}: pts={tuple(s['points'].shape)} cars={len(s['gt'])} "
                  f"car0 dxyz={[round(float(v),2) for v in g[3:6]]} (nuScenes car ~4.6,1.9,1.7)")
    print(f"  total cars in {len(ds)} samples: {tot}")
