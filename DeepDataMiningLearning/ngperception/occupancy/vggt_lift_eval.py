"""
ngperception.occupancy.vggt_lift_eval
======================================
**Training-free VGGT geometric-lift ablation** — the first VGGT direction probe.

Does a frozen VGGT (Visual Geometry Grounded Transformer, `facebook/VGGT-1B`) give better
*camera geometry* for occupancy than our current camera lift? We run VGGT on the 6 surround
cameras, take its dense per-camera metric depth, back-project with the KNOWN nuScenes
intrinsics/extrinsics into the ego voxel grid, and score class-agnostic **geometric IoU**
(occupied-vs-free) against the Occ3D-nuScenes GT on the official val scenes. No training.

This isolates VGGT's geometry from any learned semantics, so it is directly comparable to the
other *geometry-only* baselines we already measured:
    Depth-Anything mono depth-lift  geo-IoU 0.093   (our current training-free camera geometry)
    LiDAR single-sweep oracle       geo-IoU 0.167
    DINOv2 LSS (trained, leaked)     geo-IoU 0.669   (upper bound, uses learned depth+semantics)
If VGGT (frozen, no training) clears the Depth-Anything 0.093 / LiDAR 0.167 band, that validates
VGGT as the camera-path fix before any heavy trained integration.

VGGT depth is metric-ish but per-scene scale can drift; we report BOTH raw and a single global
LiDAR-median scale-aligned number (project one LiDAR sweep, s = median(lidar_d / vggt_d)).

Run:
    python -m DeepDataMiningLearning.ngperception.occupancy.vggt_lift_eval \
        --gts <gts> --nusc <nuscenes> --n 50 \
        --vggt-path /data/rnd-liu/Others/VGGT-Det-CVPR2026
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch

from .geom import backproject, to_voxels, PC_RANGE, VOXEL_SIZE, GRID_SIZE
from .datasets import Occ3DNuScenesDataset
from .evaluator import OccupancyEvaluator, FREE

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def occupied_grid(points_ego: np.ndarray) -> np.ndarray:
    """Ego points -> (200,200,16) grid, occupied voxels = class 0, else free(17)."""
    grid = np.full(tuple(GRID_SIZE), FREE, np.uint8)
    idx, _ = to_voxels(points_ego)
    if idx.shape[0]:
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 0
    return grid


def lidar_points_ego(nusc, sample):
    """One LIDAR_TOP sweep -> (M,3) ego-frame points (for scale alignment only)."""
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    lsd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lsd["filename"]))
    lcs = nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
    pc.rotate(Quaternion(lcs["rotation"]).rotation_matrix)
    pc.translate(np.array(lcs["translation"]))                       # lidar -> ego
    return pc.points[:3].T.astype(np.float32)


def vggt_scale(depth, Ks, Rs, ts, pts_ego, Hv, Wv):
    """Global scale s = median(lidar_depth / vggt_depth) over LiDAR pts that land on valid pixels."""
    num, den = [], []
    for c in range(len(CAMS)):
        cam = (pts_ego - ts[c]) @ Rs[c]                              # ego -> cam (R^T@(p-t))
        z = cam[:, 2]
        front = z > 0.5
        cam = cam[front]
        if cam.shape[0] == 0:
            continue
        uvw = cam @ Ks[c].T
        u = np.round(uvw[:, 0] / uvw[:, 2]).astype(int)
        v = np.round(uvw[:, 1] / uvw[:, 2]).astype(int)
        ld = cam[:, 2]
        inb = (u >= 0) & (u < Wv) & (v >= 0) & (v < Hv)
        u, v, ld = u[inb], v[inb], ld[inb]
        vd = depth[c][v, u]
        ok = vd > 0.3
        num.append(ld[ok]); den.append(vd[ok])
    if not num:
        return 1.0
    num = np.concatenate(num); den = np.concatenate(den)
    return float(np.median(num / np.maximum(den, 1e-3))) if num.size else 1.0


def main():
    ap = argparse.ArgumentParser(description="Training-free VGGT geometric-lift occupancy ablation.")
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--n", type=int, default=50, help="number of val frames")
    ap.add_argument("--stride", type=int, default=2, help="pixel subsample for backproject")
    ap.add_argument("--Wv", type=int, default=518, help="VGGT input width (÷14)")
    ap.add_argument("--Hv", type=int, default=294, help="VGGT input height (÷14)")
    ap.add_argument("--vggt-path", default="/data/rnd-liu/Others/VGGT-Det-CVPR2026")
    args = ap.parse_args()

    from PIL import Image
    from pyquaternion import Quaternion
    from nuscenes import NuScenes
    from nuscenes.utils import splits

    sys.path.insert(0, args.vggt_path)
    from vggt.models.vggt import VGGT
    dev = "cuda"
    print(f"[vggt-lift] loading facebook/VGGT-1B ...", flush=True)
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    for p in vggt.parameters():
        p.requires_grad = False

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    ds = Occ3DNuScenesDataset(args.gts, scenes=sorted(splits.val))
    n = min(args.n, len(ds))
    print(f"[vggt-lift] {n} official-val frames | VGGT input {args.Wv}x{args.Hv} | stride {args.stride}",
          flush=True)

    ev_raw, ev_align = OccupancyEvaluator(), OccupancyEvaluator()
    scales = []
    for i in range(n):
        s = ds[i]
        sample = nusc.get("sample", s.sample_token)
        imgs, Ks, Rs, ts = [], [], [], []
        for cam in CAMS:
            sd = nusc.get("sample_data", sample["data"][cam])
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            img = Image.open(os.path.join(nusc.dataroot, sd["filename"])).convert("RGB")
            ow, oh = img.size
            sx, sy = args.Wv / ow, args.Hv / oh
            imgs.append(np.asarray(img.resize((args.Wv, args.Hv)), np.float32) / 255.0)
            K = np.array(cs["camera_intrinsic"], np.float32)
            Ks.append((np.diag([sx, sy, 1.0]).astype(np.float32) @ K))
            Rs.append(Quaternion(cs["rotation"]).rotation_matrix.astype(np.float32))
            ts.append(np.array(cs["translation"], np.float32))

        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].to(dev)   # [1,6,3,H,W]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = vggt(x)
        depth = out["depth"][0, ..., 0].float().cpu().numpy()                    # (6,Hv,Wv)

        pts_ego = lidar_points_ego(nusc, sample)
        s_scale = vggt_scale(depth, Ks, Rs, ts, pts_ego, args.Hv, args.Wv)
        scales.append(s_scale)

        raw, align = [], []
        for c in range(len(CAMS)):
            campts, _ = backproject(depth[c], Ks[c], stride=args.stride)          # (M,3) cam frame
            raw.append(campts @ Rs[c].T + ts[c])                                  # cam -> ego
            campts_a, _ = backproject(depth[c] * s_scale, Ks[c], stride=args.stride)
            align.append(campts_a @ Rs[c].T + ts[c])
        ev_raw.add(occupied_grid(np.concatenate(raw)), s.semantics, s.mask_camera)
        ev_align.add(occupied_grid(np.concatenate(align)), s.semantics, s.mask_camera)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] scale~{np.median(scales):.2f} "
                  f"geo_raw={ev_raw.summarize(False)['geo_IoU']:.3f} "
                  f"geo_align={ev_align.summarize(False)['geo_IoU']:.3f}", flush=True)

    print("\n=== VGGT training-free geometric-lift (official val) ===")
    print(f"median depth scale (LiDAR-aligned): {np.median(scales):.3f}")
    r = ev_raw.summarize(False)["geo_IoU"]
    a = ev_align.summarize(False)["geo_IoU"]
    print(f"geo-IoU  raw VGGT depth      : {r:.3f}")
    print(f"geo-IoU  LiDAR scale-aligned : {a:.3f}")
    print("--- reference (geometry-only) ---")
    print("Depth-Anything mono lift : 0.093   LiDAR oracle : 0.167   DINOv2 trained(leaked): 0.669")


if __name__ == "__main__":
    main()
