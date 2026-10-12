"""
ngperception.occupancy.cache_sonata_feat
========================================
Cache **frozen Sonata (PTv3) point features** voxelised onto the Occ3D grid — the *point-FM
arm* of the backbone study (the LiDAR-side counterpart to our camera-side FM benchmark).

Why.  Every backbone in our benchmark is camera-side, while the LiDAR branch of the fusion
column is a small 3-D CNN over **raw voxel occupancy**.  **Sonata** [Wu et al., CVPR'25,
arXiv:2503.16429] is a self-supervised point FM that needs **no teacher at all**, so swapping
it in tests whether the label-free ceiling we measured is a *teacher-quality* bound or a
*pretext/transfer* bound.

**Honest caveats (they shape the interpretation).** Sonata is pretrained on **indoor** scans
only (ScanNet/ScanNet++/S3DIS/ARKitScenes/HM3D/Structured3D/ASE) and its input features are
`coord + color`; its published *outdoor* numbers come from **fine-tuning**, not frozen
transfer.  nuScenes LiDAR has no colour, so we feed intensity as greyscale.  A frozen probe
here is therefore an **out-of-domain** test by construction — which is exactly the question
our benchmark asks of every FM, and a negative is a real result.

Output: `<token>.npz` with `idx` (M,3) int16 occupied-voxel indices and `feat` (M,C) float16
PCA-reduced features on the 200x200x16 Occ3D grid — sparse, so ~2 MB/sample instead of 650 MB.

    python -m DeepDataMiningLearning.ngperception.occupancy.cache_sonata_feat \
        --gts <gts> --nusc <nusc> --out <dir> --cap 400 --dim 32
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch

from .datasets import Occ3DNuScenesDataset
from .models.lss_occ import XBOUND, YBOUND, ZBOUND


def _grid(b):
    lo, hi, step = b
    return lo, step, int(round((hi - lo) / step))


def sonata_points(nusc, token, transform, model, device="cuda"):
    """Frozen Sonata features for one keyframe sweep -> (coord_ego (N,3), feat (N,C))."""
    from pyquaternion import Quaternion as Q
    s = nusc.get("sample", token)
    lsd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    raw = np.fromfile(os.path.join(nusc.dataroot, lsd["filename"]), dtype=np.float32).reshape(-1, 5)
    lcs = nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
    ego = raw[:, :3] @ Q(lcs["rotation"]).rotation_matrix.T + np.array(lcs["translation"])
    ego = ego.astype(np.float32)
    inten = np.clip(raw[:, 3:4] / 255.0, 0, 1).astype(np.float32)
    point = {"coord": ego.copy(),
             "color": np.repeat(inten, 3, axis=1) * 255.0,   # no colour on LiDAR -> intensity grey
             "normal": np.zeros_like(ego)}
    point = transform(point)
    for k in point:
        if isinstance(point[k], torch.Tensor):
            point[k] = point[k].to(device, non_blocking=True)
    with torch.inference_mode():
        pt = model(point)
        for _ in range(2):                                    # feature up-cast (as in the demo)
            parent = pt.pop("pooling_parent"); inv = pt.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, pt.feat[inv]], dim=-1); pt = parent
        while "pooling_parent" in pt.keys():
            parent = pt.pop("pooling_parent"); inv = pt.pop("pooling_inverse")
            parent.feat = pt.feat[inv]; pt = parent
        # NB: the default pipeline CenterShifts + GridSamples, so `pt.coord` is NOT the ego
        # frame our Occ3D grid lives in. Scatter features back to the ORIGINAL points via the
        # GridSample inverse and voxelise with the untouched ego coords.
        feat = pt.feat[pt.inverse].float().cpu().numpy()       # (N_orig, C)
        return ego, feat


def voxelize(coord, feat, dim_pca=None):
    """Mean-pool point features into the Occ3D grid -> (idx (M,3) int16, feat (M,C))."""
    x0, xs, nx = _grid(XBOUND); y0, ys, ny = _grid(YBOUND); z0, zs, nz = _grid(ZBOUND)
    ix = np.floor((coord[:, 0] - x0) / xs).astype(np.int64)
    iy = np.floor((coord[:, 1] - y0) / ys).astype(np.int64)
    iz = np.floor((coord[:, 2] - z0) / zs).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz, f = ix[ok], iy[ok], iz[ok], feat[ok]
    if len(f) == 0:
        return np.zeros((0, 3), np.int16), np.zeros((0, f.shape[1]), np.float16)
    flat = (ix * ny + iy) * nz + iz
    uniq, inv = np.unique(flat, return_inverse=True)
    acc = np.zeros((len(uniq), f.shape[1]), np.float32)
    np.add.at(acc, inv, f)
    cnt = np.bincount(inv, minlength=len(uniq))[:, None]
    acc /= np.maximum(cnt, 1)
    uz = uniq % nz; uy = (uniq // nz) % ny; ux = uniq // (nz * ny)
    return np.stack([ux, uy, uz], 1).astype(np.int16), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True); ap.add_argument("--nusc", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--dim", type=int, default=32, help="PCA dim for the cached voxel features")
    ap.add_argument("--fit", type=int, default=30, help="samples used to fit the PCA basis")
    ap.add_argument("--sonata-path", default="/home/lkk688/Developer/DeepDataMiningLearning/Others/sonata")
    args = ap.parse_args()

    from nuscenes import NuScenes
    sys.path.insert(0, args.sonata_path)
    import sonata
    os.makedirs(args.out, exist_ok=True)
    model = sonata.model.load("sonata", repo_id="facebook/sonata").cuda().eval()
    for p in model.parameters():
        p.requires_grad = False
    transform = sonata.transform.default()
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)
    n = min(args.cap, len(occ))

    # ---- pass 1: fit a PCA basis so the cache is small but keeps the dominant variance ----
    pca_f = os.path.join(args.out, "_pca.npz")
    if os.path.isfile(pca_f):
        z = np.load(pca_f); mean, comp = z["mean"], z["comp"]
    else:
        print(f"[sonata] fitting PCA({args.dim}) on {args.fit} samples ...", flush=True)
        buf = []
        for i in range(min(args.fit, n)):
            _, f = voxelize(*sonata_points(nusc, occ[i].sample_token, transform, model))
            if len(f): buf.append(f[np.random.RandomState(i).choice(len(f), min(2000, len(f)), False)])
        X = np.concatenate(buf); mean = X.mean(0)
        # top-k right singular vectors = PCA components
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        comp = Vt[:args.dim]
        np.savez(pca_f, mean=mean, comp=comp)
        print(f"[sonata] PCA basis {comp.shape} from {X.shape[0]} voxels", flush=True)

    done = skip = 0
    for i in range(n):
        tok = occ[i].sample_token
        outp = os.path.join(args.out, tok + ".npz")
        if os.path.isfile(outp):
            skip += 1; continue
        idx, f = voxelize(*sonata_points(nusc, tok, transform, model))
        fr = ((f - mean) @ comp.T).astype(np.float16) if len(f) else np.zeros((0, comp.shape[0]), np.float16)
        np.savez_compressed(outp, idx=idx, feat=fr)
        done += 1
        if done % 50 == 0:
            print(f"  [{i+1}/{n}] cached={done} skip={skip} voxels/sample~{len(idx)}", flush=True)
    print(f"[sonata] done: wrote {done}, skipped {skip} -> {args.out}")


if __name__ == "__main__":
    main()
