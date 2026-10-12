"""
ngperception.occupancy.cache_ldcm_depth
=======================================
Precompute a **frozen LDCM metric-depth prior** for the LSS occupancy lift — the
*depth-FM arm* of the backbone study (companion to `cache_vggt_depth.py`).

Why a second depth-FM arm.  Our VGGT depth-prior ablation (#2/#3) was a **wash**, and the
mechanism we gave was *redundancy*: a learned depth head **plus LiDAR depth supervision**
already carries that signal.  **LDCM** [Yu et al., ICLR'26, arXiv:2605.30115] differs *in
kind* — it is **metric** and **conditioned on the sparse observations** (it takes the
projected LiDAR as an input), so the regimes where it should pay are the ones the VGGT null
never tested: **sparse / low-beam** LiDAR.  Hence `--beams`.

Ladder (`--variant`), which decomposes *where* any gain comes from — LDCM's own `infer`
exposes all three:
  * `mono_depth`   — the frozen MoGe monocular prior alone (does **not** see the sparse depth)
  * `coarse_depth` — + **Poisson gradient-field completion** (this is the step that injects
                     the sparse LiDAR and makes the prior metric); no refinement network
  * `depth_pred`   — the full LDCM (DINOv2 dual-encoder refinement on top)

Output format is byte-identical to `cache_vggt_depth.py` (`<token>.npy`, `(6,fH,fW)` float16,
block-min pooled = nearest visible surface per patch), so it drops into the SAME
`train_lss.py --vggt-depth-cache` slot — a controlled same-slot A/B.  Use
`--depth-prior-metric` when training on these (LDCM is already metric; VGGT is up-to-scale).

    python -m DeepDataMiningLearning.ngperception.occupancy.cache_ldcm_depth \
        --gts <gts> --nusc <nuscenes> --out <cache_dir> --cap 2100 \
        --variant depth_pred --beams 0 --ldcm-path <repo>/Others/LDCM
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from .geom import CAMS
from .datasets import Occ3DNuScenesDataset

VARIANTS = ("depth_pred", "coarse_depth", "mono_depth")


def sparse_lidar_depth(nusc, sample, cam_sd, Q, H, W, beams=0):
    """Project the LiDAR keyframe sweep into one camera -> sparse **metric** depth (H,W).

    Mirrors `datasets_train._lidar_depth`'s geometry exactly, but keeps *metres* (no binning)
    because that is what LDCM consumes.  `beams>0` keeps only that many of the 32 rings
    (uniform stride) to emulate a lower-beam LiDAR.
    """
    lsd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    raw = np.fromfile(os.path.join(nusc.dataroot, lsd["filename"]), dtype=np.float32).reshape(-1, 5)
    if beams and beams < 32:
        keep = (raw[:, 4].astype(int) % (32 // beams)) == 0        # uniform ring decimation
        raw = raw[keep]
    pts = raw[:, :3]
    lcs = nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
    ego = pts @ Q(lcs["rotation"]).rotation_matrix.T + np.array(lcs["translation"])   # lidar->ego

    cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    R_c2e = Q(cs["rotation"]).rotation_matrix.astype(np.float32)
    t_c2e = np.array(cs["translation"], np.float32)
    K = np.array(cs["camera_intrinsic"], np.float32)
    cam = (ego - t_c2e) @ R_c2e                                    # ego -> cam
    cam = cam[cam[:, 2] > 0.5]
    uvw = cam @ K.T
    ow, oh = cam_sd["width"], cam_sd["height"]
    u = (uvw[:, 0] / uvw[:, 2]) * (W / ow)
    v = (uvw[:, 1] / uvw[:, 2]) * (H / oh)
    d = cam[:, 2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d = u[inb].astype(int), v[inb].astype(int), d[inb]
    dm = np.zeros((H, W), np.float32)
    order = np.argsort(-d)                                         # nearest wins on collision
    dm[v[order], u[order]] = d[order]
    return dm


def main():
    ap = argparse.ArgumentParser(description="Cache a frozen LDCM metric-depth prior for the lift.")
    ap.add_argument("--gts", required=True)
    ap.add_argument("--nusc", required=True)
    ap.add_argument("--out", required=True, help="cache dir for <token>.npy (6,fH,fW)")
    ap.add_argument("--cap", type=int, default=2100)
    ap.add_argument("--H", type=int, default=252)
    ap.add_argument("--W", type=int, default=700)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--variant", choices=VARIANTS, default="depth_pred")
    ap.add_argument("--beams", type=int, default=0,
                    help="0 = all 32 rings; else keep N rings (16/8/4) to emulate low-beam LiDAR")
    ap.add_argument("--ldcm-path", default="/home/lkk688/Developer/DeepDataMiningLearning/Others/LDCM")
    ap.add_argument("--moge", default="Ruicheng/moge-2-vits-normal")
    args = ap.parse_args()

    from PIL import Image
    from nuscenes import NuScenes
    from pyquaternion import Quaternion as Q
    fH, fW = args.H // args.patch, args.W // args.patch
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, args.ldcm_path)
    from ldcm import LDCMModel
    print(f"[cache] loading LDCM (variant={args.variant}, beams={args.beams or 32}) ...", flush=True)
    model = LDCMModel.from_pretrained("pkqbajng/LDCM", moge_path=args.moge).cuda().eval()
    for p in model.parameters():
        p.requires_grad = False

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)               # same order train_lss sees
    n = min(args.cap, len(occ))
    print(f"[cache] {n} samples -> {args.out} | input {args.W}x{args.H} -> grid {fH}x{fW}", flush=True)

    done = skip = 0
    for i in range(n):
        s = occ[i]
        outp = os.path.join(args.out, s.sample_token + ".npy")
        if os.path.isfile(outp):
            skip += 1
            continue
        sample = nusc.get("sample", s.sample_token)
        imgs, priors = [], []
        for cam in CAMS:
            sd = nusc.get("sample_data", sample["data"][cam])
            img = Image.open(os.path.join(nusc.dataroot, sd["filename"])).convert("RGB")
            imgs.append(np.asarray(img.resize((args.W, args.H)), np.float32) / 255.0)
            priors.append(sparse_lidar_depth(nusc, sample, sd, Q, args.H, args.W, args.beams))
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).cuda()          # (6,3,H,W) in [0,1]
        pr = torch.from_numpy(np.stack(priors))[:, None].cuda()                  # (6,1,H,W) metres, 0=miss
        with torch.inference_mode():
            out = model.infer(x, pr)
        d = out[args.variant]
        d = d[:, 0] if d.dim() == 4 else d                                       # (6,H,W)
        if d.shape[-2:] != (args.H, args.W):
            d = F.interpolate(d[:, None].float(), size=(args.H, args.W),
                              mode="bilinear", align_corners=False)[:, 0]
        # block-min downsample = nearest visible surface per patch (matches cache_vggt_depth)
        d = d.float().view(6, fH, args.patch, fW, args.patch).amin(dim=(2, 4))   # (6,fH,fW)
        np.save(outp, d.cpu().numpy().astype(np.float16))
        done += 1
        if done % 50 == 0:
            print(f"  [{i+1}/{n}] cached={done} skip={skip}", flush=True)
    print(f"[cache] done: wrote {done}, skipped {skip} existing -> {args.out}")


if __name__ == "__main__":
    main()
