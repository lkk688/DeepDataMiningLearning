"""
ngperception.occupancy.visibility
=================================

Independent per-sensor visibility masks for the Occ3D-nuScenes grid.

**Why this exists.** Occ3D ships `mask_lidar` and `mask_camera`, and it is tempting to
use them to study what each sensor can and cannot see. You cannot: they are **nested by
construction**. Measured over 60 random frames, `mask_camera` is an exact subset of
`mask_lidar` in 60/60 — the ground truth is built from accumulated LiDAR, so camera
visibility is only ever defined inside LiDAR-observed space. Any "sensor complementarity"
computed from them is an artefact of the labelling pipeline, not a property of the sensors.

This module builds the two masks **independently**:

``vis_lidar``
    Ray-cast from the LiDAR sensor origin along every return. Voxels traversed before the
    hit are observed (free), the hit voxel is observed (occupied), everything beyond the
    hit is unknown. Uses only the raw point cloud and the sensor calibration.

``vis_camera``
    For each of the 6 cameras, a z-buffer over the **GT scene geometry**: a voxel is
    camera-visible if it falls in that camera's frustum and no *occupied* voxel lies
    strictly in front of it along the same pixel ray. This is "what a perfect camera at
    this pose would see, given the true scene".

    This uses GT geometry (which is LiDAR-derived) for the *occluder* test, but it never
    consults ``mask_lidar``, so unlike Occ3D's mask it **can** mark a voxel camera-visible
    that the LiDAR never observed — a voxel in a beam gap, or past the last return but
    inside an unoccluded frustum. That asymmetry is the whole point of the measurement.

Both masks treat "visible" as *the sensor has information about this voxel* (ray passage),
which is the same convention Occ3D uses and the one that makes free space meaningful.

Usage
-----
    python -m DeepDataMiningLearning.ngperception.occupancy.visibility \
        --nusc /data/.../v1.0-trainval --gts /data/.../v1.0-trainval/gts \
        --max-samples 200 --out-dir output/visibility

Writes ``meta.json`` (config, seed, git commit, date), ``per_sample.csv`` (one row per
frame) and ``summary.json`` (the aggregate stratification), plus optional ``masks/`` npz.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import subprocess
import sys
from datetime import datetime

import numpy as np

from .geom import CAMS, FREE, GRID_SIZE, PC_RANGE, VOXEL_SIZE, sample_cameras

IMG_W, IMG_H = 1600, 900


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _git(repo: str, *args):
    try:
        return subprocess.check_output(["git", "-C", repo, *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# grid helpers
# --------------------------------------------------------------------------- #

def voxel_centers() -> np.ndarray:
    """(200,200,16,3) ego-frame centre of every voxel."""
    gx, gy, gz = (int(v) for v in GRID_SIZE)
    xs = PC_RANGE[0] + (np.arange(gx) + 0.5) * VOXEL_SIZE
    ys = PC_RANGE[1] + (np.arange(gy) + 0.5) * VOXEL_SIZE
    zs = PC_RANGE[2] + (np.arange(gz) + 0.5) * VOXEL_SIZE
    return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)


def _to_idx(pts: np.ndarray):
    idx = np.floor((pts - PC_RANGE[:3]) / VOXEL_SIZE).astype(np.int32)
    ok = np.all((idx >= 0) & (idx < GRID_SIZE), axis=-1)
    return idx, ok


# --------------------------------------------------------------------------- #
# LiDAR visibility
# --------------------------------------------------------------------------- #

def lidar_visibility(points_ego: np.ndarray, origin_ego: np.ndarray,
                     step_frac: float = 0.5) -> np.ndarray:
    """Mark every voxel a LiDAR ray passes through, up to and including its return.

    points_ego : (N,3) return positions in the ego frame
    origin_ego : (3,)  sensor origin in the ego frame
    """
    vis = np.zeros(tuple(int(v) for v in GRID_SIZE), bool)
    if points_ego.shape[0] == 0:
        return vis

    d = points_ego - origin_ego[None, :]
    rng = np.linalg.norm(d, axis=1)
    keep = rng > 1e-3
    d, rng, pts = d[keep], rng[keep], points_ego[keep]
    unit = d / rng[:, None]

    step = VOXEL_SIZE * step_frac
    n_steps = int(np.ceil(rng.max() / step)) + 1
    # march in chunks so the (N, n_steps, 3) tensor never materialises at once
    chunk = max(1, int(4e6 // max(n_steps, 1)))
    ts = (np.arange(n_steps, dtype=np.float32) * step)[None, :]           # (1,S)
    for s in range(0, pts.shape[0], chunk):
        u = unit[s:s + chunk]
        r = rng[s:s + chunk]
        t = np.minimum(ts, r[:, None])                                     # clamp at the hit
        samp = origin_ego[None, None, :] + u[:, None, :] * t[:, :, None]   # (n,S,3)
        idx, ok = _to_idx(samp.reshape(-1, 3))
        idx = idx[ok]
        if idx.size:
            vis[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return vis


# --------------------------------------------------------------------------- #
# camera visibility
# --------------------------------------------------------------------------- #

def camera_visibility(semantics: np.ndarray, cams: list, pix_step: int = 4, max_range: float = 60.0,
                      step_frac: float = 0.5) -> np.ndarray:
    """Forward ray-cast visibility from the 6 surround cameras.

    Rays are shot from each camera centre through a grid of pixels and marched outward;
    every voxel traversed is marked visible, up to and including the first *occupied* one,
    after which the ray stops. This mirrors the LiDAR convention exactly, so both masks
    mean "the sensor has information about this voxel".

    Two earlier designs were wrong and are recorded so they are not retried:

    * **Z-buffer over projected voxel centres.** A 0.4 m voxel at 6.5 m subtends ~78 px
      while the buffer bins were a few px, so neighbouring surface voxels projected to
      widely spaced bins and far voxels leaked through the gaps.
    * **Backward per-target ray testing** (camera -> target, "is anything occupied in
      between?"). Correct for surfaces perpendicular to the ray, badly wrong at grazing
      incidence: a camera 1.5 m up looking at ground 20 m away sends a ray that passes
      through many *ground* voxels before reaching the target ground voxel, so the target
      is falsely shadowed. On real frames this marked only ~7 % of occupied voxels
      camera-visible. The self-test missed it because its test wall was perpendicular.
    """
    occ = semantics != FREE
    vis = np.zeros(occ.shape, bool)
    step = VOXEL_SIZE * step_frac
    n_steps = int(np.ceil(max_range / step)) + 1
    ts = (np.arange(1, n_steps, dtype=np.float32) * step)[None, :]        # skip t=0

    for cam in cams:
        R, t, K = cam["R"], cam["t"], cam["K"]
        us = np.arange(0, IMG_W, pix_step, dtype=np.float64)
        vs = np.arange(0, IMG_H, pix_step, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        # pixel -> camera-frame direction (z forward), then camera -> ego rotation
        dirs_cam = np.stack([(uu.ravel() - K[0, 2]) / K[0, 0],
                             (vv.ravel() - K[1, 2]) / K[1, 1],
                             np.ones(uu.size)], axis=1)
        dirs = dirs_cam @ R.T                                             # (Nr,3) ego
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        chunk = max(1, int(4e6 // n_steps))
        for s0 in range(0, dirs.shape[0], chunk):
            u = dirs[s0:s0 + chunk]
            samp = t[None, None, :] + u[:, None, :] * ts[0][None, :, None]   # (n_r,S,3)
            idx, ok = _to_idx(samp.reshape(-1, 3))
            n_r = u.shape[0]
            lin = np.full(idx.shape[0], -1, np.int64)
            flat_ok = np.flatnonzero(ok)
            lin[flat_ok] = np.ravel_multi_index(
                (idx[flat_ok, 0], idx[flat_ok, 1], idx[flat_ok, 2]), occ.shape)
            hit = np.zeros(idx.shape[0], bool)
            hit[flat_ok] = occ.reshape(-1)[lin[flat_ok]]
            hit = hit.reshape(n_r, -1)
            # first occupied step per ray; rays that never hit run to max_range
            first = np.where(hit.any(1), hit.argmax(1), hit.shape[1] - 1)
            keep = np.arange(hit.shape[1])[None, :] <= first[:, None]
            sel = lin.reshape(n_r, -1)[keep & (lin.reshape(n_r, -1) >= 0)]
            if sel.size:
                vis.reshape(-1)[sel] = True
    return vis


# --------------------------------------------------------------------------- #
# per-sample driver
# --------------------------------------------------------------------------- #

def analyse_sample(nusc, token: str, gts_root: str) -> dict | None:
    from pyquaternion import Quaternion

    s = nusc.get("sample", token)
    scene = nusc.get("scene", s["scene_token"])["name"]
    f = osp.join(gts_root, scene, token, "labels.npz")
    if not osp.exists(f):
        return None
    d = np.load(f)
    sem = d["semantics"]

    # --- LiDAR: raw returns -> ego frame, plus the sensor origin ---
    lsd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    lcs = nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
    Rl = Quaternion(lcs["rotation"]).rotation_matrix
    tl = np.array(lcs["translation"], np.float64)
    raw = np.fromfile(osp.join(nusc.dataroot, lsd["filename"]),
                      dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    pts_ego = raw @ Rl.T + tl

    vis_l = lidar_visibility(pts_ego, tl)
    vis_c = camera_visibility(sem, sample_cameras(nusc, token))

    occ = sem != FREE
    mc = d["mask_camera"].astype(bool)
    ml = d["mask_lidar"].astype(bool)

    def frac(m, denom):
        return float(m.sum()) / max(int(denom.sum()), 1)

    return dict(
        token=token, scene=scene,
        n_occ=int(occ.sum()),
        # independent masks, over OCCUPIED voxels
        occ_both=frac(occ & vis_l & vis_c, occ),
        occ_cam_only=frac(occ & ~vis_l & vis_c, occ),
        occ_lid_only=frac(occ & vis_l & ~vis_c, occ),
        occ_neither=frac(occ & ~vis_l & ~vis_c, occ),
        # coverage of the whole grid
        grid_lidar=float(vis_l.mean()), grid_camera=float(vis_c.mean()),
        # agreement with Occ3D's own masks -- a correctness check, not a result
        iou_ours_vs_occ3d_lidar=float((vis_l & ml).sum() / max((vis_l | ml).sum(), 1)),
        iou_ours_vs_occ3d_camera=float((vis_c & mc).sum() / max((vis_c | mc).sum(), 1)),
        occ3d_cam_subset_of_lidar=bool((mc & ~ml).sum() == 0),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nusc", default=None, help="nuScenes v1.0-trainval root")
    ap.add_argument("--gts", default=None, help="extracted Occ3D gts/ directory")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic geometric checks and exit")
    ap.add_argument("--save-masks", action="store_true",
                    help="also write per-sample npz masks (large)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    set_global_seed(args.seed)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = osp.join(args.out_dir, f"{stamp}_visibility")
    os.makedirs(out, exist_ok=True)
    if args.save_masks:
        os.makedirs(osp.join(out, "masks"), exist_ok=True)

    repo = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
    with open(osp.join(out, "meta.json"), "w") as fh:
        json.dump(dict(date=datetime.now().isoformat(timespec="seconds"),
                       args=vars(args), seed=args.seed, host=os.uname().nodename,
                       python=sys.executable,
                       git=dict(commit=_git(repo, "rev-parse", "HEAD"),
                                branch=_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                                dirty=bool(_git(repo, "status", "--porcelain")))), fh, indent=2)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc, verbose=False)

    tokens = [s["token"] for s in nusc.sample]
    random.shuffle(tokens)
    rows = []
    for tok in tokens:
        if len(rows) >= args.max_samples:
            break
        r = analyse_sample(nusc, tok, args.gts)
        if r is None:
            continue
        rows.append(r)
        if len(rows) % 20 == 0:
            print(f"[vis] {len(rows)}/{args.max_samples}", flush=True)

    if not rows:
        print("[vis] no samples analysed -- check --gts layout")
        return 1

    cols = list(rows[0].keys())
    with open(osp.join(out, "per_sample.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {k: mean(k) for k in cols if k not in ("token", "scene")}
    summary["n_samples"] = len(rows)
    with open(osp.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== independent visibility, {len(rows)} frames ===")
    print(f"{'occupied voxels seen by':32s}{'share':>8s}")
    for k, lbl in (("occ_both", "LiDAR + camera"),
                   ("occ_cam_only", "camera only"),
                   ("occ_lid_only", "LiDAR only"),
                   ("occ_neither", "NEITHER (fully occluded)")):
        print(f"{lbl:32s}{summary[k] * 100:7.2f}%")
    print(f"\ngrid coverage: lidar {summary['grid_lidar']*100:.1f}%  "
          f"camera {summary['grid_camera']*100:.1f}%")
    print("correctness checks (not results):")
    print(f"  IoU(ours_lidar, Occ3D mask_lidar)   = {summary['iou_ours_vs_occ3d_lidar']:.3f}")
    print(f"  IoU(ours_camera, Occ3D mask_camera) = {summary['iou_ours_vs_occ3d_camera']:.3f}")
    print(f"  Occ3D mask_camera subset of mask_lidar in "
          f"{summary['occ3d_cam_subset_of_lidar']*100:.0f}% of frames (expected 100)")
    print(f"\nwritten to {out}")
    return 0



# --------------------------------------------------------------------------- #
# self-test -- validates against the DEFINITION, not against Occ3D's masks
# --------------------------------------------------------------------------- #

def self_test() -> int:
    """Synthetic-scene checks. Occ3D's own masks are the wrong reference here: they are
    built from *accumulated* LiDAR (our single-sweep recall rises 0.45 -> 0.72 as we
    accumulate 1 -> 40 sweeps) and they appear to use a narrower notion of "observed"
    than ray passage (our precision against them sits at 0.38-0.47 regardless). Those are
    definitional gaps, so agreement with them cannot validate this code. These tests check
    the property we actually claim: a voxel is visible iff a ray reaches it unobstructed.
    """
    gx, gy, gz = (int(v) for v in GRID_SIZE)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    # ---- LiDAR: one ray straight down +x, hitting a wall at x = +10 m ----
    origin = np.array([0.0, 0.0, 0.0])
    hit = np.array([[10.0, 0.0, 0.0]])
    vis = lidar_visibility(hit, origin)
    ix_of = lambda x: int(np.floor((x - PC_RANGE[0]) / VOXEL_SIZE))
    iy0, iz0 = int(np.floor((0.0 - PC_RANGE[1]) / VOXEL_SIZE)), int(np.floor((0.0 - PC_RANGE[2]) / VOXEL_SIZE))
    check("lidar: voxel before the return is observed", vis[ix_of(5.0), iy0, iz0])
    check("lidar: the return voxel itself is observed", vis[ix_of(9.9), iy0, iz0])
    check("lidar: voxel BEHIND the return is unknown", not vis[ix_of(12.0), iy0, iz0])
    check("lidar: voxel off the ray is unknown", not vis[ix_of(5.0), iy0 + 10, iz0])

    # ---- camera: a wall in front of CAM_FRONT must shadow what is behind it ----
    # R is camera->ego (same convention as geom.sample_cameras). nuScenes camera axes are
    # x right / y down / z forward, so for a forward-looking camera on the ego (x fwd,
    # y left, z up) the columns are the ego-frame images of the camera basis vectors:
    #   cam x (right)   -> ego -y
    #   cam y (down)    -> ego -z
    #   cam z (forward) -> ego +x
    centers = voxel_centers()
    sem = np.full((gx, gy, gz), FREE, np.uint8)
    wall_x = ix_of(8.0)
    sem[wall_x, :, :] = 1                                   # occupied slab at x = +8 m
    R_c2e = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])
    cam = [dict(name="CAM_FRONT", R=R_c2e, t=np.array([1.7, 0.0, 1.5]),
                K=np.array([[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1.0]]))]
    izc = int(np.floor((1.4 - PC_RANGE[2]) / VOXEL_SIZE))
    vc = camera_visibility(sem, cam)
    check("camera: voxel in front of the wall is visible", vc[ix_of(5.0), iy0, izc])
    check("camera: the wall itself is visible", vc[wall_x, iy0, izc])
    check("camera: voxel behind the wall is shadowed", not vc[ix_of(12.0), iy0, izc])
    check("camera: voxel behind the ego is not visible to a forward cam",
          not vc[ix_of(-10.0), iy0, izc])

    # ---- GRAZING SURFACE: a flat ground slab must be visible far down-range ----
    # This is the case backward per-target ray testing gets wrong: the ray to a distant
    # ground voxel passes through many nearer *ground* voxels, which falsely shadow it.
    sem_g = np.full((gx, gy, gz), FREE, np.uint8)
    izg = int(np.floor((-0.2 - PC_RANGE[2]) / VOXEL_SIZE))
    sem_g[:, :, izg] = 11                                    # drivable-surface slab
    vg = camera_visibility(sem_g, cam)
    check("camera: near ground is visible", vg[ix_of(6.0), iy0, izg])
    check("camera: ground 20 m down-range is visible (grazing)", vg[ix_of(20.0), iy0, izg])
    check("camera: ground behind the ego is not visible to a forward cam",
          not vg[ix_of(-15.0), iy0, izg])

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
