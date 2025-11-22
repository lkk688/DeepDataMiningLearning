#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_nuscenes_projection.py

Validate your nuScenes calibration by projecting LiDAR points into camera images.
- Colors points by camera-depth (Turbo colormap) for high contrast
- Sorts by depth so nearer points draw on top
- Robust percentile clipping for visibility
- Saves one image per camera per sample

Usage example:
  python viz_nuscenes_projection.py \
      --dataroot /data/nuscenes \
      --version v1.0-trainval \
      --out-dir viz_out \
      --cams FRONT FRONT_LEFT FRONT_RIGHT \
      --max-count 20 \
      --resize 1408 512 \
      --every-n 5
"""

from __future__ import annotations
import os
import os.path as osp
import argparse
import sys
from typing import Iterator, List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nuScenes devkit
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
except Exception as e:
    print("ERROR: nuScenes devkit is required. pip install nuscenes-devkit")
    raise

# --------------------------- Small utils --------------------------------- #

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _camera_order(nusc: NuScenes) -> List[str]:
    # Standard nuScenes camera names in a stable order.
    all_cams = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    # keep only those present in the dataset version
    have = set()
    for s in nusc.sample:
        have.update(s['data'].keys())
    return [c for c in all_cams if c in have]

def _find_nus_version(dataroot: str, preferred: Optional[str]) -> str:
    candidates = ['v1.0-mini', 'v1.0-trainval', 'v1.0-test']
    if preferred is not None:
        if osp.isdir(osp.join(dataroot, preferred)):
            return preferred
        else:
            print(f"[WARN] Requested version '{preferred}' not found under {dataroot}. Trying to auto-detect.")
    found = [v for v in candidates if osp.isdir(osp.join(dataroot, v))]
    if not found:
        msg = [
            f"nuScenes version folder not found under: {dataroot}",
            f"Requested/guessed: '{preferred}'" if preferred else "Requested/guessed: <none>",
            "Found versions here: none",
            "Fix by either:",
            "  1) moving/symlinking your data so a valid version folder exists, or",
            "  2) passing --version with one of: v1.0-mini, v1.0-trainval, v1.0-test",
        ]
        raise AssertionError("\n".join(msg))
    # prefer trainval if present
    if 'v1.0-trainval' in found:
        return 'v1.0-trainval'
    return found[0]

def _quat_rot(q: Quaternion) -> np.ndarray:
    """Quaternion -> 3x3 rotation matrix (float32)."""
    return q.rotation_matrix.astype(np.float32)

def _Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 from R(3x3), t(3,)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.astype(np.float32)
    return T

def _build_lidar2cam_chain(nusc: NuScenes,
                           sample: dict,
                           cam_token: str,
                           lidar_sd: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct transforms to go from LiDAR frame to camera image:
      lidar2cam_4x4, cam2lidar_4x4, lidar2img_4x4, K4 (3x3 intrinsics padded to 4x4)

    Chain (spaces / times):
      [lidar] --calib_lidar--> [ego @ lidar_t] --ego_pose--> [global]
              --inv ego_pose@cam_t--> [ego @ cam_t] --calib_cam--> [cam]
              --K--> image

    We handle different ego poses (LiDAR vs camera timestamps).
    """
    # LiDAR calibrated + ego pose
    lidar_calib = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    R_lidar = _quat_rot(Quaternion(lidar_calib['rotation']))
    t_lidar = np.array(lidar_calib['translation'], dtype=np.float32)
    T_lidar_to_egoL = _Rt(R_lidar, t_lidar)
    R_egoL = _quat_rot(Quaternion(lidar_ego['rotation']))
    t_egoL = np.array(lidar_ego['translation'], dtype=np.float32)
    T_egoL_to_global = _Rt(R_egoL, t_egoL)

    # Camera calibrated + ego pose
    cam_sd = nusc.get('sample_data', cam_token)
    cam_calib = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ego = nusc.get('ego_pose', cam_sd['ego_pose_token'])
    R_cam = _quat_rot(Quaternion(cam_calib['rotation']))
    t_cam = np.array(cam_calib['translation'], dtype=np.float32)
    T_cam_to_egoC = _Rt(R_cam, t_cam)
    R_egoC = _quat_rot(Quaternion(cam_ego['rotation']))
    t_egoC = np.array(cam_ego['translation'], dtype=np.float32)
    T_egoC_to_global = _Rt(R_egoC, t_egoC)

    # World alignment across timestamps:
    # lidar->egoL->global->egoC->cam
    T_global_to_egoC = np.linalg.inv(T_egoC_to_global)
    T_egoL_to_cam = T_global_to_egoC @ T_egoL_to_global @ T_cam_to_egoC  # actually egoL->cam? (see below)
    # We want lidar->cam:
    # lidar -> egoL (T_lidar_to_egoL)
    # egoL  -> global (T_egoL_to_global)
    # global-> egoC (inv T_egoC_to_global)
    # egoC  -> cam (inv cam->egoC)
    T_egoC_to_cam = np.linalg.inv(T_cam_to_egoC)
    lidar2cam = T_egoC_to_cam @ np.linalg.inv(T_egoC_to_global) @ T_egoL_to_global @ T_lidar_to_egoL

    # Intrinsics
    K = np.array(cam_calib['camera_intrinsic'], dtype=np.float32)  # [3,3]
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K
    # 3x4 proj: K [I|0] @ lidar2cam (4x4) -> we store as 4x4 with K padded
    lidar2img = K4 @ lidar2cam

    # Sanity shapes
    assert lidar2cam.shape == (4, 4)
    assert lidar2img.shape == (4, 4)
    return lidar2cam, np.linalg.inv(lidar2cam), lidar2img, K4

def _load_lidar_xyz(lidar_path: str) -> np.ndarray:
    """
    Load LiDAR .bin via nuScenes helper (x,y,z,intensity,ring). Returns (N,3) xyz.
    """
    pc = LidarPointCloud.from_file(lidar_path)
    # pc.points: [4, N] = x,y,z,intensity by nuScenes convention
    pts = pc.points[:3, :].T.astype(np.float32)  # (N,3)
    return pts

# -------------------- Projection & rendering (depth color) ---------------- #

def project_points_lidar_to_img_with_depth(points_xyz: np.ndarray, P_4x4: np.ndarray):
    """
    Args:
      points_xyz: (N,3)
      P_4x4     : lidar2img 4x4 (K4 @ lidar2cam)

    Returns:
      uv   : (M,2) pixel coords for valid points (z>0 in camera)
      zcam: (M,)   camera-frame depth for valid points
      mask: (N,)   validity mask
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    assert P_4x4.shape == (4, 4)
    N = points_xyz.shape[0]
    homo = np.concatenate([points_xyz, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N,4]
    cam = (P_4x4 @ homo.T).T  # [N,4] in homogeneous image space; depth is cam-Z before K
    # Get camera-Z by undoing K: since P=K4@T, cam coords are (T@X)_[:3]
    # But we don't have T@X from here; we can recover z by dividing before K. Simpler:
    # Build lidar2cam separately if needed, but we already returned z via chain above.
    # Alternative: use depth as cam[:,2] only if P_4x4 == K4 @ lidar2cam and cam[:,2] equals camera-Z * fx? It's not.
    # Safer: require both lidar2img and lidar2cam; here we approximate using w component:
    # Instead we’ll pass in true camera Z separately when available.
    # To keep API, we estimate normalized depth = cam[:,2]; acceptable for coloring. Or handle outside.
    z = cam[:, 2].copy()
    mask = z > 1e-6
    cam = cam[mask]
    uv = cam[:, :2] / cam[:, 2:3]
    zcam = z[mask]
    return uv, zcam, mask

def overlay_depth_points_on_image(
    img: Image.Image,
    uv: np.ndarray,
    zcam: np.ndarray,
    *,
    point_size_px: int | None = None,
    cmap: str = "turbo",
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
    alpha: float = 0.95,
    add_colorbar: bool = False,
) -> Image.Image:
    assert uv.ndim == 2 and uv.shape[1] == 2
    assert zcam.ndim == 1 and uv.shape[0] == zcam.shape[0]
    w, h = img.size

    if zcam.size > 0:
        zmin, zmax = np.percentile(zcam, clip_percentiles)
        zmin = max(1e-3, float(zmin))
        zmax = float(max(zmin + 1e-3, zmax))
        zc = np.clip(zcam, zmin, zmax)
    else:
        zmin, zmax = 1.0, 50.0
        zc = zcam

    # far->near so near points render last (on top)
    order = np.argsort(zc)
    uv = uv[order]
    zc = zc[order]

    if point_size_px is None:
        point_size_px = max(1, int(0.0015 * max(w, h)))  # adaptive

    norm = (zc - zmin) / (zmax - zmin + 1e-12)

    fig_w = w / 100.0
    fig_h = h / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(img)

    if uv.size > 0:
        sc = ax.scatter(uv[:, 0], uv[:, 1],
                        s=point_size_px,
                        c=norm,
                        cmap=cmap,
                        linewidths=0,
                        alpha=alpha)
        if add_colorbar:
            cax = fig.add_axes([0.86, 0.08, 0.02, 0.3])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label("Depth (relative)", rotation=90)

    ax.set_axis_off()
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(vis)

def render_lidar_depth_overlay(
    img: Image.Image,
    points_xyz: np.ndarray,
    lidar2img_4x4: np.ndarray,
    *,
    point_size_px: int | None = None,
    cmap: str = "turbo",
    clip_percentiles=(2.0, 98.0),
    alpha: float = 0.95,
    add_colorbar: bool = False,
) -> Image.Image:
    uv, zcam, mask = project_points_lidar_to_img_with_depth(points_xyz, lidar2img_4x4)
    w, h = img.size
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return overlay_depth_points_on_image(
        img, uv[inside], zcam[inside],
        point_size_px=point_size_px,
        cmap=cmap,
        clip_percentiles=clip_percentiles,
        alpha=alpha,
        add_colorbar=add_colorbar,
    )

# --------------------------- Iterator over samples ----------------------- #

def iter_nuscenes_samples(
    dataroot: str,
    version: Optional[str] = None,
    max_count: int = -1,
    every_n: int = 1,
    cams: Optional[List[str]] = None,
) -> Iterator[Tuple[str, List[Tuple[str, str, np.ndarray]]]]:
    """
    Yields per-sample bundles:

    Returns tuple:
      (lidar_path, cam_items)

    where cam_items is a list of tuples per camera:
      (cam_name, image_path, lidar2img_4x4)

    Shapes & conventions:
      - lidar2img_4x4: float32 [4,4] = K4 @ lidar2cam
      - image path is absolute
      - LiDAR path is absolute pointing at nuScenes LIDAR_TOP .bin

    Notes:
      - We compute transforms carefully across different timestamps (LiDAR vs camera).
      - You can limit to specific cams via --cams; otherwise we use all 6 standard cams present.
    """
    version = _find_nus_version(dataroot, version)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    all_cams = _camera_order(nusc) if cams is None else cams

    count = 0
    for idx, sample in enumerate(nusc.sample):
        if every_n > 1 and (idx % every_n != 0):
            continue
        if 0 <= max_count <= count:
            break

        # LiDAR TOP
        lidar_token = sample['data'].get('LIDAR_TOP', None)
        if lidar_token is None:
            continue
        lidar_sd = nusc.get('sample_data', lidar_token)
        lidar_path = osp.join(dataroot, lidar_sd['filename'])
        if not osp.isfile(lidar_path):
            continue

        cam_items: List[Tuple[str, str, np.ndarray]] = []
        ok = True
        for cam_name in all_cams:
            cam_token = sample['data'].get(cam_name, None)
            if cam_token is None:
                ok = False
                break
            cam_sd = nusc.get('sample_data', cam_token)
            img_path = osp.join(dataroot, cam_sd['filename'])
            if not osp.isfile(img_path):
                ok = False
                break

            lidar2cam, cam2lidar, lidar2img, K4 = _build_lidar2cam_chain(
                nusc, sample, cam_token, lidar_sd
            )
            # Validate shapes
            if lidar2img.shape != (4, 4):
                print(f"[WARN] Bad lidar2img shape {lidar2img.shape} for {cam_name}; skipping sample.")
                ok = False
                break
            cam_items.append((cam_name, img_path, lidar2img))

        if not ok or not cam_items:
            continue

        yield (lidar_path, cam_items)
        count += 1

# --------------------------- Main driver --------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, default="data/nuscenes", help="nuScenes dataroot containing v1.0-*/")
    ap.add_argument("--version", type=str, default=None, help="One of v1.0-mini, v1.0-trainval, v1.0-test (auto-detect if omitted)")
    ap.add_argument("--out-dir", type=str, default="viz_out", help="Output directory for overlays")
    ap.add_argument("--cams", type=str, nargs="*", default=None,
                    help="Subset of cameras to render (e.g., --cams CAM_FRONT CAM_FRONT_LEFT). Defaults to all available.")
    ap.add_argument("--max-count", type=int, default=50, help="Max number of samples to render (-1 = all)")
    ap.add_argument("--every-n", type=int, default=1, help="Stride through samples (e.g., 5 renders every 5th sample)")
    ap.add_argument("--resize", type=int, nargs=2, default=None, metavar=('W','H'),
                    help="Resize output image to WxH (e.g., --resize 1408 512)")
    ap.add_argument("--point-size", type=int, default=None, help="Point size in pixels (auto if omitted)")
    ap.add_argument("--add-colorbar", action="store_true", help="Draw a small colorbar at right")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    it = iter_nuscenes_samples(
        dataroot=args.dataroot,
        version=args.version,
        max_count=args.max_count,
        every_n=args.every_n,
        cams=args.cams
    )

    total = 0
    for lidar_path, cam_items in it:
        try:
            pts = _load_lidar_xyz(lidar_path)  # (N,3)
        except Exception as e:
            print(f"[WARN] Failed to load LiDAR: {lidar_path} ({e})")
            continue

        # basename from token-ish stem
        base = osp.splitext(osp.basename(lidar_path))[0].replace('.', '_')

        for cam_name, img_path, lidar2img in cam_items:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"[WARN] Failed to read image: {img_path} ({e})")
                continue

            if args.resize is not None:
                W, H = int(args.resize[0]), int(args.resize[1])
                img = img.resize((W, H), Image.BILINEAR)

            vis = render_lidar_depth_overlay(
                img=img,
                points_xyz=pts,
                lidar2img_4x4=lidar2img,
                point_size_px=args.point_size,
                cmap="turbo",
                clip_percentiles=(2, 98),
                alpha=0.95,
                add_colorbar=args.add_colorbar
            )

            out_name = f"{base}_{cam_name}_points.jpg"
            out_path = osp.join(args.out_dir, out_name)
            vis.save(out_path, quality=95)
            print(f"[OK] Saved {out_path}")

        total += 1

    print(f"[DONE] Rendered {total} samples to: {args.out_dir}")

if __name__ == "__main__":
    main()