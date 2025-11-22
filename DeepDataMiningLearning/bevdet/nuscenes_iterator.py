# nuscenes_iterator.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import os.path as osp
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

# 依赖 nuScenes-devkit
# pip install nuscenes-devkit
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


# --------- 工具函数 ---------

def _find_nus_version(dataroot: str, version: Optional[str]) -> str:
    """确保 version 正确，且目录存在。"""
    valid = ['v1.0-mini', 'v1.0-trainval', 'v1.0-test']
    if version is None:
        # 尝试猜测
        for v in valid:
            if osp.isdir(osp.join(dataroot, v)):
                return v
        msg = [
            f"nuScenes version folder not found under: {dataroot}",
            "Requested/guessed: 'v1.0-trainval'",
            "Found versions here: none",
            "Fix by either:",
            "  1) moving/symlinking your data so a valid version folder exists, or",
            f"  2) explicitly pass version in {valid}"
        ]
        raise AssertionError("\n".join(msg))
    assert version in valid, f"Invalid nuScenes version '{version}', choose from {valid}"
    path = osp.join(dataroot, version)
    assert osp.isdir(path), f"Database version not found: {path}"
    return version


def _intrinsic_4x4(intri_3x3: List[List[float]]) -> np.ndarray:
    K = np.array(intri_3x3, dtype=np.float32)
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K
    return K4


def _se3_to_4x4(q: Quaternion, t: np.ndarray) -> np.ndarray:
    """nuScenes 的外参：sensor->ego 或 ego->global 等，给出 4x4 矩阵。"""
    R = q.rotation_matrix.astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.astype(np.float32)
    return T


def _build_lidar2cam_chain(nusc: NuScenes,
                           sample: dict,
                           cam_token: str,
                           lidar_sd: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算一帧里：lidar->cam, cam->lidar, lidar->img(=K@lidar->cam), cam内参 K(4x4)
    注意：nuScenes 时间戳不同，链式：LidarSensor->Ego(lidar时刻)->Global
                      ->Ego(cam时刻)->CamSensor
    """
    # 相机 sd / cs / ego
    cam_sd = nusc.get('sample_data', cam_token)
    cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ep = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    # LiDAR sd / cs / ego
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    # sensor->ego
    T_ego_lidar = _se3_to_4x4(Quaternion(lidar_cs['rotation']), np.array(lidar_cs['translation']))
    T_ego_cam   = _se3_to_4x4(Quaternion(cam_cs['rotation']),   np.array(cam_cs['translation']))

    # ego(lidar)->global, ego(cam)->global
    T_global_ego_lidar = _se3_to_4x4(Quaternion(lidar_ep['rotation']), np.array(lidar_ep['translation']))
    T_global_ego_cam   = _se3_to_4x4(Quaternion(cam_ep['rotation']),   np.array(cam_ep['translation']))

    # 链：lidar(sensor)->ego(lidar)->global->ego(cam)->cam(sensor)
    T_ego_cam_inv = np.linalg.inv(T_ego_cam)
    T_global_ego_cam_inv = np.linalg.inv(T_global_ego_cam)
    lidar2cam = T_ego_cam_inv @ T_global_ego_cam_inv @ T_global_ego_lidar @ T_ego_lidar

    # 反变换：
    cam2lidar = np.linalg.inv(lidar2cam)

    # 相机内参 4x4
    K4 = _intrinsic_4x4(cam_cs['camera_intrinsic'])

    # lidar->image 齐次：K @ lidar->cam
    lidar2img = K4 @ lidar2cam

    return lidar2cam, cam2lidar, lidar2img, K4


# def _camera_order(nusc: NuScenes) -> List[str]:
#     """固定相机顺序（与训练/配置一致）"""
#     # 常见顺序：['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
#     # 也可按你配置文件的顺序来。这里取 nuScenes 官方常见顺序：
#     return ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

def _camera_order(nusc):
    # Standard nuScenes order used by many repos
    order = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    # Keep only those present in this dataset version
    present = [k for k in order if k in nusc.sample[0]['data']]
    return present

import numpy as np
from pyquaternion import Quaternion

def _SE3(R, t):
    """Create 4x4 from rotation matrix R (3x3) and translation t (3,)."""
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = np.asarray(t, np.float32)
    return T

def _inv(T):
    """Inverse of 4x4 SE(3)."""
    R = T[:3,:3]; t = T[:3,3]
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3,:3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def build_lidar2cam_and_lidar2img(nusc, lidar_sd, cam_sd):
    """
    Returns:
      lidar2cam_4x4, camK4x4, lidar2img_4x4
    All are float32 numpy arrays.
    Uses the CORRECT time-aware chain:
      LIDAR(cs@lidar_time) -> EGO(lidar_time) -> GLOBAL
            -> EGO(cam_time) -> CAM(cs@cam_time) -> image
    """
    # ---- LiDAR calibrated sensor + ego pose (at LiDAR time)
    lidar_cs   = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ep   = nusc.get('ego_pose',          lidar_sd['ego_pose_token'])
    T_lidar_cs = _SE3(Quaternion(lidar_cs['rotation']).rotation_matrix, lidar_cs['translation'])
    T_lidar_ep = _SE3(Quaternion(lidar_ep['rotation']).rotation_matrix, lidar_ep['translation'])

    # ---- Camera calibrated sensor + ego pose (at Camera time)
    cam_cs   = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ep   = nusc.get('ego_pose',          cam_sd['ego_pose_token'])
    T_cam_cs = _SE3(Quaternion(cam_cs['rotation']).rotation_matrix, cam_cs['translation'])
    T_cam_ep = _SE3(Quaternion(cam_ep['rotation']).rotation_matrix, cam_ep['translation'])

    # ---- Camera intrinsics (nuScenes gives 3x3 K)
    K = np.array(cam_cs['camera_intrinsic'], dtype=np.float32)  # [3,3]
    K4 = np.eye(4, dtype=np.float32); K4[:3,:3] = K

    # ---- Chain:
    # LiDAR sensor -> ego(lidar) -> global -> ego(cam) -> camera sensor
    T_lidar_to_cam = _inv(T_cam_cs) @ _inv(T_cam_ep) @ T_lidar_ep @ T_lidar_cs

    # ---- Pixel projection matrix for homogeneous 3D (x_cam,y_cam,z_cam,1):
    # u ~ K * [R|t] * X_lidar
    lidar2img = K4 @ T_lidar_to_cam

    return T_lidar_to_cam.astype(np.float32), K4.astype(np.float32), lidar2img.astype(np.float32)

def scale_lidar2img_for_resize(lidar2img, W0, H0, W, H):
    sx, sy = W / float(W0), H / float(H0)
    P = lidar2img.copy()
    P[0, :] *= sx
    P[1, :] *= sy
    return P
import os.path as osp
from typing import Iterator, List, Dict, Tuple, Optional

import numpy as np
import torch
import warnings
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


# ----------------------------- Helpers -----------------------------

def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 SE(3) from rotation (3x3) and translation (3,)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    """Inverse of 4x4 SE(3)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = (-R.T @ t).astype(np.float32)
    return Ti


def _camera_order(nusc: NuScenes) -> List[str]:
    """
    Canonical nuScenes camera order used by many repos.
    Only keep cameras that exist in this dataset/version.
    """
    order = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    sample0 = nusc.sample[0]
    return [c for c in order if c in sample0['data']]


def _discover_versions(dataroot: str) -> List[str]:
    known = ['v1.0-mini', 'v1.0-trainval', 'v1.0-test']
    return [v for v in known if osp.isdir(osp.join(dataroot, v))]


def _find_nus_version(dataroot: str, version: Optional[str]) -> str:
    """
    Pick a valid nuScenes version folder under dataroot.
    If version is None, prefer 'v1.0-trainval' if present, else 'v1.0-mini'.
    """
    found = _discover_versions(dataroot)
    if not found:
        msg = [
            f"nuScenes version folder not found under: {dataroot}",
            "Requested/guessed: {}".format('v1.0-trainval' if version is None else version),
            "Found versions here: none",
            "Fix by either:",
            "  1) moving/symlinking your data so a valid version folder exists, or",
            "  2) passing --nus-version with one of: v1.0-mini, v1.0-trainval, v1.0-test"
        ]
        raise AssertionError("\n".join(msg))

    if version is None:
        if 'v1.0-trainval' in found:
            return 'v1.0-trainval'
        if 'v1.0-mini' in found:
            return 'v1.0-mini'
        return found[0]

    assert version in found, (
        f"Requested version '{version}' not present under {dataroot}. "
        f"Found: {found}"
    )
    return version


def _build_lidar2cam_and_lidar2img(nusc: NuScenes, lidar_sd: dict, cam_sd: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build time-aware transforms for one LiDAR sample_data and one camera sample_data.

    Chain (correct):
      LIDAR(cs@lidar_time) -> EGO(lidar_time) -> GLOBAL
         -> EGO(cam_time)  -> CAM(cs@cam_time) -> image

    Returns:
      lidar2cam_4x4 (float32),
      K4 (4x4 camera intrinsics as homogeneous, float32),
      lidar2img_4x4 (float32)
    """
    # LiDAR: calibrated sensor + ego pose (at LiDAR time)
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ep = nusc.get('ego_pose',          lidar_sd['ego_pose_token'])
    T_lidar_cs = _se3(Quaternion(lidar_cs['rotation']).rotation_matrix, np.array(lidar_cs['translation']))
    T_lidar_ep = _se3(Quaternion(lidar_ep['rotation']).rotation_matrix, np.array(lidar_ep['translation']))

    # Camera: calibrated sensor + ego pose (at Camera time)
    cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ep = nusc.get('ego_pose',          cam_sd['ego_pose_token'])
    T_cam_cs = _se3(Quaternion(cam_cs['rotation']).rotation_matrix, np.array(cam_cs['translation']))
    T_cam_ep = _se3(Quaternion(cam_ep['rotation']).rotation_matrix, np.array(cam_ep['translation']))

    # Intrinsics K (3x3) -> homogeneous 4x4
    K = np.asarray(cam_cs['camera_intrinsic'], dtype=np.float32)  # [3,3]
    if K.shape != (3, 3):
        raise ValueError(f"camera_intrinsic must be 3x3, got {K.shape}")
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K

    # Time-aware chain (see docstring)
    T_lidar_to_cam = _inv(T_cam_cs) @ _inv(T_cam_ep) @ T_lidar_ep @ T_lidar_cs

    # Final projection: pixel_homo = K4 @ T_lidar_to_cam @ X_lidar_homo
    lidar2img = K4 @ T_lidar_to_cam

    return T_lidar_to_cam.astype(np.float32), K4.astype(np.float32), lidar2img.astype(np.float32)


def _scale_proj_for_resize(lidar2img: np.ndarray, W0: int, H0: int, W: int, H: int) -> np.ndarray:
    """Scale lidar2img (4x4) for image resize W0xH0 -> WxH."""
    if any(v <= 0 for v in (W0, H0, W, H)):
        raise ValueError(f"Invalid sizes: {(W0, H0)} -> {(W, H)}")
    sx, sy = W / float(W0), H / float(H0)
    P = lidar2img.copy()
    P[0, :] *= sx
    P[1, :] *= sy
    return P


# ----------------------- The main iterator ------------------------

def iter_nuscenes_samples(
    dataroot: str,
    version: Optional[str] = None,
    max_count: int = -1,
    resize_to: Optional[Tuple[int, int]] = None,  # (W, H); if None, no scaling applied
    require_all_cams: bool = True,
) -> Iterator[Tuple[str, List[str], Dict[str, torch.Tensor], str]]:
    """
    Iterate over nuScenes samples, yielding tuples:
      (lidar_path, image_paths[N], metainfo(dict of torch Tensors), basename)

    Returned metainfo keys (all CPU tensors):
      - cam2img:        [N, 4, 4]  float32
      - lidar2cam:      [N, 4, 4]  float32
      - cam2lidar:      [N, 4, 4]  float32
      - lidar2img:      [N, 4, 4]  float32 (scaled if resize_to is given)
      - img_aug_matrix: [1, N, 3, 3] float32 (identity if unknown)
      - lidar_aug_matrix: [1, 4, 4]  float32 (identity if unknown)
      - ori_img_hw:     [N, 2]  int64; (H0, W0) for each camera before resize
      - num_pts_feats:  scalar int32 (nuScenes typical = 5)

    Notes / invariants:
      * Camera order is fixed: [FRONT, FR, BR, BACK, BL, FL]  (subset if missing).
      * Transforms are built with the correct time-aware chain (LiDAR time -> GLOBAL -> Cam time).
      * If 'resize_to' is provided, only 'lidar2img' is scaled (cam2img/K4 is kept raw).
      * Filesystem presence of LiDAR & image files is checked; missing ones are skipped.
      * If 'require_all_cams' is True, samples missing ANY ordered camera are skipped.

    Raises clearly explained exceptions when structure invariants are violated.
    """
    version = _find_nus_version(dataroot, version)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    cam_names = _camera_order(nusc)
    if not cam_names:
        raise RuntimeError("No camera modalities found in this nuScenes version.")

    num_yielded = 0

    for sample in nusc.sample:
        if 0 <= max_count == num_yielded:
            break

        # 1) LiDAR sample_data and its file
        lidar_token = sample['data'].get('LIDAR_TOP', None)
        if lidar_token is None:
            if require_all_cams:
                continue
            warnings.warn("Sample has no LIDAR_TOP; skipping.")
            continue

        lidar_sd = nusc.get('sample_data', lidar_token)
        lidar_rel = lidar_sd['filename']  # e.g., "sweeps/LIDAR_TOP/..."
        lidar_path = osp.join(dataroot, lidar_rel)
        if not osp.isfile(lidar_path):
            warnings.warn(f"LiDAR file missing on disk: {lidar_path}; skipping sample.")
            continue

        # 2) Per-camera loop: files and transforms
        img_paths: List[str] = []
        lidar2cam_list, cam2lidar_list = [], []
        lidar2img_list, cam2img_list = [], []
        ori_hw_list = []

        ok = True
        for cam in cam_names:
            cam_token = sample['data'].get(cam, None)
            if cam_token is None:
                msg = f"Sample missing camera {cam}"
                if require_all_cams:
                    warnings.warn(msg + "; skipping sample.")
                    ok = False
                    break
                else:
                    warnings.warn(msg + "; continuing without it.")
                    continue

            cam_sd = nusc.get('sample_data', cam_token)

            # Resolve file and check existence
            img_rel = cam_sd['filename']  # e.g., "samples/CAM_FRONT/..."
            img_path = osp.join(dataroot, img_rel)
            if not osp.isfile(img_path):
                msg = f"Image file missing on disk: {img_path}"
                if require_all_cams:
                    warnings.warn(msg + "; skipping sample.")
                    ok = False
                    break
                else:
                    warnings.warn(msg + "; continuing without it.")
                    continue

            # Original image size from sample_data
            W0, H0 = int(cam_sd['width']), int(cam_sd['height'])
            if W0 <= 0 or H0 <= 0:
                warnings.warn(f"Invalid original size for {img_path}: (W0,H0)=({W0},{H0}); skipping sample.")
                ok = False
                break
            ori_hw_list.append([H0, W0])

            # Build time-aware transforms
            T_l2c, K4, T_l2i = _build_lidar2cam_and_lidar2img(nusc, lidar_sd, cam_sd)

            # Optional resize scaling (only lidar2img)
            if resize_to is not None:
                W, H = int(resize_to[0]), int(resize_to[1])
                if W <= 0 or H <= 0:
                    raise ValueError(f"resize_to must be positive (W,H), got {resize_to}")
                T_l2i = _scale_proj_for_resize(T_l2i, W0, H0, W, H)

            img_paths.append(img_path)
            lidar2cam_list.append(T_l2c)
            cam2lidar_list.append(_inv(T_l2c))
            lidar2img_list.append(T_l2i)
            cam2img_list.append(K4)

        if not ok or len(img_paths) == 0:
            continue

        # 3) Stack and validate shapes
        def _stack(name, arrs, shape_tail):
            if len(arrs) == 0:
                raise RuntimeError(f"{name}: empty list after per-camera loop.")
            A = np.stack(arrs, axis=0).astype(np.float32)
            if A.shape[1:] != shape_tail:
                raise ValueError(f"{name} wrong shape {A.shape}, expected [N,{','.join(map(str, shape_tail))}].")
            return torch.from_numpy(A)

        N = len(img_paths)
        cam2img   = _stack("cam2img",   cam2img_list,   (4, 4))
        lidar2cam = _stack("lidar2cam", lidar2cam_list, (4, 4))
        cam2lidar = _stack("cam2lidar", cam2lidar_list, (4, 4))
        lidar2img = _stack("lidar2img", lidar2img_list, (4, 4))
        ori_img_hw = torch.as_tensor(ori_hw_list, dtype=torch.int64)  # [N,2] (H0, W0)

        # Identity augmentations (if you don't have real augs)
        img_aug_matrix   = torch.eye(3, dtype=torch.float32)[None, None, ...].repeat(1, N, 1, 1)  # [1,N,3,3]
        lidar_aug_matrix = torch.eye(4, dtype=torch.float32)[None, ...]                            # [1,4,4]

        # 4) Pack metainfo (CPU tensors; your pipeline can .to(device) later)
        metainfo: Dict[str, torch.Tensor] = dict(
            cam2img=cam2img,
            lidar2cam=lidar2cam,
            cam2lidar=cam2lidar,
            lidar2img=lidar2img,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
            ori_img_hw=ori_img_hw,
            num_pts_feats=torch.tensor(5, dtype=torch.int32),
        )

        # 5) Final sanity checks
        for k, v in metainfo.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    warnings.warn(f"[NaN/Inf] detected in metainfo[{k}] for sample {sample['token']}.")

        basename = lidar_sd['token']  # a stable unique id
        num_yielded += 1
        yield (lidar_path, img_paths, metainfo, basename)


# =======================
# nuScenes visualization
# =======================
import os, os.path as osp
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt


def _load_nus_lidar(lidar_path: str) -> np.ndarray:
    """
    Load nuScenes LiDAR .bin (.pcd.bin) to (N, 5?) float32.
    We only use the first 3 columns (x,y,z). Falls back to (N,4).
    """
    arr = np.fromfile(lidar_path, dtype=np.float32)
    if arr.size % 5 == 0:
        pts = arr.reshape(-1, 5)
    elif arr.size % 4 == 0:
        pts = arr.reshape(-1, 4)
    else:
        # last resort: try columns = 5 then trim
        n = arr.size // 5
        pts = arr[: n * 5].reshape(-1, 5)
    return pts


def _project_points_xyz(points_xyz: np.ndarray, P_4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points (N,3) in LiDAR frame into image using lidar2img (4x4).
    Returns:
      uv  : (M, 2) pixel coords for valid points (z>0)
      mask: (N,) bool mask of valid points
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3, "points_xyz must be [N,3]"
    assert P_4x4.shape == (4, 4), "lidar2img must be 4x4"

    N = points_xyz.shape[0]
    homo = np.concatenate([points_xyz, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N,4]
    cam = (P_4x4 @ homo.T).T  # [N,4]
    z = cam[:, 2]
    valid = z > 1e-6
    cam = cam[valid]
    uv = cam[:, :2] / cam[:, 2:3]
    return uv, valid


def _overlay_points_on_image(img: Image.Image, uv: np.ndarray, s: int = 2) -> Image.Image:
    """
    Draw projected points as white dots using matplotlib for speed.
    """
    w, h = img.size
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(img)
    if uv.size > 0:
        ax.scatter(uv[:, 0], uv[:, 1], s=s, c='w', linewidths=0, alpha=0.9)
    ax.set_axis_off()
    canvas = fig.canvas
    canvas.draw()
    vis = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(vis)


def _save_image(img: Image.Image, path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    img.save(path, quality=95)


import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1) Project with depth returned -------------------------------------
def project_points_lidar_to_img_with_depth(points_xyz: np.ndarray, P_4x4: np.ndarray):
    """
    Args:
      points_xyz: (N,3) in LiDAR frame
      P_4x4     : lidar2img 4x4
    Returns:
      uv   : (M,2) pixel coords for valid points (z>0)
      zcam: (M,)   camera-frame depth for valid points
      mask: (N,)   validity mask (z>0)
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    assert P_4x4.shape == (4, 4)
    N = points_xyz.shape[0]
    homo = np.concatenate([points_xyz, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N,4]
    cam = (P_4x4 @ homo.T).T  # [N,4]
    z = cam[:, 2]
    mask = z > 1e-6
    cam = cam[mask]
    uv = cam[:, :2] / cam[:, 2:3]
    zcam = z[mask]
    return uv, zcam, mask

# --- 2) Depth-colored overlay (percentile clipping + near-on-top) -------
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
    """
    Draws points colored by depth (z in camera frame).
    - Clips depth to [p2, p98] for contrast
    - Sorts so near points are plotted last (appear on top)
    """
    assert uv.ndim == 2 and uv.shape[1] == 2
    assert zcam.ndim == 1 and uv.shape[0] == zcam.shape[0]
    w, h = img.size

    # Robust clipping range
    if zcam.size > 0:
        zmin, zmax = np.percentile(zcam, clip_percentiles)
        zmin = max(1e-3, float(zmin))
        zmax = float(max(zmin + 1e-3, zmax))
        zc = np.clip(zcam, zmin, zmax)
    else:
        zmin, zmax = 1.0, 50.0
        zc = zcam

    # Sort by depth: far -> near so near points draw on top
    order = np.argsort(zc)  # increasing depth
    uv = uv[order]
    zc = zc[order]

    # Adaptive point size
    if point_size_px is None:
        point_size_px = max(1, int(0.0015 * max(w, h)))  # ~0.15% of longer side

    # Normalize for colormap
    norm = (zc - zmin) / (zmax - zmin + 1e-12)

    # Render with tight layout (no axes)
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
        # Optional colorbar (tiny; anchored)
        if add_colorbar:
            cax = fig.add_axes([0.86, 0.08, 0.02, 0.3])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label("Depth (relative)", rotation=90)

    ax.set_axis_off()
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(vis)

# --- 3) Convenience wrapper that combines both steps --------------------
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
    # Keep only points that land inside the image bounds
    w, h = img.size
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv_in = uv[inside]
    z_in = zcam[inside]
    return overlay_depth_points_on_image(
        img, uv_in, z_in,
        point_size_px=point_size_px,
        cmap=cmap,
        clip_percentiles=clip_percentiles,
        alpha=alpha,
        add_colorbar=add_colorbar,
    )
    
def viz_nuscenes_projection(
    dataroot: str,
    out_dir: str,
    version: Optional[str] = None,
    resize_to: Optional[Tuple[int, int]] = (704, 256),
    max_count: int = 10,
    stride: int = 1,
) -> Dict[str, Any]:
    """
    Quick projection sanity-check for nuScenes:
      * Iterates samples via `iter_nuscenes_samples`
      * Loads LiDAR, projects to each camera (time-aware chain)
      * Saves per-camera overlays to out_dir
      * Reports coverage stats (fraction of points landing in each image)

    Args:
      dataroot  : nuScenes root containing v1.0-*
      out_dir   : where to save overlays
      version   : 'v1.0-trainval' / 'v1.0-mini' / 'v1.0-test' (auto-detect if None)
      resize_to : (W,H) to RESIZE images; must match the iterator’s scaling of lidar2img
                  If None, no resize & iterator should be called with resize_to=None as well.
      max_count : max number of samples to visualize
      stride    : take every `stride`-th sample

    Returns:
      summary dict with basic stats per camera.
    """
    os.makedirs(out_dir, exist_ok=True)

    # IMPORTANT: pass the SAME `resize_to` to iterator so lidar2img matches image size.
    it = iter_nuscenes_samples(
        dataroot=dataroot,
        version=version,
        max_count=-1 if (max_count is None or max_count <= 0) else max_count * stride,
        resize_to=resize_to,
        require_all_cams=True,
    )

    cam_names: Optional[List[str]] = None
    stats: Dict[str, Dict[str, float]] = {}  # cam -> {'in_view_ratio': avg, 'count': n}

    processed = 0
    yielded = 0
    for (lidar_path, img_paths, metainfo, basename) in it:
        processed += 1
        if (processed - 1) % stride != 0:
            continue

        # Initialize camera names on first sample (consistent order)
        if cam_names is None:
            cam_names = [osp.basename(osp.dirname(p)) for p in img_paths]  # e.g., CAM_FRONT
            for c in cam_names:
                stats[c] = dict(in_view_ratio=0.0, count=0.0)

        # Load LiDAR
        pts = _load_nus_lidar(lidar_path)  # (N,5)
        xyz = pts[:, :3].astype(np.float32)

        # Tensors from metainfo (CPU → numpy)
        lidar2img = metainfo['lidar2img'].numpy()  # (6, 4, 4) [Ncam,4,4]

        for cam_idx, img_path in enumerate(img_paths):
            cam_name = cam_names[cam_idx] if cam_names else f"CAM{cam_idx}"
            # Load + optional resize
            img = Image.open(img_path).convert('RGB')
            if resize_to is not None:
                img = img.resize(resize_to, Image.BILINEAR) #(704, 256)

            # # Project
            # uv, valid = _project_points_xyz(xyz, lidar2img[cam_idx])
            # # Keep those inside image bounds
            # w, h = img.size
            # inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            # uv_in = uv[inside]

            # # Update simple coverage stats
            # ratio = float(uv_in.shape[0]) / max(1, xyz.shape[0])
            # stats[cam_name]['in_view_ratio'] += ratio
            # stats[cam_name]['count'] += 1.0

            # # Render & save
            # vis = _overlay_points_on_image(img, uv_in, s=1)
            uv, zcam, mask = project_points_lidar_to_img_with_depth(points_xyz, lidar2img_4x4)
            # Keep only points that land inside the image bounds
            w, h = img.size
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            uv_in = uv[inside]
            z_in = zcam[inside]
            return overlay_depth_points_on_image(
                img, uv_in, z_in,
                point_size_px=point_size_px,
                cmap=cmap,
                clip_percentiles=clip_percentiles,
                alpha=alpha,
                add_colorbar=add_colorbar,
            )
            out_name = f"{basename}_{cam_name}_points.jpg"
            _save_image(vis, osp.join(out_dir, out_name))

        yielded += 1
        if max_count and yielded >= max_count:
            break

    # Average ratios
    for cam_name, d in stats.items():
        c = max(1.0, d['count'])
        d['in_view_ratio'] = d['in_view_ratio'] / c

    print("[viz_nuscenes_projection] Done. Per-camera in-view ratios:")
    for k, v in stats.items():
        print(f"  {k:>16s}: {v['in_view_ratio']:.4f} over {int(v['count'])} samples")

    return dict(
        samples_processed=processed,
        samples_visualized=yielded,
        per_camera_stats=stats,
        resize_to=resize_to,
    )


if __name__ == "__main__":
    # Example:
    # dataroot should contain v1.0-trainval (or mini)
    dataroot = "data/nuscenes"
    out_dir  = "infer_out_viz"

    # IMPORTANT: if your inference resizes images to (704,256),
    # pass the SAME resize_to here so `lidar2img` is scaled in the iterator.
    summary = viz_nuscenes_projection(
        dataroot=dataroot,
        out_dir=out_dir,
        version=None,             # auto-detect if None
        resize_to=(704, 256),     # match your model’s input size
        max_count=10,
        stride=1,
    )