#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal MMDet3D inference on nuScenes files (no dataset wrappers).

- Uses a robust nuScenes iterator that returns BEVFusion-friendly metainfo:
  cam2img, lidar2cam, cam2lidar, lidar2img -> [N,4,4]
  img_aug_matrix -> [N,3,4]
  lidar_aug_matrix -> [4,4]
- Prepares batch_inputs_dict (imgs [B,N,3,256,704], points list length B)
  and Det3DDataSample with .metainfo correctly set.
- Works for BEVFusion-based models; also fine for other detectors as long as
  they implement .predict(batch_inputs_dict, data_samples).

Requirements: torch, numpy, pillow, nuscenes-devkit, mmcv, mmengine, mmdet3d
"""

import os
import os.path as osp
import time
import argparse
from typing import List, Tuple, Dict, Iterator, Optional

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
# --- nuScenes ---
from nuscenes.nuscenes import NuScenes

# --- MMDet3D core imports (no Dataset/Runner) ---
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
import os
import os.path as osp
from typing import Iterator, Tuple, List, Dict, Optional

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


# ==========================
# Utilities
# ==========================

def set_determinism(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_tensor(x, dtype=torch.float32, device=None):
    t = torch.as_tensor(x, dtype=dtype)
    return t.to(device) if device is not None else t




def _mat44(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from rotation (3x3) and translation (3,)."""
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R.astype(np.float32)
    M[:3, 3] = t.astype(np.float32)
    return M


def _T_from_pose(rotation: List[float], translation: List[float]) -> np.ndarray:
    """
    Make a 4x4 pose from NuScenes dict fields.
    - rotation: w, x, y, z (quaternion, NuScenes format)
    - translation: [x, y, z]
    Returns T_world_from_frame (i.e., transforms from the local 'frame' to world).
    """
    q = Quaternion(rotation)  # w, x, y, z
    R = q.rotation_matrix.astype(np.float32)  # (3,3)
    t = np.array(translation, dtype=np.float32)  # (3,)
    return _mat44(R, t)  # world_from_frame


def _find_nus_version(dataroot: str, version: Optional[str]) -> str:
    """
    Resolve a valid nuScenes 'version' directory under dataroot.
    - If 'version' is provided, validate it exists.
    - Else try common defaults in priority order.
    Raises AssertionError with a friendly message if not found.
    """
    candidates = [v for v in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test'] if osp.isdir(osp.join(dataroot, v))]
    if version is not None:
        path = osp.join(dataroot, version)
        assert osp.isdir(path), (
            f"nuScenes version folder not found under: {dataroot}\n"
            f"Requested version: '{version}'\n"
            f"Found versions here: {', '.join(candidates) if candidates else 'none'}"
        )
        return version
    # Auto-pick
    assert len(candidates) > 0, (
        f"nuScenes version folder not found under: {dataroot}\n"
        f"Requested/guessed: 'v1.0-trainval'\n"
        f"Found versions here: none\n"
        f"Fix by either:\n"
        f"  1) moving/symlinking your data so a valid version folder exists, or\n"
        f"  2) passing --nus-version with one of: v1.0-mini, v1.0-trainval, v1.0-test"
    )
    # Prefer trainval > mini > test
    for v in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test']:
        if v in candidates:
            return v
    return candidates[0]


def _camera_order(nusc: NuScenes) -> List[str]:
    """
    Standard camera order for nuScenes 6 cams:
      ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    If some scenes/datasets differ, this function can be adapted as needed.
    """
    # Keep canonical order (matches many open-source configs)
    return [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ]

def quat_to_rot(qwxyz):
    w, x, y, z = qwxyz
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    return R


def _pose_to_T(ego) -> np.ndarray:
    R = quat_to_rot(ego['rotation'])
    t = np.array(ego['translation'], dtype=np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, :3] = R; T[:3, 3] = t
    return T


def _calib_to_T(calib) -> np.ndarray:
    R = quat_to_rot(calib['rotation'])
    t = np.array(calib['translation'], dtype=np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, :3] = R; T[:3, 3] = t
    return T

def _build_lidar2cam_chain(nusc: NuScenes, sample: dict, cam_token: str, lidar_sd: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute transforms needed by BEVFusion:

    Returns (all float32 numpy):
      - T_lidar2cam:  (4,4), camera frame ← lidar frame
      - T_cam2lidar:  (4,4), lidar frame  ← camera frame
      - T_lidar2img:  (4,4), projection matrix (K in top-left 3x3) * T_lidar2cam
      - K4:           (4,4), camera intrinsics stored as a 4x4 with K in top-left and K4[3,3]=1

    Construction:
    1) For each sensor (lidar & camera), global poses:
         T_world_from_lidar  = T_world_from_ego(lidar)  @ T_ego_from_lidar
         T_world_from_cam    = T_world_from_ego(cam)    @ T_ego_from_cam
       where:
         T_world_from_ego(...)  from ego_pose
         T_ego_from_sensor      from calibrated_sensor
    2) Lidar → Cam:
         T_cam_from_lidar = (T_world_from_cam)^(-1) @ T_world_from_lidar
    3) Intrinsics K:
         K from calibrated_sensor (fx, fy, cx, cy) → 3x3
         Store as 4x4 K4 (K in 0:3,0:3; K4[3,3]=1)
    4) Lidar → Image:
         T_lidar2img = K4 @ T_lidar2cam  (compatible with 4x4 storage; BEVFusion uses [:3] parts as needed)
    """
    # Camera sample_data & calibrations
    cam_sd = nusc.get('sample_data', cam_token)
    cam_calib = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ego   = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    # Lidar calibrations
    lidar_calib = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ego   = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    # Poses: world_from_sensor = world_from_ego @ ego_from_sensor
    T_world_from_cam   = _T_from_pose(cam_ego['rotation'],   cam_ego['translation']) @ \
                         _T_from_pose(cam_calib['rotation'], cam_calib['translation'])
    T_world_from_lidar = _T_from_pose(lidar_ego['rotation'],   lidar_ego['translation']) @ \
                         _T_from_pose(lidar_calib['rotation'], lidar_calib['translation'])

    # T_cam_from_lidar = (world_from_cam)^-1 @ (world_from_lidar)
    T_cam_from_world  = np.linalg.inv(T_world_from_cam)
    T_lidar2cam       = (T_cam_from_world @ T_world_from_lidar).astype(np.float32)
    T_cam2lidar       = np.linalg.inv(T_lidar2cam).astype(np.float32)

    # Intrinsics K (3x3)
    fx, fy = cam_calib['camera_intrinsic'][0][0], cam_calib['camera_intrinsic'][1][1]
    cx, cy = cam_calib['camera_intrinsic'][0][2], cam_calib['camera_intrinsic'][1][2]
    K      = np.array([[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

    # Store intrinsics in a (4,4) for consistency with many BEVFusion configs
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K

    # Lidar to image (stored as a 4x4 “projection-like” matrix)
    # BEVFusion typically uses the top 3 rows of (K4 @ T_lidar2cam).
    T_lidar2img = (K4 @ T_lidar2cam).astype(np.float32)

    return T_lidar2cam, T_cam2lidar, T_lidar2img, K4


def iter_nuscenes_samples(
    dataroot: str,
    version: Optional[str] = None,
    max_count: int = -1,
) -> Iterator[Tuple[str, List[str], Dict[str, np.ndarray], str]]:
    """
    Iterate nuScenes samples and yield:
        (lidar_path, image_paths, metainfo, basename)

    metainfo (all **CPU numpy.float32**, not torch tensors):

      Required keys & shapes for BEVFusion:

      - 'cam2img'     : (N, 4, 4)
            Camera intrinsics per camera as 4x4 (K in the top-left 3x3, [3,3]=1).
            We provide 4x4 to match many repos; only the top 3x3 K is used in math.

      - 'lidar2img'   : (N, 4, 4)
            Per camera projection-like matrix: K4 @ lidar2cam.
            BEVFusion often slices [:3, :] internally.

      - 'cam2lidar'   : (N, 4, 4)
            Each camera's transform to lidar frame.

      - 'lidar2cam'   : (N, 4, 4)
            Lidar frame to camera frame.

      - 'img_aug_matrix' : (N, 3, 4)
            Per camera image augmentation matrix:
              - top-left 3x3 is usually resize/rotation (identity if you feed raw images)
              - last column is 0 (no translation in raw feed)
            Shape must be (N, 3, 4). (BEVFusion accesses [:, :3, :3] and [:, :3, 3].)

      - 'lidar_aug_matrix': (4, 4)
            Lidar augmentation (identity if none). Shape must be (4,4).

      - 'num_pts_feats'   : int
            Number of point features per point (e.g., 5 for nuScenes: x,y,z,intensity,timestamp).
            A plain Python int is fine.

    Other notes:
      * N is the number of cameras (typically 6).
      * All file paths are absolute and must exist.
      * We do **not** apply any image resize/flip here. If your model expects pre-resized
        inputs, either resize the images before stacking or encode that in img_aug_matrix.

    Args:
      dataroot: nuScenes root folder containing v1.0-*/ samples/ sweeps/ maps/ etc.
      version:  'v1.0-trainval' | 'v1.0-mini' | 'v1.0-test' (auto-detect if None).
      max_count: stop after this many samples (<=0 means all).

    Yields:
      (lidar_path, img_paths, metainfo, basename)
    """
    version = _find_nus_version(dataroot, version)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    cam_names = _camera_order(nusc)

    for i, sample in enumerate(nusc.sample):
        if 0 <= max_count and i >= max_count:
            break

        # LIDAR_TOP is required for BEVFusion
        lidar_token = sample['data'].get('LIDAR_TOP', None)
        if lidar_token is None:
            continue
        lidar_sd = nusc.get('sample_data', lidar_token)
        lidar_path = osp.join(dataroot, lidar_sd['filename'])
        if not osp.isfile(lidar_path):
            continue  # skip if missing on disk

        # Collect per-camera paths & transforms in a fixed order
        img_paths: List[str] = []
        lidar2cam_list: List[np.ndarray] = []
        cam2lidar_list: List[np.ndarray] = []
        lidar2img_list: List[np.ndarray] = []
        cam2img_list: List[np.ndarray] = []

        ok = True
        for cam in cam_names:
            cam_token = sample['data'].get(cam, None)
            if cam_token is None:
                ok = False
                break
            cam_sd = nusc.get('sample_data', cam_token)
            img_path = osp.join(dataroot, cam_sd['filename'])
            if not osp.isfile(img_path):
                ok = False
                break
            img_paths.append(img_path)

            T_l2c, T_c2l, T_l2i, K4 = _build_lidar2cam_chain(nusc, sample, cam_token, lidar_sd)
            lidar2cam_list.append(T_l2c)
            cam2lidar_list.append(T_c2l)
            lidar2img_list.append(T_l2i)
            cam2img_list.append(K4)

        if not ok:
            continue

        # Stack everything into numpy float32 with the exact shapes required
        N = len(img_paths)
        cam2img    = np.stack(cam2img_list,    axis=0).astype(np.float32)   # (N,4,4)
        lidar2cam  = np.stack(lidar2cam_list,  axis=0).astype(np.float32)   # (N,4,4)
        cam2lidar  = np.stack(cam2lidar_list,  axis=0).astype(np.float32)   # (N,4,4)
        lidar2img  = np.stack(lidar2img_list,  axis=0).astype(np.float32)   # (N,4,4)

        # No image aug here → identity per camera in [N,3,4], last column 0
        img_aug_matrix = np.zeros((N, 3, 4), dtype=np.float32)
        for k in range(N):
            img_aug_matrix[k, :3, :3] = np.eye(3, dtype=np.float32)

        # No lidar aug → identity (4,4)
        lidar_aug_matrix = np.eye(4, dtype=np.float32)

        metainfo: Dict[str, np.ndarray] = dict(
            cam2img=cam2img,
            lidar2img=lidar2img,
            cam2lidar=cam2lidar,
            lidar2cam=lidar2cam,
            img_aug_matrix=img_aug_matrix,     # (N,3,4)
            lidar_aug_matrix=lidar_aug_matrix, # (4,4)
            num_pts_feats=5,                   # int is OK
        )

        # A stable name you can log — lidar sample_data token is convenient
        basename = lidar_sd['token']
        yield (lidar_path, img_paths, metainfo, basename)

def load_nus_lidar_points(path: str) -> np.ndarray:
    """
    nuScenes LiDAR is .pcd.bin with (x,y,z,intensity,ring) float32.
    Return (N,5) float32; if file is different, try best-effort fallbacks.
    """
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 5 == 0:
        return arr.reshape(-1, 5)
    # Fallbacks
    if arr.size % 4 == 0:
        xyz_i = arr.reshape(-1, 4)
        ring = np.zeros((xyz_i.shape[0], 1), dtype=np.float32)
        return np.concatenate([xyz_i, ring], axis=1)
    raise ValueError(f"Unexpected LiDAR binary format: {path}, float32 count={arr.size}")


def load_and_stack_images(img_paths: List[str], size_hw=(256, 704)) -> torch.Tensor:
    """
    Load N images, resize to HxW, return [N,3,H,W] float32 normalized [0..1].
    """
    H, W = size_hw
    imgs = []
    for p in img_paths:
        im = Image.open(p).convert('RGB').resize((W, H), Resampling.BILINEAR)
        np_im = np.asarray(im, dtype=np.float32) / 255.0  # H W 3
        chw = torch.from_numpy(np_im).permute(2, 0, 1).contiguous()  # 3 H W
        imgs.append(chw)
    x = torch.stack(imgs, dim=0)  # [N,3,H,W]
    return x


def maybe_set_attn_chunk(model: torch.nn.Module, chunk: Optional[int]) -> bool:
    """
    Try to set model.view_transform.attn_chunk if available.
    Return True if supported, else False.
    """
    if chunk is None:
        return True
    try:
        vt = getattr(model, 'view_transform', None)
        if vt is None:
            return False
        if hasattr(vt, 'attn_chunk'):
            vt.attn_chunk = int(chunk)
            print(f"[INFO] set view_transform.attn_chunk = {chunk}")
            return True
    except Exception as e:
        print(f"[WARN] Failed to set attn_chunk: {e}")
    return False


def build_model_from_cfg(cfg_path: str, checkpoint: str, device: str = 'cuda'):
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = MODELS.build(cfg.model)
    model.to(device)
    model.eval()
    # load ckpt (strict=False to allow weightsonly)
    _ = load_checkpoint(model, checkpoint, map_location=device)
    return model, cfg


# ==========================
# Visualization (optional)
# ==========================

def project_pts_to_img(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    pts_cam: [3,N] in camera frame, K: [3,3]; return [2,N] pixel coords.
    Only for quick overlay; ignores distortion and masks depth<=0 outside.
    """
    uvw = K @ pts_cam
    uv = uvw[:2] / np.clip(uvw[2:3], 1e-6, None)
    return uv


import numpy as np
import cv2
from PIL import Image

def quick_draw_boxes_on_image(img_pil, boxes, lidar2cam, cam_K4,
                              color=(0, 255, 0), thickness=2):
    """
    Overlay 3D boxes (in LiDAR coordinates) on a single camera image.

    Args:
        img_pil:  PIL.Image RGB image
        boxes:    list/array of boxes (each len 7, 9, or 10)
        lidar2cam: np.ndarray (4,4) LiDAR→Cam extrinsic
        cam_K4:   np.ndarray (4,4) camera intrinsics (K in top-left 3x3)
        color:    tuple (B,G,R) draw color
        thickness: line thickness (int)
    Returns:
        PIL.Image with projected boxes drawn.
    """
    # Convert to OpenCV BGR
    img_cv = np.asarray(img_pil)[:, :, ::-1].copy()
    H, W, _ = img_cv.shape

    def corners_3d(b):
        # safe parse 7/9/10-d boxes
        arr = np.array(b).reshape(-1).astype(np.float32)
        x, y, z, dx, dy, dz, yaw = arr[:7]
        hx, hy, hz = dx/2, dy/2, dz/2
        corners = np.array([
            [ hx,  hy,  hz],
            [ hx, -hy,  hz],
            [-hx, -hy,  hz],
            [-hx,  hy,  hz],
            [ hx,  hy, -hz],
            [ hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx,  hy, -hz],
        ], dtype=np.float32).T  # 3x8
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
        pts = R @ corners
        pts[0,:] += x
        pts[1,:] += y
        pts[2,:] += z
        return pts

    P = cam_K4 @ lidar2cam  # 4x4 combined transform

    for b in boxes:
        pts_l = corners_3d(b)                              # [3,8]
        pts_h = np.vstack([pts_l, np.ones((1,8),np.float32)])  # [4,8]
        proj = P @ pts_h                                   # [4,8]
        zs = proj[2,:] + 1e-6
        us, vs = proj[0,:]/zs, proj[1,:]/zs
        mask = (zs>0)&(us>=0)&(us<W)&(vs>=0)&(vs<H)
        if mask.sum()<4: continue
        pts2d = np.stack([us,vs],axis=1).astype(np.int32)

        # connect box edges
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for i,j in edges:
            if mask[i] and mask[j]:
                cv2.line(img_cv, tuple(pts2d[i]), tuple(pts2d[j]), color, thickness)

    return Image.fromarray(img_cv[:, :, ::-1])

def enrich_meta_for_predict(mi: dict, img_paths: list, img_hw=None, num_pts_feats: int = 5):
    """
    Ensure BEVFusion/TransFusion-style heads have the meta they expect.

    Args:
      mi:            dict you already built (cam2img, lidar2img, lidar2cam, cam2lidar, img_aug_matrix, lidar_aug_matrix, ...)
      img_paths:     list of camera image file paths (len = N cams)
      img_hw:        optional (H, W); if None we read the first image to get size
      num_pts_feats: default 5 for nuScenes (x,y,z,intensity,timestamp)
    """
    # ---- Required by TransFusionHead.predict_by_feat ----
    # These must be *classes*, not strings:
    from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode
    mi['box_type_3d'] = LiDARInstance3DBoxes       # callable class
    mi['box_mode_3d'] = Box3DMode.LIDAR            # enum value

    # ---- Commonly used / safe defaults ----
    # Some heads/pipelines read these (especially if you trained with data aug):
    mi.setdefault('pcd_scale_factor', 1.0)         # float
    if 'pcd_rotation' not in mi:
        mi['pcd_rotation'] = np.eye(3, dtype=np.float32)       # (3,3)
    if 'pcd_trans' not in mi:
        mi['pcd_trans'] = np.zeros(3, dtype=np.float32)        # (3,)
    # If code branches use ori_* provide copies
    mi.setdefault('ori_cam2img',  mi['cam2img'].copy())        # (N,4,4)
    mi.setdefault('ori_lidar2img', mi['lidar2img'].copy())     # (N,4,4)

    # nuScenes label map (optional but nice for visualization)
    mi.setdefault('classes', [
        'car','truck','construction_vehicle','bus','trailer',
        'barrier','motorcycle','bicycle','pedestrian','traffic_cone'
    ])

    # img_shape/ori_shape are frequently accessed; give per-camera shapes
    if img_hw is None:
        # lazily read one image to get H,W
        import cv2
        im0 = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
        assert im0 is not None, f'Failed to read image: {img_paths[0]}'
        H, W = im0.shape[:2]
    else:
        H, W = img_hw

    N = len(img_paths)
    # BEVFusion/mmdet3d metas usually keep shapes as tuples (H, W, 3)
    per_cam_shape = [(H, W, 3)] * N
    mi.setdefault('img_shape', per_cam_shape)
    mi.setdefault('ori_shape', per_cam_shape)

    # Number of point features
    mi['num_pts_feats'] = int(num_pts_feats)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _extract_xyz_dxdydz_yaw(b):
    """
    Accepts:
      - array/tensor shape (..., 7|9|10)
      - LiDARInstance3DBoxes (uses .tensor)
    Returns: x,y,z,dx,dy,dz,yaw as floats
    Notes:
      - If len>=7, we take the first 7. If len>7 (e.g., 9 with vx,vy), we ignore extras.
      - mmdet3d convention for LiDAR boxes is [x, y, z, dx, dy, dz, yaw, (vx, vy)?]
      - dx: length along x-axis, dy: width along y-axis, dz: height (z).
    """
    # Unwrap LiDARInstance3DBoxes
    try:
        from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
        if isinstance(b, LiDARInstance3DBoxes):
            b = b.tensor  # [N, 7|9]
            # If N==1, squeeze
            if b.ndim == 2 and b.shape[0] == 1:
                b = b[0]
    except Exception:
        pass

    arr = _to_numpy(b).reshape(-1)  # 1D
    if arr.shape[0] < 7:
        raise ValueError(f"bbox must have ≥7 values, got {arr.shape}")

    x, y, z, dx, dy, dz, yaw = arr[:7].astype(np.float32)
    return float(x), float(y), float(z), float(dx), float(dy), float(dz), float(yaw)

def corners_3d(b):
    """
    Returns 8 corners (3x8) for a single 3D box in LiDAR coords.
    Box convention: center (x,y,z), size (dx,dy,dz), yaw (rad, about +Z).
    """
    x, y, z, dx, dy, dz, yaw = _extract_xyz_dxdydz_yaw(b)

    # Local corners before rotation (centered at origin)
    # dx along x, dy along y, dz along z
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    # 8 corners (x, y, z)
    corners = np.array([
        [ hx,  hy,  hz],
        [ hx, -hy,  hz],
        [-hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy, -hz],
        [-hx, -hy, -hz],
        [-hx,  hy, -hz],
    ], dtype=np.float32).T  # 3x8

    # Rotation around Z
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners  # 3x8

def box_center_top(b):
    """Handy: returns center (3,) and top-center (3,) for text placement."""
    x, y, z, dx, dy, dz, yaw = _extract_xyz_dxdydz_yaw(b)
    center = np.array([x, y, z], dtype=np.float32)
    top = np.array([x, y, z + dz/2.0], dtype=np.float32)
    return center, top

import numpy as np
import cv2
from PIL import Image

def _corners_3d(box):
    """
    box: len 7/9/10 → (x, y, z, dx, dy, dz, yaw, ...)
    Returns corners in LiDAR frame: [3, 8]
    """
    arr = np.array(box, dtype=np.float32).reshape(-1)
    x, y, z, dx, dy, dz, yaw = arr[:7]
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    # 8 corners in the box local frame (x forward, y left, z up for nuScenes LiDAR convention)
    corners = np.array([
        [ hx,  hy,  hz],
        [ hx, -hy,  hz],
        [-hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy, -hz],
        [-hx, -hy, -hz],
        [-hx,  hy, -hz],
    ], dtype=np.float32).T  # [3,8]

    c, s = np.cos(yaw).astype(np.float32), np.sin(yaw).astype(np.float32)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    pts = R @ corners
    pts[0, :] += x
    pts[1, :] += y
    pts[2, :] += z
    return pts  # [3,8]


def _project_lidar_corners_to_img(P_4x4, corners_lidar, W, H):
    """
    P_4x4: lidar2img 4x4 (or K4 @ lidar2cam)
    corners_lidar: [3,8]
    Returns:
      pts2d: (8,2) int32
      vis_mask: (8,) bool in-front & in-FOV
    """
    pts_h = np.vstack([corners_lidar, np.ones((1, corners_lidar.shape[1]), dtype=np.float32)])  # [4,8]
    proj = P_4x4 @ pts_h  # [4,8]
    zs = proj[2, :] + 1e-6
    us = proj[0, :] / zs
    vs = proj[1, :] / zs
    mask = (zs > 0) & (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    pts2d = np.stack([us, vs], axis=1).astype(np.int32)  # (8,2)
    return pts2d, mask


def quick_draw_boxes_on_image_dual(
    img_pil,
    pred_boxes,        # (N_pred, 7/9/10) in LiDAR frame
    gt_boxes,          # (N_gt, 7/9/10) in LiDAR frame
    lidar2img_4x4=None,
    lidar2cam_4x4=None,
    cam_K4=None,
    color_pred=(0, 255, 0),
    color_gt=(0, 0, 255),
    thickness=2,
    diag_prefix="[VIS]"
):
    """
    Draw predicted (green) and ground-truth (red) 3D boxes on the *resized* image.

    You MUST provide either:
      - lidar2img_4x4  (preferred), or
      - lidar2cam_4x4 + cam_K4 (then P = cam_K4 @ lidar2cam_4x4)

    We automatically scale P to match the resized image size.

    Returns: PIL.Image with overlays
    """
    # Convert to OpenCV BGR
    img_cv = np.asarray(img_pil)[:, :, ::-1].copy()
    H_resized, W_resized = img_cv.shape[:2]

    # Figure out original size BEFORE your resize (we can infer it from the PIL image you loaded)
    # If you still have the original image on disk, best practice is to open it first, record (H0, W0),
    # then resize to (W_resized, H_resized). Here, we assume you created `img_pil` via:
    #   img_pil = Image.open(path).convert('RGB'); H0,W0 = img_pil.size[::-1]; img_pil = img_pil.resize((704,256))
    # If you only have the resized image now, pass (H0, W0) externally if needed.
    # For your current code you *do* have the original PIL before resize, so do:
    #   H0, W0 = orig_img.size[::-1]
    # To keep this function self-contained, we store the original size in EXIF if you pass it in. Otherwise,
    # we assume no EXIF and fall back to scaling using the resized shape (sx=sy=1), which will still draw *something*.
    if hasattr(img_pil, "_original_size_"):
        H0, W0 = img_pil._original_size_
        sx, sy = W_resized / float(W0), H_resized / float(H0)
    else:
        # Fallback: try to get original size via a hint field we can attach outside.
        H0 = getattr(img_pil, "_H0", None)
        W0 = getattr(img_pil, "_W0", None)
        if (H0 is not None) and (W0 is not None):
            sx, sy = W_resized / float(W0), H_resized / float(H0)
        else:
            # Last resort: no scaling (may project off-screen if intrinsics are from original size)
            sx, sy = 1.0, 1.0

    # Build projection matrix P
    if lidar2img_4x4 is not None:
        P = np.array(lidar2img_4x4, dtype=np.float32).copy()
    else:
        assert (lidar2cam_4x4 is not None) and (cam_K4 is not None), \
            "Need either lidar2img_4x4 or both lidar2cam_4x4 and cam_K4."
        P = (np.array(cam_K4, dtype=np.float32) @ np.array(lidar2cam_4x4, dtype=np.float32)).copy()

    # Scale P to match resized image: multiply row 0 by sx and row 1 by sy
    P[0, :] *= sx
    P[1, :] *= sy

    # Edge list for a wireframe cuboid
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

    def _draw_group(boxes, color, tag):
        infront_total = 0
        fov_total = 0
        for b in boxes:
            corners = _corners_3d(b)  # [3,8]
            pts2d, mask = _project_lidar_corners_to_img(P, corners, W_resized, H_resized)
            infront_total += int(mask.any())
            fov_total += int(mask.sum() >= 4)
            # draw edges only for visible corners
            for i, j in edges:
                if mask[i] and mask[j]:
                    cv2.line(img_cv, tuple(pts2d[i]), tuple(pts2d[j]), color, thickness, lineType=cv2.LINE_AA)
            # draw center point (project center too)
            ctr = np.array([corners[0].mean(), corners[1].mean(), corners[2].mean(), 1.0], dtype=np.float32)
            ctrp = P @ ctr
            if ctrp[2] > 1e-6:
                u, v = int(ctrp[0]/ctrp[2]), int(ctrp[1]/ctrp[2])
                if 0 <= u < W_resized and 0 <= v < H_resized:
                    cv2.circle(img_cv, (u, v), 3, color, -1, lineType=cv2.LINE_AA)
        print(f"{diag_prefix} {tag}: boxes={len(boxes)}, any-in-front={infront_total}, >=4-corners-in-FOV={fov_total}")

    # Draw predicted (green) and GT (red)
    _draw_group(pred_boxes, color_pred, "PRED")
    _draw_group(gt_boxes,   color_gt,   "GT")

    return Image.fromarray(img_cv[:, :, ::-1])  # back to RGB PIL


import numpy as np
from PIL import Image
import cv2

def project_points_lidar_to_image(points_xyz, P_4x4, W, H, max_pts=20000):
    """Return (u,v,mask) for a subset of LiDAR points."""
    if len(points_xyz) == 0:
        return np.empty((0,2), int), np.zeros((0,), bool)
    pts = points_xyz[:max_pts]
    pts_h = np.c_[pts, np.ones(len(pts), dtype=np.float32)]  # [N,4]
    proj = (P_4x4 @ pts_h.T).T  # [N,4]
    z = proj[:, 2] + 1e-6
    u = proj[:, 0] / z
    v = proj[:, 1] / z
    m = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uv = np.stack([u, v], 1).astype(np.int32)
    return uv, m

def overlay_projected_points(img_pil, lidar_pts, lidar2img_4x4, H0W0=None):
    """Draw a few projected LiDAR points as white dots to verify extrinsics/intrinsics."""
    img_cv = np.asarray(img_pil)[:, :, ::-1].copy()
    H, W = img_cv.shape[:2]
    P = np.array(lidar2img_4x4, np.float32).copy()
    # Scale intrinsics rows if image was resized
    if H0W0 is not None:
        H0, W0 = H0W0
        sx, sy = W / float(W0), H / float(H0)
        P[0, :] *= sx; P[1, :] *= sy
    uv, m = project_points_lidar_to_image(lidar_pts[:, :3], P, W, H, max_pts=30000)
    uv = uv[m]
    for (u, v) in uv[::10]:  # stride to avoid heavy draw
        cv2.circle(img_cv, (u, v), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    return Image.fromarray(img_cv[:, :, ::-1])


import numpy as np
import cv2
from PIL import Image

_EDGES = [(0,1),(1,2),(2,3),(3,0),
          (4,5),(5,6),(6,7),(7,4),
          (0,4),(1,5),(2,6),(3,7)]

def _make_P_scaled(lidar2img_4x4=None, lidar2cam_4x4=None, camK4=None, H_resized=256, W_resized=704, H0W0=None):
    assert (lidar2img_4x4 is not None) or (lidar2cam_4x4 is not None and camK4 is not None), \
        "Provide lidar2img or (lidar2cam + camK4)."
    if lidar2img_4x4 is not None:
        P = np.array(lidar2img_4x4, np.float32).copy()
    else:
        P = (np.array(camK4, np.float32) @ np.array(lidar2cam_4x4, np.float32)).copy()
    if H0W0 is not None:
        H0, W0 = H0W0
        sx, sy = W_resized / float(W0), H_resized / float(H0)
        P[0, :] *= sx; P[1, :] *= sy
    return P

def _corners_3d_from_box(b, swap_lw=False, yaw_offset=0.0):
    """b: (x,y,z, dx,dy,dz,yaw[, ...]) or (x,y,z,w,l,h,yaw) if swap_lw=True."""
    t = np.array(b, np.float32).reshape(-1)
    x, y, z = t[0:3]
    dx, dy, dz = t[3:6]
    yaw = t[6] + yaw_offset
    if swap_lw:
        dx, dy = dy, dx
    hx, hy, hz = dx/2.0, dy/2.0, dz/2.0

    # nuScenes LiDAR frame: x forward, y left, z up
    corners = np.array([
        [ hx,  hy,  hz],
        [ hx, -hy,  hz],
        [-hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy, -hz],
        [-hx, -hy, -hz],
        [-hx,  hy, -hz],
    ], np.float32).T
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], np.float32)
    pts = R @ corners
    pts[0, :] += x; pts[1, :] += y; pts[2, :] += z
    return pts  # [3,8]

def _project_corners(P_4x4, corners_lidar, W, H):
    pts_h = np.vstack([corners_lidar, np.ones((1, corners_lidar.shape[1]), np.float32)])  # [4,8]
    proj = P_4x4 @ pts_h
    z = proj[2, :] + 1e-6
    u = proj[0, :] / z
    v = proj[1, :] / z
    mask = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts2d = np.stack([u, v], 1).astype(np.int32)
    return pts2d, mask

def draw_boxes_dual(
    img_pil,
    pred_boxes, gt_boxes,
    lidar2img_4x4=None, lidar2cam_4x4=None, camK4=None,
    orig_size=None,                 # (H0, W0)
    color_pred=(0,255,0), color_gt=(0,0,255),
    thickness=2,
    score_thr=None, pred_scores=None,
    max_range_m=80.0,
    swap_lw_pred=False, swap_lw_gt=False,
    yaw_off_pred=0.0, yaw_off_gt=0.0,
    diag_prefix="[VIS]"
):
    img_cv = np.asarray(img_pil)[:, :, ::-1].copy()
    H, W = img_cv.shape[:2]
    P = _make_P_scaled(lidar2img_4x4, lidar2cam_4x4, camK4, H, W, H0W0=orig_size)

    def _filter_and_draw(boxes, color, tag, swap_lw, yaw_off, scores=None):
        n_infront = n_fov4 = n_drawn = 0
        for i, b in enumerate(boxes):
            if scores is not None and score_thr is not None:
                if float(scores[i]) < float(score_thr): 
                    continue
            # simple range filter
            if np.linalg.norm(b[:2]) > max_range_m:
                continue
            corners = _corners_3d_from_box(b, swap_lw=swap_lw, yaw_offset=yaw_off)
            pts2d, m = _project_corners(P, corners, W, H)
            n_infront += int(m.any())
            n_fov4 += int(m.sum() >= 4)
            # draw edges for visible segments
            for a, c in _EDGES:
                if m[a] and m[c]:
                    cv2.line(img_cv, tuple(pts2d[a]), tuple(pts2d[c]), color, thickness, lineType=cv2.LINE_AA)
            # draw center
            ctr = np.array([corners[0].mean(), corners[1].mean(), corners[2].mean(), 1.0], np.float32)
            ctrp = P @ ctr
            if ctrp[2] > 0:
                u, v = int(ctrp[0]/ctrp[2]), int(ctrp[1]/ctrp[2])
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(img_cv, (u, v), 3, color, -1, lineType=cv2.LINE_AA)
                    n_drawn += 1
        print(f"{diag_prefix} {tag}: in-front(any)={n_infront}, in-FOV(>=4)={n_fov4}, drawn={n_drawn}")

    _filter_and_draw(pred_boxes, color_pred, "PRED", swap_lw_pred, yaw_off_pred, scores=pred_scores)
    _filter_and_draw(gt_boxes,   color_gt,   "GT",   swap_lw_gt,   yaw_off_gt,   scores=None)
    return Image.fromarray(img_cv[:, :, ::-1])
# ==========================
# Main
# ==========================
import inspect
import math
import torch
import torch.nn as nn

# ---------------------------
# 0) PARAM COUNT
# ---------------------------
def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train

def human_params(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

def human_flops(n: float) -> str:
    if n >= 1e12: return f"{n/1e12:.2f} TFLOPs"
    if n >= 1e9:  return f"{n/1e9:.2f} GFLOPs"
    if n >= 1e6:  return f"{n/1e6:.2f} MFLOPs"
    return f"{n:.0f} FLOPs"

# ---------------------------
# 2) FLOPs via mmengine (API-agnostic)
# ---------------------------
def flops_mmengine_safe(model, inputs_args, inputs_kwargs, device="cuda"):
    """
    Call mmengine.analysis.get_model_complexity_info with whatever it supports.
    Returns a numeric FLOPs or raises.
    """
    from mmengine.analysis import get_model_complexity_info

    model = model.to(device).eval()
    # Temporarily use forward_dummy if present
    restore = None
    if hasattr(model, "forward_dummy"):
        restore = model.forward
        model.forward = model.forward_dummy  # type: ignore

    try:
        sig = inspect.signature(get_model_complexity_info).parameters
        kwargs = {}
        # Prefer 'inputs' if available (most robust across versions)
        if "inputs" in sig:
            # Pack args+kwargs the same way we will actually call forward_dummy
            if inputs_args and not inputs_kwargs:
                kwargs["inputs"] = inputs_args
            elif not inputs_args and inputs_kwargs:
                # mmengine expects a tuple of args; if only kwargs are used, we still pass empty args
                kwargs["inputs"] = ()
                kwargs["input_constructor"] = None  # ignored if not in signature
            else:
                kwargs["inputs"] = inputs_args  # kwargs generally aren’t passed via mmengine
        elif "input_shape" in sig:
            # Infer shape from first tensor in args/kwargs
            def _first_tensor():
                for a in inputs_args:
                    if torch.is_tensor(a):
                        return tuple(a.shape)
                    if isinstance(a, dict):
                        for v in a.values():
                            if torch.is_tensor(v):
                                return tuple(v.shape)
                for v in inputs_kwargs.values():
                    if torch.is_tensor(v):
                        return tuple(v.shape)
                raise RuntimeError("Cannot infer input_shape; pass a tensor in args/kwargs.")
            kwargs["input_shape"] = _first_tensor()
        # Quiet options if present
        if "show_table" in sig: kwargs["show_table"] = False
        if "show_arch"  in sig: kwargs["show_arch"]  = False
        if "as_strings" in sig: kwargs["as_strings"] = False

        result = get_model_complexity_info(model, **kwargs)

        # Normalize return
        flops = None
        if isinstance(result, tuple) and len(result) >= 1:
            flops = result[0]
        elif isinstance(result, dict) and "flops" in result:
            flops = result["flops"]
        elif isinstance(result, str):
            flops = result
        else:
            raise RuntimeError(f"Unexpected get_model_complexity_info return: {type(result)}")

        if isinstance(flops, str):
            toks = flops.strip().split()
            if len(toks) >= 2:
                val = float(toks[0]); unit = toks[1].upper()
                scale = {"TFLOPS":1e12, "GFLOPS":1e9, "MFLOPS":1e6, "KFLOPS":1e3, "FLOPS":1.0}.get(unit, 1.0)
                flops = val * scale
            else:
                raise RuntimeError(f"Cannot parse FLOPs string: {flops}")

        return float(flops)
    finally:
        if restore is not None:
            model.forward = restore

# ---------------------------
# 3) LAST-RESORT HOOK-BASED FLOPs (no extra libs)
#    Counts Conv{1,2,3}d / Linear / ConvTranspose{2,3}d.
#    (Ignores BN/ReLU/Pooling; Attention not covered)
# ---------------------------
def _conv2d_flops(m: nn.Conv2d, x, y):
    # FLOPs per output element = Cin/group * Kh * Kw * 2 (mul+add)
    # Total = Cout * Hout * Wout * Cin/group * Kh * Kw * 2
    Cin = m.in_channels
    Cout = m.out_channels
    Kh, Kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
    groups = m.groups
    # y shape: (B, Cout, Hout, Wout)
    B, _, Hout, Wout = y.shape
    return B * Cout * Hout * Wout * (Cin // groups) * Kh * Kw * 2

def _convtranspose2d_flops(m: nn.ConvTranspose2d, x, y):
    # Similar to conv2d (transpose conv math is symmetric in FLOPs count)
    Cin = m.in_channels
    Cout = m.out_channels
    Kh, Kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
    groups = m.groups
    B, _, Hout, Wout = y.shape
    return B * Cout * Hout * Wout * (Cin // groups) * Kh * Kw * 2

def _linear_flops(m: nn.Linear, x, y):
    # FLOPs per output = in_features * 2, total = B * out_features * in_features * 2
    in_f = m.in_features
    out_f = m.out_features
    B = x[0].shape[0] if torch.is_tensor(x[0]) else 1
    return B * in_f * out_f * 2

SUPPORTED = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ConvTranspose2d, nn.ConvTranspose3d)

def estimate_flops_by_hooks(model: nn.Module, run_fn, *args, **kwargs) -> int:
    flops = 0
    handles = []

    def hook(module, inp, out):
        nonlocal flops
        try:
            if isinstance(module, nn.Conv2d):
                flops += _conv2d_flops(module, inp[0], out)
            elif isinstance(module, nn.ConvTranspose2d):
                flops += _convtranspose2d_flops(module, inp[0], out)
            elif isinstance(module, nn.Linear):
                flops += _linear_flops(module, inp, out)
            # (Extend with Conv1d/3d/Transpose3d if used)
        except Exception:
            pass  # keep going even if some shapes are dynamic/unsupported

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))

    try:
        with torch.no_grad():
            _ = run_fn(*args, **kwargs)
    finally:
        for h in handles:
            h.remove()

    return int(flops)

# ----------------------- visualization helpers ----------------------- #

import argparse
import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from os import path as osp
from typing import Optional, Tuple, List, Union

# ----------------------- Visualization Helpers ----------------------- #

def project_points(points_xyz: np.ndarray, P_4x4: np.ndarray):
    """
    Projects 3D points into 2D image space.
    
    Args:
        points_xyz: [N, 3] float32 - Points in Source Frame (usually LiDAR).
        P_4x4:      [4, 4] float32 - Projection Matrix (Source -> Image).
                    If source is LiDAR, this is (Camera_Intrinsics @ Lidar2Cam_Extrinsics).
    
    Returns:
        uv: [N, 2] float32 - 2D pixel coordinates (u, v).
        zc: [N]    float32 - Depth (z) in Camera Frame.
        mask: [N]  bool    - Mask of points strictly in front of camera (z > epsilon).
    """
    N = points_xyz.shape[0]
    # Add homogeneous coord: [N, 3] -> [N, 4]
    homo = np.concatenate([points_xyz[:, :3], np.ones((N, 1), dtype=np.float32)], axis=1)
    
    # Project: [N, 4] @ [4, 4].T -> [N, 4]
    cam = (P_4x4 @ homo.T).T
    
    # Depth in camera frame
    z = cam[:, 2].copy()
    mask = z > 1e-6  # Keep points in front of camera
    
    # Perspective division (x/z, y/z)
    uv = cam[mask, :2] / cam[mask, 2:3]
    zc = z[mask]
    
    return uv, zc, mask

def overlay_points_depth(img: Image.Image, uv: np.ndarray, zc: np.ndarray,
                         point_size_px: Optional[int] = None,
                         cmap='turbo', alpha=0.95, 
                         max_depth: float = 80.0) -> Image.Image:
    """
    Draws depth-colored points on the image using Matplotlib.
    
    Args:
        img: PIL Image (Background).
        uv:  [M, 2] Pixel coordinates of valid points.
        zc:  [M]    Depth values of valid points.
        max_depth:  Maximum depth for color normalization (meters).
    """
    w, h = img.size
    
    if zc.size > 0:
        # Normalize depth 0 to 1 based on max_depth
        norm = np.clip(zc, 0, max_depth) / max_depth
    else:
        norm = zc

    # Sort by depth (far -> near) so near points draw on top
    idx = np.argsort(norm)[::-1] 
    uv = uv[idx]
    norm = norm[idx]

    if point_size_px is None:
        # Auto-scale point size based on image resolution
        point_size_px = max(1, int(0.0015 * max(w, h)))

    # Use matplotlib for high-quality colormap rendering
    fig = plt.figure(figsize=(w/100., h/100.), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(img)
    
    if uv.size > 0:
        ax.scatter(uv[:, 0], uv[:, 1], s=point_size_px, c=norm, cmap=cmap, linewidths=0, alpha=alpha)
    
    ax.set_axis_off()
    fig.canvas.draw()
    
    # Convert Matplotlib figure back to Numpy/PIL
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(vis)

def box_corners_3d(bxyzwhl: np.ndarray) -> np.ndarray:
    """
    Computes the 8 corners of a 3D box.
    
    Args:
        bxyzwhl: [7] float32 - [x, y, z, dx, dy, dz, yaw]
                 x, y, z: Bottom-center of the box (LiDAR frame).
                 dx, dy, dz: Length, Width, Height.
                 yaw: Rotation around Z-axis.
                 
    Returns:
        pts: [8, 3] float32 - Coordinates of the 8 corners in LiDAR frame.
    """
    x, y, z, dx, dy, dz, yaw = bxyzwhl.tolist()
    
    # Rotation Matrix (Yaw only)
    cosa, sina = math.cos(yaw), math.sin(yaw)
    R = np.array([[cosa, -sina, 0], 
                  [sina,  cosa, 0], 
                  [0,     0,    1]], dtype=np.float32)
    
    # Dimensions (l=dx, w=dy, h=dz)
    l, w, h = dx, dy, dz
    
    # 1. Create corners relative to center (0,0,0)
    #    Note: In MMDet3D, 'z' is usually the BOTTOM of the box.
    #    So relative Z coordinates range from 0 (bottom) to h (top).
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float32)
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float32)
    z_c = np.array([ 0,    0,    0,    0,    h,    h,    h,    h  ], dtype=np.float32) 
    
    # Stack: [8, 3]
    pts = np.stack([x_c, y_c, z_c], axis=1)
    
    # 2. Rotate and Translate
    #    pts @ R.T rotates the points
    #    + [x, y, z] moves the box to its global position
    pts = (R @ pts.T).T + np.array([x, y, z], dtype=np.float32)
    
    return pts

def draw_boxes_on_image(img: Image.Image,
                        boxes: np.ndarray,
                        P_lidar2img: np.ndarray,
                        color=(0, 255, 0),
                        width=2) -> Image.Image:
    """
    Draws 3D boxes projected onto the image.
    
    Args:
        boxes: [N, 7] float32 - [x, y, z, dx, dy, dz, yaw] (LiDAR frame)
    """
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    
    if boxes is None or boxes.shape[0] == 0:
        return img

    for b in boxes:
        b7 = b[:7].astype(np.float32)
        corners = box_corners_3d(b7)  # [8, 3]
        
        # Project corners to Image Plane
        uv, zc, m = project_points(corners, P_lidar2img)
        
        # Basic Clipping: 
        # If fewer than 8 corners projected (some behind camera), skip or handle partially.
        # For simple viz, we often skip if any corner is behind cam (zc <= 0).
        if uv.shape[0] != 8: continue 
        
        # Edges connecting the 8 corners
        # Bottom face: 0-1, 1-2, 2-3, 3-0
        # Top face:    4-5, 5-6, 6-7, 7-4
        # Pillars:     0-4, 1-5, 2-6, 3-7
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        
        for i, j in edges:
            x1, y1 = uv[i]
            x2, y2 = uv[j]
            # Draw line
            d.line((x1, y1, x2, y2), fill=color, width=width)
            
    return img

def overlay_points_and_boxes(img: Image.Image,
                             pts_xyz: np.ndarray,
                             P_l2i: Optional[np.ndarray] = None,
                             pred_boxes: Optional[np.ndarray] = None,
                             gt_boxes: Optional[np.ndarray] = None,
                             # Additional args for flexibility
                             lidar2cam_4x4: Optional[np.ndarray] = None,
                             camK4: Optional[np.ndarray] = None,
                             depth_coloring: bool = True,
                             max_depth: float = 80.0,
                             box_color_pred: Tuple[int,int,int] = (0, 255, 0),
                             box_color_gt: Tuple[int,int,int] = (255, 0, 0),
                             thickness: int = 2) -> Image.Image:
    """
    Master visualization function.
    1. Projects LiDAR points to image (optional depth coloring).
    2. Projects and draws 3D Bounding Boxes (Pred & GT).
    """
    
    # 1. Construct Projection Matrix (LiDAR -> Image) if not provided
    if P_l2i is None:
        if lidar2cam_4x4 is not None and camK4 is not None:
            # P = K @ Extrinsics
            # camK4 is usually 4x4 (or 3x3 expanded). Assuming 4x4 here or compatible logic.
            if camK4.shape == (3, 3):
                # Expand to 4x4 identity if it was 3x3
                K_exp = np.eye(4, dtype=np.float32)
                K_exp[:3, :3] = camK4
                camK4 = K_exp
            
            P_l2i = camK4 @ lidar2cam_4x4
        else:
            print("Warning: No projection matrix provided. Returning original image.")
            return img

    vis = img

    # 2. Draw Points (Depth Colored)
    if pts_xyz is not None and len(pts_xyz) > 0:
        uv, z, m = project_points(pts_xyz, P_l2i)
        w, h = img.size
        # Filter points within image bounds
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        
        if depth_coloring:
            vis = overlay_points_depth(vis, uv[inside], z[inside], 
                                       point_size_px=None, cmap='turbo', 
                                       alpha=0.95, max_depth=max_depth)
    
    # 3. Draw GT Boxes
    if gt_boxes is not None and len(gt_boxes) > 0:
        vis = draw_boxes_on_image(vis, gt_boxes, P_l2i, color=box_color_gt, width=thickness)

    # 4. Draw Pred Boxes
    if pred_boxes is not None and len(pred_boxes) > 0:
        vis = draw_boxes_on_image(vis, pred_boxes, P_l2i, color=box_color_pred, width=thickness)

    return vis
    
#ap.add_argument("--config", type=str, default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to MMDet3D config .py")
#    ap.add_argument("--checkpoint", type=str, default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Path to model checkpoint .pth")
def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--config", type=str, default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to MMDet3D config .py")
    #ap.add_argument("--checkpoint", type=str, default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Path to model checkpoint .pth")
    ap.add_argument("--config", type=str, default="work_dirs/mybevfusion7_newv2/mybevfusion7_crossattnaux_paintingv2.py", help="Path to MMDet3D config .py")
    ap.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_newv2/epoch_2.pth", help="Path to model checkpoint .pth")
    ap.add_argument("--dataroot", type=str, default="data/nuscenes", help="nuScenes dataroot")
    ap.add_argument("--nus-version", type=str, default="v1.0-trainval", help="nuScenes version (e.g., v1.0-mini, v1.0-trainval)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--attn-chunk", type=int, default=None, help="Set view_transform.attn_chunk if supported")
    ap.add_argument("--max-samples", type=int, default=10, help="Limit samples for quick test")
    ap.add_argument("--resize", type=int, nargs=2, default=[704,256]) # W H
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="infer_out")
    ap.add_argument("--save-vis", action="store_true", default=True, help="Save simple image overlays")
    args = ap.parse_args()

    set_determinism(args.seed)
    ensure_dir(args.out_dir)

    # 1) Build model (no dataset/runner)
    model, cfg = build_model_from_cfg(args.config, args.checkpoint, device=args.device)
    device = args.device
    # Params
    total, trainable = count_params(model)
    print(f"Params (total/trainable): {human_params(total)} / {human_params(trainable)}")
    print(f"~Model size (fp32): {total*4/1e6:.1f} MB")

    # 2) Optionally set attn_chunk on CrossAttnLSSTransform
    _ = maybe_set_attn_chunk(model, args.attn_chunk)

    W, H = args.resize
    # 3) Iterate nuScenes samples
    count = 0
    t_all = []
    lat_ms = []
    peak_mem_mb = 0.0
    torch.cuda.reset_peak_memory_stats(args.device)
    # with torch.inference_mode(), torch.autocast(device_type='cuda' if 'cuda' in args.device else 'cpu',
    #                                             dtype=torch.float16 if 'cuda' in args.device else torch.float32):
    # 
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):    
        #torch.bfloat16 not good for sparse conv
        for lidar_path, img_paths, metainfo, basename in iter_nuscenes_samples(
                dataroot=args.dataroot, version=args.nus_version, max_count=args.max_samples):

            # lidar_path = 'data/nuscenes/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin'
            # img_paths  = [
            #     "data/nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONTxxx.png",
            #     "data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-18-11-07-57+0800__CAM_F",
            #     "/path/to/CAM_BACK_RIGHT.png",
            #     "/path/to/CAM_BACK.png",
            #     "/path/to/CAM_BACK_LEFT.png",
            #     "/path/to/CAM_FRONT_LEFT.png",
            # ]
            
            # metainfo = { 'num_pts_feats' =5
            #     "img_aug_matrix":  (6, 3, 4)
            #     "lidar_aug_matrix": (4, 4)
            #     "cam2img":   [cam_intr_3x3_or_4x4_per_cam, ...], [6, 4, 4]
            #     "lidar2img": [lidar2img_3x4_or_4x4_per_cam, ...], [6, 4, 4]
            #     "cam2lidar": [cam2lidar_4x4_per_cam, ...], [6, 4, 4]
            #     "lidar2cam": [lidar2cam_4x4_per_cam, ...],[6, 4, 4]
            # }
            enrich_meta_for_predict(metainfo, img_paths)
            # 4) Build batch_inputs_dict
            # imgs: [B,N,3,256,704]
            #size_hw=(H, W)
            imgs_ = load_and_stack_images(img_paths, size_hw=(H, W))  #(256, 704) [N,3,H,W], float32
            imgs_ = imgs_.unsqueeze(0).to(args.device, non_blocking=True)  # [1,N,3,H,W]
            # points: list of length B, each [num, feat]
            pts = load_nus_lidar_points(lidar_path)  # (N,5)
            # remove NaN/Inf rows
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]
            # Clamp to model point cloud range (matches your config)
            # point_cloud_range = [-54,-54,-5, 54,54,3]
            xmin,ymin,zmin,xmax,ymax,zmax = -54.0,-54.0,-5.0, 54.0,54.0,3.0
            m = (pts[:,0] >= xmin) & (pts[:,0] <= xmax) \
            & (pts[:,1] >= ymin) & (pts[:,1] <= ymax) \
            & (pts[:,2] >= zmin) & (pts[:,2] <= zmax)
            pts = pts[m]

            pts_t = torch.from_numpy(pts).to(args.device, non_blocking=True).float()
            #pts_t = torch.from_numpy(pts).contiguous().float()   # keep on CPU!
            points_list = [pts_t] #list of [34720, 5]

            batch_inputs_dict = dict(
                imgs=imgs_,
                points=points_list
            )

            # 5) Build Det3DDataSample with correct metainfo
            ds = Det3DDataSample()
            # Make sure tensors are on same device
            mi = {
                k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                for k, v in metainfo.items()
            }
            ds.set_metainfo(mi)
            batch_data_samples = [ds] #list of <Det3DDataSample
            
            # --- Timed forward with proper CUDA sync ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 6) Forward predict
            preds = model.predict(batch_inputs_dict, batch_data_samples)

            # forward
            amp_dtype=torch.bfloat16
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            lat_ms.append(dt)
            
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated()
                peak_mem_mb = max(peak_mem_mb, mem / (1024.0 * 1024.0))
                
            #preds = model.predict(batch_inputs_dict, batch_data_samples)
            t1 = time.time()
            t_all.append((t1 - t0) * 1000.0)
            print(f"{basename} | | {dt:.1f} ms "
                  f"| peak_gpu={peak_mem_mb:.1f} MB")

            count += 1
            print(f"[{count}] {basename}: {t_all[-1]:.1f} ms")
            
            # 5) Extract pred boxes for viz
            pred_boxes = np.zeros((0, 7), dtype=np.float32)
            if isinstance(preds, list) and len(preds) > 0:
                ds_pred = preds[0]
                if hasattr(ds_pred, 'pred_instances_3d') and hasattr(ds_pred.pred_instances_3d, 'bboxes_3d'):
                    b = ds_pred.pred_instances_3d.bboxes_3d
                    if hasattr(b, 'tensor'):
                        arr = b.tensor.detach().cpu().numpy() # shape: [K, 7] or [K, 9]
                        if arr.shape[1] >= 7:
                            pred_boxes = arr[:, :7].astype(np.float32)
            
            # 6) Visualization
            if args.save_vis:
                # --- Setup Camera Index (Front) ---
                cam_idx = 1 if len(img_paths) >= 2 else 0 
                img = Image.open(img_paths[cam_idx]).convert('RGB').resize((W, H), Image.Resampling.BILINEAR)
                
                # --- Prepare Matrices ---
                P_l2i = None
                lidar2cam = None
                K4 = None
                
                if 'lidar2img' in metainfo:
                    # Case A: Pre-computed LiDAR -> Image matrix available
                    P_l2i = metainfo['lidar2img'][cam_idx] # [4, 4]
                else:
                    # Case B: Construct from Extrinsics + Intrinsics
                    K4 = metainfo['cam2img'][cam_idx]      # [4, 4]
                    lidar2cam = metainfo['lidar2cam'][cam_idx] # [4, 4]

                pts_xyz = pts[:, :3] # [N, 3]

                # --- Single Unified Call ---
                vis = overlay_points_and_boxes(
                    img=img,
                    pts_xyz=pts_xyz,
                    P_l2i=P_l2i,                    # Pass if we have it
                    pred_boxes=pred_boxes,
                    gt_boxes=None,                  # Pass gt_boxes here if you have them
                    lidar2cam_4x4=lidar2cam,        # Fallback if P_l2i is None
                    camK4=K4,                       # Fallback if P_l2i is None
                    depth_coloring=True,
                    max_depth=80.0,                 # Coloring range 0-80m
                    box_color_pred=(0, 255, 0),
                    box_color_gt=(255, 0, 0),
                    thickness=2
                )

                out_path = osp.join(args.out_dir, f"{basename}_front_overlay.jpg")
                vis.save(out_path, quality=95)
                # print(f"Saved: {out_path}")

    if t_all:
        print(f"[DONE] {count} frames. Latency ms (mean/median): "
              f"{np.mean(t_all):.1f} / {np.median(t_all):.1f}")
    else:
        print("[DONE] No frames processed.")


if __name__ == "__main__":
    main()


#latency=66.2 ms | peak_gpu=1981.5 MB

#"work_dirs/mybevfusion7_new/mybevfusion7_crossattnaux_painting.py"
#"work_dirs/mybevfusion7_new/epoch_4.pth"
#latency=55.1 ms | peak_gpu=852.7 MB

#work_dirs/mybevfusion7_newv2/epoch_2.pth
#work_dirs/mybevfusion7_newv2/mybevfusion7_crossattnaux_paintingv2.py
#54.7 ms | peak_gpu=852.7 MB

#work_dirs/mybevfusion9_new/epoch_6.pth
#work_dirs/mybevfusion9_new/mybevfusion9_new.py
#59.4 ms | peak_gpu=795.9 MB

#work_dirs/mybevfusion9_new2/epoch_4.pth
#work_dirs/mybevfusion9_new2/mybevfusion9_new2.py
#59.3 ms | peak_gpu=797.8 MB

#original
# Params (total/trainable): 40.80M / 39.80M
# ~Model size (fp32): 163.2 MB

# Params (total/trainable): 38.28M / 38.28M
# ~Model size (fp32): 153.1 MB

# Params (total/trainable): 38.61M / 11.09M
# ~Model size (fp32): 154.4 MB