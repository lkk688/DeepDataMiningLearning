#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_resolver.py
A tiny dataset front-end that yields (lidar_file, image_file_or_None, calib_dict_or_None, basename)
for dataset in {'any','kitti','waymokitti','nuscenes'}.

Calib dict format:
  {
    "lidar2img": np.ndarray of shape (3,4)
  }
"""

from __future__ import annotations
import os
import os.path as osp
import json
import glob
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, List

import numpy as np

# ------------------------------
# Utilities
# ------------------------------

def _safe_read_json(p: str) -> Optional[dict]:
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _as_np34(x) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape == (3, 4):
            return arr
    except Exception:
        pass
    return None

def _find_first_file(stem: str, folder: str, exts: Tuple[str, ...]) -> Optional[str]:
    for ext in exts:
        p = osp.join(folder, stem + ext)
        if osp.isfile(p):
            return p
    return None

def _list_dir_with_ext(folder: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    for e in exts:
        files.extend(glob.glob(osp.join(folder, f"*{e}")))
    return sorted(files)

# ------------------------------
# ANY-mode iterator
# ------------------------------

def iter_any(
    lidar_dir: str,
    image_dir: Optional[str] = None,
    calib_dir: Optional[str] = None,
    limit: int = -1
) -> Iterator[Tuple[str, Optional[str], Optional[Dict], str]]:
    """
    ANY mode: we assume flat folders. We pair by basename.
    - LiDAR: required
    - Image: optional
    - Calib: JSON with {"lidar2img": [[...],[...],[...]]}, optional
    """
    assert osp.isdir(lidar_dir), f"lidar-dir not found: {lidar_dir}"
    lidar_files = _list_dir_with_ext(lidar_dir, (".pcd", ".bin", ".npy"))
    if limit > 0:
        lidar_files = lidar_files[:limit]

    for lf in lidar_files:
        stem = Path(lf).stem
        imgf = None
        calib = None

        if image_dir and osp.isdir(image_dir):
            imgf = _find_first_file(stem, image_dir, (".jpg", ".png", ".jpeg", ".bmp"))

        if calib_dir and osp.isdir(calib_dir):
            cj = _find_first_file(stem, calib_dir, (".json",))
            if cj:
                jd = _safe_read_json(cj)
                if jd is not None:
                    m = _as_np34(jd.get("lidar2img", None))
                    if m is not None:
                        calib = {"lidar2img": m}

        yield lf, imgf, calib, stem

# ------------------------------
# KITTI / WaymoKITTI iterator
# ------------------------------

def _read_kitti_calib(calib_file: str) -> Dict[str, np.ndarray]:
    """
    Read a KITTI-style calib.txt and return dict with P2, R0_rect, Tr_velo_to_cam.
    """
    # KITTI format lines: "P2: ...", "R0_rect: ...", "Tr_velo_to_cam: ..."
    P2 = None
    R0_rect = None
    Tr = None
    with open(calib_file, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = np.fromstring(line[len("P2:"):], sep=' ')
                P2 = vals.reshape(3, 4)
            elif line.startswith("R0_rect:") or line.startswith("R_rect:"):
                vals = np.fromstring(line.split(":")[1], sep=' ')
                R0 = vals.reshape(3, 3)
                # make it 4x4
                R0_rect = np.eye(4, dtype=np.float32)
                R0_rect[:3, :3] = R0
            elif line.startswith("Tr_velo_to_cam:") or line.startswith("Tr_velodyne_to_cam:"):
                vals = np.fromstring(line.split(":")[1], sep=' ')
                Tr = np.eye(4, dtype=np.float32)
                Tr[:3, :] = vals.reshape(3, 4)

    if P2 is None or R0_rect is None or Tr is None:
        raise RuntimeError(f"Incomplete calib: {calib_file}")

    return {"P2": P2.astype(np.float32), "R0_rect": R0_rect.astype(np.float32), "Tr_velo_to_cam": Tr.astype(np.float32)}

def _kitti_lidar2img(calib: Dict[str, np.ndarray]) -> np.ndarray:
    """
    3x4 projection: lidar2img = P2 @ R0_rect @ Tr_velo_to_cam
    """
    P2 = calib["P2"]
    R0 = calib["R0_rect"]
    Tr = calib["Tr_velo_to_cam"]
    Rt = (R0 @ Tr)[:3, :]  # 3x4
    return (P2 @ Rt).astype(np.float32)  # 3x4

def iter_kitti_like(
    root: str,
    frame_number: str = "-1",
    split: str = "training",
    use_cam: str = "image_2",
    lidar_rel: str = "velodyne",
    calib_rel: str = "calib"
) -> Iterator[Tuple[str, Optional[str], Optional[Dict], str]]:
    """
    KITTI / Waymo2KITTI structure:
      root/
        training/ (or testing/)
          image_2/
          velodyne/
          calib/
    """
    base = osp.join(root, split)
    img_dir = osp.join(base, use_cam)
    lid_dir = osp.join(base, lidar_rel)
    cal_dir = osp.join(base, calib_rel)

    assert osp.isdir(lid_dir), f"LiDAR dir not found: {lid_dir}"
    if not osp.isdir(img_dir):
        img_dir = None
    if not osp.isdir(cal_dir):
        cal_dir = None

    def _one(stem: str):
        # LiDAR file
        lf = _find_first_file(stem, lid_dir, (".bin", ".pcd", ".npy"))
        if lf is None:
            return
        # Image file
        imgf = _find_first_file(stem, img_dir, (".png", ".jpg", ".jpeg")) if img_dir else None
        # Calib
        cfile = _find_first_file(stem, cal_dir, (".txt",)) if cal_dir else None
        calib = None
        if cfile:
            k = _read_kitti_calib(cfile)
            calib = {"lidar2img": _kitti_lidar2img(k)}
        yield lf, imgf, calib, stem

    if frame_number != "-1":
        stem = str(frame_number).zfill(6)
        yield from _one(stem)
    else:
        # enumerate stems from lidar folder
        for lf in _list_dir_with_ext(lid_dir, (".bin", ".pcd", ".npy")):
            stem = Path(lf).stem
            yield from _one(stem)

# ------------------------------
# nuScenes iterator (devkit)
# ------------------------------

def _try_import_nuscenes():
    try:
        from nuscenes.nuscenes import NuScenes
        from pyquaternion import Quaternion
        return NuScenes, Quaternion
    except Exception as e:
        raise ImportError("nuscenes-devkit is required for --dataset nuscenes. `pip install nuscenes-devkit`.") from e

def _rt_mat(rotation, translation):
    from pyquaternion import Quaternion
    R = Quaternion(rotation).rotation_matrix
    t = np.array(translation, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def _nus_lidar2img(nusc, sample, camera_channel='CAM_FRONT') -> np.ndarray:
    # sample_data
    sd_cam = nusc.get('sample_data', sample['data'][camera_channel])
    sd_lid = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
    cs_lid = nusc.get('calibrated_sensor', sd_lid['calibrated_sensor_token'])
    ep_cam = nusc.get('ego_pose', sd_cam['ego_pose_token'])
    ep_lid = nusc.get('ego_pose', sd_lid['ego_pose_token'])

    K = np.array(cs_cam['camera_intrinsic'], dtype=np.float32)  # (3,3)

    T_lidar_to_ego_lid = _rt_mat(cs_lid['rotation'], cs_lid['translation'])
    T_ego_lid_to_global = _rt_mat(ep_lid['rotation'], ep_lid['translation'])
    T_global_to_ego_cam = np.linalg.inv(_rt_mat(ep_cam['rotation'], ep_cam['translation']))
    T_ego_cam_to_cam   = np.linalg.inv(_rt_mat(cs_cam['rotation'], cs_cam['translation']))

    T_lidar_to_cam = T_ego_cam_to_cam @ T_global_to_ego_cam @ T_ego_lid_to_global @ T_lidar_to_ego_lid
    lidar2cam = T_lidar_to_cam[:3, :]
    lidar2img = K @ lidar2cam
    return lidar2img.astype(np.float32)

def _normalize_nus_version(dataroot: str, version: str | None) -> str:
    """Return a valid nuScenes version folder present under dataroot."""
    want = (version or "").strip()
    valid_names = ["v1.0-mini", "v1.0-trainval", "v1.0-test"]

    # Exact match & exists -> use it
    if want in valid_names and osp.isdir(osp.join(dataroot, want)):
        return want

    # Accept shorthand inputs
    aliases = {
        "mini": "v1.0-mini",
        "trainval": "v1.0-trainval",
        "test": "v1.0-test"
    }
    if want in aliases and osp.isdir(osp.join(dataroot, aliases[want])):
        return aliases[want]

    # Auto-detect by preference order if user didn’t specify or misspelled
    for cand in (want, aliases.get(want, ""), "v1.0-trainval", "v1.0-mini", "v1.0-test"):
        if cand and osp.isdir(osp.join(dataroot, cand)):
            return cand

    # Final fallback: list what exists to help the user
    found = [d for d in valid_names if osp.isdir(osp.join(dataroot, d))]
    msg = [
        f"nuScenes version folder not found under: {dataroot}",
        f"Requested/guessed: {version!r}",
        f"Found versions here: {found if found else 'none'}",
        "Fix by either:",
        "  1) moving/symlinking your data so a valid version folder exists, or",
        "  2) passing --nus-version with one of: v1.0-mini, v1.0-trainval, v1.0-test"
    ]
    raise AssertionError("\n".join(msg))


def iter_nuscenes(
    dataroot: str,
    version: str = "v1.0-mini",
    camera: str = "CAM_FRONT",
    limit: int = -1,
    tokens_file: Optional[str] = None
) -> Iterator[Tuple[str, Optional[str], Optional[Dict], str]]:
    """
    Iterate nuScenes samples with LiDAR_TOP + chosen camera.
    Yields lidar_file (.pcd), image_file, calib={'lidar2img': 3x4}, basename=sample_token.
    """
    NuScenes, _ = _try_import_nuscenes()
    version = _normalize_nus_version(dataroot, version)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    tokens: List[str]
    if tokens_file and osp.isfile(tokens_file):
        with open(tokens_file, "r") as f:
            tokens = [ln.strip() for ln in f if ln.strip()]
    else:
        tokens = [s['token'] for s in nusc.sample]

    if limit > 0:
        tokens = tokens[:limit]

    for token in tokens:
        sample = nusc.get('sample', token)
        # lidar file
        sd_lid = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_file = nusc.get_sample_data_path(sd_lid['token'])
        # image file
        if camera not in sample['data']:
            # fall back to CAM_FRONT if requested channel missing
            camera = 'CAM_FRONT'
        sd_cam = nusc.get('sample_data', sample['data'][camera])
        image_file = nusc.get_sample_data_path(sd_cam['token'])
        # calib
        lidar2img = _nus_lidar2img(nusc, sample, camera_channel=camera)
        calib = {"lidar2img": lidar2img}
        yield lidar_file, image_file, calib, token

# ------------------------------
# Public selector
# ------------------------------

def make_iterator(
    dataset: str,
    input_path: str,
    frame_number: str,
    # any:
    lidar_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    calib_dir: Optional[str] = None,
    # nuscenes:
    nus_version: str = "v1.0-mini",
    nus_camera: str = "CAM_FRONT",
    nus_limit: int = -1,
    nus_tokens_file: Optional[str] = None,
    # kitti-like:
    split: str = "training",
    use_cam: str = "image_2",
) -> Iterator[Tuple[str, Optional[str], Optional[Dict], str]]:
    """
    Dispatcher over dataset types. Returns an iterator over samples.
    """
    dataset = dataset.lower()
    if dataset == "any":
        assert lidar_dir is not None, "--lidar-dir is required for dataset=any"
        return iter_any(lidar_dir, image_dir, calib_dir, limit=nus_limit)
    elif dataset in ("kitti", "waymokitti"):
        return iter_kitti_like(
            root=input_path, frame_number=frame_number, split=split, use_cam=use_cam,
            lidar_rel="velodyne", calib_rel="calib"
        )
    elif dataset == "nuscenes":
        return iter_nuscenes(
            dataroot=input_path, version=nus_version, camera=nus_camera,
            limit=nus_limit, tokens_file=nus_tokens_file
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")