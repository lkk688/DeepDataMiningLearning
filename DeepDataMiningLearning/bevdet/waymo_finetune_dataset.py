"""
WaymoFineTuneDataset: an mmdet3d-compatible Dataset that adapts our
``Waymo3DDataset`` v2-parquet loader for BEVFusion fine-tuning.

Wraps the existing ``WaymoMMDet3DZeroShot`` wrapper (which handles
LiDAR/camera/calibration loading) and exposes the
``BaseDataset``+``Det3DDataset`` interface that mmengine's training loop
expects.

Key design:
  * Info pkl entries store ``lidar_path = 'waymo://{seg}/{ts}/{idx}'``
    (built by ``build_waymo_finetune_infos.py``). The integer ``idx``
    is the frame index in the underlying ``Waymo3DDataset.frame_index``.
  * ``parse_data_info`` is called per sample; we lazily look up the
    LiDAR + images via the underlying loader.
  * Y-FLIP convention: the info-pkl GT boxes are already y-flipped (cy→-cy,
    yaw→-yaw). We apply the SAME y-flip to LiDAR points and projection
    matrices at load time, so the model sees a consistent y-flipped frame
    during fine-tune. This matches the V6 eval where predictions are
    y-flipped back to vehicle frame after inference.
"""

from __future__ import annotations

import os.path as osp
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

import sys
sys.path.insert(0, '/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning')
from DeepDataMiningLearning.detection3d.dataset_waymo_mmdet3d import (
    WaymoMMDet3DZeroShot, _load_camera_calibration, SLOT_TO_WAYMO_CAM,
    NUM_CAMS)


# Module-level singleton: one loader per Waymo root + split. Avoids re-
# scanning parquets every time the DataLoader spawns a new worker.
_LOADER_CACHE: Dict[str, WaymoMMDet3DZeroShot] = {}
# (seg, ts) -> frame_idx lookup for each loader. The info.pkl built by
# build_waymo_finetune_infos_fast.py uses a global counter as sample_idx,
# which doesn't match Waymo3DDataset.frame_index ordering — we look up
# the actual loader index from (seg, ts) instead.
_FRAME_LOOKUP: Dict[str, Dict] = {}


def _get_loader(root: str, split: str) -> WaymoMMDet3DZeroShot:
    key = f'{root}::{split}'
    if key not in _LOADER_CACHE:
        loader = WaymoMMDet3DZeroShot(
            root_dir=root, split=split, max_frames=None)
        _LOADER_CACHE[key] = loader
        # Build (seg, ts) → idx lookup. frame_index entry = (fname, ts, seg).
        lookup = {}
        for i, (fname, ts, seg) in enumerate(loader.base.frame_index):
            lookup[(seg, int(ts))] = i
        _FRAME_LOOKUP[key] = lookup
    return _LOADER_CACHE[key]


def _lookup_frame_idx(root: str, split: str, seg: str, ts: int) -> int:
    _get_loader(root, split)
    key = f'{root}::{split}'
    return _FRAME_LOOKUP[key][(seg, int(ts))]


_WAYMO_URI_RE = re.compile(r'^waymo://([^/]+)/(\d+)/(\d+)(?:/.+)?$')
_WAYMO_V1_URI_RE = re.compile(r'^waymo_v1://([^/]+)/(\d+)(?:/.+)?$')


def _parse_uri(uri: str):
    """Parse either waymo:// (v2) or waymo_v1:// (extracted v1) URI.
    Returns (variant, segment, ts_or_None, frame_idx)."""
    m = _WAYMO_URI_RE.match(uri)
    if m:
        return 'v2', m.group(1), int(m.group(2)), int(m.group(3))
    m = _WAYMO_V1_URI_RE.match(uri)
    if m:
        return 'v1', m.group(1), None, int(m.group(2))
    raise ValueError(f'not a waymo:// or waymo_v1:// URI: {uri}')


# Y-flip matrices used at load time (same as the V6 prediction-side flip).
M_YFLIP_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
M_YFLIP_4 = np.eye(4, dtype=np.float32); M_YFLIP_4[:3, :3] = M_YFLIP_3


@DATASETS.register_module()
class WaymoFineTuneDataset(NuScenesDataset):
    """Drop-in NuScenesDataset for Waymo v2 fine-tuning.

    Inherits NuScenesDataset so existing data pipelines (transforms,
    augmentations, samplers, evaluators) work without modification.

    Args:
        waymo_root: filesystem root for waymo201 (default).
        waymo_split: 'validation' or 'training' for the underlying loader.
        y_flip_on_load: if True, apply M_yflip to LiDAR + projection
            matrices when reading a frame. The info-pkl GT was already
            y-flipped by the builder; setting this False would mean the
            model sees mirrored GT against un-mirrored points (bad).
    """
    METAINFO = {
        'classes': (
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
        ),
    }

    def __init__(self, *args,
                 waymo_root: str = '/fs/atipa/data/rnd-liu/Datasets/waymo201',
                 waymo_split: str = 'validation',
                 y_flip_on_load: bool = True,
                 **kwargs):
        self._waymo_root = waymo_root
        self._waymo_split = waymo_split
        self._y_flip = bool(y_flip_on_load)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------- read
    def parse_data_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve waymo:// or waymo_v1:// URIs and stash routing info."""
        info = dict(info)   # copy
        variant, seg, ts, idx = _parse_uri(info['lidar_points']['lidar_path'])
        info['_waymo_variant'] = variant
        info['_waymo_frame_idx'] = idx
        info['_waymo_seg'] = seg
        if ts is not None:
            info['_waymo_ts'] = ts
        return super().parse_data_info(info)


# ----------------------------------------------------------------- transforms

from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadWaymoFrameFromInfo(BaseTransform):
    """Pipeline transform that materializes LiDAR + images + projection
    matrices from a Waymo info dict.

    Reads ``_waymo_seg / _waymo_ts / _waymo_frame_idx`` (stashed by
    ``WaymoFineTuneDataset.parse_data_info``) and uses the singleton
    ``Waymo3DDataset`` to fetch the actual data. Applies the y-flip to
    LiDAR + projection matrices on the fly when ``y_flip=True``.

    After this transform:
      * results['points']         : (N, 5) float32 tensor in y-flipped frame
      * results['img']            : (6, 3, H, W) float32 tensor
      * results['lidar2img']      : list of 6 (4,4) np arrays
      * results['lidar2cam']      : same
      * results['cam2img']        : same
      * results['img_aug_matrix'] : list of 6 (4,4) np arrays
      * results['lidar_aug_matrix']: (4,4) np array (identity)
    """

    def __init__(self,
                 waymo_root: str = '/fs/atipa/data/rnd-liu/Datasets/waymo201',
                 waymo_split: str = 'validation',
                 waymo_v1_root: str = ('/fs/atipa/data/rnd-liu/MyRepo/'
                                       'DeepDataMiningLearning/data/'
                                       'waymo_v1_extracted'),
                 y_flip: bool = False,
                 num_sweeps: int = 1,
                 sweep_dt_seconds: float = 0.1):
        self.waymo_root = waymo_root
        self.waymo_split = waymo_split
        self.waymo_v1_root = waymo_v1_root
        self.y_flip = bool(y_flip)
        # Multi-sweep config — for the v1 path, we look back `num_sweeps-1`
        # frames in the same segment and transform their points into the
        # current frame's coords via pose deltas. Each past frame's
        # timestamp column is set to -k*sweep_dt_seconds so the model can
        # use it as a temporal feature.
        self.num_sweeps = max(1, int(num_sweeps))
        self.sweep_dt = float(sweep_dt_seconds)

    def transform(self, results: Dict) -> Dict:
        from mmdet3d.structures.points import LiDARPoints
        variant = results.get('_waymo_variant', 'v2')
        if variant == 'v1':
            return self._transform_v1(results)
        # ----- v2 path (parquet) -----
        loader = _get_loader(self.waymo_root, self.waymo_split)
        seg = str(results['_waymo_seg'])
        ts  = int(results['_waymo_ts'])
        idx = _lookup_frame_idx(self.waymo_root, self.waymo_split, seg, ts)
        frame = loader.get_frame(idx)
        ds = frame['data_samples'][0]
        meta = ds.metainfo

        # LiDAR — wrap in LiDARPoints so downstream transforms
        # (translate/rotate/scale) can dispatch correctly.
        points = frame['inputs']['points'][0]   # (N, 5) torch tensor
        if self.y_flip:
            xyz = points[:, :3]
            yflip = torch.from_numpy(M_YFLIP_3.T).to(xyz.dtype)
            xyz_flip = xyz @ yflip
            points = torch.cat([xyz_flip, points[:, 3:]], dim=1)
        results['points'] = LiDARPoints(points, points_dim=points.shape[-1])

        # Cameras: Pack3DDetInputs expects a list of (H, W, 3) numpy arrays
        # (one per camera), not a stacked tensor. Convert from (Nc, 3, H, W)
        # torch float → list of (H, W, 3) numpy uint8/float arrays.
        img_t = frame['inputs']['img'][0]   # (Nc, 3, H, W) float
        img_list = []
        for c in range(img_t.shape[0]):
            arr = img_t[c].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
            img_list.append(arr.astype(np.float32))
        results['img'] = img_list

        # Projection matrices (post-multiply by M_yflip if y_flip ON, because
        # the points are now in y-flipped frame).
        lidar2img = meta['lidar2img']   # list of 6 (4,4)
        lidar2cam = meta['lidar2cam']
        cam2img   = meta['cam2img']
        cam2lidar = meta['cam2lidar']
        if self.y_flip:
            lidar2img = [m @ M_YFLIP_4 for m in lidar2img]
            lidar2cam = [m @ M_YFLIP_4 for m in lidar2cam]
            cam2lidar = [M_YFLIP_4 @ m for m in cam2lidar]
        results['lidar2img']      = lidar2img
        results['lidar2cam']      = lidar2cam
        results['cam2img']        = cam2img
        results['cam2lidar']      = cam2lidar
        results['img_aug_matrix'] = meta['img_aug_matrix']
        results['lidar_aug_matrix'] = meta['lidar_aug_matrix']

        # Number of point feats (for downstream voxelization)
        results['num_pts_feats'] = 5

        # mmdet3d / mmcv pipelines expect these shape keys even when no
        # further image transform is applied — set them to the
        # already-resized dims so downstream code doesn't trip on missing keys.
        H, W = img_t.shape[-2], img_t.shape[-1]
        results['ori_shape'] = (H, W)
        results['img_shape'] = (H, W)
        results['pad_shape'] = (H, W)
        results['scale_factor'] = 1.0
        return results

    def __repr__(self):
        return (f'LoadWaymoFrameFromInfo(waymo_root={self.waymo_root}, '
                f'split={self.waymo_split}, '
                f'v1_root={self.waymo_v1_root}, y_flip={self.y_flip})')

    # ------------------------------------------------------------------ v1
    def _transform_v1(self, results: Dict) -> Dict:
        """Load a frame from waymo_v1_extracted/{seg}/f_{idx}.npz + JPEGs.
        Mirrors the v2 path's output schema."""
        from mmdet3d.structures.points import LiDARPoints
        from PIL import Image

        seg = str(results['_waymo_seg'])
        fidx = int(results['_waymo_frame_idx'])
        seg_dir = osp.join(self.waymo_v1_root, seg)
        npz_path = osp.join(seg_dir, f'f_{fidx:04d}.npz')
        d = np.load(npz_path)
        # extract_waymo_v1.py stored points in VEHICLE frame already
        # (frame_utils.convert_range_image_to_point_cloud applies the
        # extrinsic internally). T_v_l is kept for reference only —
        # DO NOT apply it again here.
        lidar = d['lidar'].astype(np.float32)         # (N, 5)
        T_wv_cur = d['pose_world_from_vehicle'].astype(np.float64)

        # ----- Multi-sweep: load past frames + transform into current frame -----
        if self.num_sweeps > 1:
            T_cw_cur = np.linalg.inv(T_wv_cur)
            sweep_pts = [lidar]  # current already has ts_offset=0 in col 4
            for s in range(1, self.num_sweeps):
                past_idx = fidx - s
                if past_idx < 0:
                    break
                past_path = osp.join(seg_dir, f'f_{past_idx:04d}.npz')
                if not osp.exists(past_path):
                    break
                dp = np.load(past_path)
                pts_p = dp['lidar'].astype(np.float32)
                T_wv_p = dp['pose_world_from_vehicle'].astype(np.float64)
                T_cur_from_past = (T_cw_cur @ T_wv_p).astype(np.float32)
                xyz_h = np.concatenate([
                    pts_p[:, :3], np.ones((pts_p.shape[0], 1), dtype=np.float32)
                ], axis=1)
                xyz_cur = (xyz_h @ T_cur_from_past.T)[:, :3]
                pts_p_t = pts_p.copy()
                pts_p_t[:, :3] = xyz_cur
                pts_p_t[:, 4] = -s * self.sweep_dt   # negative time = past
                sweep_pts.append(pts_p_t)
            lidar = np.concatenate(sweep_pts, axis=0).astype(np.float32)

        points = torch.from_numpy(lidar)
        if self.y_flip:
            xyz = points[:, :3]
            yflip = torch.from_numpy(M_YFLIP_3.T).to(xyz.dtype)
            xyz_flip = xyz @ yflip
            points = torch.cat([xyz_flip, points[:, 3:]], dim=1)
        results['points'] = LiDARPoints(points, points_dim=points.shape[-1])

        # Cameras + projection matrices (built from per-cam calib .npz files
        # extracted by extract_waymo_v1.py).
        # Slot mapping mirrors the v2 wrapper:
        #   slot0=FRONT(1), 1=FRONT_RIGHT(3), 2=FRONT_LEFT(2),
        #   3=BACK(dummy), 4=SIDE_LEFT(4), 5=SIDE_RIGHT(5)
        SLOT_TO_WAYMO = {0: 1, 1: 3, 2: 2, 3: None, 4: 4, 5: 5}
        R_OC_WC = np.array([[0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64)

        imgs_list = []          # list of (H, W, 3) float arrays
        l2img_list = []
        l2cam_list = []
        c2img_list = []
        c2lid_list = []
        img_aug_list = []       # bake the resize into projection matrices
        FINAL_H, FINAL_W = 256, 704
        RESIZE = 0.48
        for slot in range(6):
            waymo_id = SLOT_TO_WAYMO[slot]
            if waymo_id is None:
                # Dummy BACK
                img = np.zeros((FINAL_H, FINAL_W, 3), dtype=np.float32)
                imgs_list.append(img)
                l2img_list.append(np.eye(4, dtype=np.float32))
                l2cam_list.append(np.eye(4, dtype=np.float32))
                c2img_list.append(np.eye(4, dtype=np.float32))
                c2lid_list.append(np.eye(4, dtype=np.float32))
                img_aug_list.append(np.eye(4, dtype=np.float32))
                continue
            jpg_path = osp.join(seg_dir, f'f_{fidx:04d}_cam_{waymo_id}.jpg')
            calib_path = osp.join(seg_dir, f'f_{fidx:04d}_cam_{waymo_id}_calib.npz')
            calib = np.load(calib_path)
            K = calib['intrinsic'].astype(np.float64)
            c2v = calib['cam2vehicle'].astype(np.float64)
            W_orig = int(calib['width']); H_orig = int(calib['height'])

            # Load image, resize + crop to FINAL_H × FINAL_W
            pil = Image.open(jpg_path)
            pil = pil.resize((int(W_orig * RESIZE), int(H_orig * RESIZE)),
                              Image.BILINEAR)
            new_W = pil.width; new_H = pil.height
            crop_h = max(0, new_H - FINAL_H); crop_w = max(0, (new_W - FINAL_W) // 2)
            pil = pil.crop((crop_w, crop_h, crop_w + FINAL_W, crop_h + FINAL_H))
            arr = np.asarray(pil, dtype=np.float32)   # (H, W, 3)
            imgs_list.append(arr)

            # img_aug = resize + translate (crop)
            M_aug = np.eye(4, dtype=np.float64)
            M_aug[0, 0] = RESIZE; M_aug[1, 1] = RESIZE
            M_aug[0, 3] = -float(crop_w); M_aug[1, 3] = -float(crop_h)

            lidar2cam = np.linalg.inv(c2v)
            cam2img = K @ R_OC_WC
            lidar2img = M_aug @ cam2img @ lidar2cam

            l2img_list.append(lidar2img.astype(np.float32))
            l2cam_list.append(lidar2cam.astype(np.float32))
            c2img_list.append(cam2img.astype(np.float32))
            c2lid_list.append(c2v.astype(np.float32))
            img_aug_list.append(M_aug.astype(np.float32))

        results['img'] = imgs_list
        results['lidar2img'] = l2img_list
        results['lidar2cam'] = l2cam_list
        results['cam2img']   = c2img_list
        results['cam2lidar'] = c2lid_list
        results['img_aug_matrix'] = img_aug_list
        results['lidar_aug_matrix'] = np.eye(4, dtype=np.float32)
        results['num_pts_feats'] = 5
        results['ori_shape'] = (FINAL_H, FINAL_W)
        results['img_shape'] = (FINAL_H, FINAL_W)
        results['pad_shape'] = (FINAL_H, FINAL_W)
        results['scale_factor'] = 1.0
        return results
