"""
Waymo v2.0.1 → mmdet3d-compatible wrapper for cross-dataset zero-shot eval.

Wraps the existing ``Waymo3DDataset`` (which returns ``(lidar [N,5],
target_dict)``) and emits the dict shape ``model.test_step`` expects from
our B10c-style BEVFusionCA model:

    {
        'inputs':       {
            'points': [points_t (N, 5)],          # vehicle frame, with time_delta
            'img':    [imgs_t (Nc, 3, H, W)],     # post-ImageAug3D
        },
        'data_samples': [Det3DDataSample(metainfo=dict(
            cam2img=...,           # [Nc, 4, 4]   (intrinsics + homogeneous)
            lidar2img=...,         # [Nc, 4, 4]   composed for the view-transform
            cam2lidar=...,         # [Nc, 4, 4]
            lidar2cam=...,         # [Nc, 4, 4]
            img_aug_matrix=...,    # [Nc, 4, 4]   center-crop + resize transform
            lidar_aug_matrix=...,  # [4, 4]       identity at eval
            box_type_3d=LiDARInstance3DBoxes,
            sample_idx=int,
            num_pts_feats=5,
            ori_cam2img=...,       # for record-keeping
            ori_lidar2img=...,     # for record-keeping
            # Waymo-specific extras (for downstream eval):
            waymo_segment=str,
            waymo_timestamp=int,
            waymo_gt_boxes_vehicle=np.ndarray[M, 7],
            waymo_gt_types=np.ndarray[M] (Waymo class IDs 1-4),
        ))],
    }

**Camera mapping (Waymo 5 cams → nuScenes-style 6 slots):**

  nuScenes order (canonical for B10c eval):
    CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

  Waymo cam IDs (1-5):
    1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT

  Mapping into the 6 nuScenes slots:
    slot 0 (FRONT)        ← Waymo cam 1 (FRONT)
    slot 1 (FRONT_RIGHT)  ← Waymo cam 3 (FRONT_RIGHT)
    slot 2 (FRONT_LEFT)   ← Waymo cam 2 (FRONT_LEFT)
    slot 3 (BACK)         ← BLACK DUMMY  (Waymo has no rear camera)
    slot 4 (BACK_LEFT)    ← Waymo cam 4 (SIDE_LEFT)
    slot 5 (BACK_RIGHT)   ← Waymo cam 5 (SIDE_RIGHT)

  The dummy back image is all-zeros + matching extrinsic (whatever; will
  be projected past the BEV range). Its lidar2img is set to an
  invalidating large translation so the view-transform won't lift any
  points onto it.

**Image preprocessing:** match B10c's eval pipeline (256×704 final, mean
crop 0.0, scale 0.48). Waymo input is 1280×1920. The img_aug_matrix
encodes the per-camera 2D affine: scale 0.48 + crop to top-of-image.

This is a research wrapper — not optimized; loads one frame at a time.
"""
from __future__ import annotations
import os
import os.path as osp
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

# Make ``DeepDataMiningLearning.*`` importable when run from this file's dir.
_HERE = osp.dirname(osp.abspath(__file__))
_DDML_ROOT = osp.dirname(osp.dirname(_HERE))
if _DDML_ROOT not in sys.path:
    sys.path.insert(0, _DDML_ROOT)

from DeepDataMiningLearning.detection3d.dataset_waymo3dv201 import Waymo3DDataset


# -----------------------------------------------------------------------------
# Camera slot mapping
# -----------------------------------------------------------------------------

# Six "slots" used by B10c's image branch. The order matches the nuScenes
# config's iteration of info['images'] (which is the dict-insertion order:
# FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT).
NUS_CAM_ORDER = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
NUM_CAMS = 6   # B10c expects 6 cams

# Map nuScenes slot index -> Waymo camera ID (or None for the BACK slot).
SLOT_TO_WAYMO_CAM = {
    0: 1,    # CAM_FRONT       ← Waymo FRONT
    1: 3,    # CAM_FRONT_RIGHT ← Waymo FRONT_RIGHT
    2: 2,    # CAM_FRONT_LEFT  ← Waymo FRONT_LEFT
    3: None, # CAM_BACK        ← (Waymo has no rear cam — dummy black image)
    4: 4,    # CAM_BACK_LEFT   ← Waymo SIDE_LEFT
    5: 5,    # CAM_BACK_RIGHT  ← Waymo SIDE_RIGHT
}

# Waymo string name → int sensor ID (fallback when 'camera_id' is missing).
WAYMO_CAM_NAME_TO_ID = {
    'FRONT': 1, 'FRONT_LEFT': 2, 'FRONT_RIGHT': 3,
    'SIDE_LEFT': 4, 'SIDE_RIGHT': 5,
    # Just in case the base dataset uppercases inconsistently:
    'Front': 1, 'Front_Left': 2, 'Front_Right': 3,
    'Side_Left': 4, 'Side_Right': 5,
}


# -----------------------------------------------------------------------------
# Calibration helpers
# -----------------------------------------------------------------------------

def _load_camera_calibration(cam_calib_parquet: str) -> Dict[int, Dict]:
    """
    Parse Waymo v2.0.1 camera_calibration parquet → dict keyed by camera
    ID 1-5 with intrinsic + 4x4 extrinsic (vehicle ← camera).

    Per the v2.0.1 schema, ``extrinsic.transform`` is the **vehicle←camera**
    transform stored as a 16-element row-major array. So
    ``T_vc[3,3] = transform`` and ``cam2vehicle = T_vc`` directly.
    """
    df = pq.read_table(cam_calib_parquet).to_pandas()
    out: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        cam_id = int(row['key.camera_name'])
        K = np.eye(4, dtype=np.float64)
        K[0, 0] = row['[CameraCalibrationComponent].intrinsic.f_u']
        K[1, 1] = row['[CameraCalibrationComponent].intrinsic.f_v']
        K[0, 2] = row['[CameraCalibrationComponent].intrinsic.c_u']
        K[1, 2] = row['[CameraCalibrationComponent].intrinsic.c_v']
        extrinsic = np.asarray(
            row['[CameraCalibrationComponent].extrinsic.transform'],
            dtype=np.float64,
        ).reshape(4, 4)
        out[cam_id] = {
            'intrinsic_4x4':       K,                     # camera-frame → pixel
            'cam2vehicle':         extrinsic,             # 4x4, vehicle ← camera
            'width':  int(row['[CameraCalibrationComponent].width']),
            'height': int(row['[CameraCalibrationComponent].height']),
        }
    return out


def _build_lidar2img(
    cam_calib: Dict[int, Dict],
    img_aug_matrix_per_cam: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Compose the per-camera 4x4 ``lidar2img`` that BEVFusionCA needs.

    Waymo has only ONE coordinate frame on the ego side ("vehicle"). The
    pipeline uses it as both "lidar" and "ego" since lidar GTs are already
    expressed in vehicle frame. So:

        lidar2cam   =  inv(cam2vehicle)
        cam2lidar   =  cam2vehicle
        cam2img     =  intrinsic_4x4   (Waymo: front-x, left-y, up-z
                                        camera-frame  →  fx*x/z + cx pixel
                                        ⚠ this is the OpenCV convention; Waymo's
                                        camera frame conforms after we account
                                        for the vehicle←camera transform which
                                        encodes the camera's rotation)
        lidar2img   =  img_aug @ cam2img @ lidar2cam

    Returns (lidar2img, cam2img, cam2lidar, lidar2cam) — each a list of
    length NUM_CAMS, each entry a 4x4 numpy array.
    """
    # Convention: Waymo's vehicle frame is +x forward, +y left, +z up.
    # Waymo's camera frame (per official docs) is +x forward, +y left,
    # +z up — but the intrinsic projection uses the *standard* OpenCV
    # convention (+z forward, +x right, +y down). The transform from
    # Waymo's camera frame to OpenCV's frame is a fixed rotation:
    #     R_opencv_from_waymo_cam = [[0, -1,  0],   # opencv X = -waymo Y (right)
    #                                [0,  0, -1],   # opencv Y = -waymo Z (down)
    #                                [1,  0,  0]]   # opencv Z =  waymo X (forward)
    # We compose this into cam2img.
    R_oc_wc = np.array([
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0,  0.0],
    ])
    T_oc_wc = np.eye(4, dtype=np.float64)
    T_oc_wc[:3, :3] = R_oc_wc

    lidar2img_list, cam2img_list = [], []
    cam2lidar_list, lidar2cam_list = [], []

    for slot in range(NUM_CAMS):
        waymo_cam = SLOT_TO_WAYMO_CAM[slot]
        if waymo_cam is None or waymo_cam not in cam_calib:
            # Dummy BACK camera: place far behind the vehicle so its
            # frustum doesn't intersect anything in BEV (-100 m on +x).
            dummy_K = np.eye(4, dtype=np.float64)
            dummy_K[0, 0] = 1000.0
            dummy_K[1, 1] = 1000.0
            dummy_K[0, 2] = 352.0
            dummy_K[1, 2] = 128.0

            dummy_c2v = np.eye(4, dtype=np.float64)
            dummy_c2v[:3, 3] = [-100.0, 0.0, 1.5]   # camera 100 m behind ego
            dummy_R = np.array([
                [-1.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0],
                [ 0.0, 0.0, 1.0],
            ])
            dummy_c2v[:3, :3] = dummy_R

            lidar2cam = np.linalg.inv(dummy_c2v)
            cam2lidar = dummy_c2v
            cam2img = T_oc_wc @ dummy_K   # OpenCV-conv + dummy intrinsics
        else:
            calib = cam_calib[waymo_cam]
            cam2vehicle = calib['cam2vehicle']        # 4x4
            K = calib['intrinsic_4x4']                # 4x4
            lidar2cam = np.linalg.inv(cam2vehicle)
            cam2lidar = cam2vehicle
            cam2img = K @ T_oc_wc                     # opencv-conv intrinsic

        # Apply ImageAug3D's 2D affine to the projection matrix.
        img_aug = img_aug_matrix_per_cam[slot]        # 4x4
        lidar2img = img_aug @ cam2img @ lidar2cam     # composes all three

        lidar2img_list.append(lidar2img.astype(np.float32))
        cam2img_list.append(cam2img.astype(np.float32))
        cam2lidar_list.append(cam2lidar.astype(np.float32))
        lidar2cam_list.append(lidar2cam.astype(np.float32))

    return lidar2img_list, cam2img_list, cam2lidar_list, lidar2cam_list


# -----------------------------------------------------------------------------
# Image preprocessing (matches our B10c eval pipeline)
# -----------------------------------------------------------------------------

def _resize_crop_to_target(
    image_hwc_uint8: np.ndarray,
    final_h: int = 256,
    final_w: int = 704,
    resize: float = 0.48,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replicate ``ImageAug3D`` (test mode) on a single image:
      - resize whole image by ``resize`` factor
      - crop bottom-aligned strip of ``final_h × final_w``
    Returns (resized_cropped_chw_float32, img_aug_matrix_4x4).
    """
    import cv2
    H, W = image_hwc_uint8.shape[:2]
    new_W, new_H = int(W * resize), int(H * resize)
    resized = cv2.resize(image_hwc_uint8, (new_W, new_H),
                         interpolation=cv2.INTER_LINEAR)

    # Bot-aligned crop matching test_pipeline (bot_pct_lim=[0.0, 0.0] →
    # crop_h = newH - final_h, center-x):
    crop_h_start = max(0, new_H - final_h)
    crop_w_start = max(0, (new_W - final_w) // 2)
    cropped = resized[crop_h_start:crop_h_start + final_h,
                      crop_w_start:crop_w_start + final_w]
    # Pad if Waymo image is too small (rare; nuScenes is 1600×900, Waymo
    # 1920×1280 — both safely larger than 704×256 at 0.48 scale).
    if cropped.shape[0] < final_h or cropped.shape[1] < final_w:
        pad = np.zeros((final_h, final_w, 3), dtype=cropped.dtype)
        pad[:cropped.shape[0], :cropped.shape[1]] = cropped
        cropped = pad

    # Build the 4x4 img_aug_matrix that maps ORIGINAL pixel coords →
    # AUGMENTED pixel coords (used to compose lidar2img above).
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = resize
    M[1, 1] = resize
    M[0, 3] = -float(crop_w_start)
    M[1, 3] = -float(crop_h_start)

    # CHW float32 normalization (mean/std matches torchvision/Imagenet).
    img_chw = cropped.transpose(2, 0, 1).astype(np.float32)
    return img_chw, M


# -----------------------------------------------------------------------------
# Main wrapper
# -----------------------------------------------------------------------------

class WaymoMMDet3DZeroShot:
    """
    Per-frame iterator yielding mmdet3d-compatible inputs for B10c
    zero-shot evaluation. Caches per-segment camera calibration so we
    don't re-read parquet for every frame.
    """

    WAYMO_CAM_DECODED_KEY = '[CameraImageComponent].image'

    def __init__(self,
                 root_dir: str,
                 split: str = 'validation',
                 max_frames: Optional[int] = None,
                 num_sweeps: int = 1,
                 image_final_size: Tuple[int, int] = (256, 704),
                 image_resize: float = 0.48):
        self.base = Waymo3DDataset(
            root_dir=root_dir, split=split,
            max_frames=max_frames, num_sweeps=num_sweeps,
        )
        self.image_h, self.image_w = image_final_size
        self.image_resize = float(image_resize)

        self.cam_calib_dir = osp.join(root_dir, split, 'camera_calibration')
        self._calib_cache: Dict[str, Dict[int, Dict]] = {}

    def __len__(self) -> int:
        return len(self.base.frame_index)

    def _segment_calib(self, fname: str) -> Dict[int, Dict]:
        if fname not in self._calib_cache:
            self._calib_cache[fname] = _load_camera_calibration(
                osp.join(self.cam_calib_dir, fname),
            )
        return self._calib_cache[fname]

    def get_frame(self, idx: int) -> Dict[str, Any]:
        """Return one mmdet3d-style frame ready for ``model.test_step(...)``.

        Returns dict with keys ``inputs`` and ``data_samples``, mirroring
        the structure ``model.test_step`` expects in our pipeline.
        """
        from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

        # 1) base loader fetches lidar + per-cam image + GT boxes (in Waymo
        # vehicle frame: +x forward, +y left, +z up).
        lidar, target = self.base[idx]            # lidar:[N,5], target: dict
        fname, ts, seg = self.base.frame_index[idx]
        cam_calib = self._segment_calib(fname)

        # ---------------------------------------------------------------
        # FRAME ALIGNMENT — empirical result, May 2026.
        #
        # The nuScenes-trained model produces predictions in a y-MIRRORED
        # frame relative to the input. Verified by brute-forcing all 8
        # axis-aligned 2D transforms of model output: only `y_flip`
        # recovers Vehicle AP (0.014 → 0.539 @ score=0.10 ≤50m,
        # 0.675 @ score=0.01 ≤50m).
        #
        # The fix is post-processing in eval_waymo_zeroshot.py — model
        # predictions get a y-flip before GT matching. We do NOT modify
        # the LiDAR points or projection matrices fed to the model: that
        # was tried and produced WORSE AP (V5: 0.032) because it changes
        # what the model "sees", which alters the prediction frame in
        # ways the simple post-flip can't fully undo.
        # ---------------------------------------------------------------

        # 2) build per-cam image array + img_aug_matrix.
        # ``target['surround_views']`` is a list of per-camera dicts with
        # 'image' (H, W, 3 uint8), 'camera_name' (1-5).
        surround = target.get('surround_views', []) or []
        # Base dataset stores int sensor ID under 'camera_id' (1=FRONT, 2=F_L,
        # 3=F_R, 4=S_L, 5=S_R). 'camera_name' is the *string* like 'FRONT'.
        cam_id_to_img = {}
        for d in surround:
            if 'image' not in d:
                continue
            cid = d.get('camera_id')
            if cid is None:
                cn = d.get('camera_name')
                cid = WAYMO_CAM_NAME_TO_ID.get(str(cn))
            if cid is None:
                continue
            cam_id_to_img[int(cid)] = d['image']
        imgs_chw = []
        img_aug_per_cam: List[np.ndarray] = []
        for slot in range(NUM_CAMS):
            waymo_id = SLOT_TO_WAYMO_CAM[slot]
            if waymo_id is not None and waymo_id in cam_id_to_img:
                arr = cam_id_to_img[waymo_id]
                if torch.is_tensor(arr):
                    arr = arr.cpu().numpy()
                arr = np.asarray(arr, dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    img_chw, M = _resize_crop_to_target(
                        arr, final_h=self.image_h, final_w=self.image_w,
                        resize=self.image_resize,
                    )
                else:
                    img_chw = np.zeros((3, self.image_h, self.image_w),
                                       dtype=np.float32)
                    M = np.eye(4, dtype=np.float64)
            else:
                # Dummy BACK: black image + identity aug matrix.
                img_chw = np.zeros((3, self.image_h, self.image_w),
                                   dtype=np.float32)
                M = np.eye(4, dtype=np.float64)
            imgs_chw.append(img_chw)
            img_aug_per_cam.append(M)

        imgs_tensor = torch.from_numpy(np.stack(imgs_chw, axis=0)).float()  # [Nc,3,H,W]

        # 3) project matrices — unchanged from raw vehicle frame
        lidar2img, cam2img, cam2lidar, lidar2cam = _build_lidar2img(
            cam_calib, img_aug_per_cam,
        )

        # 4) GT (Waymo vehicle frame — same convention as nuScenes LiDAR)
        gt_boxes = target.get('boxes_3d')
        gt_labels = target.get('labels')
        if torch.is_tensor(gt_boxes):
            gt_boxes_np = gt_boxes.cpu().numpy()
        else:
            gt_boxes_np = np.asarray(gt_boxes) if gt_boxes is not None else np.zeros((0, 7))
        if torch.is_tensor(gt_labels):
            gt_labels_np = gt_labels.cpu().numpy()
        else:
            gt_labels_np = np.asarray(gt_labels) if gt_labels is not None else np.zeros((0,), dtype=np.int64)

        # 5) Build the data_sample. NB: B10c reads ``lidar2img`` /
        # ``cam2img`` / ``img_aug_matrix`` / ``lidar_aug_matrix`` from
        # metainfo as **lists of 4x4 arrays** indexed by camera slot.
        ds = Det3DDataSample()
        ds.set_metainfo(dict(
            sample_idx=int(idx),
            num_pts_feats=5,
            box_type_3d=LiDARInstance3DBoxes,
            lidar2img=lidar2img,
            cam2img=cam2img,
            cam2lidar=cam2lidar,
            lidar2cam=lidar2cam,
            ori_lidar2img=lidar2img,    # no original-vs-augmented distinction
            ori_cam2img=cam2img,        # since we already baked img_aug in
            img_aug_matrix=img_aug_per_cam,
            lidar_aug_matrix=np.eye(4, dtype=np.float64),
            # Waymo extras for downstream eval / debugging:
            waymo_segment=str(seg),
            waymo_timestamp=int(ts),
            waymo_gt_boxes_vehicle=gt_boxes_np.astype(np.float32),
            waymo_gt_types=gt_labels_np.astype(np.int64),
        ))

        return dict(
            inputs=dict(points=[lidar.float()], img=[imgs_tensor]),
            data_samples=[ds],
        )


# -----------------------------------------------------------------------------
# Convenience: iterate
# -----------------------------------------------------------------------------

def iterate(root_dir: str, split: str = 'validation',
            max_frames: Optional[int] = None, **kwargs):
    """Generator that yields ``frame_dict`` for every frame in the dataset."""
    ds = WaymoMMDet3DZeroShot(root_dir=root_dir, split=split,
                               max_frames=max_frames, **kwargs)
    for i in range(len(ds)):
        yield ds.get_frame(i)
