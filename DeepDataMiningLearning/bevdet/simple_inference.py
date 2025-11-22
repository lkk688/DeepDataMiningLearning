import os, json, argparse, contextlib
import torch, numpy as np
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmengine.registry import init_default_scope
from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes, DepthInstance3DBoxes

def make_default_metas(num_cams=6, img_shape=(256, 704, 3), use_lidar_coords=True):
    """
    Build a minimal-but-correct metainfo dict for BEVFusion.predict().
    Fill real calibration matrices in production; identities are only for a dry run.
    """
    import numpy as np

    # identity intrinsics/extrinsics as placeholders (replace with real calibration)
    K34 = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3,1), dtype=np.float32)], axis=1)  # 3x4
    T44 = np.eye(4, dtype=np.float32)

    metas = dict(
        # image geometry
        img_shape=img_shape,          # (H, W, C)
        ori_shape=img_shape,
        pad_shape=img_shape,
        num_cams=num_cams,

        # camera intrinsics/extrinsics per camera
        cam2img=[K34.copy() for _ in range(num_cams)],
        ori_cam2img=[K34.copy() for _ in range(num_cams)],
        lidar2cam=[T44.copy() for _ in range(num_cams)],
        cam2lidar=[T44.copy() for _ in range(num_cams)],
        lidar2img=[T44.copy() for _ in range(num_cams)],  # some pipelines also read this

        # optional ego/global (safe defaults)
        lidar2ego=T44.copy(),
        ego2lidar=T44.copy(),
        ego2global=T44.copy(),
        timestamp=0.0,

        # REQUIRED by head.predict_by_feat: callable class + mode
        box_type_3d = LiDARInstance3DBoxes if use_lidar_coords else CameraInstance3DBoxes,
        box_mode_3d = Box3DMode.LIDAR     if use_lidar_coords else Box3DMode.CAM,

        # classes (adjust to your dataset)
        classes = ['car','truck','construction_vehicle','bus','trailer',
                   'barrier','motorcycle','bicycle','pedestrian','traffic_cone'],
    )
    return metas

# add near top of your file
def _import_det3d_sample():
    try:
        from mmdet3d.structures.det3d_data_sample import Det3DDataSample
    except Exception:
        # some versions re-export at package root
        from mmdet3d.structures import Det3DDataSample
    return Det3DDataSample

def make_data_samples_from_metas(metas_list):
    """metas_list: list of dicts (one per batch item)."""
    Det3DDataSample = _import_det3d_sample()
    data_samples = []
    for m in metas_list:
        ds = Det3DDataSample()
        # IMPORTANT: use the official API so .metainfo exists
        ds.set_metainfo(m)
        data_samples.append(ds)
    return data_samples

def build_model_from_cfg(cfg_path, ckpt_path, device='cuda'):
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    model.to(device).eval()
    return model, cfg

def make_multimodal_batch(imgs, pts, metas):
    """
    imgs: FloatTensor [B,N,C,H,W]
    pts:  list[FloatTensor], len=B
    metas: dict for a single sample (B=1)
    """
    assert imgs.ndim == 5 and imgs.size(0) == 1, "Expect B=1 for this helper."
    batch_inputs_dict = {'imgs': imgs, 'points': pts}
    # BEVFusion.predict wants List[Det3DDataSample]
    data_samples = make_data_samples_from_metas([metas])  # B=1
    return {'batch_inputs_dict': batch_inputs_dict, 'data_samples': data_samples}



@torch.no_grad()
def predict_one(model, batch, amp_dtype=None):
    """
    batch: dict with keys:
      - 'batch_inputs_dict': {'imgs': ..., 'points': ...}
      - 'data_samples': List[Det3DDataSample]  (B elements)
    """
    model.eval()
    ctx = (torch.autocast('cuda', dtype=amp_dtype)
           if (amp_dtype is not None and torch.cuda.is_available()) else contextlib.nullcontext())
    with ctx:
        bid = batch['batch_inputs_dict']
        ds  = batch['data_samples']  # List[Det3DDataSample]
        if hasattr(model, 'predict'):
            return model.predict(bid, ds)
        # generic fallback (mmdet3d BaseDet3D forwards to predict anyway)
        return model.forward(bid, ds, mode='predict')
    
def set_attn_chunk_if_any(model, chunk):
    if chunk is None: return False
    ok = False
    for m in model.modules():
        if hasattr(m, 'attn_chunk'):
            try: m.attn_chunk = int(chunk); ok = True
            except Exception: pass
    return ok

def main_old():
    import argparse, os, json, time
    import numpy as np
    import torch
    from mmengine import Config, init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.registry import MODELS
    from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes, Det3DDataSample

    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str, default="work_dirs/mybevfusion7_new/mybevfusion7_crossattnaux_painting.py", help="Path to MMDet3D config .py")
    #parser.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_new/epoch_4.pth", help="Path to model checkpoint .pth")
    #parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument("--config", type=str, default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to MMDet3D config .py")
    parser.add_argument("--checkpoint", type=str, default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Path to model checkpoint .pth")
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--points', default="data/nuscenes/samples/LIDAR_TOP/n015-2018-11-21-19-58-31+0800__LIDAR_TOP__1542801733448313.pcd.bin", help='LiDAR file (.bin/.npy/.pcd) or npy path')
    parser.add_argument('--images', nargs='+', default=["data/nuscenes/samples/CAM_FRONT/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801733412460.jpg"], help='Multi-view image files in order (front,...).')
    parser.add_argument('--img-hw', type=int, nargs=2, default=[256, 704], help='(H W) after your pipeline resize')
    parser.add_argument('--bf16', action='store_true', help='use torch.bfloat16 autocast')
    parser.add_argument('--attn-chunk', type=int, default=None, help='Override view_transform.attn_chunk if available')
    parser.add_argument('--warmup', type=int, default=5, help='warmup iterations')
    parser.add_argument('--out-json', default='pred_one.json')
    args = parser.parse_args()

    # --------------------------
    # 1) Build model from config
    # --------------------------
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    model = MODELS.build(cfg.model)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    # Optionally set attn_chunk
    if args.attn_chunk is not None and hasattr(model, 'view_transform'):
        vt = getattr(model, 'view_transform', None)
        if vt is not None and hasattr(vt, 'attn_chunk'):
            try:
                vt.attn_chunk = int(args.attn_chunk)
                print(f'[INFO] Set view_transform.attn_chunk = {vt.attn_chunk}')
            except Exception as e:
                print(f'[WARN] Failed to set attn_chunk: {e}')

    # --------------------------------
    # 2) Load inputs (points + images)
    # --------------------------------
    def load_points_any(path: str) -> torch.Tensor:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.bin':
            pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        elif ext == '.npy':
            pts = np.load(path)
        elif ext == '.pcd':
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pts.shape[1] == 3:
                # add dummy intensity
                pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), np.float32)], axis=1)
        else:
            raise ValueError(f'Unsupported point file: {path}')
        return torch.from_numpy(pts)

    def load_image_chw(path: str, hw: tuple[int,int]) -> torch.Tensor:
        # Minimal loader: PIL -> resize -> CHW float32 0..255 (BEVFusion preproc will normalize)
        from PIL import Image
        H, W = hw
        img = Image.open(path).convert('RGB').resize((W, H))
        arr = np.asarray(img, dtype=np.uint8)  # H W 3
        chw = torch.from_numpy(arr).permute(2,0,1).contiguous().float()  # 3 H W
        return chw

    points_tensor = load_points_any(args.points).to(args.device, non_blocking=True)
    H, W = args.img_hw
    imgs_list = [load_image_chw(p, (H, W)).to(args.device, non_blocking=True) for p in args.images]
    # Model expects [B, N, C, H, W]
    imgs_bnc_hw = torch.stack(imgs_list, dim=0).unsqueeze(0)  # (1, N, 3, H, W)

    # ------------------------------------------
    # 3) Build a correct Det3DDataSample + metas
    # ------------------------------------------
    def make_default_metas(num_cams: int, img_shape_hw3, use_lidar_coords=True):
        K34 = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3,1), dtype=np.float32)], axis=1)  # 3x4
        T44 = np.eye(4, dtype=np.float32)
        box_type = LiDARInstance3DBoxes if use_lidar_coords else CameraInstance3DBoxes
        box_mode = Box3DMode.LIDAR if use_lidar_coords else Box3DMode.CAM
        metas = dict(
            img_shape=img_shape_hw3,
            ori_shape=img_shape_hw3,
            pad_shape=img_shape_hw3,
            num_cams=num_cams,
            cam2img=[K34.copy() for _ in range(num_cams)],
            ori_cam2img=[K34.copy() for _ in range(num_cams)],
            lidar2cam=[T44.copy() for _ in range(num_cams)],
            cam2lidar=[T44.copy() for _ in range(num_cams)],
            lidar2img=[T44.copy() for _ in range(num_cams)],
            lidar2ego=T44.copy(),
            ego2lidar=T44.copy(),
            ego2global=T44.copy(),
            timestamp=0.0,
            box_type_3d=box_type,   # callable class, not string!
            box_mode_3d=box_mode,   # enum, not string!
            classes=['car','truck','construction_vehicle','bus','trailer',
                     'barrier','motorcycle','bicycle','pedestrian','traffic_cone'],
        )
        return metas

    ds = Det3DDataSample()
    ds.set_metainfo(make_default_metas(
        num_cams=len(args.images),
        img_shape_hw3=(H, W, 3),
        use_lidar_coords=True
    ))

    # ----------------------------
    # 4) Build batch and predict()
    # ----------------------------
    batch_inputs_dict = {
        'points': [points_tensor],   # list length == B
        'imgs': imgs_bnc_hw          # (B, N, 3, H, W)
    }
    data_samples = [ds]              # List[Det3DDataSample], length == B

    # Warmup + one timed call
    amp_dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else None
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            if amp_dtype is not None:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    _ = model.predict(batch_inputs_dict, data_samples)
            else:
                _ = model.predict(batch_inputs_dict, data_samples)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        t0 = time.perf_counter()
        if amp_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                preds = model.predict(batch_inputs_dict, data_samples)
        else:
            preds = model.predict(batch_inputs_dict, data_samples)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = (time.perf_counter() - t0) * 1000.0  # ms

    # ----------------------------
    # 5) Report + dump predictions
    # ----------------------------
    # Basic GPU memory peek
    mem_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_bytes = torch.cuda.max_memory_allocated()
        mem_mb = float(mem_bytes) / (1024.0**2)

    print(f'[OK] Inference: {dt:.2f} ms, peak GPU mem ~ {mem_mb:.1f} MB')
    # Normalize a lightweight serializable dict
    out = {
        'latency_ms': dt,
        'peak_mem_mb': mem_mb,
        'attn_chunk': getattr(getattr(model, 'view_transform', object()), 'attn_chunk', None),
        'predictions': []
    }
    # preds is List[Det3DDataSample]
    def dsample_to_dict(dsample: Det3DDataSample):
        outd = {}
        pred = getattr(dsample, 'pred_instances_3d', None)
        if pred is not None:
            # Convert to plain lists for JSON
            if hasattr(pred, 'scores_3d') and pred.scores_3d is not None:
                outd['scores_3d'] = pred.scores_3d.detach().cpu().tolist()
            if hasattr(pred, 'labels_3d') and pred.labels_3d is not None:
                outd['labels_3d'] = pred.labels_3d.detach().cpu().tolist()
            if hasattr(pred, 'bboxes_3d') and pred.bboxes_3d is not None:
                # Instance3DBoxes.tensor: (N, 7) or (N, 9) depending on vel/yaw
                outd['bboxes_3d'] = pred.bboxes_3d.tensor.detach().cpu().tolist()
        return outd

    for d in preds:
        out['predictions'].append(dsample_to_dict(d))

    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[OK] Wrote {args.out_json}')

from nuscenes_iterator import iter_nuscenes_samples
# 在 simple_inference.py 里添加或替换相应逻辑
from pathlib import Path
import numpy as np
import cv2
import torch

from mmdet3d.structures import Det3DDataSample
from nuscenes_iterator import iter_nuscenes_samples

# 你的图像预处理，要与训练一致（bgr_to_rgb/mean/std/resize）
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def load_img_to_chw(fp: str, final_hw=(256, 704)) -> torch.Tensor:
    img = cv2.imread(fp, cv2.IMREAD_COLOR)  # BGR
    if final_hw is not None:
        img = cv2.resize(img, (final_hw[1], final_hw[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    # bgr_to_rgb
    img = img[..., ::-1]
    # norm
    img = (img - MEAN) / STD
    # HWC->CHW tensor
    chw = torch.from_numpy(img.copy()).permute(2,0,1).contiguous().float()
    return chw

def load_lidar_as_tensor(fp: str) -> torch.Tensor:
    pts = np.fromfile(fp, dtype=np.float32).reshape(-1, 5)  # nuScenes .bin 通常是 5 列
    return torch.from_numpy(pts.copy()).float()

def one_sample_to_batch(lidar_path: str, img_paths: list, metainfo: dict) -> tuple:
    imgs = [load_img_to_chw(p) for p in img_paths]     # list[N_cam] of CHW
    points = load_lidar_as_tensor(lidar_path)          # [N,5]

    # 包装成 Det3DDataSample —— 只放 metainfo 即可
    ds = Det3DDataSample(metainfo=metainfo)

    # batch_inputs_dict 要与你的模型 forward/predict 接口匹配
    # 大多数 BEVFusion 实现接受：
    # - 'points': list[Tensor], 每元素是一帧点云
    # - 'img' 或 'imgs': list[N_cam] CHW
    batch_inputs_dict = {
        'points': [points],
        'imgs': imgs
    }
    data_samples = [ds]
    return batch_inputs_dict, data_samples

def normalize_batch_inputs_dict_for_predict(bid: dict, device):
    out = dict(bid)  # shallow copy
    imgs = out.get('imgs', out.get('img', None))
    if imgs is None:
        return out

    # If imgs is list[Tensor(C,H,W)], stack to [N,C,H,W]
    if isinstance(imgs, list):
        assert len(imgs) > 0, "empty imgs list"
        # ensure all on device and contiguous float
        imgs = [t.to(device, non_blocking=True).contiguous().float() for t in imgs]
        imgs = torch.stack(imgs, dim=0)             # [N,C,H,W]
    elif torch.is_tensor(imgs):
        # Accept [N,C,H,W] or [1,N,C,H,W] → squeeze to [N,C,H,W]
        if imgs.ndim == 5 and imgs.size(0) == 1:
            imgs = imgs[0]
        assert imgs.ndim == 4, f"expected [N,C,H,W], got {imgs.shape}"
        imgs = imgs.to(device, non_blocking=True).contiguous().float()
    else:
        raise TypeError(f"Unsupported imgs type: {type(imgs)}")

    out['imgs'] = imgs
    # points → list[Tensor] still fine
    if 'img' in out:  # keep only the key the model reads
        out.pop('img', None)
    return out

import torch

def _to_4x4(m):
    """Accepts torch/np tensors of (..., 3x3), (3x4) or (4x4) and returns (..., 4x4) torch tensor."""
    if not torch.is_tensor(m):
        m = torch.as_tensor(m)
    if m.shape[-2:] == (4, 4):
        return m
    eye = torch.eye(4, dtype=m.dtype, device=m.device).expand(*m.shape[:-2], 4, 4).clone()
    if m.shape[-2:] == (3, 3):
        eye[..., :3, :3] = m
        # implicit zero translation
        return eye
    if m.shape[-2:] == (3, 4):
        eye[..., :3, :4] = m
        return eye
    raise ValueError(f"Unsupported matrix shape {tuple(m.shape)}; expected (...,3,3), (...,3,4) or (...,4,4)")

def normalize_metas_to_4x4(batch_input_metas):
    """
    Ensure metas keys used by BEVFusion VT are [B, N, 4, 4]:
    'img_aug_matrix', 'lidar_aug_matrix', and (optionally) 'lidar2img'/'cam2img' to homogeneous.
    """
    keys_44 = ['img_aug_matrix', 'lidar_aug_matrix']
    opt_keys = ['lidar2img', 'cam2img', 'lidar2cam', 'cam2lidar']  # some repos use these in VT
    for meta in batch_input_metas:
        for k in keys_44:
            if k in meta:
                m = torch.stack([_to_4x4(mi) for mi in meta[k]], dim=0)  # [N,4,4]
                meta[k] = m
            else:
                # supply identity if absent
                N = len(meta.get('img_shape', [])) or len(meta.get('cam2img', [])) or 6
                meta[k] = torch.eye(4).repeat(N, 1, 1)
        # (Optional) homogenize camera matrices if you want consistent math later
        for k in opt_keys:
            if k in meta:
                meta[k] = torch.stack([_to_4x4(mi) for mi in meta[k]], dim=0)
    return batch_input_metas

def normalize_imgs_for_bevfusion(bid, device):
    """Make imgs be [B,N,C,H,W] FloatTensor, and move to device."""
    imgs = bid.get('imgs', bid.get('img', None))
    if imgs is None:
        return bid
    if isinstance(imgs, list):  # list of N tensors [C,H,W]
        imgs = torch.stack([im.to(device, non_blocking=True).contiguous().float() for im in imgs], dim=0)  # [N,C,H,W]
        imgs = imgs.unsqueeze(0)  # [1,N,C,H,W]
    elif torch.is_tensor(imgs):
        imgs = imgs.to(device, non_blocking=True).contiguous().float()
        if imgs.ndim == 4:  # [N,C,H,W] -> [1,N,C,H,W]
            imgs = imgs.unsqueeze(0)
        elif imgs.ndim != 5:
            raise ValueError(f"Unexpected imgs.ndim={imgs.ndim}, shape={imgs.shape}")
    else:
        raise TypeError(f"Unsupported imgs type: {type(imgs)}")
    out = dict(bid)
    out['imgs'] = imgs
    out.pop('img', None)
    return out


import os
import cv2
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

# Optional (recommended) to make a true Det3DDataSample:
try:
    from mmdet3d.structures import Det3DDataSample
    from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, CameraInstance3DBoxes
    from mmdet3d.registry import DATA_SAMPLERS  # not used; just to show typical deps
except Exception:
    Det3DDataSample = None
    LiDARInstance3DBoxes = None
    CameraInstance3DBoxes = None

# --------------------------------------------------------------------------------------
# 1) Basic loaders
# --------------------------------------------------------------------------------------

def load_lidar_points(lidar_path: str) -> torch.Tensor:
    """
    Load LiDAR as a Tensor [M, D]. Supports .bin (KITTI) and .npy.
    - KITTI velodyne .bin: float32 [x,y,z,reflectance]
    - NuScenes LIDAR_TOP .bin: may need your own loader; .npy is easiest.
    """
    ext = os.path.splitext(lidar_path)[1].lower()
    if ext == ".bin":
        arr = np.fromfile(lidar_path, dtype=np.float32)
        if arr.size % 4 == 0:
            arr = arr.reshape(-1, 4)
        else:
            # If your format includes timestamp/intensity etc., reshape accordingly
            raise ValueError(f"Unexpected .bin size {arr.size}; expected multiple of 4.")
    elif ext == ".npy":
        arr = np.load(lidar_path)  # shape [M, D]
    else:
        raise ValueError(f"Unsupported lidar format: {ext}")
    return torch.from_numpy(arr).contiguous()

def load_image_chw(img_path: str) -> torch.Tensor:
    """
    Load one RGB image and return tensor [3, H, W], float32 in [0,255] range.
    (Det3DDataPreprocessor in your model will normalize/scale if configured.)
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> RGB
    chw = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()  # [3,H,W]
    return chw

# --------------------------------------------------------------------------------------
# 2) Meta normalization helpers (handles dict or Det3DDataSample)
# --------------------------------------------------------------------------------------

def _to_4x4(m):
    """Return a 4x4 torch tensor from (...,3x3), (...,3x4), or (...,4x4)."""
    if isinstance(m, np.ndarray):
        m = torch.from_numpy(m)
    elif not torch.is_tensor(m):
        m = torch.as_tensor(m)
    if m.shape[-2:] == (4, 4):
        return m
    eye = torch.eye(4, dtype=m.dtype, device=m.device).expand(*m.shape[:-2], 4, 4).clone()
    if m.shape[-2:] == (3, 3):
        eye[..., :3, :3] = m
        return eye
    if m.shape[-2:] == (3, 4):
        eye[..., :3, :4] = m
        return eye
    raise ValueError(f"Unsupported matrix shape {tuple(m.shape)}; expected (...,3,3), (...,3,4) or (...,4,4)")

def _stack_4x4_list(seq, device=None, dtype=None):
    """Convert a sequence of matrices to [N,4,4] torch tensor."""
    mats = []
    for x in seq:
        t = _to_4x4(x)
        if device is not None: t = t.to(device)
        if dtype is not None: t = t.to(dtype)
        mats.append(t)
    return torch.stack(mats, dim=0)

def _is_datasample(x):
    return (Det3DDataSample is not None) and isinstance(x, Det3DDataSample)

def _get_metadict(meta):
    if _is_datasample(meta):
        return meta.metainfo
    if isinstance(meta, dict):
        return meta
    raise TypeError(f"Unsupported meta type {type(meta)}; expected Det3DDataSample or dict")

def _set_metakey(meta, key, value):
    if _is_datasample(meta) and hasattr(meta, 'set_metainfo'):
        d = dict(meta.metainfo)
        d[key] = value
        meta.set_metainfo(d)
    elif _is_datasample(meta):
        meta.metainfo[key] = value
    else:
        meta[key] = value

def normalize_metas_to_4x4(batch_data_samples):
    """
    Ensure BEVFusion-relevant metas are [N,4,4] per camera:
      - 'img_aug_matrix' (required) : list len N of (3x3 or 4x4); fallback identities
      - 'lidar_aug_matrix' (required) : list len N; fallback identities
      - Optional (homogenize if present): 'lidar2img','cam2img','lidar2cam','cam2lidar'
    Mutates in place and returns the same list.
    """
    keys_44 = ['img_aug_matrix', 'lidar_aug_matrix']
    opt_keys = ['lidar2img', 'cam2img', 'lidar2cam', 'cam2lidar']

    for meta in batch_data_samples:
        md = _get_metadict(meta)

        # infer N from any present per-camera key
        N = 0
        for k in ('img_aug_matrix', 'cam2img', 'lidar2img'):
            if k in md and md[k] is not None:
                N = len(md[k]); break
        if N == 0:
            N = md.get('num_cams', 6)  # fallback (nuScenes typical)

        device = None

        # Required keys
        for k in keys_44:
            seq = md.get(k, None)
            if seq is None or len(seq) == 0:
                stacked = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
            else:
                stacked = _stack_4x4_list(seq, device=device)
            _set_metakey(meta, k, stacked)

        # Optional keys
        for k in opt_keys:
            seq = md.get(k, None)
            if seq is None or len(seq) == 0:
                continue
            stacked = _stack_4x4_list(seq, device=device)
            _set_metakey(meta, k, stacked)

    return batch_data_samples

def normalize_imgs_for_bevfusion(batch_inputs_dict, device):
    """
    Ensure 'imgs' exists and is [B,N,C,H,W] FloatTensor on `device`.
    Accepts:
      - 'img': list of N [C,H,W] tensors, or a single [N,C,H,W]/[B,N,C,H,W] tensor
      - 'imgs': same as above
    """
    imgs = batch_inputs_dict.get('imgs', None)
    if imgs is None:
        imgs = batch_inputs_dict.get('img', None)
    if imgs is None:
        return batch_inputs_dict  # lidar-only

    if isinstance(imgs, list):
        # [N, C, H, W]
        imgs = torch.stack([im.to(device, non_blocking=True).contiguous().float()
                            for im in imgs], dim=0).unsqueeze(0)  # -> [1,N,C,H,W]
    elif torch.is_tensor(imgs):
        imgs = imgs.to(device, non_blocking=True).contiguous().float()
        if imgs.ndim == 4:  # [N,C,H,W] -> [1,N,C,H,W]
            imgs = imgs.unsqueeze(0)
        elif imgs.ndim != 5:
            raise ValueError(f"Unexpected imgs.ndim={imgs.ndim}, shape={imgs.shape}")
    else:
        raise TypeError(f"Unsupported imgs type: {type(imgs)}")

    out = dict(batch_inputs_dict)
    out['imgs'] = imgs
    out.pop('img', None)
    return out

import torch

def _is_empty_value(v) -> bool:
    """Return True only if value is meaningfully empty."""
    if v is None:
        return True
    if torch.is_tensor(v):
        return v.numel() == 0
    if isinstance(v, (list, tuple, dict, str)):
        return len(v) == 0
    return False

def _ensure_3d(mats: torch.Tensor) -> torch.Tensor:
    # Accept [N,*,*] or [1,N,*,*]; squeeze leading 1 if present
    if not torch.is_tensor(mats):
        mats = torch.as_tensor(mats)
    if mats.ndim == 4 and mats.size(0) == 1:
        mats = mats.squeeze(0)
    if mats.ndim != 3:
        raise ValueError(f"Expected [N,*,*] or [1,N,*,*], got {tuple(mats.shape)}")
    return mats

def _to4x4_batch(mats) -> torch.Tensor:
    """Normalize per-camera matrices to [N,4,4]. Supports 3x3, 3x4, 4x4."""
    if isinstance(mats, (list, tuple)):
        mats = torch.stack([torch.as_tensor(m) for m in mats], dim=0)
    mats = _ensure_3d(mats)

    N, R, C = mats.shape
    if (R, C) == (4, 4):
        return mats

    out = torch.eye(4, dtype=mats.dtype, device=mats.device).expand(N, 4, 4).clone()
    if (R, C) == (3, 3):
        out[:, :3, :3] = mats
        return out
    if (R, C) == (3, 4):
        out[:, :3, :4] = mats
        return out
    raise ValueError(f"Unsupported matrix shape {R}x{C} (expected 3x3, 3x4, or 4x4)")

def ensure_img_aug_in_meta(data_meta: dict, num_cams: int):
    """Ensure data_meta['img_aug_matrix'] exists and is [N,4,4]."""
    key = 'img_aug_matrix'
    if (key not in data_meta) or _is_empty_value(data_meta[key]):
        data_meta[key] = torch.eye(4, dtype=torch.float32).expand(num_cams, 4, 4).clone()
    else:
        data_meta[key] = _to4x4_batch(data_meta[key])

def ensure_lidar_aug_in_meta(data_meta: dict, num_cams: int):
    """Same normalization for optional lidar_aug_matrix."""
    key = 'lidar_aug_matrix'
    if (key not in data_meta) or _is_empty_value(data_meta[key]):
        data_meta[key] = torch.eye(4, dtype=torch.float32).expand(num_cams, 4, 4).clone()
    else:
        data_meta[key] = _to4x4_batch(data_meta[key])
# --------------------------------------------------------------------------------------
# 3) Main API: one_sample_to_batch
# --------------------------------------------------------------------------------------

def one_sample_to_batch(
    lidar_path: str,
    img_paths: List[str],
    metainfo: Dict,
    device: Optional[torch.device] = None,
    box_space: str = "lidar"
) -> Tuple[Dict, List]:
    """
    Build (batch_inputs_dict, batch_data_samples) for BEVFusion.predict()

    Args:
      lidar_path: path to a single LiDAR file (.bin, .npy)
      img_paths: list of N camera image paths (ordered consistently with metas)
      metainfo: dict with per-camera matrices (lists of length N):
        Required (lists, len N): 'img_aug_matrix', 'lidar_aug_matrix'
        Optional (lists, len N): 'cam2img','lidar2img','cam2lidar','lidar2cam'
        Optional scalars: 'pcd_rotation','pcd_scale_factor','pcd_trans'
        Optional: 'ori_shape','img_shape','pad_shape' per camera, etc.
        Set 'box_type_3d' & 'box_mode_3d' if you will decode predictions later.
      device: torch.device to place image tensors
      box_space: 'lidar' or 'camera' — used to pre-set box_type_3d
    Returns:
      batch_inputs_dict: {'imgs': [B,N,C,H,W], 'points': [list length B]}
      batch_data_samples: [Det3DDataSample or dict], len B
    """
    # 1) Load LiDAR
    points = load_lidar_points(lidar_path)  # [43400, 4] [M, D]
    points_list = [points]  # B=1

    # 2) Load images
    imgs_chw = [load_image_chw(p) for p in img_paths]  # list of N [3,H,W] tensors

    # 3) Construct batch_inputs_dict
    batch_inputs_dict = {
        'img': imgs_chw,     # normalized to 'imgs' later
        'points': points_list
    }

    # 4) Build DataSample (or dict) + metainfo
    N = len(img_paths)
    data_meta = dict(metainfo)  # shallow copy; you manage ordering

    # Required keys: if not present, create identity lists
    # def _ensure_list_of_identities(key):
    #     if key not in data_meta or not data_meta[key]:
    #         data_meta[key] = [np.eye(4, dtype=np.float32) for _ in range(N)]

    # _ensure_list_of_identities('img_aug_matrix')
    # _ensure_list_of_identities('lidar_aug_matrix')

    # Helpful hints for downstream code
    data_meta.setdefault('num_cams', N)
    data_meta.setdefault('img_paths', img_paths)
    data_meta.setdefault('lidar_path', lidar_path)

    # Box type for prediction decoding (HEADS use this on predict_by_feat)
    if box_space == "lidar" and LiDARInstance3DBoxes is not None:
        data_meta.setdefault('box_type_3d', LiDARInstance3DBoxes)
        data_meta.setdefault('box_mode_3d', None)
    elif box_space == "camera" and CameraInstance3DBoxes is not None:
        data_meta.setdefault('box_type_3d', CameraInstance3DBoxes)
        data_meta.setdefault('box_mode_3d', None)
    else:
        # Safe fallback if classes not importable; you can set later
        pass

    if Det3DDataSample is not None:
        ds = Det3DDataSample()
        ds.set_metainfo(data_meta)
        batch_data_samples = [ds]
    else:
        batch_data_samples = [data_meta]  # plain dict also works if your model accepts it

    # 5) Normalize metas to [N,4,4] & images to [B,N,C,H,W] on device
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_data_samples = normalize_metas_to_4x4(batch_data_samples)
    batch_inputs_dict = normalize_imgs_for_bevfusion(batch_inputs_dict, device)

    # Sanity prints (optional)
    # print("imgs:", batch_inputs_dict['imgs'].shape)  # [1,N,C,H,W]
    # print("points:", batch_inputs_dict['points'][0].shape)
    # print("img_aug_matrix:", _get_metadict(batch_data_samples[0])['img_aug_matrix'].shape)

    return batch_inputs_dict, batch_data_samples

def _downgrade_to_3x4(mi, key):
    # If you previously normalized to 4x4, convert back to 3x4 for LSS expectations
    if key in mi:
        M = torch.as_tensor(mi[key])
        if M.ndim == 3 and M.shape[1:] == (4,4):
            mi[key] = M[:, :3, :4]  # keep rotation+translation rows only
            
def main():
    import argparse
    import time
    from pathlib import Path
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    parser = argparse.ArgumentParser("Simple BEVFusion/CrossAttnLSS inference (no evaluator)")
    # model / device
    parser.add_argument("--config", type=str, default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py", help="Path to MMDet3D config .py")
    parser.add_argument("--checkpoint", type=str, default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="Path to model checkpoint .pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", type=str, default="fp32", choices=["bf16", "fp16", "fp32"],
                        help="Autocast dtype for inference")
    parser.add_argument("--attn-chunk", type=int, default=None,
                        help="Override model.view_transform.attn_chunk if the module exposes it")

    # dataset selection
    parser.add_argument("--dataset", type=str, default="nuscenes",
                        choices=["nuscenes", "any", "kitti"],
                        help="Basic source of frames. 'nuscenes' uses the NuScenes devkit; "
                             "'any' expects you to assemble batches manually elsewhere; "
                             "'kitti' not implemented here (placeholder).")
    # nuscenes args
    parser.add_argument("--nus-root", type=str, default="data/nuscenes",
                        help="NuScenes dataroot (folder containing v1.0-*)")
    parser.add_argument("--nus-version", type=str, default=None,
                        help="NuScenes version: v1.0-mini, v1.0-trainval, or v1.0-test. "
                             "If omitted, we auto-detect by scanning the root.")

    # runtime / io
    parser.add_argument("--max-count", type=int, default=8, help="Max frames to run (-1 = all)")
    parser.add_argument("--out-dir", type=str, default="simple_infer_out",
                        help="Where to save qualitative outputs (png/ply)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="If set, Open3D saves PLY instead of opening a window.")
    parser.add_argument("--save-vis", action="store_true", default=True,
                        help="If set, save qualitative 2D and Open3D visualizations.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- build model -----
    # (If you already have a builder in-file, use that. Otherwise this assumes one.)
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model, config_model = build_model_from_cfg(args.config, args.checkpoint, args.device)  # <- your existing builder
    model.eval()

    # optional: override attn_chunk on the fly
    if args.attn_chunk is not None:
        try:
            if hasattr(model, "view_transform") and hasattr(model.view_transform, "attn_chunk"):
                old = getattr(model.view_transform, "attn_chunk")
                setattr(model.view_transform, "attn_chunk", int(args.attn_chunk))
                print(f"[INFO] attn_chunk: {old} -> {model.view_transform.attn_chunk}")
            else:
                print("[INFO] Model does not expose view_transform.attn_chunk; skipping override.")
        except Exception as e:
            print(f"[WARN] Failed to set attn_chunk: {e}")

    # autocast dtype
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None
    }[args.amp]

    # ----- choose iterator -----
    if args.dataset == "nuscenes":
        iterator = iter_nuscenes_samples(
            dataroot=args.nus_root,
            version=args.nus_version,
            max_count=args.max_count
        )
    elif args.dataset == "kitti":
        raise NotImplementedError("KITTI iterator is not implemented in this main().")
    else:  # "any"
        raise NotImplementedError(
            "'any' mode expects you to add a custom iterator that yields "
            "(lidar_path, image_paths, metainfo, basename)."
        )

    # ----- run -----
    num_done = 0
    lat_ms = []
    peak_mem_mb = 0.0

    with torch.no_grad():
        for lidar_path, image_paths, metainfo, basename in iterator:
            # lidar_path = "/path/to/lidar.bin"
            # img_paths  = [
            #     "/path/to/CAM_FRONT.png",
            #     "/path/to/CAM_FRONT_RIGHT.png",
            #     "/path/to/CAM_BACK_RIGHT.png",
            #     "/path/to/CAM_BACK.png",
            #     "/path/to/CAM_BACK_LEFT.png",
            #     "/path/to/CAM_FRONT_LEFT.png",
            # ]
            
            # metainfo = {
            #     "img_aug_matrix":  [1, 6, 3, 3]
            #     "lidar_aug_matrix": [1, 4, 4]
            #     "cam2img":   [cam_intr_3x3_or_4x4_per_cam, ...], [6, 4, 4]
            #     "lidar2img": [lidar2img_3x4_or_4x4_per_cam, ...], [6, 4, 4]
            #     "cam2lidar": [cam2lidar_4x4_per_cam, ...], [6, 4, 4]
            #     "lidar2cam": [lidar2cam_4x4_per_cam, ...],[6, 4, 4]
            # }
            N = len(image_paths)  # e.g., 6 for nuScenes
            ensure_img_aug_in_meta(metainfo, num_cams=N) #img_aug_matrix: [1, 6, 3, 3] lidar_aug_matrix [1, 4, 4], 'num_pts_feats'=5
            # optional if you rely on it
            if 'lidar_aug_matrix' in metainfo or True:  # keep / drop as you need
                ensure_lidar_aug_in_meta(metainfo, num_cams=N) #img_aug_matrix: [6, 4, 4] lidar_aug_matrix [1, 4, 4]
            
            for ds in batch_data_samples:
                mi = ds.metainfo
                _downgrade_to_3x4(mi, 'img_aug_matrix')
                _downgrade_to_3x4(mi, 'lidar_aug_matrix')
            
            # assemble batch for one sample
            batch_inputs_dict, batch_data_samples = one_sample_to_batch(
                lidar_path=lidar_path,
                img_paths=image_paths,
                metainfo=metainfo,
                device=None,            # auto GPU/CPU
                box_space="lidar"       # or "camera" depending on your head/postproc
            )
            # forward
            if amp_dtype is None:
                t0 = time.perf_counter()
                preds = model.predict(batch_inputs_dict, batch_data_samples)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                dt = (time.perf_counter() - t0) * 1000.0
            else:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    t0 = time.perf_counter()
                    preds = model.predict(batch_inputs_dict, batch_data_samples)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    dt = (time.perf_counter() - t0) * 1000.0

            lat_ms.append(dt)
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated()
                peak_mem_mb = max(peak_mem_mb, mem / (1024.0 * 1024.0))

            # unpack predictions
            pred0 = preds[0]
            instances = getattr(pred0, "pred_instances_3d", None)
            nobj = 0
            if instances is not None and hasattr(instances, "labels_3d"):
                nobj = int(instances.labels_3d.shape[0])

            print(f"[{num_done:04d}] {basename} | objects={nobj} | {dt:.1f} ms "
                  f"| peak_gpu={peak_mem_mb:.1f} MB")

            # optional qualitative saves
            if args.save_vis and instances is not None:
                # Convert to plain dict for your visualize_with_open3d helper
                pred_dict = dict(
                    bboxes_3d=instances.bboxes_3d.tensor.cpu().numpy() if hasattr(instances.bboxes_3d, "tensor") else
                              (instances.bboxes_3d.cpu().numpy() if hasattr(instances.bboxes_3d, "cpu") else
                               np.asarray(instances.bboxes_3d)),
                    labels_3d=instances.labels_3d.cpu().numpy(),
                    scores_3d=instances.scores_3d.cpu().numpy() if hasattr(instances, "scores_3d") else None,
                    metainfo=dict(classes=getattr(pred0, "metainfo", {}).get("classes", None))
                )

                # image for 2D overlay: pick the first camera
                img0 = image_paths[0] if len(image_paths) > 0 else None
                # no GT here; pass empty
                try:
                    visualize_with_open3d(
                        lidar_file=lidar_path,
                        predictions_dict=pred_dict,
                        gt_bboxes=[],
                        out_dir=str(out_dir),
                        basename=str(basename),
                        headless=args.headless,
                        img_file=img0,
                        calib_file=None  # (optional) you can pass a packed calib if your drawer supports it
                    )
                except Exception as e:
                    print(f"[WARN] Visualization failed for {basename}: {e}")

            num_done += 1
            if 0 <= args.max_count <= num_done:
                break

    # ----- summary -----
    import json
    import numpy as np
    summ = dict(
        count=num_done,
        latency_ms_mean=float(np.mean(lat_ms) if len(lat_ms) else 0.0),
        latency_ms_p50=float(np.percentile(lat_ms, 50) if len(lat_ms) else 0.0),
        latency_ms_p90=float(np.percentile(lat_ms, 90) if len(lat_ms) else 0.0),
        latency_ms_p95=float(np.percentile(lat_ms, 95) if len(lat_ms) else 0.0),
        latency_ms_max=float(np.max(lat_ms) if len(lat_ms) else 0.0),
        peak_mem_mb=float(peak_mem_mb),
        attn_chunk=getattr(getattr(model, "view_transform", object()), "attn_chunk", None),
        amp=args.amp,
        dataset=args.dataset,
        nus_root=args.nus_root if args.dataset == "nuscenes" else None,
        nus_version=args.nus_version if args.dataset == "nuscenes" else None
    )
    with open(out_dir / "simple_infer_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(f"[OK] Wrote {out_dir / 'simple_infer_summary.json'}")

if __name__ == "__main__":
    main()