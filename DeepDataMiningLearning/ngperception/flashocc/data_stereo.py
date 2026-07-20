"""
flashocc.data_stereo
====================
Build the BEVDet4D-stereo `img_inputs` tuple for a nuScenes keyframe, faithfully following FlashOcc's
`PrepareImageInputs` (test path) — so the ported BEVStereo4DOCC can be fed real val data and reproduce
the supervised ceiling (mIoU 37.84). Camera-only; no BEVDet info-pkl needed (built from the devkit).

img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda], where the view
axis is ordered frame-major: [f0_cam0..cam5, f1_cam0..cam5, f2_cam0..cam5] with f0=current keyframe,
f1/f2 = previous keyframes (temporal + stereo ref). Test aug only (resize+center-crop, no flip/rot).
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
INPUT_HW = (256, 704)                                    # (fH, fW)
SRC_HW = (900, 1600)
_MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # mmlab ImageNet norm (0-255 scale)
_STD = np.array([58.395, 57.12, 57.375], np.float32)


def _normalize(pil):
    # FlashOcc mmlabNormalize: imnormalize(np.array(PIL_RGB), mean, std, to_rgb=True). The to_rgb
    # cvtColor(BGR2RGB) REVERSES the already-RGB channels -> the trained model sees BGR. Match it.
    a = np.asarray(pil, np.float32)[..., ::-1]              # RGB -> BGR (the to_rgb reversal)
    a = (a - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def _sample_aug(H, W):
    """Test-time resize + center crop (matches PrepareImageInputs.sample_augmentation, is_train=False)."""
    fH, fW = INPUT_HW
    resize = float(fW) / float(W)                           # 704/1600 = 0.44
    resize_dims = (int(W * resize), int(H * resize))        # (704, 396)
    newW, newH = resize_dims
    crop_h = int((1 - 0.0) * newH) - fH                     # 396 - 256 = 140
    crop_w = int(max(0, newW - fW) / 2)                     # 0
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    return resize, resize_dims, crop


def _img_transform(pil, resize, resize_dims, crop):
    """Apply resize+crop; return transformed PIL + post_rot(3x3)+post_tran(3) encoding it."""
    img = pil.resize(resize_dims).crop(crop)
    post_rot = torch.eye(2) * resize
    post_tran = torch.zeros(2) - torch.Tensor(crop[:2])     # no flip/rotate at test
    R = torch.eye(3); T = torch.zeros(3)
    R[:2, :2] = post_rot; T[:2] = post_tran
    return img, R, T


def _sensor_transforms(nusc, sd_token):
    """sensor2ego (4x4) + ego2global (4x4) for a camera sample_data, from calib + ego pose."""
    from pyquaternion import Quaternion
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    s2e = torch.eye(4); s2e[:3, :3] = torch.Tensor(Quaternion(cs["rotation"]).rotation_matrix)
    s2e[:3, 3] = torch.Tensor(cs["translation"])
    e2g = torch.eye(4); e2g[:3, :3] = torch.Tensor(Quaternion(ep["rotation"]).rotation_matrix)
    e2g[:3, 3] = torch.Tensor(ep["translation"])
    return s2e, e2g


def _prev_keyframes(nusc, sample_token, n):
    """The current sample token + n previous keyframe tokens (repeat the oldest if the scene starts)."""
    toks = [sample_token]
    s = nusc.get("sample", sample_token)
    for _ in range(n):
        s = nusc.get("sample", s["prev"]) if s["prev"] else s
        toks.append(s["token"])
    return toks                                             # [curr, prev1, prev2, ...]


def build_img_inputs(nusc, sample_token, num_frame=3):
    """Return the BEVDet4D img_inputs tuple (unbatched) for one keyframe. imgs are CAM-major
    (cam*num_frame+frame); calib tensors are FRAME-major (frame*6+cam) — matches the ported model.
    Adjacent frames reuse the CURRENT frame's per-cam image augmentation (as in the source)."""
    frame_toks = _prev_keyframes(nusc, sample_token, num_frame - 1)   # [key, prev1, prev2]
    grid = {}                                              # (frame, cam) -> dict of tensors
    aug = {}                                               # cam -> (resize, dims, crop) from current frame
    for fi, tok in enumerate(frame_toks):
        sample = nusc.get("sample", tok)
        for ci, cam in enumerate(CAMS):
            sd_tok = sample["data"][cam]
            sd = nusc.get("sample_data", sd_tok)
            pil = Image.open(nusc.get_sample_data_path(sd_tok)).convert("RGB")
            if fi == 0:
                aug[ci] = _sample_aug(pil.height, pil.width)
            resize, dims, crop = aug[ci]                   # adjacent reuse current aug
            img, R, T = _img_transform(pil, resize, dims, crop)
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            s2e, e2g = _sensor_transforms(nusc, sd_tok)
            grid[(fi, ci)] = dict(img=_normalize(img), s2e=s2e, e2g=e2g,
                                  K=torch.Tensor(np.array(cs["camera_intrinsic"], np.float32)), R=R, T=T)
    imgs = [grid[(f, c)]["img"] for c in range(6) for f in range(num_frame)]      # CAM-major
    idx_fm = [(f, c) for f in range(num_frame) for c in range(6)]                 # FRAME-major
    s2e = [grid[k]["s2e"] for k in idx_fm]; e2g = [grid[k]["e2g"] for k in idx_fm]
    K = [grid[k]["K"] for k in idx_fm]; pr = [grid[k]["R"] for k in idx_fm]; pt = [grid[k]["T"] for k in idx_fm]
    return (torch.stack(imgs), torch.stack(s2e), torch.stack(e2g),
            torch.stack(K), torch.stack(pr), torch.stack(pt), torch.eye(3))
