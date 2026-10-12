"""Sensor-level corruptions for the fusion-robustness probe.

A single mmdet3d transform, :class:`SensorCorruption`, is inserted immediately
after the *loading* transforms of whatever test pipeline a model uses.  At that
point the result dict holds raw sensor data in a model-independent form::

    results['img']        list[np.ndarray]  6 x (900, 1600, 3) float32 RGB, 0..255
    results['points']     LiDARPoints       (N, 5) = x y z intensity dt
    results['lidar2cam']  np.ndarray        (6, 4, 4)
    results['cam2img']    np.ndarray        (6, 4, 4)
    results['lidar2img']  np.ndarray        (6, 4, 4)   == cam2img @ lidar2cam
    results['cam2lidar']  np.ndarray        (6, 4, 4)   == inv(lidar2cam)

Corrupting *here* -- before ImageAug3D, PointsRangeFilter and voxelisation --
keeps every condition at the sensor level, so "16-beam LiDAR" means the same
physical thing for every model under test regardless of what its own pipeline
does downstream.  That is the property the cross-model ranking comparison needs.

Randomness is drawn from a generator seeded by ``(seed, condition, sample_id)``,
so a condition produces *identical* corrupted input for every model.  Without
this the ranking comparison would be confounded by corruption noise.

Honest limitations, to be reported with any result produced from this file:

* Corruptions are synthetic, following the usual practice of the BEV robustness
  literature (nuScenes-C / RoboBEV).  They are not measured adverse-weather data.
* ``lidar_beam_decimate`` bins by elevation angle over the *accumulated* 9-sweep
  cloud.  Points from earlier sweeps are ego-motion compensated, so their
  elevation relative to the current pose is only approximately their true beam.
  The decimation is therefore a good approximation of a lower-beam sensor, not
  an exact resampling of one.
* ``cam_fog`` is a uniform airlight blend, not a depth-dependent scattering
  model; it removes contrast without respecting scene depth.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

# nuScenes camera order as emitted by ``BEVLoadMultiViewImageFromFiles``,
# which iterates ``results['images']`` in info-file order.
NUSCENES_CAMS: List[str] = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# Convenience groups usable in place of explicit camera names.
CAM_GROUPS: Dict[str, List[str]] = {
    'all': list(NUSCENES_CAMS),
    'front': ['CAM_FRONT'],
    'front_arc': ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
    'back_arc': ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
}


def _sample_rng(seed: int, condition: str, sample_id) -> np.random.Generator:
    """A generator that depends only on (seed, condition, sample) -- not model."""
    key = f'{seed}|{condition}|{sample_id}'.encode()
    digest = hashlib.sha256(key).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, 'little'))


def _resolve_views(views: Sequence[str], cam_names: List[str]) -> List[int]:
    """Map camera names / group names to indices into ``results['img']``."""
    wanted: List[str] = []
    for v in views:
        wanted.extend(CAM_GROUPS.get(v, [v]))
    idx = []
    for name in wanted:
        if name not in cam_names:
            raise KeyError(f'unknown camera {name!r}; have {cam_names}')
        idx.append(cam_names.index(name))
    return sorted(set(idx))


# --------------------------------------------------------------------------- #
# camera-side corruptions
# --------------------------------------------------------------------------- #

def cam_drop(results, rng, *, views=('all',), cam_names) -> None:
    """Total loss of one or more cameras: the view goes black."""
    for i in _resolve_views(views, cam_names):
        results['img'][i] = np.zeros_like(results['img'][i])


def cam_blur(results, rng, *, sigma: float, views=('all',), cam_names) -> None:
    """Defocus / motion blur, as an isotropic Gaussian of the given sigma (px)."""
    import cv2
    ksize = int(2 * round(3 * sigma) + 1)
    for i in _resolve_views(views, cam_names):
        img = results['img'][i]
        results['img'][i] = cv2.GaussianBlur(img, (ksize, ksize), sigma)


def cam_dark(results, rng, *, gain: float, gamma: float = 1.0,
             views=('all',), cam_names) -> None:
    """Low light: gamma then linear gain, both applied in [0, 1] space."""
    for i in _resolve_views(views, cam_names):
        img = np.clip(results['img'][i] / 255.0, 0.0, 1.0)
        img = np.power(img, gamma) * gain
        results['img'][i] = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.float32)


def cam_fog(results, rng, *, t: float, airlight: float = 220.0,
            views=('all',), cam_names) -> None:
    """Uniform airlight blend: ``img * (1 - t) + A * t``. Contrast loss, no depth."""
    for i in _resolve_views(views, cam_names):
        img = results['img'][i]
        results['img'][i] = np.clip(
            img * (1.0 - t) + airlight * t, 0.0, 255.0).astype(np.float32)


def cam_noise(results, rng, *, sigma: float, views=('all',), cam_names) -> None:
    """Additive Gaussian sensor noise, sigma in 0..255 units."""
    for i in _resolve_views(views, cam_names):
        img = results['img'][i]
        noise = rng.normal(0.0, sigma, size=img.shape)
        results['img'][i] = np.clip(img + noise, 0.0, 255.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# LiDAR-side corruptions
# --------------------------------------------------------------------------- #

def _keep_points(results, mask: torch.Tensor) -> None:
    """Replace the point cloud by the masked subset, keeping the container type."""
    points = results['points']
    if mask.sum() == 0:  # never hand an empty cloud to a sparse encoder
        mask = torch.zeros_like(mask)
        mask[:1] = True
    results['points'] = points[mask]


def lidar_beam_decimate(results, rng, *, keep_beams: int,
                        total_beams: int = 32) -> None:
    """Emulate a lower-beam sensor by keeping every ``total/keep``-th elevation bin."""
    if keep_beams >= total_beams:
        return
    xyz = results['points'].tensor[:, :3]
    r_xy = torch.linalg.norm(xyz[:, :2], dim=1).clamp(min=1e-6)
    elev = torch.atan2(xyz[:, 2], r_xy)
    lo, hi = torch.quantile(elev, torch.tensor([0.001, 0.999], dtype=elev.dtype))
    span = (hi - lo).clamp(min=1e-6)
    beam = ((elev - lo) / span * total_beams).floor().clamp(0, total_beams - 1)
    stride = max(1, total_beams // keep_beams)
    _keep_points(results, (beam.long() % stride) == 0)


def lidar_dropout(results, rng, *, p: float) -> None:
    """Drop a fraction ``p`` of returns uniformly at random."""
    n = results['points'].tensor.shape[0]
    keep = torch.from_numpy(rng.random(n) >= p)
    _keep_points(results, keep)


def lidar_fov(results, rng, *, fov_deg: float, center_deg: float = 0.0) -> None:
    """Restrict the cloud to an azimuth wedge -- a partially blinded scanner."""
    xyz = results['points'].tensor[:, :3]
    az = torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0]))
    delta = (az - center_deg + 180.0) % 360.0 - 180.0
    _keep_points(results, delta.abs() <= (fov_deg / 2.0))


def lidar_noise(results, rng, *, sigma: float) -> None:
    """Range jitter, applied as isotropic Gaussian noise on xyz (metres)."""
    pts = results['points']
    n = pts.tensor.shape[0]
    jitter = torch.from_numpy(
        rng.normal(0.0, sigma, size=(n, 3))).to(pts.tensor.dtype)
    pts.tensor[:, :3] += jitter
    results['points'] = pts


# --------------------------------------------------------------------------- #
# calibration corruption
# --------------------------------------------------------------------------- #

def _small_se3(rng, rot_deg: float, trans_m: float) -> np.ndarray:
    """A random SE(3) with rotation magnitude ``rot_deg`` and shift ``trans_m``."""
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-9)
    theta = np.deg2rad(rot_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    t = rng.normal(size=3)
    t = t / max(np.linalg.norm(t), 1e-9) * trans_m
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def calib_extrinsic(results, rng, *, rot_deg: float, trans_m: float,
                    views=('all',), cam_names) -> None:
    """Perturb camera extrinsics and propagate to every derived matrix.

    The error is applied on the *camera* side (``lidar2cam' = T_err @ lidar2cam``)
    and ``lidar2img`` / ``cam2lidar`` are recomputed from it, so the result stays
    a consistent calibration -- just a wrong one, as a miscalibrated rig would be.
    """
    lidar2cam = np.array(results['lidar2cam'], dtype=np.float64)
    cam2img = np.array(results['cam2img'], dtype=np.float64)
    for i in _resolve_views(views, cam_names):
        lidar2cam[i] = _small_se3(rng, rot_deg, trans_m) @ lidar2cam[i]
    results['lidar2cam'] = lidar2cam.astype(np.float32)
    results['lidar2img'] = np.stack(
        [cam2img[i] @ lidar2cam[i] for i in range(len(lidar2cam))]
    ).astype(np.float32)
    results['cam2lidar'] = np.stack(
        [np.linalg.inv(lidar2cam[i]) for i in range(len(lidar2cam))]
    ).astype(np.float32)


OPS = {
    'cam_drop': cam_drop,
    'cam_blur': cam_blur,
    'cam_dark': cam_dark,
    'cam_fog': cam_fog,
    'cam_noise': cam_noise,
    'lidar_beam_decimate': lidar_beam_decimate,
    'lidar_dropout': lidar_dropout,
    'lidar_fov': lidar_fov,
    'lidar_noise': lidar_noise,
    'calib_extrinsic': calib_extrinsic,
}

# ops that need the camera-name list injected
_NEEDS_CAMS = {'cam_drop', 'cam_blur', 'cam_dark', 'cam_fog', 'cam_noise',
               'calib_extrinsic'}


@TRANSFORMS.register_module()
class SensorCorruption(BaseTransform):
    """Apply a named list of sensor corruptions to one sample.

    Args:
        condition: name of the condition, used only to seed the per-sample RNG.
            Two runs with the same condition name and seed corrupt identically.
        ops: list of ``dict(type=..., **kwargs)``, applied in order.
        seed: global seed for the probe.
    """

    def __init__(self, condition: str, ops: Optional[Sequence[dict]] = None,
                 seed: int = 0) -> None:
        self.condition = condition
        self.ops = list(ops or [])
        self.seed = seed
        for op in self.ops:
            if op['type'] not in OPS:
                raise KeyError(f"unknown corruption {op['type']!r}; "
                               f'have {sorted(OPS)}')

    def transform(self, results: dict) -> dict:
        if not self.ops:
            return results
        sample_id = results.get('token', results.get('sample_idx', 0))
        rng = _sample_rng(self.seed, self.condition, sample_id)
        cam_names = (list(results['images'].keys())
                     if 'images' in results else list(NUSCENES_CAMS))
        for op in self.ops:
            kwargs = {k: v for k, v in op.items() if k != 'type'}
            if op['type'] in _NEEDS_CAMS:
                kwargs['cam_names'] = cam_names
            OPS[op['type']](results, rng, **kwargs)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(condition={self.condition!r}, '
                f'ops={self.ops!r}, seed={self.seed})')
