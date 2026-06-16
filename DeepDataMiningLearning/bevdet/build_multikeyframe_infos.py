"""
Offline builder for a multi-keyframe nuScenes info pkl.

Why this exists
---------------
The modern mmdet3d ``nuscenes_infos_*.pkl`` files (the ones the configs in
this project actually load) carry NO ``lidar_sweeps`` field. The
``LoadPointsFromMultiSweeps`` transform falls into a degenerate
``pad_empty_sweeps=True`` branch and just duplicates the keyframe cloud
N times with ``time=0`` — so what looks like "10-sweep" training has
actually been single-keyframe training. We spotted this while planning
Stage 2 of the temporal redesign.

What this script produces
-------------------------
A new pkl whose top-level dict matches the modern mmdet3d schema but
where every sample also has a ``lidar_sweeps`` list populated with
``sweeps_num`` real sample_data records walked back along the
LIDAR_TOP ``prev`` chain — across the keyframe boundary if needed.

For ``sweeps_num = 30`` and nuScenes' 20 Hz LiDAR, this covers
~1.5 s — about 3× the original "single-keyframe" multi-sweep span and
the temporal window we want for B9 (Stage 2 of the temporal work).

Each sweep entry contains the fields the existing
``LoadPointsFromMultiSweeps`` reads:

    lidar_points.lidar_path     str
    lidar_points.lidar2sensor   4x4 list  (modern mmdet3d convention,
                                           same as update_infos_to_v2.py)
    lidar_points.lidar2ego      4x4 list
    ego2global                  4x4 list
    timestamp                   float seconds
    sample_data_token           str

Sensor → keyframe-lidar math
----------------------------
Verbatim from ``nuscenes_converter.py::obtain_sensor2top``:

    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @
        (inv(e2g_r_mat).T @ inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @
        (inv(e2g_r_mat).T @ inv(l2e_r_mat).T)
    T -= e2g_t @ (inv(e2g_r_mat).T @ inv(l2e_r_mat).T) +
         l2e_t @ inv(l2e_r_mat).T
    sensor2lidar_rotation    = R.T   # so p_lidar = p_sensor @ R.T + T
    sensor2lidar_translation = T

Then converted to ``lidar2sensor`` (the inverse, per
``update_infos_to_v2.py``):

    lidar2sensor[:3, :3] = sensor2lidar_rotation.T            # = R
    lidar2sensor[:3, 3]  = -lidar2sensor[:3, :3] @ T          # = -R @ T

Usage
-----
    python projects/bevdet/build_multikeyframe_infos.py \\
        --in-pkl  data/nuscenes/nuscenes_infos_train_25pct.pkl \\
        --out-pkl data/nuscenes/nuscenes_infos_train_25pct_mkf30.pkl \\
        --sweeps-num 30

Run separately for train and val. ~minutes per file on the 25 % train
subset (5988 samples × 30 sweeps × a few dict reads).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
except ImportError as e:
    raise SystemExit(
        "nuscenes-devkit is required:\n  pip install nuscenes-devkit"
    ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_t_to_4x4(rot_quat, trans) -> np.ndarray:
    """Build a 4x4 transform from a quaternion + translation."""
    M = np.eye(4)
    M[:3, :3] = Quaternion(rot_quat).rotation_matrix
    M[:3, 3] = np.asarray(trans, dtype=np.float64)
    return M


def _obtain_sensor2lidar(
    nusc: NuScenes,
    sensor_token: str,
    l2e_t: np.ndarray, l2e_r_mat: np.ndarray,
    e2g_t: np.ndarray, e2g_r_mat: np.ndarray,
):
    """
    Port of ``nuscenes_converter.obtain_sensor2top`` plus the
    ``update_infos_to_v2`` lidar2sensor transformation. Returns a dict
    in the modern mmdet3d ``lidar_sweeps[i]`` schema.

    l2e_*, e2g_* are the CURRENT keyframe's lidar→ego and ego→global
    transforms.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    l2e_r_s = cs_record['rotation']
    l2e_t_s = np.asarray(cs_record['translation'], dtype=np.float64)
    e2g_r_s = pose_record['rotation']
    e2g_t_s = np.asarray(pose_record['translation'], dtype=np.float64)

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    # sensor→ego_sweep→global→ego_now→lidar_now (verbatim from converter)
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) \
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    # sensor2lidar_rotation = R.T  (per converter; matches loader convention)
    s2l_rot = R.T
    s2l_trans = T

    # Convert to modern lidar2sensor block (per update_infos_to_v2.py)
    lidar2sensor = np.eye(4)
    lidar2sensor[:3, :3] = s2l_rot.T                          # = R
    lidar2sensor[:3, 3] = -1.0 * (s2l_rot.T @ s2l_trans)      # = -R @ T

    # Build per-sweep ego2global and lidar2ego for completeness.
    sweep_ego2global = _quat_t_to_4x4(e2g_r_s, e2g_t_s)
    sweep_lidar2ego = _quat_t_to_4x4(l2e_r_s, l2e_t_s)

    # Preserve the dev-kit's full filename (e.g.
    # 'samples/LIDAR_TOP/...' for past keyframes,
    # 'sweeps/LIDAR_TOP/...' for non-keyframe sweeps). det3d_dataset's
    # parse_data_info uses 'samples' membership in the path to choose
    # data_prefix['pts'] vs data_prefix['sweeps'] — see
    # mmdet3d/datasets/det3d_dataset.py:286-295.
    return {
        'lidar_points': {
            'lidar_path': sd_rec['filename'],
            'lidar2sensor': lidar2sensor.astype(np.float32).tolist(),
            'lidar2ego': sweep_lidar2ego.astype(np.float32).tolist(),
            'num_pts_feats': 5,
        },
        'ego2global': sweep_ego2global.astype(np.float32).tolist(),
        'timestamp': sd_rec['timestamp'] / 1e6,   # μs → s, per modern schema
        'sample_data_token': sd_rec['token'],
    }


def _current_keyframe_l2e_e2g(nusc: NuScenes, sample_token: str):
    """Compute the current keyframe's (l2e_t, l2e_r, e2g_t, e2g_r) matrices."""
    sample = nusc.get('sample', sample_token)
    lidar_sd_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', lidar_sd_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    l2e_t = np.asarray(cs_rec['translation'], dtype=np.float64)
    l2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
    e2g_t = np.asarray(pose_rec['translation'], dtype=np.float64)
    e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix

    return lidar_sd_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(in_pkl: str, out_pkl: str, sweeps_num: int, data_root: str, version: str):
    print(f"[mkf] Loading nuScenes dev-kit ({version}) from {data_root} ...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    print(f"[mkf]   dev-kit samples: {len(nusc.sample)}")

    with open(in_pkl, 'rb') as f:
        in_data = pickle.load(f)
    assert isinstance(in_data, dict) and 'data_list' in in_data, \
        f"Expected modern mmdet3d pkl with 'data_list' key, got {list(in_data)[:5]}"
    data_list = in_data['data_list']
    print(f"[mkf] Input samples: {len(data_list)}")

    t0 = time.time()
    n_short = 0
    for i, info in enumerate(data_list):
        sample_token = info['token']
        cur_lidar_sd, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat = \
            _current_keyframe_l2e_e2g(nusc, sample_token)

        # Walk the sample_data.prev chain. Each prev-step is one 20 Hz
        # LiDAR sample_data record (sweeps + past keyframes, mixed).
        sweeps = []
        sd_rec = nusc.get('sample_data', cur_lidar_sd)
        while len(sweeps) < sweeps_num and sd_rec['prev'] != '':
            prev_token = sd_rec['prev']
            sweep_entry = _obtain_sensor2lidar(
                nusc, prev_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat,
            )
            sweeps.append(sweep_entry)
            sd_rec = nusc.get('sample_data', prev_token)
        if len(sweeps) < sweeps_num:
            n_short += 1
        info['lidar_sweeps'] = sweeps

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(data_list) - i - 1)
            print(f"[mkf]   {i+1}/{len(data_list)}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                  f"avg-sweeps={sum(len(d['lidar_sweeps']) for d in data_list[:i+1])/(i+1):.1f}")

    print(f"[mkf] Done in {time.time() - t0:.0f}s. "
          f"{n_short}/{len(data_list)} samples had fewer than {sweeps_num} sweeps "
          f"(start-of-scene boundary).")

    os.makedirs(os.path.dirname(os.path.abspath(out_pkl)), exist_ok=True)
    tmp = out_pkl + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(in_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_pkl)
    print(f"[mkf] Wrote {out_pkl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-pkl', required=True)
    ap.add_argument('--out-pkl', required=True)
    ap.add_argument('--sweeps-num', type=int, default=30)
    ap.add_argument('--data-root', default='data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    args = ap.parse_args()
    build(args.in_pkl, args.out_pkl, args.sweeps_num, args.data_root, args.version)


if __name__ == '__main__':
    main()
