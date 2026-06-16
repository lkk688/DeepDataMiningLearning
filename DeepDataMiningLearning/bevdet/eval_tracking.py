"""
End-to-end driver for nuScenes 3D MOT evaluation on top of B10c's
per-sample 3D detections.

What it does
------------
1. Loads the val info pkl to learn ``sample_idx -> sample_token`` mapping.
2. Loads the nuScenes dev-kit to learn ``sample_token -> {ego_pose,
   calibrated_sensor, scene, prev, next, timestamp}``.
3. Reads our existing detection JSONs from ``<runs_dir>/<sample_idx>/
   <sample_idx>_3d_dets.json`` (produced by ``unified_inference_loop``).
4. For each scene, in chronological order:
   * transforms each LiDAR-frame box → global frame using the per-sample
     ``lidar2ego @ ego2global`` extrinsics from the dev-kit;
   * filters to the 7 tracking classes (drops construction_vehicle,
     traffic_cone, barrier);
   * feeds the frame to a fresh ``HungarianTracker3D`` instance.
5. Collects all tracklet histories → nuScenes tracking-format JSON.
6. Runs the official ``TrackingEval`` and prints AMOTA / AMOTP.

Side benefit: by setting ``--no-velocity``, we get an IoU-only baseline
that ablates the value of our FG-TCA flow head's per-cell velocity
prediction. ``B10c-with-velocity vs B10c-no-velocity`` is the
direct paper measurement for the tracker chapter.

Usage
-----
::

    python eval_tracking.py \\
        --runs-dir   /tmp/unified_b10c \\
        --val-pkl    /path/to/nuscenes_infos_val_mkf30.pkl \\
        --dataroot   /path/to/data/nuscenes \\
        --out-dir    /tmp/track_b10c_velocity \\
        [--no-velocity]   # for the IoU-only ablation
"""
from __future__ import annotations
import argparse
import json
import os
import os.path as osp
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

# Make ``DeepDataMiningLearning.bevdet.tracker3d`` importable.
_HERE = osp.dirname(osp.abspath(__file__))
_DDML_ROOT = osp.dirname(osp.dirname(_HERE))   # ..../MyRepo/DeepDataMiningLearning
if _DDML_ROOT not in sys.path:
    sys.path.insert(0, _DDML_ROOT)

from DeepDataMiningLearning.bevdet.tracker3d import (  # noqa: E402
    HungarianTracker3D,
    boxes_lidar_to_global,
    DET_CLASSES_10,
    DET_TO_TRACK,
    TRACKING_CLASSES_7,
)


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_sample_idx_to_token(val_pkl: str) -> Dict[int, str]:
    """Build a mapping ``sample_idx -> sample_token`` from the val info pkl."""
    with open(val_pkl, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'data_list' in data:
        infos = data['data_list']
    else:
        infos = data
    out: Dict[int, str] = {}
    for i, info in enumerate(infos):
        idx = int(info.get('sample_idx', i))
        out[idx] = info['token']
    print(f'[eval_tracking] sample_idx → token map built: {len(out)} entries')
    return out


def build_dets_iter(runs_dir: str,
                     idx_to_token: Dict[int, str]) -> Dict[int, str]:
    """
    Return only the sample_idx values that have a 3d_dets.json on disk.
    Drops missing samples (they won't contribute to tracking).
    """
    have: Dict[int, str] = {}
    for idx, tok in idx_to_token.items():
        det_path = osp.join(runs_dir, str(idx), f'{idx}_3d_dets.json')
        if osp.isfile(det_path):
            have[idx] = tok
    print(f'[eval_tracking] {len(have)} / {len(idx_to_token)} samples have 3D det JSONs')
    return have


def load_dets_json(runs_dir: str, sample_idx: int) -> List[Dict]:
    """
    Read a single sample's per-frame det JSON. Returns list of detection
    dicts each with keys: ``box_lidar`` (np.ndarray [9]), ``score``,
    ``label_det`` (10-class name).
    """
    p = osp.join(runs_dir, str(sample_idx), f'{sample_idx}_3d_dets.json')
    with open(p) as f:
        d = json.load(f)
    boxes = np.asarray(d['boxes_lidar'], dtype=np.float64) if d['boxes_lidar'] \
            else np.zeros((0, 9), dtype=np.float64)
    scores = np.asarray(d['scores'], dtype=np.float64) if d['scores'] \
             else np.zeros((0,), dtype=np.float64)
    labels = np.asarray(d['labels'], dtype=np.int64) if d['labels'] \
             else np.zeros((0,), dtype=np.int64)
    out = []
    for i in range(boxes.shape[0]):
        lab = int(labels[i])
        if lab < 0 or lab >= len(DET_CLASSES_10):
            continue
        det_name = DET_CLASSES_10[lab]
        track_name = DET_TO_TRACK.get(det_name)
        if track_name is None:
            continue       # not a tracked class
        out.append({
            'box_lidar':   boxes[i],
            'score':       float(scores[i]),
            'class_det':   det_name,
            'class_track': track_name,
        })
    return out


# -----------------------------------------------------------------------------
# nuScenes dev-kit wrappers
# -----------------------------------------------------------------------------

def get_extrinsics(nusc, sample_token: str
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(lidar2ego_R, lidar2ego_t, ego2global_R, ego2global_t)`` for
    the LiDAR sensor at ``sample_token``.

    These are the same matrices our offline mkf30 builder used and match
    the convention in mmdet3d's lidar boxes.
    """
    sample = nusc.get('sample', sample_token)
    lidar_sd_token = sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', lidar_sd_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])

    lidar2ego_R = Quaternion(cs['rotation']).rotation_matrix.astype(np.float64)
    lidar2ego_t = np.asarray(cs['translation'], dtype=np.float64)
    ego2global_R = Quaternion(pose['rotation']).rotation_matrix.astype(np.float64)
    ego2global_t = np.asarray(pose['translation'], dtype=np.float64)
    return lidar2ego_R, lidar2ego_t, ego2global_R, ego2global_t


def group_samples_by_scene(nusc, sample_tokens: List[str]
                            ) -> Dict[str, List[Tuple[str, int]]]:
    """
    Group sample_tokens by scene, sorted by timestamp (chronological).
    Returns ``{scene_token: [(sample_token, timestamp_us), ...]}``.
    """
    scenes: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for tok in sample_tokens:
        sample = nusc.get('sample', tok)
        scenes[sample['scene_token']].append((tok, int(sample['timestamp'])))
    for k in scenes:
        scenes[k].sort(key=lambda x: x[1])
    return scenes


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def run_tracking(args) -> Dict[str, List[Dict]]:
    print(f'[eval_tracking] runs_dir={args.runs_dir}')
    print(f'[eval_tracking] val_pkl={args.val_pkl}')
    print(f'[eval_tracking] dataroot={args.dataroot}')
    print(f'[eval_tracking] use_velocity={not args.no_velocity}')

    # 1) sample_idx -> sample_token, filtered to samples we have JSONs for
    idx_to_token = load_sample_idx_to_token(args.val_pkl)
    idx_to_token = build_dets_iter(args.runs_dir, idx_to_token)
    if not idx_to_token:
        raise RuntimeError(f'No 3D-det JSONs found under {args.runs_dir}/')

    # 2) Load dev-kit
    from nuscenes.nuscenes import NuScenes
    print(f'[eval_tracking] Loading NuScenes dev-kit ({args.version}) ...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # 3) Group by scene
    token_to_idx = {t: i for i, t in idx_to_token.items()}
    scenes = group_samples_by_scene(nusc, list(idx_to_token.values()))
    print(f'[eval_tracking] {len(scenes)} scenes covering {len(idx_to_token)} samples')

    # 4) Track each scene independently. Accumulate per-token output records.
    all_records: Dict[str, List[Dict]] = defaultdict(list)
    t_start = time.time()
    for scene_idx, (scene_tok, ordered) in enumerate(scenes.items()):
        tracker = HungarianTracker3D(
            use_velocity=(not args.no_velocity),
            max_age=args.max_age,
            birth_score=args.birth_score,
            retention_score=args.retention_score,
        )
        prev_ts_us: Optional[int] = None
        for (sample_tok, ts_us) in ordered:
            idx = token_to_idx[sample_tok]
            dets_lidar = load_dets_json(args.runs_dir, idx)

            # LiDAR-frame → global-frame box conversion.
            l2e_R, l2e_t, e2g_R, e2g_t = get_extrinsics(nusc, sample_tok)
            if dets_lidar:
                boxes_lidar = np.stack([d['box_lidar'] for d in dets_lidar])
                boxes_global = boxes_lidar_to_global(
                    boxes_lidar, l2e_R, l2e_t, e2g_R, e2g_t,
                )
                for i, d in enumerate(dets_lidar):
                    d['box_global'] = boxes_global[i]

            dt = ((ts_us - prev_ts_us) * 1e-6) if prev_ts_us is not None else 0.5
            prev_ts_us = ts_us
            tracker.update(dets_lidar, dt_seconds=dt, sample_token=sample_tok)

        # Collect this scene's tracklet history into the global output dict.
        scene_recs = tracker.finalize_history()
        for tok, recs in scene_recs.items():
            all_records[tok].extend(recs)

        if (scene_idx + 1) % 20 == 0 or scene_idx + 1 == len(scenes):
            elapsed = time.time() - t_start
            eta = elapsed / (scene_idx + 1) * (len(scenes) - scene_idx - 1)
            print(f'[eval_tracking]   scene {scene_idx+1}/{len(scenes)}  '
                  f'elapsed={elapsed:.0f}s  eta={eta:.0f}s')

    # Make sure every val sample has an entry (empty if no detections survived).
    for tok in idx_to_token.values():
        all_records.setdefault(tok, [])

    return dict(all_records)


def save_tracking_json(records: Dict[str, List[Dict]], out_path: str) -> None:
    os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
    payload = {
        'meta': {
            'use_camera':  False,
            'use_lidar':   True,
            'use_radar':   False,
            'use_map':     False,
            'use_external': False,
        },
        'results': records,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    n_tracklets = sum(len(v) for v in records.values())
    print(f'[eval_tracking] tracking JSON → {out_path}'
          f'   ({len(records)} samples, {n_tracklets} tracklet-points)')


def _patch_legacy_deps():
    """
    nuScenes-devkit's tracking eval was written against very old
    motmetrics (<=1.1.3) + numpy (<1.20) + Python (<=3.9). Modern
    environments break it at three layers; we patch each in place
    BEFORE importing anything from ``nuscenes.eval.tracking``.

    1. ``collections.Iterable`` was moved to ``collections.abc`` in
       Python 3.10. motmetrics 1.1.3 still does ``from collections
       import Iterable``.
    2. ``np.bool`` / ``np.float`` / ``np.int`` were removed in numpy
       >=1.24. motmetrics + nuscenes-devkit both reference them.
    3. nuscenes-devkit's ``MOTAccumulatorCustom`` overrides ``events``
       and ``new_event_dataframe_with_data`` with a custom-but-buggy
       rebuild path that doesn't survive newer pandas. We delete the
       override so the parent class's working ``events`` is used.
    """
    import collections, collections.abc
    for _name in ('Iterable', 'Mapping', 'MutableMapping', 'Sequence', 'Callable'):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(collections.abc, _name))

    if not hasattr(np, 'bool'):
        np.bool = bool          # type: ignore[attr-defined]
    if not hasattr(np, 'float'):
        np.float = float        # type: ignore[attr-defined]
    if not hasattr(np, 'int'):
        np.int = int            # type: ignore[attr-defined]


def run_official_eval(tracking_json: str, dataroot: str, version: str,
                       eval_set: str, out_dir: str) -> Dict:
    """Run the official nuScenes TrackingEval (AMOTA / AMOTP)."""
    _patch_legacy_deps()

    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.mot import MOTAccumulatorCustom

    # Drop the buggy override of ``events`` so motmetrics' built-in
    # implementation is used instead.
    if hasattr(MOTAccumulatorCustom, 'events') and isinstance(
            MOTAccumulatorCustom.__dict__.get('events'), property):
        del MOTAccumulatorCustom.events

    # motmetrics 1.1.3's MOTAccumulator.new_event_dataframe_with_data
    # assumes the events list has at least one row (tevents[0]) and
    # raises IndexError when an accumulator recorded zero events for
    # a class. Patch ``MOTAccumulator.events`` to return an empty
    # MOTAccumulator-style DataFrame in that case.
    import motmetrics as _mm
    _orig_acc_events = _mm.MOTAccumulator.events.fget

    def _safe_acc_events(self):
        if not self._indices:
            return MOTAccumulatorCustom.new_event_dataframe()
        return _orig_acc_events(self)
    _mm.MOTAccumulator.events = property(_safe_acc_events)

    cfg = config_factory('tracking_nips_2019')
    os.makedirs(out_dir, exist_ok=True)
    print('\n' + '=' * 60 + '\n  Running official TrackingEval\n' + '=' * 60)
    evaluator = TrackingEval(
        config=cfg,
        result_path=tracking_json,
        eval_set=eval_set,
        output_dir=out_dir,
        nusc_version=version,
        nusc_dataroot=dataroot,
        verbose=True,
    )
    metrics = evaluator.main()
    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True,
                    help='Directory of per-sample 3D detection JSONs '
                         '(unified_inference_loop output).')
    ap.add_argument('--val-pkl', required=True,
                    help='nuScenes val info pkl (mkf30) for sample_idx→token.')
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--eval-set', default='val')
    ap.add_argument('--out-dir', default='/tmp/track_out')
    ap.add_argument('--no-velocity', action='store_true',
                    help='Ablation: turn off velocity-based propagation.')
    ap.add_argument('--max-age', type=int, default=4)
    ap.add_argument('--birth-score', type=float, default=0.20)
    ap.add_argument('--retention-score', type=float, default=0.10)
    ap.add_argument('--skip-eval', action='store_true',
                    help='Just produce tracking JSON, skip official eval.')
    args = ap.parse_args()

    records = run_tracking(args)
    tracking_json = osp.join(args.out_dir, 'tracking_results.json')
    save_tracking_json(records, tracking_json)

    if not args.skip_eval:
        run_official_eval(
            tracking_json=tracking_json,
            dataroot=args.dataroot,
            version=args.version,
            eval_set=args.eval_set,
            out_dir=args.out_dir,
        )


if __name__ == '__main__':
    main()
