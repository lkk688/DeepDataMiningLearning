"""Pack real nuScenes frames into Qwen-Drive's `PerceptionFrame` layout.

WHY THIS IS THE CRITICAL PATH
-----------------------------
Three separate blockers in `QWEN_DRIVE_LIDAR_TOKENS.md` are the *same* blocker -- the demo
set is 6 pre-packed frames with no nuScenes sample token, so nothing of ours can be aligned
to them:

* **the tokeniser prerequisite (§5.5).** The current LiDAR tokens are an 8x8 height
  histogram; no negative claim about LiDAR utility survives that ("your encoder was a
  histogram", and the referee is right). `extract.py` produces frozen TransFusion queries --
  but only for frames our detector can be run on, i.e. real nuScenes samples.
* **task (b), the headroom measurement.** Qwen-Drive vs BEVFusion-LC has to be on the *same
  frames*, which means frames both models can consume.
* **scale.** 6 frames is a plumbing gate and nothing more.

So: build the packer, and all three unlock.

WHAT IS FAITHFUL AND WHAT IS NOT
--------------------------------
Faithful:

* **the occupancy grid needs no resampling.** Occ3D-nuScenes and Qwen-Drive's nuScenes occ
  head are voxel-identical -- 200x200x16 over [-40,-40,-1, 40,40,5.4] at 0.4 m in the ego
  frame, (X,Y,Z). Asserted against the released config by
  `qwen_drive_labels.assert_grid_compatible`, not assumed.
* **the prompt layout** is copied from a released nuScenes demo frame verbatim, including
  the clockwise camera order (FRONT, FRONT_RIGHT, BACK_RIGHT, BACK, BACK_LEFT, FRONT_LEFT),
  which differs from the order used elsewhere in this repo.
* **calibration** is composed to the release's own convention: `sensor2lidar_*` such that
  `p_lidar = R p_cam + t` (what `geometry.build_lidar2img` inverts), from nuScenes'
  camera->ego and lidar->ego extrinsics.

Not faithful, and recorded rather than hidden:

* **`map` GT is written as ZEROS.** The released layout wants a 200x400 raster at 0.15 m
  over +-30 x +-15 m from the nuScenes map expansion; we do not build it. A `NO_MAP_GT`
  marker file is written into every frame directory and the manifest records it, so map
  segmentation must not be scored on packed frames. Silently writing zeros would read as "the
  model predicts nothing correctly".
* **occupancy labels are coarsened** 18 -> 10 (`qwen_drive_labels`), which is lossy in a
  direction that flatters Qwen-Drive. Score BOTH sides in the common taxonomy; the manifest
  carries the per-frame `merged_frac` so the size of the concession stays on the record.
* **boxes** are written in the LiDAR frame with z at the box bottom, matching what
  `head.get_bboxes` returns, but that convention is inferred from the decode path rather than
  from a spec. Detection numbers from packed frames should not be trusted until `--verify`
  is extended to check a box against an image; occupancy does not depend on it.

SAMPLING
--------
`--n` takes a **shuffled** subset and the manifest reports the number of distinct *scenes*,
not frames. Consecutive nuScenes frames are not independent samples (rule 7.10: `--limit
150` once gave 4 distinct scenes and a single-scene artefact looked like a dataset property).

VERIFY FIRST
------------
`--verify` projects the LiDAR cloud through the composed `lidar2img` of every camera and
reports the fraction of points landing in the image with positive depth. **Run it against a
released demo frame in the same call** (`--verify-demo`): the two statistics must be
comparable. A wrong extrinsic here does not crash -- it silently produces a model that looks
bad, which is the most expensive failure mode available (§7.8).
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_drive_labels import (assert_grid_compatible, collapse_report,  # noqa: E402
                               occ3d_to_qd)

#: Qwen-Drive's own nuScenes camera order (clockwise), from data/demo/perception/*/frame.json
QD_CAM_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
QD_VIEW_TAG = {'CAM_FRONT': '<FRONT VIEW>', 'CAM_FRONT_RIGHT': '<FRONT RIGHT VIEW>',
               'CAM_BACK_RIGHT': '<BACK RIGHT VIEW>', 'CAM_BACK': '<BACK VIEW>',
               'CAM_BACK_LEFT': '<BACK LEFT VIEW>', 'CAM_FRONT_LEFT': '<FRONT LEFT VIEW>'}
QD_INSTRUCTION = 'Analyze the scene.'
MAP_SHAPE = (200, 400)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _git(repo, *a):
    try:
        return subprocess.check_output(['git', '-C', str(repo), *a], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def _pose(nusc, sd_token):
    """calibrated_sensor of a sample_data -> (R sensor->ego, t sensor->ego)."""
    from pyquaternion import Quaternion
    sd = nusc.get('sample_data', sd_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    return (Quaternion(cs['rotation']).rotation_matrix,
            np.array(cs['translation'], np.float64), sd)


def frame_calib(nusc, sample) -> dict:
    """Compose the release's calib.npz fields for one sample.

    `sensor2lidar_*` is the transform taking a CAMERA-frame point to the LIDAR frame:
    ``p_lidar = R p_cam + t``, which is what `geometry.build_lidar2img` inverts. nuScenes
    gives camera->ego and lidar->ego, so
    ``R_c2l = R_l2e^T R_c2e``  and  ``t_c2l = R_l2e^T (t_c2e - t_l2e)``.
    """
    R_l2e, t_l2e, _ = _pose(nusc, sample['data']['LIDAR_TOP'])
    K, R_c2l, t_c2l, paths, sizes = [], [], [], [], []
    for cam in QD_CAM_ORDER:
        R_c2e, t_c2e, sd = _pose(nusc, sample['data'][cam])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        K.append(np.array(cs['camera_intrinsic'], np.float64))
        R_c2l.append(R_l2e.T @ R_c2e)
        t_c2l.append(R_l2e.T @ (t_c2e - t_l2e))
        paths.append(osp.join(nusc.dataroot, sd['filename']))
        sizes.append((sd['width'], sd['height']))
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = R_l2e
    lidar2ego[:3, 3] = t_l2e
    return dict(cam_intrinsic=np.stack(K).astype(np.float32),
                sensor2lidar_rotation=np.stack(R_c2l).astype(np.float32),
                sensor2lidar_translation=np.stack(t_c2l).astype(np.float32),
                lidar2ego=lidar2ego.astype(np.float32),
                _paths=paths, _sizes=sizes)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def project_stats(points_lidar: np.ndarray, calib: dict, sizes) -> dict:
    """Fraction of LiDAR points that land inside each image with positive depth.

    This is the positive control on the extrinsics. A transposed rotation or a swapped
    translation sign does not raise -- it moves the points off the sensor, and the only
    symptom downstream is "the model is bad".
    """
    from qwen_drive_perception import geometry
    fracs, depths = [], []
    p = np.concatenate([points_lidar[:, :3],
                        np.ones((points_lidar.shape[0], 1))], 1).astype(np.float32)
    for i, (w, h) in enumerate(sizes):
        l2i = geometry.build_lidar2img(calib['cam_intrinsic'][i],
                                       calib['sensor2lidar_rotation'][i],
                                       calib['sensor2lidar_translation'][i])
        q = p @ np.asarray(l2i, np.float32).T
        z = q[:, 2]
        front = z > 0.1
        u, v = q[front, 0] / z[front], q[front, 1] / z[front]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        fracs.append(float(inside.sum()) / max(points_lidar.shape[0], 1))
        depths.append(float(np.median(z[front][inside])) if inside.any() else float('nan'))
    return dict(per_cam_in_image=fracs, total_in_image=float(np.sum(fracs)),
                per_cam_median_depth=depths)


def verify_demo(demo_dir: Path) -> dict:
    """Same statistic on a released demo frame -- the known-good reference."""
    from qwen_drive_perception.dataset import PerceptionFrame
    f = PerceptionFrame(demo_dir)
    calib = dict(cam_intrinsic=f.cam_intrinsic,
                 sensor2lidar_rotation=f.sensor2lidar_rotation,
                 sensor2lidar_translation=f.sensor2lidar_translation)
    sizes = [f.image(c).size for c in f.cam_order]
    return project_stats(f.lidar, calib, sizes)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nusc', default='/data/rnd-liu/Datasets/nuScenes/v1.0-trainval')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--gts', default=None, help='Occ3D gts dir (default <nusc>/gts)')
    ap.add_argument('--split', choices=['val', 'train', 'all'], default='val')
    ap.add_argument('--tokens', default=None,
                    help='file of sample tokens (one per line): pack EXACTLY these, in this '
                         'order, ignoring --split/--n/--seed. Required when the frames must '
                         'line up with another pipeline that chose them -- e.g. the '
                         'occupancy trainer, whose subset is the deterministic first-N and '
                         'must be matched token-for-token, not re-sampled.')
    ap.add_argument('--n', type=int, default=200, help='0 = all')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--link-images', action='store_true', default=True,
                    help='symlink camera jpgs instead of copying (default)')
    ap.add_argument('--copy-images', dest='link_images', action='store_false')
    ap.add_argument('--verify', type=int, default=3,
                    help='run the projection control on the first N packed frames')
    ap.add_argument('--verify-demo',
                    default='/data/rnd-liu/Others/Qwen-Drive-1.0/data/demo/perception/'
                            '90162f90eceb4ada9e595bc1adb71b5f',
                    help='released nuScenes demo frame to use as the known-good reference')
    args = ap.parse_args()
    sys.path.insert(0, '/data/rnd-liu/Others/Qwen-Drive-1.0/src')

    set_global_seed(args.seed)
    gts = Path(args.gts or osp.join(args.nusc, 'gts'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open('/data/rnd-liu/Others/Qwen-Drive-1.0-4B/perception/config.json') as fh:
        assert_grid_compatible(json.load(fh))
    print('occ grid: voxel-identical to Occ3D-nuScenes (asserted against the released '
          'config)')

    from nuscenes import NuScenes
    from nuscenes.utils import splits
    nusc = NuScenes(version=args.version, dataroot=args.nusc, verbose=False)
    want = None if args.split == 'all' else set(getattr(splits, args.split))

    by_token = {}
    samples = []
    for s in nusc.sample:
        scene = nusc.get('scene', s['scene_token'])['name']
        if not (gts / scene / s['token'] / 'labels.npz').exists():
            continue
        by_token[s['token']] = (scene, s)
        if want is None or scene in want:
            samples.append((scene, s))

    if args.tokens:
        want_tok = [ln.strip() for ln in open(args.tokens) if ln.strip()]
        missing = [t for t in want_tok if t not in by_token]
        if missing:
            raise SystemExit(f'{len(missing)} tokens have no Occ3D GT, e.g. {missing[:3]}')
        samples = [by_token[t] for t in want_tok]
        print(f'packing EXACTLY the {len(samples)} tokens listed in {args.tokens}')
    else:
        print(f'{len(samples)} {args.split} samples with Occ3D GT '
              f'({len({s for s, _ in samples})} scenes)')
        # Shuffle BEFORE subsetting: the sample table is scene-ordered (rule 7.10).
        random.shuffle(samples)
        if args.n:
            samples = samples[:args.n]
    print(f'packing {len(samples)} frames from '
          f'{len({s for s, _ in samples})} distinct scenes')

    ref = None
    if args.verify and args.verify_demo:
        try:
            ref = verify_demo(Path(args.verify_demo))
            print(f'reference (released demo frame): '
                  f'{ref["total_in_image"] * 100:.1f}% of LiDAR points land in an image')
        except Exception as e:                       # noqa: BLE001
            print(f'reference demo check unavailable: {type(e).__name__}: {e}')

    manifest = []
    for i, (scene, s) in enumerate(samples):
        tok = s['token']
        d = out / tok
        (d / 'images').mkdir(parents=True, exist_ok=True)
        calib = frame_calib(nusc, s)

        for cam, src in zip(QD_CAM_ORDER, calib.pop('_paths')):
            dst = d / 'images' / f'{cam}.jpg'
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.link_images:
                dst.symlink_to(src)
            else:
                shutil.copyfile(src, dst)
        sizes = calib.pop('_sizes')

        sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        pts = np.fromfile(osp.join(nusc.dataroot, sd['filename']),
                          np.float32).reshape(-1, 5)[:, :3]
        np.save(d / 'lidar.npy', pts)

        lab = np.load(gts / scene / tok / 'labels.npz')
        occ18 = lab['semantics']
        occ10 = occ3d_to_qd(occ18).astype(np.int8)
        rep = collapse_report(occ18)

        # mask_camera / mask_lidar travel with the frame: the released GT and Occ3D
        # disagree overwhelmingly about UNOBSERVED space (measured on the two nuScenes demo
        # tokens: ~90 % of the voxels they call occupied and Occ3D calls free lie outside
        # mask_camera; restricting to it lifts occupied IoU 0.60 -> 0.84/0.87). Scoring
        # outside the mask charges Qwen-Drive for predicting where Occ3D declines to label.
        np.savez(d / 'gt.npz', occ=occ10,
                 mask_camera=lab['mask_camera'].astype(np.uint8),
                 mask_lidar=lab['mask_lidar'].astype(np.uint8),
                 map=np.zeros(MAP_SHAPE, np.int8),          # see NO_MAP_GT
                 boxes=np.zeros((0, 9), np.float32), labels=np.zeros((0,), np.int64))
        np.savez(d / 'calib.npz', **calib)
        (d / 'frame.json').write_text(json.dumps(dict(
            dataset_type='nuscenes', cam_order=QD_CAM_ORDER,
            content=[x for cam in QD_CAM_ORDER
                     for x in ({'text': QD_VIEW_TAG[cam]}, {'image': cam})]
            + [{'text': QD_INSTRUCTION}]), indent=2))
        (d / 'NO_MAP_GT').write_text(
            'gt.npz["map"] is all zeros: the 200x400 @0.15 m map raster is NOT built by\n'
            'pack_nuscenes.py. Do not score map segmentation on this frame.\n'
            'gt.npz["boxes"] is empty for the same reason -- occupancy only.\n')

        row = dict(token=tok, scene=scene, n_lidar=int(pts.shape[0]),
                   occ_merged_frac=rep['merged_frac'], n_occupied=rep['n_occupied'])
        if ref is not None and i < args.verify:
            st = project_stats(pts, calib, sizes)
            row['verify'] = st
            print(f'  [{tok[:8]}] {st["total_in_image"] * 100:.1f}% of points in an image '
                  f'(reference {ref["total_in_image"] * 100:.1f}%)')
        manifest.append(row)
        if (i + 1) % 25 == 0:
            print(f'  packed {i + 1}/{len(samples)}', flush=True)

    meta = dict(date=datetime.now().isoformat(timespec='seconds'), args=vars(args),
                n_frames=len(manifest),
                n_scenes=len({r['scene'] for r in manifest}),
                mean_occ_merged_frac=float(np.mean([r['occ_merged_frac']
                                                    for r in manifest])),
                map_gt='ABSENT (zeros) -- see NO_MAP_GT in each frame dir',
                boxes_gt='ABSENT (empty) -- occupancy only',
                reference_demo=ref,
                git=dict(commit=_git(Path(__file__).parent, 'rev-parse', 'HEAD'),
                         dirty=bool(_git(Path(__file__).parent, 'status', '--porcelain'))))
    (out / 'manifest.json').write_text(json.dumps(
        dict(meta=meta, frames=manifest), indent=2))
    print(f'\npacked {len(manifest)} frames from {meta["n_scenes"]} scenes -> {out}')
    print(f'mean fraction of occupied GT voxels in a MERGED class: '
          f'{meta["mean_occ_merged_frac"]:.3f}  '
          f'(that much of the taxonomy is coarsened away -- quote it)')
    if ref is not None:
        vs = [r['verify']['total_in_image'] for r in manifest if 'verify' in r]
        if vs:
            print(f'projection control: packed {np.mean(vs) * 100:.1f}% vs '
                  f'demo {ref["total_in_image"] * 100:.1f}% of LiDAR points in an image '
                  f'-- these must be comparable, or the extrinsics are wrong')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
