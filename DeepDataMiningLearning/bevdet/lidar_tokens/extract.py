"""Extract a frozen LiDAR detector's object queries, for use as LLM tokens.

The plan this belongs to: **frozen VLM + frozen LiDAR detector + train only a projector**
that maps detector queries into the language model's token space. The idea is to avoid the
one thing that makes LiDAR-in-a-VLA hard — there is no web-scale corpus to pretrain a point
cloud encoder on, so a from-scratch LiDAR adapter is data-starved next to a vision tower
that has seen billions of images. Feeding the LLM a *trained detector's* output turns the
problem from "learn to read point clouds" into "describe what the detector found".

**This script is step 0, and step 0 is not the projector.** Before building an apparatus to
read these features, establish that the features contain what we intend to read (§7.9 of
`ngperception/docs/RESEARCH_DIRECTIONS.md` — the FusionOcc FOV probe was carefully designed
and returned nothing because the effect it perturbed was ~0 in the region it measured). So
this dumps the queries, and `probe_m0.py` asks a linear probe whether simple 3D facts are
recoverable from them at all. If a linear probe cannot read them, no projector will.

WHY THE **LiDAR-ONLY** DETECTOR, NOT BEVFusion-LC
-------------------------------------------------
This defaults to `bevfusion_lidar_voxel0075_...` (the NDS 0.6922 anchor), **not** the
lidar-camera checkpoint, and that is not a convenience choice. The VLM already sees all six
cameras. Queries from a *fusion* detector carry camera-derived features, so tokens built
from them would partly duplicate what the language model can already see, and the two
questions this whole line exists to answer would both become unanswerable:

* "does the LLM read the LiDAR?" -- a gain could come from the detector's camera reading.
* "what does LiDAR add on top of the cameras?" -- the baseline is no longer camera-only.

With the LiDAR-only detector, everything in the token stream is information the VLM does
not otherwise have. `--config`/`--checkpoint` can still point at the LC model, deliberately,
as a separate arm.

TOKEN ALIGNMENT
---------------
Saved per frame is the nuScenes **sample token**, not `sample_idx`. In these infos
`sample_idx` is a plain integer row index (0, 1, 2, ...) while `token` is the 32-hex sample
token -- and the token is the only thing that aligns with `pack_nuscenes.py` output, whose
frame directories are named by it. `--tokens` accepts either a file of tokens or a packer
`manifest.json`, subsets the dataset to exactly those samples, and **fails if any requested
token is missing** rather than silently extracting a different set.

What is saved per frame:
  query_feat  (P, C)  final decoder-layer query features -- the projector's input
  centers     (P, 3)  decoded box centres, ego frame
  scores      (P,)    per-query max class score
  labels      (P,)    per-query argmax class
  targets     dict    3D facts derived from GT boxes (see TARGETS below)

TARGETS are chosen to be things a LiDAR detector should know and a single image answers
badly -- metric counts and metric distances, not semantics:
  n_veh_20m   vehicles whose centre lies within 20 m
  d_front     distance to the nearest GT object in the front 60 deg wedge, capped at 50 m
  n_rear      objects behind the ego (x < 0)
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

VEHICLE = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle'}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _git(repo, *a):
    try:
        return subprocess.check_output(['git', '-C', repo, *a], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gt_targets(gt_bboxes_3d, gt_labels_3d, class_names) -> dict:
    """3D facts derived from ground-truth boxes, in the ego frame."""
    if gt_bboxes_3d is None or len(gt_bboxes_3d) == 0:
        return dict(n_veh_20m=0.0, d_front=50.0, n_rear=0.0, n_total=0.0)
    c = np.asarray(gt_bboxes_3d.gravity_center.cpu())
    lab = np.asarray(gt_labels_3d.cpu())
    names = np.array([class_names[i] if 0 <= i < len(class_names) else '?' for i in lab])

    rng = np.linalg.norm(c[:, :2], axis=1)
    is_veh = np.isin(names, list(VEHICLE))
    # front 60 deg wedge: |atan2(y, x)| < 30 deg
    az = np.degrees(np.arctan2(c[:, 1], c[:, 0]))
    front = np.abs(az) < 30.0
    d_front = float(rng[front].min()) if front.any() else 50.0
    return dict(n_veh_20m=float(((rng < 20.0) & is_veh).sum()),
                d_front=min(d_front, 50.0),
                n_rear=float((c[:, 0] < 0).sum()),
                n_total=float(len(c)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='projects/BEVFusion/configs/'
                    'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
                    help='LiDAR-ONLY by default -- see the module docstring')
    ap.add_argument('--checkpoint', default='modelzoo_mmdetection3d/bevfusion_lidar_'
                    'voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth')
    ap.add_argument('--tokens', default=None,
                    help='file of nuScenes sample tokens (one per line) or a '
                         'pack_nuscenes.py manifest.json; the dataset is subset to exactly '
                         'these samples and a missing token is an error')
    ap.add_argument('--out', required=True, help='output .npz')
    ap.add_argument('--max-frames', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    want = None
    if args.tokens:
        txt = open(args.tokens).read()
        if args.tokens.endswith('.json'):
            want = [r['token'] for r in json.loads(txt)['frames']]
        else:
            want = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        print(f'{len(want)} tokens requested from {args.tokens}')

    set_global_seed(args.seed)

    import torch as _t
    if not getattr(_t.load, '_patched', False):          # trusted local checkpoints
        _o = _t.load

        def _l(*a, **k):
            k.setdefault('weights_only', False)
            return _o(*a, **k)
        _l._patched = True
        _t.load = _l

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    init_default_scope('mmdet3d')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join(osp.dirname(osp.abspath(args.out)), '_runner')
    cfg.load_from = args.checkpoint
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 4
    cfg.val_dataloader = cfg.test_dataloader

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model.eval().cuda()
    class_names = list(cfg.get('class_names', []))

    if want is None:
        loader = runner.test_dataloader
        ds = loader.dataset
    else:
        # Subset by token BEFORE the dataloader exists: mutating `data_list` afterwards
        # leaves the sampler with the old length. Build the dataset, subset it in place,
        # then hand the instance to build_dataloader (which accepts a built Dataset).
        from mmengine.registry import DATASETS
        ds = DATASETS.build(cfg.test_dataloader.dataset)
        pos = {}
        for i in range(len(ds)):
            pos[ds.get_data_info(i)['token']] = i
        missing = [t for t in want if t not in pos]
        if missing:
            raise SystemExit(f'{len(missing)} requested tokens are not in this split, '
                             f'e.g. {missing[:3]} -- wrong --config split?')
        ds.get_subset_([pos[t] for t in want])
        dl_cfg = dict(cfg.test_dataloader)
        dl_cfg['dataset'] = ds
        dl_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)
        loader = Runner.build_dataloader(dl_cfg)
        print(f'dataset subset to {len(ds)} samples')

    # The final decoder layer's output IS the query feature the head decodes from.
    cap = {}

    def hook(_m, _a, out):
        cap['q'] = out.detach()                     # (B, C, P)
    model.bbox_head.decoder[-1].register_forward_hook(hook)

    # `Det3DDataSample.metainfo` does NOT carry `token`; asking for it returns None and any
    # fallback to `sample_idx` yields a row index ('0', '1', ...), which silently produces an
    # npz that aligns with nothing. The dataset's own data_list is the authority, and the
    # sampler is `shuffle=False`, so row i of the loader is `order_tokens[i]`.
    order_tokens = [ds.get_data_info(i)['token'] for i in range(len(ds))]
    if want is not None and order_tokens != list(want):
        raise SystemExit('dataset order does not match the requested token order')
    bad = [t for t in order_tokens[:5] if not (isinstance(t, str) and len(t) == 32)]
    if bad:
        raise SystemExit(f'these do not look like nuScenes sample tokens: {bad}')

    Q, CEN, SC, LB, TG, TOK = [], [], [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= args.max_frames:
                break
            out = model.test_step(data)[0]
            pred = out.pred_instances_3d
            # Keep ALL P queries' features: that is the projector's input and it is a
            # fixed-size set. The decoded predictions vary per frame (score threshold /
            # NMS), so pad them to P instead of truncating the features to match --
            # truncating would silently change the token count from frame to frame.
            q = cap['q'][0].permute(1, 0).float().cpu().numpy()          # (P, C)
            P = q.shape[0]
            n = int(len(pred.scores_3d))
            order = torch.argsort(pred.scores_3d, descending=True).cpu().numpy()[:P]
            cen = np.zeros((P, 3), np.float32)
            sc = np.zeros((P,), np.float32)
            lb = np.full((P,), -1, np.int64)
            m = min(n, P)
            if m:
                cen[:m] = np.asarray(pred.bboxes_3d.gravity_center.cpu())[order[:m]]
                sc[:m] = np.asarray(pred.scores_3d.cpu())[order[:m]]
                lb[:m] = np.asarray(pred.labels_3d.cpu())[order[:m]]
            Q.append(q)
            CEN.append(cen)
            SC.append(sc)
            LB.append(lb)
            gt = out.eval_ann_info if hasattr(out, 'eval_ann_info') else {}
            TG.append(gt_targets(getattr(out, 'gt_instances_3d', None) and
                                 out.gt_instances_3d.bboxes_3d,
                                 getattr(out, 'gt_instances_3d', None) and
                                 out.gt_instances_3d.labels_3d, class_names))
            TOK.append(order_tokens[i])          # see the note above metainfo
            if (i + 1) % 100 == 0:
                print(f'[extract] {i + 1}/{args.max_frames}', flush=True)

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        query_feat=np.stack(Q), centers=np.stack(CEN), scores=np.stack(SC),
        labels=np.stack(LB), token=np.asarray(TOK, dtype='<U32'),
        **{f'tgt_{k}': np.asarray([t[k] for t in TG]) for k in TG[0]})
    meta = dict(date=datetime.now().isoformat(timespec='seconds'), args=vars(args),
                n_frames=len(Q), query_shape=list(np.stack(Q).shape),
                query_dim=int(np.stack(Q).shape[-1]),
                n_queries=int(np.stack(Q).shape[-2]),
                modality='lidar-only' if 'lidar-cam' not in args.config else 'lidar+camera',
                class_names=class_names,
                git=dict(commit=_git(osp.dirname(osp.abspath(__file__)), 'rev-parse', 'HEAD')))
    with open(osp.splitext(args.out)[0] + '.meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'\nsaved {len(Q)} frames -> {args.out}\nquery tensor {np.stack(Q).shape}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
