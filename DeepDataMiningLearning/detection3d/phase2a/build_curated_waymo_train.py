"""Build a class-balanced Waymo v1.4.3 train info pkl.

Oversamples frames containing Pedestrian (cls 8) and Cyclist (cls 7),
which are severely under-represented in the natural Waymo distribution:

  Natural ratio (4,453 frames):
    cars        ubiquitous (100% of frames)
    pedestrians 84.8% frames, but 36% of instances are diluted
    cyclists    22.7% of frames, only 0.7% of instances

Curation strategy (no synthetic frames, only replication):
  - Frames containing >=1 cyclist:      replicate 5× total (4× extra)
  - Frames containing >=5 pedestrians:  replicate 2× total (1× extra)
  - Other frames:                       keep as-is (1× total)

A frame that satisfies both rules is counted in BOTH replication
groups, accumulating to up to 5+1 = 6× in the curated mix.

Writes ``waymo_v1_infos_train_curated.pkl`` alongside the original.
"""
from __future__ import annotations

import argparse
import collections
import copy
import pickle
import sys
from pathlib import Path


CYC_LABEL = 7    # bicycle
PED_LABEL = 8    # pedestrian


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/fs/atipa/data/rnd-liu/MyRepo/'
                    'DeepDataMiningLearning/data/waymo_finetune/'
                    'waymo_v1_infos_train.pkl')
    ap.add_argument('--dst', default='/fs/atipa/data/rnd-liu/MyRepo/'
                    'DeepDataMiningLearning/data/waymo_finetune/'
                    'waymo_v1_infos_train_curated.pkl')
    ap.add_argument('--cyc-replicate', type=int, default=5,
                    help='total copies (incl. original) of cyclist frames')
    ap.add_argument('--ped-min', type=int, default=5,
                    help='ped-rich threshold (min instances)')
    ap.add_argument('--ped-replicate', type=int, default=2,
                    help='total copies (incl. original) of ped-rich frames')
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    print(f'Reading {src} ...')
    with src.open('rb') as f:
        d = pickle.load(f)
    metainfo = d.get('metainfo', {})
    data = d['data_list']
    print(f'  source frames: {len(data)}')

    # Classify each frame
    cyc_frames, ped_frames, other_frames = [], [], []
    cat_counter = collections.Counter()
    for f in data:
        cnt = collections.Counter()
        for inst in f.get('instances', []):
            l = inst.get('bbox_label_3d')
            if l is not None:
                cnt[l] += 1
        n_cyc = cnt.get(CYC_LABEL, 0)
        n_ped = cnt.get(PED_LABEL, 0)
        cat_counter.update(cnt)
        if n_cyc >= 1:
            cyc_frames.append(f)
        if n_ped >= args.ped_min:
            ped_frames.append(f)
        if n_cyc < 1 and n_ped < args.ped_min:
            other_frames.append(f)

    print(f'  cyclist frames (>=1 cyc):   {len(cyc_frames)}  '
          f'-> {args.cyc_replicate}× = {len(cyc_frames)*args.cyc_replicate}')
    print(f'  ped-rich frames (>={args.ped_min}): {len(ped_frames)}  '
          f'-> {args.ped_replicate}× = {len(ped_frames)*args.ped_replicate}')
    print(f'  other frames:               {len(other_frames)}  -> 1× kept')

    # Build curated list. Note: cyc_frames and ped_frames may overlap.
    # CRITICAL: deep-copy every replica. Det3DDataset.parse_data_info
    # mutates ``info['lidar_points']['lidar_path']`` in-place during
    # load_data_list (prefix-joins ``data_prefix['pts']='velodyne'``),
    # so sharing dict references between replicas crashes the second
    # iteration with "not a waymo_v1:// URI: velodyne/waymo_v1://...".
    curated = []
    for f in cyc_frames:
        for _ in range(args.cyc_replicate):
            curated.append(copy.deepcopy(f))
    for f in ped_frames:
        for _ in range(args.ped_replicate):
            curated.append(copy.deepcopy(f))
    # Add others exactly once (already-replicated cyc/ped frames have
    # representation; don't double-count them as "other").
    for f in other_frames:
        curated.append(copy.deepcopy(f))

    # Compute final per-class instance counts in the curated mix
    new_cat = collections.Counter()
    for f in curated:
        for inst in f.get('instances', []):
            l = inst.get('bbox_label_3d')
            if l is not None:
                new_cat[l] += 1
    print()
    print(f'  curated frames: {len(curated)}  '
          f'({len(curated)/len(data):.2f}× original)')
    print(f'  natural instance counts:  car={cat_counter[0]:>6}  '
          f'ped={cat_counter[8]:>6}  cyc={cat_counter[7]:>6}')
    print(f'  curated instance counts:  car={new_cat[0]:>6}  '
          f'ped={new_cat[8]:>6}  cyc={new_cat[7]:>6}')
    nat_total = sum(cat_counter.values())
    cur_total = sum(new_cat.values())
    print(f'  natural class share:      car={100*cat_counter[0]/nat_total:.1f}%  '
          f'ped={100*cat_counter[8]/nat_total:.1f}%  cyc={100*cat_counter[7]/nat_total:.1f}%')
    print(f'  curated class share:      car={100*new_cat[0]/cur_total:.1f}%  '
          f'ped={100*new_cat[8]/cur_total:.1f}%  cyc={100*new_cat[7]/cur_total:.1f}%')

    out_dict = {'metainfo': metainfo, 'data_list': curated}
    print(f'Writing {dst} ...')
    with dst.open('wb') as f:
        pickle.dump(out_dict, f)
    print(f'  {dst.stat().st_size/1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
