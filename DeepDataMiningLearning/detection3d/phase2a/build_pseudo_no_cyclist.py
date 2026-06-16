"""S3: Build a pseudo info.pkl matching waymo_v1_infos_train_pseudo.pkl
but with all Cyclist pseudo-labels removed. Used to train Mixed+PL-NoCyc:
if Cyclist AP still rises vs Mixed, the gain is from general scene
exposure, not the 21 cyclist labels.
"""
from __future__ import annotations

import pickle
from pathlib import Path

NUS_BICYCLE_IDX = 7   # 'bicycle' = Cyclist in our 10-class mapping

SRC = Path('/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
            'data/waymo_finetune/waymo_v1_infos_train_pseudo.pkl')
DST = Path('/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/'
            'data/waymo_finetune/waymo_v1_infos_train_pseudo_nocyc.pkl')


def main():
    with SRC.open('rb') as f:
        d = pickle.load(f)
    data = d['data_list']
    n_inst_before = sum(len(fr['instances']) for fr in data)
    n_cyc_before = sum(1 for fr in data for inst in fr['instances']
                        if inst.get('bbox_label_3d') == NUS_BICYCLE_IDX)
    # Filter cyclist instances out of every frame
    n_dropped_frames = 0
    out = []
    for fr in data:
        kept = [inst for inst in fr['instances']
                if inst.get('bbox_label_3d') != NUS_BICYCLE_IDX]
        if not kept:
            n_dropped_frames += 1
            continue
        fr2 = dict(fr)
        fr2['instances'] = kept
        out.append(fr2)
    n_inst_after = sum(len(fr['instances']) for fr in out)
    print(f'source frames:        {len(data)}')
    print(f'kept frames:          {len(out)}  (dropped {n_dropped_frames} '
          f'that had only cyclist instances)')
    print(f'total instances:      {n_inst_before} -> {n_inst_after}  '
          f'(dropped {n_inst_before - n_inst_after}, '
          f'of which {n_cyc_before} were cyclists)')

    out_d = dict(metainfo=d['metainfo'], data_list=out)
    with DST.open('wb') as f:
        pickle.dump(out_d, f)
    print(f'wrote {DST}  ({DST.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
