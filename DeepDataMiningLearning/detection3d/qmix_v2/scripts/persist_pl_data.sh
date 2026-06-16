#!/bin/bash
# Persist the extracted Waymo per-frame npz (GT + re-extracted PL segments)
# to the canonical robust dataset store under /data/rnd-liu/Datasets, so it
# survives HPC session loss (the original PL extraction was symlinked into
# scratch /tmp and was wiped). Run after PL re-extraction completes.
#
# Source of truth going forward: /data/rnd-liu/Datasets/waymo_v1_extracted
set -u
REPO_ROOT=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_v1_extracted
CANON=/data/rnd-liu/Datasets/waymo_v1_extracted
LOG=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/detection3d/qmix_v2/scripts/persist_pl_data.log
echo "persist started $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"

# wait for PL re-extraction to complete
until [ -f /tmp/reextract2_done.flag ]; do sleep 60; done
echo "## extraction done; rsync -> $CANON" >> "$LOG"

mkdir -p "$CANON"
rsync -a --info=stats2 "$REPO_ROOT/" "$CANON/" >> "$LOG" 2>&1
echo "## rsync done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# verify all PL + GT frames resolve against the canonical store
conda run --no-capture-output -n py310 python - <<'PY' >> "$LOG" 2>&1
import pickle, os
root='/data/rnd-liu/Datasets/waymo_v1_extracted'
for nm,p in [('GT','/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune/waymo_v1_infos_train.pkl'),
             ('PL','/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune/waymo_v1_infos_train_pseudo_scaled.pkl')]:
    dl=pickle.load(open(p,'rb'))['data_list']; ok=miss=0
    for f in dl:
        seg,fidx=f['lidar_points']['lidar_path'].replace('waymo_v1://','').split('/')
        if os.path.isfile(os.path.join(root,seg,f'f_{int(fidx):04d}.npz')): ok+=1
        else: miss+=1
    print(f'[verify-canon] {nm}: resolvable={ok} missing={miss}')
PY
touch /tmp/persist_done.flag
echo "PERSIST DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
