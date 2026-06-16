#!/bin/bash
# Re-extract the 37 pseudo-label Waymo segments lost when scratch /tmp was
# wiped on HPC session loss. Pull only the needed tfrecords from 3 training
# tars, run extract_waymo_v1.py -> per-frame npz, then re-run B32 (Q-Mix v2).
set -u
REPO=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
TFDIR=/fs/atipa/data/rnd-liu/Datasets/waymo143/training/training
EXTRACT=$REPO/DeepDataMiningLearning/detection3d/phase2a/extract_waymo_v1.py
OUT=$REPO/data/waymo_v1_extracted
TFTMP=/tmp/pl_tfrec
LOG=/tmp/reextract_pl.log
echo "PL re-extract started $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"

mkdir -p "$TFTMP"
# 1) Pull the needed tfrecords from their tars (selective extraction)
cd "$REPO"
conda run --no-capture-output -n py310 python - <<'PY' >> "$LOG" 2>&1
import json, os, subprocess, glob
TFDIR='/fs/atipa/data/rnd-liu/Datasets/waymo143/training/training'
seg2tar=json.load(open('/tmp/seg2tar.json'))
# These tars are non-standard (multi-stream): a single tar call with several
# --wildcards patterns exits non-zero. Extract ONE segment per call and
# verify by file existence, tolerating the benign non-zero exit.
done=0; failed=[]
for seg,(tar,member) in seg2tar.items():
    subprocess.run(['tar','-xf',os.path.join(TFDIR,tar),'-C','/tmp/pl_tfrec',
                    '--wildcards',f'*{seg}*'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hit=glob.glob(f'/tmp/pl_tfrec/**/*{seg}*.tfrecord', recursive=True)
    if hit: done+=1
    else: failed.append(seg)
    if done % 10 == 0: print(f'[pl-tfrec] {done} extracted...', flush=True)
print(f'[pl-tfrec] extracted {done}/{len(seg2tar)}; failed={len(failed)}', flush=True)
if failed: print('FAILED:', failed[:5], flush=True)
# flatten: extractor reads *.tfrecord directly from --tfrec-dir
for root,_,files in os.walk('/tmp/pl_tfrec'):
    for f in files:
        if f.endswith('.tfrecord') and root!='/tmp/pl_tfrec':
            os.replace(os.path.join(root,f), os.path.join('/tmp/pl_tfrec',f))
n=len([f for f in os.listdir('/tmp/pl_tfrec') if f.endswith('.tfrecord')])
print(f'[pl-tfrec] {n} tfrecords ready in /tmp/pl_tfrec', flush=True)
PY
echo "## tfrecords pulled $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 2) Extract per-frame npz into the shared extracted root (appends 37 seg dirs)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  conda run --no-capture-output -n py310 python "$EXTRACT" \
  --tfrec-dir "$TFTMP" --out-dir "$OUT" >> "$LOG" 2>&1
echo "## npz extraction done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 3) Verify PL frames now resolve
conda run --no-capture-output -n py310 python - <<'PY' >> "$LOG" 2>&1
import pickle, os
root='/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_v1_extracted'
dl=pickle.load(open('/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune/waymo_v1_infos_train_pseudo_scaled.pkl','rb'))['data_list']
ok=miss=0
for f in dl:
    seg,fidx=f['lidar_points']['lidar_path'].replace('waymo_v1://','').split('/')
    if os.path.isfile(os.path.join(root,seg,f'f_{int(fidx):04d}.npz')): ok+=1
    else: miss+=1
print(f'[verify] PL resolvable={ok} missing={miss}')
open('/tmp/pl_reextract_ok','w').write(f'{ok},{miss}')
PY
rm -rf "$TFTMP"   # free the ~tens of GB of tfrecords
touch /tmp/reextract_done.flag
echo "PL RE-EXTRACT DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
