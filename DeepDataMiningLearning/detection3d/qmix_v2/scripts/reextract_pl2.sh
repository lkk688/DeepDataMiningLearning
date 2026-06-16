#!/bin/bash
# Clean re-extract: pull 37 PL tfrecords (per-segment, tolerant) -> npz into
# REAL dirs in the shared extracted root (broken symlinks already removed).
# No premature cleanup; keep tfrecords until npz verified.
set -u
REPO=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
EXTRACT=$REPO/DeepDataMiningLearning/detection3d/phase2a/extract_waymo_v1.py
OUT=/data/rnd-liu/Datasets/waymo_v1_extracted
TFTMP=/tmp/pl_tfrec
LOG=$(dirname "$0")/reextract_pl2.log
echo "PL re-extract v2 started $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"
mkdir -p "$TFTMP"

# 1) pull tfrecords (one segment per tar call; tolerate benign non-zero)
conda run --no-capture-output -n py310 python - <<'PY' >> "$LOG" 2>&1
import json, os, subprocess, glob
TFDIR='/fs/atipa/data/rnd-liu/Datasets/waymo143/training/training'
seg2tar=json.load(open('/tmp/seg2tar.json'))
done=0; failed=[]
for seg,(tar,member) in seg2tar.items():
    if glob.glob(f'/tmp/pl_tfrec/*{seg}*.tfrecord'):
        done+=1; continue
    subprocess.run(['tar','-xf',os.path.join(TFDIR,tar),'-C','/tmp/pl_tfrec',
                    '--wildcards',f'*{seg}*'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # flatten any nested path
    for h in glob.glob(f'/tmp/pl_tfrec/**/*{seg}*.tfrecord', recursive=True):
        if os.path.dirname(h)!='/tmp/pl_tfrec':
            os.replace(h, os.path.join('/tmp/pl_tfrec', os.path.basename(h)))
    if glob.glob(f'/tmp/pl_tfrec/*{seg}*.tfrecord'): done+=1
    else: failed.append(seg)
print(f'[pl-tfrec] ready {done}/{len(seg2tar)} failed={len(failed)}', flush=True)
PY
echo "## tfrecords ready $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 2) npz extraction
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  conda run --no-capture-output -n py310 python "$EXTRACT" \
  --tfrec-dir "$TFTMP" --out-dir "$OUT" >> "$LOG" 2>&1
echo "## npz extraction done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 3) verify resolvability
conda run --no-capture-output -n py310 python - <<'PY' >> "$LOG" 2>&1
import pickle, os
root='/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_v1_extracted'
dl=pickle.load(open('/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/data/waymo_finetune/waymo_v1_infos_train_pseudo_scaled.pkl','rb'))['data_list']
ok=miss=0
for f in dl:
    seg,fidx=f['lidar_points']['lidar_path'].replace('waymo_v1://','').split('/')
    if os.path.isfile(os.path.join(root,seg,f'f_{int(fidx):04d}.npz')): ok+=1
    else: miss+=1
print(f'[verify] PL resolvable={ok} missing={miss}', flush=True)
open('/tmp/pl_reextract_ok','w').write(f'{ok},{miss}')
PY
[ -f /tmp/pl_reextract_ok ] && rm -rf "$TFTMP"   # only clean after verify
touch /tmp/reextract2_done.flag
echo "PL RE-EXTRACT v2 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
