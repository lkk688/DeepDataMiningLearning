#!/usr/bin/env bash
# Integration test: does the RELEASED model produce sensible occupancy on frames we packed
# ourselves from real nuScenes? (pack_nuscenes.py -> QWEN_DRIVE_LIDAR_TOKENS.md task (b))
#
# `--inject none` is the released path with no extra tokens, eval only, so the number is
# Qwen-Drive's own occupancy accuracy on real nuScenes val frames -- the first half of the
# headroom measurement. The second half is BEVFusion-LC on the SAME tokens.
#
# Waits for run_next.sh to release the card first: one arm holds ~41 GiB and two unrelated
# alpasim jobs hold ~34 GiB of the 93 GiB.
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1

# See the note in run_step_tokeniser.sh: never wait on a process PATTERN here.
need_mib=${NEED_MIB:-45000}
while :; do
  free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) \
         - $(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits | head -1) ))
  [ "$free" -ge "$need_mib" ] && break
  sleep 30
done
echo "card has ${free} MiB free at $(date '+%F %T')"

PACK=${PACK:-/tmp/claude-1368/-fs-atipa-data-rnd-liu-MyRepo-DeepDataMiningLearning/9aa9d800-f198-4f09-bd4a-437fa1cfeb75/scratchpad/packtest}
$PY train_projector.py --inject none --frames "$PACK" \
    --exp-name packed_nuscenes --out-root outputs > logs/packed_eval.log 2>&1
echo "exit=$? -> logs/packed_eval.log"
grep -E "frames:|\[eval@|patch embed" logs/packed_eval.log
