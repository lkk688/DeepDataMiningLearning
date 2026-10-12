#!/usr/bin/env bash
# Next step: swap the 8x8 height-histogram tokeniser for a FROZEN LiDAR-ONLY detector's
# TransFusion object queries, and measure how much the tokeniser actually matters.
# Doc: QWEN_DRIVE_LIDAR_TOKENS.md §5.5 (the prerequisite) and §4 (rung 0).
#
# 1. extract queries for exactly the packed tokens (LiDAR-only BEVFusion, NDS 0.6922)
# 2. rung 0 with the histogram      -- the cheap lower bound, on real nuScenes frames
# 3. rung 0 with the queries        -- same frames, same probe, better tokeniser
#
# THE READ: `self vs other` in step 3 vs step 2. Same VLM, same images, same untrained
# projector architecture; only the tokeniser differs. If the detector's queries do not carry
# more into the BEV head than a height histogram, the channel is the limit, not the encoder
# -- and that is a much stronger statement than either number alone.
set -u
cd "$(dirname "$0")"
PY=/home/010796032/miniconda3/envs/py310/bin/python
export CUDA_HOME=/data/rnd-liu/cudatk
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export PATH=$(dirname $PY):$PATH
export PYTHONUNBUFFERED=1
MM=/data/rnd-liu/MyRepo/mmdetection3d
PACK=${PACK:-/data/rnd-liu/Datasets/nuScenes/packed_qd_val120}
Q=$PACK/lidar_queries.npz

# Wait for free GPU memory, NOT for a process pattern.
# `pgrep -f <pattern>` deadlocked here for 6h45m: the launching shell's command line
# contains this script's own heredoc, pattern included, so the script matched its own
# parent forever. Bracketing the pattern stops a grep matching ITSELF; it does nothing
# about a parent whose cmdline embeds the script text. Poll the resource instead.
need_mib=${NEED_MIB:-45000}
while :; do
  free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) \
         - $(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits | head -1) ))
  [ "$free" -ge "$need_mib" ] && break
  sleep 30
done
echo "card has ${free} MiB free at $(date '+%F %T')"

echo "===== 1. extract frozen LiDAR-only detector queries for the packed tokens"
if [ ! -f "$Q" ]; then
  ( cd $MM && $PY /data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/bevdet/lidar_tokens/extract.py \
      --tokens $PACK/manifest.json --out $Q --max-frames 100000 ) \
      > logs/extract_queries.log 2>&1
  echo "  exit=$? -> logs/extract_queries.log"
else
  echo "  reusing $Q"
fi
tail -3 logs/extract_queries.log 2>/dev/null

[ -f "$Q" ] || { echo "extraction produced no npz; stopping"; exit 1; }

echo
echo "===== 2. rung 0, histogram tokeniser, real nuScenes frames"
$PY train_projector.py --mode sensitivity --frames $PACK --max-frames 2 \
    --tokenizer pillar --exp-name sens_pillar_nusc --out-root outputs \
    > logs/sens_pillar_nusc.log 2>&1
echo "  exit=$?"
grep -E "^  \[|^pair|vs |^n = |queries:" logs/sens_pillar_nusc.log

echo
echo "===== 3. rung 0, frozen-detector queries, SAME frames"
$PY train_projector.py --mode sensitivity --frames $PACK --max-frames 2 \
    --tokenizer queries --queries $Q --exp-name sens_queries_nusc --out-root outputs \
    > logs/sens_queries_nusc.log 2>&1
echo "  exit=$?"
grep -E "^  \[|^pair|vs |^n = |queries:" logs/sens_queries_nusc.log
