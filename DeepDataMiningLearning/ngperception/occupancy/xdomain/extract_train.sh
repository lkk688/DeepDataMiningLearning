#!/usr/bin/env bash
# Phase 4: extract single-frame training data (t0 only, 40 keyframes/scene for diversity) from the
# rendered TRAIN scenes -> /tmp/shield/xdomain_train/<scene>/. LSS is single-frame so num_frame=1.
set -uo pipefail
OUR=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/closedloop_occ
XD=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/occupancy/xdomain
VENV="$OUR/alpasim/.venv/bin/python"
OUT=/tmp/shield/xdomain_train; mkdir -p "$OUT"
NF="${1:-40}"
n=0
for asl in $(find "$OUR/av2_surround_train" -name rollout.asl 2>/dev/null); do
  scene=$(echo "$asl" | grep -oE "av2_surround_train/[^/]+" | cut -d/ -f2 | cut -c1-8)
  [ -f "$OUT/$scene/f000.npz" ] && { echo "have $scene"; continue; }
  echo "== extract $scene =="
  "$VENV" "$XD/av2_extract_temporal.py" "$asl" "$OUT/$scene" "$NF" 1 2>&1 | tail -1
  n=$((n+1))
done
echo "extracted $n new scenes; total train frames: $(ls "$OUT"/*/f*.npz 2>/dev/null | wc -l)"
