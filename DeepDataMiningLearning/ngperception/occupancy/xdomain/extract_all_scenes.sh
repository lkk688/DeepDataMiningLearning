#!/usr/bin/env bash
# Extract temporal keyframes for every rendered AV2 surround scene -> /tmp/shield/xdomain_scenes/<scene>/
# Run after renders complete. Uses the alpasim venv (protobuf reader).
set -uo pipefail
OUR=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/closedloop_occ
XD=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/ngperception/occupancy/xdomain
VENV="$OUR/alpasim/.venv/bin/python"
OUT=/tmp/shield/xdomain_scenes; mkdir -p "$OUT"
NF="${1:-12}"
for asl in $(find "$OUR/av2_surround" -name rollout.asl 2>/dev/null); do
  scene=$(echo "$asl" | grep -oE "av2_surround/[^/]+" | cut -d/ -f2)
  echo "== extract $scene =="
  "$VENV" "$XD/av2_extract_temporal.py" "$asl" "$OUT/$scene" "$NF" 2>&1 | tail -2
done
echo "n scenes extracted:"; ls -d "$OUT"/*/ 2>/dev/null | wc -l
