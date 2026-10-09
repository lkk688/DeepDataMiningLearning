#!/usr/bin/env bash
# The 4-arm capacity gate for LiDAR tokens at a frozen VLM's input.
# Doc: ngperception/docs/QWEN_DRIVE_LIDAR_TOKENS.md §4   Index: RESEARCH_DIRECTIONS.md §4.7
#
# PRE-REGISTERED READING (write it down before running -- §7.8):
#
#   none    the released path. The number to beat.
#   zero    K zero-embedding tokens, no training. Separates "the prompt got K tokens
#           longer" from "the tokens carried LiDAR". If `zero` already differs from `none`
#           by as much as the trained arms do, nothing has been shown.
#   lidar   projector trained on this frame's points.
#   shuffle projector trained on ANOTHER frame's points. The control.
#
#   PASS  : lidar beats none clearly AND beats shuffle clearly.
#   FAIL-1: lidar cannot beat none  -> the injection channel does not reach the BEV head
#           usefully. The route stops. (A zero gradient is a bug, not this result; the
#           script aborts on one.)
#   FAIL-2: lidar beats none but shuffle matches it -> the projector is being used as free
#           per-frame parameters, not as a LiDAR reading. The route stops.
#
#   This is an OVERFIT test on 6 frames. Even a PASS says only that the channel is open.
#   It does not say LiDAR helps, and it cannot: 6 frames, no held-out set.
#
# RUN THE CHEAP PROBE FIRST. At ~100 s/step this script is ~5 h, and a FAIL-1 from it would
# be ambiguous between "the channel is too narrow" and "60 steps was not enough". So ask the
# capacity question in its cheapest form first (~85 min):
#
#   python train_projector.py --inject lidar --max-frames 1 --epochs 50 --lr 3e-3 #       --exp-name capacity1f --out-root outputs
#
# If the loss on that single frame does not move, stop -- nothing here will help.
#
# Resumable: an arm whose output dir already contains a log.json with the final epoch is
# skipped. Runs strictly sequentially -- one arm peaks ~33.5 GiB (fp32 attention math) or
# ~27.2 GiB with --attn-math bf16, which costs no extra time.
#
# COST: with --patch-embed matmul (the default, see QWEN_DRIVE_LIDAR_TOKENS.md §4) the
# forward is ~1 s instead of ~93 s, so this script is minutes, not hours. Every timing in
# git history before 2026-09-09 was ~99 % one pathological Conv3d layer.

set -u
cd "$(dirname "$0")"

PY=${PY:-/home/010796032/miniconda3/envs/py310/bin/python}
export CUDA_HOME=${CUDA_HOME:-/data/rnd-liu/cudatk}
export PATH=$(dirname "$PY"):$PATH
export CPATH=${CPATH:-$CUDA_HOME/targets/x86_64-linux/include}
export PYTHONUNBUFFERED=1

OUT=${OUT:-outputs/gate}
EPOCHS=${EPOCHS:-40}
LR=${LR:-1e-3}
SEED=${SEED:-0}
mkdir -p "$OUT" logs

for ARM in none zero lidar shuffle; do
  DONE=$(ls -d "$OUT"/*_"$ARM" 2>/dev/null | while read -r d; do
           [ -f "$d/log.json" ] && grep -q '"epoch"' "$d/log.json" && echo "$d"; done | tail -1)
  if [ -n "${DONE:-}" ]; then
    echo "== skip $ARM (already in $DONE)"
    continue
  fi
  echo "== $ARM  $(date '+%F %T')"
  $PY train_projector.py --inject "$ARM" --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" \
      --eval-every 5 --out-root "$OUT" --exp-name gate \
      > "logs/gate_${ARM}.log" 2>&1
  echo "   exit=$? -> logs/gate_${ARM}.log"
done

echo
echo "== summary"
for d in "$OUT"/*/; do
  [ -f "$d/log.json" ] || continue
  $PY - "$d" <<'PYEOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
log = json.loads((d / 'log.json').read_text())
ev = [r for r in log if r['split'] == 'eval']
arm = d.name.split('_')[-1]
if ev:
    print(f'{arm:8s} eval@{ev[0]["epoch"]:<3d} loss {ev[0]["loss"]:.4f} mIoU {ev[0]["miou"]:.4f}'
          f'   ->  eval@{ev[-1]["epoch"]:<3d} loss {ev[-1]["loss"]:.4f} mIoU {ev[-1]["miou"]:.4f}')
PYEOF
done
