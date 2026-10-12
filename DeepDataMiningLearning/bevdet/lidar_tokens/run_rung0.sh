#!/usr/bin/env bash
# Rung 0 of the LiDAR-token gate: how wide is the injection channel at all?
# Doc: ngperception/docs/QWEN_DRIVE_LIDAR_TOKENS.md §4
#
# Trains nothing. One frame, one set of images; only the injected tokens vary:
#   none / zero / self-cloud / other-cloud / random  (untrained projector)
#   trained / trained_other                          (if a projector.pt is given)
#
# THE READ is the pairwise table:
#   self vs other        -> BANDWIDTH. Only the point cloud differs. ~0 means the channel
#                           cannot carry LiDAR content and no optimiser will change that.
#   random vs zero       -> upper bound on what ANY token content can do.
#   none vs zero         -> the cost of the K extra sequence positions alone.
#   trained vs self      -> what training achieved *in predictions*. This is the number
#                           that separates information from calibration when the loss falls
#                           but mIoU does not (RESEARCH_DIRECTIONS.md §2.6 constant prior).
#
# Usage:  ./run_rung0.sh [WAIT_PID]
#   WAIT_PID: block until that PID exits first. Wait on a PID or on free memory, NEVER on a
#   process name pattern: `pgrep -f` also matches the launching shell, whose command line
#   contains this script's own text, and the wait never ends (it cost 6h45m once).
#   Also: `kill` on a launcher leaves the worker reparented to init and still holding
#   memory -- verify kills with `ps -eo pid,cmd | grep -E "<brack>eted"`.

set -u
cd "$(dirname "$0")"

PY=${PY:-/home/010796032/miniconda3/envs/py310/bin/python}
export CUDA_HOME=${CUDA_HOME:-/data/rnd-liu/cudatk}
export PATH=$(dirname "$PY"):$PATH
export CPATH=${CPATH:-$CUDA_HOME/targets/x86_64-linux/include}
export PYTHONUNBUFFERED=1
mkdir -p logs outputs

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "waiting for pid $WAIT_PID to exit ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  sleep 30                        # let the allocator actually release
fi

CKPT=$(ls -t outputs/*_capacity1f_lidar/projector.pt 2>/dev/null | head -1)
ARGS=(--mode sensitivity --inject lidar --max-frames 2 --exp-name sensitivity
      --out-root outputs)
if [ -n "${CKPT:-}" ]; then
  echo "probing trained projector: $CKPT"
  ARGS+=(--projector-ckpt "$CKPT")
else
  echo "no trained projector found; untrained probes only"
fi

$PY train_projector.py "${ARGS[@]}" > logs/sensitivity.log 2>&1
echo "exit=$? -> logs/sensitivity.log"
