#!/usr/bin/env bash
# Start one OpenAI-compatible server, wait until it answers, run bench_client.py against it, stop it.
#
#   bash mac_inference/serve_bench.sh <label> <venv> <model> -- <server command ...>
#
# Examples (each engine in its own venv; see LAPTOP_QUICKSTART.md Step 6e):
#   bash mac_inference/serve_bench.sh mlx_4bit ~/venvs/mlxlm mlx-community/Qwen3.8-27B-4bit -- \
#       mlx_lm.server --model mlx-community/Qwen3.8-27B-4bit --port 8000 --chat-template-args '{"enable_thinking": false}'
#   bash mac_inference/serve_bench.sh lithos_4bit_dspark ~/venvs/lithos mlx-community/Qwen3.8-27B-4bit -- \
#       lithos-metal serve --model mlx-community/Qwen3.8-27B-4bit --max-context 8192
set -uo pipefail
cd "$(dirname "$0")"
label=$1; venv=$2; model=$3; shift 4
OUT=${OUT:-results.jsonl}
mkdir -p logs
source "$venv/bin/activate"
t0=$(date +%s)
/usr/bin/time -l "$@" > "logs/$label.server.log" 2>&1 &         # -l: peak memory footprint at exit
TP=$!
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do
  sleep 2
  kill -0 $TP 2>/dev/null || { echo "$label: server exited"; tail -20 "logs/$label.server.log"; exit 1; }
done
echo "$label: ready after $(( $(date +%s) - t0 ))s"
python "${CLIENT:-bench_client.py}" --model "$model" --label "$label" --out "$OUT" ${CLIENT_ARGS:-}
# The process that actually serves port 8000 (oMLX's launcher forks a detached `omlx-server`).
LP=$(lsof -t -nP -iTCP:8000 -sTCP:LISTEN | head -1)
peak=$(footprint -p "$LP" 2>/dev/null | awk '/phys_footprint_peak/ {print $2, $3}')
SP=$(pgrep -P $TP)
kill -INT $SP 2>/dev/null; sleep 5; kill -TERM $SP 2>/dev/null   # mlx_lm.server ignores SIGINT
kill -TERM "$LP" 2>/dev/null; sleep 3; kill -KILL "$LP" 2>/dev/null
wait $TP
echo "$label: done; peak memory footprint of the serving process: ${peak:-n/a}"
