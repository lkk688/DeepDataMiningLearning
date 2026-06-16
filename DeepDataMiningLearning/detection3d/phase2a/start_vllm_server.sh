#!/bin/bash
# Start a local OpenAI-compatible vLLM server for the P1a VLM ambiguity
# ablation. Uses apptainer + the team's pre-built vllm SIF images.
#
# Default model: Qwen/Qwen3.6-27B-FP8 (served as "qwen-27b") — the
# team's preferred VLM. Switch to Gemma-4-31B via `MODEL=gemma`.
#
# Usage:
#   bash start_vllm_server.sh                    # default Qwen3.6-27B-FP8
#   MODEL=gemma   bash start_vllm_server.sh      # google/gemma-4-31b-it
#   PORT=8001  bash start_vllm_server.sh
#
# Important: on this HPC, clients that hit localhost must bypass the
# proxy (otherwise the request is routed to the corporate proxy and
# times out). Either set `NO_PROXY=*` in client env, OR call with
# `curl --noproxy "*" -X POST http://0.0.0.0:8000/v1/chat/completions`.
set -u
SIF=${SIF:-"/data/rnd-liu/images/vllm.sif"}
MODEL_KEY=${MODEL:-qwen}
PORT=${PORT:-8000}

if [ "$MODEL_KEY" = "qwen" ]; then
  MODEL_ID="Qwen/Qwen3.6-27B-FP8"
  SERVED_NAME="qwen-27b"
  EXTRA_FLAGS=(
    --dtype auto
    --gpu-memory-utilization 0.6
    --max-model-len 262144
    --kv-cache-dtype fp8
    --enable-chunked-prefill
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --max-num-seqs 16
  )
elif [ "$MODEL_KEY" = "gemma" ]; then
  SIF=${SIF:-"/data/rnd-liu/images/vllm-latest.sif"}
  MODEL_ID="google/gemma-4-31b-it"
  SERVED_NAME="gemma-4-31b"
  EXTRA_FLAGS=(
    --dtype bfloat16
    --kv-cache-dtype fp8
    --gpu-memory-utilization 0.9
    --max-model-len 32768
    --enable-chunked-prefill
    --enable-prefix-caching
    --trust-remote-code
    --enable-auto-tool-choice
    --reasoning-parser gemma4
    --tool-call-parser gemma4
  )
else
  echo "MODEL must be 'qwen' or 'gemma' (got: $MODEL_KEY)" >&2
  exit 2
fi

echo "Launching vLLM via apptainer:"
echo "  SIF:    $SIF"
echo "  model:  $MODEL_ID  (served as $SERVED_NAME)"
echo "  port:   $PORT"
echo

apptainer run --nv --cleanenv \
  --bind /data/rnd-liu:/data/rnd-liu \
  "$SIF" \
  "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  "${EXTRA_FLAGS[@]}" \
  --host 0.0.0.0 --port "$PORT"
