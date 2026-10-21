#!/bin/zsh
# Splash (incoai/splash) on this Mac with the same clients as the other engines. Thinking off server-side.
cd "$(dirname "$0")"
S=~/opt/splash-1.4.0-arm64-macos26
SPL=($S/python/bin/python3 $S/install/launcher.py serve --port 8000 --default-reasoning-effort none --idle-release off)
M=${1:-mlx-community/Qwen3.8-27B-4bit}; L=${2:-splash_27b}; shift 2 2>/dev/null
OUT=results.jsonl bash serve_bench.sh $L ~/venvs/mlxlm $M -- "${SPL[@]}" --model $M "$@"
CLIENT=prefill_client.py OUT=prefill.jsonl CLIENT_ARGS="--sizes 1024 4096 16384 32768" bash serve_bench.sh pf_$L ~/venvs/mlxlm $M -- "${SPL[@]}" --model $M "$@"
echo ALLDONE_SPLASH_$L
