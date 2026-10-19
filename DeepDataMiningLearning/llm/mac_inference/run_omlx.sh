#!/bin/zsh
# oMLX runs: isolated base dir ~/omlx_bench (models/ + model_settings.json), see LAPTOP_QUICKSTART.md Step 6e
cd "$(dirname "$0")"
B=~/omlx_bench; export OMLX_BASE_PATH=$B
for id in "$@"; do
  OUT=results.jsonl bash serve_bench.sh omlx_$id ~/venvs/omlx $id -- omlx serve --model-dir $B/models --port 8000
  CLIENT=prefill_client.py OUT=prefill.jsonl bash serve_bench.sh pf_omlx_$id ~/venvs/omlx $id -- omlx serve --model-dir $B/models --port 8000
done
echo ALLDONE_OMLX
