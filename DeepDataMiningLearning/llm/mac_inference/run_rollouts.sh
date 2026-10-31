#!/bin/zsh
# The concurrency / pass@k rollout test (rollout_client.py) for Strata and Splash, one server at a time.
cd "$(dirname "$0")"
D=${STRATA_DIR:-~/opt/strata}; S=~/opt/splash-1.4.0-arm64-macos26
SPL=($S/python/bin/python3 $S/install/launcher.py serve --port 8000 --default-reasoning-effort none --idle-release off)
EXTRA_JSON='{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}' CLIENT=rollout_client.py OUT=results_m5pro_rollouts.jsonl \
  bash serve_bench.sh ro_strata_fn ~/venvs/mlxlm qwen3.8-flash-next-q2_0 -- $D/.venv/bin/python $D/serve/server.py --engine strata --config $D/strata-q2_0.json --port 8000
pkill -f build-metal/strata; sleep 10   # the engine is a child of the server; make sure its memory is free
for pair in "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 ro_splash_fn" "mlx-community/Qwen3.8-27B-4bit ro_splash_27b"; do
  set -- ${=pair}
  CLIENT=rollout_client.py OUT=results_m5pro_rollouts.jsonl bash serve_bench.sh $2 ~/venvs/mlxlm $1 -- "${SPL[@]}" --model $1
done
echo ALLDONE_ROLLOUTS
