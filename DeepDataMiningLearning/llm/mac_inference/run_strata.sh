#!/bin/zsh
# Strata (niko1221/strata) on this Mac with the same clients as the other engines; thinking off per request.
# Built and prepared with:  ./setup.sh --model Q2_0 --family qwen --gguf-dir <dir with the two Q2_0 shards> --vision no
#                           --context 65536 --yes --no-start   (in the Strata checkout; STRATA_DIR below)
cd "$(dirname "$0")"
D=${STRATA_DIR:-~/opt/strata}
STR=($D/.venv/bin/python $D/serve/server.py --engine strata --config $D/strata-q2_0.json --port 8000)
M=qwen3.8-flash-next-q2_0; L=${1:-strata_fn}
export EXTRA_JSON='{"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": false}}'
OUT=results.jsonl bash serve_bench.sh $L ~/venvs/mlxlm $M -- "${STR[@]}"
pkill -f build-metal/strata; sleep 10
CLIENT=prefill_client.py OUT=prefill.jsonl CLIENT_ARGS="--sizes 1024 4096 16384 32768" bash serve_bench.sh pf_$L ~/venvs/mlxlm $M -- "${STR[@]}"
pkill -f build-metal/strata; sleep 10
CLIENT=rollout_client.py OUT=results_m5pro_rollouts.jsonl bash serve_bench.sh ro_$L ~/venvs/mlxlm $M -- "${STR[@]}"
pkill -f build-metal/strata; sleep 10
echo ALLDONE_STRATA_$L
