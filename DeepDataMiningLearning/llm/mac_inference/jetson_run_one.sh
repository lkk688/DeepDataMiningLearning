#!/bin/bash
# Copy this directory to ~/llmbench/llm/mac_inference on the Jetson, with a venv at ~/llmbench (tokenizers) for the clients.
# One server per configuration: decode (bench_client) + prefill/agent (prefill_client) against the same container.
#   run_one.sh <label> <port> <model> -- docker run ...
cd ~/llmbench/llm/mac_inference
export OUT=~/llmbench/results_thor.jsonl EXTRA_JSON="{\"chat_template_kwargs\": {\"enable_thinking\": false}}"
PY=~/llmbench/bin/python
label=$1 port=$2 model=$3; shift 4
python3 memory_balloon.py 4   # return pooled GPU memory + page cache (no sudo); vLLM sees only MemFree
name=bench_$label; docker rm -f $name > /dev/null 2>&1
echo "$label: start, used $(free -g | awk "/^Mem:/ {print \$3}") GB free $(free -g | awk "/^Mem:/ {print \$4}") GB"
t0=$(date +%s)
"$1" "$2" -d --name $name "${@:3}" > /dev/null || { echo "$label: docker run failed"; exit 1; }
until curl -sf http://127.0.0.1:$port/v1/models > /dev/null 2>&1; do
  sleep 5
  docker ps -q -f name="^${name}$" | grep -q . || { echo "$label: container exited"; docker logs $name > ~/llmbench/logs/$label.failed.log 2>&1; tail -8 ~/llmbench/logs/$label.failed.log; docker rm $name > /dev/null 2>&1; exit 1; }
done
echo "$label: ready after $(( $(date +%s) - t0 ))s"
( while sleep 1; do free -b | awk "/^Mem:/ {print \$3}"; done ) > logs/$label.mem & MP=$!
$PY bench_client.py --url http://127.0.0.1:$port --model "$model" --label $label --out $OUT
$PY prefill_client.py --url http://127.0.0.1:$port --model "$model" --label pf_$label --out $OUT --sizes 1024 4096 16384 32768
kill $MP; docker logs $name > logs/$label.server.log 2>&1
used_run=$(free -g | awk "/^Mem:/ {print \$3}")
ts=$(date +%s); docker stop -t 120 $name > /dev/null; echo "$label: graceful stop (SIGTERM) took $(( $(date +%s) - ts ))s"
docker rm $name > /dev/null; sleep 30
echo "$label: used while serving ${used_run} GB, 30 s after graceful stop $(free -g | awk "/^Mem:/ {print \$3}") GB"
echo "$label: done; peak used $(sort -n logs/$label.mem | tail -1 | awk "{printf \"%.1f GB\", \$1/1e9}"); after stop used $(free -g | awk "/^Mem:/ {print \$3}") GB"
