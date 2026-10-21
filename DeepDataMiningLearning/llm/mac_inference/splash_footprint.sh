#!/bin/zsh
# Peak memory of a Splash server (the engine/splash child, not the launcher) while it prefills one long prompt.
# zsh splash_footprint.sh <model> <prompt_tokens> [splash serve options...]
S=~/opt/splash-1.4.0-arm64-macos26; M=$1; N=$2; shift 2; cd ${0:A:h}
lsof -ti :8000 | xargs kill 2>/dev/null; sleep 2
BASE=$(zsh sysmem.sh)
$S/python/bin/python3 $S/install/launcher.py serve --port 8000 --default-reasoning-effort none --idle-release off --model $M "$@" > logs/footprint.server.log 2>&1 &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 3; done
P=$(pgrep -f "engine/splash" | head -1); echo "base: $BASE"
( while kill -0 $P 2>/dev/null; do zsh sysmem.sh >> logs/footprint.samples; sleep 2; done ) &
W=$!
idle=$(footprint -p $P | awk '/phys_footprint:/ {print $2, $3; exit}')
~/venvs/mlxlm/bin/python prefill_client.py --model $M --label fp --out /dev/null --sizes $N --agent-size 1024 > /dev/null
loaded=$(footprint -p $P | awk '/phys_footprint:/ {print $2, $3; exit}')
peak=$(footprint -p $P | awk '/phys_footprint_peak:/ {print $2, $3; exit}')
echo "model=$M idle=$idle after_${N}=$loaded peak=$peak"; echo "loaded: $(head -1 logs/footprint.samples)"; echo "peak:   $(sort -n logs/footprint.samples | tail -1)"
kill $W; lsof -ti :8000 | xargs kill; wait 2>/dev/null
