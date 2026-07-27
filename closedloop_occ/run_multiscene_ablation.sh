#!/usr/bin/env bash
# #1 Multi-scene closed-loop ablation: ego-only (no perception) vs +occ (DA3 depth forward-cap),
# across N held-out AV2 recon scenes. Turns the 1-scene proof into a statistic:
# collision_at_fault / collision_any / offroad / lane_centering per (scene, arm) -> CSV.
set -uo pipefail
OUR=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/closedloop_occ
ALP="$OUR/alpasim"
TH=/data/cmpe258-sp24/010892622/thesis/e2e/diffusion/thesis-nurec
RECON="$TH/data/argoverse2_reconstructions/val"
N="${1:-8}"                                    # number of scenes
OUT="$OUR/ablation_multiscene"; mkdir -p "$OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "scene,arm,collision_at_fault,collision_any,offroad,lane_centering,progress,exit" > "$CSV"

# --- env (no SLURM, local apptainer) ---
unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NODELIST 2>/dev/null || true
export ALPASIM_SINGULARITY_BIN=apptainer
: "${http_proxy:=http://172.16.1.2:3128}"; export http_proxy; export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"; export NO_PROXY="$no_proxy"; export no_grpc_proxy="$no_proxy"
export UV_NO_SYNC=1
export HF_TOKEN HUGGING_FACE_HUB_TOKEN NGC_API_KEY 2>/dev/null || true

metric () { grep -F "$2" "$1" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | head -1; }

cleanup () { for p in $(ps -eo pid,cmd | grep -E "apptainer exec|alpasim_wizard|pycena|physics_server|alpasim_runtime|alpasim_driver|alpasim_controller" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done; sleep 4; }

run_one () {  # scene  arm(ego|occ)
  local scene=$1 arm=$2
  grep -q "^${scene},${arm}," "$CSV" && { echo "[skip] $scene $arm"; return; }
  cleanup
  local LD="$OUT/${arm}_${scene}"; rm -rf "$LD"; mkdir -p "$LD"
  local CTMP="$LD/ctmp"; mkdir -p "$CTMP"; export APPTAINER_BIND="$CTMP:/tmp"
  local OCC=false DEV=cpu; [ "$arm" = occ ] && { OCC=true; DEV=cuda; }
  echo "[run] $scene $arm (use_occ=$OCC)"
  cd "$ALP"
  timeout 900 "$ALP/.venv/bin/alpasim_wizard" \
    deploy=local_singularity_hostvenv topology=1gpu \
    driver=occ driver.model.use_occ=$OCC driver.model.device=$DEV \
    +cameras=av2_1cam +exp/sim=av2_recon '~wizard.external_services.driver' \
    "scenes.local_usdz_dir=$RECON" "scenes.scene_ids=[$scene]" \
    wizard.log_dir="$LD" > "$LD/wizard.log" 2>&1
  local ec=$?
  local M="$LD/aggregate/metrics_results.txt"
  local caf=$(metric "$M" "collision_at_fault"); local can=$(metric "$M" "collision_any")
  local off=$(metric "$M" "offroad "); local lc=$(metric "$M" "lane_centering_abs_offset")
  local pr=$(metric "$M" "progress ")
  echo "${scene},${arm},${caf:-NA},${can:-NA},${off:-NA},${lc:-NA},${pr:-NA},${ec}" >> "$CSV"
  echo "[done] $scene $arm -> caf=${caf:-NA} can=${can:-NA} lc=${lc:-NA} (exit $ec)"
}

SCENES=$(ls "$RECON" | head -"$N")
echo "=== multi-scene ablation on $N scenes -> $CSV ==="
for scene in $SCENES; do
  [ -f "$RECON/$scene/$scene.usdz" ] || continue
  run_one "$scene" ego
  run_one "$scene" occ
done
cleanup
echo "=== DONE. summary ==="
python3 - "$CSV" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
for arm in ("ego","occ"):
    a=[r for r in rows if r["arm"]==arm]
    def col(k): return [float(r[k]) for r in a if r[k] not in ("NA","")]
    caf,can,lc=col("collision_at_fault"),col("collision_any"),col("lane_centering")
    if caf: print(f"{arm}: n={len(a)} collision_at_fault mean={st.mean(caf):.3f} | collision_any mean={st.mean(can):.3f} | lane_center mean={st.mean(lc):.3f}")
PY
