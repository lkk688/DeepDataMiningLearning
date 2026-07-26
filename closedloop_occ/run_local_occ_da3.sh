#!/usr/bin/env bash
# M1a: CLOSED-LOOP drive with our occ_driver (stub CV = ego-only / no-occ ablation arm).
# Same local no-SLURM apptainer stack as run_local_gtreplay.sh, but driver=occ actually
# DRIVES (no force_gt): the OccModel returns a trajectory each step, the controller tracks
# it, the renderer re-renders the ego's new viewpoint -> genuine closed loop.
set -uo pipefail
OUR=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/closedloop_occ
ALP="$OUR/alpasim"
TH=/data/cmpe258-sp24/010892622/thesis/e2e/diffusion/thesis-nurec
RECON="$TH/data/argoverse2_reconstructions/val"
SCENE="${1:-02678d04-cc9f-3148-9f95-1ba66347dff9}"
LOG_DIR="$OUR/m1_occ_da3"

unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NODELIST 2>/dev/null || true
export ALPASIM_SINGULARITY_BIN=apptainer
CTMP="$LOG_DIR/ctmp"; rm -rf "$CTMP"; mkdir -p "$CTMP"
export APPTAINER_BIND="$CTMP:/tmp"
: "${http_proxy:=http://172.16.1.2:3128}"; export http_proxy; export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"; export NO_PROXY="$no_proxy"
export no_grpc_proxy="$no_proxy"
export UV_NO_SYNC=1
export HF_TOKEN HUGGING_FACE_HUB_TOKEN NGC_API_KEY 2>/dev/null || true

mkdir -p "$LOG_DIR"
cd "$ALP"
echo "== M1a closed-loop occ-drive (DA3-occ +occ arm): scene=$SCENE  log_dir=$LOG_DIR =="

"$ALP/.venv/bin/alpasim_wizard" \
  deploy=local_singularity_hostvenv topology=1gpu \
  driver=occ driver.model.use_occ=true driver.model.device=cuda +cameras=av2_1cam +exp/sim=av2_recon \
  '~wizard.external_services.driver' \
  "scenes.local_usdz_dir=$RECON" \
  "scenes.scene_ids=[$SCENE]" \
  wizard.log_dir="$LOG_DIR"
echo "== WIZARD_EXIT=$? =="
echo "-- results --"; find "$LOG_DIR/aggregate" -name 'results-summary.json' -o -name 'metrics_results.txt' 2>/dev/null | head
