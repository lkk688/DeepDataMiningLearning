#!/usr/bin/env bash
# M0 smoke test: run AlpaSim closed-loop GT-replay DIRECTLY on this node (no SLURM,
# no sbatch/srun). Uses our copied alpasim + host venv, the thesis's read-only
# containers (via data/sif symlinks) and one already-reconstructed scene. The
# no-SLURM SingularityDeployment override runs each microservice as a plain
# `apptainer exec --nv` background process; services co-locate on localhost.
set -uo pipefail
OUR=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/closedloop_occ
ALP="$OUR/alpasim"
TH=/data/cmpe258-sp24/010892622/thesis/e2e/diffusion/thesis-nurec
RECON="$TH/data/argoverse2_reconstructions/val"
SCENE="${1:-02678d04-cc9f-3148-9f95-1ba66347dff9}"
LOG_DIR="$OUR/m0_gtreplay"

# We are inside a SLURM interactive allocation, but the user wants NO SLURM: unset
# SLURM_JOB_ID so wizard.create sets slurm_job_id=0 -> our SingularityDeployment
# no-SLURM branch runs each service as a plain `apptainer exec` (no srun).
unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NODELIST 2>/dev/null || true
# Apptainer is our container runtime (backend reads ALPASIM_SINGULARITY_BIN).
export ALPASIM_SINGULARITY_BIN=apptainer
# Bind a FRESH, on-DISK /tmp into every service container. Two problems this solves:
#  (1) the NuRec renderer writes a deterministic /tmp/collector_<hash>.slang; a stale
#      copy owned by another user on the shared host /tmp is read-only -> Permission
#      denied. A fresh dir we own has no clash.
#  (2) --writable-tmpfs makes /tmp an in-MEMORY overlay that is too small for NuRec's
#      temp usage -> "No space left on device". An on-disk dir has ample space.
CTMP="$LOG_DIR/ctmp"; rm -rf "$CTMP"; mkdir -p "$CTMP"
export APPTAINER_BIND="$CTMP:/tmp"
# Cluster reaches the internet via proxy; bypass loopback so intra-node gRPC is direct.
: "${http_proxy:=http://172.16.1.2:3128}"; export http_proxy; export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"; export NO_PROXY="$no_proxy"
export no_grpc_proxy="$no_proxy"
# Never mutate the prebuilt venv at runtime.
export UV_NO_SYNC=1
# Tokens forwarded into containers (scene is cached; usually unused at runtime).
export HF_TOKEN HUGGING_FACE_HUB_TOKEN NGC_API_KEY 2>/dev/null || true

mkdir -p "$LOG_DIR"
cd "$ALP"
echo "== M0 local GT-replay: scene=$SCENE  log_dir=$LOG_DIR =="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true

"$ALP/.venv/bin/alpasim_wizard" \
  deploy=local_singularity_hostvenv topology=1gpu \
  driver=manual +cameras=av2_1cam +exp/sim=av2_recon \
  '~wizard.external_services.driver' \
  'driver.inference.use_cameras=[ring_front_center]' \
  "scenes.local_usdz_dir=$RECON" \
  "scenes.scene_ids=[$SCENE]" \
  runtime.simulation_config.force_gt_duration_us=20000000 \
  runtime.simulation_config.skip_driver_during_force_gt=true \
  wizard.log_dir="$LOG_DIR"
echo "== WIZARD_EXIT=$? =="
echo "-- results --"; find "$LOG_DIR/aggregate" "$LOG_DIR/eval" "$LOG_DIR/rollouts" \
  -name 'results-summary.json' -o -name '*.mp4' -o -name 'metrics*.parquet' 2>/dev/null | head
