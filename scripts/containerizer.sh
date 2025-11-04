#!/usr/bin/env bash
# ==============================================================================
# containerizer.sh — One-command Docker workflow for AI + ROS 2 + OpenMMLab
#
# WHAT THIS SCRIPT DOES
#   1) Build a development image FROM a base (e.g., NVIDIA CUDA Ubuntu) and
#      optionally install ROS 2 + OpenMMLab stack (mmengine/mmcv/mmdet/mmdet3d),
#      all into a dedicated Python venv to avoid apt/pip conflicts (blinker, etc).
#   2) Start/attach a GPU-enabled shell with host folder mounts and NVIDIA
#      recommended runtime settings; or stop/delete the container.
#   3) Commit changes, push to Docker Hub, save to a .tar.gz, and scp to a remote.
#
# HIGHLIGHTS
#   - DEFAULT CONTAINER NAME: ai-ros2
#   - Virtualenv (/opt/venv) ensures clean Python deps (fixes 'blinker' uninstall)
#   - ROS 2 sourcing is CONDITIONAL (no more /opt/ros/... missing errors)
#   - NVIDIA: --gpus all, --ipc=host, bigger --shm-size, memlock ulimit
#
# QUICK START
#   chmod +x containerizer.sh
#   ./containerizer.sh init
#   # (edit .containerizer.env if desired: IMAGE_REPO, ROS2_DISTRO, etc.)
#   ./containerizer.sh build --ros2 true --mmdet3d true
#   ./containerizer.sh shell -m ~/my_ws
#
# COMMON COMMANDS
#   ./containerizer.sh stop
#   ./containerizer.sh delete
#   ./containerizer.sh commit -m "add deps + configs" -t v0.1
#   ./containerizer.sh push
#   ./containerizer.sh zip
#   ./containerizer.sh scp
#   ./containerizer.sh selftest   # print test commands for ROS2, PyTorch, etc.
#
# SANITY TEST EXAMPLES (run INSIDE the container; also printed by `selftest`)
#   # GPU & CUDA:
#   nvidia-smi
#   python - <<'PY'
#   import torch; print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available()); print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
#   PY
#
#   # ROS 2:
#   ros2 --help
#   source /opt/ros/$ROS_DISTRO/setup.bash 2>/dev/null || true
#   ros2 pkg list | head
#   # Try a DDS ping (in two shells):
#   #   Shell A: ros2 run demo_nodes_cpp talker
#   #   Shell B: ros2 run demo_nodes_cpp listener
#
#   # OpenMMLab / MMDet3D:
#   python - <<'PY'
#   import mmengine, mmcv, mmdet
#   print("mmengine:", mmengine.__version__)
#   print("mmcv:", mmcv.__version__)
#   print("mmdet:", mmdet.__version__)
#   try:
#       import mmdet3d
#       print("mmdet3d:", mmdet3d.__version__)
#   except Exception as e:
#       print("mmdet3d not installed or failed:", e)
#   PY
#
#   # OpenCV (headless):
#   python - <<'PY'
#   import cv2; import numpy as np
#   img = np.zeros((100,100,3), dtype=np.uint8); cv2.circle(img,(50,50),20,(255,255,255),-1)
#   print("opencv:", cv2.__version__, "shape:", img.shape)
#   PY
#
# REQUIREMENTS
#   - Docker (with NVIDIA Container Toolkit)
#   - bash 4+, gzip, scp (optional)
#
# ==============================================================================

set -euo pipefail

# -------------------- Defaults (override via .containerizer.env or CLI flags) --------------------
PROJECT_NAME="${PROJECT_NAME:-ai-ros2}"
IMAGE_REPO="${IMAGE_REPO:-$PROJECT_NAME}"          # e.g. myuser/ai-ros2 (for push)
IMAGE_TAG="${IMAGE_TAG:-latest}"                   # build/run tag
CONTAINER_NAME="${CONTAINER_NAME:-$PROJECT_NAME}"  # default 'ai-ros2'
HOST_MOUNT_DIR="${HOST_MOUNT_DIR:-$PWD/workspace}" # host dir -> /workspace

# Base image: good general purpose for CUDA + Ubuntu 22.04
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04}"

# Feature toggles for generated Dockerfile
INSTALL_ROS2="${INSTALL_ROS2:-true}"               # true|false
ROS2_DISTRO="${ROS2_DISTRO:-humble}"               # Ubuntu 22.04 -> humble; Ubuntu 24.04 -> jazzy
INSTALL_MMDET3D="${INSTALL_MMDET3D:-true}"         # true|false

# PyTorch/CUDA wheels channel tag: cu121|cu122|cu124 (matches your base image/driver)
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu124}"

# NVIDIA runtime defaults
SHM_SIZE="${SHM_SIZE:-8g}"
EXTRA_DOCKER_RUN_ARGS="${EXTRA_DOCKER_RUN_ARGS:-}" # e.g. "-p 8888:8888"
REMOVE_ON_EXIT="${REMOVE_ON_EXIT:-false}"          # true -> docker run --rm; false -> keep container for commit

# Color helpers
c() { local code="$1"; shift; printf "\033[%sm%s\033[0m\n" "$code" "$*"; }
info(){ c "1;34" "ℹ $*"; }
ok()  { c "1;32" "✔ $*"; }
warn(){ c "1;33" "⚠ $*"; }
err() { c "1;31" "✘ $*"; }

usage() {
cat <<EOF
containerizer.sh — all-in-one container workflow (GPU + ROS 2 + OpenMMLab)

USAGE:
  ./containerizer.sh init
  ./containerizer.sh build [--base IMG] [--ros2 true|false] [--ros2-distro humble|jazzy] [--mmdet3d true|false] [--torch-cuda cu124] [-t TAG]
  ./containerizer.sh shell [-m /host/dir]
  ./containerizer.sh stop
  ./containerizer.sh delete
  ./containerizer.sh commit -m "message" [-t newtag]
  ./containerizer.sh push
  ./containerizer.sh zip [outfile.tar.gz]
  ./containerizer.sh scp [outfile.tar.gz]
  ./containerizer.sh selftest     # print recommended validation commands
  ./containerizer.sh help

CONFIG (override via .containerizer.env or env vars):
  PROJECT_NAME="$PROJECT_NAME"
  IMAGE_REPO="$IMAGE_REPO"
  IMAGE_TAG="$IMAGE_TAG"
  CONTAINER_NAME="$CONTAINER_NAME"
  HOST_MOUNT_DIR="$HOST_MOUNT_DIR"
  BASE_IMAGE="$BASE_IMAGE"
  INSTALL_ROS2="$INSTALL_ROS2"  ROS2_DISTRO="$ROS2_DISTRO"
  INSTALL_MMDET3D="$INSTALL_MMDET3D"  TORCH_CUDA_TAG="$TORCH_CUDA_TAG"
  SHM_SIZE="$SHM_SIZE"
  REMOVE_ON_EXIT="$REMOVE_ON_EXIT"
  EXTRA_DOCKER_RUN_ARGS="$EXTRA_DOCKER_RUN_ARGS"

NOTES:
  - Ubuntu 22.04 -> ROS2_DISTRO=humble; Ubuntu 24.04 -> ROS2_DISTRO=jazzy
  - Default keeps container after exit (REMOVE_ON_EXIT=false) so you can 'commit' later.
EOF
}

load_env() {
  if [[ -f .containerizer.env ]]; then
    # shellcheck disable=SC1091
    source .containerizer.env
  fi
}

ensure_workspace() {
  mkdir -p "$HOST_MOUNT_DIR"
}

ensure_gpu_runtime_note() {
  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon not reachable. Start Docker first."
    exit 1
  fi
  if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    warn "GPU test failed. Ensure NVIDIA drivers + nvidia-container-toolkit are installed and configured."
    warn "Proceeding anyway (CPU-only will still work)."
  fi
}

image_ref() {
  printf "%s:%s" "$IMAGE_REPO" "$IMAGE_TAG"
}

generate_dockerfile_if_missing() {
  local df="Dockerfile.generated"
  info "Generating $df with BASE_IMAGE=$BASE_IMAGE, ROS2=$INSTALL_ROS2 ($ROS2_DISTRO), MMDetection3D=$INSTALL_MMDET3D, TORCH=$TORCH_CUDA_TAG"
  cat > "$df" <<'DOCKERFILE'
# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_ROS2=true
ARG ROS2_DISTRO=humble
ARG INSTALL_MMDET3D=true
ARG TORCH_CUDA_TAG=cu124

# ---- Base system packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion ca-certificates curl wget git vim nano \
    build-essential cmake ninja-build pkg-config \
    python3 python3-pip python3-venv python3-dev \
    tzdata locales sudo unzip zip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

SHELL ["/bin/bash", "-lc"]

# ---- Optional: ROS 2 (desktop + dev tools), only if requested ----
RUN if [[ "${INSTALL_ROS2}" == "true" ]]; then \
      set -eux; \
      apt-get update && apt-get install -y software-properties-common; \
      add-apt-repository universe; \
      apt-get update && apt-get install -y curl gnupg lsb-release; \
      curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg; \
      echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $$(. /etc/os-release && echo $$UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list; \
      apt-get update && apt-get install -y \
        ros-${ROS2_DISTRO}-desktop \
        python3-colcon-common-extensions \
        ros-dev-tools; \
      # Add conditional sourcing ONLY if ROS is installed:
      echo '[ -f /opt/ros/'"${ROS2_DISTRO}"'/setup.bash ] && source /opt/ros/'"${ROS2_DISTRO}"'/setup.bash' >> /etc/bash.bashrc; \
    fi

# ---- Python virtualenv to avoid apt/distutils conflicts (e.g., blinker) ----
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

# Pre-empt apt's blinker and similar by forcing venv wheels
RUN pip install -U pip wheel setuptools \
 && pip install -U --ignore-installed "blinker>=1.7"

# ---- PyTorch (match CUDA wheels) ----
RUN pip install --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
    torch torchvision torchaudio

# ---- OpenMMLab core ----
RUN pip install -U openmim \
 && mim install "mmengine>=0.10.4" \
 && mim install "mmcv>=2.1.0" \
 && mim install "mmdet>=3.2.0"

# ---- Optional: MMDetection3D ----
RUN if [[ "${INSTALL_MMDET3D}" == "true" ]]; then \
      set -eux; \
      if ! mim install "mmdet3d>=1.4.0"; then \
        echo "mim install mmdet3d failed — attempting source install"; \
        cd /opt && git clone --depth=1 https://github.com/open-mmlab/mmdetection3d.git && cd mmdetection3d; \
        pip install -r requirements/runtime.txt; \
        pip install -v -e .; \
      fi; \
    fi

# ---- Prefer headless OpenCV in servers (avoid duplicate GUI libs) ----
RUN pip install -U "opencv-python-headless>=4.10.0" && pip uninstall -y opencv-python || true

# ---- Entry: /workspace and ROS-safe entrypoint ----
# ---- Entrypoint: ROS-safe, venv already on PATH via ENV ----
WORKDIR /workspace

# Create a tiny, robust entrypoint that sources the first available ROS 2 setup
RUN cat >/usr/local/bin/ctr-entrypoint <<'BASH'
#!/usr/bin/env bash
set -e
# Source first available ROS 2 setup (humble/jazzy/…)
if [ -d /opt/ros ]; then
  shopt -s nullglob
  for d in /opt/ros/*; do
    if [ -f "$d/setup.bash" ]; then
      # shellcheck disable=SC1090
      source "$d/setup.bash"
      break
    fi
  done
fi
exec "$@"
BASH
RUN chmod +x /usr/local/bin/ctr-entrypoint

ENTRYPOINT ["/usr/local/bin/ctr-entrypoint"]
CMD ["/bin/bash"]
DOCKERFILE
  ok "Dockerfile.generated created."
}

cmd_init() {
  ensure_workspace
  if [[ ! -f .containerizer.env ]]; then
    cat > .containerizer.env <<EOF
# Copy/edit these defaults, then re-run commands.
PROJECT_NAME=$PROJECT_NAME
IMAGE_REPO=$IMAGE_REPO
IMAGE_TAG=$IMAGE_TAG
CONTAINER_NAME=$CONTAINER_NAME
HOST_MOUNT_DIR=$HOST_MOUNT_DIR
BASE_IMAGE=$BASE_IMAGE
INSTALL_ROS2=$INSTALL_ROS2
ROS2_DISTRO=$ROS2_DISTRO
INSTALL_MMDET3D=$INSTALL_MMDET3D
TORCH_CUDA_TAG=$TORCH_CUDA_TAG
SHM_SIZE=$SHM_SIZE
REMOVE_ON_EXIT=$REMOVE_ON_EXIT
EXTRA_DOCKER_RUN_ARGS=$EXTRA_DOCKER_RUN_ARGS
EOF
    ok "Created .containerizer.env (edit as needed)."
  else
    warn ".containerizer.env already exists; not overwriting."
  fi
  ok "Workspace ready at $HOST_MOUNT_DIR"
}

cmd_build() {
  load_env
  ensure_workspace
  ensure_gpu_runtime_note

  # CLI overrides
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --base) BASE_IMAGE="$2"; shift 2;;
      --ros2) INSTALL_ROS2="$2"; shift 2;;
      --ros2-distro) ROS2_DISTRO="$2"; shift 2;;
      --mmdet3d) INSTALL_MMDET3D="$2"; shift 2;;
      --torch-cuda) TORCH_CUDA_TAG="$2"; shift 2;;
      --tag|-t) IMAGE_TAG="$2"; shift 2;;
      *) err "Unknown flag: $1"; usage; exit 1;;
    esac
  done

  generate_dockerfile_if_missing

  info "Building $(image_ref) from $BASE_IMAGE"
  docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg INSTALL_ROS2="$INSTALL_ROS2" \
    --build-arg ROS2_DISTRO="$ROS2_DISTRO" \
    --build-arg INSTALL_MMDET3D="$INSTALL_MMDET3D" \
    --build-arg TORCH_CUDA_TAG="$TORCH_CUDA_TAG" \
    -t "$(image_ref)" -f Dockerfile.generated .
  ok "Build complete: $(image_ref)"
}

cmd_shell() {
  load_env
  ensure_workspace

  # Optional override for mount
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--mount) HOST_MOUNT_DIR="$2"; shift 2;;
      *) err "Unknown flag: $1"; usage; exit 1;;
    esac
  done

  local cid
  cid="$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" || true

  # If already running, just exec into it
  if [[ -n "$cid" ]] && docker ps -q -f "id=$cid" >/dev/null; then
    info "Container $CONTAINER_NAME already running. Attaching shell..."
    exec docker exec -it "$CONTAINER_NAME" bash
  fi

  # Build run args
  local run_rm_arg=()
  if [[ "$REMOVE_ON_EXIT" == "true" ]]; then run_rm_arg=(--rm); fi

  info "Starting container $CONTAINER_NAME from $(image_ref)"
  mkdir -p "$HOST_MOUNT_DIR"
  docker run -it "${run_rm_arg[@]}" \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1:-1 \
    --shm-size="$SHM_SIZE" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$HOST_MOUNT_DIR":/workspace \
    -v "$HOME/.cache":/root/.cache \
    $EXTRA_DOCKER_RUN_ARGS \
    "$(image_ref)" bash
}

cmd_stop() {
  load_env
  if docker ps -q -f "name=^${CONTAINER_NAME}$" >/dev/null; then
    docker stop "$CONTAINER_NAME" >/dev/null || true
    ok "Stopped $CONTAINER_NAME"
  else
    warn "No running container named $CONTAINER_NAME"
  fi
}

cmd_delete() {
  load_env
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  ok "Deleted (stopped/removed) $CONTAINER_NAME"
}

cmd_commit() {
  load_env
  local msg="" new_tag=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--message) msg="$2"; shift 2;;
      -t|--tag) new_tag="$2"; shift 2;;
      *) err "Unknown flag: $1"; usage; exit 1;;
    esac
  done
  if [[ -z "$msg" ]]; then err "Commit message required. Use: commit -m \"message\""; exit 1; fi
  if [[ -z "$new_tag" ]]; then new_tag="$(date +%Y%m%d%H%M)"; fi

  if ! docker ps -aq -f "name=^${CONTAINER_NAME}$" >/dev/null; then
    err "Container $CONTAINER_NAME not found. Start it with 'shell' first, make changes, then commit."
    exit 1
  fi

  local new_image="${IMAGE_REPO}:${new_tag}"
  info "Committing $CONTAINER_NAME -> $new_image"
  docker commit -m "$msg" "$CONTAINER_NAME" "$new_image" >/dev/null
  ok "Committed as $new_image"
  info "Tip: set IMAGE_TAG=$new_tag in .containerizer.env if you want to run this tag by default."
}

cmd_push() {
  load_env
  info "Pushing $(image_ref)"
  docker push "$(image_ref)"
  ok "Pushed $(image_ref)"
}

cmd_zip() {
  load_env
  local out="${1:-$(echo "$(image_ref)" | tr '/:' '__').tar.gz}"
  info "Saving $(image_ref) to $out (gzip)..."
  docker save "$(image_ref)" | gzip > "$out"
  ok "Saved image archive: $out"
  info "To load elsewhere:  docker load -i $out"
}

cmd_scp() {
  load_env
  local out="${1:-$(echo "$(image_ref)" | tr '/:' '__').tar.gz}"
  if [[ ! -f "$out" ]]; then
    warn "$out not found. Creating it now via 'zip'..."
    cmd_zip "$out"
  fi

  read -rp "Destination (e.g., user@server:/path/): " dest
  read -rp "SSH port [22]: " port
  port="${port:-22}"

  info "scp -P $port $out $dest"
  scp -P "$port" "$out" "$dest"
  ok "Uploaded $out to $dest"
  info "On the destination: docker load -i $(basename "$out")"
}

cmd_selftest() {
  cat <<'EOT'

=================== SELF-TEST: RUN THESE INSIDE THE CONTAINER ===================

# 0) GPU present?
nvidia-smi

# 1) PyTorch + CUDA
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

# 2) ROS 2 basics
ros2 --help
source /opt/ros/$ROS_DISTRO/setup.bash 2>/dev/null || true
ros2 pkg list | head
# Two terminals:
#   A: ros2 run demo_nodes_cpp talker
#   B: ros2 run demo_nodes_cpp listener

# 3) OpenMMLab / MMDet3D
python - <<'PY'
import mmengine, mmcv, mmdet
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
try:
    import mmdet3d
    print("mmdet3d:", mmdet3d.__version__)
except Exception as e:
    print("mmdet3d not installed or failed:", e)
PY

# 4) OpenCV (headless)
python - <<'PY'
import cv2, numpy as np
img = np.zeros((100,100,3), dtype=np.uint8)
print("opencv:", cv2.__version__, "OK:", img.shape)
PY

# 5) File mount check
ls -la /workspace

===============================================================================

EOT
}

# -------------------- Main dispatch --------------------
case "${1:-help}" in
  help|-h|--help) usage ;;

  init)        shift; cmd_init "$@" ;;
  build)       shift; cmd_build "$@" ;;
  shell)       shift; cmd_shell "$@" ;;
  stop)        shift; cmd_stop "$@" ;;
  delete)      shift; cmd_delete "$@" ;;
  commit)      shift; cmd_commit "$@" ;;
  push)        shift; cmd_push "$@" ;;
  zip)         shift; cmd_zip "$@" ;;
  scp)         shift; cmd_scp "$@" ;;
  selftest)    shift; cmd_selftest "$@" ;;

  *) usage; exit 1;;
esac