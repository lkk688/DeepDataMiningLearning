#!/usr/bin/env bash
# ==============================================================================
# isaac_containerizer.sh — One-command workflow on NVIDIA Isaac ROS base (nvcr.io)
#
# Features
#   • TARGET-aware BASE_IMAGE (x86_64 default, Jetson/aarch64 optional)
#   • X11 / Wayland GUI forwarding helpers for RViz/Gazebo
#   • GPU-ready run with NVIDIA capabilities (graphics/video/compute/display)
#   • Build/shell/stop/delete/commit/push/zip/scp/selftest
#
# Quick start (x86_64 host, X11):
#   ./isaac_containerizer.sh init
#   ./isaac_containerizer.sh login             # NGC API key
#   ./isaac_containerizer.sh pull
#   ./isaac_containerizer.sh build
#   xhost +local:root                          # (X11 helper)
#   ./isaac_containerizer.sh shell --gui x11 -m ~/ws
#
# Jetson:
#   ./isaac_containerizer.sh init
#   sed -i 's/^TARGET=.*/TARGET=jetson/' .isaac_containerizer.env
#   ./isaac_containerizer.sh login
#   ./isaac_containerizer.sh pull
#   ./isaac_containerizer.sh build
#   xhost +local:root
#   ./isaac_containerizer.sh shell --gui x11 -m ~/ws
#
# Wayland (often requires non-root):
#   ./isaac_containerizer.sh shell --gui wayland --user host -m ~/ws
#
# Self-test inside container:
#   nvidia-smi
#   ros2 pkg list | head
#   ros2 run demo_nodes_cpp talker   # and in another shell: listener
# ==============================================================================

set -euo pipefail

# -------------------- Defaults (override via .isaac_containerizer.env or env) ---
PROJECT_NAME="${PROJECT_NAME:-ai-isaac-ros}"
IMAGE_REPO="${IMAGE_REPO:-$PROJECT_NAME}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-$PROJECT_NAME}"
HOST_MOUNT_DIR="${HOST_MOUNT_DIR:-$PWD/workspace}"

# Target & ROS 2 distro
TARGET="${TARGET:-x86_64}"                 # x86_64 | jetson
ROS2_DISTRO="${ROS2_DISTRO:-humble}"       # humble (22.04) or jazzy (24.04)

# Resolve arch key and default BASE_IMAGE
case "$TARGET" in
  x86_64) ARCH_KEY="x86_64" ;;
  jetson) ARCH_KEY="aarch64" ;;
  *) echo "Unknown TARGET=$TARGET (use x86_64|jetson)"; exit 1;;
esac
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/isaac/ros:${ARCH_KEY}-ros2_${ROS2_DISTRO}}"

# Python & extras (installed in venv to avoid apt/pip clashes)
USE_VENV="${USE_VENV:-true}"
INSTALL_TORCH="${INSTALL_TORCH:-false}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu124}"  # cu121|cu122|cu124
INSTALL_MMDET3D="${INSTALL_MMDET3D:-false}"
EXTRA_APT="${EXTRA_APT:-}"                 # e.g. 'ffmpeg tmux'
EXTRA_PIP="${EXTRA_PIP:-}"                 # e.g. 'matplotlib jupyterlab'

# Runtime
SHM_SIZE="${SHM_SIZE:-8g}"
REMOVE_ON_EXIT="${REMOVE_ON_EXIT:-false}"
EXTRA_DOCKER_RUN_ARGS="${EXTRA_DOCKER_RUN_ARGS:-}"

# -------------------- GUI forwarding (X11/Wayland) ------------------------------
# We don’t force these globally; they’re added on-demand by:  --gui x11|wayland
# For X11 on host (Linux):
#   xhost +local:root     # allow root in local containers
# For Wayland:
#   You’ll usually need to run as host user:  --user host
#   And have WAYLAND_DISPLAY & XDG_RUNTIME_DIR set on host.

# -------------------- UI helpers ------------------------------------------------
c() { local code="$1"; shift; printf "\033[%sm%s\033[0m\n" "$code" "$*"; }
info(){ c "1;34" "ℹ $*"; }
ok()  { c "1;32" "✔ $*"; }
warn(){ c "1;33" "⚠ $*"; }
err() { c "1;31" "✘ $*"; }

usage() {
cat <<EOF
isaac_containerizer.sh — extend NVIDIA Isaac ROS base and manage it

USAGE:
  ./isaac_containerizer.sh init
  ./isaac_containerizer.sh login
  ./isaac_containerizer.sh pull
  ./isaac_containerizer.sh build [--no-cache]
  ./isaac_containerizer.sh shell [-m /host/dir] [--gui x11|wayland] [--user host]
  ./isaac_containerizer.sh stop | delete
  ./isaac_containerizer.sh commit -m "msg" [-t tag]
  ./isaac_containerizer.sh push
  ./isaac_containerizer.sh zip [outfile.tar.gz]
  ./isaac_containerizer.sh scp [outfile.tar.gz]
  ./isaac_containerizer.sh selftest
  ./isaac_containerizer.sh x11-allow | x11-revoke

CONFIG (.isaac_containerizer.env):
  TARGET=$TARGET            (x86_64 [default] or jetson)
  ROS2_DISTRO=$ROS2_DISTRO  (humble|jazzy)
  BASE_IMAGE=$BASE_IMAGE
  USE_VENV=$USE_VENV  INSTALL_TORCH=$INSTALL_TORCH TORCH_CUDA_TAG=$TORCH_CUDA_TAG
  INSTALL_MMDET3D=$INSTALL_MMDET3D
  EXTRA_APT="$EXTRA_APT"    EXTRA_PIP="$EXTRA_PIP"
  SHM_SIZE=$SHM_SIZE        REMOVE_ON_EXIT=$REMOVE_ON_EXIT
EOF
}

# -------------------- Internals -------------------------------------------------
load_env() { [[ -f .isaac_containerizer.env ]] && source .isaac_containerizer.env || true; }
ensure_workspace(){ mkdir -p "$HOST_MOUNT_DIR"; }
image_ref(){ printf "%s:%s" "$IMAGE_REPO" "$IMAGE_TAG"; }
ensure_docker(){ docker info >/dev/null 2>&1 || { err "Docker not reachable."; exit 1; }; }
ensure_gpu_note(){
  if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    warn "GPU test failed. Check NVIDIA drivers + nvidia-container-toolkit."
  fi
}
get_container_state(){ docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "absent"; }

# Compose GUI args based on mode and user
compose_gui_args() {
  local mode="$1" ; shift || true
  local -a args=()
  case "$mode" in
    x11)
      # Host prep: xhost +local:root
      [[ -z "${DISPLAY:-}" ]] && warn "DISPLAY is empty on host; X11 may not work."
      args+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
      # Enable full graphics+display caps
      args+=(-e NVIDIA_DRIVER_CAPABILITIES=all)
      # Reduce MIT-SHM issues with X11
      args+=(-e QT_X11_NO_MITSHM=1)
      ;;
    wayland)
      # Wayland sockets are per-user; usually requires --user host
      local host_uid; host_uid="$(id -u)"
      local xdg="${XDG_RUNTIME_DIR:-/run/user/$host_uid}"
      local wld="${WAYLAND_DISPLAY:-wayland-0}"
      [[ -S "$xdg/$wld" ]] || warn "Wayland socket $xdg/$wld not found; Wayland may fail."
      args+=(-e XDG_RUNTIME_DIR="/run/user/$host_uid" -e WAYLAND_DISPLAY="$wld")
      args+=(-v "$xdg":"/run/user/$host_uid":rw)
      # Encourage Qt/Gtk to use Wayland
      args+=(-e QT_QPA_PLATFORM=wayland -e GDK_BACKEND=wayland)
      # Full graphics capability
      args+=(-e NVIDIA_DRIVER_CAPABILITIES=all)
      ;;
    ""|none) ;;
    *) warn "Unknown GUI mode '$mode' (use x11|wayland)";;
  esac
  printf '%s\n' "${args[@]}"
}

# -------------------- Commands --------------------------------------------------
cmd_init() {
  ensure_workspace
  if [[ ! -f .isaac_containerizer.env ]]; then
    cat > .isaac_containerizer.env <<EOF
PROJECT_NAME=$PROJECT_NAME
IMAGE_REPO=$IMAGE_REPO
IMAGE_TAG=$IMAGE_TAG
CONTAINER_NAME=$CONTAINER_NAME
HOST_MOUNT_DIR=$HOST_MOUNT_DIR

# Target & ROS
TARGET=$TARGET
ROS2_DISTRO=$ROS2_DISTRO
BASE_IMAGE=$BASE_IMAGE

# Python & extras
USE_VENV=$USE_VENV
INSTALL_TORCH=$INSTALL_TORCH
TORCH_CUDA_TAG=$TORCH_CUDA_TAG
INSTALL_MMDET3D=$INSTALL_MMDET3D
EXTRA_APT=$EXTRA_APT
EXTRA_PIP=$EXTRA_PIP

# Runtime
SHM_SIZE=$SHM_SIZE
REMOVE_ON_EXIT=$REMOVE_ON_EXIT
EXTRA_DOCKER_RUN_ARGS=$EXTRA_DOCKER_RUN_ARGS
EOF
    ok "Created .isaac_containerizer.env"
  else
    warn ".isaac_containerizer.env exists; not overwriting."
  fi
  ok "Workspace ready at $HOST_MOUNT_DIR"
}

cmd_login() {
  ensure_docker
  local key="${NGC_API_KEY:-}"
  if [[ -z "$key" ]]; then read -rsp "Enter your NGC API key: " key; echo; fi
  info "Logging in to nvcr.io…"
  echo "$key" | docker login nvcr.io -u '$oauthtoken' --password-stdin
  ok "nvcr.io login OK"
}

cmd_pull() {
  load_env; ensure_docker
  info "Pulling $BASE_IMAGE"
  docker pull "$BASE_IMAGE"
  ok "Pulled base image."
}

generate_dockerfile() {
  local df="Dockerfile.generated"
  info "Generating $df (FROM $BASE_IMAGE)"
  cat > "$df" <<'DOCKERFILE'
# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG USE_VENV=true
ARG INSTALL_TORCH=false
ARG TORCH_CUDA_TAG=cu124
ARG INSTALL_MMDET3D=false
ARG EXTRA_APT=""
ARG EXTRA_PIP=""
ARG ROS2_DISTRO=humble

SHELL ["/bin/bash", "-lc"]

# Isaac ROS base already has ROS 2 + CUDA/TensorRT. Add common tools:
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion ca-certificates curl wget git vim nano \
    python3-venv python3-pip python3-dev build-essential \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Optional extra apt packages
RUN if [[ -n "${EXTRA_APT}" ]]; then \
      apt-get update && apt-get install -y --no-install-recommends ${EXTRA_APT} && rm -rf /var/lib/apt/lists/*; \
    fi

# Keep Python deps separate
# Keep Python deps separate (overwrite any preexisting pip/python shims)
RUN if [[ "${USE_VENV}" == "true" ]]; then \
      python3 -m venv /opt/venv && \
      /opt/venv/bin/pip install -U pip wheel setuptools && \
      ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
      ln -sf /opt/venv/bin/python /usr/local/bin/python; \
    else \
      python3 -m pip install -U pip wheel setuptools; \
    fi

# Optional: PyTorch (CUDA wheels)
RUN if [[ "${INSTALL_TORCH}" == "true" ]]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch torchvision torchaudio; \
    fi

# Optional: OpenMMLab stack
RUN pip install -U openmim "numpy<2.0" || true \
 && if [[ "${INSTALL_MMDET3D}" == "true" ]]; then \
      mim install "mmengine>=0.10.4" && \
      mim install "mmcv>=2.1.0" && \
      mim install "mmdet>=3.2.0" && \
      if ! mim install "mmdet3d>=1.4.0"; then \
        cd /opt && git clone --depth=1 https://github.com/open-mmlab/mmdetection3d.git && cd mmdetection3d && \
        pip install -r requirements/runtime.txt && pip install -v -e .; \
      fi; \
    fi

# Optional: extra pip
RUN if [[ -n "${EXTRA_PIP}" ]]; then pip install ${EXTRA_PIP}; fi

WORKDIR /workspace
DOCKERFILE
  ok "Dockerfile.generated created."
}

cmd_build() {
  load_env; ensure_docker; ensure_gpu_note; ensure_workspace
  local NO_CACHE="false"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-cache) NO_CACHE="true"; shift ;;
      *) err "Unknown flag: $1"; usage; exit 1;;
    esac
  done

  generate_dockerfile

  info "Building $(image_ref) FROM $BASE_IMAGE"
  local args=(
    --build-arg BASE_IMAGE="$BASE_IMAGE"
    --build-arg USE_VENV="$USE_VENV"
    --build-arg INSTALL_TORCH="$INSTALL_TORCH"
    --build-arg TORCH_CUDA_TAG="$TORCH_CUDA_TAG"
    --build-arg INSTALL_MMDET3D="$INSTALL_MMDET3D"
    --build-arg EXTRA_APT="$EXTRA_APT"
    --build-arg EXTRA_PIP="$EXTRA_PIP"
    --build-arg ROS2_DISTRO="$ROS2_DISTRO"
    -t "$(image_ref)" -f Dockerfile.generated .
  )
  if [[ "$NO_CACHE" == "true" ]]; then
    DOCKER_BUILDKIT=1 docker build --no-cache "${args[@]}"
  else
    docker build "${args[@]}"
  fi
  ok "Build complete: $(image_ref)"
}

get_container_state() {
  local s
  s="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  # strip CR/LF and spaces
  s="${s//$'\r'/}"; s="${s//$'\n'/}"; s="${s//[[:space:]]/}"
  if [[ -z "$s" ]]; then echo "absent"; else echo "$s"; fi
}

cmd_shell() {
  load_env; ensure_workspace

  local GUI_MODE="none" USER_MODE="container"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--mount) HOST_MOUNT_DIR="$2"; shift 2;;
      --gui) GUI_MODE="$2"; shift 2;;
      --user) USER_MODE="$2"; shift 2;;
      *) err "Unknown flag: $1"; usage; exit 1;;
    esac
  done

  IFS=$'\n' read -r -d '' -a GUI_ARGS < <(compose_gui_args "$GUI_MODE" && printf '\0')
  USER_ARGS=(); [[ "$USER_MODE" == "host" ]] && USER_ARGS=(--user "$(id -u)":"$(id -g)")

  local state; state="$(get_container_state)"
  local run_rm_arg=(); [[ "${REMOVE_ON_EXIT}" == "true" ]] && run_rm_arg=(--rm)

  case "$state" in
    running)
      info "Container $CONTAINER_NAME is running. Attaching…"
      exec docker exec -it "${USER_ARGS[@]}" "$CONTAINER_NAME" bash
      ;;
    exited|created|paused|dead)
      info "Found existing container ($state). Starting and attaching…"
      exec docker start -ai "$CONTAINER_NAME"
      ;;
    absent)
      info "Starting new container $CONTAINER_NAME from $(image_ref)"
      mkdir -p "$HOST_MOUNT_DIR"
      exec docker run -it "${run_rm_arg[@]}" \
        --name "$CONTAINER_NAME" \
        --gpus all --ipc=host --ulimit memlock=-1:-1 --shm-size="$SHM_SIZE" \
        -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v "$HOST_MOUNT_DIR":/workspace -v "$HOME/.cache":/root/.cache \
        "${GUI_ARGS[@]}" ${EXTRA_DOCKER_RUN_ARGS:-} "${USER_ARGS[@]}" \
        "$(image_ref)" bash
      ;;
    *)
      warn "State '$state' not recognized; treating as absent."
      docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
      exec docker run -it "${run_rm_arg[@]}" \
        --name "$CONTAINER_NAME" \
        --gpus all --ipc=host --ulimit memlock=-1:-1 --shm-size="$SHM_SIZE" \
        -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v "$HOST_MOUNT_DIR":/workspace -v "$HOME/.cache":/root/.cache \
        "${GUI_ARGS[@]}" ${EXTRA_DOCKER_RUN_ARGS:-} "${USER_ARGS[@]}" \
        "$(image_ref)" bash
      ;;
  esac
}

cmd_stop()   { load_env; docker ps -q -f "name=^${CONTAINER_NAME}$" >/dev/null && docker stop "$CONTAINER_NAME" >/dev/null || warn "Not running."; ok "Stopped (if running)."; }
cmd_delete() { load_env; docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; ok "Deleted $CONTAINER_NAME"; }

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
  [[ -z "$msg" ]] && { err "Commit requires -m \"message\""; exit 1; }
  [[ -z "$new_tag" ]] && new_tag="$(date +%Y%m%d%H%M)"
  [[ -z "$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" ]] && { err "Container not found."; exit 1; }
  local new_image="${IMAGE_REPO}:${new_tag}"
  info "Committing $CONTAINER_NAME -> $new_image"
  docker commit -m "$msg" "$CONTAINER_NAME" "$new_image" >/dev/null
  ok "Committed as $new_image"
  info "Set IMAGE_TAG=$new_tag in .isaac_containerizer.env to use it by default."
}

cmd_push() { load_env; info "Pushing $(image_ref)"; docker push "$(image_ref)"; ok "Pushed."; }

cmd_zip() {
  load_env
  local out="${1:-$(echo "$(image_ref)" | tr '/:' '__').tar.gz}"
  info "Saving $(image_ref) -> $out"
  docker save "$(image_ref)" | gzip > "$out"
  ok "Saved $out"
  info "Load elsewhere: docker load -i $out"
}

cmd_scp() {
  load_env
  local out="${1:-$(echo "$(image_ref)" | tr '/:' '__').tar.gz}"
  [[ -f "$out" ]] || { warn "$out not found; creating…"; cmd_zip "$out"; }
  read -rp "Destination (user@host:/path/): " dest
  read -rp "SSH port [22]: " port; port="${port:-22}"
  info "Uploading…"; scp -P "$port" "$out" "$dest"
  ok "Uploaded. On remote: docker load -i $(basename "$out")"
}

cmd_selftest() {
cat <<'EOT'

=================== SELF-TEST (run inside the container) ===================

# GPU
nvidia-smi

# ROS 2 (Isaac ROS base ships ROS 2)
ros2 --help | head
ros2 pkg list | head
# Two terminals:
#   A: ros2 run demo_nodes_cpp talker
#   B: ros2 run demo_nodes_cpp listener

# GUI quick checks
#   RViz2:   rviz2
#   Gazebo:  gazebo    (or ign gazebo, depending on distro)

# Python env
python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch; import torch.cuda as cu
    print("Torch:", torch.__version__, "CUDA available:", cu.is_available())
except Exception as e:
    print("Torch not installed or failed:", e)
PY

# Workspace mount
ls -la /workspace

=============================================================================
EOT
}

# Convenience helpers for X11 on host
cmd_x11_allow()  { xhost +local:root  >/dev/null 2>&1 || true; ok "X11: allowed local root"; }
cmd_x11_revoke() { xhost -local:root  >/dev/null 2>&1 || true; ok "X11: revoked local root"; }

# -------------------- Dispatcher ----------------------------------------------
case "${1:-help}" in
  help|-h|--help) usage ;;
  init)           shift; cmd_init "$@" ;;
  login)          shift; cmd_login "$@" ;;
  pull)           shift; cmd_pull "$@" ;;
  build)          shift; cmd_build "$@" ;;
  shell)          shift; cmd_shell "$@" ;;
  stop)           shift; cmd_stop "$@" ;;
  delete)         shift; cmd_delete "$@" ;;
  commit)         shift; cmd_commit "$@" ;;
  push)           shift; cmd_push "$@" ;;
  zip)            shift; cmd_zip "$@" ;;
  scp)            shift; cmd_scp "$@" ;;
  selftest)       shift; cmd_selftest "$@" ;;
  x11-allow)      shift; cmd_x11_allow "$@" ;;
  x11-revoke)     shift; cmd_x11_revoke "$@" ;;
  *) usage; exit 1;;
esac