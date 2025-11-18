#!/usr/bin/env bash
# cuda_local.sh
# Modes:
#   auto (default): if CUDA version already installed -> run check only; else install then check
#   install:        install specified CUDA version then check
#   check:          just run checks (uses /usr/local/cuda or a provided path)
#
# Syntax:
#   ./cuda_local.sh [auto|install] CUDA_VERSION [INSTALL_PREFIX=/usr/local] [EXPORT_TO_BASH=1]
#   ./cuda_local.sh check [CUDA_HOME_OR_DIR=/usr/local/cuda]
#
# CUDA_VERSION ∈ {110,111,112,113,114,115,116,117,118,120,121,122,123,124,125,126,128}

set -euo pipefail

# ---------- logging ----------
log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

# ---------- URLs & folders ----------
declare -A URLS FOLDERS
URLS[110]="https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run"
URLS[111]="https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run"
URLS[112]="https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run"
URLS[113]="https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run"
URLS[114]="https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run"
URLS[115]="https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run"
URLS[116]="https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run"
URLS[117]="https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run"
URLS[118]="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
URLS[120]="https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run"
URLS[121]="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run"
URLS[122]="https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run"
URLS[123]="https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run"
URLS[124]="https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run"
URLS[125]="https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run"
URLS[126]="https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run"
URLS[128]="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run"

FOLDERS[110]="cuda-11.0"; FOLDERS[111]="cuda-11.1"; FOLDERS[112]="cuda-11.2"
FOLDERS[113]="cuda-11.3"; FOLDERS[114]="cuda-11.4"; FOLDERS[115]="cuda-11.5"
FOLDERS[116]="cuda-11.6"; FOLDERS[117]="cuda-11.7"; FOLDERS[118]="cuda-11.8"
FOLDERS[120]="cuda-12.0"; FOLDERS[121]="cuda-12.1"; FOLDERS[122]="cuda-12.2"
FOLDERS[123]="cuda-12.3"; FOLDERS[124]="cuda-12.4"; FOLDERS[125]="cuda-12.5"
FOLDERS[126]="cuda-12.6"; FOLDERS[128]="cuda-12.8"

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
content_length() {
  if have curl; then curl -sIL "$1" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{print $2}' | tail -n1; 
  elif have wget; then wget --spider --server-response -O - "$1" 2>&1 | awk 'BEGIN{IGNORECASE=1}/Content-Length:/{print $2}' | tail -n1;
  fi
}
download_with_resume() {
  local url="$1" out="$2" tries="${3:-6}"
  local expect; expect="$(content_length "$url" || true)"
  log "Expected size (bytes): ${expect:-unknown}"
  local i rc; i=1
  while (( i<=tries )); do
    log "Download attempt $i/$tries → $out"
    if have aria2c; then set +e; aria2c -x 16 -s 16 -k 1M -o "$(basename "$out")" -d "$(dirname "$out")" "$url"; rc=$?; set -e
    elif have curl; then set +e; curl -L --retry 8 --retry-all-errors --speed-time 30 --speed-limit 100000 -C - -o "$out" "$url"; rc=$?; set -e
    elif have wget; then set +e; wget -c --tries=8 --read-timeout=30 --timeout=30 -O "$out" "$url"; rc=$?; set -e
    else die "Need curl or wget (or aria2c)."; fi
    if [[ -f "$out" ]]; then
      if [[ -n "$expect" ]]; then
        local have_sz; have_sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
        log "Downloaded size: $have_sz bytes"
        [[ "$have_sz" == "$expect" ]] && { log "Download verified by Content-Length."; return 0; }
      else [[ $rc -eq 0 ]] && return 0; fi
    fi
    warn "Download incomplete; retrying..."
    sleep 5; i=$((i+1))
  done
  die "Failed to download complete file after $tries attempts."
}
check_space_gb() {
  local path="$1" need="$2"
  local avail_kb; avail_kb=$(df -Pk "$path" | awk 'NR==2{print $4}')
  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  (( avail_gb >= need )) || die "Not enough free space on $path: need ${need}GB, have ${avail_gb}GB."
}
auto_sudo() {
  local target="$1"
  if [[ -w "$target" ]]; then echo ""; return 0; fi
  if [[ $EUID -eq 0 ]]; then echo ""; return 0; fi
  have sudo || die "Need root to write $target (sudo not found)."
  echo "sudo"
}
nvcc_path() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then echo "${CUDA_HOME}/bin/nvcc"; return; fi
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then echo "/usr/local/cuda/bin/nvcc"; return; fi
  command -v nvcc || true
}

# ---------- check (env + compile + run) ----------
check_cuda() {
  local cuda_dir="${1:-/usr/local/cuda}"
  local NVCC="${cuda_dir}/bin/nvcc"
  log "Checking CUDA at: $cuda_dir"

  if [[ ! -x "$NVCC" ]]; then
    warn "nvcc not found at $NVCC"
    if command -v nvcc >/dev/null 2>&1; then NVCC="$(command -v nvcc)"; warn "Using nvcc from PATH: $NVCC"; else die "nvcc not found."; fi
  fi

  "$NVCC" -V || true

  # decide SM flags
  local SM_FLAGS=""
  if have nvidia-smi; then
    local CAP; CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
    if [[ "$CAP" =~ ^[0-9]+$ ]]; then SM_FLAGS="-gencode arch=compute_${CAP},code=sm_${CAP}"; fi
  fi
  if [[ -z "$SM_FLAGS" ]]; then
    SM_FLAGS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90"
  fi

  # Tiny smoketest
  local TEST_CU OUT_BIN
  TEST_CU="$(mktemp /tmp/cuda_test.XXXXXX.cu)"
  cat > "$TEST_CU" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); return 10; } }while(0)
__global__ void hello(int *out){ *out = 42; }
int main(){
  int n=0; CK(cudaGetDeviceCount(&n));
  if(n<1){ std::fprintf(stderr,"No CUDA device.\n"); return 2; }
  int *d=nullptr, h=-1; CK(cudaMalloc(&d,sizeof(int)));
  hello<<<1,1>>>(d);
  CK(cudaPeekAtLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h,d,sizeof(int),cudaMemcpyDeviceToHost));
  CK(cudaFree(d));
  std::printf("CUDA OK. Devices=%d, KernelValue=%d\n", n, h);
  return (h==42)?0:3;
}
CU
  OUT_BIN="/tmp/cuda_test.out"

  set +e
  "$NVCC" -O2 $SM_FLAGS "$TEST_CU" -o "$OUT_BIN" 2>/tmp/cuda_test_build.log
  local BUILD_RC=$?
  if (( BUILD_RC != 0 )); then
    warn "nvcc compile failed; see /tmp/cuda_test_build.log"
    return 20
  fi
  "$OUT_BIN"
  local RUN_RC=$?
  set -e
  if (( RUN_RC == 0 )); then
    log "CUDA test ran successfully."
    return 0
  else
    warn "CUDA runtime test returned code $RUN_RC (driver/GPU availability or permissions issue?)."
    return 21
  fi
}

# ---------- install (toolkit) ----------
install_cuda() {
  local ver="$1" prefix="${2:-/usr/local}" export_bash="${3:-1}"
  local RUN_URL="${URLS[$ver]:-}" CUDA_FOLDER="${FOLDERS[$ver]:-}"
  [[ -n "$RUN_URL" && -n "$CUDA_FOLDER" ]] || die "Unsupported CUDA_VERSION '$ver'."
  local OS ARCH; OS="$(uname -s)"; ARCH="$(uname -m)"
  [[ "$OS" == "Linux" ]] || die "Linux only (got: $OS)."
  [[ "$ARCH" == "x86_64" ]] || warn "Architecture is '$ARCH' (runfiles are for x86_64)."

  if have nvidia-smi; then log "NVIDIA driver detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"; else warn "No 'nvidia-smi' found (driver may be missing)."; fi
  if grep -qi microsoft /proc/version 2>/dev/null; then warn "WSL detected; ensure Windows NVIDIA driver + WSL CUDA support are set up."; fi

  local INSTALL_DIR="${prefix%/}/${CUDA_FOLDER}"
  check_space_gb /tmp 6
  check_space_gb "$(dirname "$INSTALL_DIR")" 4

  local WORKDIR INSTALLER
  WORKDIR="$(mktemp -d)"; trap 'rm -rf "$WORKDIR"' RETURN
  INSTALLER="$WORKDIR/$(basename "$RUN_URL")"

  log "Downloading CUDA runfile: $RUN_URL"
  download_with_resume "$RUN_URL" "$INSTALLER"
  chmod +x "$INSTALLER"

  local SUDO; SUDO=$(auto_sudo "$INSTALL_DIR")
  if [[ ! -d "$INSTALL_DIR" ]]; then log "Creating install directory: $INSTALL_DIR"; $SUDO mkdir -p "$INSTALL_DIR"; fi

  log "Running installer (toolkit only) to: $INSTALL_DIR"
  $SUDO sh "$INSTALLER" --silent --toolkit --toolkitpath="$INSTALL_DIR" --no-man-page --override --no-drm

  log "Creating/updating symlink: /usr/local/cuda -> $INSTALL_DIR"
  local SUDO_USR; SUDO_USR=$(auto_sudo "/usr/local")
  $SUDO_USR ln -sfn "$INSTALL_DIR" /usr/local/cuda

  local ENV_BLOCK
  ENV_BLOCK=$(cat <<EOF
# >>> CUDA (${CUDA_FOLDER}) >>>
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export CPATH=\$CUDA_HOME/include\${CPATH:+:\$CPATH}
export LIBRARY_PATH=\$CUDA_HOME/lib64:\$CUDA_HOME/lib64/stubs\${LIBRARY_PATH:+:\$LIBRARY_PATH}
export CMAKE_PREFIX_PATH=\$CUDA_HOME\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}
# <<< CUDA (${CUDA_FOLDER}) <<<
EOF
)
  if [[ "${export_bash}" == "1" ]]; then
    log "Appending environment exports to ~/.bashrc"
    printf "\n%s\n" "$ENV_BLOCK" >> "$HOME/.bashrc"
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" || true
  else
    warn "EXPORT_TO_BASH=0 — remember to export env in your shell:"
    echo "$ENV_BLOCK"
  fi

  log "Install step complete: ${CUDA_FOLDER} at ${INSTALL_DIR}"
}

# ---------- detect if installed ----------
is_version_installed() {
  local cuda_folder="$1"
  [[ -x "/${cuda_folder}/bin/nvcc" ]] && return 0
  [[ -x "/usr/local/${cuda_folder}/bin/nvcc" ]] && return 0
  [[ -x "/usr/local/cuda/bin/nvcc" ]] && /usr/local/cuda/bin/nvcc -V 2>/dev/null | grep -q "${cuda_folder/cuda-/}" && return 0
  return 1
}

# ---------- main ----------
MODE="${1:-auto}"

if [[ "$MODE" == "check" ]]; then
  TARGET="${2:-/usr/local/cuda}"
  check_cuda "$TARGET"
  exit $?
fi

# install or auto require a version
CUDA_VERSION="${2:-}"; [[ -n "$CUDA_VERSION" ]] || die "Usage: $0 [auto|install] CUDA_VERSION [INSTALL_PREFIX=/usr/local] [EXPORT_TO_BASH=1]"
INSTALL_PREFIX="${3:-/usr/local}"
EXPORT_BASHRC="${4:-1}"

RUN_URL="${URLS[$CUDA_VERSION]:-}"; CUDA_FOLDER="${FOLDERS[$CUDA_VERSION]:-}"
[[ -n "$RUN_URL" && -n "$CUDA_FOLDER" ]] || die "Unsupported CUDA_VERSION '$CUDA_VERSION'."

if [[ "$MODE" == "auto" ]]; then
  if is_version_installed "usr/local/${CUDA_FOLDER}"; then
    log "Detected ${CUDA_FOLDER} already installed. Running check only..."
    check_cuda "/usr/local/${CUDA_FOLDER}" || exit $?
    exit 0
  fi
  log "${CUDA_FOLDER} not found. Proceeding with install..."
  install_cuda "$CUDA_VERSION" "$INSTALL_PREFIX" "$EXPORT_BASHRC"
  log "Running post-install checks..."
  check_cuda "/usr/local/${CUDA_FOLDER}" || exit $?
  exit 0
elif [[ "$MODE" == "install" ]]; then
  install_cuda "$CUDA_VERSION" "$INSTALL_PREFIX" "$EXPORT_BASHRC"
  log "Running post-install checks..."
  check_cuda "/usr/local/${CUDA_FOLDER}" || exit $?
  exit 0
else
  die "Unknown mode '$MODE'. Use: auto | install | check"
fi


# ./cuda_local.sh auto 126
# # or simply:
# ./cuda_local.sh 126
# ./cuda_local.sh install 128 /usr/local 1
# #Only check
# ./cuda_local.sh check
# #./cuda_local.sh check /usr/local/cuda-12.6