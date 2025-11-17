#!/usr/bin/env bash
# cuda_local_install.sh
# Syntax: ./cuda_local_install.sh CUDA_VERSION [INSTALL_PREFIX=/usr/local] [EXPORT_TO_BASH=1]
# CUDA_VERSION ∈ {110,111,112,113,114,115,116,117,118,120,121,122,123,124,125,126,128}

set -euo pipefail

log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

trap 'err "Installation failed (line $LINENO). Check the logs above."' ERR

CUDA_VERSION="${1:-}"
INSTALL_PREFIX="${2:-/usr/local}"
EXPORT_BASHRC="${3:-1}"

[[ -n "$CUDA_VERSION" ]] || {
  echo "Usage: $0 CUDA_VERSION [INSTALL_PREFIX=/usr/local] [EXPORT_TO_BASH=1]"
  exit 2
}

declare -A URLS
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

declare -A FOLDERS
FOLDERS[110]="cuda-11.0"; FOLDERS[111]="cuda-11.1"; FOLDERS[112]="cuda-11.2"
FOLDERS[113]="cuda-11.3"; FOLDERS[114]="cuda-11.4"; FOLDERS[115]="cuda-11.5"
FOLDERS[116]="cuda-11.6"; FOLDERS[117]="cuda-11.7"; FOLDERS[118]="cuda-11.8"
FOLDERS[120]="cuda-12.0"; FOLDERS[121]="cuda-12.1"; FOLDERS[122]="cuda-12.2"
FOLDERS[123]="cuda-12.3"; FOLDERS[124]="cuda-12.4"; FOLDERS[125]="cuda-12.5"
FOLDERS[126]="cuda-12.6"; FOLDERS[128]="cuda-12.8"

RUN_URL="${URLS[$CUDA_VERSION]:-}"
CUDA_FOLDER="${FOLDERS[$CUDA_VERSION]:-}"
[[ -n "$RUN_URL" && -n "$CUDA_FOLDER" ]] || die "Unsupported CUDA_VERSION '$CUDA_VERSION'."

OS="$(uname -s)"; ARCH="$(uname -m)"
[[ "$OS" == "Linux" ]] || die "Linux only (got: $OS)."
if [[ "$ARCH" != "x86_64" ]]; then
  warn "Architecture is '$ARCH' (these runfiles are for x86_64)."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA driver detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
else
  warn "No 'nvidia-smi' in PATH. Driver may be missing; toolkit will install but runtime needs a driver."
fi
if grep -qi microsoft /proc/version 2>/dev/null; then
  warn "WSL detected; ensure Windows NVIDIA driver + WSL CUDA support are set up."
fi

INSTALL_DIR="${INSTALL_PREFIX%/}/${CUDA_FOLDER}"
SUDO=""; [[ $EUID -ne 0 || "$INSTALL_PREFIX" = "/usr/local" ]] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
content_length() {
  if have curl; then
    curl -sIL "$1" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{sz=$2}END{if(sz!="")print sz}'
  elif have wget; then
    wget --spider --server-response -O - "$1" 2>&1 | awk 'BEGIN{IGNORECASE=1}/Content-Length:/{sz=$2}END{if(sz!="")print sz}'
  else
    echo ""
  fi
}
check_space_gb() {
  local path="$1" need="$2"
  local avail_kb; avail_kb=$(df -Pk "$path" | awk 'NR==2{print $4}')
  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  (( avail_gb >= need )) || die "Not enough free space on $path: need ${need}GB, have ${avail_gb}GB."
}
download_with_resume() {
  local url="$1" out="$2" tries="${3:-6}"
  local expect; expect="$(content_length "$url" || true)"
  log "Expected size (bytes): ${expect:-unknown}"
  local i=1 rc=1
  while (( i<=tries )); do
    log "Download attempt $i/$tries → $out"
    if have aria2c; then
      # parallel download if aria2c exists
      set +e
      aria2c -x 16 -s 16 -k 1M -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
      rc=$?
      set -e
    elif have curl; then
      set +e
      curl -L --retry 8 --retry-all-errors --speed-time 30 --speed-limit 100000 \
           -C - -o "$out" "$url"
      rc=$?
      set -e
    elif have wget; then
      set +e
      wget -c --tries=8 --read-timeout=30 --timeout=30 -O "$out" "$url"
      rc=$?
      set -e
    else
      die "Need curl or wget (or aria2c) to download."
    fi

    # accept curl exit 18 (partial) if size matches
    if [[ -f "$out" ]]; then
      if [[ -n "$expect" ]]; then
        local have_sz; have_sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
        log "Downloaded size: $have_sz bytes"
        if [[ "$have_sz" = "$expect" ]]; then
          log "Download verified by Content-Length."
          return 0
        fi
      else
        # no size available; accept rc==0
        [[ $rc -eq 0 ]] && return 0
      fi
    fi
    warn "Download incomplete; retrying..."
    sleep 5
    i=$((i+1))
  done
  die "Failed to download complete file after $tries attempts."
}

# ---------- preflight space checks ----------
# Rough needs: ~5GB in /tmp for the runfile + extraction, ~4GB in target dir
check_space_gb /tmp 6
check_space_gb "$(dirname "$INSTALL_DIR")" 4

# ---------- download ----------
WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT
INSTALLER="$WORKDIR/$(basename "$RUN_URL")"

log "Downloading CUDA runfile: $RUN_URL"
download_with_resume "$RUN_URL" "$INSTALLER"

chmod +x "$INSTALLER"

# ---------- install ----------
if [[ ! -d "$INSTALL_DIR" ]]; then
  log "Creating install directory: $INSTALL_DIR"
  $SUDO mkdir -p "$INSTALL_DIR"
fi

log "Running installer (toolkit only) to: $INSTALL_DIR"
$SUDO sh "$INSTALLER" \
  --silent --toolkit \
  --toolkitpath="$INSTALL_DIR" \
  --no-man-page --override --no-drm

# ---------- symlink ----------
log "Creating/updating symlink: /usr/local/cuda -> $INSTALL_DIR"
$SUDO ln -sfn "$INSTALL_DIR" /usr/local/cuda

# ---------- exports ----------
CUDA_HOME_EXPORT="/usr/local/cuda"
ENV_BLOCK=$(cat <<EOF
# >>> CUDA (${CUDA_FOLDER}) >>>
export CUDA_HOME=${CUDA_HOME_EXPORT}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export CPATH=\$CUDA_HOME/include\${CPATH:+:\$CPATH}
export LIBRARY_PATH=\$CUDA_HOME/lib64:\$CUDA_HOME/lib64/stubs\${LIBRARY_PATH:+:\$LIBRARY_PATH}
export CMAKE_PREFIX_PATH=\$CUDA_HOME\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}
# <<< CUDA (${CUDA_FOLDER}) <<<
EOF
)

if [[ "$EXPORT_BASHRC" == "1" ]]; then
  log "Appending environment exports to ~/.bashrc"
  printf "\n%s\n" "$ENV_BLOCK" >> "$HOME/.bashrc"
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
else
  warn "EXPORT_TO_BASH=0 — remember to export env in your shell:"
  echo "$ENV_BLOCK"
fi

# ---------- final checks ----------
log "Verifying nvcc"
if [[ -x "${CUDA_HOME_EXPORT}/bin/nvcc" ]]; then
  "${CUDA_HOME_EXPORT}/bin/nvcc" -V || true
else
  warn "nvcc not found at ${CUDA_HOME_EXPORT}/bin/nvcc (source ~/.bashrc or open a new shell)."
fi

log "Building & running a tiny CUDA test"
TEST_CU="$(mktemp /tmp/cuda_test.XXXXXX.cu)"
cat > "$TEST_CU" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void hello(int *out){ if(threadIdx.x==0 && blockIdx.x==0) *out = 42; }
int main(){
  int n=0; if(cudaGetDeviceCount(&n)!=cudaSuccess){std::fprintf(stderr,"No runtime.\n"); return 1;}
  if(n<1){std::fprintf(stderr,"No CUDA-capable device.\n"); return 2;}
  int *d=nullptr,h=0; cudaMalloc(&d,sizeof(int)); hello<<<1,1>>>(d);
  cudaMemcpy(&h,d,sizeof(int),cudaMemcpyDeviceToHost); cudaFree(d);
  std::printf("CUDA OK. Devices=%d, Value=%d\n", n, h);
  return (h==42)?0:3;
}
CU
OUT_BIN="/tmp/cuda_test.out"
set +e
"${CUDA_HOME_EXPORT}/bin/nvcc" -O2 "$TEST_CU" -o "$OUT_BIN" 2>/tmp/cuda_test_build.log
BUILD_RC=$?
set -e
if (( BUILD_RC != 0 )); then
  warn "nvcc compile failed; see /tmp/cuda_test_build.log"
else
  set +e; "$OUT_BIN"; RUN_RC=$?; set -e
  if (( RUN_RC == 0 )); then
    log "CUDA test ran successfully."
  else
    warn "CUDA runtime test returned code $RUN_RC (driver/GPU issue?)."
  fi
fi

log "Done. Installed ${CUDA_FOLDER} at: ${INSTALL_DIR}"
log "Symlink: /usr/local/cuda -> ${INSTALL_DIR}"