#!/usr/bin/env bash
# cuda_local_install.sh
# Syntax: cuda_install CUDA_VERSION INSTALL_PREFIX EXPORT_TO_BASH
#   CUDA_VERSION in {110,111,112,113,114,115,116,117,118,120,121,122,123,124,125,126,128}
#   INSTALL_PREFIX default: /usr/local
#   EXPORT_TO_BASH in {0,1}; default: 1 (append to ~/.bashrc)
#
# Example:
#   chmod +x cuda_local_install.sh
#   ./cuda_local_install.sh 126 ~/nvidia 1
#   source ~/.bashrc

set -euo pipefail

############### Logging helpers ###############
log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

trap 'err "Installation failed (line $LINENO). Check the logs above."' ERR

############### Inputs & defaults ###############
CUDA_VERSION="${1:-}"
INSTALL_PREFIX="${2:-/usr/local}"
EXPORT_BASHRC="${3:-1}"

if [[ -z "${CUDA_VERSION}" ]]; then
  cat >&2 <<'USAGE'
Usage: ./cuda_local_install.sh CUDA_VERSION [INSTALL_PREFIX] [EXPORT_TO_BASH]
  CUDA_VERSION: one of {110,111,112,113,114,115,116,117,118,120,121,122,123,124,125,126,128}
  INSTALL_PREFIX: default /usr/local
  EXPORT_TO_BASH: 1 to append to ~/.bashrc (default), 0 to skip

Examples:
  ./cuda_local_install.sh 126                    # installs to /usr/local/cuda-12.6 and exports paths
  ./cuda_local_install.sh 126 ~/nvidia 1         # installs to ~/nvidia/cuda-12.6 and exports paths
  ./cuda_local_install.sh 118 /opt/nvidia 0      # installs to /opt/nvidia/cuda-11.8, do not edit .bashrc
USAGE
  exit 2
fi

############### URLs (x86_64 Linux runfiles) ###############
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

[[ -n "${URLS[$CUDA_VERSION]:-}" ]] || die "Unsupported CUDA_VERSION '$CUDA_VERSION'."

RUN_URL="${URLS[$CUDA_VERSION]}"
CUDA_FOLDER="${FOLDERS[$CUDA_VERSION]}"
INSTALL_DIR="${INSTALL_PREFIX%/}/${CUDA_FOLDER}"

############### Basic environment checks ###############
OS="$(uname -s)"
ARCH="$(uname -m)"
[[ "$OS" == "Linux" ]] || die "This installer supports Linux only (got: $OS)."
[[ "$ARCH" == "x86_64" ]] || warn "Non-x86_64 architecture detected ($ARCH). Ensure the runfile matches your arch."

if grep -qi microsoft /proc/version 2>/dev/null; then
  warn "WSL detected. CUDA requires NVIDIA GPU passthrough from Windows. Ensure your WSL/NVIDIA setup supports CUDA."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
  log "NVIDIA driver detected: $DRIVER_VER"
else
  warn "No 'nvidia-smi' found. Driver may be absent or not in PATH. Toolkit can install, but runtime will need an NVIDIA driver."
fi

# downloader
DL=""
if command -v curl >/dev/null 2>&1; then DL="curl -L -o"; fi
if command -v wget >/dev/null 2>&1; then DL="${DL:-wget -O}"; fi
[[ -n "$DL" ]] || die "Need 'curl' or 'wget' to download the installer."

# sudo helper (only if lacking write perms)
SUDO=""
if [[ ! -w "$(dirname "$INSTALL_PREFIX")" ]] || [[ "$INSTALL_PREFIX" == "/usr/local" && $EUID -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else
    [[ $EUID -eq 0 ]] || die "Need root privileges for installing to '$INSTALL_PREFIX'. Re-run with sudo."
  fi
fi

############### Download & Install ###############
WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

INSTALLER="$WORKDIR/$(basename "$RUN_URL")"
log "Downloading CUDA runfile: $RUN_URL"
$DL "$INSTALLER" "$RUN_URL"

# Create install dir
if [[ ! -d "$INSTALL_DIR" ]]; then
  log "Creating install directory: $INSTALL_DIR"
  $SUDO mkdir -p "$INSTALL_DIR"
fi

log "Running installer (toolkit only) to: $INSTALL_DIR"
# Notes:
#  --toolkit          : install only the CUDA Toolkit
#  --toolkitpath=PATH : install path
#  --silent           : non-interactive
#  --no-man-page      : skip man pages (smaller)
#  --override         : overwrite if needed
#  --no-drm           : skip DRM
$SUDO sh "$INSTALLER" \
  --silent --toolkit \
  --toolkitpath="$INSTALL_DIR" \
  --no-man-page --override --no-drm

############### Symlink /usr/local/cuda ###############
# Always point /usr/local/cuda -> INSTALL_DIR (even if INSTALL_PREFIX != /usr/local)
log "Creating/updating symlink: /usr/local/cuda -> $INSTALL_DIR"
if [[ $EUID -ne 0 ]]; then
  $SUDO ln -sfn "$INSTALL_DIR" /usr/local/cuda
else
  ln -sfn "$INSTALL_DIR" /usr/local/cuda
fi

############### Exports ###############
CUDA_HOME_EXPORT="/usr/local/cuda"  # use the symlink for convenience
ENV_LINES=$(cat <<EOF
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

if [[ "${EXPORT_BASHRC}" == "1" ]]; then
  log "Appending environment exports to ~/.bashrc"
  {
    echo
    echo "$ENV_LINES"
  } >> "$HOME/.bashrc"
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
else
  warn "EXPORT_TO_BASH=0; remember to set your env (example below):"
  echo
  echo "$ENV_LINES"
  echo
fi

############### Final checks ###############
log "Verifying nvcc"
NVCC_BIN="${CUDA_HOME_EXPORT}/bin/nvcc"
if [[ -x "$NVCC_BIN" ]]; then
  "$NVCC_BIN" -V || true
else
  warn "nvcc not found at $NVCC_BIN (PATH may not be refreshed in this shell). Try: source ~/.bashrc"
fi

log "Compiling a minimal CUDA test (this verifies toolkit + driver runtime)"
TEST_CU="$(mktemp /tmp/cuda_test.XXXXXX.cu)"
cat > "$TEST_CU" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void hello(int *out){ if(threadIdx.x==0 && blockIdx.x==0) *out = 42; }
int main(){
  int devCount=0; cudaError_t e = cudaGetDeviceCount(&devCount);
  if(e != cudaSuccess){ std::fprintf(stderr,"cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e)); return 1; }
  if(devCount < 1){ std::fprintf(stderr,"No CUDA-capable device found.\n"); return 2; }
  int *d=nullptr, h=0; cudaMalloc(&d, sizeof(int));
  hello<<<1,1>>>(d);
  cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d);
  std::printf("CUDA OK. DeviceCount=%d, KernelValue=%d\n", devCount, h);
  return (h==42)?0:3;
}
CU

OUT_BIN="/tmp/cuda_test.out"
set +e
"$NVCC_BIN" -O2 "$TEST_CU" -o "$OUT_BIN" 2>/tmp/cuda_test_build.log
BUILD_RC=$?
set -e
if [[ $BUILD_RC -ne 0 ]]; then
  warn "nvcc compile failed. See /tmp/cuda_test_build.log"
else
  set +e
  "$OUT_BIN"
  RUN_RC=$?
  set -e
  if [[ $RUN_RC -eq 0 ]]; then
    log "CUDA test ran successfully."
  else
    warn "CUDA runtime test returned code $RUN_RC (driver/GPU availability issue?)."
  fi
fi

log "Done. Installed ${CUDA_FOLDER} at: ${INSTALL_DIR}"
log "Symlink: /usr/local/cuda -> ${INSTALL_DIR}"
log "Open a NEW shell or 'source ~/.bashrc' to ensure your environment is updated."