#!/usr/bin/env bash
# Compile CUDA extension packages on Windows without a manual "Developer Command Prompt".
#
#   bash scripts/build_extensions.sh gsplat            # vendored 3dgs/ (gsplat)
#   bash scripts/build_extensions.sh 2dsugar           # diff-surfel-rasterization, simple-knn, ...
#   bash scripts/build_extensions.sh all
#
# torch.utils.cpp_extension needs cl.exe + nvcc on PATH. vcvars64.bat provides cl.exe, but it
# only works inside cmd.exe, so we generate a throwaway batch file and run it with cmd //c.
# TORCH_CUDA_ARCH_LIST is pinned to this machine's GPU (default 8.9 = RTX 40-series laptop);
# building every architecture takes 10x longer for no benefit.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
WIN_ROOT="$(pwd -W 2>/dev/null || cygpath -w "$ROOT" 2>/dev/null || echo "$ROOT")"

PYTHON="${PYTHON:-$ROOT/.venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"
win() { cygpath -w "$1" 2>/dev/null || echo "$1"; }
PYTHON_WIN="$(win "$PYTHON")"
ROOT_WIN="$(win "$ROOT")"
ARCH="${TORCH_CUDA_ARCH_LIST:-8.9}"
BUILD_NO_CUDA="${BUILD_NO_CUDA:-0}"
JOBS="${JOBS:-}"

find_vcvars() {
  local base candidate
  for base in "/c/Program Files (x86)/Microsoft Visual Studio" "/c/Program Files/Microsoft Visual Studio"; do
    for candidate in \
      "$base/18/BuildTools/VC/Auxiliary/Build/vcvars64.bat" \
      "$base/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat" \
      "$base/2022/Community/VC/Auxiliary/Build/vcvars64.bat" \
      "$base/2022/Professional/VC/Auxiliary/Build/vcvars64.bat" \
      "$base/2022/Enterprise/VC/Auxiliary/Build/vcvars64.bat"; do
      [ -f "$candidate" ] && { cygpath -w "$candidate" 2>/dev/null || echo "$candidate"; return 0; }
    done
  done
  return 1
}

VCCALL="$(find_vcvars || true)"
if [ -z "$VCCALL" ]; then
  echo "vcvars64.bat not found. Install Visual Studio Build Tools with the 'C++ build tools' workload." >&2
  exit 1
fi
echo "[msvc] $VCCALL"
echo "[arch] TORCH_CUDA_ARCH_LIST=$ARCH"
echo "[python] $PYTHON"
command -v nvcc >/dev/null && echo "[nvcc] $(nvcc --version | grep -o 'release [0-9.]*')"

run_batch() {  # run_batch <label> <pip args...>
  local label="$1"; shift
  local bat="$ROOT/data/.cache/build/${label}.bat"
  mkdir -p "$(dirname "$bat")"
  {
    echo '@echo off'
    echo "call \"$VCCALL\""
    # torch.utils.cpp_extension._check_abi() refuses a pre-activated VC env unless this is set,
    # otherwise it would activate the toolchain a second time.
    echo "set DISTUTILS_USE_SDK=1"
    echo "set TORCH_CUDA_ARCH_LIST=$ARCH"
    echo "set BUILD_NO_CUDA=$BUILD_NO_CUDA"
    if [ -n "$JOBS" ]; then echo "set MAX_JOBS=$JOBS"; fi
    echo "cd /d \"$ROOT_WIN\""
    echo "\"$PYTHON_WIN\" $*"
    echo "exit /b %ERRORLEVEL%"
  } > "$bat"
  echo "[build] $label: $*"
  cmd.exe //c "$(win "$bat")"
  local rc=$?
  echo "[result] $label rc=$rc"
  return $rc
}

target="${1:-all}"
case "$target" in
  gsplat)
    run_batch gsplat "-m pip install --no-build-isolation -v \"./3dgs\""
    ;;
  2dsugar)
    # 2D-SuGaR needs three extensions; each is a separate pip build.
    run_batch diff-surfel "-m pip install --no-build-isolation -v \"./2dsugar/gaussian_splatting_2d/submodules/diff-surfel-rasterization\""
    run_batch simple-knn  "-m pip install --no-build-isolation -v \"./2dsugar/gaussian_splatting_2d/submodules/simple-knn\""
    run_batch diff-gauss  "-m pip install --no-build-isolation -v \"./2dsugar/gaussian_splatting/submodules/diff-gaussian-rasterization\""
    ;;
  all)
    bash "$0" gsplat && bash "$0" 2dsugar
    ;;
  *)
    echo "usage: build_extensions.sh [gsplat|2dsugar|all]" >&2; exit 2
    ;;
esac
