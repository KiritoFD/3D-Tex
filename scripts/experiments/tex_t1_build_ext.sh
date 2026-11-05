#!/bin/bash
# Tex-T1: 编译 SUPRA-Tex 运行时 CUDA 扩展（在基础环境装好之后运行）
set -uo pipefail
PY=/opt/conda/envs/supra-tex/bin/python
EXT=/root/3dtex/third_party/_ext
LOG=/root/3dtex/outputs/tex_t1_supra_20261105/logs/build_ext.log
export CUDA_HOME=/opt/conda/envs/nvcc124
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.9"
mkdir -p "$(dirname "$LOG")"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "=== nvdiffrast ==="
"$PY" -m pip install --no-build-isolation --no-deps "$EXT/nvdiffrast" >> "$LOG" 2>&1 || say "nvdiffrast FAILED"

say "=== CuMesh (+cubvh) ==="
"$PY" -m pip install --no-build-isolation --no-deps "$EXT/CuMesh" >> "$LOG" 2>&1 || say "CuMesh FAILED"

say "=== FlexGEMM ==="
"$PY" -m pip install --no-build-isolation --no-deps "$EXT/FlexGEMM" >> "$LOG" 2>&1 || say "FlexGEMM FAILED"

say "=== o-voxel (TRELLIS.2) ==="
"$PY" -m pip install --no-build-isolation --no-deps "$EXT/TRELLIS.2/o-voxel" >> "$LOG" 2>&1 || say "o-voxel FAILED"

say "=== eval deps (scikit-image / lpips) ==="
"$PY" -m pip install scikit-image lpips >> "$LOG" 2>&1 || say "eval deps FAILED"

say "=== verify ==="
"$PY" - <<'PY' >> "$LOG" 2>&1
import torch
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
import nvdiffrast.torch as dr
print("nvdiffrast ok")
import cumesh; print("cumesh ok")
import flex_gemm; print("flex_gemm ok")
import o_voxel; print("o_voxel ok")
import lpips, skimage; print("lpips/skimage ok")
PY
say "build stage done"
