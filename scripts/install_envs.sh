#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-all}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
command -v conda >/dev/null || exit 1
mk() { conda run -n "$1" python --version >/dev/null 2>&1 || conda create -y -n "$1" "python=$2"; }
vggt() { mk recon-vggt 3.10; conda run -n recon-vggt python -m pip install -r "$ROOT/vggt/requirements.txt" -r "$ROOT/vggt/requirements_demo.txt"; }
gsplat() { mk recon-3dgs 3.11; conda run -n recon-3dgs python -m pip install torch==2.9.1 torchvision==0.24.1; BUILD_EXPERIMENTAL=0 conda run -n recon-3dgs python -m pip install -e "$ROOT/3dgs" --no-build-isolation; conda run -n recon-3dgs python -m pip install -r "$ROOT/3dgs/examples/requirements.txt"; }
sugar() { if ! conda run -n 2d-sugar python --version >/dev/null 2>&1; then conda env create -f "$ROOT/2dsugar/environment.yml"; fi; for path in gaussian_splatting_2d/submodules/diff-surfel-rasterization gaussian_splatting_2d/submodules/simple-knn gaussian_splatting/submodules/diff-gaussian-rasterization; do conda run -n 2d-sugar python -m pip install -e "$ROOT/2dsugar/$path" --no-build-isolation; done; }
case "$TASK" in vggt) vggt;; 3dgs) gsplat;; 2dsugar) sugar;; all) vggt;gsplat;sugar;; *) exit 2;; esac
