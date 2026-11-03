# WSL2 render environment (accelerated differentiable rasterization)

Windows has no working wheels for nvdiffrast / pytorch3d on this machine, so the
acceleration path for Phase B (M2/M3, 512^2 full-view) lives in WSL2. The pure
torch bucket rasterizer (`scripts/m2_anchor_opt.py --renderer bucket`) remains
the Windows fallback and correctness baseline. Metrics are only comparable
within the same renderer + resolution; cross-renderer comparisons are labelled
as such wherever they appear.

## Host / platform

| item | value |
|---|---|
| WSL distro | `ext4` (Ubuntu 24.04.3), WSL2, mirrored networking |
| GPU | RTX 4070 Laptop 8 GB (`nvidia-smi` works inside WSL) |
| Windows driver | 617.42 |
| CUDA toolkit (WSL) | 12.8 at `/usr/local/cuda` (`nvcc` V12.8.93) |
| Python | 3.12.3 (system), venv at `~/venvs/render` |
| Repo path | `/mnt/g/GitHub/3D-Tex` (mounted; do not restructure) |
| Proxy | host proxy `http://127.0.0.1:7897` (required for git/pip over the network; unstable for >1 GB downloads) |

EGL/OpenGL is **not** needed: nvdiffrast runs on `RasterizeCudaContext`
(headless CUDA), no `RasterizeGLContext`, no display.

## Verified library versions

| package | version | install route | status |
|---|---|---|---|
| torch | 2.14.1+cu126 | Windows-side `pip download` then local install (below) | import + `cuda.is_available()` OK |
| nvdiffrast | 0.4.0 | source build (`~/nvdiffrast`) | import + rasterize/interpolate/antialias OK |
| pytorch3d | 0.7.9 | source build with `-std=c++20` patch (below) | import + `knn_points` CUDA + `rasterize_meshes` OK |
| scikit-image | 0.26.0 | pip via proxy (needed by `scripts/tsdf_mesh.py`) | OK |
| triton | 3.8.0 | wheel dependency of torch | OK |

## Reproduction (one page)

```bash
# 0. WSL prerequisites: CUDA 12.8 toolkit at /usr/local/cuda, python3.12-venv,
#    build-essential, ninja-build, git. Proxy for network access:
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
git config --global http.proxy http://127.0.0.1:7897

# 1. venv
python3 -m venv ~/venvs/render && source ~/venvs/render/bin/activate

# 2. torch 2.14.1+cu126 — DO NOT pip-install over the WSL network directly
#    (2.5 GB download dies repeatedly). Download on the WINDOWS side instead,
#    from a CLEAN throwaway venv (an activated venv with torch already present
#    makes the resolver silently skip the nvidia-* wheels):
#      python -m venv outputs\piptmp && outputs\piptmp\Scripts\activate
#      pip download torch --platform manylinux_2_28_x86_64 --python-version 312 \
#        --implementation cp --abi cp312 --only-binary=:all: \
#        --index-url https://download.pytorch.org/whl/cu126 -d outputs\wsl_wheels
#      pip download cuda-bindings==12.9.4 nvidia-cufile-cu12==1.11.1.6 (same flags)
#    then in WSL:
pip install --no-index --find-links /mnt/g/GitHub/3D-Tex/outputs/wsl_wheels torch

# 3. nvdiffrast 0.4.0 (build isolation must be OFF: setup needs torch)
git clone https://github.com/NVlabs/nvdiffrast ~/nvdiffrast   # not /tmp (wiped on WSL restart)
cd ~/nvdiffrast && CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  pip install . --no-build-isolation

# 4. pytorch3d 0.7.9 — patch C++ standard first: torch 2.14 ATen headers
#    require C++20, but setup.py hardcodes -std=c++17
git clone https://github.com/facebookresearch/pytorch3d ~/pytorch3d
sed -i 's/-std=c++17/-std=c++20/g' ~/pytorch3d/setup.py
cd ~/pytorch3d && CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  pip install . --no-build-isolation

# 5. misc
pip install scikit-image   # tsdf_mesh.write_ply
```

## Failure log (what broke, precisely)

- `pip install torch` in WSL: repeated timeouts on the 2.5 GB wheel even via
  proxy → Windows-side download + `--no-index` install. A killed install left a
  truncated `site-packages` (`import torch` → `Bus error (signal 7)`); fixed by
  deleting the venv and reinstalling from the verified local wheels
  (`zipfile.testzip` clean).
- `pip download` from the activated Windows `.venv` (torch 2.13 already
  installed): resolver reported "Requirement already satisfied" and skipped the
  `nvidia-*`/`triton` wheels → use a clean throwaway venv for downloading.
- pytorch3d attempt 1: build died with access violation `-1073741571`
  (parallel compile). attempt 2 (`MAX_JOBS=1`):
  `ATen.h:5:2: error: #error C++20 or later compatible compiler is required to
  use ATen.` — setup.py hardcodes `-std=c++17` while gcc is 14.3 and torch
  2.14 requires C++20. Fixed by the sed patch above; attempt 3 succeeded.
- nvdiffrast: default pip build isolation fails (`ModuleNotFoundError: torch`
  during setup) → `--no-build-isolation`.

## Renderer validation (validate_nvd.py)

`outputs/wsl_wheels/validate_nvd.py` renders the M2 baseline mesh
(`carve_v35c30`, ~400k faces) at the aligned anchor16 cameras with both
renderers and compares depth/alpha/vertex-gradients, plus speed and peak VRAM.
Key facts established during bring-up:

- nvdiffrast NDC has **y pointing down** (image convention, not GL): the
  correct transform is `x_ndc = 2fx/W·x/z + (2cx/W − 1)`,
  `y_ndc = 2fy/H·y/z + (2cy/H − 1)`, `w = z`. Verified against the GT
  foreground mask: coverage 0.2372 vs GT 0.2372, IoU 0.947 (GL-style y-flip
  drops IoU to 0.45).
- The bucket rasterizer caps candidates per pixel (`max_cand`); with
  index-ordered truncation it drops whole spatial regions (alpha IoU vs GT
  fell to 0.49). `build_buckets` now orders each tile's candidates
  nearest-first and sends >8-tile-span triangles to a global list (also fixes
  the MemoryError from sliver triangles). Even so, the bucket renderer is a
  fidelity-limited fallback; nvdiffrast is the mainline.
- Results JSON: `outputs/wsl_wheels/validate_nvd.json` (see GEOMETRY_OPTIMIZATION
  Phase B / M2 for the numbers in context).

## Validation numbers (2026-11-03, carve_v35c30 mesh, aligned anchor16 view 0)

| check | result |
|---|---|
| depth agreement (bucket vs nvd, shared fg px, 256^2) | L1 0.049, median rel **1.13e-4** |
| alpha vs GT fg (IoU) | nvd **0.947** / bucket 0.49 (max_cand truncation) |
| grad sign agreement (bucket vs nvd) | 0.55 — expected: bucket smears grad over candidates, nvd concentrates on the covering tri; FD check is the gate |
| finite-difference check (nvd, depth+silhouette loss, 8 vertices × 3 axes, eps 1e-4) | **23/24 sign agreement** (same level as the M1 bucket check 24/24) |
| speed fwd+bwd (nvd, full frame) | **4.6 ms @256^2, 6.4 ms @512^2** (peak 0.17/0.19 GiB) |
| speed fwd+bwd (bucket, 8192 px @256^2) | 394 ms (full frame ~8×) → nvd is ~700× faster at equal pixels |
