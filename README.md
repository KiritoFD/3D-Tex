# 3D-Tex

Minimal **capture -> VGGT cameras/points -> (3DGS | 2D-SuGaR -> Mesh/UV)** research baseline.
This is a *per-scene optimization* testbed; **no generic reconstruction model is trained**.
All upstream implementation directories are peers at the repository root and kept unmodified:

```text
.
├── vggt/                       # facebookresearch/vggt (Git submodule)
├── 3dgs/                       # nerfstudio-project/gsplat (Git submodule)
├── 2dsugar/                    # prajwalcr/2d-sugar (Git submodule)
├── scripts/                    # small wrappers; no model fork
├── configs/upstreams.json      # upstream URLs and exact pinned Git SHAs
├── tests/
├── docs/
├── data/scenes/<scene>/
│   ├── images/                 # images shared by both branches
│   ├── sparse/                 # VGGT COLMAP export
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   ├── points3D.bin
│   │   └── 0/                  # links/copies of the same 3 files for 2D-SuGaR
│   └── sample_manifest.json
└── outputs/
    ├── logs/
    ├── 3dgs/<scene>/
    └── 2dsugar/<scene>/{model,mesh}/
```

## Requirements

- Linux + NVIDIA CUDA GPU (the scripts default to `TORCH_CUDA_ARCH_LIST=8.9` for RTX 4090).
- Conda, Git, Git LFS if needed, GPU driver, and an available CUDA compiler (`nvcc`) for extension builds.
- Enough RAM/disk for source repos, pretrained VGGT/VGGSfM weights, and rendered scenes.
- **Three isolated Conda environments:** VGGT pins `torch==2.3.1`; current gsplat example pins `torch==2.9.1`; 2D-SuGaR has an older Python 3.9 / CUDA 11.8 environment. Do **not** try to merge environments in phase 1.

See `docs/COMPATIBILITY.md` before installing. First environment setup downloads pretrained weights when first needed. Source and model licenses remain upstream.

## One-time setup

```bash
# If cloned from a Git repository or from the provided .bundle:
bash scripts/bootstrap.sh

# If extracted from the provided .zip, the same bootstrap script clones upstreams.

# Install each environment separately (installation may take a while):
bash scripts/install_envs.sh vggt
bash scripts/install_envs.sh 3dgs
bash scripts/install_envs.sh 2dsugar

python3 scripts/doctor.py
```

To prefetch the VGGT weights (otherwise the first reconstruction downloads them):

```bash
bash scripts/prefetch_weights.sh vggt
# Optional tracker weights for --ba:
bash scripts/prefetch_weights.sh tracker
```

`--ba` uses the VGGSfM tracker and may download additional pretrained weights/DINO features on first use.

## First smoke test (no external dataset download)

Use the official VGGT sample images bundled in the `vggt` source repository:

```bash
# Small samples: fern (20 images), flower (25), kitchen (25).
python3 scripts/get_samples.py --scenes fern flower

# Phase 1: predict a common COLMAP scene from fern RGB images.
python3 scripts/run_vggt.py --scene fern --ba

# Phase 2A: train a 3D Gaussian model with gsplat.
python3 scripts/run_3dgs.py --scene fern --steps 7000

# Phase 2B: 2D Gaussian + SuGaR mesh refinement + UV export.
# Default mode omits the unrelated Metric3D prior in order to test
# the same VGGT cameras and point cloud used in Phase 2A.
python3 scripts/run_2dsugar.py --scene fern --iterations 7000 --refine-iterations 2000
```

Or: `bash scripts/run_all.sh fern`.

For a full 2D-SuGaR comparison, run the upstream depth/normal prior pathway separately:

```bash
python3 scripts/run_2dsugar.py --scene fern --priors metric3d \
  --iterations 30000 --refine-iterations 15000
```

This downloads a large Metric3D model, creates additional depth/normal `.npy` maps and uses VGGT COLMAP tracks. The upstream helper processes **PNG files only** and makes intrinsic/scale assumptions; inspect quality before treating this as a valid geometry improvement. The no-prior baseline remains the cleaner VGGT integration comparison.

**Smoke test** means the workflow executes and produces models, not that a 7k-step result is a converged benchmark. Use a longer schedule for quality measurements. High-quality mesh extraction also requires a sufficiently dense, well-constrained scene.

## Commands for your own capture

```bash
mkdir -p data/scenes/object01/images
# Copy 20-100 undistorted overlapping RGB images into this folder.
python3 scripts/run_vggt.py --scene object01 --ba
python3 scripts/run_3dgs.py --scene object01 --steps 30000
python3 scripts/run_2dsugar.py --scene object01 --iterations 30000 --refine-iterations 15000
```

For all wrappers use `--help`; use `--dry-run` to inspect generated upstream commands. Run one GPU job at a time on a single 4090.

## Critical evaluation caveat

These small VGGT examples **do not contain ground-truth geometry or independent test camera poses**. Both branches share the same VGGT-estimated cameras and points. gsplat and 2D-SuGaR can hold out every 8th image from their own optimization, but **VGGT itself has already seen all images**. Therefore this is a *pipeline integration smoke test*, not a leak-free novel-view benchmark.

For actual geometry accuracy and held-out rendering evaluation, see `docs/EVALUATION.md`; prepare DTU official calibration / GT geometry separately. Never call a pipeline smoke-test PSNR a clean benchmark.

## Reproducibility and repository publishing

Upstream Git SHAs are pinned in both Git submodule gitlinks and `configs/upstreams.json`.
A `git clone --recurse-submodules` of a published repository recovers all three upstreams and their nested dependencies. Local data, weight caches, and outputs are ignored by Git.

To create a new **private** GitHub repository after cloning the downloadable Git bundle:

```bash
# Requires authenticated GitHub CLI (gh auth login).
bash scripts/publish_github.sh
```

The repository has not been uploaded automatically; creation via the connected GitHub tool was unavailable in this session.

## Upstream references

- VGGT: https://github.com/facebookresearch/vggt
- gsplat: https://github.com/nerfstudio-project/gsplat
- 2D-SuGaR: https://github.com/prajwalcr/2d-sugar

This wrapper contains only integration scripts and metadata; upstream files are downloaded from their original repositories and retain their respective licenses.