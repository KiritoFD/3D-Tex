# 3D-Tex

**Research goal: multi-view RGB → VGGT-Ω cameras/depth → surface Mesh → UV atlas → faithful textures (optional PBR). No 3D Gaussian stage.**

> **Status:** Architecture and literature survey updated 2026-11-01. The **new mesh/UV end-to-end implementation is NOT present yet**. The current `vggt/` is the pinned **original VGGT** (not VGGT-Ω); the `2dsugar/` source is a retained, optional historical reference, **not used in the new main pipeline**. The old standalone `3dgs/` vendored code and runner have been removed.

## Technical design & papers

- **[Mesh + UV implementation plan](docs/MESH_UV_PIPELINE.md)** — all stage contracts; robust depth fusion / TSDF / topology optimization / xatlas / visibility-aware texture baking / seam loss, tests and GPU gates.
- **[2025–2026 SOTA comparison](docs/SOTA_2026.md)** — verified authors, institutions, venues, geometric/rendering/texturing results, code status, honest limitations.
- **[VGGT-Ω adapter](docs/VGGT_OMEGA_ADAPTER.md)** — model API, lack of point-map head, K/pose/depth alignment, checkpoint version and RAM/VRAM budgets.
- **[No-Gaussian decision](docs/DECISION_NO_GAUSSIANS.md)** — why and what was removed.
- **[Direct UV VAE roadmap](docs/DIRECT_UV.md)** — follows a stable UV carrier.
- **[Paper index and PDF downloader](docs/papers/README.md)**.

### Proposed stages (not currently runnable)

```text
RGB train images
  └─ VGGT-Ω [NOT YET PORTED] ──> intrinsics/extrinsics + depth + confidence
        └─ multiview consistency + TSDF [PLANNED]
              ├─ baseline mesh -> xatlas -> texture baking [PLANNED]
              ├─ adapted ExMesh mesh/UV optimizer [PLANNED, not upstream default]
              └─ GeoSVR sparse voxel geometry baseline [PLANNED]
                                     └─ fixed Mesh + UV -> texture blending/repair/PBR [PLANNED]
```

No part of the above requires 3DGS or 2DGS, but the ExMesh **official** upstream implementation obtains its initial mesh from PGSR; replacing this with our VGGT-Ω/TSDF mesh is an unverified integration experiment.

## What is actually available now

```text
vggt/           pinned original VGGT inference + COLMAP exporter
2dsugar/        legacy optional 2DGS/SuGaR research source (not mainline)
scripts/        ETH3D downloader, current VGGT wrapper, paper downloader
tests/          CPU dataset, HTTP, and original-VGGT/COLMAP wiring regression tests
docs/           architecture, method review, implementation roadmap
data/           Git-ignored input data
outputs/        Git-ignored outputs
```

### CPU test/data preparation

```bash
git clone https://github.com/KiritoFD/3D-Tex.git && cd 3D-Tex
python3 -m unittest discover -s tests -v
python3 scripts/doctor.py
python3 scripts/download_eth3d.py --dry-run
python3 scripts/download_papers.py --list

# optional real ETH3D images + scans (p7zip required, no GPU)
python3 scripts/download_eth3d.py --scenes pipes
```

The official ETH3D three-scene image/laser download and extraction was previously verified on GitHub Actions (~1.254 GiB compressed / ~1.901 GiB extracted). Those data checks do **not** establish a trained VGGT-Ω or Mesh/UV reconstruction pipeline.

The original `scripts/run_vggt.py` still invokes original VGGT (CUDA expected). **`scripts/run_all.sh` intentionally refuses to run** until the newly specified no-Gaussian integration is implemented. Do not publish synthetic wrapper results as geometric/texture performance.

See [docs/DATASETS.md](docs/DATASETS.md), [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md), [docs/UPSTREAMS.md](docs/UPSTREAMS.md) for data/license/dependency boundaries.
