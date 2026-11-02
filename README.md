# 3D-Tex

**Research goal: multi-view RGB → VGGT-Ω cameras/depth → surface Mesh → UV atlas → faithful textures (optional PBR). No 3D Gaussian stage.**

> **Status (2026-11-02):** the A0 path **now runs end to end** on ETH3D `kicker`:
> `VGGT-Ω → consistency filter → TSDF → mesh.ply → ExMesh → OBJ/MTL/albedo/GLB`.
> [QUALITY_ATTRIBUTION.md](docs/QUALITY_ATTRIBUTION.md) records what is verified against
> ETH3D ground truth (Ω poses 0.59% / 0.57°, focals −0.25%) and what is still weak
> (12 of 25 views fit in 8 GiB, open boundaries, ExMesh only smoke-tested at 200 iters).
> `vggt_omega/` is a vendored source mirror of VGGT-Ω; the local `vggt/` remains the pinned
> **original VGGT** (used only for its COLMAP exporter/legacy baseline). `2dsugar/` is a
> retained optional historical reference; `3dgs/` is removed.

## Technical design & papers

- **[Mesh + UV implementation plan](docs/MESH_UV_PIPELINE.md)** — all stage contracts; robust depth fusion / TSDF / topology optimization / xatlas / visibility-aware texture baking / seam loss, tests and GPU gates.
- **[2025–2026 SOTA comparison](docs/SOTA_2026.md)** — verified authors, institutions, venues, geometric/rendering/texturing results, code status, honest limitations.
- **[VGGT-Ω adapter](docs/VGGT_OMEGA_ADAPTER.md)** — model API, lack of point-map head, K/pose/depth alignment, checkpoint version and RAM/VRAM budgets.
- **[No-Gaussian decision](docs/DECISION_NO_GAUSSIANS.md)** — why and what was removed.
- **[Direct UV VAE roadmap](docs/DIRECT_UV.md)** — follows a stable UV carrier.
- **[Paper index and PDF downloader](docs/papers/README.md)**.

### Stages (status as of 2026-11-02)

```text
RGB train images
  └─ VGGT-Ω [DONE] ──> depths + confidences + K/T + pixel affine (camera z-depth)
        └─ multiview consistency + TSDF [DONE]  -> mesh_init.ply (surface, not solid)
              ├─ baseline mesh -> xatlas -> texture baking [DONE, superseded by ExMesh]
              └─ adapted ExMesh mesh/UV optimizer [DONE, 200-iter smoke only]
                                     └─ fixed Mesh + UV -> texture blending/repair/PBR [PLANNED]
```

`GeoSVR` remains a planned geometry baseline. None of the above requires 3DGS or 2DGS.
The ExMesh **official** stack initialises from PGSR; replacing that with our VGGT-Ω/TSDF mesh
is the integration this repository now performs, and it is validated only as a smoke run —
see [QUALITY_ATTRIBUTION.md](docs/QUALITY_ATTRIBUTION.md) §4 for the limits.

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
