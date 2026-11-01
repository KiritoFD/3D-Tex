# Architecture and source/status contract (2026-11-01)

## What we WILL build

`train RGB → VGGT-Ω K/T/depth/confidence → robust multi-view depth fusion → TSDF Mesh → mesh optimization (adapted ExMesh or GeoSVR) → UV atlas → observed texture bake → seam optimization → optional generated/PBR missing-material maps`.

Full design and acceptance criteria: **[MESH_UV_PIPELINE.md](MESH_UV_PIPELINE.md)**. State of the art: **[SOTA_2026.md](SOTA_2026.md)**.

## What is PRESENT in the repo

| Component | Current reality | Operational status |
|---|---|---|
| `scripts/download_eth3d.py` | ETH3D data, calibration and strictly isolated heldout views | **CPU-tested, real GitHub Actions download previously passed** |
| `vggt/`, `scripts/run_vggt.py` | **Original VGGT** COLMAP exporter and wrapper | CPU CLI/shape tests only; actual CUDA inference not verified |
| `2dsugar/`, `scripts/run_2dsugar.py` | 2DGS + SuGaR legacy comparison; uses CUDA Gaussian extensions | **Historical/optional, not in no-Gaussian mainline** |
| `3dgs/`, `scripts/run_3dgs.py` | **Removed** by architecture decision | Not part of build or pipeline |
| `vggt_omega/` | not vendored | Pending checkpoint/license and adapter |
| Depth/pose adapter + TSDF fusion | not implemented | Pending |
| Mesh optimizer integration | not implemented | Pending |
| UV atlas & camera-weighted texture baking | not implemented | Pending |
| Heldout RGB render + GT mesh metrics | not implemented | Pending |

## Geometry and evaluation contracts

- No intermediate Gaussian train or gaussians as final representation.
- Only train views go to inference and refinement. ETH3D `heldout/images` and `ground_truth` are never model inputs.
- The model-to-original-image intrinsics transform, world-to-camera pose convention, scale ambiguity, and depth definition must be explicit. No silent use of old VGGT point-map outputs with new Ω API.
- A0 = robust depth/TSDF baseline, A1 = adapted ExMesh (replace author PGSR init), A2 = GeoSVR sparse voxels; compare only when evaluation protocol matches.
- Output is explicit `vertices/faces/face-corner UV/texels + observed/generated masks`; missing surfaces must remain identifiable.
- New pipeline **must pass GPU/CPU output acceptance** before documenting a working end-to-end command.

## Legacy entry points

`scripts/run_vggt.py` and `scripts/run_2dsugar.py` are old experiments only. `scripts/run_all.sh` exits nonzero rather than silently invoking old Gaussian stages. Old source and prior files remain in Git history for reproducibility.
