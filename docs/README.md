# 3D-Tex research and implementation notes

**Active target: no-Gaussian Mesh + UV from VGGT-Ω. The A0 path (Ω geometry → TSDF → mesh → ExMesh → OBJ/GLB) now runs end to end on ETH3D `kicker`; see [QUALITY_ATTRIBUTION.md](QUALITY_ATTRIBUTION.md) for what is verified and what is still weak.**

Latest experiment: [VIEW_SELECTION.md](experiments/VIEW_SELECTION.md) separates
input interference from fusion degradation, reports foreign-view failure, and
records the LOO precision mismatch plus corrected rerun protocol.

0. **[P0_GEOMETRY_OPTIMIZATION.md](experiments/P0_GEOMETRY_OPTIMIZATION.md)** — pose/depth isolation, PGSR 10k negative result, voxel-scale filtering and untouched test-view measurements.
0. **[GT_TSDF_DIAGNOSTIC.md](GT_TSDF_DIAGNOSTIC.md)** — direct ExMesh initialization research and source-GT depth TSDF comparison, with provenance, measurements and commands.
0. **[PIPELINE_FIXES.md](PIPELINE_FIXES.md)** — corrected foreground, TSDF, cameras and texture pipeline; lego reproduction and honest quality limits.
1. [MESH_UV_PIPELINE.md](MESH_UV_PIPELINE.md) — preferred architecture, mathematical data contract, TSDF filtering, mesh optimization alternatives, UV texturing, gate/metrics.
2. [SOTA_2026.md](SOTA_2026.md) — verified 2025–2026 research comparison with affiliations, venues, exact reported numbers and non-transferability notes.
3. [VGGT_OMEGA_ADAPTER.md](VGGT_OMEGA_ADAPTER.md) — official Ω API, 1B checkpoint versions, required K/pose/depth adapter and memory figures.
4. [DECISION_NO_GAUSSIANS.md](DECISION_NO_GAUSSIANS.md) — recorded architecture decision and source removal.
5. [DIRECT_UV.md](DIRECT_UV.md) — downstream UV VAE / cross-seam latent research after fixed mesh.
6. [DATASETS.md](DATASETS.md), [EVALUATION.md](EVALUATION.md) — current ETH3D smoke data; train-only alignment and geometry/texture metrics.
7. [CPU_VALIDATION.md](CPU_VALIDATION.md) — CPU-tested and not-tested boundaries.
8. [papers/README.md](papers/README.md) — paper links + optional PDF caching script.
9. [UPSTREAMS.md](UPSTREAMS.md), [COMPATIBILITY.md](COMPATIBILITY.md) — what actually lives in repo.
10. **[QUALITY_ATTRIBUTION.md](QUALITY_ATTRIBUTION.md)** — 质量归因：against ETH3D GT, VGGT-Ω's poses (0.59% / 0.57°) and intrinsics (−0.25% focal) are accurate; the bad meshes were our post-processing bugs. Includes before/after numbers.
11. **[ISSUES_2026_11_02.md](ISSUES_2026_11_02.md)** — 问题总结：11 个已定位并修复的缺陷（内参差 10 倍、仿射推导、TSDF 符号、GLB 材质、torch 崩溃等），以及仍未解决的背景外壳 / 闭合性 / 评估缺口。
12. **[EXMESH_ADAPTER.md](EXMESH_ADAPTER.md)** — ExMesh compile patches (`<cfloat>`, `<cstdint>`, mask resize), the input directory contract, and the OBJ/GLB exporter for the checkpoint tensors.

**Historical code:** `2dsugar/` is retained for comparison only. `3dgs/` is removed. `vggt/` is still original VGGT, **not** VGGT-Ω.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/download_eth3d.py --dry-run
python3 scripts/download_papers.py --list
python3 scripts/doctor.py
```

Paper PDFs are fetched **on demand**, not committed. `scripts/run_all.sh` explicitly refuses to run the retired 3DGS branch and does not claim the new Mesh/UV path has been implemented.
