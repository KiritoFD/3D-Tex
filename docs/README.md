# 3D-Tex research and implementation notes

**Active target: no-Gaussian Mesh + UV from VGGT-Ω. This target is a design, not an already working pipeline.**

1. [MESH_UV_PIPELINE.md](MESH_UV_PIPELINE.md) — preferred architecture, mathematical data contract, TSDF filtering, mesh optimization alternatives, UV texturing, gate/metrics.
2. [SOTA_2026.md](SOTA_2026.md) — verified 2025–2026 research comparison with affiliations, venues, exact reported numbers and non-transferability notes.
3. [VGGT_OMEGA_ADAPTER.md](VGGT_OMEGA_ADAPTER.md) — official Ω API, 1B checkpoint versions, required K/pose/depth adapter and memory figures.
4. [DECISION_NO_GAUSSIANS.md](DECISION_NO_GAUSSIANS.md) — recorded architecture decision and source removal.
5. [DIRECT_UV.md](DIRECT_UV.md) — downstream UV VAE / cross-seam latent research after fixed mesh.
6. [DATASETS.md](DATASETS.md), [EVALUATION.md](EVALUATION.md) — current ETH3D smoke data; train-only alignment and geometry/texture metrics.
7. [CPU_VALIDATION.md](CPU_VALIDATION.md) — CPU-tested and not-tested boundaries.
8. [papers/README.md](papers/README.md) — paper links + optional PDF caching script.
9. [UPSTREAMS.md](UPSTREAMS.md), [COMPATIBILITY.md](COMPATIBILITY.md) — what actually lives in repo.

**Historical code:** `2dsugar/` is retained for comparison only. `3dgs/` is removed. `vggt/` is still original VGGT, **not** VGGT-Ω.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/download_eth3d.py --dry-run
python3 scripts/download_papers.py --list
python3 scripts/doctor.py
```

Paper PDFs are fetched **on demand**, not committed. `scripts/run_all.sh` explicitly refuses to run the retired 3DGS branch and does not claim the new Mesh/UV path has been implemented.
