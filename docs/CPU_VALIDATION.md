# CPU validation and GPU acceptance boundaries (2026-11-01)

## Previously verified

- [ETH3D GitHub Actions](https://github.com/KiritoFD/3D-Tex/actions/runs/41308468752): offline split/unit/CLI validation plus **real** pipes/relief/kicker archive downloads, extraction and GT scan presence. Approximately **1.254 GiB compressed, 1.901 GiB expanded** dataset files (excluding model/checkpoint sizes).
- The current wrappers test uses **synthetic COLMAP binary fixtures**. No neural VGGT-Ω model/BA/depth inference, no voxel/CUDA Mesh training and no UV-texture baking is run by this test.
- The retired 3DGS standalone module has been deleted from the tree and is no longer installed by default. Historical `2dsugar/` remains optional for research.

## Runs without GPU

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/download_eth3d.py --dry-run
python3 scripts/download_papers.py --list
python3 scripts/doctor.py
bash -n scripts/install_envs.sh scripts/run_all.sh
```

`scripts/run_all.sh` exits nonzero on purpose: the old pipeline was retired before the new one was implemented. This failure is an API guard, not evidence of a working replacement.

## Future acceptance gates

0. Ω checkpoint access, `vggt_omega/` adapter and camera/depth/uncertainty export; validate pixel transformation, coordinates, scale ambiguity and 12/25-view GPU peaks.
1. CPU/GPU robust TSDF from Ω depth/pose to mesh; synthetic known-depth geometry and ETH3D heldout geometry evaluation.
2. xatlas UV, high-resolution observed-view texel bake, z-test, texture coverage and seam metrics; actual OBJ/GLB validation.
3. Two independent geometry arms: adapted ExMesh initialized from Ω/TSDF (not official PGSR), and GeoSVR with Ω geometry priors; report each dependency/metric scope.
4. Fixed mesh UV atlas diffusion/PBR research only after earlier quality gates.

See [MESH_UV_PIPELINE.md](MESH_UV_PIPELINE.md).
