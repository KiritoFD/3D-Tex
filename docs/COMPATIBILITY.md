# Environment and dependency policy (2026-11-01)

**Active engineering:** original VGGT vendored source, ETH3D download and CPU tests. Next: VGGT-Ω checkpoint-gated model + CPU sparse TSDF + xatlas, followed by optional CUDA mesh/voxel refiners.

- Original `vggt/` is still available with conda environment `recon-vggt`; no claim it is VGGT-Ω.
- `3dgs/` and `recon-3dgs` have been removed from the active build; `scripts/install_envs.sh all` installs original VGGT only. There is no 3DGS launch command.
- `2dsugar/` remains **legacy source**, manually selectable via `bash scripts/install_envs.sh 2dsugar`, and requires old CUDA 11.8 / differentiable Gaussian rasterizer builds. It is not an implicit dependency of A0/A1/A2.
- Proposed new CPU geometry dependencies (Open3D, NumPy, optional xatlas Python binding) are **not pinned or installed yet**; install/version/verify only when their code is implemented. Do not suggest the new pipeline is executable.
- ExMesh upstream uses CUDA `nvdiffrast`, `pytorch3d` and a PGSR initializer; replacing PGSR with an Ω/TSDF mesh is a new code integration.
- GeoSVR upstream uses customized sparse voxel CUDA rasterizer, older Torch/CUDA environment and significant CPU RAM for mesh extraction. Isolate its dependencies.
- VGGT-Ω official checkpoint download requires access request on Hugging Face; weights are neither stored nor vendored here. Consult its original license separately.
- One 4090 24GB is a reasonable first target. Official Ω A100 25-frame peak 7.80GB; ExMesh DTU original workflow 4.5GB; GeoSVR appendix ~11–14GB. These are **not** measured on the proposed integrated pipeline.

[Complete design](MESH_UV_PIPELINE.md) | [Method review](SOTA_2026.md) | [Upstream provenance](UPSTREAMS.md)
