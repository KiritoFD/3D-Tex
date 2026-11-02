# Pipeline corrections and lego reconstruction (2026-11-02)

The previous output was a smoke test. Camera accuracy and cross-view depth agreement do not establish surface accuracy. This reconstruction reuses the same 16 Omega predictions and checkpoint; no inference or model replacement is hidden in the comparison.

## Confirmed defects and corrections

| Stage | Defect | Correction |
|---|---|---|
| Foreground | No segmentation before fusion; depth-quantile masks incorrectly called foreground | `--mask-source alpha` or `files`, transformed using the recorded pixel affine; original foreground saved separately from depth validity |
| Consistency | A low-confidence or background target pixel could support another view | Require valid foreground and confidence in the supporting view; lego uses two supporting views, 3% relative and 0.002 absolute tolerance |
| Fusion | Sample splatting counted a voxel multiple times per frame | Chunked projective integration, exactly one update per voxel per frame |
| Extraction | Filling unknown space with +1 created artificial zero crossings | Keep triangles only in cells with eight observed corners; include the voxel-centre offset |
| Coordinates | Principal points scaled twice if depth size changed; half-pixel conventions differed | Scale each intrinsic row once; use integer image coordinates consistently |
| ExMesh | Resized kicker images retained original focal length; all views shared first-frame K | Scale K with actual copied image size and export a camera entry per frame |
| ExMesh priors | Model-grid priors and masks resized without accounting for crop | Transform model-grid depth/masks to copied-image grid; preserve RGBA; no depth-percentile removal |
| Texture | Original face indices indexed UV-remapped vertices | Use UV face indices for remapped positions and normals |
| Texture | No occlusion check despite the docstring claim | Check projected texel depth against filtered depth before RGB accumulation |
| Export | PLY-to-OBJ wrote original positions with UV-remapped face indices | Write remapped positions; use shared checked GLB material exporter |

The ExMesh renderer assumes a centred principal point. The adapter now fails on an unsupported off-axis calibration instead of silently replacing it. Normal texture weighting is two-sided because the initial TSDF can contain open surfaces. Fusion remains a dense CPU grid with chunked evaluation, not a sparse hash implementation.

## Provenance and limits

- Scene: `nerf_lego16`, 16 training views, raw Omega depth at 512x512.
- Mask: dataset-provided RGBA alpha, **GT foreground**, not predicted segmentation. External predicted masks can instead be supplied with `--mask-source files --mask-dir ...`; they must be in original-image coordinates.
- Raw checkpoint SHA256: `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
- Corrected filter retains 1,224,395 pixels (29.2% of all image pixels). Its inlier statistic is foreground-restricted and is not directly comparable to the old background-inclusive statistic.
- Corrected TSDF: 256 voxel divisions, truncation 4 voxels, uniform per-view weights; 270,227 faces, 41,985 boundary edges, zero nonmanifold edges before cleanup. These numbers do **not** prove geometrical accuracy or completeness.
- ExMesh input drops components smaller than 100 faces, then simplifies to 80,000 faces. This cleanup is specific to this single-object experiment.
- ExMesh training uses 400x400 images (`-r 2`), a 1024x1024 texture atlas and 10,000 iterations. The 2048x2048 observed TSDF texture is exported separately.
- No surface GT metric or heldout test is claimed. Comparison renders use four **training** cameras and unlit textures. Missing surfaces and small mechanisms can remain wrong even when silhouettes improve.

## Reproduction

The measured CPU environment is pinned in `configs/geometry_requirements.txt` (`pip install -r configs/geometry_requirements.txt`). Blender is used only for preview rendering, and ExMesh uses its separate existing WSL CUDA environment.

```powershell
.venv/Scripts/python.exe scripts/consistency_filter.py --scene nerf_lego16 --out data/scenes/nerf_lego16/geometry/filtered_fixed --mask-source alpha --min-agree 2 --tau-rel 0.03 --tau-abs 0.002
.venv/Scripts/python.exe scripts/tsdf_mesh.py --scene nerf_lego16 --filtered data/scenes/nerf_lego16/geometry/filtered_fixed --out data/scenes/nerf_lego16/mesh_fixed --voxel-div 256 --trunc-voxels 4 --weight-mode uniform
.venv/Scripts/python.exe scripts/bake_texture.py --scene nerf_lego16 --mesh data/scenes/nerf_lego16/mesh_fixed/mesh_init.ply --filtered data/scenes/nerf_lego16/geometry/filtered_fixed --out deliverables/lego_fixed_tsdf --texels 2048 --max-faces 100000 --min-component-share 0.99
.venv/Scripts/python.exe scripts/exmesh_adapter.py --scene nerf_lego16 --out data/scenes/nerf_lego16/exmesh_fixed --filtered data/scenes/nerf_lego16/geometry/filtered_fixed --mesh data/scenes/nerf_lego16/mesh_fixed/mesh_init.ply --max-faces 80000 --min-component-faces 100
```

The adapter creates the cleaned 80,000-face input described above using these explicit flags. The upstream ExMesh checkout is `/root/src/exmesh` in WSL. Its existing build and mask-resize patches are described in [EXMESH_ADAPTER.md](EXMESH_ADAPTER.md). One additional local runtime patch lazily creates LPIPS only when positive test iterations are requested; LPIPS is not used by this training loss. The original training script was backed up as `train.py.before_lego_fixed`.

```bash
cd /root/src/exmesh
EXMESH_TEX=1024 /root/venvs/recon128/bin/python -u train.py \
  -s /mnt/g/GitHub/3D-Tex/data/scenes/nerf_lego16/exmesh_fixed \
  -m /mnt/g/GitHub/3D-Tex/data/scenes/nerf_lego16/exmesh_fixed/out \
  -r 2 --iterations 10000 --save_iterations 1000 3000 10000 --test_iterations -1
```

Render using `blender -b --python scripts/render_asset.py -- --asset <mesh.glb> --out <directory>`. Use `scripts/compare_assets.py --before <old-render-dir> --after <new-render-dir>` to write matching-camera panels and training-view PSNR/IoU. The `.blend` preview uses emission materials to inspect baked colour independently of lighting.

## Verification

Four focused regressions cover real kicker focal scaling and per-frame calibration, alpha mapping, unknown-band surface extraction, and one observation per frame/voxel. Run `python -m unittest discover -s tests -p test_geometry_fixes.py -v`. GLB texture/materials are read back and Blender renders are visually inspected. This is a bug-fix check, not benchmark validation.

## Completed output

Training completed at iteration 10,000. The final asset is `deliverables/lego_fixed/mesh.glb` (102,849 vertices, 201,153 faces, 1024x1024 learned texture), with `mesh.obj`, `mesh.mtl`, `albedo.png` and a packed Blender preview. The four-view figure and reference/before/after comparison are in that directory. The separately baked CPU baseline is `deliverables/lego_fixed_tsdf/`.

Across the four matching **training** views, mean silhouette IoU changed from 0.7116 to 0.9636 and foreground RGB PSNR from 16.92 to 19.17 dB. These are diagnostic render metrics, not heldout or surface GT scores. This comparison changes foreground input, filtering, fusion, calibration and training budget together; it does not isolate a single fix. The background shell is removed, but the mesh still has rough surfaces, missing small rods and inaccurate fine detail. A full 10,000-step run did not solve those remaining reconstruction errors.

The validation script itself also retained the old conjugated K conversion and original-K/model-depth mismatch; both were corrected. Historical reports produced by that script should be regenerated before being treated as current evidence.
