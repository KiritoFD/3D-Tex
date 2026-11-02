# P0 pose/depth isolation and RGB geometry optimization (2026-11-02)

## Outcome

P0 identifies significant pose and depth effects separately. Replacing depth alone causes a larger surface-accuracy loss than replacing pose alone, but their effects are correlated and cannot be added as independent percentages. Correcting pose alone does not fix the predicted depth's surface noise.

The first PGSR comparison is a **negative result**: it produces convincing training RGB renders but its exported depth and resulting mesh are geometrically worse in this 16-view configuration. It also fails the heldout mesh appearance comparison. PGSR is therefore retained as an explicit comparison, not presented as the improved reconstruction.

A two-voxel consistency threshold improves Omega's surface precision and F-score, while reducing completeness. This is removal of inconsistent measurements, not recovery of missing structure.

## P0 protocol

- Same 16 original training views and dataset alpha masks; no test images enter inference or optimization.
- One Sim(3) estimated from the 16 training camera centres; scale is applied to raw predicted z-depth. No per-view scale/shift fitting.
- Fix **GT intrinsics for every case**. Resample predicted depths using only the intrinsic mapping (nearest-neighbour); extrinsic changes are kept separate. This excludes intrinsic variation from the pose factor, but resampling is itself a sampling limitation.
- Cross GT/aligned-Omega extrinsics with GT/scaled-Omega depth. The previous broad consistency filter is not applied to these four raw-depth cases; they differ from the previously published filtered Omega mesh.
- Use one frozen GT-derived diagnostic ROI, voxel grid and truncation band for every case. Voxel size is 0.01150144 source units, truncation is four voxels; uniform fusion weights, no smoothing or ExMesh.
- Render all meshes from identical GT cameras, with identical grey matcap lighting.
- Accuracy: 50,000 fixed-seed, area-weighted mesh surface points to exact source-scene triangles using BVH nearest distance.
- Completeness: 50,000 fixed-seed **observed GT-depth points** to mesh triangles. This assesses observed surfaces, not full CAD completeness. F-score uses these precision/recall sets at one and two voxel tolerances.
- Source scene is the same public copy used by the previous diagnostic (mean alpha IoU 0.9868); source mismatch remains a limitation. These are diagnostics, **not official benchmark scores**.

| Case | Accuracy mean / voxel (lower) | Observed completeness mean / voxel (lower) | F-score @ 1 voxel | F-score @ 2 voxels |
|---|---:|---:|---:|---:|
| GT pose + GT depth | 0.360 | 0.311 | 0.9357 | 0.9946 |
| GT pose + Omega depth | 1.375 | 0.855 | 0.6021 | 0.8351 |
| Omega pose + GT depth | 0.747 | 0.625 | 0.7670 | 0.9535 |
| Omega pose + Omega depth | 1.403 | 0.809 | 0.6206 | 0.8396 |
| Previous Omega percentage-filter depth | 1.233 | 0.850 | 0.6399 | 0.8580 |
| Previous corrected ExMesh 10k (external mesh baseline) | 1.107 | 1.069 | 0.6190 | 0.8502 |
| Omega pose + PGSR depth (10k) | 3.182 | 2.201 | 0.3324 | 0.5377 |
| Omega pose + Omega depth (2-voxel filter) | 0.676 | 1.217 | 0.6927 | 0.9020 |
| Omega pose + PGSR depth (2-voxel + alpha filter) | 3.056 | 3.687 | 0.2700 | 0.4694 |

For the four core P0 cases, F-score@1 voxel is 0.9357 with GT inputs, 0.7670 with only predicted pose, 0.6021 with only predicted depth, and 0.6206 with both predicted. The latter two are similar: pose/depth coupling and the nonlinear fusion/filtering prevent an additive error attribution.

On the common GT pixel grid, raw Omega mean absolute relative depth error is 1.242%, median 0.701%. The 5-pixel finite-difference surface normal operator gives unsigned mean error 32.16° and median 29.00°. On pixels with GT-normal variation below 2°, mean normal error is 15.61° (50,759 pixels). This excludes depth-mask boundaries, but it is still a stencil-derived diagnostic rather than independent exact GT normal supervision.

## PGSR comparison

Implementation: the PGSR snapshot bundled inside [ExMesh](https://github.com/Fan-Treasure/ExMesh) revision `2cecbe513a0e2e021cd5cab509761ddc458ee526`; method reference: [official PGSR](https://github.com/zju3dv/PGSR).

- Initialization: 30,000 deterministic samples from filtered **Omega depth**, in Omega gauge, plus Omega cameras. No source-scene GT camera/depth/mesh is an optimization input. Dataset alpha masks are the same GT foreground inputs as earlier runs and are declared as such.
- 16 original 800² RGBs, trained at 400² (`-r 2`), NCC at 800² (`--ncc_scale 0.5`). No camera BA is performed by this run.
- 10,000 iterations; official single/multiview geometry losses activate after iteration 4,000. Densification ends at 7,000, point budget 400,000, final count 399,075.
- Neighbour angle 60°: the 30° default gives zero neighbours for 3 of 16 views. This is a **changed hyperparameter** and a limitation of this sparse-view experiment, not an official PGSR reproduction. Camera errors, sparse overlaps, plane warps at larger angles, and view-dependent appearance may all affect the result; this experiment does not isolate them.
- The existing CUDA rasterizer/knn builds are reused. `pytorch3d.transforms.quaternion_to_matrix` is replaced by PGSR's own normalized CUDA `build_rotation` helper; its only dependency use was that conversion. A second patch exposes the already-computed rasterized alpha for export. `pgsr_geometry.py patch` applies these idempotently, preserving local backup files.
- Geometry export: float32 optimized `plane_depth` at 512², original foreground masks, alpha saved separately as **coverage**, not calibrated depth confidence.

Raw PGSR depth mean absolute relative error increases from Omega's 1.242% to 3.141%, and stencil-derived normal mean error increases from 32.16° to 44.34°. This occurs despite plausible native Gaussian RGB renders. It is not evidence that PGSR cannot work on other views/cameras/settings, nor evidence that the current exported mesh is an improvement.

## Two-voxel filtering

The new `--voxel-size` / `--max-voxel-error` filter mode makes the acceptance tolerance explicit in fusion-grid units and disables the percentage term. It also reports pairwise median residuals in voxel units (including occlusion/disagreement within the eligible foreground, so this is not an accuracy score).

The native Omega grid has voxel size 0.002722 in its gauge. A two-voxel threshold is 0.005444; require two other supporting views. For PGSR additionally require rasterized alpha >=0.9. Omega retains 928,008 pixels, PGSR retains 883,475. The extra PGSR alpha condition means their post-filter comparison is not a pure optimizer-only ablation.

Relative to the existing 3%-plus-absolute filter, F-score improves from 0.6399 to 0.6927 and mesh-to-source error drops from 1.233 to 0.676 voxels, while observed completeness error worsens from 0.850 to 1.217. Relative to raw Omega depth, mesh-to-source mean error drops from 1.403 to 0.676 diagnostic voxels, but observed completeness error rises from 0.809 to 1.217 voxels. Precision@1 voxel rises from 0.554 to 0.780, observed recall falls from 0.705 to 0.623. The filtering improvement must therefore be reported with its holes/coverage cost. PGSR's filtering does not rescue its overall geometry.

Display assets use a separate **GT-free production ROI** derived from the original Omega foreground depths (`--roi-geometry`), not the GT diagnostic bounds. The numerical table above uses the frozen common GT diagnostic grid. Do not mix those grids' mesh statistics.

## Untouched test-view mesh appearance

Indices 0,50,100,150 from the original test split were declared before evaluation. Their GT cameras are mapped into Omega gauge using the train-only Sim(3). They are not used for training, stopping, checkpoint selection or parameter choice. Both meshes use the same cameras, 512² render size, unlit textures and white-background RGB comparison. This is four-view PSNR/IoU only; LPIPS and the full 200-view benchmark were not run.

| Mesh | Mean PSNR on white background | Mean foreground PSNR | Mean silhouette IoU |
|---|---:|---:|---:|
| Previous corrected ExMesh 10k | 19.41 dB | 17.30 dB | 0.9317 |
| PGSR 10k -> TSDF -> observed bake | 17.30 dB | 16.43 dB | 0.8802 |

This comparison includes different geometry optimizers and different texture construction (learned ExMesh 1024² versus 2048² observed bake), so appearance differences cannot be attributed solely to geometry.

The two-voxel display asset was also evaluated on the same four test views: white-background PSNR **16.21 dB**, foreground PSNR **14.23 dB**, silhouette IoU **0.8414**. Its missing-surface penalty outweighs its precision gain for appearance. It is a geometry-precision diagnostic, not a replacement for the previous display asset. The previous ExMesh asset itself has surface F-score@1 voxel **0.6190**, mean accuracy **1.107** voxels and observed completeness error **1.069** voxels under the same source-triangle metric (measured as an external mesh baseline, not another TSDF ablation).

## Reproduction and artifacts

```powershell
.venv/Scripts/python.exe scripts/diagnostics/pose_depth_ablation.py
blender -b --disable-autoexec --python scripts/diagnostics/measure_surfaces.py --
.venv/Scripts/python.exe scripts/adapters/pgsr_geometry.py prepare
```

```bash
cd /mnt/g/GitHub/3D-Tex
/root/venvs/recon128/bin/python scripts/adapters/pgsr_geometry.py patch
cd /root/src/exmesh/PGSR
/root/venvs/recon128/bin/python -u train.py   -s /mnt/g/GitHub/3D-Tex/data/scenes/nerf_lego16/pgsr_workdir   -m /mnt/g/GitHub/3D-Tex/data/scenes/nerf_lego16/pgsr_workdir/out   -r 2 --iterations 10000 --ncc_scale 0.5 --multi_view_max_angle 60   --densify_until_iter 7000 --max_all_points 400000   --save_iterations 5000 10000 --test_iterations -1
cd /mnt/g/GitHub/3D-Tex
/root/venvs/recon128/bin/python scripts/adapters/pgsr_geometry.py export
```

```powershell
.venv/Scripts/python.exe scripts/diagnostics/pose_depth_ablation.py --optimized-geometry data/scenes/nerf_lego16/geometry/pgsr --only-optimized
blender -b --disable-autoexec --python scripts/diagnostics/measure_surfaces.py -- --case pred_pose_pgsr_depth
.venv/Scripts/python.exe scripts/consistency_filter.py --scene nerf_lego16 --out data/scenes/nerf_lego16/geometry/omega_2vox --mask-source alpha --min-agree 2 --voxel-size 0.002722 --max-voxel-error 2
.venv/Scripts/python.exe scripts/consistency_filter.py --scene nerf_lego16 --geometry data/scenes/nerf_lego16/geometry/pgsr --out data/scenes/nerf_lego16/geometry/pgsr_consistent --mask-source alpha --min-agree 2 --voxel-size 0.002722 --max-voxel-error 2 --conf-threshold 0.9
```

Native display-mesh export (no GT diagnostic bounds):

```powershell
.venv/Scripts/python.exe scripts/tsdf_mesh.py --scene nerf_lego16 --filtered data/scenes/nerf_lego16/geometry/omega_2vox --roi-geometry data/scenes/nerf_lego16/geometry/filtered_fixed --out data/scenes/nerf_lego16/mesh_omega_2vox --voxel-div 256 --trunc-voxels 4 --weight-mode uniform
.venv/Scripts/python.exe scripts/bake_texture.py --scene nerf_lego16 --geometry data/scenes/nerf_lego16/geometry/omega --filtered data/scenes/nerf_lego16/geometry/omega_2vox --mesh data/scenes/nerf_lego16/mesh_omega_2vox/mesh_init.ply --out deliverables/lego_omega_2vox --texels 2048 --max-faces 100000 --min-component-share 0.99
.venv/Scripts/python.exe scripts/diagnostics/heldout_views.py prepare
blender -b --python scripts/render_asset.py -- --asset deliverables/lego_omega_2vox/mesh.glb --out deliverables/lego_omega_2vox/heldout --cameras outputs/lego_heldout/cameras.json --view-indices 0 1 2 3
```

For the raw PGSR export, substitute `geometry/pgsr` for the depth/geometry roots and output to `mesh_pgsr` / `lego_pgsr`. The filtered PGSR mesh uses `geometry/pgsr_consistent` for depth/visibility and `geometry/pgsr` for calibration, with the same original Omega ROI. Source geometry metrics can be rerun with `measure_surfaces.py -- --case <name>`; the previous ExMesh baseline uses `--case previous_exmesh_mesh --mesh deliverables/lego_fixed/mesh.obj --mesh-is-omega`.

- `deliverables/lego_p0/optimization_comparison.png`: previous ExMesh / two-voxel filtering / PGSR coloured view, with their limitations labelled.
- `deliverables/lego_p0/`: four core meshes, common-grid optimized/filter cases, `comparison.png`, depth/normal error maps, `ablation_report.json` and `surface_metrics.json`.
- `deliverables/lego_pgsr/`: raw PGSR-depth TSDF display asset; retained as a negative result, including four heldout renders.
- `deliverables/lego_pgsr_consistent/`: post-filter PGSR display asset, with increased missing surfaces.
- `deliverables/lego_omega_2vox/`: the precision-improved Omega display asset, also with increased missing surfaces.
- `outputs/lego_heldout/metrics.json`: heldout comparison. `outputs/lego_pgsr_train.log` records the completed training.

Both GitHub Actions workflows were deleted at the user's request. No replacement CI workflow or CI-only requirement file is retained. Local diagnostic execution remains available.

## Next geometry work

This run supports testing small-pose refinement and a depth-anchored RGB geometry objective before more unconstrained PGSR training. It does not establish which particular PGSR loss or neighbour configuration caused the regression. More views, COLMAP MVS and further PGSR/ExMesh schedules were not run in this phase. No current candidate simultaneously improves surface precision, completeness and heldout appearance.

## Local verification

The new plane-normal regression and four existing geometry checks pass (five targeted tests); all new scripts compile, and the P0 script verifies the Sim(3) camera-centre transformation numerically. Generated geometry, GLB material/UV metadata and Blender previews were inspected. There is no claim of remote CI success.
