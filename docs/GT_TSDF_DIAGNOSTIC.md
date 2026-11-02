# Direct mesh initialization and GT-depth TSDF diagnostic

## What the published methods actually do

Sources checked on 2026-11-02:

| Method | Initialization and surface construction | Implication here |
|---|---|---|
| ExMesh default | The paper uses a 7k-step 2DGS model followed by 256³ TSDF; the current README recommends 5k-step PGSR followed by its TSDF mesh export. Geometry and UV are then optimized explicitly. | Calling the optimization explicit-mesh does not make its default initialization TSDF-free. |
| ExMesh sphere ablation | A sphere of about 1,000 faces with random colours can deform/subdivide into the target. The authors report slower convergence and slightly lower geometric accuracy. | Bypassing TSDF is supported in principle; it is an alternative initialization experiment, not a guaranteed quality improvement. |
| DMesh++ | The project demonstrates random-grid and point-cloud initialization, and learns mesh connectivity using multi-view colour and depth. | An example of a different topology-learning route; its demonstrated colour+depth supervision is material to the comparison. |
| MeshSplatting | Optimize a disconnected semi-transparent triangle soup, establish connectivity with restricted Delaunay triangulation, then refine toward opaque connected triangles. | A non-TSDF route with different representation and connectivity machinery; it is not simply deleting the TSDF command from ExMesh. |

Primary sources: [ExMesh paper, sections 4.1 and 4.3](https://arxiv.org/html/2606.07288), [ExMesh README](https://github.com/Fan-Treasure/ExMesh), [DMesh++ project](https://sonsang.github.io/dmesh2-project/), [MeshSplatting project](https://meshsplatting.github.io/).

The practical next direct-ExMesh experiment is a coarse sphere localized from the input cameras/foreground or Omega foreground points, keeping the same RGB views and masks. That experiment has **not** been run by this diagnostic. A depth-grid triangulation or Poisson reconstruction is another initialization, with its own discontinuity/normal/overlap issues; those are engineering proposals, not results reproduced here.

## Ground-truth caveat

The cached NeRF synthetic archive has 8-bit depth PNGs only in its test split. [NeRF collaborator statement](https://github.com/bmild/nerf/issues/77): these files were accidentally included and were not validated by the original project. [NeRFiller's conversion](https://github.com/ethanweber/nerfiller/blob/main/nerfiller/scripts/blender_to_nerfstudio.py) uses `(1 - q/255) * 8`, but this does not turn an unverified quantized export into exact surface ground truth.

Instead this diagnostic uses the public [Robust e-NeRF lego Blender source](https://huggingface.co/datasets/wengflow/robust-e-nerf/blob/main/scenes/lego.blend). This is an externally published source-scene copy, not a neural reconstruction. Its geometry must first match the NeRF RGB silhouettes under the original GT cameras; any mismatch must be reported. Embedded Blender scripts are not executed.

## Controlled experiment

- Reuse exactly the original 16 training-image names and corresponding GT train camera transforms.
- Use the same 512x512 depth grid as Omega and the same original RGBs for texture.
- Evaluate the source mesh including modifiers in Blender, build a world-space BVH, and take first ray intersections. Save float32 **camera z-depth**, not ray length and not 8-bit depth.
- Keep this in `data/scenes/nerf_lego_gt16/geometry/gt/`; do not place GT in the Omega inference inputs or overwrite its predictions.
- Pass the resulting depth/cameras directly to the same corrected CPU TSDF, at 256 divisions and truncation 4 voxels with uniform view weights. No ExMesh optimization or surface smoothing is used for the GT-TSDF result.
- Render both colour and plain geometry. TSDF is still a discretized reconstruction of visible GT surfaces, not the original source mesh itself; hidden surfaces, sub-voxel features and topology can remain missing.

The joint replacement of camera and depth tests whether the same fusion pipeline can retain clean geometry given clean inputs. It does not separate Omega camera error from Omega depth error. Texture metrics also include source-image illumination/view dependence and must not be interpreted as surface error.

## Commands

Download the public source scene and prepare the isolated camera/image manifest, then run Blender with:

```powershell
.venv/Scripts/python.exe scripts/fetch_url.py --url https://huggingface.co/datasets/wengflow/robust-e-nerf/resolve/main/scenes/lego.blend --out data/.cache/nerf-syn/lego_source.blend -n 8
.venv/Scripts/python.exe scripts/diagnostics/prepare_lego_gt.py
blender -b --disable-autoexec --python scripts/diagnostics/lego_gt_depth.py -- --source data/.cache/nerf-syn/lego_source.blend --scene data/scenes/nerf_lego_gt16
.venv/Scripts/python.exe scripts/tsdf_mesh.py --scene nerf_lego_gt16 --filtered data/scenes/nerf_lego_gt16/geometry/gt --out data/scenes/nerf_lego_gt16/mesh --voxel-div 256 --trunc-voxels 4 --weight-mode uniform
.venv/Scripts/python.exe scripts/bake_texture.py --scene nerf_lego_gt16 --geometry data/scenes/nerf_lego_gt16/geometry/gt --filtered data/scenes/nerf_lego_gt16/geometry/gt --mesh data/scenes/nerf_lego_gt16/mesh/mesh_init.ply --out deliverables/lego_gt_tsdf --texels 2048 --max-faces 100000 --min-component-share 0.99
```

## Measured result

The source file SHA256 is `91c2824364ac9cd957fb86f33329bd4d98e8c493be05c5f08727395a02deae90`, matching the published file. First-hit source silhouettes versus the dataset's alpha masks have mean IoU **0.9868**, minimum **0.9725**, maximum **0.9940**. This public source copy closely matches the dataset but is not asserted to match it pixel-perfectly; source rig evaluation, silhouettes and camera pixel conventions can differ.

| Same TSDF settings, before simplification | Omega inputs | Source GT inputs |
|---|---:|---:|
| Views / depth grid | 16 / 512² | 16 / 512² |
| Faces | 270,227 | 251,156 |
| Boundary edges | 41,985 | 30,964 |
| Nonmanifold edges | 0 | 0 |

The GT result preserves substantially more regular baseplate studs, roof surfaces and narrow rods, while still losing small structures and leaving holes. These remain limited by view coverage, sampling, TSDF truncation and extraction. Boundary count is a structural diagnostic, not a surface-accuracy score. No ExMesh training was applied to the GT mesh.

For an additional input diagnostic, fit a single Sim(3) to the **16 training camera centres only**. The predicted-to-GT scale is **4.72325**, mean camera centre residual **0.02891 source units**, and mean rotation error **0.314°**. Comparing the scale-adjusted raw Omega z-depth to the source z-depth at corresponding foreground pixels gives mean absolute relative error **1.213%**, median **0.732%**, and 90th percentile **2.183%**. This comparison includes residual pose/intrinsic and source-scene mismatch; it is not an official depth benchmark. Mean absolute depth difference **0.04550 source units** is about **3.96 times** the GT grid's **0.011501** voxel size. Thus apparently small global depth errors can still create several-voxel local surface disturbances.

Artifacts:

- `deliverables/lego_gt_tsdf/mesh.glb`, `mesh.obj` + `material.mtl` + `albedo.png`: 99,999-face display export, 2048² observed texture.
- `deliverables/lego_gt_tsdf/mesh_raw.ply`: original 251,156-face TSDF extraction.
- `deliverables/lego_gt_tsdf/preview.blend`: packed colour preview, opens in material preview.
- `deliverables/lego_gt_tsdf/raw_geometry/preview.blend`: original geometry with a fixed grey matcap and no texture or smoothing.
- `deliverables/lego_gt_tsdf/comparison.png`: input / Omega-TSDF / GT-TSDF, with colour and raw geometry rows. Each reconstruction uses the camera estimate appropriate to that input; these are corresponding input viewpoints rather than identical numerical poses.
- `deliverables/lego_gt_tsdf/input_comparison_report.json`, `mesh_report.json`, and `data/scenes/nerf_lego_gt16/geometry/gt/source_report.json`: numerical evidence.

Both coloured exports use the same 99% component coverage, 100k-face simplification cap, 2048² atlas and depth-checked texture bake. The raw geometry row uses the unsimplified meshes. Generated assets and downloaded source data remain git-ignored.

## Conclusion and next experiment

The result rejects the claim that TSDF alone explains the poor output: cleaner camera/depth inputs produce cleaner geometry through the same fusion implementation. It also does not exonerate post-processing, because the GT extraction itself retains discretization and visibility defects. The next controlled direct-ExMesh run should compare sphere initialization against the existing Omega-TSDF initialization using fixed RGB/cameras/masks and report convergence, raw geometry and surface metrics separately. This document does not claim that sphere initialization has been reproduced locally.
