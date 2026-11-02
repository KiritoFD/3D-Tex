# Capture-scheme controlled experiments (lab notebook)

Status: group 1 (view count / angular density) complete. Renderer note: this axis
uses the dataset's own 100 pre-rendered path-traced train views (uniform-azimuth
subsets), because Cycles GPU initialisation crashes on this host and EEVEE hits an
access violation (3.5 GiB free VRAM is insufficient); new-render axes (lighting,
blur, texture) are blocked on this and tracked below.

Protocol: identical to P0/P0.5 — train-only Sim(3) on camera centres, nearest
regrid onto the GT K grid, frozen P0 diagnostic grid (voxel 0.01150144, trunc 4
voxels, uniform weights), measured against the source BVH with 50k fixed-seed
samples in both directions. GT depth is evaluation-only. Sharp/flat structure
split: per-source-triangle max dihedral > 40° (27.0% of 2,035,378 triangles).

## Group 1 — view count (uniform-azimuth subsets of the dataset train views)

Commands (per N in 8/16/32/64):

```powershell
.venv/Scripts/python.exe scripts/capture/build_dataset_subset.py --views N --out data/scenes/capture_setN --name setN
blender -b --disable-autoexec --python scripts/capture/capture_render.py -- --out data/scenes/capture_setN --views N --name setN --pass-mode depth --depth-resolution 512 --poses-json data/scenes/capture_setN/poses.json
.venv/Scripts/python.exe scripts/adapters/vggt_omega_geometry.py --scene-dir data/scenes/capture_setN --checkpoint data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt --device cuda --aggregator-dtype bfloat16 --image-resolution 512
.venv/Scripts/python.exe scripts/capture/capture_eval.py --capture data/scenes/capture_setN --out outputs/capture_setN
blender -b --disable-autoexec --python scripts/diagnostics/measure_surfaces.py -- --out outputs/capture_setN --source-flags outputs/lego_source_edge_flags.npy
```

Subsets: uniform azimuth rank over the 100 train views (mean gap: set8 47.7°,
set16 23.4°, set32 11.5°, set64 5.7°; dataset trajectory keeps some near-duplicate
views at 64). Manifests: `data/scenes/capture_setN/{capture_subset.json,poses.json,geometry/gt/manifest.json}`.
64-view VGGT inference peaks at 6,335 MiB VRAM / 611 s on the 8 GiB RTX 4070
laptop (first OOM retry recovered); 32 views: 4,490 MiB / 20 s.

### Depth / normal error (image space, GT grid)

| Config | depth abs-rel mean / median | normal mean° | flat° | edge° |
|---|---:|---:|---:|---:|
| set8 | 2.055% / 1.381% | 33.35 | 16.55 | 35.32 |
| set16 | 1.654% / 1.055% | 32.92 | 15.51 | 34.96 |
| set32 | 1.462% / 0.888% | 32.53 | 15.38 | 34.63 |
| set64 | 1.324% / 0.805% | 32.03 | 14.64 | 34.16 |
| anchor16 (original 16, P0) | 1.242% / 0.701% | 32.16 | 15.61 | 33.86 |

### Mesh metrics (frozen P0 grid)

| Config | F@1vox ↑ | P / R @1vox | accuracy (vox) ↓ | completeness (vox) ↓ | acc_sharp / acc_flat (vox) | P_sharp@1vox | R_sharp@1vox |
|---|---:|---:|---:|---:|---:|---:|---:|
| set8 | 0.5064 | 0.531 / 0.484 | 1.628 | 1.356 | 2.481 / 1.136 | 0.282 | 0.451 |
| set16 | 0.5255 | 0.503 / 0.550 | 1.623 | 1.098 | 2.343 / 1.214 | 0.270 | 0.537 |
| set32 | **0.5352** | 0.460 / 0.639 | 1.813 | 0.906 | 2.583 / 1.362 | 0.228 | 0.578 |
| set64 | 0.4940 | 0.394 / 0.664 | 2.292 | 0.879 | 3.101 / 1.736 | 0.190 | 0.631 |
| anchor16 | **0.6206** | 0.554 / 0.705 | 1.403 | 0.809 | — | — | — |

### Findings (view-count axis)

1. **Depth error shrinks slowly with N** (2.06% → 1.32%), but the **normal error
   is flat (33.35° → 32.03°)**: the mid-frequency warp does not average out with
   more views. Knee = saturation, not improvement.
2. **F-score peaks at 32 views then degrades** (0.5352 → 0.4940 at 64): recall
   keeps rising (0.48 → 0.66) while precision collapses (0.53 → 0.39). More
   views add more consistently-biased depth sheets; uniform TSDF averaging
   integrates them into a thicker, less accurate surface.
3. **The original non-uniform 16 views beat every uniform subset, including
   64 views** (0.6206 vs 0.4940; depth 1.24% vs 1.32%). The anchor16 trajectory
   mixes elevations 9.7°–85.7° (mean 39°); the ring subsets sit near one
   elevation. **Elevation/parallax diversity beats azimuth density.**
4. **Thin/sharp structures degrade with more views**: precision_sharp@1vox falls
   0.282 → 0.190 (8 → 64) while recall_sharp rises 0.451 → 0.631. View count
   does not un-melt the posts/studs; it trades their precision for coverage.
   Sharp-region accuracy stays ~2–3 voxels vs ~1–1.7 flat everywhere.
5. Anchor16 remains the best capture scheme measured so far. "Add more views"
   is not a fix for the waviness; the residual is a per-view consistent bias.

Visual review: clay renders saved under `outputs/capture_setN/renders/view_*.png`
(inspection pending — media upload service failed during this session; files are
on disk for review; numbers above are complete and unaffected).

## Blocked/pending axes

- New-render axes (lighting direction/shadow, defocus/motion blur, texture,
  resolution via re-render) require a working render engine: Cycles GPU init
  crashes (Microsoft C++ exception) and BLENDER_EEVEE access-violates on this
  host's 8 GiB GPU; Cycles CPU ≈ 25 min/view at 800²/64spp is impractical.
  Resolution axis via image downsampling of the dataset views is still possible
  without new renders.
