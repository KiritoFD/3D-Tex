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

## Group 1 — view count (uniform-azimuth subsets of the dataset train views)

### Visual review (34 renders, reviewed in a separate session; integrated here)

Renders under `outputs/capture_setN/renders/`, `deliverables/lego_p05/renders/`
and `deliverables/lego_p05/crops/` were inspected image by image. Conclusions:

1. **The warp is two superimposed scales**: a low-frequency "dunes / dried
   riverbed" component (wavelength ≈ 1/8–1/4 of the vehicle width, strongest on
   the bucket interior and the base plate) plus a high-frequency isotropic
   "grit/cotton-wool" grain that bridges lattice openings into fluffy webs and
   half-blocks window holes. It is neither stripe- nor staircase-shaped.
2. **GT depth through the identical TSDF/voxel pipeline reconstructs the stud
   array, open window holes and thin wheel spokes; predicted depth melts all of
   them.** Voxel resolution and the reconstruction pipeline are therefore NOT
   the bottleneck — the depth values are. This is a premise-level judgement for
   both roadmap documents.
3. The five P0.5 variants (repro / bilinear_both / bilateral / robust_fusion /
   combined) are visually indistinguishable at full-frame and crop level,
   consistent with their <3% metric spread.
4. **Indicator blind spots (important)**:
   - The silhouette "burr cloud" GROWS with view count: 8 views = fat skirt,
     32 = fine fringe, 64 = body-wide spikes. These fly-aways fall outside the
     GT evaluation band and the surface samples, so abs_rel / F-score cannot
     see them — set64 looks best on paper while gaining a new visual defect.
   - The cab-roof crater ("volcano") is present at 8/16/32/64 views — a
     systematic top-surface failure, not a density problem.
   - Fluffy bridges IMPROVE completeness while destroying lattice structures.
   - Back-facing/heldout orientations are visibly worse; the global mean hides
     orientation non-uniformity.

These motivated the new metrics below (burr rate, orientation split, structure
survival, scale-split residual) and the wording of the capture guidelines
("densifying one ring past a point trades silhouette quality for coverage").

### New metrics added after the visual review

- `burr_rate` (capture_eval): fraction of mesh surface samples projecting
  outside the dilated GT foreground or floating ≥2 voxels in front of the GT
  surface, any of the GT views — targets the burr cloud the F-score misses.
- `orientation_split` (measure_surfaces): accuracy/completeness binned by
  azimuth sector — exposes back-facing degradation hidden by the mean.
- `structure` (capture_eval): interior-hole fill rate (windows/gaps wrongly
  filled) and observed-surface coverage (mesh depth within ±2 voxels of GT) —
  structure survival instead of pixel error.
- `residual_lowfreq_rms` / `residual_highfreq_rms` (capture_eval): Gaussian
  σ=8 px split of the per-view depth residual — quantifies the dunes (low) and
  grit (high) components separately.
- `outlier_rate_*` (measure_surfaces): fraction of accuracy/completeness
  samples beyond 2/4 voxels.

## Group 2 — dual-ring elevation diversity (16 low + 8 high)

Motivation: anchor16 (elevations 9.7°–85.7°) beat every uniform ring, so test a
controlled dual-ring from the same dataset pool: 16 views from elevation band
[10°,50°) + 8 views from [60°,91°), uniform azimuth within each band (24 views
total, mean azimuth gap 15.3°).

```powershell
.venv/Scripts/python.exe scripts/capture/build_dataset_subset.py --bands "10,50,16;60,91,8" --out data/scenes/capture_dualring --name dualring
# then the same depth/VGGT/eval/measure chain as group 1
```

| Config | depth abs-rel mean/median | normal° | flat° | edge° | F@1vox | P / R | accuracy (vox) | compl. (vox) | acc_sharp / flat | P_sharp | R_sharp | burr_rate | hole_fill | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dualring24 | 1.386% / 0.784% | 32.42 | **14.44** | 34.89 | **0.5678** | 0.503 / 0.652 | 1.713 | 0.896 | 2.515 / 1.264 | 0.265 | 0.619 | 0.0274 | 0.990 | 0.389 |
| set16 (single ring) | 1.654% / 1.055% | 32.92 | 15.51 | 34.96 | 0.5255 | 0.503 / 0.550 | 1.623 | 1.098 | 2.343 / 1.214 | 0.270 | 0.537 | — | — | — |
| set32 (single ring) | 1.462% / 0.888% | 32.53 | 15.38 | 34.63 | 0.5352 | 0.460 / 0.639 | 1.813 | 0.906 | 2.583 / 1.362 | 0.228 | 0.578 | — | — | — |
| anchor16 | 1.242% / 0.701% | 32.16 | 15.61 | 33.86 | **0.6206** | 0.554 / 0.705 | 1.403 | 0.809 | — | — | — | — | — | — |

Findings:

1. **24 dual-ring views beat 32 single-ring views** (F 0.5678 vs 0.5352) with
   fewer images — elevation diversity is worth more than azimuth density,
   confirming the group-1 trajectory hypothesis. Precision stays at set16's
   level (0.503) while recall exceeds set32 (0.652 vs 0.639).
2. **Best flat-region normal error of any config** (14.44° vs 15.38–15.61° for
   the rings and anchor16): high-elevation views measurably improve flat-area
   geometry.
3. anchor16 (0.6206) still leads — its depth error (1.24%) is the lowest of
   all configs; the gap vs dualring (1.386%) suggests its specific placement is
   still better than a clean two-band split, or that part of its advantage is
   trajectory luck. Dual-ring is the best *designed* scheme so far.
4. Orientation split (new metric): dualring accuracy is worst at azimuth
   135..180 (2.40 vox) and −180..−135 (2.28) vs best 45..90 (1.38) — the
   back-facing degradation seen visually is now quantified (~1.7× spread).
5. burr_rate 0.0274 and hole_fill_rate 0.990 (99% of interior holes filled) are
   the first baselines for the new metrics; hole filling confirms the visual
   "fluffy bridging" — window/lattice gaps are almost entirely filled in.
   coverage_rate 0.389 (mesh within ±2 voxels of GT depth) reflects the same
   surface deviation as the accuracy numbers (±2 voxels is tight; pose error is
   included).

## E2 status — COLMAP PatchMatch external baseline (in progress)

- Windows COLMAP 4.2.1 CUDA binary (414 MB, verified `colmap -h`) crashes with
  STATUS_STACK_BUFFER_OVERRUN (0xC0000409) the moment PatchMatch starts on this
  host — same native-CUDA instability as Blender EEVEE/Cycles-GPU on this box
  (torch CUDA works). Retried at max_image_size 400: same crash. No CPU
  PatchMatch exists (COLMAP MVS is CUDA-only).
- Fallback in progress: building COLMAP 4.2.1 from source inside WSL (CUDA
  toolkit 13.4, sm_89). Dependency chain so far: OpenImageIO (+tools), CGAL,
  OpenCV. Build is running.
- The pipeline side is ready and tested up to the crash point: aligned-pose
  export (`export_aligned_poses.py`, identical Sim(3) as the TSDF protocol),
  COLMAP model writer + per-reference stereo cfg (`colmap_mvs.py`), .bin depth/
  normal parser. Once a working colmap binary exists, E2 runs anchor16 + set32
  through the frozen evaluation.

## Blocked/pending axes

- New-render axes (lighting direction/shadow, defocus/motion blur, texture,
  resolution via re-render) require a working render engine: Cycles GPU init
  crashes (Microsoft C++ exception) and BLENDER_EEVEE access-violates on this
  host's 8 GiB GPU; Cycles CPU ≈ 25 min/view at 800²/64spp is impractical.
  Resolution axis via image downsampling of the dataset views is still possible
  without new renders.
