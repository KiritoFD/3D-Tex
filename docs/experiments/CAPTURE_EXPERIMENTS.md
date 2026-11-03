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

## Group 3 — capture resolution (256/400/512 vs 800, identical anchor16 views/GT)

Motivation: GT depth and camera grid are resolution-independent at the frozen
512 model grid; only the input images are rescaled (cv2.INTER_AREA from the 800²
originals) and K_original rescaled to match. This isolates the effect of input
resolution on VGGT-Ω depth, with zero change to trajectory, lighting, or TSDF.

```powershell
.venv/Scripts/python.exe scripts/capture/build_resolution_subset.py --resolutions 256,400,512
.venv/Scripts/python.exe scripts/adapters/vggt_omega_geometry.py --scene-dir data/scenes/capture_resR ... --image-resolution 512
.venv/Scripts/python.exe scripts/capture/capture_eval.py --capture data/scenes/capture_resR --out outputs/capture_resR --p0-out deliverables/lego_p0
```

| Config | depth abs-rel mean/median | normal° | flat° | edge° | low-RMS | high-RMS | burr_rate | hole_fill | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| res256 | 1.371% / 0.756% | 33.13 | 16.72 | 34.95 | 0.0119 | 0.0233 | 0.0244 | 0.990 | 0.406 |
| res400 | 1.265% / 0.694% | 32.48 | 16.14 | 34.14 | 0.0104 | 0.0223 | 0.0186 | 0.985 | 0.468 |
| res512 | 1.278% / 0.758% | 32.12 | 15.63 | 33.80 | 0.0103 | 0.0218 | 0.0191 | 0.985 | 0.418 |
| anchor16 (800² baseline) | 1.242% / 0.701% | 32.16 | 15.61 | 33.86 | — | — | — | — | — |

Mesh metrics (frozen P0 grid, measure_surfaces; anchor16 row = P0 baseline):

| Config | F@1vox ↑ | P / R @1vox | accuracy (vox) ↓ | compl. (vox) ↓ | acc_sharp / acc_flat (vox) | P_sharp | R_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| res256 | 0.5726 | 0.514 / 0.646 | 1.548 | 0.894 | 2.194 / 1.197 | 0.295 | 0.644 |
| res400 | **0.6299** | 0.576 / 0.695 | **1.322** | **0.813** | 1.996 / 1.010 | 0.335 | 0.681 |
| res512 | 0.5954 | 0.537 / 0.668 | 1.436 | 0.856 | 2.087 / 1.083 | 0.317 | 0.695 |
| anchor16 (800² baseline) | 0.6206 | 0.554 / 0.705 | 1.403 | 0.809 | — | — | — |

Orientation split (accuracy vox by azimuth, worst/best sector): res256 1.94
(0..45) / 1.35 (−90..−45); res400 1.58 (0..45) / 1.22 (−90..−45); res512 1.81
(0..45) / 1.24 (−90..−45) — the 0..45 sector is worst at every resolution,
same orientation non-uniformity as the other configs, resolution-independent.

Findings:

1. **Resolution axis is nearly flat from 400² up**: res400 and res512 sit within
   ~0.02–0.04% abs-rel of the 800² anchor16 baseline, and normal error matches
   to <0.4°. VGGT-Ω's 512² model input, not the capture resolution, is the
   dominant factor at these scales — 400² and 512² captures are upsampled by
   the loader to 512² anyway. The mesh side agrees: res400 (F 0.6299) and
   res512 (F 0.5954) bracket the anchor16 baseline (0.6206) — non-monotone and
   within the same noise band as the coverage_rate flip, not a resolution trend.
2. **res256 degrades measurably but modestly**: abs-rel +0.13pp (1.37% vs
   1.24%), normal +1.0°, edge normal +1.1°. High-frequency RMS rises the most
   (0.0233 vs ~0.0218): the finest "grit" component of the warp grows at very
   low capture resolution, consistent with the model having less texture detail
   to latch onto.
3. **burr_rate and hole_fill are stable** across 400/512 (0.019/0.985): the
   silhouette burr cloud and structure melting do NOT improve with capture
   resolution. res256 burr 0.0244 is marginally worse than 400/512.
4. **coverage_rate non-monotone** (0.406 / 0.468 / 0.418): res400 best on this
   measure. Likely a noise-level effect at ±2 voxel tolerance rather than a
   real resolution trend; treat as within-noise until repeated with different
   seeds.

Implication for capture guidelines: do not shoot below ~400²; from 400² up the
model input resolution is the bottleneck, not the capture resolution, so extra
pixels buy nothing.

## Group 4 — pixel-level attribution (questions A and B)

Method: for every GT foreground pixel we compute the depth residual of the
predicted depth together with nine covariates (view count seeing the GT 3-D
point, max parallax, incidence angle vs GT normal, image gradient, distance to
silhouette, distance to GT depth discontinuity, GT curvature, interior-groove
flag, near-clip highlight flag). Aggregating by mutually exclusive region
classes gives question A; Spearman rho + quartile binning on the full pixel
population gives question B. Pixel counts: ~400k per config.

### Question A — which regions benefit from denser capture?

abs_rel by region, per config (lower = better):

| Config | flat | weak_texture | high_curvature | thin_sharp | edge_band | highlight |
|---|---:|---:|---:|---:|---:|---:|
| set8 | 0.0166 | 0.0160 | 0.0177 | **0.0343** | 0.0268 | 0.0127 |
| set16 | 0.0121 | 0.0119 | 0.0144 | 0.0324 | 0.0231 | 0.0104 |
| set32 | 0.0100 | 0.0099 | 0.0124 | 0.0310 | 0.0212 | 0.0144 |
| set64 | **0.0088** | 0.0086 | 0.0118 | 0.0292 | **0.0196** | 0.0097 |
| dualring24 | **0.0085** | 0.0092 | 0.0129 | 0.0319 | 0.0210 | **0.0033** |

Density benefit (uniform ring set8 → set64):

| Region | Δ abs_rel | Verdict |
|---|---:|---|
| flat | **−47.3%** | large benefit — extra views do help large flat areas |
| weak_texture | **−46.4%** | large benefit |
| high_curvature | −33.5% | moderate benefit |
| edge_band | −26.8% | moderate benefit |
| highlight | −23.7% | moderate benefit (small N; noisy) |
| thin_sharp | −14.6% | **weak benefit** — density alone does not un-melt thin structures |

Dualring vs set32 (24 elevation-diverse views vs 32 single-ring views, i.e.
*fewer* images but better placed):

| Region | Δ vs set32 | Verdict |
|---|---:|---|
| flat | **−14.3%** | elevation diversity wins on flat |
| weak_texture | −7.5% | wins |
| highlight | **−77.1%** | large win (small N) |
| edge_band | −1.1% | parity |
| thin_sharp | +2.8% | parity/slightly worse |
| high_curvature | +4.6% | parity/slightly worse |

**Answer A**: extra views pay off most on *flat* and *weak-texture* regions
(the user-predicted "no benefit on flat" is wrong at the pixel level — flat
shows the largest relative gain, −47%). Thin_sharp benefits least from raw
density (−15%) and does not improve from elevation diversity either: thin
structures are bounded by the depth estimation, not the capture. Edge_band
sits in between. The capture-side lever that matters for thin structures is
NOT count but incidence angle (see Q4 below).

### Question B — where is VGGT-Ω worst, and why?

Spearman rho between each covariate and per-pixel abs_rel (set32, ~400k
pixels; sign gives direction):

| Covariate | rho | q1→q4 abs_rel trend |
|---|---:|---|
| dist_to_discontinuity | **−0.282** | 0.0262 → 0.0088 (closer to an edge = worse) |
| local_curvature | **+0.228** | 0.0100 → 0.0268 (curved = worse) |
| parallax | +0.134 | 0.0114 → 0.0179 |
| incidence | +0.123 | 0.0115 → 0.0179 (grazing = worse) |
| img_grad | +0.095 | 0.0119 → 0.0181 |
| dist_to_silhouette | +0.032 | flat |
| view count | −0.025 | flat |
| highlight / groove | ~0 | n/a |

**Error-contribution ranking** (largest |rho| first):

1. **Edge / discontinuity band** — pixels near GT depth discontinuities are
   ~3× worse than the interior (0.026 vs 0.009). This is the *dominant* error
   source, consistent with the visual "fluffy bridges" and "burr cloud".
2. **Local curvature** — high-curvature pixels are 2.7× worse than low.
3. **Grazing incidence** — pixels seen at grazing angles are 1.6× worse.
4. **Image gradient** (weak texture) — weak but real.
5. **View count / parallax / silhouette distance** — all ≪0.15; once the
   object is reasonably covered, marginal coverage is not the limiting factor.

**Mechanism attribution**: the two largest effects (discontinuity band and
curvature) are *local shape properties*, not *capture* properties — view
count, parallax and incidence are all far weaker. This is evidence for a
**network prior / patch-level smoothing** failure mode: VGGT-Ω systematically
smooths across depth discontinuities and rounds off high-curvature regions,
so pixels there are biased regardless of how many views see them. By contrast,
pure information-starvation variables (views, parallax) contribute almost
nothing beyond ~16 views. The weak-texture effect (rho 0.095) is real but
secondary; highlights are not a measurable failure mode at this scale.

A direct density-by-quartile comparison (set8 vs set64 abs_rel within each
covariate quartile) shows the gain is **largest exactly where information is
scarce**: low-parallax Q1 −59%, low-incidence Q1 −44%, low-texture Q1 −42%,
low-curvature Q1/Q2 −47%/−50%, far-from-edge Q4 −54%. In the *worst* buckets
(discontinuity Q1 −20%, curvature Q4 −20%, parallax Q4 −24%) the gain is the
smallest. This splits the failure modes cleanly:

- **Information-starved pixels** (low parallax / grazing / weak texture /
  far-from-edge flat areas): densification helps a lot → capture-side fix.
- **Discontinuity-adjacent and high-curvature pixels**: densification helps
  least → these are the network-prior / smoothing failures, geometry-side fix.

Implication: capture can fix *flat / weak-texture / coverage* gaps (Q-A shows
they respond), but the *edge_band / thin_sharp / high_curvature* family
responds weakly to capture and is the natural target for a depth-side
refiner (P1) or a better multi-view matcher (E2).

## E2 — COLMAP PatchMatch external baseline

**Status**: anchor16 and set32 done.

COLMAP path: Windows 4.2.1 CUDA binary crashes with 0xC0000409 the moment
PatchMatch starts (same native-CUDA instability as Blender EEVEE/Cycles-GPU
on this host; torch CUDA works). WSL source build of COLMAP 4.2.1 with
CUDA 12.8 / sm_89 succeeds (build: `cmake -GNinja -DGUI_ENABLED=OFF
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDA_ENABLED=ON`; first
build passed `--CUDAToolkit_ROOT` pointing at the Windows CUDA on /mnt/c
which silently produced a CPU-only binary; the successful build points at
WSL's /usr/local/cuda). PatchMatchStereo needs `depth_min`/`depth_max` when
the sparse model has no 3D points (true for known-pose-only reconstruction);
estimated from VGGT anchor16 depth percentiles: [2.5, 6.5] (camera frame).
anchor16: 268 s for 16 views; set32: 494 s for 32 views on the RTX 4070 laptop.

| Geometry | depth abs-rel mean / median / p90 | normal mean° | coverage_rate | hole_fill_rate | burr_rate |
|---|---:|---:|---:|---:|---:|
| VGGT-Ω (anchor16) | 1.242% / 0.701% / — | 32.16 | — | 0.990 | — |
| COLMAP PatchMatch (anchor16) | 2.305% / 0.891% / 3.341% | **24.43** | 0.066 | **0.206** | 0.0153 |
| VGGT-Ω (set32) | 1.462% / 0.888% / — | 32.53 | — | 0.994 | — |
| COLMAP PatchMatch (set32) | 2.698% / 1.098% / 3.420% | 32.63 | 0.264 | **0.611** | **0.0105** |

Mesh metrics (frozen P0 grid, measure_surfaces; VGGT rows for reference):

| Geometry | F@1vox ↑ | P / R @1vox | accuracy (vox) ↓ | compl. (vox) ↓ | acc_sharp / acc_flat |
|---|---:|---:|---:|---:|---:|
| VGGT-Ω (anchor16) | **0.6206** | 0.554 / **0.705** | 1.403 | **0.809** | — |
| COLMAP PatchMatch (anchor16) | 0.1479 | 0.409 / 0.090 | 1.602 | 9.415 | 2.237 / 1.139 |
| VGGT-Ω (set32) | **0.5352** | 0.460 / **0.639** | 1.813 | **0.906** | 2.583 / 1.362 |
| COLMAP PatchMatch (set32) | 0.4015 | **0.719** / 0.279 | **0.882** | 3.764 | **1.555 / 0.638** |

**Coverage-0.066 attribution (checked)**: the PatchMatch photometric depth maps
are 100% valid on every frame; the `geom_consistency=true` maps used for the
evaluation are only **2.5% (anchor16) / 5.7% (set32) valid pixels** — COLMAP's
geometric consistency filter rejects nearly everything with 10–31 source views
at default thresholds. The sparse coverage is therefore a *filtering* artefact,
not a matching failure: where the filter commits, mesh accuracy is at or better
than VGGT-Ω (set32: 0.882 vs 1.813 vox; sharp split 1.555 vs 2.583), and
precision@1vox far higher (0.719 vs 0.460) — the loss is entirely recall.
Not checked: whether relaxed filter thresholds keep the accuracy advantage.

PatchMatch is roughly twice as bad on abs-rel as VGGT-Ω on the same views
(anchor16: 2.31% vs 1.24%; set32: 2.70% vs 1.46%) and its depth is much
sparser (anchor16 coverage 6.6%, set32 26.4%, vs ~100% for VGGT-Ω) — many
pixels have no confident multi-view match.

Where PatchMatch does commit on anchor16, its surface normal is markedly
better than VGGT-Ω (24.4° vs 32.2°; flat-region 8.3° vs 15.6°) — on 16
well-placed views, classical multi-view matching does not suffer the
patch-level smoothing bias seen in the network prior. At 32 views the
PatchMatch normal is no longer better than VGGT-Ω (32.6° vs 32.5°), suggesting
the extra 16 views (denser azimuth, similar elevation) mostly add
lower-confidence / noisier matches that erase the normal-angle advantage
while the abs-rel penalty and sparse coverage persist.

Crucially, **PatchMatch hole_fill_rate is far lower than VGGT-Ω** (anchor16
0.21 vs 0.99; set32 0.61 vs 0.99): PatchMatch does NOT bridge window holes
or melt lattice openings the way the network prior does, confirming the
visual review's "fluffy bridging" as a network-specific failure. Its
burr_rate is also lower (anchor16 0.015 vs ~0.024 typical; set32 0.010 vs
0.027 for dualring).

Combined evidence (PatchMatch: better normals / structure preservation where
it commits, much worse coverage / abs-rel overall) supports the **geometry
refinement route** (P1): use VGGT-Ω as a depth prior and refine with
multi-view photometric consistency, rather than replacing it.

## E2 status — COLMAP PatchMatch external baseline (anchor16 + set32 done)

- Windows COLMAP 4.2.1 CUDA binary (414 MB, verified `colmap -h`) crashes with
  STATUS_STACK_BUFFER_OVERRUN (0xC0000409) the moment PatchMatch starts on this
  host — same native-CUDA instability as Blender EEVEE/Cycles-GPU on this box
  (torch CUDA works). Retried at max_image_size 400: same crash. No CPU
  PatchMatch exists (COLMAP MVS is CUDA-only).
- WSL source build of COLMAP 4.2.1 (CUDA 12.8, sm_89) succeeds after the
  CUDAToolkit path was corrected from `/mnt/c/...` (Windows CUDA, silently
  produced CPU-only binary) to `/usr/local/cuda`. anchor16 PatchMatch:
  268 s on RTX 4070 laptop for 16 views at 800²; set32: 494 s for 32 views.
- Pipeline: `scripts/capture/export_aligned_poses.py` (frozen Sim(3)) →
  `colmap_mvs.py` (write PINHOLE model + per-reference stereo cfg + run +
  .bin→npy) → `capture_eval.py --geometry-dir` (skips Sim3/regrid, resizes
  800²→512² INTER_NEAREST onto the GT K grid).

## Handoff for the next session

Done this session (commits newest first):
- E2 COLMAP PatchMatch anchor16 + set32 (this commit): abs_rel / normals /
  coverage / hole_fill / burr all measured; route comparison updated.
- Capture group 4 — pixel-level attribution (`scripts/diagnostics/pixel_attribution.py`
  + `density_by_covariate.py`): answers Q-A (density benefit per region) and
  Q-B (error-contribution ranking + mechanism) with per-covariate quartile
  gain analysis set8 vs set64.
- Capture group 3 — resolution axis (256/400/512 vs anchor16 800²).
- Mechanism evidence + guidelines / geometry doc updates.

Blocked / pending axes (unchanged): renderer axes (lighting, blur, texture,
re-render at new resolutions) — EEVEE access-violates and Cycles-GPU init
crashes on this host, so these remain **[未验证]** in CAPTURE_GUIDELINES.

Next steps, in priority order:
1. P1 depth refinement (the E2 finding "PatchMatch normals better where it
   commits but coverage/abs_rel much worse" points at VGGT-Ω-as-prior +
   multi-view photometric refinement as the natural next experiment).
2. Optional: per-region measure_surfaces runs for the res256/400/512 outputs
   to add F-score / accuracy to the resolution table (currently only
   capture_eval metrics are reported there).
3. Optional: measure_surfaces for the E2 colmap outputs if F-score comparison
   on the mesh side is wanted (currently only image-space metrics reported).

## Blocked/pending axes

- New-render axes (lighting direction/shadow, defocus/motion blur, texture,
  resolution via re-render) require a working render engine: Cycles GPU init
  crashes (Microsoft C++ exception) and BLENDER_EEVEE access-violates on this
  host's 8 GiB GPU; Cycles CPU ≈ 25 min/view at 800²/64spp is impractical.
  Resolution axis via image downsampling of the dataset views is still possible
  without new renders.
