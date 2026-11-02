# GEOMETRY_OPTIMIZATION — geometry roadmap after P0 / P0.5 / capture group 1

Purpose: which geometry route fits which failure mode, expected gain, cost, and
the trigger signals for switching. Cross-references: capture-side avoidable
issues are in `docs/experiments/CAPTURE_GUIDELINES.md`; raw data in
`docs/experiments/P0_GEOMETRY_OPTIMIZATION.md`,
`docs/experiments/P0.5_RESAMPLING_NORMAL_DIAGNOSIS.md`,
`docs/experiments/CAPTURE_EXPERIMENTS.md`.

## Current best numbers (frozen P0 protocol, 16 original views)

| Geometry | F@1vox | P / R | accuracy (vox) | completeness (vox) |
|---|---:|---:|---:|---:|
| VGGT-Ω depth → TSDF (baseline) | 0.6206 | 0.554 / 0.705 | 1.403 | 0.809 |
| + 2-voxel consistency filter | 0.6927 | 0.780 / 0.623 | 0.676 | 1.217 |
| PGSR 10k → TSDF (negative result) | 0.3324 | 0.308 / 0.361 | 3.182 | 2.201 |
| ExMesh 10k (existing asset) | 0.6190 | 0.615 / 0.624 | 1.107 | 1.069 |
| GT cameras + GT depth (ceiling) | 0.9357 | 0.919 / 0.953 | 0.360 | 0.311 |

Target for the next phase: **F@1vox > 0.70 with recall ≥ 0.705** and lower
normal error. Gap from the full-recall baseline: +0.08 F without losing recall.

## What has been ruled out (measured, this repo)

1. **Resampling / grid alignment**: ~0.1° of the 32.16° normal error (0.3%).
   Nearest remains the correct default; bilinear is slightly harmful at
   silhouettes. (P0.5 Part A)
2. **Fusion-side tricks**: bilinear TSDF sampling, bilateral depth filtering
   (σc 0.02–0.10), Huber robust fusion, and their combination are all within
   −0.01 F of baseline; combined is worst (0.6102). The depth deviation is
   view-consistent, so multi-view averaging cannot remove it. (P0.5 Part B)
3. **Small-window depth smoothing**: the warp is mid-frequency (≈20–100 px
   wavelength, sub-percent amplitude); 5 px bilateral windows cannot touch it,
   and pixel roughness is already lower than GT (0.0096% vs 0.067%).
4. **More views on one ring**: saturates by 32 views and reverses by 64
   (F 0.5064/0.5255/0.5352/0.4940 at 8/16/32/64). Thin-structure precision
   *degrades* with more views (0.28→0.19). (capture group 1)
5. **PGSR 10k in the 16-view configuration**: depth 1.24%→3.14%, normal
   32.16°→44.34°, F 0.3324. Retained as a negative result, not a fix. (P0)

## Failure-mode map

| Failure mode | Signature measured | Root cause | Candidate fix |
|---|---|---|---|
| Mid-frequency surface warp ("dunes", normal 32°) | smooth depth, 1.24% rel err, view-consistent | VGGT-Ω local shape bias | P1 RGB-guided depth refinement; E2 PatchMatch external baseline |
| Melted thin structures (posts, studs, holes) | sharp-region accuracy 2–3 vox, P_sharp 0.19–0.28 | depth bias at high-curvature areas + TSDF thickening | same as above; capture diversity helps recall only |
| Pose/depth coupling | GT-pose+pred-depth (0.6021) ≈ pred-pose+pred-depth (0.6206) | entangled prediction | joint small refinement, not independent pose fixes (P0) |
| Precision-vs-completeness trade | 2-voxel filter: F 0.6927 but recall 0.623 | consistency filtering deletes surface | adaptive per-pixel confidence, not harder thresholds |

## Route comparison

| Route | Expected gain | Cost | When to switch |
|---|---|---|---|
| P1: multi-view photometric depth refinement (Omega depth as prior, ZNCC/SSIM + normal/depth-prior regularisation) | directly targets the measured warp; literature (EG 2025) supports the formulation | medium (custom optimisation loop) | **main line, start now** |
| E2: COLMAP PatchMatch with VGGT poses → depth → same TSDF/eval | independent multi-view measurement; answers "is RGB matching better than Omega depth" | low (existing tools) | run in parallel; if it beats Omega depth on the same views, prioritise depth fusion by confidence |
| URGT (CVPR 2026) refinement | normal-error oriented refiner; needs gauge handling for z-depth | medium | after E2, if depth-refinement direction validates |
| GeoSVR / ADELE | stronger geometry representation (sparse voxels / adaptive tetrahedra) | high | only if depth-side fixes plateau below target |
| ExMesh continuation | bounded by initial geometry quality | medium | not before depth improves; 10k already at F 0.6190 |
| More PGSR tuning | negative in this configuration | — | do not retry without changing view coverage/geometry supervision |

## Interaction with capture (cross-reference)

- Some "geometry" problems are capture problems: single-ring oversampling
  *hurts* precision (group 1). Before tuning geometry, apply
  CAPTURE_GUIDELINES.md: elevation-diverse views, ~12° ring spacing, top views
  of flat planes.
- The mid-frequency warp is a VGGT property under *all* tested capture schemes
  (8–64 views); it is not avoidable by capture. Capture controls coverage and
  depth-error magnitude (2.06% ↔ 1.24%), not the warp.
