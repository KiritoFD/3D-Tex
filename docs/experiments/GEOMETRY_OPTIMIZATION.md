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

0. **Voxel resolution and the TSDF pipeline are NOT the bottleneck** (visual
   review, independent session): GT depth through the identical TSDF/voxel/eval
   pipeline reproduces the stud array, open window holes and thin wheel spokes,
   while predicted depth melts all of them on the same grid. The defect lives
   in the predicted depth values themselves, so geometry work must change the
   depth estimation, not the meshing. Structure survival (holes open, studs
   present, spokes connected) is adopted as a switching criterion alongside
   F-score — a method is only an improvement if it raises F-score without
   lowering structure survival.
1. **Resampling / grid alignment**: ~0.1° of the 32.16° normal error (0.3%).
   Nearest remains the correct default; bilinear is slightly harmful at
   silhouettes. (P0.5 Part A)
2. **Fusion-side tricks**: bilinear TSDF sampling, bilateral depth filtering
   (σc 0.02–0.10), Huber robust fusion, and their combination are all within
   −0.01 F of baseline; combined is worst (0.6102). The depth deviation is
   view-consistent, so multi-view averaging cannot remove it. (P0.5 Part B)
3. **Small-window depth smoothing**: the warp is two superimposed scales — a
   mid-frequency "dune" component (~20–100 px / 1/8–1/4 vehicle width) plus a
   high-frequency "grit" grain; 5 px bilateral windows cannot touch the dunes,
   and pixel roughness is already lower than GT (0.0096% vs 0.067%).
4. **More views on one ring**: saturates by 32 views and reverses by 64
   (F 0.5064/0.5255/0.5352/0.4940 at 8/16/32/64). Thin-structure precision
   *degrades* with more views (0.28→0.19), and the silhouette burr cloud grows
   from a skirt at 8 views to body-wide spikes at 64 — densification trades
   silhouette quality for coverage. (capture group 1 + visual review)
5. **PGSR 10k in the 16-view configuration**: depth 1.24%→3.14%, normal
   32.16°→44.34°, F 0.3324. Retained as a negative result, not a fix. (P0)

## Failure-mode map

| Failure mode | Signature measured | Root cause | Candidate fix |
|---|---|---|---|
| Edge / discontinuity band smoothing | per-pixel abs_rel 0.026 (band) vs 0.009 (interior); Spearman rho −0.28 with dist-to-edge — *largest* error source | VGGT-Ω patch-level smoothing bleeds across depth discontinuities; capture-side fixes (more views) barely help | ~~P1 dense photometric refinement~~ (measured negative); gated sparse multi-view update (E2-style) is the remaining candidate |
| High-curvature rounding | abs_rel rho +0.23 with GT curvature; curvature q4 is 2.7× q1 | network prior smooths curved regions | same as above |
| Mid-frequency surface warp ("dunes", normal 32°) | smooth depth, 1.24% rel err, view-consistent | VGGT-Ω local shape bias | ~~P1 dense photometric refinement~~ (measured negative — dunes are photometrically invisible on flat plastic); representation-side routes |
| Melted thin structures (posts, studs, holes) | sharp-region accuracy 2–3 vox, P_sharp 0.19–0.28; density benefit only −15% from 8→64 views | depth bias at high-curvature areas + TSDF thickening; capture density alone cannot un-melt | same as above; capture diversity helps recall only |
| Pose/depth coupling | GT-pose+pred-depth (0.6021) ≈ pred-pose+pred-depth (0.6206) | entangled prediction | joint small refinement, not independent pose fixes (P0) |
| Precision-vs-completeness trade | 2-voxel filter: F 0.6927 but recall 0.623 | consistency filtering deletes surface | adaptive per-pixel confidence, not harder thresholds |

## Region-split evidence (capture vs geometry responsibility)

Pixel-level attribution over ~400k GT pixels per config (group 4):

| Region | abs_rel (set32) | Density benefit 8→64 | Fix lives in |
|---|---:|---:|---|
| flat | 0.0100 | **−47%** | capture (densify) |
| weak_texture | 0.0099 | **−46%** | capture (densify) |
| highlight | 0.0144 | −24% (small N) | capture (avoid highlights) |
| high_curvature | 0.0124 | −33% | mixed |
| edge_band | 0.0212 | −27% | **geometry** (smoothing prior) |
| thin_sharp | 0.0310 | **−15%** | **geometry** (smoothing prior) |

Regions above the "geometry" line barely respond to capture density; they are
the natural target of P1 (depth refinement) and the benchmark for E2 (if a
classical matcher beats VGGT-Ω there, the smoothing is a model property, not
an information limit).

## Route comparison

| Route | Expected gain | Cost | When to switch |
|---|---|---|---|
| P1: multi-view photometric depth refinement (Omega depth as prior, ZNCC/SSIM + normal/depth-prior regularisation) | **measured negative** (P1_DEPTH_REFINEMENT.md): every config degrades geometry (best F 0.598 vs 0.621, normals +7° to +18°) while improving ZNCC past the GT ceiling; photo gain is anti-correlated with depth gain (spearman +0.09) and a plane-sweep argmin has no skill (spearman +0.05, worse than prior in 78% of pixels). Cause: glossy weakly-textured synthetic appearance makes dense photometric consistency nearly geometry-free | measured (done) | ruled out as formulated; the remaining gated-sparse variant was also measured neutral (E2-followup row) — photometric depth-value direction closed |
| E2: COLMAP PatchMatch with VGGT poses → depth → same TSDF/eval | **measured**: abs-rel 2× worse (2.31% vs 1.24% anchor16; 2.70% vs 1.46% set32), coverage 6.6–26% (vs ~100%), but normals better at 16 views and hole_fill much lower (0.21 vs 0.99 anchor16) — does NOT melt windows/lattice; mesh side: where the geometric-consistency filter commits (only 2.5–5.7% of pixels), set32 accuracy 0.882 vox vs VGGT 1.813, precision 0.719 vs 0.460 — recall is the entire loss | measured (done) | measured; motivated the gated-sparse-update, which was then measured neutral (E2-followup row) |
| URGT (CVPR 2026) refinement | normal-error oriented refiner; needs gauge handling for z-depth | medium | after E2, if depth-refinement direction validates |
| E2-followup: gate threshold sweep + sparse gated fusion (VGGT base, COLMAP depth only on gate pixels) | **measured neutral** (P1_DEPTH_REFINEMENT.md Follow-up A): knee at cost2_nc2 (3.7% coverage @ 2.34% abs_rel, collapse at cost3 → 14.8% @ 10.8%); fusion at trustworthy gates (≤4.4% px) is exactly neutral (F 0.619–0.621 vs 0.621), looser gates monotonically harmful (F down to 0.48); COLMAP abs_rel > VGGT at every level, and 3–4% of pixels is too sparse to move the TSDF surface | measured (done) | closed — depth-value photometric direction fully ruled out |
| **B: Mesh Splatting (ICLR 2026) with added geometric supervision** | representation-side route now that the depth-value direction is closed; DMTet stage permits topology changes (the only tested route family that can re-open window holes / re-connect rods); RGB+normal losses must be augmented with depth/silhouette/free-space anchors to avoid repeating the P1 failure | medium-high (plan below; feasibility probe done — **no public code found**, re-check before starting) | **next main line** |
| GeoSVR / ADELE | stronger geometry representation (sparse voxels / adaptive tetrahedra) | high | only if depth-side fixes plateau below target |
| ExMesh continuation | bounded by initial geometry quality | medium | not before depth improves; 10k already at F 0.6190 |
| More PGSR tuning | negative in this configuration | — | do not retry without changing view coverage/geometry supervision |

## Phase B plan — Mesh Splatting with geometric anchors (for the next session)

Paper: **"Mesh Splatting for End-to-end Multiview Surface Reconstruction"**
(Zhang, Wu, Chen; ICLR 2026 Poster; arXiv 2601.21400; OpenReview PSgps4JXTb).
Multi-layer semi-transparent mesh "softening" differentiable w.r.t. a base
mesh, splatting renderer, ~20 min optimization; **DMTet reparameterization in
the early stage (≈5k steps) for topology control, continuous remeshing later
(≈10k steps)**. NOT the same work as CVPR 2026 "MeshSplatting: Differentiable
Rendering with Opaque Meshes" (Held et al., arXiv 2512.06818,
meshsplatting.github.io) — do not mix them up.

Feasibility probe (2026-11-03): paper and reviews verified; **no public code
repository found** (checked OpenReview page, arXiv listing, author pages and
GitHub search — nothing released as of this date). Before any implementation
work: re-check for an official release; if absent, either email the authors or
treat "reimplement from the paper" as the real cost (renderer + DMTet +
remeshing stack; kaolin/diff-gaussian-rasterization ecosystems cover the
building blocks on paper). Windows compatibility unknown until code exists;
the splatting rasterizers in this ecosystem are CUDA+Linux-first, so WSL is
the likely runtime (same pattern as our COLMAP build). VRAM: 8 GiB is below
the typical default for multi-layer mesh splatting at 800²; budget for
image-resolution reduction (model grid 512² or 256²) and/or fewer layers.

Integration plan (when code/reimplementation exists):
1. Input: VGGT cameras + depth → initial mesh from the EXISTING frozen TSDF
   pipeline (baseline mesh, F 0.6207) as the DMTet-stage initialisation —
   do not train from a primitive shape; the goal is to repair, not to
   rediscover.
2. Required added supervision (their default is RGB + normal losses, which on
   our evidence would reproduce "ZNCC up, geometry down"):
   - **trusted depth anchors**: gate-passing COLMAP pixels (Follow-up A gate
     logic, cost1/cost2 levels) as sparse hard depth constraints;
   - **silhouette loss**: rendered alpha vs image alpha matte (dataset masks,
     no GT geometry involved);
   - **free-space constraint**: rays through the background matte must not
     generate solid surface — directly attacks the burr cloud;
   - **edge-aware normal/curvature regularisation** on the base mesh, to
     protect true creases instead of flattening them.
3. Acceptance: same frozen protocol (capture_eval + measure_surfaces +
   heldout 4-view renders), reporting P and R separately, structure survival
   (stud array / rod & spoke connectivity / window holes open), burr rate —
   per the acceptance criteria in the P1 follow-up. Selection by geometric
   metrics only; no photometric-loss-based early stopping.

## Phase C plan — ADELE-style residual SDF (research plan only, do not start)

Representation: s(x) = s0(x) + λ_res · s_res(x), where s0 is a TSDF fused from
the highest-confidence VGGT depth (confidence-weighted fusion, existing
infrastructure), discretised on a LOW-resolution SDF grid; s_res is a learned
residual field with a small initial coefficient, optimised coarse-to-fine.
Two-stage supervision: volume rendering first (density well-posed early),
mesh rendering later (surface refinement). Motivation vs Phase B: the residual
parameterisation keeps the good prior exactly and localises where the model
may deviate, instead of re-optimising the whole surface — conceptually the
SDF-space version of the P1 residual idea, but with the multiplicative-failure
lessons applied (small λ_res, coarse-to-fine, explicit anchors).
Status: **ADELE official code is NOT released** (as of 2026-11-03); this is a
self-implementation plan, high cost. Trigger to switch from B to C: Phase B
cannot recover structure either (studs/window holes/rods still dead after
DMTet+remeshing with anchors), or B is blocked by the missing codebase.

## Interaction with capture (cross-reference)

- Some "geometry" problems are capture problems: single-ring oversampling
  *hurts* precision (group 1). Before tuning geometry, apply
  CAPTURE_GUIDELINES.md: elevation-diverse views, ~12° ring spacing, top views
  of flat planes.
- The mid-frequency warp is a VGGT property under *all* tested capture schemes
  (8–64 views); it is not avoidable by capture. Capture controls coverage and
  depth-error magnitude (2.06% ↔ 1.24%), not the warp.
