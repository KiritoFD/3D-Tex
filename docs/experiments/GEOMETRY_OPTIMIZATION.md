# GEOMETRY_OPTIMIZATION — geometry roadmap after P0 / P0.5 / capture group 1

**2026-11-03 local execution:** corrected M2 and voxel-tolerance priors now run in
WSL; diagnostic F reaches 0.73685 and a separately initialised native prediction
control reaches 0.72687. Both worsen mesh normal error and still close most holes,
so neither replaces the adopted reference. See [complete results and provenance](local_next/README.md).
The official ICLR Mesh Splatting CUDA component compiled and passed a small
forward/backward probe; full DTU/DMTet/remeshing reproduction is still pending.
Historical burr values use the pre-fix vertical projection and are invalid;
this does not invalidate the distance/recall metrics.

Smooth lattice displacement has also been measured: F 0.7024–0.7041 with lower
mesh normal error, but sharp recall drops 0.6935→0.664–0.666. It is not adopted.
See [lattice tradeoffs](local_next/LATTICE_AND_CAPTURE.md) and the
[capture/alignment audit](local_next/CAPTURE_RESULTS.md); mesh scores from
different Sim3 fitting sets must not be mixed.

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
| **v35c30 carve + confidence (adopted default, "无代价档")** | **0.6315** | 0.581 / 0.692 | 1.350 | 0.837 |
| + 2-voxel consistency filter | 0.6927 | 0.780 / 0.623 | 0.676 | 1.217 |
| PGSR 10k → TSDF (negative result) | 0.3324 | 0.308 / 0.361 | 3.182 | 2.201 |
| ExMesh 10k (existing asset) | 0.6190 | 0.615 / 0.624 | 1.107 | 1.069 |
| GT cameras + GT depth (ceiling) | 0.9357 | 0.919 / 0.953 | 0.360 | 0.311 |

**Adopted reference for all downstream experiments: v35c30+conf** (free-space
carve at vote_frac 0.35 / confirm_max 0.3 + confidence-weighted fusion;
P1_DEPTH_REFINEMENT.md Follow-up C). It is the *no-cost* operating point —
F +0.011, P and R both ≥ old baseline, burr −21%, heldout IoU +0.001 — not the
max-F point. Deliberate aggressive alternatives, full trade table in Follow-up C:
v25c40+conf (F 0.6365, P 0.600 / R 0.678) and v15c50+conf (F 0.6398, P 0.617 /
R 0.664, burr −39%, hole_fill 0.753). The 2-voxel filter stays the max-F /
min-recall option.

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
| Tiling probe + mixed-trajectory fusion (Follow-up D) | **probe positive, fusion negative**: structure loss is effective-resolution (crop/mixed runs restore stud arrays, rims, spokes; edge normals −6°); mixed full+close-up sets stay anchored (0.048) while crop-only sets break trajectory (×8). BUT fusing crop depth into the voxel TSDF fails in all four rules tried (coexist 0.5775 / adjudicate 0.5653 / replace 0.4969 / internally-consistent 0.4886 vs 0.6315 target): crop depth's absolute placement resists sub-voxel registration (p90 23% disagreement, not pose-shaped) | measured (done) | capture-side: adopt mixed full+close-up capture; fusion-side: closed — crop depth is an anchor source for Phase B, not TSDF input |
| **Geometry-side supervision, no photometric loss (Follow-up C)** | **measured working trade**: cross-view free-space carving (votes+confirms, aligned gauge) + VGGT confidence-weighted fusion. Monotone trade curve: F 0.6207→0.6398, P 0.554→0.617, R 0.705→0.664, acc 1.403→1.230 vox, burr −39%, window holes genuinely open (hole_fill 0.985→0.753), P_sharp 0.326→0.371; deleted pixels ~2× enriched for wrong geometry, completeness pays proportionally; near-free point v35c30+conf (F 0.6315, burr −21%, IoU +0.001). Crater untouched; studs/rods not recoverable by deletion | measured (done) | adopt v35c30+conf as default fusion; aggressive points documented |
| **B: Mesh Splatting (ICLR 2026) with added geometric supervision** | representation-side route now that the depth-value direction is closed; DMTet stage permits topology changes (the only tested route family that can re-open window holes / re-connect rods); RGB+normal losses must be augmented with depth/silhouette/free-space anchors to avoid repeating the P1 failure | medium-high (plan below; feasibility probe done — **no public code found**, re-check before starting) | **next main line** |
| GeoSVR / ADELE | stronger geometry representation (sparse voxels / adaptive tetrahedra) | high | only if depth-side fixes plateau below target |
| ExMesh continuation | bounded by initial geometry quality | medium | not before depth improves; 10k already at F 0.6190 |
| More PGSR tuning | negative in this configuration | — | do not retry without changing view coverage/geometry supervision |

## Phase B plan — Mesh Splatting with geometric anchors (feasibility decomposition, 2026-11-03)

Paper: **"Mesh Splatting for End-to-end Multiview Surface Reconstruction"**
(Zhang, Wu, Chen; ICLR 2026 Poster; arXiv 2601.21400; OpenReview PSgps4JXTb).
Multi-layer semi-transparent mesh "softening" differentiable w.r.t. a base
mesh, splatting renderer, ~20 min optimization; DMTet early (≈5k steps),
continuous remeshing later (≈10k). NOT the same work as CVPR 2026
"MeshSplatting: Differentiable Rendering with Opaque Meshes" (Held et al.,
arXiv 2512.06818, meshsplatting.github.io).

**Feasibility probe: no public code found** (OpenReview, arXiv, author pages,
GitHub — nothing released as of 2026-11-03). Therefore the plan below targets a
**minimal viable subset** — "multi-layer softening + geometric anchor
supervision on our existing TSDF mesh" — not a replication of the full
DMTet+remeshing stack, and no big-repo clones or environment changes before
the feasibility questions are answered.

### Component decomposition (minimal subset)

| # | Component | What it must do | Existing alternative in this repo / ecosystem | Verdict |
|---|---|---|---|---|
| 1 | Differentiable renderer for a triangle mesh (silhouette + depth, no shading needed) | rasterize mesh to per-view alpha + depth with gradients w.r.t. vertex positions | nvdiffrast (Windows wheels exist, used by many mesh-optim repos); pytorch3d SoftSilhouetteRasterizer (pure torch, installs on Windows); as a last resort a custom splat rasterizer (torch scatter, the mesh is only ~400k tris) | **viable** |
| 2 | Multi-layer softening | turn the base mesh into K semi-transparent layers at offsets δ_k along vertex normals, blend weights by distance to base mesh | ~100 lines of torch on top of component 1 (layers = K renders of the same mesh with vertex offsets) | **viable** |
| 3 | Anchor losses (no RGB) | (a) trusted depth anchors: gate-passing COLMAP pixels (Follow-up A gate, cost1/cost2) and/or mixed-run crop depth in structure regions (Follow-up D) as hard depth constraints; (b) silhouette loss: rendered alpha vs dataset alpha matte; (c) free-space: penalty on any rendered occupancy along background rays (reuses geometry_pruning votes, now on the differentiable side); (d) edge-aware normal/curvature regulariser on the base mesh | all data already produced (gates, masks, vote maps) | **viable** |
| 4 | Optimisation loop | Adam on vertex positions (optionally per-vertex scalar offset along normals), 400k verts; a few hundred iterations at 512², 16 views | refine_depth.py already runs this scale (peak <2 GiB) | **viable** |
| 5 | Evaluation | frozen protocol (capture_eval + measure_surfaces + heldout renders); selection by geometric metrics only | exists | **viable** |
| 6 | Deferred: DMTet reparameterisation + remeshing (topology change) | only needed if the crater/studs need actual topology surgery | kaolin DMTet (pip, Linux-first; WSL path same as COLMAP); decide after the minimal subset result | deferred |

VRAM budget: 16 views × 512² depth/alpha render at ~400k tris ≪ 2 GiB per
layer set; K=3–5 layers still fits 8 GiB. Runtime estimate: minutes per run.

### Time-box and give-up conditions

- **M1 (spike)**: components 1–2 working — render the baseline mesh
  differentiably at the 16 train views, verify gradients flow (one Adam step
  reduces a toy target), all on Windows torch. If nvdiffrast/pytorch3d cannot
  be installed on this host in one attempt, fall back to WSL (proven pattern),
  and if that fails too, stop — subset is dead.
  **M1 PASSED (2026-11-03)**: nvdiffrast has no Windows PyPI wheel (would need
  an MSVC source build; pytorch3d likewise), so the fallback was used — a
  pure-torch tile-bucketed soft rasterizer (`outputs/m1_spike/m1_rasterizer.py`)
  renders soft depth+alpha of the 406k-tri baseline mesh with gradients to
  vertices: **24/24 finite-difference sign agreements** (magnitudes within
  ~10%), peak VRAM 0.14 GiB (< 3 GiB budget), ~70 s total at 256². M2 may
  proceed with this renderer; nvdiffrast remains an optional speed upgrade via
  source build if the custom path proves too slow at 512².
- **M2 (anchor run)**: baseline mesh + anchor losses only (no RGB, no
  softening), 300–800 iters. Acceptance: F@1vox ≥ the v35c30+conf reference
  (0.6315) with structure survival not worse; burr ↓ without new fringe.
- **M3 (softening run)**: add multi-layer softening. Acceptance: **F@1vox ≥
  0.65** AND a measurable structure improvement vs M2 (hole_fill ↓ from 0.98
  without recall collapse, or crater visibly reduced in renders).
- **Give-up conditions**: M1 fails for environment reasons; M2 cannot reach
  the 0.6315 reference; M3 cannot reach F 0.65 or leaves structure survival
  unchanged → stop and switch to Phase C (residual SDF) or a stronger depth
  estimator, per the route table.
- Hard rule for M2/M3: **no RGB/photometric loss and no loss-based model
  selection** anywhere (P1 negative result); selection by geometric metrics
  only, on the same frozen protocol with heldout renders.

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
