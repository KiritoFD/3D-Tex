# P1 depth refinement — per-view residual refinement of VGGT-Ω depth (2026-11-03)

Status: **complete — negative result with mechanism analysis.** The formulated
refinement (dense multi-view photometric + consistency + edge-aware
regularisation on top of the VGGT-Ω depth prior) measurably *degrades* the
geometry on this data, in every tested configuration. The cause is an
objective–data mismatch (glossy, weakly-textured synthetic appearance), shown
below with three independent diagnostics. GT depth/normals/mesh were used for
evaluation and diagnosis only, never as optimisation inputs.

## Method

Refinement model (implemented in `scripts/refine_depth.py`):

- Per view: `D' = D_omega · exp(δ)`, where δ is a sum of bilinearly-upsampled
  low-resolution grids (default 16² + 64² + 128² on the 512² model grid), so the
  correction is multiplicative, smooth and VRAM-cheap (332k parameters for 16
  views; optimisation peak fits easily in the 8 GiB card, ~150–460 s per run).
- Losses per iteration (8192 sampled foreground pixels/view, 4 nearest source
  views by camera centre):
  - `L_photo`: multi-view photometric consistency — **per-channel ZNCC** of
    5×5 patches between reference and source views, warped through the refined
    depth, occlusion-masked (z_proj beyond source depth ×1.5% rejected).
    Grayscale ZNCC was measured to be nearly signal-free on this data (GT
    ceiling 0.26, see below), so the correlation is computed on the 75-dim
    per-channel patch vector (GT ceiling 0.645).
  - `L_depth`: multi-view depth consistency — huber of the relative difference
    between projected refined depth and source refined depth, same mask.
  - `L_smooth` + `L_curv`: edge-aware (RGB-gradient-weighted) first-order TV and
    isotropic second-order penalty **on δ** — keeps the correction low-order
    without forcing the refined surface itself to be planar.
  - `L_prior`: huber(δ, β) — bounds the residual, prevents drift from the prior.
- Views: anchor16 only (the 16 original training views), predicted poses and
  intrinsics exactly as exported by the VGGT-Ω adapter. An optional
  `--joint-pose` mode adds a small 6-DOF correction per view (rotvec +
  translation, L2-anchored), motivated by the measured pose contribution below.
- Evaluation is the unchanged frozen protocol: refined depth is exported in the
  omega layout and `capture_eval.py` runs the identical default path (train-only
  Sim3, nearest regrid, pred-pose TSDF, frozen P0 grid), then
  `measure_surfaces.py` (Blender BVH) for the mesh metrics. The baseline export
  reproduced the P0 case bit-identically (234,215 verts / 406,187 faces /
  F@1vox 0.6207), which validates the harness.

## Runs and headline results

| Run | Config | photo ZNCC (train) | abs_rel | normal° (flat/edge) | F@1vox | P / R | acc (vox) | compl (vox) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | 0.575 | **1.242%** | **32.16 (15.61/33.86)** | **0.6207** | 0.554 / 0.705 | **1.403** | 0.809 |
| run1 (main) | grids 16/64/128, λ(1,1,.5,.1,1), 1200 it | 0.643→**0.726** | 2.391% | 49.87 (47.49/41.32) | 0.5746 | 0.463 / 0.756 | 2.047 | 0.720 |
| run2 | grids 16/64, cap .55, λ(1,20,1,1,5 β.01), 400 it | →0.779* | 1.780% | 42.94 (35.93/36.92) | 0.5602 | 0.474 / 0.686 | 1.839 | 0.823 |
| run3 | run2 + joint pose (0.05°, 0.0013) | →0.800* | 1.556% | 39.11 (31.85/36.21) | 0.5978 | 0.521 / 0.702 | 1.707 | 0.800 |

*capped units, not comparable with run1.

Structure / auxiliary metrics:

| Run | low-freq RMS | high-freq RMS | burr | hole_fill | coverage | outlier acc>2vox | acc_sharp / flat | P_sharp | R_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0101 | 0.0218 | 0.0188 | 0.985 | 0.432 | 0.2446 | 2.076 / 1.058 | 0.326 | 0.721 |
| run1 | 0.0202 | 0.0310 | 0.0237 | 0.996 | 0.170 | 0.3706 | 2.964 / 1.484 | 0.230 | 0.756 |
| run2 | 0.0148 | 0.0234 | 0.0185 | 0.984 | 0.222 | 0.3459 | 2.667 / 1.347 | 0.237 | 0.675 |
| run3 | 0.0131 | 0.0231 | 0.0185 | 0.979 | 0.374 | 0.3059 | 2.527 / 1.273 | 0.277 | 0.685 |

Every refined run is worse on abs_rel, normal, F-score, precision_sharp and the
scale-split residuals. The stage goal (F@1vox > 0.70 at recall ≥ 0.705 with
lower normal error) is **not reached**; no tested λ combination moves any run
above the baseline. Notably run1 *raises* recall to 0.756 while collapsing
precision — the same "thicker surface" trade as over-dense capture (group 1).

## Why it fails — three diagnostics (all decisive, GT used for diagnosis only)

1. **The photometric objective is nearly geometry-free on this data.**
   5×5 per-channel ZNCC at 1.27M sampled pixels: pred 0.574, GT-pose+GT-depth
   ceiling 0.645 (grayscale: 0.090 vs 0.260 — glossy specular plastic violates
   Lambertian consistency even with perfect geometry). The entire
   geometry-carrying signal is the 0.07 gap; run1's optimizer reached ZNCC
   0.726 — *above* the GT ceiling — while tripling depth error: the
   parameterization has more adversarial capacity (fitting specularity and
   aliasing by warping depth) than the signal it is supposed to chase.
2. **Photometric gain is anti-correlated with geometric gain.** Per-pixel
   (127k px) Spearman(ZNCC gain, error change) = **+0.086** for run1 (+0.15 in
   the top-10% textured pixels — improving ZNCC there *increased* error
   1.03%→1.93%). It is not an optimizer/λ problem: the loss landscape's
   downhill direction is away from the true surface.
3. **A plane-sweep argmin has no skill either.** Sweeping 25 multiplicative
   ratios ∈ [0.94, 1.06] with the same capped ZNCC cost (median over 4
   sources): Spearman(argmin−1, true ratio−1) = **+0.053**; only 21.9% of
   pixels land closer to GT than the prior (worse than chance); applying the
   argmin doubles the ratio error (0.0122→0.0283). Gradient-free global search
   of the same objective is equally dead — the failure is the objective, not
   the optimiser.
4. **Pose error is a real but small contributor.** GT poses raise ZNCC from
   0.090 to 0.155 (grayscale) with predicted depth unchanged, motivating run3's
   joint pose refinement; the learned corrections were tiny (0.05°, 0.0013)
   and reduced but did not remove the damage (F 0.5978 vs 0.5602 depth-only).

Reconciliation with E2 (why PatchMatch can be better where it commits):
COLMAP commits on only 2.5% of pixels (16 views) after geometric-consistency
filtering — photometric information on this data exists only for a sparse,
multiply-confirmed subset. Dense per-pixel refinement of *all* pixels cannot
extract it; that is exactly what diagnostics 2–3 show.

## Visual review (clay renders, BLENDER_WORKBENCH, GT train cameras 0/4/8/12)

Files: `outputs/p1_refine/renders/{baseline,run1,run3}/view_0..3.png`
(baseline render verified pixel-identical to P0.5 `repro_nearest`).

- **run1** (`run1/view_0.png`, `view_2.png`): the whole body turns into a rough
  "rock/coral" texture; the base-plate dunes are not flattened — they are
  replaced by finer, denser speckle; silhouette grows a debris field of spikes
  and fly-aways (bright speckles); thin structures remain melted and
  additionally chewed. Matches acc 1.40→2.05 vox, flat-normal 15.6°→47.5°.
- **run3** (`run3/view_0.png`, `view_2.png`): global shape and dune scale
  unchanged vs baseline (joint-pose corrections are sub-degree), but a fine
  grain covers every surface, the silhouette fringe thickens slightly and
  isolated bright speckles appear; posts/studs/window holes show no recovery.
  Matches F 0.598 vs 0.621, normal +7°.
- No run changes hole bridging (`hole_fill` 0.98–1.0) or burr materially
  (0.019–0.024): the structure failures are inherited, not created — but
  nothing in the photometric signal repairs them either.

## λ sensitivity summary

Damage decreases monotonically as photo influence is capped/anchored
(run1 → run2 → run3 on F: 0.575/0.560/0.598 vs baseline 0.621), i.e. the best
measured setting of the photometric weight on this data is zero. The residual
grids' capacity also matters: dropping the 128² grid (run1→run2) removes the
worst speckle. Joint-pose absorbs the small systematic misalignment and is the
only change that improved a refined run (run3 vs run2).

## Conclusions

1. **P1 as formulated is a dead end on this data** — measured, mechanism
   understood, and not fixable by λ tuning, residual capacity, robust capping
   or joint pose refinement. The VGGT-Ω depth prior is already better than
   anything raw-RGB photometric consistency can reach here, because glossy
   synthetic appearance makes the dense photometric objective nearly
   geometry-free (GT ceiling ZNCC 0.26 grayscale / 0.65 per-channel).
2. The failure is **not** "optimisation smoothing away detail" — it is
   photometric noise being converted into new mid/high-frequency surface
   error (low-freq RMS 0.010→0.013–0.020, high-freq RMS 0.022→0.023–0.031).
   Mid-frequency dunes are photometrically invisible (flat plastic has no
   patch contrast), so they survive untouched while the optimizer adds damage
   elsewhere.
3. **Remaining photometric direction consistent with all evidence**: a
   PatchMatch-style winner-take-all per-pixel search *with* multi-view
   geometric-consistency gating, using the VGGT depth as a search prior and
   keeping the prior wherever evidence is weak (E2 shows committed pixels are
   genuinely better: set32 acc 0.882 vox vs 1.813). Dense refinement of every
   pixel is ruled out by diagnostics 2–3.
4. Otherwise: accept VGGT-Ω depth as the final depth source for this object
   class and pursue capture-side (CAPTURE_GUIDELINES) or representation-side
   routes instead of depth-value refinement.

## Reproduction

```powershell
.venv/Scripts/python.exe scripts/refine_depth.py --omega data/scenes/nerf_lego16/geometry/omega --images data/scenes/nerf_lego16/images --out outputs/p1_refine/run_main --iters 1200 --samples 8192
.venv/Scripts/python.exe scripts/refine_depth.py ... --out outputs/p1_refine/run2 --iters 400 --grids 16,64 --lambda-depth 20 --lambda-smooth 1 --lambda-curv 1 --lambda-prior 5 --prior-beta 0.01 --photo-cap 0.55
.venv/Scripts/python.exe scripts/refine_depth.py ... --out outputs/p1_refine/run3 (run2 flags + --joint-pose)
.venv/Scripts/python.exe scripts/capture/capture_eval.py --capture <scene with geometry/gt + geometry/omega> --out <eval out>
blender -b --disable-autoexec --python scripts/diagnostics/measure_surfaces.py -- --out <eval out> --source-flags outputs/lego_source_edge_flags.npy
```

Diagnostics (scripts and outputs under `outputs/p1_refine/`, gitignored):
`diag_zncc.py` (ceiling/config sweep), `diag_correlation.py` (photo-vs-geometry
per-pixel correlation), `diag_sweep.py` (plane-sweep argmin skill).
Refine logs: `outputs/p1_refine/run*/refine_log.json`; eval reports:
`outputs/p1_refine/eval_*/ablation_report.json` + `surface_metrics.json`.

---

# Follow-up A (2026-11-03): geometric-consistency gate sweep + sparse gated fusion

Question (updated acceptance criteria): relax COLMAP's `geom_consistency`
filter step by step, find the coverage-vs-accuracy knee, then fuse sparsely —
replace VGGT depth with COLMAP depth ONLY on gate-passing pixels (VGGT kept
everywhere else) — and test whether a small number of trusted corrections
improves accuracy without degrading completeness, silhouette or structure.
Precision/recall are reported separately; selection is by geometric metrics
only (no ZNCC/PSNR/loss-based early stopping anywhere in this section).

## A.1 Gate threshold sweep (anchor16, COLMAP 4.2.1 CUDA, WSL build)

Swept `filter_geom_consistency_max_cost` × `filter_min_num_consistent`
(full PatchMatch re-run per level, ~5 min each; bins: `outputs/colmap/sweep/`).
COLMAP-only metrics through the frozen protocol (`capture_eval --geometry-dir`):

| Level | valid px % | abs_rel mean/med | normal° | flat° | hole_fill | coverage_rate |
|---|---:|---:|---:|---:|---:|---:|
| cost1_nc2 (E2 default) | 2.54 | 2.36% / 0.89% | 24.43 | 8.3 | 0.206 | 0.066 |
| cost2_nc2 | 3.72 | **2.34%** / 0.91% | 29.25 | 11.83 | 0.298 | 0.090 |
| cost3_nc2 | 14.84 | 10.75% / 1.60% | 43.22 | 30.07 | 0.778 | 0.155 |
| cost4_nc2 | 14.84 | = cost3 (identical maps) | — | — | — | — |
| cost8_nc2 | 14.84 | = cost3 (identical maps) | — | — | — | — |
| cost1_nc1 | 14.08 | 12.57% / 1.70% | 36.98 | 16.54 | 0.788 | 0.126 |
| cost4_nc1 | 30.98 | 19.37% / 4.09% | 50.40 | 43.90 | 0.970 | 0.111 |

**Knee: `cost2_nc2`.** Coverage rises 2.5%→3.7% at unchanged accuracy
(2.34%); at cost3 the gate collapses outright (14.8% coverage but 10.8%
error), and cost4/cost8 produce byte-identical maps (threshold saturates).
Dropping `min_num_consistent` to 1 behaves like the cost relaxation (14–31%
coverage, 12.6–19.4% error). Answer to "放宽到多少覆盖率时精度还显著优于
VGGT": **on abs_rel, never** (COLMAP ≥2.34% vs VGGT 1.24% everywhere); **on
normals, only up to ~3.7% coverage** (24.4° at cost1, 29.3° at cost2, both
better than VGGT's 32.16°; gone by cost3). The trustworthy gate is intrinsically
sparse at 16 views.

## A.2 Sparse gated fusion (gate% = replaced pred-grid pixel fraction; hard replace unless noted)

Fused depth (`scripts/gated_fusion.py`: z-buffer splat into the predicted gauge
via the frozen train-only Sim3, verified bit-consistent with the E2 aligned
poses) through the IDENTICAL default protocol as the 0.6207 baseline:

| Level | gate% | F@1vox | P / R @1vox | acc (vox) | compl (vox) | abs_rel | normal° (flat) | burr | hole_fill | P_sharp / R_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | **0.6207** | 0.554 / 0.705 | 1.403 | 0.809 | **1.24%** | **32.16 (15.61)** | **0.0188** | 0.985 | **0.326** / 0.721 |
| cost1_nc2 | 3.3 | 0.6211 | 0.553 / 0.708 | **1.393** | 0.806 | 1.39% | 34.54 (19.56) | 0.0210 | 0.984 | 0.321 / 0.717 |
| cost2_nc2 | 4.4 | 0.6187 | 0.552 / 0.703 | 1.402 | **0.812** | 1.45% | 35.10 (20.17) | 0.0230 | 0.984 | 0.326 / 0.714 |
| cost1_nc2, min-cluster 25 | 2.7 | 0.6198 | 0.553 / 0.705 | 1.394 | 0.810 | 1.33% | 33.62 (18.00) | 0.0201 | 0.983 | 0.318 / 0.711 |
| cost2_nc2, min-cluster 25 | 3.8 | 0.6175 | 0.552 / 0.701 | 1.396 | 0.816 | 1.38% | 34.14 (18.64) | 0.0215 | 0.982 | 0.313 / 0.710 |
| cost4_nc2 | 15.9 | 0.5898 | 0.520 / 0.681 | 1.600 | 0.839 | 5.79% | 44.33 (36.32) | 0.0314 | 0.988 | 0.287 / 0.702 |
| cost4_nc2, feather 2px | 20.4 | 0.5828 | 0.512 / 0.677 | 1.669 | 0.843 | 6.69% | 46.25 (39.78) | 0.0309 | 0.986 | 0.276 / 0.696 |
| cost1_nc1 | 16.6 | 0.5847 | 0.508 / 0.689 | 1.791 | 0.823 | 7.12% | 47.11 (37.91) | 0.0307 | 0.988 | 0.272 / 0.708 |
| cost4_nc1 | 31.5 | 0.4815 | 0.408 / 0.588 | 2.919 | 1.037 | 16.52% | 52.69 (49.59) | 0.0422 | 0.981 | 0.193 / 0.581 |

**Verdict: sparse gated fusion is exactly neutral where COLMAP is trustworthy
(gate ≤4.4% of pixels: F within ±0.002 of baseline, accuracy −0.01 vox,
completeness and structure unchanged), and monotonically harmful past the
knee.** No tested operating point improves any metric while keeping the
others: the "accuracy up, completeness down" trade-off does not materialise
because at the trustworthy gate levels there is no measurable effect at all —
3–4% of pixels is too few to move the TSDF surface. Cluster filtering
(min-cluster 25) removes the isolated-splat noise (normals 34.5°→33.6°) but
stays neutral. Feathering only spreads the damage at loose levels.

Why the E2 advantage does not transfer: the committed-pixel advantage was
*normal/accuracy of an isolated sparse surface*, not lower depth error
(COLMAP abs_rel 2.34% > VGGT 1.24% at every level). Replacing 3% of pixels in
a dense, view-consistent field injects (a) slightly higher-error values and
(b) spike discontinuities at gate boundaries, while the region-coherent
advantage PatchMatch has cannot express itself at 3% coverage.

## A.3 Held-out views (4 predeclared test cameras, clay renders + silhouette IoU)

Renders: `outputs/gated_fusion/renders/{baseline,cost1_nc2_b0,cost2_nc2_b0,cost4_nc2_b0}/view_0..3.png`
(same WORKBENCH clay path as P0.5; IoU via `heldout_views.py compare`):

| Case | silhouette IoU (4 heldout views) |
|---|---:|
| baseline | 0.9377 |
| cost1_nc2 | 0.9364 |
| cost2_nc2 | 0.9365 |
| cost4_nc2 | **0.9110** |

Visual review (image-by-image): baseline and cost1/cost2 are indistinguishable
at full-frame level — same dune field, same melted posts, same stud loss, no
new texture or crispness; structure survival is inherited unchanged (studs NOT
recovered as an array, posts/spokes still melted, window holes still bridged —
the gate has no signal there either). cost4 shows body-wide spike debris and a
thickened silhouette fringe, matching its IoU/F drop. "更光滑" does not occur
at any level; the strict-gate variants are simply the baseline with 3% of
pixels nudged.

## A.4 Conclusions for the record

1. The COLMAP geometric-consistency gate is all-or-nothing on this data: knee
   at cost2_nc2 (3.7% coverage @ 2.34%), collapse at cost3 (14.8% @ 10.8%).
   There is no middle operating point with meaningfully more trustworthy
   coverage.
2. Sparse gated fusion cannot answer "用少量可信修正提升精度" affirmatively
   at 16 views: trustworthy corrections are too sparse to matter (neutral),
   and denser corrections are not trustworthy (harmful). The full trade-off
   table above is the working-point record.
3. This closes the photometric direction on the depth-value level on this
   scene (P1 dense refinement negative; sparse gated replacement neutral).
   Remaining routes are representation-side (see GEOMETRY_OPTIMIZATION.md
   sections B and C) and capture-side.

---

# Follow-up B (2026-11-03): cross-scene — is the P1 failure a glossy-plastic special case?

Method: the **held-out-source generalisation test** (GT-free, calibrated on
lego). The run3 refinement recipe is re-run per scene with 4 randomly chosen
views excluded from being photometric SOURCE views (`--holdout 4`); afterwards
we measure the mean per-channel 5×5 ZNCC against train sources and against the
held-out sources, at the initial and refined depth
(`outputs/p1_refine/diag_generalization.py`). If the photometric gain were
geometrically real it must transfer to unseen sources; adversarial gains do
not. Calibration: on lego the same recipe is independently known from GT to be
harmful (F 0.598 vs 0.621 baseline), and its transfer ratio is 0.28 — so any
scene scoring ≤ that band is judged "not applicable", and a scene would need a
clearly higher ratio to be a candidate.

| Scene | Surface / imagery | pred ZNCC (train src) | GT ceiling | train gain | held-out gain | transfer ratio | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| lego (anchor16) | glossy plastic, path-traced | 0.564 | 0.645 per-channel (0.26 grayscale) | +0.033 | +0.009 | 0.28 | ✗ (calibration: F 0.598 < 0.621, harmful) |
| eth3d pipes (12v) | industrial metal pipes, real DSLR | 0.308 | n/a (no dense GT in repo) | +0.012 | −0.000 | −0.01 | ✗ |
| eth3d kicker (12v) | mixed glossy/wood foosball, real DSLR | 0.425 | n/a | +0.033 | −0.004 | −0.11 | ✗ |
| eth3d relief (25v) | matte carved stone, strong texture, real DSLR | 0.782 | n/a | +0.014 | +0.003 | 0.24 | ✗ |

(abs_rel / normal° / F-score per scene are reported where dense GT exists —
lego only; ETH3D scenes carry laser-scan GT but no dense depth/mesh in this
repo, marked n/a rather than approximated. The generalisation test is the
decisive evidence and needs no GT.)

Findings:

1. **The failure is NOT a glossy-plastic special case.** On real photographs
   across metallic (pipes), mixed-material (kicker) and matte strongly-textured
   (relief) surfaces, the refinement's ZNCC gain does not transfer to unseen
   source views either (ratios −0.11 to +0.24, all at or below the
   calibrated-harmful lego band).
2. **Base photo-consistency varies 2.5× by surface** (pred-gauge ZNCC: pipes
   0.31, kicker 0.43, lego 0.56, relief 0.78): matte textured stone is far more
   multi-view consistent than glossy plastic — texture/BRDF genuinely control
   how much photometric signal exists. BUT even at relief's 0.78 the residual
   gains available to an optimizer are adversarial: the easy consistency is
   already correct, and the remaining slack is specular/aliasing/noise, same
   mechanism as lego diagnostics 2–3.
3. Texture is therefore **necessary but not sufficient** for photometric
   geometry: it raises the consistency ceiling, but dense refinement still
   finds no transferable signal on any tested scene. On this evidence the
   photometric route is closed generally, not just for glossy synthetic
   objects; representation-side routes (GEOMETRY_OPTIMIZATION.md B/C) are the
   way forward.

Runs: `outputs/xscene/{lego,pipes,kicker,relief}/refine_log.json`
(refinement run3-recipe + holdout 4; lego 439 s, pipes 314 s, kicker 260 s,
relief 356 s on the shared 8 GiB GPU).

---

# Follow-up C (2026-11-03): geometry-side supervision, NO photometric optimisation

Targets the three visually-located defects (fluffy window bridges, silhouette
burr cloud, cab-roof crater) with purely geometric evidence: (1) cross-view
silhouette/free-space carving, (2) VGGT confidence-weighted TSDF fusion.
No RGB loss, no ZNCC/PSNR anywhere in the pipeline; selection is by the
geometric metrics only.

## C.0 Mechanism diagnosis first (why naive carving cannot work, and what does)

Each depth pixel entering the baseline fusion was unprojected and reprojected
into all 16 views (`outputs/geometry_pruning/diag/`). Two per-pixel signals:
`votes` (fraction of views where the point lands on a background pixel of the
GT-foreground protocol mask, eroded 1 px) and `confirms` (fraction of views
where it lands on foreground with depth consistent to ±1.5% with that view's
unmasked predicted depth). Findings:

- Raw predicted poses and depth are mutually inconsistent at the few-percent
  level (pair reprojection median mismatch up to 16%); votes must therefore be
  computed in the frozen train-only Sim(3)-aligned gauge, not the raw gauge.
- Bridge/burr pixels are NOT isolated outliers: they are view-consistent
  (their own unmasked depth values confirm each other across views — the same
  bias as the dunes). A pure "projects to background in many views" vote marks
  them, but it also marks every legitimate edge pixel at similar rates; the
  separation only appears when combined with the confirm signal: bridges/burr
  = votes high AND confirms low (magenta in the diag maps), legit surface =
  confirms high (cyan). Strict thresholds (votes≥0.5 & confirms≤0.2) delete
  only 1.27% of foreground pixels and change nothing — the defects are
  view-consistent and survive.

So carving works only as a *tunable precision/recall trade*, not as a clean
defect detector. The trade curve below is the result.

## C.1 Free-space / silhouette carving (scripts/geometry_pruning.py)

Rule: delete a depth pixel when `votes ≥ vote_frac AND confirms ≤ confirm_max`
(aligned gauge; deletion exported in the raw predicted grid; eval protocol
unchanged). Four operating points + GT-side classification of the deleted
pixels (GT used for evaluation only):

| Variant | deleted fg px | deleted-px abs_rel mean/med | >5% off (deleted vs kept) | F@1vox | P / R | acc (vox) | compl (vox) | burr | hole_fill | P_sharp / R_sharp | heldout IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | — | 3.6% | 0.6207 | 0.554 / 0.705 | 1.403 | 0.809 | 0.0188 | 0.985 | 0.326 / 0.721 | 0.9377 |
| v50c20 (strict) | 1.27% | 1.07% / 0.63% | 2.9% | 0.6249 | 0.560 / 0.708 | 1.393 | 0.810 | 0.0190 | 0.985 | 0.326 / 0.722 | — |
| v35c30 | 3.65% | 1.52% / 0.69% | 6.1% | 0.6311 | 0.570 / 0.706 | 1.365 | 0.818 | **0.0148** | 0.983 | 0.340 / 0.716 | **0.9386** |
| v25c40 | 8.45% | 1.73% / 0.76% | 7.4% | 0.6349 | 0.587 / 0.692 | 1.312 | 0.851 | 0.0130 | 0.916 | 0.345 / 0.682 | 0.9332 |
| v15c50 (aggressive) | 13.07% | 1.79% / 0.80% | 7.7% | **0.6386** | 0.605 / 0.676 | **1.251** | 0.886 | **0.0115** | **0.753** | **0.366** / 0.645 | 0.9291 |

Image-space (same protocol): abs_rel 1.242% → 1.131% and normal 32.16° →
31.50° monotonically from strict to aggressive; coverage_rate 0.432 → 0.475.

**Cost/benefit accounting.** Deleted pixels are ~2× enriched for wrong
geometry (>5% outliers: 6–8% vs 3.6% kept), but the majority of deleted
pixels are still good surface (median error 0.7–0.8%) — that is the
completeness price: recall 0.705 → 0.676 and R_sharp 0.721 → 0.645 at the most
aggressive point. In return: F +0.018, precision +0.051, accuracy −0.15 vox,
burr −39%, window holes genuinely open (hole_fill 0.985 → 0.753; visible in
renders — the top window frame goes from a white bridge to a clear through
hole), P_sharp +0.04, and abs_rel/normal also improve. Heldout silhouette IoU
peaks at v35c30 (0.9386, better than baseline) and degrades only slightly at
aggressive levels (0.929).

Operating-point read: **v35c30 is a near-free lunch** (F +0.010, P and R both
≥ baseline, burr −21%, IoU +0.001, completeness cost +0.009 vox); v25c40 and
v15c50 are deliberate precision/structure-for-recall trades. The cab-roof
crater is unaffected by carving (it is not a background/free-space violation).

## C.2 VGGT confidence-weighted TSDF fusion

VGGT-Ω ships per-pixel depth confidence (`omega/confidence/*.npy`, range
1.1–72.8, median 4.5). Wired as TSDF weights via an opt-in flag
(`capture_eval.py --weight-mode confidence`; regridded like depth, globally
rescaled p99→8 since fuse() clips at 8; a global rescale is a no-op except at
the clip boundary; default remains uniform — protocol unchanged).

| Variant | F@1vox | P / R | acc (vox) | compl (vox) | burr | hole_fill | P_sharp / R_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform (baseline) | 0.6207 | 0.554 / 0.705 | 1.403 | 0.809 | 0.0188 | 0.985 | 0.326 / 0.721 |
| confidence | 0.6226 | 0.568 / 0.690 | 1.378 | 0.831 | 0.0187 | 0.983 | 0.333 / 0.696 |

Small precision win, small recall cost — direction consistent with carving.

## C.3 Combination (orthogonal, stacks)

| Variant | F@1vox | P / R | acc (vox) | compl (vox) | burr | hole_fill | P_sharp / R_sharp | heldout IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v35c30 + conf | 0.6315 | 0.581 / 0.692 | 1.350 | 0.837 | 0.0148 | 0.983 | 0.339 / 0.694 | — |
| v25c40 + conf | 0.6365 | 0.600 / 0.678 | 1.287 | 0.867 | 0.0130 | 0.916 | 0.353 / 0.664 | — |
| v15c50 + conf | **0.6398** | 0.617 / 0.664 | **1.230** | 0.901 | 0.0115 | **0.753** | **0.371** / 0.630 | 0.9290 |

Confidence weighting adds ~+0.01 precision and −0.012 recall at every carving
level. Best F of this phase: 0.6398 (v15c50+conf), still below the P0 2-voxel
consistency filter's 0.6927 (P 0.780 / R 0.623, compl 1.217) — but that filter
achieves its F by deleting far more surface (completeness 1.217 vs 0.901) and
has no burr/structure benefit; the carve+conf curve offers structurally
cleaner operating points (burr −39%, holes open, P_sharp 0.371) at moderate
recall cost.

## C.4 Conclusions

1. Geometry-side supervision works as a **tunable trade**, not a free fix:
   the defects it removes are real (burr −39%, windows genuinely open,
   F +0.019 max), and the deleted pixels are ~2× enriched for wrong geometry,
   but completeness/recall pay proportionally because most deleted pixels are
   still decent surface.
2. The cab-roof crater is untouched (not a free-space violation); studs and
   rods are not recovered (carving cannot add information, only remove).
3. Recommendation: adopt **v35c30+conf** as the new default fusion for
   downstream work (near-free: F 0.6315, burr −21%, IoU +0.001, completeness
   cost +0.03 vox vs baseline), keep v25c40/v15c50 as documented aggressive
   operating points when precision/structure matters more than recall.
4. What carving cannot do, Phase B (Mesh Splatting with anchors) must:
   actually re-open geometry (topology change), not just delete it.

---

# Follow-up D (2026-11-03): tiling / crop probe — is the structure loss an effective-resolution problem?

Question: the melted studs / wheel spokes / window frames are tens of pixels
across at full-frame capture. The group-3 resolution axis (whole-image
downsampling 256–800²) was nearly flat, so it cannot say whether *structure-scale*
resolution is the limit. Probe: crop a structure region per view around the
projected 3-D region (back-projected from the anchor16 predicted geometry, no
GT used), re-run VGGT-Ω on the crops, compare depth/normal quality in-region.

Setups (all 16 anchor views, model input 512²):
- **crop-only**: 16 crop images per region (crop sides 293–519 px → effective
  zoom 1.5–2.7×), poses from the crop run itself.
- **mixed**: 16 full images + 16 roofwin crops + 16 wheels crops in ONE joint
  run (48 images; full images anchor the trajectory).

## D.1 Trajectory health (train-only Sim3 residual on camera centres)

| Run | scale | resid med | resid max | Verdict |
|---|---:|---:|---:|---|
| full-frame anchor16 | 4.7232 | 0.029 | 0.054 | reference |
| crop-only roofwin | 4.1188 | **0.245** | 0.477 | trajectory broken (8× worse) |
| crop-only wheels | 4.6969 | **0.167** | 0.403 | broken |
| mixed (full views) | 4.8524 | 0.048 | 0.131 | ≈ anchored (full images keep registration) |

Crop-only sets destroy multi-view registration: each view's crop shows a
different part of the object, so the tracker drifts. Keeping the full-frame
images in the same run anchors it back to near-baseline.

## D.2 Region depth / normal quality (scale-matched per view; GT evaluation only)

| Region | abs_rel base / crop / mixed | normal flat° base / crop / mixed | normal edge° base / crop / mixed |
|---|---|---|---|
| roofwin | 1.54% / 2.54% / 1.70% | 17.51 / 15.50 / 18.27 | 32.96 / **26.83** / **26.61** |
| wheels | 1.40% / 1.78% / **1.36%** | 15.81 / 13.82 / **13.66** | 35.40 / **28.91** / **29.44** |

## D.3 Structure recovery (normal-map visual review, GT | base | crop | mixed)

Images: `outputs/tiling/vis/{roofwin,wheels}_{r_0,r_13,r_33}_normals_gt_base_tiling_mix.png`.
- roofwin r_0/r_13: GT shows the roof/bucket stud array as crisp rows of
  circles; full-frame base melts it into blobs; **crop and mixed runs both
  restore the stud array as countable discrete rows**, and the cab window
  frame regains sharp edges.
- wheels r_0: GT shows both wheel rims with spoke pattern + plate stud rows;
  base melts them; **crop and mixed restore both rims with visible spokes and
  the plate stud array** (slightly cleaner in mixed than crop-only).
- Structure recovery is identical between crop-only and mixed; mixed fixes the
  trajectory, so the structure gain becomes usable.

## D.4 Conclusions

1. **The melted structure is an effective-resolution problem, not (only) a
   network-prior limit.** At 1.5–2.7× effective zoom the same network recovers
   stud arrays, wheel rims and spokes that are melted at full-frame 512² —
   the prior CAN represent them; full-frame capture does not give the region
   enough tokens.
2. **But crops alone break registration** (residual ×8, abs_rel 1.5→2.5%):
   naive tiling is not a usable geometry pipeline. The usable form is
   **mixed full-frame + close-up sets**: trajectory stays anchored (0.048)
   while edge normals improve −6° and abs_rel matches or beats the baseline
   in-region (wheels 1.36% vs 1.40%).
3. The cab-roof crater persists in crop runs too (roof region is deepened,
   not repaired) — that one is a genuine local-shape prior failure, not
   resolution.
4. Follow-up uses (not yet implemented): mixed-set capture for
   structure-critical objects; crop-depth as high-confidence structural
   anchors for Phase B; per-region TSDF fusion at the frozen voxel with
   mixed-run crop depth as a replacement for baseline depth in structure
   regions (needs a consistency rule for the boundary).

## D.5 Mixed-trajectory fusion attempt (negative result, 2026-11-03)

Goal: turn the mixed run's crop depth into fused geometry (F ≥ v35c30+conf
0.6315 target). Pipeline (`scripts/mixed_fusion.py`): crop pose := matching
full view's pose (+ optional multi-view refinement), intrinsics := shifted
full K, depth := crop depth × robust overlap scalar, then fused through the
frozen protocol (48-frame merged GT: 16 full + 32 crop frames).

Alignment measurements: scalar alone leaves median 8.4% mismatch (initially
miscomputed correspondence; after fixing, 1.1% median / 3.1% p90 against the
own view). The multi-view refinement (register crop points against all 16
full views, robust 10% gate) reaches median ~3.5–3.7% but **p90 stays ~23%**:
the disagreement tail is not pose-shaped — it is the crop run's own
fine-scale shape error at structure boundaries, unfixable by any pose
correction.

Fusion variants through the identical eval (mixed 48-frame GT set):

| Variant | F@1vox | P / R | acc (vox) | compl (vox) | Note |
|---|---:|---:|---:|---:|---|
| target: v35c30+conf (16-frame GT) | 0.6315 | 0.581 / 0.692 | 1.350 | 0.837 | — |
| coexist (crop ×4 weight, anchor16 full) | 0.5775 | 0.501 / 0.682 | 1.662 | 0.853 | double-sheets in conflict areas |
| adjudicate (mutual exclusion by conf) | 0.5653 | 0.485 / 0.678 | 1.688 | 0.868 | kill-mask makes it worse |
| replace (crop rules the whole rectangle) | 0.4969 | 0.404 / 0.644 | 2.074 | 0.936 | crop surface alone is misplaced |
| consistent (mixed-run full + crops) | 0.4886 | 0.477 / 0.501 | 1.644 | 1.194 | mixed full depth is worse; recall collapses |

**Conclusion: mixed-trajectory crop fusion into the voxel TSDF is not viable.**
The crop depth carries excellent local structure (Follow-up D: studs/spokes
in normals) but its *absolute surface placement* cannot be registered to the
full trajectory at sub-voxel accuracy: the multi-view refinement fixes the
median (~1–3.7%) while the structure-carrying tail (p90 ~23%) is not
pose-shaped and resists any registration. Every fusion rule either
double-sheets the surface (coexist/adjudicate) or imports the offset surface
(replace). This is the depth-value level again: photometric-free registration
cannot rescue it. The crop data remains useful as *anchors* for Phase B
(where an optimizer can move the surface), not as direct TSDF input.
