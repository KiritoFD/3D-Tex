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
