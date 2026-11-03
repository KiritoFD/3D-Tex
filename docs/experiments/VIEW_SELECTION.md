# View Selection: per-view marginal contribution to VGGT-Ω geometry

Status: core groups (E0/E1/E2) and extreme groups B1–B4/B6 measured end-to-end;
B5 (unrelated views) evaluated; the initial LOO sweep was stopped due to a
precision mismatch (see audit below), with matched-bf16 reruns required. **Headline: adding
neutral views does NOT interfere with the model's depth on the core views; the
mesh damage from "more views" happens at the fusion stage (precision dilution).
Extreme/grazing views are the exception — they DO interfere.**

## Motivation

The capture view-count ablation (CAPTURE_EXPERIMENTS.md group 1) showed that going
16→64 uniform views makes depth *more accurate* (abs-rel 1.65%→1.32%) while the fused
mesh gets *worse* (F@1vox 0.5255→0.4940). Two confounded mechanisms can produce this:

1. **Interference**: extra views change VGGT-Ω's attention, degrading the per-view
   depth/normal predictions themselves.
2. **Fusion**: extra views only add more consistently-biased depth sheets, and TSDF
   averaging integrates them into a thicker, less accurate surface.

This experiment separates the two, asks which *individual photos* carry positive or
negative marginal contribution, and derives a deployable (no-GT) view-scoring module.

## Protocol (frozen)

- Checkpoint `vggt_omega_1b_512.pt` (sha256 c02da418…), input resolution 512,
  aggregator bf16, heads fp32 — identical to the anchor16/set*N* baselines.
- A = anchor16, the repo's best capture set (the original 16 train views:
  r_0, r_7, r_13, r_20, r_26, r_33, r_40, r_46, r_53, r_59, r_66, r_73, r_79, r_86,
  r_92, r_99; elevations 9.7°–85.7°). Byte-identical image copies; the core-frame
  relative order is the same in every run (sorted filenames).
- Alignment: train-only Sim(3) on the evaluated frames' camera centres (identical
  rule as capture_eval.py in every group). Depth scaled by the Sim(3) scale.
- Evaluation grid: GT K grid 512², nearest regrid, GT foreground mask — identical to
  the frozen P0/capture protocol. Sanity check: this repo's eval of the *existing*
  anchor16 omega export reproduces the published baseline exactly
  (abs-rel 1.2416% vs 1.242%, normal 32.156° vs 32.16°).
- Mesh: frozen P0 TSDF grid (voxel 0.01150144, trunc 4 voxels, origin/shape from
  deliverables/lego_p0), uniform weights, nearest fusion sampling, then
  measure_surfaces (50k fixed-seed samples vs source BVH) + hole_fill/burr.
- GT depth/foreground for the non-anchor views: same BVH first-hit method as the
  capture pipeline (capture_render.py --pass-mode depth, 512²), no render engine.
- Renderer limitation: EEVEE/Cycles crash on this host, so **no new images were
  rendered**. Candidate groups are built from the 100 path-traced train views, the
  200 heldout test views (nerf_syn.zip), ETH3D photos, and deterministic image
  transforms of existing images.

## Core groups (disentangle interference vs fusion)

B_core = 16 neutral extra views: uniform-azimuth-rank pick over the 84 non-anchor
train views (r_1, r_6, r_9, r_15, r_23, r_24, r_30, r_32, r_41, r_42, r_60, r_80,
r_83, r_84, r_87, r_98).

| Group | Input | Evaluated / fused | Question |
|---|---|---|---|
| E0 | A (16) | A's 16 views | reference |
| E1 | A+B_core (32) | **only A's 16 views** | does the model get worse on the same reference views? |
| E2 | A+B_core (32) | all 32 views, fused | does the full chain improve? |

E1 and E2 share **one** VGGT-Ω run (A+B_core); only the evaluated/fused frame set
differs. Cross-fusion: E0's mesh = A16(E0 run), E1's mesh = A16(E1 run), E2's mesh =
all 32 — so "model change" (E0 vs E1 mesh, same 16 depths slots) and "fusion change"
(E1 vs E2 mesh) are measured separately.

### Results (depth/normal on A's reference views, image space)

| Run | abs-rel mean | normal° | flat° | edge° | RPE rot° |
|---|---:|---:|---:|---:|---:|
| E0 (A only) | 1.242% | 32.16 | 15.61 | 33.86 | 0.380 |
| E1 (A+B_core, A evaluated) | **1.220%** | 32.15 | 15.74 | 33.86 | 0.335 |

E1 ≈ E0 (−0.02pp abs-rel): **no interference on the core views from 16 neutral extra
views.** (GT source equivalence checked: re-running E0 against the ab GT instead of
the gt16 GT changes abs-rel by 0.0003pp.)

### Results (mesh, frozen P0 grid)

| Fusion | Depths fused | F@1vox | P / R @1vox | acc (vox) | compl (vox) | P_sharp | R_sharp | hole_fill | burr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 = E0 | core16 (A-only run) | **0.6212** | 0.555 / 0.705 | 1.398 | 0.809 | 0.327 | 0.721 | 0.985 | 0.019 |
| F1 = E1 | core16 (A+B run) | 0.6123 | 0.565 / 0.668 | 1.380 | 0.821 | 0.322 | 0.610 | 0.985 | 0.020 |
| F2 = E2 | all 32 | 0.5736 | 0.481 / 0.710 | 1.769 | 0.805 | 0.252 | 0.709 | 0.998 | 0.019 |

Reading: F0→F1 isolates the model change (same 16 depth slots, predictions from
different input sets) — small. F1→F2 isolates the fusion change (same model run,
16 vs 32 depths fused) — **precision collapses 0.565→0.481** while recall is flat:
the extra depth sheets thicken the surface. This reproduces the mechanism behind the
old set16→set64 mesh degradation, now attributable to fusion, not attention.

## Extreme candidate groups (A + 16 typed views each)

| Group | Construction | Mechanism tested |
|---|---|---|
| B1 near-duplicate | per anchor view the angularly nearest remaining train view (mean 7.7°, min 1.6°) | redundant tokens interfering with attention |
| B2 low quality | B1's viewpoints + gaussian blur σ2px + noise σ8/255 + exposure ×1.5 clip | acquisition quality |
| B3 appearance shift | B1's viewpoints + gamma 1.8, exposure ×0.8, channel gains (1.12,1.00,0.88) | appearance inconsistency — **proxy for specular/highlight variation, NOT a real lighting change** (renderer unavailable) |
| B4 extreme views | 8 smallest foreground-fraction + 8 lowest-elevation views from train remainder + 200 heldout test views | weak overlap / grazing / occlusion |
| B5 unrelated | 16 ETH3D images (6 pipes + 5 kicker + 5 relief) | anomaly rejection (stress test, not normal capture) |
| B6 high value | greedy max-min angular coverage over remaining train views (min 15.8°–33.7° from A∪picked) | positive-control: coverage A lacks |

B2/B3 share B1's viewpoints on purpose: B1↔B2 isolates quality at matched
viewpoints; B1↔B3 isolates appearance shift at matched viewpoints.
Each group gets: E1-style eval (A's 16 depth/normal) + E2-style eval (fuse all 32).
B5 is E1-style only — fusing a different scene's geometry into the lego grid is not
meaningful; instead we report its effect on A plus the model's own confidence on the
foreign views (rejection signal).

### Results

Image space (E1-style: error on A's 16 reference views; E0 = 1.242% / 32.16°):

| Group | abs-rel | Δ vs E0 | normal° | RPE rot° |
|---|---:|---:|---:|---:|
| E1 (B_core neutral) | 1.220% | −0.02pp | 32.15 | 0.335 |
| B1 near-duplicate | 1.236% | −0.01pp | 32.04 | 0.367 |
| B2 low quality | 1.261% | +0.02pp | 32.09 | 0.354 |
| B3 appearance proxy | 1.192% | −0.05pp | 31.92 | 0.353 |
| B4 extreme | **1.395%** | **+0.15pp** | **32.71** | 0.332 |
| B6 high value | **1.362%** | **+0.12pp** | 32.28 | 0.342 |
| B5 unrelated | **17.252%** | **+16.01pp** | **46.82** | **11.579** |

Mesh (c16 = fuse only A's depths from that run; full = fuse all 32):

| Group | F@1vox c16 | F@1vox full | P/R full | acc full (vox) | P_sharp full |
|---|---:|---:|---:|---:|---:|
| E0 reference | 0.6212 | — | — | — | — |
| B_core (E1/E2) | 0.6123 | 0.5736 | 0.481/0.710 | 1.769 | 0.252 |
| B1 near-duplicate | **0.6419** | 0.5686 | 0.486/0.685 | 1.770 | 0.281 |
| B2 low quality | 0.6021 | 0.5334 | 0.452/0.650 | 1.915 | 0.255 |
| B3 appearance proxy | 0.6337 | 0.5630 | 0.483/0.676 | 1.802 | 0.284 |
| B4 extreme | 0.6090 | **0.4521** | 0.406/0.509 | 1.838 | 0.237 |
| B6 high value | 0.6022 | 0.5028 | 0.441/0.585 | 1.842 | 0.254 |

Findings:

1. **Attention interference is absent for ordinary views.** Near-duplicates, blurred/
   noisy and tone-shifted views leave A's depth within ±0.05pp — the model is robust
   to redundant or degraded context. B1's presence even *improves* the core-16 mesh
   (0.6419 vs 0.6212): redundant support views act like a weak ensemble for the
   covered viewpoints.
2. **Extreme/grazing views are the exception.** B4 degrades A's own depth
   (+0.15pp abs-rel, +0.55° normal) *and* its full fusion is catastrophic
   (F 0.4521, recall also drops): the model both mis-predicts those views and lets
   them distort the shared frame.
3. **"High value" coverage views (B6) do not pay off.** A's depth worsens slightly
   (+0.12pp) and the full mesh is worse than neutral B_core (0.5028 vs 0.5736) —
   distant viewpoints add the most inconsistent depth sheets. Under the current
   fusion rule, coverage is not convertible into accuracy.
4. **Every 32-view fusion loses to the best 16-view fusion** (all full ≤ 0.569 vs
   e0 0.6212), with precision collapsing across all groups — the harm is generic to
   naive uniform TSDF averaging, not to any view type. View selection alone cannot
   fix this; it must pair with a fusion rule that suppresses inconsistent sheets.

## Leave-one-out marginal contribution

Set S = A + B_core (32 views). Full run once, then 31-view runs dropping each view
in turn (sequential, resume-safe, VRAM-gated). Per deletion, geometry error is
measured on the **retained 31 views** (all have GT):

    Δ_i = E(S) − E(S\{i})      (mean abs-rel / normal° over the retained views)

Δ_i > 0 ⇔ view i was a net *negative* contributor. Secondary reference: retained A-core
views only (their count changes if a core view is removed). The 32 single deletions are the exhaustive
k=1 distribution, so the equal-count random-deletion control is their own
mean/spread (bootstrap CI reported) — no extra GPU runs are burned re-sampling a
subset of itself. A joint-deletion check (drop all flagged views at once vs the same
number of random views) tests the known caveat that two redundant views can be
individually removable but not jointly.

### Results / protocol audit (2026-11-03, 18:45 CST)

**No valid individual negative-contributor ranking yet.** Eight deletion runs
completed (`r_0,r_1,r_13,r_15,r_20,r_23,r_24,r_26`), but their manifests record
`aggregator_dtype=torch.float32`, whereas the full AB baseline records
`torch.bfloat16`. LOO's subprocess omitted `--aggregator-dtype`; `auto` picked
fp32. Any apparent improvement mixes numerical precision with view deletion.
Those files remain under `outputs/viewsel/loo` for audit and are excluded from
the conclusions. The active process tree was stopped before further wasted runs.

Fix: `loo_inference_flags` inherits dtype, image resolution and seed from the
full-run manifest; `validate_loo_protocol` rejects cached or new exports with a
different dtype, heads dtype, resolution, model grid, seed or checkpoint hash.
The same rule applies to joint deletion. Regression check covers inherited
bf16 flags and rejection of a cached fp32 export. Correct reruns use a fresh
`outputs/viewsel/loo_bf16` directory. Partial tabulations explicitly report
`complete=false`; their mean is not an exhaustive random-deletion control.

The group results above remain valid: E0/E1/E2 and B1–B6 full exports use the
matched bf16 protocol. B5 is a stress test containing foreign scene images;
it demonstrates severe interference but does not calibrate a normal-capture
rejection threshold. Its 16 core views reach abs-rel 17.252%, normal 46.82°,
and relative rotation error 11.579° (`outputs/viewsel/b5_eval.json`).

### Next: distinguish removing an input from omitting its fusion contribution

Keep AB's full 32-view prediction and all 32 camera centres for the same Sim3.
Omit only four views from TSDF, chosen by the lowest no-GT foreground
cross-view reprojection inlier ratio: `r_53,r_42,r_26,r_32` (0.294–0.315).
Compare against equal-count seeded random exclusions and full E2, retaining
the same observed GT point set, voxel grid and all-view structure evaluation.
This tests fusion selection without rerunning VGGT or changing the alignment.
It is distinct from input LOO, which changes model predictions too.

`capture_eval.py --exclude-fusion-stems` implements this control and records
the excluded stems and alignment-view count. No GT depth error is used to
choose the four views. The diagnostic still uses GT foreground/grid as in P0,
so a positive result would not yet establish an end-to-end deployment gain.

## Deployable view-selection signals (no GT)

`scripts/view_selection.py score` computes, per view, from the images and the
full-run omega export only: Laplacian variance (focus), over/under-exposure
fraction, mean luminance, VGGT confidence mean/p10, min angular distance to peers
(predicted poses), min relative baseline, cross-view depth reprojection inlier
ratio, and max camera+register token cosine similarity (VGGT-Ω frame tokens).
Validation: Spearman rank correlation of each signal against the LOO Δ_i.

### Results

pending

## Deployable module design

pending (depends on the correlation outcome)

## Limitations

- B3 is a tone-curve proxy for appearance inconsistency, not a true specular/lighting
  change; a real highlight group is blocked on the renderer (EEVEE/Cycles crash on
  this host) and can be added later without changing the harness.
- Marginal contribution is set-dependent: Δ_i is measured against S = A+B_core and
  does not transfer verbatim to other sets (the joint-deletion control quantifies
  the worst case of this).
- Single scene (nerf-synthetic lego), single checkpoint, single resolution (512²
  model grid). ETH3D appears only as *foreign* views in B5.
- E1/E2 share one VGGT run, so frame *positions* in the batch differ between the
  E0 (16-view) and E1 (32-view) inputs; VGGT tokenises frames independently before
  global attention, but positional batch effects are not separately ablated.

## Reproduce

```powershell
# selection + run dirs + GT (BVH depth pass; no render engine needed)
.venv/Scripts/python.exe scripts/view_selection.py select
.venv/Scripts/python.exe scripts/view_selection.py build
blender -b --disable-autoexec --python scripts/capture/capture_render.py -- `
    --out data/scenes/viewsel/runs/ab --views 32 --name ab --pass-mode depth `
    --depth-resolution 512 --poses-json data/scenes/viewsel/runs/ab/poses.json   # same for b1,b4,b6
.venv/Scripts/python.exe scripts/view_selection.py assemble --case <case> --gt-src <gt dir> [--gt-stems ...] [--omega-src <run>/geometry/omega]

# inference (VRAM-gated, bf16 to match the frozen protocol)
.venv/Scripts/python.exe scripts/view_selection.py infer --run-dir data/scenes/viewsel/runs/ab --aggregator-dtype bfloat16 --save-tokens

# evaluation: image-space (this script) + frozen mesh protocol (capture_eval + measure_surfaces)
.venv/Scripts/python.exe scripts/view_selection.py eval --omega <omega> --gt <gt> --out <json>
.venv/Scripts/python.exe scripts/capture/capture_eval.py --capture data/scenes/viewsel/cases/<case> --out outputs/viewsel/mesh/<case>
blender -b --disable-autoexec --python scripts/diagnostics/measure_surfaces.py -- `
    --out outputs/viewsel/mesh/<case> --source-flags outputs/lego_source_edge_flags.npy

# LOO + deployable signals
.venv/Scripts/python.exe scripts/view_selection.py score --run-dir data/scenes/viewsel/runs/ab --out outputs/viewsel/scores_ab.json
.venv/Scripts/python.exe scripts/view_selection.py loo --phase run --base data/scenes/viewsel/runs/ab --gt data/scenes/viewsel/cases/e2/geometry/gt --work outputs/viewsel/loo
.venv/Scripts/python.exe scripts/view_selection.py loo --phase tabulate --base data/scenes/viewsel/runs/ab --gt ... --work outputs/viewsel/loo --scores outputs/viewsel/scores_ab.json
```

For the corrected sweep, replace `--work outputs/viewsel/loo` with
`--work outputs/viewsel/loo_bf16` in both commands. Supply `--anchor-stems`
with the comma-separated `anchor16` list from `selection.json` for core references.
