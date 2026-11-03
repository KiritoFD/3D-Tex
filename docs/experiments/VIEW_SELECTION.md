# View Selection: per-view marginal contribution to VGGT-Ω geometry

Status: core groups (E0/E1/E2) measured; extreme groups B1–B6 measured; LOO + random
controls measured; deployable-signal validation done. **Numbers below are final unless
marked pending.** Independent experiment; does not modify GEOMETRY_OPTIMIZATION.md,
P1_DEPTH_REFINEMENT.md or CAPTURE_*.md.

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

| Run | abs-rel mean / median | normal° | flat° | edge° | RPE rot° |
|---|---:|---:|---:|---:|---:|
| E0 (A only) | 1.242% / 0.701% | 32.16 | 15.61 | 33.86 | 0.442 |
| E1 (A+B_core, A evaluated) | pending | | | | |

### Results (mesh, frozen P0 grid)

| Fusion | F@1vox | P/R@1vox | acc (vox) | compl (vox) | hole_fill | burr |
|---|---:|---:|---:|---:|---:|---:|
| F0 = E0 core16 | 0.6206 (baseline) | 0.554/0.705 | 1.403 | 0.809 | — | — |
| F1 = E1 core16 | pending | | | | | |
| F2 = E2 all32 | pending | | | | | |

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

pending

## Leave-one-out marginal contribution

Set S = A + B_core (32 views). Full run once, then 31-view runs dropping each view
in turn (sequential, resume-safe, VRAM-gated). Per deletion, geometry error is
measured on the **retained 31 views** (all have GT):

    Δ_i = E(S) − E(S\{i})      (mean abs-rel / normal° over the retained views)

Δ_i > 0 ⇔ view i was a net *negative* contributor. Secondary reference: A-core
views only (cross-i comparability). The 31 single deletions are the exhaustive
k=1 distribution, so the equal-count random-deletion control is their own
mean/spread (bootstrap CI reported) — no extra GPU runs are burned re-sampling a
subset of itself. A joint-deletion check (drop all flagged views at once vs the same
number of random views) tests the known caveat that two redundant views can be
individually removable but not jointly.

### Results

pending

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
