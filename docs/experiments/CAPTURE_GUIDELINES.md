# CAPTURE_GUIDELINES — how to photograph objects so VGGT-Omega does not degrade

Scope: evidence-backed capture rules derived from controlled experiments on the
NeRF-Synthetic Lego scene (GT geometry, VGGT-Omega 1B 512, frozen P0 evaluation
protocol). Every rule is labelled **[实测支持]** (measured), **[经验推断]**
(extrapolation from measured evidence), or **[未验证]** (not yet tested). Raw
experiment data: `docs/experiments/CAPTURE_EXPERIMENTS.md`; geometry context:
`docs/experiments/GEOMETRY_OPTIMIZATION.md`.

## View planning

| Rule | Evidence |
|---|---|
| Do NOT assume more views fix surface waviness. Between 8→64 uniform-azimuth views, depth error improves only 2.06%→1.32%, normal error stays ~32–33°, and F-score@1vox peaks at 32 views (0.5352) then *degrades* at 64 (0.4940) as precision collapses. | **[实测支持]** CAPTURE_EXPERIMENTS group 1 |
| Favour elevation/parallax diversity over azimuth density. The non-uniform 16-view dataset trajectory (elevations 9.7°–85.7°) beats all uniform ring subsets including 64 views (F 0.6206 vs 0.4940). Include high-elevation (top-down-ish) views of flat surfaces. | **[实测支持]** group 1; high-elevation benefit inferred from trajectory composition |
| Adjacent azimuth spacing ~12° (32 views/ring) is the saturation point for a single ring; denser (≤6°) adds coverage but hurts precision AND silhouette quality (burr cloud grows from a fat skirt at 8 views to body-wide spikes at 64). Do not densify one ring to chase detail. | **[实测支持]** group 1 + visual review |
| Thin rods, studs, holes and sharp rims are NOT recovered by adding views: sharp-region precision falls 0.28→0.19 from 8→64 views. If the object has thin structures, do not expect capture density alone to fix them — see GEOMETRY_OPTIMIZATION.md. | **[实测支持]** group 1 |
| Recommended starting point for small objects on a turntable: one ring at ~30–40° elevation with ~12° spacing plus a second ring at high elevation (~70–85°) with ~22–45° spacing; 24–40 views total. | **[经验推断]** combines the measured ring density with the anchor16 elevation spread; exact ring split untested (dual-ring experiment in progress) |
| Keep the whole object in frame at all times (all tested configurations did); very close detail shots without full-frame context are untested. | **[未验证]** |

## Failure modes the metrics cannot see

Standard abs_rel / F-score / PSNR have blind spots; review renders alongside
them (from the independent 34-render visual review):

| Failure | What the metrics say | What is actually there |
|---|---|---|
| Silhouette "burr cloud" (grows with view count; worst at 64) | nothing — fly-aways fall outside the GT band and surface samples; set64 even looks best on F/depth | body-wide spikes around the silhouette; use `burr_rate` and outlier rates |
| Cab-roof crater ("volcano") at every view count | hidden inside the global mean | systematic top-surface collapse; check orientation-split bins |
| Fluffy bridges over lattice openings | completeness IMPROVES | window/lattice structure destroyed; use structure survival (hole fill rate) |
| Back-facing sides much worse than front | hidden by the global mean | bin metrics by azimuth/elevation sector |

## Surface-type decision table

Density benefit is measured in CAPTURE_EXPERIMENTS.md group 4 (pixel-level
region attribution over ~400k pixels per config, uniform ring set8 → set64):

| Surface type | Density benefit (8→64 views) | Recommended density | Evidence |
|---|---:|---|---|
| Large flat planes (base plates) | **−47% abs_rel** (large) | densify freely up to 32 views/ring; benefits flatten past that | **[实测支持]** group 4 |
| Weak-texture / solid colour | **−46%** (large) | as above | **[实测支持]** group 4 |
| High-curvature regions | −33% (moderate) | ~32 views; more helps a little | **[实测支持]** group 4 |
| Edge / discontinuity band | −27% (moderate) | moderate; capture cannot fully fix this — see "failure regions" below | **[实测支持]** group 4 |
| Highlights / near-clip | −24% (small N) | avoid highlights at capture; not a capture-density fix | **[实测支持]** group 4 |
| Thin rods / sharp rims / studs | **−15% (weak)** | density alone does NOT recover these; use elevation diversity + depth-side refinement | **[实测支持]** groups 1 & 4 |
| Grooves / interior holes | n/a (no GT pixels inside) | n/a — evaluated via hole_fill_rate structure metric instead | — |

## VGGT-Ω failure regions and mechanism

Failure-region ranking (Spearman |rho| vs per-pixel abs_rel, group 4 set32):

| Rank | Covariate | rho | Failure mode |
|---:|---|---:|---|
| 1 | distance to depth discontinuity | −0.282 | fluffy bridges, burr cloud, melted rims — smoothing across edges |
| 2 | local curvature | +0.228 | rounds off high-curvature regions |
| 3 | parallax / incidence | +0.13 | grazing views slightly worse |
| 4 | image gradient (weak texture) | +0.10 | weak but real |
| 5+ | view count, parallax, silhouette distance | ≤0.13 | **not** the bottleneck once coverage ≥16 views |

**Mechanism**: the two dominant failure modes (edges, curvature) are *local
shape* properties, not *capture* properties — adding views barely changes
their error. This is a **network-prior / patch-level smoothing** failure, not
an information-shortage failure. Capture can fix coverage-side gaps (flat,
weak-texture); geometry-side refinement must fix the edge/curvature family.

## Image parameters

| Parameter | Recommendation | Evidence |
|---|---|---|
| Input resolution | Shoot at **400²–800²** with the object filling the frame; do NOT shoot below ~400². From 400² up, VGGT-Ω's own 512² model input dominates and extra pixels buy nothing (res400 / res512 / 800² anchor16 differ by ≤0.04% abs-rel, ≤0.4° normal). At 256² depth error grows +0.13pp and high-frequency "grit" RMS rises. | **[实测支持]** group 3 |
| Focus / motion blur | must be sharp; blur robustness untested | **[未验证]** |
| Exposure / highlights | avoid clipping (dataset-style studio lighting used in all experiments) | **[经验推断]** |
| Lighting direction / shadows | single fixed soft key light used in the dataset; direction ablation untested | **[未验证]** |
| Noise | dataset images are clean path traces; noise robustness untested | **[未验证]** |
| Object size in frame | object spans ≳60% of frame height in all tested configs | **[实测支持]** at this framing only |

## Red lines (measured failures)

- <8 views / spacing >45°: depth error doubles (1.24% → 2.06%) and completeness
  drops to 0.484 recall@1vox. **[实测支持]**
- 64-view single ring (≤6° spacing): *worse* F-score than 32 views; do not
  oversample one ring. **[实测支持]**

## Environment notes

- 64-view joint VGGT-Omega inference needs ≳6.5 GiB free VRAM at 512² (RTX 4070
  laptop: first attempt OOM-retried, 611 s); 32 views: 4.5 GiB, 20 s.
  **[实测支持]**
