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

| Surface type | Views (min) | Angular spacing | Extra close-ups | Evidence |
|---|---|---|---|---|
| Large flat planes (base plates) | ~24–32 | ≤12° | no | **[经验推断]** flat regions are the least damaged (flat accuracy ≈1.1–1.7 vox vs 2–3 vox sharp) |
| High-curvature regions | ~32 | ≤12° | unknown | **[经验推断]** from global depth-error scaling |
| Thin rods / thin walls / rims | more views do NOT help precision (measured); keep ≤~12° spacing and prefer elevation diversity | — | **[未验证]** |
| Studs / holes / small concave features | as above; expect melting without geometry post-refinement | — | **[实测支持]** structure split in group 1 |
| Weak-texture / glossy surfaces | untested | untested | **[未验证]** |
| Deep holes / occluded grooves | untested | untested | **[未验证]** |

## Image parameters

| Parameter | Recommendation | Evidence |
|---|---|---|
| Input resolution | VGGT runs at 512² model input; captures at ≥800² with the object filling the frame (all experiments). Sub-512 capture and upscaling are untested. | **[实测支持]** at ≥800²; **[未验证]** below |
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
