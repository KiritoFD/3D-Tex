# Geometry, UV and appearance evaluation — do not conflate SOTA axes

**Main rule:** split test views *before* VGGT-Ω/TSDF/mesh optimization. Both intrinsics and scale can be uncertain; heldout GT cameras can be used only for metric rendering after a train-only Sim(3) fit.

## Dataset protocol

- ETH3D `pipes/relief/kicker`: 12/25/25 reconstruction views, 2/6/6 held out; run camera alignment and geometry smoke test, no canonical UV GT. Original preprocessed images/laser scans are the public GT.
- DTU: 15 named reconstruction scans + official evaluation ObsMask + laser point clouds. Use exactly one chosen camera alignment, crop and evaluated threshold/protocol. Different published CD values can have different priors/evaluation scripts.
- For UV: object models with known `mesh+UV+texture` and controlled calibrated cameras. Compare the **same fixed geometry** if isolating UV error, and report any generated unseen surface texels separately.
- Do not use heldout images to refine the mesh, select iterations, calibrate depth confidence, choose bake weights, or decide UV settings. GT object mask as training input vs predicted mask must be declared explicitly.

## Metrics (report each separately)

| Axis | Measure |
|---|---|
| Cameras | aligned R error, center/translation error, reprojection residual; train-only Sim(3) |
| Depth | train-view multiview depth reprojection consistency; heldout depth error only if actual GT exists |
| Geometry | DTU CD (mm, smaller better), F-score @ explicit tolerance, normal consistency, completeness, hole count, disconnected/non-manifold edges, faces/vertices |
| Appearance | heldout PSNR, SSIM, LPIPS; same raster resolution, gamma, exposure, background, cameras |
| UV atlas | valid [0,1] coordinates, flipped/degenerate UV triangles, overlap/coverage, chart count, effective texel resolution, gutter |
| Seams | sample **same 3D edge** from both chart UVs, mean/95th percentile RGB/Lab delta; compare after mip level and real render |
| Reconstruction provenance | observed atlas pixels vs generated/unseen pixels with separate metrics; avoid over-reporting inferred material as measured |
| Runtime | per-stage time, host RAM, GPU peak, checkpoints and per-view resolution |

If only a synthetic wrapper/CPU shape check passed, report it as **integration validation only**. The new VGGT-Ω / TSDF / ExMesh / GeoSVR / UV code has not yet passed model inference or real 3D benchmark tests.

[Method choices](SOTA_2026.md) | [Full engineering design](MESH_UV_PIPELINE.md)
