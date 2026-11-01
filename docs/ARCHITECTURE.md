# Architecture and integration contract

## Goal and scope

Current objective: obtain a small, reproducible reconstruction/texture-export baseline before changing the VAE or training a Direct-UV diffusion model. Per-scene optimization only; no generalization training.

| Stage | Input | Output | Current implementation |
|---|---|---|---|
| Download/split | ETH3D DSLR undistorted archive + scan evaluation archive | data/scenes/eth3d_*/images, heldout, ground_truth | scripts/download_eth3d.py |
| VGGT | train images only | scene/sparse/{cameras,images,points3D}.bin | scripts/run_vggt.py -> vggt/demo_colmap.py |
| Bridge | VGGT COLMAP sparse files | scene/sparse/0/ symlinks/copies | scripts/prepare_scene.py |
| 3DGS | same VGGT cameras/points and scene images | optimized Gaussian model and renders | scripts/run_3dgs.py -> 3dgs/examples/simple_trainer.py |
| 2D-SuGaR | same VGGT cameras/points and scene images | optimized 2DGS, mesh, refined texture/UV | scripts/run_2dsugar.py -> 2dsugar/train.py |
| Evaluation | independent heldout poses/images + scan GT | formal metrics | **not yet implemented** |

## Geometry contract

- VGGT exports COLMAP coordinates via pycolmap, with camera intrinsics, world-to-camera poses and sparse 3D points. The bridge does not change any geometry, only mirrors binaries from sparse/ to sparse/0/.
- Both optimization branches must load the *same* COLMAP initialization and the same image files. Do not use official ETH3D ground_truth calibration, heldout images, or scan in reconstruction inputs.
- 2D-SuGaR's optional Metric3D route is a **different experiment**: it adds pretrained depth/normal priors. Compare this to no-prior explicitly rather than treating it as a constant.
- 2D-SuGaR extracts UV texture coordinates on the **mesh**, not on the Gaussian representation. A textured OBJ is an export format, not proof of seam consistency.

## Non-obvious data split

ETH3D dataset prep first holds out 2 / 6 / 6 images in pipes / relief / kicker: those 14 images never enter VGGT. It leaves 12 / 25 / 25 reconstruction images.

The current gsplat runner sets test_every=8; its COLMAP loader uses image indices modulo 8 to withhold an **internal** test slice from Gaussian optimization. The 2D-SuGaR runner currently passes --eval and its reader also withholds indices modulo 8. These internal slices are NOT the external ETH3D heldout 14 views, and the training subsets should be verified to match exactly before claiming controlled branch comparisons. All formal novel-view scores must use the external heldout images transformed using train-only camera alignment.

## Execution and limitations

    python3 scripts/download_eth3d.py --scenes pipes
    python3 scripts/run_vggt.py --scene eth3d_pipes --ba
    python3 scripts/run_3dgs.py --scene eth3d_pipes --steps 7000 --data-factor 4
    python3 scripts/run_2dsugar.py --scene eth3d_pipes --iterations 7000 --refine-iterations 2000 --resolution 4

- Current vggt/demo_colmap.py requests torch.cuda.get_device_capability() unconditionally; direct CPU invocation can fail before inference. Its amp and tracking code also assumes CUDA. A proper CPU port requires explicit changes and may be impractical for VGGT-1B plus many images.
- gsplat/2D-SuGaR training and differentiable Gaussian rasterizers require CUDA; running their CLI wrappers on CPU is not a replacement for actual training.
- The generated scene-scale TSDF parameters are heuristics (scripts/scene_scale.py), not guaranteed optimal for each object; validate geometry visually and against GT.
- The original upstream executables are large. Keep modifications to top-level scripts/adapters, preserving vendored provenance wherever possible.

## Current verification boundary

Existing GitHub Actions validate real ETH3D archive download/extraction and CPU dataset preparation. A separate synthetic CPU pipeline test verifies command construction and COLMAP bridging using a stub VGGT producer; it **does not** execute neural inference, CUDA compilation, rendering or mesh extraction. See [CPU_VALIDATION.md](CPU_VALIDATION.md).
