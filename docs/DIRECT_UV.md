# Direct-UV research plan (NOT implemented yet)

## Research question

Can a pretrained **image diffusion model and image VAE** operate directly on an unwrapped mesh UV atlas while preserving 3D surface continuity, including seams, without training a video model or fitting a per-scene radiance field for every texture edit?

The existing VGGT/3DGS/2D-SuGaR code is only a geometry/UV baseline and source of reference renders. It is not the proposed generative model.

## Suggested minimum experiment

1. Obtain a small object mesh with a nontrivial multi-island UV atlas. Keep face indices, per-corner UV coordinates, 3D xyz, normals and an atlas-valid mask. 2D-SuGaR can supply a reconstructed mesh; for independent debugging also use an exact known mesh.
2. Rasterize per-texel surface information: 3D xyz, normal, triangle ID, barycentrics, UV island ID, coverage, and a matching map for texels adjacent in 3D but separated in UV.
3. First measure a pretrained image VAE **without any diffusion training**: UV round-trip RGB error, error within a narrow seam band, texture atlas mask behavior, and rendered-view PSNR/SSIM/LPIPS. Ordinary 2D VAE downsampling can merge unrelated surface patches separated only by a few atlas pixels.
4. Compare no-modification VAE with local geometry-aware VAE changes: seam-aware padding or cross-island token connections, based on 3D proximity and surface geodesic distance. Preserve normal 2D neighborhoods inside an island.
5. Only then train a small UV generator or adapter. If VAE reconstruction already destroys seam structure, DiT-only changes cannot fully repair it.

## Surface attention: the essential distinction

- **2D atlas distance** is fast and captures local texture within a UV island, but treats the same 3D surface split by a seam as far apart.
- **3D Euclidean distance** links nearby surface positions even across UV islands, but incorrectly connects spatially close yet physically disconnected surfaces (e.g. opposite sides of a thin shell).
- **Geodesic distance** follows mesh connectivity, avoiding shortcuts across different sheets, but graph shortest paths are expensive and imperfect around irregular triangulation or nonmanifold geometry.
- Use a **hybrid** neighborhood/positional bias (UV locality + 3D coordinates + geodesic or mesh adjacency). Start with sparse cross-seam links; don't implement all-pairs exact geodesics at high resolution.
- The image-VAE latent grid does not have a one-to-one surface correspondence. A downsampled latent token may cover several different triangles or islands. Explicitly aggregate rasterized xyz/normal/masks at each latent footprint and measure mixed-surface contamination.

Related design evidence: [DirectUV](https://arxiv.org/abs/2609.34651) uses surface-aware positional encoding; [TEXGen](https://arxiv.org/abs/2411.14740) interleaves UV convolution with point-cloud attention; [MatLat](https://arxiv.org/abs/2512.17302) emphasizes VAE locality. These are different interventions: do not assume modifying attention alone fixes a VAE that mixes disconnected surfaces.

## Evaluation: keep geometry separate from texture

| Axis | Minimum quantitative test |
|---|---|
| Geometry | camera alignment and pose error; mesh Chamfer/F-score where usable GT exists |
| Rendered texture | heldout view PSNR, SSIM, LPIPS with a fixed mesh/cameras/lighting protocol |
| Seam consistency | 3D-matched pairs across UV island borders, RGB and feature discrepancy; seam-band error |
| UV validity | in-bounds UVs, nondegenerate triangles, atlas coverage, overlap (where prohibited), island padding |
| Detail | high-frequency texture energy and perceptual detail, with fixed evaluation rasterization |

Avoid evaluating atlas pixels globally without masking and registering the UV layout. Two equally valid unwraps need not have pixelwise matching textures. ETH3D has camera/scan GT but not canonical complete UV texture GT; it cannot by itself validate learned UV completion.

## Scope order

**Now:** reconstruction baseline + CPU correctness checks. **Next:** known-mesh UV atlas + deterministic CPU raster/round-trip sanity. **Then:** VAE reconstruction experiment and cross-seam ablation. **Last:** generative fine-tuning. Do not introduce end-to-end Direct-UV training before checking the atlas/VAE information bottleneck.
