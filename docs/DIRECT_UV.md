# Direct UV / geometry-aware VAE research (downstream of stable Mesh+UV)

**Prerequisite:** complete the proposed **VGGT-Ω → Mesh → UV Atlas + observed texture** pipeline first; the new code does not exist yet. [Engineering spec](MESH_UV_PIPELINE.md) | [Published methods](SOTA_2026.md).

## What is being learned?

The deterministic geometric baseline creates `mesh.obj` + `face_uv` + `atlas_observed` + `observed_mask`. We want to use a pretrained image VAE/diffusion model to repair *unobserved* texels and produce seam-consistent texture without hallucinating on already well-measured surfaces.

**Scientific split:** 1) geometry error; 2) UV parameterization error; 3) color fusion/visibility error; 4) VAE latent mixing; 5) diffusion conditioning/generation error. Never change all five simultaneously.

## Geometry-aware UV VAE experiment

1. First with a known GT mesh and atlas containing **multiple disconnected UV islands**, run the frozen original pretrained image VAE encoder/decoder; measure seam-pair RGB error, seam-band LPIPS and heldout rendered PSNR. Keep only valid texels, record chart ID.
2. Rasterize `XYZ`, surface normal, per-face and per-chart ID, barycentric coordinates and validity mask at UV resolution. Build **cross-UV seam pairs** that share the same 3D surface edge (not pixel adjacency).
3. Diagnose downsampling. A latent token covers multiple UV texels that may map to distant 3D surfaces; an ordinary strided 2D convolution can mix them before any DiT attention. Report per-latent surface mixing, not just endpoint PSNR.
4. Try minimal geometry-aware local VAE modifications: masked local kernels, seam-pair latent links, or chart-aware downsampling. Do not use naive Euclidean proximity alone: spatially nearby but disconnected sheets would be connected incorrectly. Use graph adjacency / approximate geodesic as a supplementary bias.
5. Compare no VAE change / VAE local modification / DiT seam attention only. Fine-tune each under equal data and compute.
6. Train texture repair only on heldout UV masks; hard/soft preserve observed data. Evaluate real mesh rendering, not atlas 2D MSE across different unwrappings.

## Relevant high-priority papers

- [MatLat, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yeo_MatLat_Material_Latent_Space_for_PBR_Texture_Generation_CVPR_2026_paper.html): pretrained image model + VAE locality + correspondence-aware attention. **Closest VAE locality motivation.**
- [MV2UV, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_MV2UV_Generating_High-quality_UV_Texture_Maps_with_Multiview_Prompts_CVPR_2026_paper.pdf): UV texture generation conditioned by multi-view observations and 3D XYZ correspondence.
- [ExMesh++, arXiv 2026](https://arxiv.org/abs/2608.24109): fixed mesh/UV carrier followed by PBR and lighting recovery. **Not verified peer-reviewed or fully open-sourced.**
- [NaTex, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lai_NaTex_Seamless_Texture_Generation_as_Latent_Color_Diffusion_CVPR_2026_paper.html) and [Lafite, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Lafite_A_Generative_Latent_Field_for_3D_Native_Texturing_CVPR_2026_paper.html): alternative *3D-native* latent representations; strong seam motivation, substantially heavier model work.
- [TEXGen, 2024](https://arxiv.org/abs/2411.14740): UV CNN + point 3D attention; [DirectUV, 2026 preprint](https://arxiv.org/abs/2609.34651): explicitly UV-native diffusion. Verify each paper's training/compute protocol before borrowing numbers.

## Evaluation

UV semantics: [0,1] validity, per-corner indexing, texel coverage, edge chart pairing, overlaps, gutter/mip padding. Texture: PSNR/SSIM/LPIPS on fixed heldout cameras, visual seams, observed vs generated split. Geometry: CD/F-score evaluated separately; see [EVALUATION.md](EVALUATION.md).

**Milestone:** only consider a UV DiT fine-tuning experiment after fixed-mesh VAE reconstruction and seam pairing tests pass.
