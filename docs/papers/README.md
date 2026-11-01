# Annotated paper library (official links)

This is an index of primary research papers, not a vendored third-party source collection. PDF files are intentionally not committed because of size and redistribution/licensing differences. For the arXiv entries use:

    python3 scripts/download_papers.py --list
    python3 scripts/download_papers.py --papers vggt gaussian2d directuv matlat
    python3 scripts/download_papers.py --papers vggt directuv --dry-run

Downloaded PDFs and SHA256/size metadata stay in docs/papers/pdf/ (Git-ignored). Cite original papers, not these notes.

## A. Current baseline: camera estimation -> splatting -> mesh

| Paper | Primary source/PDF | Why we care |
|---|---|---|
| VGGT, CVPR 2025 | [arXiv:2503.11651](https://arxiv.org/pdf/2503.11651) | Feed-forward camera, point/depth, tracking initialization; check BA and calibration error. |
| 3D Gaussian Splatting, TOG 2023 | [arXiv:2308.04079](https://arxiv.org/pdf/2308.04079) | Fast volumetric rendering baseline, **not** an ideal editable surface representation. |
| gsplat, JMLR 2025 | [arXiv:2409.06765](https://arxiv.org/pdf/2409.06765) | Optimized PyTorch/CUDA Gaussian rasterization and per-scene training software. |
| 2D Gaussian Splatting, SIGGRAPH 2024 | [arXiv:2403.17888](https://arxiv.org/pdf/2403.17888) | Oriented surfel disks and depth/normal regularization for consistent surfaces. |
| SuGaR, CVPR 2024 | [arXiv:2311.12775](https://arxiv.org/pdf/2311.12775) | Surface alignment, mesh extraction and Gaussian-on-mesh refinement. |
| 2D-SuGaR, Eurographics 2026 | [Eurographics publisher/PDF](https://diglib.eg.org/items/876e3a03-c494-483b-80dc-b43ebd0e76a7) | 2DGS with monocular depth/normal priors, pruning and joint mesh-Gaussian refinement; direct upstream for 2dsugar/. |

## B. Primary topic: direct UV textures and seam continuity

| Paper | Primary source/PDF | Mechanism / implication |
|---|---|---|
| DirectUV, 2026 preprint | [arXiv:2609.34651](https://arxiv.org/pdf/2609.34651) | UV-space diffusion with **surface-aware positional encoding**; core comparison for 3D-aware attention across atlas seams. |
| TEXGen, SIGGRAPH Asia 2024 | [arXiv:2411.14740](https://arxiv.org/pdf/2411.14740) | Direct UV diffusion, alternating UV convolutions and point-cloud attention; major training-from-scratch baseline. |
| Point-UV Diffusion, ICCV 2023 | [arXiv:2308.10490](https://arxiv.org/pdf/2308.10490) | Point-level coarse texture first, UV refinement second; global structure before local detail. |
| Paint3D, CVPR 2024 | [arXiv:2312.13913](https://arxiv.org/pdf/2312.13913) | Multi-view coarse texture + UV inpainting/refinement; useful non-direct-UV baseline. |
| TexFusion, ICCV 2023 | [arXiv:2310.13772](https://arxiv.org/pdf/2310.13772) | Share latent texture during multi-view diffusion to improve 3D consistency without new UV-native diffusion training. |
| MatLat, CVPR 2026 | [arXiv:2512.17302](https://arxiv.org/pdf/2512.17302) | Image-generation priors, PBR VAE adaptation and **VAE-locality** requirement; directly relevant to seam mixing before DiT. |

## Reading priorities and cautions

1. **DirectUV + TEXGen + MatLat**: the most relevant trio for a geometry-aware UV latent model. Their contributions differ: positional encoding, point/UV mixing, and latent locality.
2. **VGGT + 2DGS + 2D-SuGaR**: understand the actual baseline code used by this repository and what geometry/UV is passed downstream.
3. **Point-UV + Paint3D + TexFusion**: contrast directly learned UV mapping with multi-view projection/inpainting.

Keep the scientific claims separate: a seam-continuous *render* does not automatically mean an atlas is free of seam-color discrepancy, and UV generation fidelity depends on the chosen parameterization and VAE decoder. See [DIRECT_UV.md](../DIRECT_UV.md).
