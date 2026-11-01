# Research papers: from VGGT-Ω to true Mesh, UV and materials

This directory holds **annotated first-party paper links and reading priorities**. PDFs are **not committed**. For arXiv papers run `python3 scripts/download_papers.py --list` and `python3 scripts/download_papers.py --papers omega exmesh exmeshpp geosvr --dry-run`. Data/ckpt/source licenses differ.

**For the complete author/venue/performance comparison** read [../SOTA_2026.md](../SOTA_2026.md). **For implementation** read [../MESH_UV_PIPELINE.md](../MESH_UV_PIPELINE.md).

## Core no-Gaussian pipeline

| Paper | Publication | Official paper/source | Relevance |
|---|---|---|---|
| VGGT-Ω — Wang et al. | CVPR 2026 Oral/Best Paper Finalist | [arXiv:2605.15195](https://arxiv.org/abs/2605.15195), [code](https://github.com/facebookresearch/vggt-omega) | joint camera + depth; point map derived by unprojection |
| GeoSVR — Li et al. | NeurIPS 2025 Spotlight | [arXiv:2509.18090](https://arxiv.org/html/2509.18090), [code](https://github.com/Fictionarry/GeoSVR) | sparse voxel geometry baseline, DTU CD 0.47 |
| ExMesh — Fan et al. | CVPR 2026 | [arXiv:2606.07288](https://arxiv.org/html/2606.07288), [code](https://github.com/Fan-Treasure/ExMesh) | topology split/merge, differentiable triangle mesh, UV map; **default PGSR bootstrap** |
| ExMesh++ — Fan et al. | **2026 arXiv preprint** | [arXiv:2608.24109](https://arxiv.org/abs/2608.24109), [project](https://fan-treasure.github.io/ExMeshpp_page/) | topology-adaptive Mesh+UV followed by fixed-carrier UV-PBR and lighting optimization |
| MeshSplatting — Held et al. | CVPR 2026 | [arXiv:2512.06818](https://arxiv.org/abs/2512.06818), [code](https://github.com/meshsplatting/mesh-splatting) | triangle soup → restricted Delaunay → opaque mesh, not 3DGS |
| DMesh++ — Son et al. | ICCV 2025 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Son_DMesh_An_Efficient_Differentiable_Mesh_for_Complex_Shapes_ICCV_2025_paper.html), [project](https://sonsang.github.io/dmesh2-project/) | differentiable mesh topology for complex point shapes |
| Open3D TSDF fusion | classical method/software | [docs](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html) | fast simple CPU baseline, explicit Mesh output |
| xatlas | UV parameterization utility | [source](https://github.com/jpcy/xatlas) | charting, parametrization, packing, per-corner UV |

## UV texture and PBR

| Paper | Publication | Link | Relevance |
|---|---|---|---|
| MV2UV — Zheng Zhang et al. | CVPR 2026 | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_MV2UV_Generating_High-quality_UV_Texture_Maps_with_Multiview_Prompts_CVPR_2026_paper.pdf) | native UV generation + multi-view XYZ attention |
| MatLat — Yeo et al. | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Yeo_MatLat_Material_Latent_Space_for_PBR_Texture_Generation_CVPR_2026_paper.html) | pretrained VAE locality and PBR material latent |
| NaTex — Lai et al. | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Lai_NaTex_Seamless_Texture_Generation_as_Latent_Color_Diffusion_CVPR_2026_paper.html) | 3D color-point latent diffusion from scratch |
| Lafite — Chen et al. | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Lafite_A_Generative_Latent_Field_for_3D_Native_Texturing_CVPR_2026_paper.html) | sparse 3D latent color field VAE + flow |
| Intrinsic Image Fusion — Kocsis et al. | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Kocsis_Intrinsic_Image_Fusion_for_Multi-View_3D_Material_Reconstruction_CVPR_2026_paper.html) | physically meaningful material/lighting decomposition for captured objects |
| TEXTRIX — Zeng et al. | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_TEXTRIX_Latent_Attribute_Grid_for_Native_Texture_Generation_and_Beyond_CVPR_2026_paper.html) | 3D attribute latent DiT instead of 2D atlas |
| TEXGen — Yang et al. | 2024 | [arXiv:2411.14740](https://arxiv.org/abs/2411.14740) | alternating UV convolutions and 3D point attention |
| DirectUV | 2026 preprint | [arXiv:2609.34651](https://arxiv.org/abs/2609.34651) | UV-space 3D surface-aware conditioning |
| Point-UV Diffusion | ICCV 2023 | [arXiv:2308.10490](https://arxiv.org/abs/2308.10490) | point-level guidance and UV refinement |
| Paint3D | CVPR 2024 | [arXiv:2312.13913](https://arxiv.org/abs/2312.13913) | multi-view projection and UV inpainting |
| TexFusion | ICCV 2023 | [arXiv:2310.13772](https://arxiv.org/abs/2310.13772) | shared texture latents |

## Retired Gaussian branches (study only)

- [2D-SuGaR, Eurographics **2026 Short Papers**](https://diglib.eg.org/items/876e3a03-c494-483b-80dc-b43ebd0e76a7), not an established overall 2026 SOTA; published DTU CD 0.67 on its protocol.
- [MILo, SIGGRAPH Asia 2025 / ACM TOG](https://anttwo.github.io/milo/): Gaussian + in-the-loop mesh refinement.
- [OMeGa, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Cao_OMeGa_Joint_Optimization_of_Explicit_Meshes_and_Gaussian_Splats_for_WACV_2026_paper.html): explicit mesh + 2DGS jointly optimized.
- [2DGS, SIGGRAPH 2024](https://arxiv.org/abs/2403.17888), [SuGaR, CVPR 2024](https://arxiv.org/abs/2311.12775): historical context only. No standalone 3DGS branch in current codebase.

## Caution

Values like **DTU CD 0.47 (GeoSVR)**, **0.58 (ExMesh)**, **Mip-NeRF360 PSNR +0.69 dB (MeshSplatting)** and **FID 24.4/26.4 (MV2UV)** measure *different tasks*. Never sort them into one universal leaderboard.

Downloaded PDF caches and checksum manifests live under `docs/papers/pdf/` (Git-ignored), never in Git. The downloader checks file signatures and enforces a byte budget; a passing PDF test is not proof of model reproduction.
