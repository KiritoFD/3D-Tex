# Benchmark cautions

Initial samples are **integration smoke tests only**: VGGT sees every example image. Do not report held-out PSNR from these scenes as leak-free benchmarking. For strict evaluation split images before VGGT; evaluate both 3DGS and 2D-SuGaR using common held-out cameras and identical image crops/exposure settings. DTU is suitable for scan-ground-truth mesh Chamfer / F-score; PSNR, SSIM, and LPIPS assess image appearance rather than surface accuracy. Keep image and geometry metrics separate.
