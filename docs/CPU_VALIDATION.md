# CPU and download validation status (2026-11-01)

## What is already verified remotely

- [GitHub Actions ETH3D 3-scene run](https://github.com/KiritoFD/3D-Tex/actions/runs/41308468752) completed successfully before this documentation update.
- Its offline job ran Python source compilation, unittest discovery, ETH3D dry-run CLI and vendored-source doctor checks.
- Its public-archive job installed 7-Zip, downloaded the *real* pipes, relief and kicker undistorted image archives plus evaluation scan archives, extracted the files, and asserted exact reconstruction/heldout counts, GT poses, scan PLY presence, disjoint filenames, and total unpacked size under 10 GiB.
- The previous recorded dataset totals were approximately 1.254 GiB of compressed archives and 1.901 GiB of extracted scene files (data only), as documented in DATASETS.md.

## CPU verification introduced here

    python3 -m unittest discover -s tests -v
    python3 scripts/download_eth3d.py --scenes pipes relief kicker --dry-run
    python3 scripts/download_papers.py --list
    python3 scripts/download_papers.py --papers vggt directuv --dry-run
    bash -n scripts/install_envs.sh scripts/run_all.sh
    python3 scripts/doctor.py

Additional regression tests cover (i) offline HTTP mock downloads and Range resume behavior; (ii) PDF content/byte limits; (iii) a synthetic VGGT COLMAP export driving prepare_scene and both pipeline wrappers with captured commands. These tests require no real camera model or GPU.

## What these tests do NOT demonstrate

- No neural VGGT forward pass, BA / VGGSfM tracking, or checkpoint download on a CPU-only host.
- No CUDA extension build, gsplat optimization, 2DGS render, mesh extraction or UV baking.
- No quantitative ETH3D evaluation of predicted cameras, novel views or geometry.
- No Direct-UV model training or VAE cross-seam comparison.

**Do not mark the whole pipeline E2E-green based on the synthetic wrapper test.** The next hardware acceptance gate should execute a 3-view VGGT pass then reduced-iteration 3DGS and 2D-SuGaR on a CUDA workstation, record logs, and check actual outputs (sparse files, checkpoints, mesh OBJ/MTL/texture and render metrics). The CPU tests only validate structural integration and catch regressions before that gate.
