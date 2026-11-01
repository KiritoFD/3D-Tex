# 3D-Tex documentation

## What is implemented

This repository currently builds a **per-scene reconstruction baseline**, not a trained Direct-UV texture generator:

    RGB images (training views only)
         |
         v
    VGGT -> cameras + point cloud -> COLMAP sparse/
         |                         |
         |                         +-> 3DGS / gsplat -> Gaussian renders
         |
         +-> 2D-SuGaR (2DGS -> mesh refinement -> textured OBJ/UV)

The 3DGS and 2D-SuGaR branches are **parallel experiments**. 3DGS is not an input to 2D-SuGaR. Original VGGT, gsplat and 2D-SuGaR source is retained in separate first-level directories, not submodules.

## Read in order

1. [ARCHITECTURE.md](ARCHITECTURE.md): current interfaces, coordinate conventions, actual optimization data and failure points.
2. [DATASETS.md](DATASETS.md): the three small ETH3D scenes, download recipe and strict heldout split.
3. [CPU_VALIDATION.md](CPU_VALIDATION.md): test matrix, exact CPU commands and GPU-only limitations.
4. [EVALUATION.md](EVALUATION.md): metrics and the train/test leakage boundary.
5. [DIRECT_UV.md](DIRECT_UV.md): the next research pipeline (mesh/UV representation, seams, VAE and attention).
6. [papers/README.md](papers/README.md): annotated paper reading list, official links and optional local PDF downloader.
7. [UPSTREAMS.md](UPSTREAMS.md), [COMPATIBILITY.md](COMPATIBILITY.md): source pins and install cautions.

## Reproduce the lightweight checks

    python3 -m unittest discover -s tests -v
    python3 scripts/download_eth3d.py --dry-run
    python3 scripts/download_papers.py --list
    python3 scripts/download_papers.py --papers vggt directuv --dry-run
    python3 scripts/doctor.py

These checks require no model weights or GPU. Model inference/training still requires GPU-specific installations and has **not** been proven by the CPU smoke tests.

For reproducibility, paper PDFs are **not** committed. Run scripts/download_papers.py to cache the selected open-access PDFs under docs/papers/pdf/ (Git-ignored). Read or cite papers using the original publishers and respect their licenses.
