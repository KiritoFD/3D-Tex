# ETH3D benchmark: small public scenes (<10GB)

## Selected scenes

| Scene | Images | Undistorted photos archive | Laser evaluation scan archive |
|---|---:|---:|---:|
| pipes | 14 | 0.1 GB | 0.1 GB |
| relief | 31 | 0.3 GB | 0.2 GB |
| kicker | 31 | 0.4 GB | 0.2 GB |

Total **~1.3 GB compressed**, using official rounded sizes. The downloader limits aggregate cached archives to **3 GiB** by default. Raw archives, their expanded temporary files and working copies are different disk consumers; the selected subset is well below the requested 10GB dataset target under normal extraction. Model weights/checkpoints are not part of dataset size.

Official source: https://www.eth3d.net/datasets
Official camera/GT format: https://www.eth3d.net/documentation

## Commands

Install the system command: sudo apt-get install p7zip-full

    # Download no data; list exact URLs/selected scenes
    python3 scripts/download_eth3d.py --dry-run

    # Download the full small validation set with laser GT
    python3 scripts/download_eth3d.py --scenes pipes relief kicker

    # Shortest real-data smoke test (~0.2GB compressed)
    python3 scripts/download_eth3d.py --scenes pipes

    # Remove cached archives to save disk
    python3 scripts/download_eth3d.py --scenes pipes relief kicker --delete-archives

Each scene is downloaded using the official per-scene undistorted DSLR images archive plus the evaluation laser scan archive. No user registration is required. The script verifies 7z integrity, calculates SHA256 archive hashes, extracts and detects calibration/images, and creates a deterministic split **before VGGT runs**.

Default split: sort filenames and hold out indices 4, 9, 14, ... (zero-indexed). For the official 14/31/31 view counts, that yields 12/25/25 input images and 2/6/6 held out, i.e. **62 reconstruction images + 14 test images**. The per-scene manifest records exactly which files went into each subset.

## Layout and pipeline

    data/scenes/eth3d_pipes/
      images/                       # only TRAIN (VGGT/3DGS/2D-SuGaR input)
      heldout/images/               # not included in VGGT or Gaussian optimization
      ground_truth/calibration/
        cameras.txt                 # official PINHOLE intrinsics
        images.txt                  # official camera poses
        poses.json                  # parsed camera GT
      ground_truth/scan_eval/*.ply  # laser geometry GT; not model input
      dataset_manifest.json
      sparse/                       # later, VGGT output and 2D-SuGaR sparse/0 bridge

    python3 scripts/run_vggt.py --scene eth3d_pipes --ba
    python3 scripts/run_3dgs.py --scene eth3d_pipes --steps 7000 --data-factor 4
    python3 scripts/run_2dsugar.py --scene eth3d_pipes --iterations 7000 --refine-iterations 2000 --resolution 4

7k/2k steps and quarter-resolution settings are **pipeline smoke checks**, NOT converged quality experiments. Increase iterations for formal comparisons. Both branches use the identical VGGT camera estimates and sparse initialization.

## Measurement protocol

1. Camera: align VGGT predicted cameras to ETH3D GT using similarity transform based only on training views; measure aligned camera-center and rotation errors.
2. Novel-view appearance: render the heldout images with true heldout cameras transformed into the predicted reconstruction frame using *train-only* similarity alignment. Metrics: PSNR, SSIM, LPIPS. The current repo **prepares** these inputs, but a held-out 3DGS/mesh renderer/evaluator is not yet wired.
3. Geometry: evaluate exported 2D-SuGaR mesh against the ETH3D evaluation scan with the official scan alignment/visibility protocol. Raw unaligned Chamfer is **not** an official ETH3D result.
4. Texture repair: this is not UV GT. ETH3D lasers have no complete UV texture target. Add a textured-asset dataset in the later UV phase.

**Leakage guard:** ground_truth/ and heldout/ are not read by VGGT and should not be copied under images/ or sparse/. The built-in gsplat test_every split is NOT the external heldout benchmark.

A CPU-only GitHub Action validates preparation against the real pipes download (the smallest scene), plus synthetic archive-format and split tests.
