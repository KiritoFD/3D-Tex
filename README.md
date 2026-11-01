# 3D-Tex

**Static-scene capture -> VGGT cameras/points -> 3DGS rendering + 2D-SuGaR mesh/UV.**

This repository contains **selected, vendored upstream source code** under three peer directories. There are **no Git submodules** and no generalization training. Each scene is optimized independently.

```text
3D-Tex/
├── vggt/                    # VGGT inference + COLMAP exporter
├── 3dgs/                    # gsplat rendering core + COLMAP trainer
├── 2dsugar/                 # 2DGS, SuGaR and required CUDA extensions
├── scripts/                 # unified setup / scene preparation / run scripts
├── configs/upstreams.json   # pinned upstream commits
├── configs/vendored_manifest.json  # source provenance (after vendoring)
├── tests/                   # CPU-only data/CLI tests
├── data/scenes/<scene>/images/
├── outputs/
└── docs/
```

### Initial setup (Linux, CUDA GPU; single RTX 4090 is sufficient for small scenes)

Source files are committed to this repository; cloning **does not** download upstream repos or datasets.

```bash
git clone https://github.com/KiritoFD/3D-Tex.git
cd 3D-Tex
bash scripts/install_envs.sh all
python3 scripts/doctor.py
```

The three isolated Conda environments intentionally retain different upstream PyTorch/CUDA dependencies. For 2D-SuGaR, a compatible CUDA toolkit / compiler is required to build the vendored kernels. CUDA training has **not** been tested in the CPU-only preparation environment.

### Tiny data and two independent reconstruction branches

```bash
# Fetch sample photos only; no training data or pretrained models checked into Git.
python3 scripts/get_samples.py --scenes fern flower
python3 scripts/run_vggt.py --scene fern --ba
python3 scripts/run_3dgs.py --scene fern --steps 7000
python3 scripts/run_2dsugar.py --scene fern --iterations 7000 --refine-iterations 2000
```

VGGT's pretrained weights are downloaded by Hugging Face at first inference; `--ba` additionally downloads VGGSfM / tracking weights. All data and model outputs remain Git-ignored.

**Unified interface:** VGGT writes `data/scenes/<scene>/sparse/` (COLMAP). The adapter mirrors its files into `sparse/0/` for 2D-SuGaR without changing cameras or point coordinates. Both branches consume the same geometry and input images.

The no-prior 2D-SuGaR path is used for the first comparison. The paper's Metric3D depth/normal prior is available with `--priors metric3d`, but mixes an additional pretrained geometry model into the comparison and requires PNG inputs.

### Validation and evaluation scope

```bash
python3 -m unittest discover -s tests -v
python3 scripts/doctor.py
python3 scripts/run_vggt.py --help
python3 scripts/run_3dgs.py --help
python3 scripts/run_2dsugar.py --help
```

**Do not report the tiny sample PSNR as a leak-free new-view benchmark.** The initial VGGT estimate sees the provided sample photos. For controlled benchmarks later, split views *before* reconstruction, or use independently calibrated test cameras and DTU geometry GT. See `docs/EVALUATION.md`.

### Vendoring / updating

`scripts/vendor_sources.py` clones the three pinned repos in a temporary directory, retains the code paths needed by this pipeline, vendors the two Gaussian rasterizers + simple-knn code, and records a source manifest. It **does not** retain their Git history, images, docs, demo UI, training datasets or model weights. The GitHub Actions source-vendoring workflow performs this once when configured; rerun manually only when updating pinned upstream commits. See `docs/UPSTREAMS.md`.

Upstream copyrights and licenses are preserved in the corresponding directories; third-party weights and code may have separate usage terms.
