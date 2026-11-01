# Dependency separation

VGGT, gsplat, and 2D-SuGaR require independent Torch/CUDA environments. Install using `bash scripts/install_envs.sh all`; use a recent NVIDIA driver, Linux, CUDA compiler, and adequate disk space. 2D-SuGaR needs legacy CUDA 11.8 for its extensions; gsplat has a newer PyTorch dependency. This is a smoke-test pipeline, **not** verified GPU training. VGGT writes `sparse/`; 2D-SuGaR expects `sparse/0/`. The adapter creates symlinks without changing reconstruction geometry.
