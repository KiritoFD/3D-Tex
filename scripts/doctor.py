"""Offline consistency check of retained upstream vendor entrypoints."""
from pathlib import Path
import shutil

from common import ROOT

REQUIRED = {
    "vggt": [
        "demo_colmap.py",
        "vggt/models/vggt.py",
        "LICENSE.txt",
    ],
    # Legacy research reference, no longer on active Mesh+UV path.
    "2dsugar": [
        "train.py",
        "gaussian_splatting_2d/train.py",
        "sugar_extractors/refined_mesh.py",
        "gaussian_splatting_2d/submodules/diff-surfel-rasterization/setup.py",
        "gaussian_splatting_2d/submodules/simple-knn/setup.py",
        "gaussian_splatting/submodules/diff-gaussian-rasterization/setup.py",
    ],
}

def main():
    for cmd in ("conda", "nvcc", "nvidia-smi"):
        print(cmd, shutil.which(cmd))
    missing = [
        str(ROOT / name / path)
        for name, paths in REQUIRED.items()
        for path in paths
        if not (ROOT / name / path).is_file()
    ]
    if missing:
        print("Missing retained source:", "; ".join(missing))
        raise SystemExit(1)
    if (ROOT / "3dgs").exists():
        print("Obsolete 3dgs vendor directory detected; mainline is no-Gaussian")
        raise SystemExit(1)
    print("Retained original-VGGT / legacy 2D-SuGaR source entrypoints OK.")
    print("VGGT-Omega + Mesh/UV pipeline NOT YET IMPLEMENTED; CUDA unverified.")

if __name__ == "__main__":
    main()
