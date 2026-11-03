"""Prepare a fresh Omega-gauge TSDF using only predicted geometry and input alpha."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from geometry_masks import load_masks
from common import scene_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="nerf_lego16")
    p.add_argument("--out", default="outputs/m2/prediction_init")
    args = p.parse_args()
    scene, out = scene_path(args.scene), Path(args.out)
    omega = scene / "geometry/omega"
    cameras = json.loads((omega / "cameras.json").read_text())
    masks = load_masks(scene, cameras["frames"], "alpha")
    filtered = out / "filtered"
    (filtered / "depth").mkdir(parents=True, exist_ok=True)
    confidence = out / "omega/confidence"
    confidence.mkdir(parents=True, exist_ok=True)
    for frame, mask in zip(cameras["frames"], masks):
        stem = Path(frame["frame"]).stem
        depth = np.load(omega / "depth" / f"{stem}.npy").squeeze()
        np.save(filtered / "depth" / f"{stem}.npy", np.where(mask, depth, 0).astype(np.float32))
        shutil.copyfile(omega / "confidence" / f"{stem}.npy", confidence / f"{stem}.npy")
    (filtered / "cameras.json").write_text(json.dumps(cameras, indent=2))
    (out / "provenance.json").write_text(json.dumps({"gt_geometry_access": False,
        "mask_source": "dataset RGBA input annotations", "cameras": "predicted",
        "roi": "predicted masked points", "carving": False,
        "note": "native prediction-mode feasibility control; different initialisation from diagnostic v35c30"}, indent=2))
    subprocess.run([sys.executable, "scripts/tsdf_mesh.py", "--scene", args.scene,
                    "--filtered", str(filtered), "--out", str(out / "mesh"),
                    "--weight-mode", "confidence", "--stride", "1"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
