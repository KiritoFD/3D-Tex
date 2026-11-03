"""Prepare a fresh Omega-gauge TSDF using only predicted geometry and input alpha."""
import argparse
import hashlib
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
from fusion.observation_weights import observation_factor


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="nerf_lego16")
    p.add_argument("--out", default="outputs/m2/prediction_init")
    p.add_argument("--observation-mode", choices=["confidence", "incidence", "boundary", "combined"],
                   default="confidence")
    args = p.parse_args()
    scene, out = scene_path(args.scene), Path(args.out)
    omega = scene / "geometry/omega"
    cameras = json.loads((omega / "cameras.json").read_text())
    masks = load_masks(scene, cameras["frames"], "alpha")
    filtered = out / "filtered"
    (filtered / "depth").mkdir(parents=True, exist_ok=True)
    confidence = out / "omega/confidence"
    confidence.mkdir(parents=True, exist_ok=True)
    input_hashes = {str(omega / "cameras.json"): sha256(omega / "cameras.json")}
    for frame, mask in zip(cameras["frames"], masks):
        stem = Path(frame["frame"]).stem
        depth = np.load(omega / "depth" / f"{stem}.npy").squeeze()
        np.save(filtered / "depth" / f"{stem}.npy", np.where(mask, depth, 0).astype(np.float32))
        source = omega / "confidence" / f"{stem}.npy"
        for path in (source, omega / "depth" / f"{stem}.npy", scene / "images" / frame["file"]):
            input_hashes[str(path)] = sha256(path)
        if args.observation_mode == "confidence":
            shutil.copyfile(source, confidence / f"{stem}.npy")
        else:
            factor = observation_factor(depth, np.asarray(frame["K_model"]), mask,
                                        args.observation_mode)
            np.save(confidence / f"{stem}.npy", np.load(source).squeeze() * factor)
    (filtered / "cameras.json").write_text(json.dumps(cameras, indent=2))
    (out / "provenance.json").write_text(json.dumps({"gt_geometry_access": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": input_hashes,
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                         (Path(__file__).resolve(), ROOT / "scripts/fusion/observation_weights.py",
                          ROOT / "scripts/tsdf_mesh.py")},
        "mask_source": "dataset RGBA input annotations", "cameras": "predicted",
        "roi": "predicted masked points", "carving": False,
        "observation_mode": args.observation_mode,
        "observation_parameters": {"boundary_pixels": 3.0, "floor": 0.25},
        "note": "native prediction-mode feasibility control; different initialisation from diagnostic v35c30"}, indent=2))
    subprocess.run([sys.executable, "scripts/tsdf_mesh.py", "--scene", args.scene,
                    "--filtered", str(filtered), "--out", str(out / "mesh"),
                    "--weight-mode", "confidence", "--stride", "1"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
