"""Prepare an isolated source-GT diagnostic with the original inference view names."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-scene", default="data/scenes/nerf_lego16")
    parser.add_argument("--archive", default="data/.cache/nerf-syn/nerf_syn.zip")
    parser.add_argument("--scene", default="data/scenes/nerf_lego_gt16")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    source, scene = Path(args.input_scene), Path(args.scene)
    root = scene / "geometry" / "gt"
    for folder in (scene / "images", root / "foreground"):
        folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        truth = json.loads(archive.read("nerf_synthetic/lego/transforms_train.json"))
    poses = {Path(f["file_path"]).stem: f["transform_matrix"] for f in truth["frames"]}
    views = json.loads((source / "geometry/omega/cameras.json").read_text())["frames"]
    focal = 400 / np.tan(truth["camera_angle_x"] / 2)
    original = np.array([[focal, 0, 400], [0, focal, 400], [0, 0, 1]])
    affine = np.diag([args.resolution / 800, args.resolution / 800, 1])
    frames = []
    for view in views:
        name, stem = view["file"], Path(view["file"]).stem
        shutil.copy2(source / "images" / name, scene / "images" / name)
        with Image.open(scene / "images" / name) as image:
            mask = np.asarray(image.getchannel("A").resize((args.resolution,) * 2, Image.Resampling.NEAREST)) > 127
        np.save(root / "foreground" / f"{stem}.npy", mask)
        extrinsic = np.linalg.inv(np.asarray(poses[stem]) @ np.diag([1, -1, -1, 1]))
        frames.append({"frame": name, "file": name, "original_width": 800, "original_height": 800,
                       "model_width": args.resolution, "model_height": args.resolution,
                       "K_original": original.tolist(), "K_model": (affine @ original).tolist(),
                       "T_world_to_cam": extrinsic.tolist(), "pixel_affine_original_to_model": affine.tolist()})
    (root / "cameras.json").write_text(json.dumps({"camera_model": "PINHOLE", "source": "GT train cameras, diagnostic only", "frames": frames}, indent=2))
    (root / "manifest.json").write_text(json.dumps({"purpose": "source GT surface -> float depth -> same CPU TSDF",
                                                  "frames": len(frames), "input_resolution": [args.resolution] * 2,
                                                  "camera_source": "nerf_synthetic/lego/transforms_train.json", "model_input": False}, indent=2))
    print(f"Prepared {len(frames)} views in {scene}")


if __name__ == "__main__":
    main()
