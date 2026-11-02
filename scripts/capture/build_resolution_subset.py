"""Build resolution-axis capture scenes: identical views/trajectory, rescaled images.

Uses the existing anchor16 view set and its GT geometry (depth/foreground/cameras
at the 512 model grid are resolution-independent); only the input images are
rescaled, and K_original is scaled to stay consistent with the new resolution.

    .venv/Scripts/python.exe scripts/capture/build_resolution_subset.py --resolutions 256,400,512
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-source", default="data/scenes/nerf_lego16",
                        help="scene with the reference images (800^2 originals)")
    parser.add_argument("--gt-source", default="data/scenes/nerf_lego_gt16/geometry/gt")
    parser.add_argument("--resolutions", default="256,400,512")
    parser.add_argument("--out-template", default="data/scenes/capture_res{R}")
    parser.add_argument("--name-template", default="res{R}")
    args = parser.parse_args()
    views = Path(args.views_source)
    gt = Path(args.gt_source)
    cameras = json.loads((gt / "cameras.json").read_text())["frames"]
    base_w = cameras[0]["original_width"]
    for res in (int(r) for r in args.resolutions.split(",")):
        scale = res / base_w
        out = Path(args.out_template.format(R=res))
        name = args.name_template.format(R=res)
        (out / "images").mkdir(parents=True, exist_ok=True)
        gt_out = out / "geometry" / "gt"
        for sub in ("depth", "foreground"):
            dst = gt_out / sub
            dst.mkdir(parents=True, exist_ok=True)
            for src_file in (gt / sub).glob("*.npy"):
                shutil.copyfile(src_file, dst / src_file.name)
        frames = []
        for frame in cameras:
            stem = Path(frame["frame"]).stem
            image = cv2.imread(str(views / "images" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(image, (res, res), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "images" / f"{stem}.png"), resized)
            k_original = (np.asarray(frame["K_original"]) * scale).tolist()
            entry = dict(frame)
            entry.update({"original_width": res, "original_height": res,
                          "K_original": k_original,
                          "pixel_affine_original_to_model": (
                              np.asarray(frame["pixel_affine_original_to_model"]) / scale).tolist()})
            frames.append(entry)
        manifest = {"scheme": name,
                    "params": {"resolution": res, "resize": "cv2.INTER_AREA from 800x800 originals",
                               "views_source": str(views), "views": len(frames)},
                    "note": "same cameras/trajectory/GT as anchor16; only the input image "
                            "resolution is rescaled. GT depth/foreground shared from gt-source."}
        (gt_out / "cameras.json").write_text(json.dumps(
            {"camera_model": "PINHOLE", "frames": frames}, indent=2) + "\n")
        (gt_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[ok] res {res}: {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
