"""Build an N-view capture subset from the dataset's 100 pre-rendered train views.

The view-count axis uses the original path-traced images (same GT scene, same
intrinsics) instead of newly rendered frames: the pre-rendered set is the highest
quality image source available, and N-view subsets selected by uniform azimuth
spacing give a controlled angular-density ablation. Writes images/, poses.json
(for capture_render.py --pass-mode depth) and capture_subset.json (selection
manifest) into the target scene dir.

    .venv/Scripts/python.exe scripts/capture/build_dataset_subset.py --views 32 --out data/scenes/capture_set32
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/scenes/nerf_lego",
                        help="extracted dataset dir with images/ and transforms_train.json")
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--azimuth-offset", type=int, default=0,
                        help="rotate the uniform pick lattice by this many dataset views")
    args = parser.parse_args()
    dataset, out = Path(args.dataset), Path(args.out)
    name = args.name or f"set{args.views}"
    meta = json.loads((dataset / "transforms_train.json").read_text())
    frames = meta["frames"]
    centres = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in frames])
    azimuths = np.degrees(np.arctan2(centres[:, 1], centres[:, 0]))
    order = np.argsort(azimuths, kind="stable")
    picks = [order[round(i * len(frames) / args.views + args.azimuth_offset) % len(frames)]
             for i in range(args.views)]
    picks = list(dict.fromkeys(int(p) for p in picks))
    if len(picks) != args.views:
        raise SystemExit(f"selection collided: {len(picks)} unique of {args.views} requested; "
                         "adjust --azimuth-offset")
    images_out = out / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    poses = []
    selected = []
    for pick in picks:
        frame = frames[pick]
        stem = Path(frame["file_path"]).stem
        shutil.copyfile(dataset / "images" / f"{stem}.png", images_out / f"{stem}.png")
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        poses.append({"stem": stem, "c2w_blender": c2w.tolist()})
        centre = c2w[:3, 3]
        selected.append({"stem": stem, "dataset_index": pick,
                         "azimuth_deg": round(float(np.degrees(np.arctan2(centre[1], centre[0]))), 2),
                         "elevation_deg": round(float(np.degrees(
                             np.arcsin(centre[2] / np.linalg.norm(centre)))), 2)})
    gaps = np.diff(sorted(s["azimuth_deg"] for s in selected))
    manifest = {"scheme": name, "views": args.views, "dataset": str(dataset),
                "selection": "uniform azimuth rank over the 100 train views",
                "azimuth_offset": args.azimuth_offset,
                "azimuth_gap_deg": {"min": float(gaps.min()), "mean": float(gaps.mean()),
                                    "max": float(gaps.max())},
                "elevation_deg": {"min": float(min(s["elevation_deg"] for s in selected)),
                                  "mean": float(np.mean([s["elevation_deg"] for s in selected])),
                                  "max": float(max(s["elevation_deg"] for s in selected))},
                "frames": selected}
    (out / "poses.json").write_text(json.dumps(poses))
    (out / "capture_subset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["azimuth_gap_deg"], indent=2))
    print(f"[ok] {args.views} views -> {out}")


if __name__ == "__main__":
    main()
