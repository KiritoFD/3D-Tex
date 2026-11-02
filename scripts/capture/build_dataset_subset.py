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
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--bands", default=None,
                        help="dual-ring selection: 'lo1,hi1,count1;lo2,hi2,count2' elevation bands; "
                             "each band picks count views by uniform azimuth rank within the band")
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--azimuth-offset", type=int, default=0,
                        help="rotate the uniform pick lattice by this many dataset views")
    args = parser.parse_args()
    dataset, out = Path(args.dataset), Path(args.out)
    name = args.name or "subset"
    meta = json.loads((dataset / "transforms_train.json").read_text())
    frames = meta["frames"]
    centres = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in frames])
    azimuths = np.degrees(np.arctan2(centres[:, 1], centres[:, 0]))
    elevations = np.degrees(np.arcsin(centres[:, 2] / np.linalg.norm(centres, axis=1)))
    if args.bands:
        bands = []
        for spec in args.bands.split(";"):
            lo, hi, count = (float(x) for x in spec.split(","))
            bands.append((lo, hi, int(count)))
        picks = []
        for lo, hi, count in bands:
            band = np.nonzero((elevations >= lo) & (elevations < hi))[0]
            order = band[np.argsort(azimuths[band], kind="stable")]
            band_picks = [order[round(i * len(order) / count) % len(order)] for i in range(count)]
            band_picks = list(dict.fromkeys(int(p) for p in band_picks))
            if len(band_picks) != count:
                raise SystemExit(f"band [{lo},{hi}) selection collided: "
                                 f"{len(band_picks)} unique of {count} from {len(order)} views")
            picks.extend(band_picks)
        name = args.name or "dualring"
    else:
        name = args.name or f"set{args.views}"
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
    manifest = {"scheme": name, "views": len(picks), "dataset": str(dataset),
                "selection": "uniform azimuth rank over the 100 train views" if not args.bands
                else f"elevation bands {args.bands}, uniform azimuth rank within each band",
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
    print(f"[ok] {len(picks)} views -> {out}")


if __name__ == "__main__":
    main()
