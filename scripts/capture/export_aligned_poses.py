"""Export aligned (source-gauge) camera poses for external MVS baselines.

Computes the same train-only Sim(3) alignment used by the P0/P0.5 protocol
(predicted camera centres -> GT camera centres) and writes the aligned
world-to-cam poses together with the GT intrinsics so external tools
(COLMAP PatchMatch etc.) see exactly the poses the TSDF fusion will use.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_scene import umeyama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True,
                        help="scene dir with geometry/gt and geometry/omega cameras.json")
    parser.add_argument("--gt-dir", default=None,
                        help="override GT geometry dir (e.g. nerf_lego_gt16/geometry/gt for anchor16)")
    parser.add_argument("--out", required=True, help="poses JSON path")
    args = parser.parse_args()
    capture = Path(args.capture)
    gt_dir = Path(args.gt_dir) if args.gt_dir else capture / "geometry" / "gt"
    gt = json.loads((gt_dir / "cameras.json").read_text())["frames"]
    pred = {f["frame"]: f for f in json.loads((capture / "geometry" / "omega" / "cameras.json")
                                              .read_text())["frames"]}
    gts = np.asarray([f["T_world_to_cam"] for f in gt])
    pts = np.asarray([pred[f["frame"]]["T_world_to_cam"] for f in gt])
    gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
    pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
    scale, rotation, translation = umeyama(pc, gc)
    aligned = pts.copy()
    aligned[:, :3, :3] = pts[:, :3, :3] @ rotation.T
    aligned[:, :3, 3] = scale * pts[:, :3, 3] - aligned[:, :3, :3] @ translation
    frames = []
    for index, frame in enumerate(gt):
        frames.append({"stem": Path(frame["frame"]).stem,
                       "K_original": frame["K_original"],
                       "width": frame["original_width"], "height": frame["original_height"],
                       "T_world_to_cam": aligned[index].tolist()})
    payload = {"capture": str(capture), "scale": float(scale),
               "rotation": rotation.tolist(), "translation": translation.tolist(),
               "note": "aligned predicted poses (source gauge) + GT intrinsics; identical "
                       "alignment to the TSDF protocol",
               "frames": frames}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[ok] {len(frames)} aligned poses -> {args.out} (scale {scale:.6f})")


if __name__ == "__main__":
    main()
