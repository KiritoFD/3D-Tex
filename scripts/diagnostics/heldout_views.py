"""Prepare four untouched test cameras using train-only alignment; compare mesh renders."""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def prepare(args):
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    alignment = json.loads(Path(args.alignment).read_text())
    scale = alignment["scale"]
    rotation, translation = np.asarray(alignment["rotation"]), np.asarray(alignment["translation"])
    frames = []
    with zipfile.ZipFile(args.archive) as archive:
        meta = json.loads(archive.read("nerf_synthetic/lego/transforms_test.json"))
        focal = 400 / np.tan(meta["camera_angle_x"] / 2)
        k = np.array([[focal,0,400],[0,focal,400],[0,0,1]])
        for index in (0,50,100,150):
            frame = meta["frames"][index]
            stem = Path(frame["file_path"]).stem
            name = f"test_{stem}.png"
            Image.open(io.BytesIO(archive.read(f"nerf_synthetic/lego/test/{stem}.png"))).save(out / "images" / name)
            gt = np.linalg.inv(np.asarray(frame["transform_matrix"]) @ np.diag([1,-1,-1,1]))
            if args.gauge == "omega":
                pose = np.eye(4)
                pose[:3,:3] = gt[:3,:3] @ rotation
                pose[:3,3] = (gt[:3,:3] @ translation + gt[:3,3]) / scale
            else:  # source gauge: raw GT test cameras for meshes fused in source units
                pose = np.eye(4)
                pose[:3] = gt
            frames.append({"frame": name, "file": name, "K_original": k.tolist(), "K_model": k.tolist(),
                           "original_width": 800, "original_height": 800, "model_width": 800, "model_height": 800,
                           "T_world_to_cam": pose.tolist(), "pixel_affine_original_to_model": np.eye(3).tolist()})
    (out / "cameras.json").write_text(json.dumps({"protocol": f"test indices 0,50,100,150; {args.gauge} gauge; evaluation only", "frames": frames}, indent=2))
    print(f"Prepared four heldout views in {out}")


def compare(args):
    root = Path(args.out)
    frames = json.loads((root / "cameras.json").read_text())["frames"]
    report = {"protocol": "four predeclared heldout test views; train-only Sim3; unlit mesh render; PSNR only, no LPIPS; not a full benchmark", "views": []}
    for i, frame in enumerate(frames):
        target = np.asarray(Image.open(root / "images" / frame["file"]).convert("RGBA").resize((512,512)), dtype=float) / 255
        mask = target[...,3] > .5
        row = {"frame": frame["frame"]}
        for label, folder in ((args.before_label, args.before), (args.after_label, args.after)):
            rendered = np.asarray(Image.open(Path(folder) / f"view_{i}.png").convert("RGBA"), dtype=float) / 255
            foreground = rendered[...,:3] * rendered[...,3:4]
            expected = target[...,:3] * target[...,3:4]
            white = foreground + 1-rendered[...,3:4]
            white_gt = expected + 1-target[...,3:4]
            predicted = rendered[...,3] > .5
            row[label] = {"psnr_white": float(-10*np.log10(max(np.mean((white-white_gt)**2),1e-12))),
                          "psnr_foreground": float(-10*np.log10(max(np.mean((foreground[mask]-expected[mask])**2),1e-12))),
                          "silhouette_iou": float((mask & predicted).sum()/max((mask | predicted).sum(),1))}
        report["views"].append(row)
    for label in (args.before_label, args.after_label):
        report[label] = {key: float(np.mean([v[label][key] for v in report["views"]])) for key in report["views"][0][label]}
    (root / args.report_name).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in (args.before_label, args.after_label)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare","compare"])
    parser.add_argument("--out", default="outputs/lego_heldout")
    parser.add_argument("--alignment", default="deliverables/lego_p0/ablation_report.json")
    parser.add_argument("--archive", default="data/.cache/nerf-syn/nerf_syn.zip")
    parser.add_argument("--before", default="outputs/lego_heldout/previous")
    parser.add_argument("--after", default="deliverables/lego_pgsr/heldout")
    parser.add_argument("--after-label", default="pgsr_mesh")
    parser.add_argument("--report-name", default="metrics.json")
    parser.add_argument("--before-label", default="previous_exmesh")
    parser.add_argument("--gauge", choices=["omega", "source"], default="omega",
                        help="prepare only: omega maps GT test cameras by the train-only Sim3 (legacy); "
                             "source keeps raw GT cameras for meshes fused in source units")
    args = parser.parse_args()
    (prepare if args.action == "prepare" else compare)(args)


if __name__ == "__main__":
    main()
