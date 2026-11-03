"""Evaluate direct source-gauge meshes without re-fusing rendered depth maps."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from ply_to_obj import read_ply
from tsdf_mesh import write_ply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--meshes", nargs="+", required=True, help="name=source-gauge PLY")
    p.add_argument("--blender", required=True)
    p.add_argument("--reference", default="outputs/geometry_pruning/eval_carve_v35c30_conf")
    p.add_argument("--gt", default="data/scenes/nerf_lego_gt16/geometry/gt")
    p.add_argument("--mesh-is-omega", action="store_true", help="evaluation-only train Sim3 to source gauge")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((Path(args.reference) / "ablation_report.json").read_text())
    report["cases"] = {}
    from capture.capture_eval import burr_rate, structure_metrics
    gt_dir = Path(args.gt)
    gt_frames = [{"depth": np.load(gt_dir / "depth" / (Path(f["frame"]).stem + ".npy")),
                  "foreground": np.load(gt_dir / "foreground" / (Path(f["frame"]).stem + ".npy")),
                  "K": np.asarray(f["K_model"]), "T": np.asarray(f["T_world_to_cam"])}
                 for f in json.loads((gt_dir / "cameras.json").read_text())["frames"]]
    for item in args.meshes:
        name, path = item.split("=", 1)
        folder = out / name
        folder.mkdir(exist_ok=True)
        vertices, faces, _ = read_ply(Path(path))
        if args.mesh_is_omega:
            vertices = report["scale"] * vertices @ np.asarray(report["rotation"]).T + report["translation"]
        if not np.isfinite(vertices).all():
            raise ValueError(f"nonfinite mesh: {path}")
        write_ply(folder / "mesh.ply", vertices, faces, np.full_like(vertices, .65), np.zeros_like(vertices))
        tri = vertices[faces]
        areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
        rng = np.random.default_rng(20261102)
        selected = rng.choice(len(faces), 50000, p=areas / areas.sum())
        a, b = np.sqrt(rng.random(50000)), rng.random(50000)
        bary = np.stack([1-a, a*(1-b), a*b], axis=1)
        points = (tri[selected] * bary[..., None]).sum(1)
        np.savez_compressed(folder / "surface_samples.npz", points=points)
        report["cases"][name] = {"mesh": str(Path(path).resolve()), "vertices": len(vertices), "faces": len(faces),
                                "structure": structure_metrics(vertices, gt_frames, report["voxel"]),
                                "burr_rate_corrected": burr_rate(points, gt_frames, report["voxel"])}
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2))
    shutil.copyfile(Path(args.reference) / "observed_gt_samples.npy", out / "observed_gt_samples.npy")
    with (out / "measure.log").open("w") as log:
        subprocess.run([args.blender, "-b", "--disable-autoexec", "--python",
                        str(ROOT / "scripts/diagnostics/measure_surfaces.py"), "--", "--out", str(out),
                        "--source-flags", str(ROOT / "outputs/lego_source_edge_flags.npy")],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((out / "surface_metrics.json").read_text())["cases"]
    print(json.dumps({name: r["fscore_1_voxel"] for name, r in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
