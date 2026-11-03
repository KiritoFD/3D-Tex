"""VGGT-input branch: fresh inference, native fusion, raw prediction diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"))


def raw_metrics(scene, config):
    import numpy as np
    from measurement_error.optimize import align
    from diagnostics.pose_depth_ablation import regrid_depth
    omega, gt = scene/"geometry/omega", scene/"geometry/gt"
    frames = json.loads((omega/"cameras.json").read_text())["frames"]
    truth = {f["frame"]: f for f in json.loads((gt/"cameras.json").read_text())["frames"]}
    s, rotation, translation = align(scene, gt)
    scale = json.loads((Path(config["master"])/"scale.json").read_text())["mm_per_source_unit"]
    errors, relative, foreground_count = [], [], 0
    for frame in frames:
        stem = Path(frame["frame"]).stem
        d = np.load(gt/"depth"/f"{stem}.npy")
        raw = np.load(omega/"depth"/f"{stem}.npy").squeeze()*s
        pk, k = np.array(frame["K_model"]), np.array(truth[frame["frame"]]["K_model"])
        yy, xx = np.indices(d.shape)
        u = (xx-k[0, 2])/k[0, 0]*pk[0, 0]+pk[0, 2]
        v = (yy-k[1, 2])/k[1, 1]*pk[1, 1]+pk[1, 2]
        predicted = regrid_depth(raw, u, v, "nearest")
        valid = (d > 0) & (predicted > 0)
        foreground_count += int((d > 0).sum())
        error = predicted[valid]-d[valid]
        errors.append(error*scale)
        relative.append(np.abs(error)/d[valid])
    errors = np.concatenate(errors)
    probe = json.loads((scene/"prediction_metrics.json").read_text())["aggregate"]
    def centers(records):
        t = np.array([f["T_world_to_cam"] for f in records])
        return -np.einsum("sji,sj->si", t[:, :3, :3], t[:, :3, 3])
    pose_error = (s*centers(frames) @ rotation.T+translation-centers([truth[f["frame"]] for f in frames]))*scale
    result = {"mae_mm": float(np.abs(errors).mean()), "rmse_mm": float(np.sqrt((errors**2).mean())),
        "abs_rel": float(np.concatenate(relative).mean()), "normal_deg": probe["normal_deg"]["mean"],
        "rpe_rot_deg": probe["rpe_rot_deg"]["mean"], "valid_pixels": len(errors),
        "depth_valid_fraction": len(errors)/foreground_count,
        "camera_center_rmse_mm": float(np.sqrt((pose_error**2).sum(1).mean())),
        "scope": "VGGT raw predictions; actual jitter poses for jitter cases; Sim3 for evaluation only"}
    (scene/"raw_prediction.json").write_text(json.dumps(result, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/measurement_error.json")
    p.add_argument("--phase", choices=["infer", "fuse", "predict_metrics", "all"], required=True)
    p.add_argument("--only")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    rows = json.loads((Path(config["rgb_sweep"]["root"])/"catalog.json").read_text())
    for row in rows:
        if args.only and row["name"] not in args.only.split(","):
            continue
        scene = Path(row["scene"])
        if not (scene/"perturbation.json").exists():
            continue
        perturbation = json.loads((scene/"perturbation.json").read_text())
        for filename in perturbation["input_order"]:
            image = scene/"images"/filename
            if not image.exists() or hashlib.sha256(image.read_bytes()).hexdigest() != perturbation["input_sha256"].get(filename):
                raise ValueError(f"Incomplete or changed RGB inputs: {image}")
        if args.phase in ("infer", "all"):
            subprocess.run([sys.executable, "scripts/view_selection.py", "infer", "--run-dir", str(scene),
                "--image-resolution", str(config["model_resolution"]), "--aggregator-dtype", config["aggregator_dtype"],
                "--min-free-commit-gib", str(config["rgb_sweep"].get("min_free_commit_gib", 8.0)),
                "--seed", str(row["seed"])], cwd=ROOT, check=True)
        manifest = json.loads((scene/"geometry/omega/manifest.json").read_text())
        if (manifest["input_sha256"] != perturbation["input_sha256"] or manifest["seed"] != row["seed"]
                or manifest["image_resolution_requested"] != config["model_resolution"]
                or manifest["aggregator_dtype"] != "torch."+config["aggregator_dtype"]):
            raise ValueError("Cached VGGT export differs from current perturbation; use a fresh output")
        if args.phase in ("fuse", "all"):
            import numpy as np
            from measurement_error.optimize import native_fusion, align
            from ply_to_obj import read_ply
            from tsdf_mesh import write_ply
            native_fusion(scene, config)
            vertices, faces, _ = read_ply(scene/"mesh/mesh_init.ply")
            s, r, t = align(scene, scene/"geometry/gt")
            vertices = s*vertices @ r.T+t
            write_ply(scene/"mesh/mesh_source.ply", vertices, faces, np.full_like(vertices, .65), np.zeros_like(vertices))
        if args.phase in ("predict_metrics", "all"):
            subprocess.run([sys.executable, "scripts/view_selection.py", "eval", "--omega", str(scene/"geometry/omega"),
                "--gt", str(scene/"geometry/gt"), "--frames", ",".join(Path(f).stem for f in perturbation["input_order"]),
                "--out", str(scene/"prediction_metrics.json")], cwd=ROOT, check=True)
            raw_metrics(scene, config)
        print("VGGT branch", row["name"], args.phase, flush=True)


if __name__ == "__main__":
    main()
