#!/usr/bin/env python3
"""Verify a prepared scene: COLMAP readability, camera-pose error vs GT, and split leakage.

This is the check that turns "the files exist" into "the geometry is right". Feed-forward
reconstruction is only defined up to a similarity transform, so we estimate that transform
with Umeyama on the *training* views and then report what it costs us:

    python3 scripts/verify_scene.py --scene eth3d_pipes
    python3 scripts/verify_scene.py --scene smoke_box --json

Ground truth sources (never used as model input, only here):
    ETH3D  -> data/scenes/<name>/ground_truth/calibration/poses.json  (train views only)
    smoke  -> data/scenes/<name>/smoke_manifest.json ground_truth_cameras
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import OUTPUTS, images_for, require_colmap, scene_path


def load_estimate(sparse: Path) -> dict[str, dict]:
    import pycolmap

    recon = pycolmap.Reconstruction(str(sparse))
    out = {}
    for image_id, image in recon.images.items():
        camera = recon.cameras[image.camera_id]
        rot = image.cam_from_world.rotation.matrix()
        trans = image.cam_from_world.translation
        centre = -rot.T @ trans
        params = np.asarray(camera.params, dtype=np.float64)
        if camera.model.name in ("PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV"):
            fx, fy = params[0], (params[1] if camera.model.name != "SIMPLE_PINHOLE" else params[0])
        else:  # unknown model: fall back to whatever the first two params are
            fx = fy = params[0]
        out[Path(image.name).name] = {
            "rot": rot, "centre": centre, "fx": float(fx), "fy": float(fy),
            "width": int(camera.width), "height": int(camera.height),
        }
    out["_num_points"] = len(recon.points3D)  # type: ignore[assignment]
    return out


def load_truth(scene: Path, names: set[str]) -> tuple[dict[str, dict], str]:
    manifest = scene / "smoke_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        truth = {x["image"]: {"rot": np.array(x["cam_from_world"])[:, :3],
                              "centre": np.array(x["center"]),
                              "fx": np.array(x["intrinsic"])[0, 0],
                              "fy": np.array(x["intrinsic"])[1, 1]}
                 for x in data.get("ground_truth_cameras", [])}
        return truth, "smoke_manifest"
    poses = scene / "ground_truth" / "calibration" / "poses.json"
    if poses.is_file():
        records = json.loads(poses.read_text())
        truth = {}
        for record in records:
            q = np.asarray(record["qvec"], dtype=np.float64)   # COLMAP world-to-camera quaternion
            t = np.asarray(record["tvec"], dtype=np.float64)
            rot = quat_to_mat(q)
            truth[record["image"]] = {"rot": rot, "centre": -rot.T @ t}
        return truth, "eth3d_poses.json"
    return {}, "none"


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with dst ~= s*R*src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    u, dvec, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    delta = np.diag([1.0, 1.0, sign])
    rot = u @ delta @ vt
    var_src = ((src - mu_s) ** 2).sum() / len(src)
    scale = float((np.diag(delta) @ dvec).sum() / max(var_src, 1e-12))
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    trace = np.clip((np.trace(rot_a.T @ rot_b) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative scene verification")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--json", action="store_true", help="also write outputs/<scene>_verify.json")
    parser.add_argument("--max-centre-error", type=float, default=0.10,
                        help="pass threshold: aligned centre RMSE / scene extent")
    args = parser.parse_args()

    scene = scene_path(args.scene)
    train_names = {p.name for p in images_for(scene)}
    sparse = require_colmap(scene)
    estimate = load_estimate(sparse)
    num_points = int(estimate.pop("_num_points"))
    est_names = set(estimate)

    heldout = {p.name for p in (scene / "heldout" / "images").glob("*")} if (scene / "heldout").is_dir() else set()
    leaked = sorted(heldout & est_names)

    common = sorted(est_names & train_names)
    truth, truth_source = load_truth(scene, set(common))
    matched = [n for n in common if n in truth]

    report: dict = {"scene": args.scene, "sparse_dir": str(sparse),
                    "estimated_cameras": len(est_names), "train_images": len(train_names),
                    "points3D": num_points, "ground_truth_source": truth_source,
                    "cameras_compared": len(matched), "heldout_leakage": leaked}

    print(f"[sparse] {sparse}: {len(est_names)} cameras, {num_points} points3D")
    print(f"[split] train images {len(train_names)}; heldout images seen by reconstruction: "
          f"{len(leaked)}{'  <-- LEAKAGE' if leaked else ' (clean)'}")
    if leaked:
        raise SystemExit(f"heldout views leaked into {sparse}: {leaked}")

    if len(matched) >= 3:
        src = np.array([estimate[n]["centre"] for n in matched])
        dst = np.array([truth[n]["centre"] for n in matched])
        scale, rot, trans = umeyama(src, dst)
        aligned = scale * src @ rot.T + trans
        extent = float(np.linalg.norm(dst - dst.mean(0), axis=1).max()) or 1.0
        centre_err = float(np.linalg.norm(aligned - dst, axis=1).mean())
        rot_err = []
        for name in matched:
            rot_est = rot @ estimate[name]["rot"].T            # world rotation of aligned estimate
            rot_err.append(angle_deg(rot_est, truth[name]["rot"].T))
        report.update({
            "alignment_scale": round(scale, 6),
            "camera_centre_rmse": round(float(np.sqrt(((aligned - dst) ** 2).sum(1).mean())), 6),
            "camera_centre_mean": round(centre_err, 6),
            "camera_centre_mean_over_extent": round(centre_err / extent, 5),
            "rotation_error_mean_deg": round(float(np.mean(rot_err)), 3),
            "rotation_error_max_deg": round(float(np.max(rot_err)), 3),
            "gt_scene_extent": round(extent, 6),
        })
        print(f"[camera] Umeyama on {len(matched)} train views: scale {scale:.4f}, "
              f"centre error {centre_err / extent * 100:.2f}% of extent "
              f"({centre_err:.4f} vs extent {extent:.3f}), "
              f"rotation {np.mean(rot_err):.2f} deg mean / {np.max(rot_err):.2f} max")
        verdict = (report["camera_centre_mean_over_extent"] <= args.max_centre_error
                   and report["rotation_error_mean_deg"] < 10.0)
        report["verdict"] = "PASS" if verdict else "REVIEW"
        print(f"[verdict] {report['verdict']} "
              f"(thresholds: centre<= {args.max_centre_error:.2%} of extent, rotation < 10 deg)")
    else:
        report["verdict"] = "NO_GT"
        print(f"[camera] no usable ground-truth cameras for {len(matched)} views "
              f"({truth_source}); geometry not scored")

    if args.json:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS / f"{args.scene}_verify.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[json] {path}")
    if report["verdict"] == "REVIEW":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
