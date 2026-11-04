#!/usr/bin/env python3
"""Phase-3 protocol calibration for ETH3D: GT ROI, units, Sim(3) residuals, sampling floor.

Reads the frozen phase-2 B0/B1 meshes and the official laser scan; changes only the
evaluation protocol (ROI, sampling density, threshold units).  No reconstruction input
and no method parameter is touched.

    python scripts/diagnostics/eth3d_protocol_calibration.py --scene eth3d_kicker \
        --scene-dir data/scenes/eth3d_kicker --phase2 outputs/72h/phase2/eth3d_kicker \
        --out outputs/72h/phase3/calibration/kicker.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_point_metrics import alignment, quat_to_mat, read_ply, sample_mesh


def read_colmap_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = np.asarray([float(v) for v in parts[4:]])
        if model in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = params[:4]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"unsupported camera model {model}")
        cameras[camera_id] = {"K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                             "width": width, "height": height, "model": model}
    return cameras


def gt_pose_map(path: Path) -> dict[str, dict]:
    return {Path(p["image"]).name: p for p in json.loads(path.read_text())}


def observed_mask(points: np.ndarray, cams: list[dict], chunk: int = 2_000_000) -> np.ndarray:
    """True when the point projects inside at least one train-view image frustum."""
    mask = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk):
        block = np.asarray(points[start:start + chunk], dtype=np.float64)
        for cam in cams:
            t, k = cam["T"], cam["K"]
            local = block @ t[:3, :3].T + t[:3, 3]
            z = local[:, 2]
            front = z > 1e-6
            u = np.rint(local[:, 0] / np.maximum(z, 1e-6) * k[0, 0] + k[0, 2]).astype(np.int64)
            v = np.rint(local[:, 1] / np.maximum(z, 1e-6) * k[1, 1] + k[1, 2]).astype(np.int64)
            mask[start:start + chunk] |= (front & (u >= 0) & (u < cam["width"]) &
                                          (v >= 0) & (v < cam["height"]))
    return mask


def summarize(values: np.ndarray, voxel: float) -> dict:
    if len(values) == 0:
        return {"mean": None, "median": None, "p90": None, "mean_in_voxels": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "mean_in_voxels": float(values.mean() / voxel)}


def fscore(accuracy: np.ndarray, completeness: np.ndarray, threshold: float) -> dict:
    if len(completeness) == 0:
        return {"threshold": threshold, "precision": None, "recall": None, "f1": None}
    precision = float((accuracy <= threshold).mean())
    recall = float((completeness <= threshold).mean())
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12))}


def floor_probe(gt: np.ndarray, query_count: int, targets: list[int], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    query = gt[rng.choice(len(gt), size=min(query_count, len(gt)), replace=False)]
    rows = {}
    for target in targets:
        take = min(target, len(gt))
        subset = gt[rng.choice(len(gt), size=take, replace=False)] if take < len(gt) else gt
        dist = cKDTree(subset).query(query, k=1, workers=-1)[0]
        rows[str(target)] = {"mean": float(dist.mean()), "median": float(np.median(dist)),
                             "p90": float(np.percentile(dist, 90))}
    return {"query_points": int(len(query)), "gt_total": int(len(gt)), "targets": rows}


def mesh_info(path: Path, count: int, seed: int) -> dict:
    verts, faces = read_ply(path)
    tri = verts[faces]
    area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    samples = sample_mesh(verts, faces, count, seed)
    return {"vertices": int(len(verts)), "triangles": int(len(faces)), "surface_area": area,
            "bbox_min": verts.min(0).tolist(), "bbox_max": verts.max(0).tolist(),
            "samples": samples, "bbox_diagonal": float(np.linalg.norm(verts.max(0) - verts.min(0)))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--phase2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--gt-samples", type=int, default=5000000)
    p.add_argument("--roi-gt-samples", type=int, default=20000000)
    p.add_argument("--arm", action="append", default=[],
                   help="name=mesh.ply (repeatable); default b0 and b1")
    p.add_argument("--seed", type=int, default=20261104)
    args = p.parse_args()

    scene_dir = Path(args.scene_dir)
    phase2 = Path(args.phase2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    gt_path = phase2 / "eval" / "gt_points.npy"
    gt = np.asarray(np.load(gt_path, mmap_mode="r"), dtype=np.float64)
    diagonal = float(np.linalg.norm(gt.max(0) - gt.min(0)))
    cameras = read_colmap_cameras(scene_dir / "ground_truth/calibration/cameras.txt")
    poses = gt_pose_map(scene_dir / "ground_truth/calibration/poses.json")
    pred_cameras = phase2 / "b0" / "geometry" / "omega" / "cameras.json"
    frames = json.loads(pred_cameras.read_text())["frames"]
    gt_cams = []
    for frame in frames:
        name = Path(frame["frame"]).name
        if name not in poses:
            continue
        pose = poses[name]
        rot = quat_to_mat(np.asarray(pose["qvec"], dtype=np.float64))
        t = np.asarray(pose["tvec"], dtype=np.float64)
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3], world_to_cam[:3, 3] = rot.T, -rot.T @ t
        cam = cameras[int(pose["camera_id"])]
        gt_cams.append({"name": name, "T": world_to_cam, "K": cam["K"],
                        "width": cam["width"], "height": cam["height"]})
    ref = -np.stack([c["T"][:3, :3].T @ c["T"][:3, 3] for c in gt_cams])
    camera_span = float(np.linalg.norm(ref.max(0) - ref.min(0)))
    align = alignment(pred_cameras, scene_dir / "ground_truth/calibration/poses.json")
    scale, rot, trans = align["scale"], np.asarray(align["rotation"]), np.asarray(align["translation"])

    grid = json.loads((phase2 / "p0_grid.json").read_text())
    voxel_gt = float(grid["voxel"]) * scale

    report = {
        "scene": args.scene,
        "label": "internal_geometry_eval",
        "gt": {"path": str(gt_path), "points": int(len(gt)),
               "bbox_min": gt.min(0).tolist(), "bbox_max": gt.max(0).tolist(),
               "diagonal_m": diagonal},
        "units": {"gt_diagonal_m": diagonal, "camera_span_m": camera_span,
                  "gt_diagonal_over_camera_span": diagonal / max(camera_span, 1e-12),
                  "voxel_input": float(grid["voxel"]), "voxel_gt": voxel_gt,
                  "interpretation": "GT and GT poses share the metric gauge; camera span is in metres"},
        "sim3": {k: align[k] for k in ("scale", "rotation", "translation", "n_views", "centre_rms_relative")},
        "sim3_residual_m": {"rms": align["centre_rms_relative"] * camera_span,
                            "voxels": align["centre_rms_relative"] * camera_span / voxel_gt},
        "floor": floor_probe(gt, 50000, [50000, 250000, 1000000, 4000000, 10000000], args.seed),
        "roi": {"train_views": len(gt_cams), "camera_span_m": camera_span},
        "arms": {},
    }

    obs = observed_mask(gt, gt_cams)
    report["roi"]["observed_fraction"] = float(obs.mean())
    report["roi"]["unobserved_fraction"] = float(1 - obs.mean())
    report["roi"]["observed_gt_points"] = int(obs.sum())

    full_tree = cKDTree(gt)
    rng = np.random.default_rng(args.seed + 1)
    arm_specs = args.arm or [f"b0={phase2 / 'b0' / 'mesh' / 'mesh_init.ply'}",
                             f"b1={phase2 / 'b1' / 'mesh' / 'mesh_init.ply'}"]
    for spec in arm_specs:
        arm, mesh_str = spec.split("=", 1)
        mesh_path = Path(mesh_str)
        if not mesh_path.exists():
            report["arms"][arm] = {"valid": False, "why": f"missing {mesh_path}"}
            continue
        info = mesh_info(mesh_path, args.samples, args.seed)
        samples = scale * info["samples"] @ rot.T + trans
        accuracy = full_tree.query(samples, k=1, workers=-1)[0]
        margin = 3 * voxel_gt
        low = samples.min(0) - margin
        high = samples.max(0) + margin
        inside = (gt >= low).all(axis=1) & (gt <= high).all(axis=1)
        roi_frustum = obs
        roi_both = obs & inside
        results = {"roi_frustum": {"gt_points": int(roi_frustum.sum())},
                   "roi_frustum_bbox": {"gt_points": int(roi_both.sum())},
                   "mesh_inside_gt_bbox": float(inside[obs].mean()) if obs.any() else 0.0,
                   "gt_inside_mesh_bbox_fraction": float(inside.mean())}
        for name, mask, cap in (("full_scan", None, args.gt_samples),
                                ("roi_frustum", roi_frustum, args.roi_gt_samples),
                                ("roi_frustum_bbox", roi_both, args.roi_gt_samples)):
            points = gt if mask is None else gt[mask]
            if len(points) > cap:
                points = points[rng.choice(len(points), size=cap, replace=False)]
            completeness = cKDTree(samples).query(points, k=1, workers=-1)[0]
            results.setdefault(name, {}).update({
                "gt_used": int(len(points)),
                "completeness": summarize(completeness, voxel_gt),
                "chamfer": float((accuracy.mean() + completeness.mean()) / 2),
                "f1_1vox": fscore(accuracy, completeness, voxel_gt),
                "f1_2vox": fscore(accuracy, completeness, 2 * voxel_gt),
            })
        report["arms"][arm] = {
            "mesh": str(mesh_path), **{k: v for k, v in info.items() if k != "samples"},
            "samples": int(len(samples)),
            "accuracy": summarize(accuracy, voxel_gt),
            "outlier_accuracy_gt2vox": float((accuracy > 2 * voxel_gt).mean()),
            "variants": results,
        }

    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "scene": args.scene,
        "floor_50k": report["floor"]["targets"]["50000"]["mean"],
        "floor_1m": report["floor"]["targets"]["1000000"]["mean"],
        "observed_fraction": report["roi"]["observed_fraction"],
        "arms": {name: {"area": arm.get("surface_area"),
                        "chamfer_full": arm.get("variants", {}).get("full_scan", {}).get("chamfer"),
                        "chamfer_roi": arm.get("variants", {}).get("roi_frustum_bbox", {}).get("chamfer"),
                        "f1_roi": (arm.get("variants", {}).get("roi_frustum_bbox", {}).get("f1_1vox") or {}).get("f1"),
                        "outlier2vox": arm.get("outlier_accuracy_gt2vox")}
                for name, arm in report["arms"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
