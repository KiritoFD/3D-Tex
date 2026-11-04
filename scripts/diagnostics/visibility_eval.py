#!/usr/bin/env python3
"""ETH3D visibility mask and masked re-evaluation (training frustum + predicted occlusion).

The mask is derived exclusively from training-view VGGT-Omega cameras and depth plus the
laser-scan points themselves (never GT depth, GT cameras or GT foreground masks):

    note: gt_points are the *evaluation* targets, so reading them is allowed; the mask
    only decides which of them enter completeness / recall.

For every scan point:
  * transform GT gauge -> Omega gauge with the train-only Sim(3);
  * project into each training view;
  * a view counts as seeing the point when it lands inside and its z is not behind the
    predicted Omega surface (z <= d + tau).

Visible = seen in >= 1 training view. Masks are stored per scene and reused by every arm.

    python scripts/diagnostics/visibility_eval.py --scene eth3d_kicker \
        --scene-dir data/scenes/eth3d_kicker --phase2 outputs/72h/phase2/eth3d_kicker \
        --arm b0=outputs/72h/phase2/eth3d_kicker/b0/mesh/mesh_init.ply \
        --arm b1=outputs/72h/phase2/eth3d_kicker/b1/mesh/mesh_init.ply \
        --out outputs/72h/phase4/a2/kicker_visibility.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_point_metrics import alignment, read_ply, sample_mesh  # noqa: E402


def load_depth(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    return arr[..., 0] if arr.ndim == 3 else arr


def summarize(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"mean": None, "median": None, "p90": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


def fscore(accuracy: np.ndarray, completeness: np.ndarray, threshold: float) -> dict:
    if len(completeness) == 0:
        return {"threshold": threshold, "precision": None, "recall": None, "f1": None}
    precision = float((accuracy <= threshold).mean())
    recall = float((completeness <= threshold).mean())
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12))}


def build_mask(gt: np.ndarray, phase2: Path, scene_dir: Path, tau_voxels: float,
               chunk: int = 500_000) -> tuple[np.ndarray, dict]:
    omega = phase2 / "b0" / "geometry" / "omega"
    frames = json.loads((omega / "cameras.json").read_text())["frames"]
    align = alignment(omega / "cameras.json", scene_dir / "ground_truth/calibration/poses.json")
    scale = float(align["scale"])
    rot = np.asarray(align["rotation"], dtype=np.float64)
    trans = np.asarray(align["translation"], dtype=np.float64)
    grid = json.loads((phase2 / "p0_grid.json").read_text())
    voxel = float(grid["voxel"])
    tau = tau_voxels * voxel

    seen_any = np.zeros(len(gt), dtype=bool)
    per_view = {}
    for frame in frames:
        stem = Path(frame["frame"]).stem
        depth = np.asarray(load_depth(omega / "depth" / f"{stem}.npy"), dtype=np.float64)
        height, width = depth.shape
        k = np.asarray(frame["K_model"], dtype=np.float64)
        t = np.asarray(frame["T_world_to_cam"], dtype=np.float64)
        hits = 0
        for start in range(0, len(gt), chunk):
            block = gt[start:start + chunk]
            pred = ((block - trans) @ rot) / scale
            local = pred @ t[:3, :3].T + t[:3, 3]
            z = local[:, 2]
            safe = np.maximum(z, 1e-9)
            u = np.rint(local[:, 0] / safe * k[0, 0] + k[0, 2]).astype(np.int64)
            v = np.rint(local[:, 1] / safe * k[1, 1] + k[1, 2]).astype(np.int64)
            inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            idx = np.nonzero(inside)[0]
            if len(idx) == 0:
                continue
            ref = depth[v[idx], u[idx]]
            ok = (ref > 0) & (z[idx] <= ref + tau)
            seen_any[start + idx[ok]] = True
            hits += int(ok.sum())
        per_view[stem] = hits

    stats = {"views": len(frames), "tau": tau, "tau_voxels": tau_voxels,
             "voxel_omega": voxel, "sim3": {k: align[k] for k in ("scale", "rotation", "translation")},
             "visible_points": int(seen_any.sum()), "fraction": float(seen_any.mean()),
             "per_view_hits": per_view}
    return seen_any, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--phase2", required=True)
    parser.add_argument("--arm", action="append", default=[], help="name=mesh.ply (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask-out", default=None)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--gt-cap", type=int, default=1_000_000)
    parser.add_argument("--tau-voxels", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    phase2 = Path(args.phase2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(np.load(phase2 / "eval" / "gt_points.npy", mmap_mode="r"), dtype=np.float64)

    visible, vstats = build_mask(gt, phase2, scene_dir, args.tau_voxels)
    mask_path = Path(args.mask_out) if args.mask_out else out.parent / "masks" / f"{args.scene}_visible.npy"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, visible)

    grid = json.loads((phase2 / "p0_grid.json").read_text())
    scale = float(vstats["sim3"]["scale"])
    voxel_gt = float(grid["voxel"]) * scale
    rot = np.asarray(vstats["sim3"]["rotation"], dtype=np.float64)
    trans = np.asarray(vstats["sim3"]["translation"], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    visible_gt = gt[visible]
    if len(visible_gt) > args.gt_cap:
        visible_gt = visible_gt[rng.choice(len(visible_gt), args.gt_cap, replace=False)]
    full_tree = cKDTree(gt)

    report = {"scene": args.scene, "label": "protocol_visibility_v1",
              "mask": str(mask_path), "visibility": vstats, "arms": {}}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        mesh_path = Path(path)
        if not mesh_path.exists():
            report["arms"][name] = {"valid": False, "why": f"missing {mesh_path}"}
            continue
        verts, faces = read_ply(mesh_path)
        tri = verts[faces]
        area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        samples = sample_mesh(verts, faces, args.samples, args.seed)
        samples_gt = scale * samples @ rot.T + trans
        accuracy = full_tree.query(samples_gt, k=1, workers=-1)[0]
        completeness_full = cKDTree(samples_gt).query(gt, k=1, workers=-1)[0]
        completeness_vis = cKDTree(samples_gt).query(visible_gt, k=1, workers=-1)[0]
        report["arms"][name] = {
            "mesh": str(mesh_path), "vertices": int(len(verts)), "triangles": int(len(faces)),
            "surface_area": area, "outlier_accuracy_gt2vox": float((accuracy > 2 * voxel_gt).mean()),
            "outlier_accuracy_gt4vox": float((accuracy > 4 * voxel_gt).mean()),
            "accuracy": summarize(accuracy),
            "completeness_full": summarize(completeness_full),
            "completeness_visible": summarize(completeness_vis),
            "chamfer_full": float((accuracy.mean() + completeness_full.mean()) / 2),
            "chamfer_visible": float((accuracy.mean() + completeness_vis.mean()) / 2),
            "f1_1vox_full": fscore(accuracy, completeness_full, voxel_gt),
            "f1_1vox_visible": fscore(accuracy, completeness_vis, voxel_gt),
            "f1_2vox_visible": fscore(accuracy, completeness_vis, 2 * voxel_gt),
        }
        print(json.dumps({"scene": args.scene, "arm": name,
                          "visible_frac": vstats["fraction"],
                          "chamfer_visible": report["arms"][name]["chamfer_visible"],
                          "outlier2vox": report["arms"][name]["outlier_accuracy_gt2vox"],
                          "f1_visible": report["arms"][name]["f1_1vox_visible"]}), flush=True)

    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
