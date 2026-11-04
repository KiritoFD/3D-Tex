#!/usr/bin/env python3
"""Blender-free Lego regression guard for phase-4 A1.

Blender 3.3.9 (used by the phase-3 triangle-exact protocol) is no longer installed on
the 4090 host, so the frozen P0 source blend cannot be opened. This script instead
re-baselines B0 and the A1 arms on one Blender-free evaluator against the frozen
observed-GT surface samples (deliverables/lego_p0/observed_gt_samples.npy), so the
*deltas* versus B0 stay meaningful while absolute Chamfer differs from the
triangle-exact table.

    python scripts/experiments/p4_lego_guard.py --out outputs/72h/phase4/a1/lego_guard.json
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
from eval_point_metrics import read_ply, sample_mesh  # noqa: E402

P0 = ROOT / "deliverables/lego_p0"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--voxel", type=float, default=0.01150144350298549)
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--arm", action="append", default=[])
    args = p.parse_args()

    gt = np.asarray(np.load(P0 / "observed_gt_samples.npy"), dtype=np.float64)
    gt_tree = cKDTree(gt)
    arms = args.arm or [
        f"b0={P0 / 'pred_pose_omega_2vox' / 'mesh.ply'}",
        f"a1_c1={ROOT / 'outputs/72h/phase4/a1/lego_reg/c1/p0/c1/mesh.ply'}",
        f"a1_c2={ROOT / 'outputs/72h/phase4/a1/lego_reg/c2/p0/c2/mesh.ply'}",
    ]
    report = {"evaluator": "blender_free_observed_gt_samples",
              "why": "Blender 3.3.9 missing on host; only 2.93 present and it cannot open the 3.0.43 source blend",
              "gt_samples": int(len(gt)), "voxel": args.voxel, "arms": {}}
    for spec in arms:
        name, path = spec.split("=", 1)
        path = Path(path)
        if not path.exists():
            report["arms"][name] = {"valid": False, "why": f"missing {path}"}
            print(f"{name}: MISSING {path}")
            continue
        verts, faces = read_ply(path)
        tri = verts[faces]
        area = float(0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        samples = sample_mesh(verts, faces, args.samples, args.seed)
        accuracy = gt_tree.query(samples, k=1, workers=-1)[0]
        completeness = cKDTree(samples).query(gt, k=1, workers=-1)[0]
        thr = args.voxel
        precision = float((accuracy <= thr).mean())
        recall = float((completeness <= thr).mean())
        row = {
            "vertices": int(len(verts)), "triangles": int(len(faces)), "surface_area": area,
            "accuracy_mean": float(accuracy.mean()), "accuracy_median": float(np.median(accuracy)),
            "completeness_mean": float(completeness.mean()),
            "completeness_median": float(np.median(completeness)),
            "chamfer_mean": float(accuracy.mean() + completeness.mean()),
            "precision_1vox": precision, "recall_1vox": recall,
            "f1_1vox": float(2 * precision * recall / max(precision + recall, 1e-12)),
        }
        report["arms"][name] = row
        print(f"{name}: chamfer={row['chamfer_mean']:.6f} f1={row['f1_1vox']:.4f} "
              f"area={area:.4f} verts={len(verts)}")

    base = report["arms"].get("b0", {})
    if base:
        for name, row in report["arms"].items():
            if not row.get("vertices"):
                continue
            row["delta_vs_b0"] = {
                k: row[k] - base[k] for k in
                ("chamfer_mean", "f1_1vox", "precision_1vox", "recall_1vox", "surface_area")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
