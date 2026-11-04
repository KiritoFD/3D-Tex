#!/usr/bin/env python3
"""Read-only attribution of ETH3D B0 outliers: is the shell inside/outside the scan ROI,
and is it multi-view consistent (i.e. invisible to free-space carving)?

For every mesh sample classified as an outlier by the frozen protocol, report
  * inside_gt_bbox / inside_gt_core  (is GT simply not covering that space?)
  * multi-view support and free-space votes for the sample in Omega gauge
    (if support ~1, free-space carving cannot remove it).

    python scripts/experiments/p4_a1_diag.py --scene eth3d_kicker \
        --scene-dir data/scenes/eth3d_kicker --phase2 outputs/72h/phase2/eth3d_kicker \
        --arm b0=outputs/72h/phase2/eth3d_kicker/b0/mesh/mesh_init.ply \
        --out outputs/72h/phase4/a1/kicker_attrib.json
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--phase2", required=True)
    p.add_argument("--arm", action="append", default=[], help="name=mesh.ply")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--tau-voxels", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20261104)
    args = p.parse_args()

    scene_dir, phase2 = Path(args.scene_dir), Path(args.phase2)
    gt = np.asarray(np.load(phase2 / "eval" / "gt_points.npy", mmap_mode="r"), dtype=np.float64)
    grid = json.loads((phase2 / "p0_grid.json").read_text())
    omega = phase2 / "b0" / "geometry" / "omega"
    align = alignment(omega / "cameras.json", scene_dir / "ground_truth/calibration/poses.json")
    scale = float(align["scale"])
    rot = np.asarray(align["rotation"], dtype=np.float64)
    trans = np.asarray(align["translation"], dtype=np.float64)
    voxel_gt = float(grid["voxel"]) * scale
    tau = args.tau_voxels * float(grid["voxel"])

    lo, hi = gt.min(0), gt.max(0)
    margin = 2 * voxel_gt
    gt_tree = cKDTree(gt)
    frames = json.loads((omega / "cameras.json").read_text())["frames"]
    depths = {}
    confs = {}
    for frame in frames:
        stem = Path(frame["frame"]).stem
        d = np.load(omega / "depth" / f"{stem}.npy")
        depths[stem] = d[..., 0] if d.ndim == 3 else d
        c = np.load(omega / "confidence" / f"{stem}.npy")
        confs[stem] = c[..., 0] if c.ndim == 3 else c

    report = {"scene": args.scene, "voxel_gt": voxel_gt, "tau_omega": tau,
              "protocol": "read-only; frozen P0 mesh, train-only Sim3, no parameter change",
              "arms": {}}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        verts, faces = read_ply(Path(path))
        samples = sample_mesh(verts, faces, args.samples, args.seed)
        gt_samples = scale * samples @ rot.T + trans
        accuracy = gt_tree.query(gt_samples, k=1, workers=-1)[0]
        outlier = accuracy > 2 * voxel_gt

        inside_bbox = ((gt_samples >= lo - margin).all(1) & (gt_samples <= hi + margin).all(1))
        # GT "core": the dense centre of the scan, ignoring sparse extremities.
        dist_gt = accuracy
        core = dist_gt <= 10 * voxel_gt

        sup = np.zeros(len(samples), dtype=np.int32)
        free = np.zeros(len(samples), dtype=np.int32)
        seen = np.zeros(len(samples), dtype=np.int32)
        conf_max = np.zeros(len(samples))
        zsum = np.zeros(len(samples))
        for frame in frames:
            stem = Path(frame["frame"]).stem
            depth = depths[stem]
            conf = confs[stem]
            h, w = depth.shape
            k = np.asarray(frame["K_model"], dtype=np.float64)
            t = np.asarray(frame["T_world_to_cam"], dtype=np.float64)
            local = samples @ t[:3, :3].T + t[:3, 3]
            z = local[:, 2]
            u = np.rint(local[:, 0] / np.maximum(z, 1e-9) * k[0, 0] + k[0, 2]).astype(np.int64)
            v = np.rint(local[:, 1] / np.maximum(z, 1e-9) * k[1, 1] + k[1, 2]).astype(np.int64)
            ok = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            ref = np.zeros(len(samples))
            ref[ok] = depth[v[ok], u[ok]]
            ok &= ref > 0
            seen += ok
            sup += ok & (np.abs(z - ref) <= tau)
            free += ok & (z - ref < -tau)
            cvals = np.zeros(len(samples))
            cvals[ok] = conf[v[ok], u[ok]]
            conf_max = np.maximum(conf_max, cvals)
            zsum += np.where(ok, z, 0.0)
        seen_safe = np.maximum(seen, 1)
        zmean = zsum / seen_safe
        sup_frac = sup / seen_safe
        free_frac = free / seen_safe

        edges = np.quantile(zmean, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        depth_bins = {}
        for i in range(5):
            m = (zmean >= edges[i]) & (zmean <= edges[i + 1])
            if m.sum() == 0:
                continue
            depth_bins[f"{edges[i]:.2f}-{edges[i + 1]:.2f}"] = {
                "n": int(m.sum()), "accuracy_median_m": float(np.median(accuracy[m])),
                "outlier_frac": float(outlier[m].mean()),
                "mean_support_frac": float(sup_frac[m].mean()),
                "conf_mean": float(conf_max[m].mean())}

        def summary(mask):
            if mask.sum() == 0:
                return {"n": 0}
            return {"n": int(mask.sum()),
                    "frac_of_samples": float(mask.mean()),
                    "inside_gt_bbox": float(inside_bbox[mask].mean()),
                    "mean_seen_views": float(seen[mask].mean()),
                    "mean_support_frac": float(sup_frac[mask].mean()),
                    "mean_free_frac": float(free_frac[mask].mean()),
                    "frac_support_ge_0.8": float((sup_frac[mask] >= 0.8).mean()),
                    "conf_mean": float(conf_max[mask].mean()),
                    "conf_median": float(np.median(conf_max[mask])),
                    "conf_p90": float(np.percentile(conf_max[mask], 90)),
                    "accuracy_median_m": float(np.median(accuracy[mask]))}
        report["arms"][name] = {
            "mesh": str(path), "samples": int(len(samples)),
            "outlier_frac": float(outlier.mean()),
            "outlier": summary(outlier),
            "inlier": summary(~outlier),
            "all": summary(np.ones(len(samples), dtype=bool)),
            "accuracy_by_camera_depth": depth_bins,
        }
        print(json.dumps({name: report["arms"][name]}, indent=2), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
