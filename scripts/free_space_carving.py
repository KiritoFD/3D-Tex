#!/usr/bin/env python3
"""GT-free background-shell suppression before TSDF fusion.

Two filters, both computed only from VGGT-Omega outputs (depth + confidence + poses):

1. confidence gate: drop pixels below a fixed global confidence quantile;
2. free-space carving: a depth pixel is removed when it projects into other training
   views and lands clearly *in front of* their observed surface (free space) more often
   than it is confirmed by them.

No ground-truth scan, mask, camera or depth is read anywhere in this script.

    python scripts/free_space_carving.py \
        --omega outputs/72h/phase2/eth3d_kicker/b0/geometry/omega \
        --out outputs/72h/phase4/a1/kicker/c1/geometry/omega \
        --conf-quantile 0.2 --tau-voxels 2 --voxel 0.018257796759500712 \
        --vote-frac 0.5 --confirm-max 0.1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_depth(path: Path) -> np.ndarray:
    arr = np.load(path)
    return arr[..., 0] if arr.ndim == 3 else arr


def project(points: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a view; returns (inside, u, v, camera-space z)."""
    t = np.asarray(frame["T_world_to_cam"], dtype=np.float64)
    k = np.asarray(frame["K_model"], dtype=np.float64)
    local = points @ t[:3, :3].T + t[:3, 3]
    z = local[:, 2]
    safe = np.maximum(z, 1e-9)
    u = np.rint(local[:, 0] / safe * k[0, 0] + k[0, 2]).astype(np.int64)
    v = np.rint(local[:, 1] / safe * k[1, 1] + k[1, 2]).astype(np.int64)
    height, width = int(frame["model_height"]), int(frame["model_width"])
    inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return inside, u, v, z


def main() -> None:
    parser = argparse.ArgumentParser(description="GT-free confidence + free-space depth carving")
    parser.add_argument("--omega", required=True, help="Omega geometry directory (depth/confidence/cameras.json)")
    parser.add_argument("--out", required=True, help="output geometry directory (omega layout)")
    parser.add_argument("--conf-quantile", type=float, default=0.2, help="drop pixels below this global confidence quantile")
    parser.add_argument("--tau-voxels", type=float, default=2.0)
    parser.add_argument("--voxel", type=float, required=True)
    parser.add_argument("--tau-rel", type=float, default=0.0, help="extra tau as a fraction of the median depth")
    parser.add_argument("--vote-frac", type=float, default=0.5, help="free-space vote fraction required to delete")
    parser.add_argument("--confirm-max", type=float, default=0.1, help="maximum support fraction to allow a deletion")
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=250_000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    omega = Path(args.omega)
    out = Path(args.out)
    cameras = json.loads((omega / "cameras.json").read_text())["frames"]
    if args.max_frames > 0:
        cameras = cameras[:args.max_frames]
    (out / "depth").mkdir(parents=True, exist_ok=True)
    shutil.copy(omega / "cameras.json", out / "cameras.json")
    conf_link = out / "confidence"
    if conf_link.is_symlink() or conf_link.exists():
        if conf_link.is_symlink():
            conf_link.unlink()
        elif conf_link.is_dir():
            shutil.rmtree(conf_link)
    try:
        conf_link.symlink_to((omega / "confidence").resolve())
    except OSError:
        shutil.copytree(omega / "confidence", conf_link)

    depths, confs, stems = [], [], []
    conf_values = []
    for frame in cameras:
        stem = Path(frame["frame"]).stem
        depth = load_depth(omega / "depth" / f"{stem}.npy").astype(np.float32)
        conf = load_depth(omega / "confidence" / f"{stem}.npy").astype(np.float32)
        if depth.shape != (frame["model_height"], frame["model_width"]):
            raise SystemExit(f"{stem}: depth {depth.shape} does not match model grid")
        depths.append(depth)
        confs.append(conf)
        stems.append(stem)
        conf_values.append(conf[depth > 0])

    all_conf = np.concatenate(conf_values) if conf_values else np.zeros(0)
    conf_threshold = float(np.quantile(all_conf, args.conf_quantile)) if len(all_conf) else 0.0
    med_depth = float(np.median(np.concatenate([d[d > 0] for d in depths]))) if all_conf.size else 0.0
    tau = float(args.tau_voxels * args.voxel + args.tau_rel * med_depth)

    frames = []
    for frame, depth in zip(cameras, depths):
        t = np.linalg.inv(np.asarray(frame["T_world_to_cam"], dtype=np.float64))
        k = np.asarray(frame["K_model"], dtype=np.float64)
        frames.append({"T_cam_to_world": t, "K": k})

    report = {"omega": str(omega), "voxel": args.voxel, "tau": tau,
              "conf_threshold": conf_threshold, "conf_quantile": args.conf_quantile,
              "vote_frac": args.vote_frac, "confirm_max": args.confirm_max,
              "min_views": args.min_views, "frames": {}}

    for index, (frame, depth, conf, stem) in enumerate(zip(cameras, depths, confs, stems)):
        height, width = depth.shape
        base_valid = (depth > 0) & np.isfinite(depth) & (conf >= conf_threshold)
        ys, xs = np.nonzero(base_valid)
        if len(ys) == 0:
            np.save(out / "depth" / f"{stem}.npy", depth)
            continue
        z = depth[ys, xs].astype(np.float64)
        cam = frames[index]["T_cam_to_world"]
        points = np.stack([(xs - frames[index]["K"][0, 2]) / frames[index]["K"][0, 0] * z,
                           (ys - frames[index]["K"][1, 2]) / frames[index]["K"][1, 1] * z,
                           z], axis=1)
        points = points @ cam[:3, :3].T + cam[:3, 3]

        seen = np.zeros(len(ys), dtype=np.int32)
        support = np.zeros(len(ys), dtype=np.int32)
        free = np.zeros(len(ys), dtype=np.int32)
        for j, (other, other_depth) in enumerate(zip(cameras, depths)):
            if j == index:
                continue
            for start in range(0, len(points), args.chunk):
                block = points[start:start + args.chunk]
                inside, u, v, zloc = project(block, other)
                if not inside.any():
                    continue
                u_i, v_i = u[inside], v[inside]
                ref = other_depth[v_i, u_i]
                ok = ref > 0
                if not ok.any():
                    continue
                idx = start + np.nonzero(inside)[0][ok]
                dz = zloc[inside][ok] - ref[ok]
                seen[idx] += 1
                support[idx] += (np.abs(dz) <= tau)
                free[idx] += dz < -tau

        seen_safe = np.maximum(seen, 1)
        free_frac = free / seen_safe
        support_frac = support / seen_safe
        delete = (seen >= args.min_views) & (free_frac >= args.vote_frac) & (support_frac <= args.confirm_max)
        tested = seen >= args.min_views
        depth_out = depth.copy()
        depth_out[ys[delete], xs[delete]] = 0.0
        low_conf = (depth > 0) & np.isfinite(depth) & (conf < conf_threshold)
        depth_out[low_conf] = 0.0
        np.save(out / "depth" / f"{stem}.npy", depth_out)
        report["frames"][stem] = {
            "valid_before_confidence": int((depth > 0).sum()),
            "kept_after_confidence": int(base_valid.sum()),
            "deleted_by_free_space": int(delete.sum()),
            "deleted_fraction_of_confident": float(delete.sum() / max(base_valid.sum(), 1)),
            "tested_fraction": float(tested.mean()),
            "deleted_fraction_of_tested": float(delete.sum() / max(int(tested.sum()), 1)),
            "mean_free_fraction": float(free_frac[tested].mean()) if tested.any() else 0.0,
            "mean_support_fraction": float(support_frac[tested].mean()) if tested.any() else 0.0,
        }
        if (index + 1) % 5 == 0:
            print(f"{index + 1}/{len(cameras)} {stem} deleted={delete.mean():.3f}", flush=True)

    report["surviving_fraction"] = float(np.mean([v["kept_after_confidence"] - v["deleted_by_free_space"]
                                                  for v in report["frames"].values()])
                                         / np.mean([v["valid_before_confidence"] for v in report["frames"].values()]))
    (out / "carve_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), "tau": tau, "conf_threshold": conf_threshold,
                      "surviving_fraction": report["surviving_fraction"],
                      "mean_deleted_fraction": float(np.mean([v["deleted_fraction_of_confident"]
                                                              for v in report["frames"].values()]))}, indent=2))


if __name__ == "__main__":
    main()
