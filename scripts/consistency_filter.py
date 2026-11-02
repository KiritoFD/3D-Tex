#!/usr/bin/env python3
"""A0 stage 1: filter VGGT-Omega depth with cross-view agreement, ready for TSDF.

Background pixels are the main failure mode of feeding raw feed-forward depth into a TSDF:
sky/background still gets a plausible depth and a non-negligible confidence, so fusing it
wraps the object in a thin shell. So instead of trusting a global confidence cut we ask
every *other* view whether it agrees:

    X_i(u,v) = D_i * K_i^-1 [u,v,1]           (camera frame, +z forward)
    X_world  = R_i^T (X_i - t_i)
    project into view j -> (u_j, v_j, z_j)
    agree    := |z_j - D_j(u_j,v_j)| <= tau_abs + tau_rel * z_j

A pixel is kept when at least `--min-agree` other views corroborate it. Optional masks
restrict to the object. Outputs a filtered depth + weight map per frame, plus statistics
that the mesh step and the docs report.

    python3 scripts/consistency_filter.py --scene eth3d_pipes --tau-rel 0.05 --min-agree 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import OUTPUTS, scene_path


def bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = np.clip(np.floor(u).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx, wy = u - x0, v - y0
    return (image[y0, x0] * (1 - wx) * (1 - wy) + image[y0, x1] * wx * (1 - wy)
            + image[y1, x0] * (1 - wx) * wy + image[y1, x1] * wx * wy)


def load_geometry(root: Path) -> dict:
    cameras = json.loads((root / "cameras.json").read_text())["frames"]
    depth, conf, K, T, names = [], [], [], [], []
    for frame in cameras:
        name = frame["frame"]
        d = np.load(root / "depth" / f"{Path(name).stem}.npy").astype(np.float64)
        c = np.load(root / "confidence" / f"{Path(name).stem}.npy").astype(np.float64)
        depth.append(d[..., 0] if d.ndim == 3 else d)
        conf.append(c[..., 0] if c.ndim == 3 else c)
        # Depth maps are stored at MODEL resolution, so unprojection must use K_model.
        # Using K_original here only ever worked because a bug made the two identical.
        K.append(np.asarray(frame["K_model"], dtype=np.float64))
        T.append(np.asarray(frame["T_world_to_cam"], dtype=np.float64))
        names.append(name)
    return {"depth": np.stack(depth), "conf": np.stack(conf), "K": np.stack(K),
            "T": np.stack(T), "names": names, "cameras": cameras}


def camera_centres(T: np.ndarray) -> np.ndarray:
    return -np.einsum("sji,sj->si", T[:, :3, :3], T[:, :3, 3])


def scene_diagonal(T: np.ndarray) -> float:
    centres = camera_centres(T)
    return float(np.linalg.norm(centres.max(0) - centres.min(0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-view depth consistency filter for TSDF")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--geometry", default=None, help="default <scene>/geometry/omega")
    parser.add_argument("--out", default=None, help="default <scene>/geometry/filtered")
    parser.add_argument("--tau-abs", type=float, default=None,
                        help="absolute tolerance in scene units (default: 0.01 x camera span)")
    parser.add_argument("--tau-rel", type=float, default=0.05)
    parser.add_argument("--min-agree", type=int, default=2, help="other views that must agree")
    parser.add_argument("--conf-quantile", type=float, default=0.0,
                        help="drop the lowest q fraction of confidence in each frame (0-1)")
    parser.add_argument("--conf-threshold", type=float, default=0.0,
                        help="absolute confidence cutoff. Unlike a per-frame quantile this is "
                             "comparable across scenes: background/sky gets far lower Omega "
                             "confidence than a real surface (measured ~3.7 vs ~28 on lego)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scene = scene_path(args.scene)
    root = Path(args.geometry).resolve() if args.geometry else scene / "geometry" / "omega"
    out = Path(args.out).resolve() if args.out else scene / "geometry" / "filtered"
    geom = load_geometry(root)
    depth, conf, K, T, names = geom["depth"], geom["conf"], geom["K"], geom["T"], geom["names"]
    cameras = geom["cameras"]
    if args.max_frames:
        take = np.linspace(0, len(names) - 1, args.max_frames).round().astype(int)
        take = np.unique(take)
        depth, conf, K, T = depth[take], conf[take], K[take], T[take]
        names = [names[i] for i in take]
    count, height, width = depth.shape
    span = scene_diagonal(T)
    tau_abs = args.tau_abs if args.tau_abs is not None else 0.01 * span
    print(f"[in] {count} frames {width}x{height}, camera span {span:.4f}, "
          f"tau = {tau_abs:.4f} + {args.tau_rel:.0%} z, min_agree={args.min_agree}")

    start = time.time()
    agree = np.zeros_like(depth, dtype=np.int16)
    samples = np.zeros_like(depth, dtype=np.int16)
    per_pair = []

    rows, cols = np.mgrid[0:height, 0:width]
    rows = rows.ravel().astype(np.float64) + 0.5
    cols = cols.ravel().astype(np.float64) + 0.5
    for i in range(count):
        d_i = depth[i].ravel()
        k_i = K[i]
        cam = np.stack([(cols - k_i[0, 2]) / k_i[0, 0] * d_i,
                        (rows - k_i[1, 2]) / k_i[1, 1] * d_i, d_i], axis=-1)
        rot_i, t_i = T[i][:3, :3], T[i][:3, 3]
        world = (rot_i.T @ (cam - t_i).T).T
        for j in range(count):
            if i == j:
                continue
            rot_j, t_j = T[j][:3, :3], T[j][:3, 3]
            cam_j = (rot_j @ world.T).T + t_j
            z = cam_j[:, 2]
            k_j = K[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = cam_j[:, 0] / z * k_j[0, 0] + k_j[0, 2]
                v = cam_j[:, 1] / z * k_j[1, 1] + k_j[1, 2]
            inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            valid = inside.reshape(depth[i].shape)
            samples[i] += valid
            if not inside.any():
                continue
            d_j = bilinear(depth[j], u[inside], v[inside])
            ok = np.abs(z[inside] - d_j) <= tau_abs + args.tau_rel * z[inside]
            frame_ok = np.zeros_like(valid)
            frame_ok.reshape(-1)[np.nonzero(inside.ravel())[0][ok]] = True
            agree[i] += frame_ok
            per_pair.append({"from": names[i], "to": names[j],
                             "overlap": int(inside.sum()), "inlier_ratio": float(ok.mean())})

    keep = agree >= args.min_agree
    if args.conf_threshold > 0:
        keep &= conf >= args.conf_threshold
    if args.conf_quantile > 0:
        cut = np.array([np.quantile(conf[i], args.conf_quantile) for i in range(count)])
        keep &= conf >= cut[:, None, None]

    out.mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    (out / "mask").mkdir(exist_ok=True)
    kept_frames = [c for c in cameras if Path(c["frame"]).stem in {Path(n).stem for n in names}]
    for i, name in enumerate(names):
        stem = Path(name).stem
        filtered = np.where(keep[i], depth[i], 0.0).astype(np.float32)
        np.save(out / "depth" / f"{stem}.npy", filtered)
        np.save(out / "mask" / f"{stem}.npy", keep[i])
    # write only the cameras whose depth actually exists, so downstream joins cannot go stale
    (out / "cameras.json").write_text(json.dumps(
        {"camera_model": "PINHOLE", "frames": kept_frames}, indent=2) + "\n")

    ratios = np.array([p["inlier_ratio"] for p in per_pair]) if per_pair else np.zeros(1)
    keep_frac = keep.mean(axis=(1, 2))
    report = {
        "scene": args.scene, "frames": count, "image_size": [width, height],
        "camera_span": span, "tau_abs": tau_abs, "tau_rel": args.tau_rel,
        "min_agree": args.min_agree,
        "conf_threshold": args.conf_threshold,
        "pairs_total": len(per_pair),
        "pair_inlier_median": float(np.median(ratios)),
        "pair_inlier_p90": float(np.percentile(ratios, 90)),
        "kept_fraction_per_frame": [round(float(x), 4) for x in keep_frac],
        "kept_fraction_mean": float(keep_frac.mean()),
        "kept_pixels_total": int(keep.sum()),
        "seconds": round(time.time() - start, 1),
        "out": str(out),
    }
    (out / "filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[keep] {keep.sum()} px ({keep_frac.mean():.1%} per frame); "
          f"finite kept depth {depth[keep].min() if keep.any() else 0:.3f}-"
          f"{depth[keep].max() if keep.any() else 0:.3f}")
    print(f"[pairs] {len(per_pair)} pairs, inlier median {report['pair_inlier_median']:.3f}, "
          f"p90 {report['pair_inlier_p90']:.3f}")
    print(f"[out] {out} ({report['seconds']}s)")
    if args.json:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        (OUTPUTS / f"{args.scene}_consistency.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
