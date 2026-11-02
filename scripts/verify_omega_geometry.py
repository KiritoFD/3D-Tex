#!/usr/bin/env python3
"""Check a Gate-0 Omega geometry export on CPU: shapes, signs, and GT camera accuracy.

Pure numpy + json, no torch: this is the decoder-side acceptance test that
docs/VGGT_OMEGA_ADAPTER.md asks for ("read the saved output on CPU ... without GPU").

    python3 scripts/verify_omega_geometry.py --scene smoke_small           # GT is the smoke manifest
    python3 scripts/verify_omega_geometry.py --scene eth3d_pipes           # GT is ETH3D poses.json

Checks
  1. every frame has depth + confidence with the model's H/W, finite and positive where valid
  2. K_original reproduces K_model through the recorded pixel affine (round-trip)
  3. world->camera convention: GT camera centres line up with the predicted ones after a
     train-only Sim(3) (Umeyama), measured in % of scene extent and in degrees
  4. depth consistency: re-project each point into every other valid frame and compare depths
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import OUTPUTS, scene_path
from verify_scene import angle_deg, load_truth, umeyama


def bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample a 2-D map at float coordinates; integer truncation adds ~a pixel of depth error."""
    height, width = image.shape[:2]
    x0 = np.clip(np.floor(u).astype(int), 0, width - 1)
    y0 = np.clip(np.floor(v).astype(int), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx, wy = u - x0, v - y0
    return (image[y0, x0] * (1 - wx) * (1 - wy) + image[y0, x1] * wx * (1 - wy)
            + image[y1, x0] * (1 - wx) * wy + image[y1, x1] * wx * wy)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU acceptance check for geometry/omega")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--geometry", default=None, help="default <scene>/geometry/omega")
    parser.add_argument("--reproj-tol-mib", type=float, default=None)
    parser.add_argument("--depth-tol", type=float, default=0.05, help="relative depth tolerance")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scene = scene_path(args.scene)
    root = Path(args.geometry).resolve() if args.geometry else scene / "geometry" / "omega"
    cameras = json.loads((root / "cameras.json").read_text())["frames"]
    manifest = json.loads((root / "manifest.json").read_text())
    print(f"[io] {root}: {len(cameras)} frames, model {manifest['model_width']}x"
          f"{manifest['model_height']}, checkpoint sha {manifest['checkpoint']['sha256'][:12]}")

    problems: list[str] = []
    depths, confs, Ks, Ts = [], [], [], []
    for frame in cameras:
        stem = Path(frame["frame"]).stem
        depth = np.load(root / "depth" / f"{stem}.npy")
        conf = np.load(root / "confidence" / f"{stem}.npy")
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]            # exporters may keep the channel; compare like for like
        if conf.ndim == 3 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if depth.shape != conf.shape:
            problems.append(f"{stem}: depth {depth.shape} != conf {conf.shape}")
        if depth.ndim != 2:
            problems.append(f"{stem}: depth must be 2-D after squeezing, got {depth.shape}")
        if not np.isfinite(depth).all():
            problems.append(f"{stem}: non-finite depth")
        if not (depth > 0).all():
            problems.append(f"{stem}: non-positive depth where the model predicted")
        # K_original should equal A^-1 K_model A exactly (float round-off only)
        affine = np.asarray(frame["pixel_affine_original_to_model"])
        k_model = np.asarray(frame["K_model"])
        k_original = np.asarray(frame["K_original"])
        residual = np.abs(np.linalg.inv(affine) @ k_model @ affine - k_original).max()
        if residual > 1e-6:
            problems.append(f"{stem}: K_original not consistent with the affine (max {residual:.2e})")
        depths.append(depth)
        confs.append(conf)
        Ks.append(k_original)
        frame["_t"] = np.asarray(frame["T_world_to_cam"])
        Ts.append(frame["_t"])
    depths = np.stack(depths)
    confs = np.stack(confs)
    Ks = np.stack(Ks)
    Ts = np.stack(Ts)
    print(f"[depth] {depths.shape} range {depths.min():.3f}-{depths.max():.3f} "
          f"| conf {confs.min():.2f}-{confs.max():.2f} mean {confs.mean():.2f}")

    # --- 3b. intrinsics sanity: a wrong focal is indistinguishable from wrong depth in a
    # consistency test, so compare the implied FOV with the source image's own FOV if the
    # scene records it (smoke manifests do).
    gt_fov = None
    smoke = scene / "smoke_manifest.json"
    if smoke.is_file():
        gt_fov = json.loads(smoke.read_text()).get("fov_deg")
    if gt_fov:
        for frame, k in zip(cameras, Ks):
            fov_x = 2 * np.degrees(np.arctan(frame["model_width"] / (2 * k[0, 0])))
            print(f"[intrinsics] {frame['frame']}: predicted horizontal FOV {fov_x:.1f} deg "
                  f"(scene {gt_fov:.1f} deg) -> ratio {fov_x / gt_fov:.3f}")

    # --- 4. cross-view depth consistency (the numbers the TSDF filter will use) -------------
    reproj = []
    height, width = depths.shape[1], depths.shape[2]
    for i in range(len(depths)):
        step = max(1, min(height, width) // 64)          # sparse sample: a check, not fusion
        sub = (slice(None, None, step), slice(None, None, step))
        d = depths[i][sub]
        k = Ks[i]
        rows, cols = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        rows = rows * step + step / 2.0
        cols = cols * step + step / 2.0                 # sample at pixel centres
        cam = np.stack([(cols - k[0, 2]) / k[0, 0] * d,
                        (rows - k[1, 2]) / k[1, 1] * d, d], -1).reshape(-1, 3)
        rot_i, t_i = Ts[i][:3, :3], Ts[i][:3, 3]
        world = (rot_i.T @ (cam - t_i).T).T
        for j in range(len(depths)):
            if i == j:
                continue
            rot_j, t_j = Ts[j][:3, :3], Ts[j][:3, 3]
            cam_j = (rot_j @ world.T).T + t_j
            z = cam_j[:, 2]
            k_j = Ks[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = cam_j[:, 0] / z * k_j[0, 0] + k_j[0, 2]
                v = cam_j[:, 1] / z * k_j[1, 1] + k_j[1, 2]
            inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if inside.sum() < 32:
                continue
            d_j = bilinear(depths[j], u[inside], v[inside])
            z_in = z[inside]
            ok = np.abs(z_in - d_j) <= args.depth_tol * z_in
            reproj.append({"pair": f"{cameras[i]['frame']}->{cameras[j]['frame']}",
                           "samples": int(inside.sum()), "inlier_ratio": float(ok.mean()),
                           "median_abs_err": float(np.median(np.abs(z_in - d_j)))})
    if reproj:
        ratios = np.array([r["inlier_ratio"] for r in reproj])
        best = sorted(reproj, key=lambda r: -r["inlier_ratio"])[:5]
        print(f"[multiview] {len(reproj)} ordered pairs, inlier(|dz|<= {args.depth_tol:.0%} z) "
              f"median {np.median(ratios):.3f}, p90 {np.percentile(ratios, 90):.3f}")
        for row in best:
            print(f"   best {row['pair']}: {row['inlier_ratio']:.3f} of {row['samples']} "
                  f"samples, median |dz| {row['median_abs_err']:.4f}")
        if np.median(ratios) < 0.4:
            problems.append(f"multiview depth agreement is low (median inlier {np.median(ratios):.2f}); "
                            "background/sky pixels are very likely being treated as surfaces")

    # --- 3. GT camera comparison (train views only) ----------------------------------------
    truth, source = load_truth(scene, {f["frame"] for f in cameras})
    matched = [f for f in cameras if f["frame"] in truth]
    if len(matched) >= 3:
        pred_c = np.array([-f["_t"][:3, :3].T @ f["_t"][:3, 3] for f in matched])
        gt_c = np.array([truth[f["frame"]]["centre"] for f in matched])
        scale, rot, trans = umeyama(pred_c, gt_c)
        aligned = scale * pred_c @ rot.T + trans
        extent = float(np.linalg.norm(gt_c - gt_c.mean(0), axis=1).max()) or 1.0
        centre_err = float(np.linalg.norm(aligned - gt_c, axis=1).mean())
        rot_err = [angle_deg(rot @ f["_t"][:3, :3].T, truth[f["frame"]]["rot"].T) for f in matched]
        print(f"[camera vs {source}] {len(matched)} views: scale {scale:.4f}, "
              f"centre {centre_err / extent * 100:.2f}% of extent, "
              f"rotation {np.mean(rot_err):.2f} deg mean / {np.max(rot_err):.2f} max")
        if centre_err / extent > 0.10 or max(rot_err) > 15.0:
            problems.append(f"camera error too large vs {source}: "
                            f"{centre_err / extent:.1%} extent, {max(rot_err):.1f} deg")
    else:
        print(f"[camera] no GT for {len(matched)} views ({source}); geometry not scored")

    for issue in problems:
        print(f"[PROBLEM] {issue}")
    verdict = "PASS" if not problems else "FAIL"
    print(f"[verdict] {verdict}")

    if args.json:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS / f"{args.scene}_omega_geometry.json"
        path.write_text(json.dumps({"scene": args.scene, "verdict": verdict,
                                    "problems": problems, "multiview_pairs": reproj},
                                   indent=2) + "\n")
        print(f"[json] {path}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
