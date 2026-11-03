"""Evaluate one capture-scheme geometry export on the frozen P0 diagnostic protocol.

Pipeline: VGGT-Omega geometry of one capture config -> train-only Sim(3) alignment
on camera centres -> nearest-neighbour regrid onto the GT K grid (identical to P0)
-> depth/normal error against the config's rendered GT depth -> TSDF fusion in the
frozen P0 grid (origin/voxel/shape asserted from deliverables/lego_p0) -> mesh +
fixed-seed surface samples. The output folder mirrors the P0 layout so
diagnostics/measure_surfaces.py --out <out> produces the source-triangle metrics.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, minimum_filter


def scale_split_residual(reference, d, valid, sigma=8.0):
    """Split the relative depth residual into low/high frequency parts (dunes vs grit)."""
    residual = np.zeros_like(d)
    residual[valid] = (d[valid] - reference[valid]) / reference[valid]
    weight = valid.astype(np.float32)
    low = gaussian_filter(residual * weight, sigma) / np.maximum(gaussian_filter(weight, sigma), 1e-9)
    high = residual - low
    return (float(np.sqrt((low[valid] ** 2).mean())) if valid.any() else 0.0,
            float(np.sqrt((high[valid] ** 2).mean())) if valid.any() else 0.0)


def burr_rate(points, gt_frames, voxel, dilation=2):
    """Fraction of surface samples floating in front of the observed GT surface.

    A sample is a burr in a view when it projects inside the image, lands outside the
    (2px-dilated) GT foreground, and is at least 2 voxels in front of the GT surface
    band... for background pixels the band reference is the farthest visible depth, so
    anything projecting there is flagged whenever it is closer than the scene max depth.
    """
    flags = np.zeros(len(points), dtype=bool)
    for frame in gt_frames:
        depth, k, t = frame["depth"], frame["K"], frame["T"]
        fg = binary_dilation(frame["foreground"], iterations=dilation)
        cam = points @ t[:3, :3].T + t[:3, 3]
        z = cam[:, 2]
        front = z > 1e-6
        u = np.rint(cam[:, 0] / np.maximum(z, 1e-6) * k[0, 0] + k[0, 2]).astype(int)
        v = np.rint(cam[:, 1] / np.maximum(z, 1e-6) * k[1, 1] + k[1, 1]).astype(int)
        inside = front & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        u_i, v_i, z_i = u[inside], v[inside], z[inside]
        in_fg = fg[v_i, u_i]
        gt_z = depth[v_i, u_i]
        # outside the object silhouette: any projection counts as a burr when the pixel
        # is background and the sample is closer than the scene far plane of that view
        far = depth[depth > 0].max() if (depth > 0).any() else np.inf
        burr_bg = (~in_fg) & (z_i < far)
        # on the silhouette but floating in front of the measured surface
        burr_fg = in_fg & (gt_z > 0) & (z_i < gt_z - 2 * voxel)
        flags[np.nonzero(inside)[0][burr_bg | burr_fg]] = True
    return float(flags.mean())


def rasterize(vertices, k, t, shape):
    """Vertex z-buffer of a mesh in one view (depth map proxy for structure tests)."""
    cam = vertices @ t[:3, :3].T + t[:3, 3]
    z = cam[:, 2]
    valid = z > 1e-6
    u = np.rint(cam[valid, 0] / z[valid] * k[0, 0] + k[0, 2]).astype(int)
    v = np.rint(cam[valid, 1] / z[valid] * k[1, 1] + k[1, 2]).astype(int)
    keep = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    buffer = np.full(shape, np.inf, dtype=np.float64)
    np.minimum.at(buffer, (v[keep], u[keep]), z[valid][keep])
    return buffer


def structure_metrics(vertices, gt_frames, voxel):
    """Interior-hole fill rate and observed-surface coverage of a mesh per GT view.

    hole_fill_rate: fraction of pixels that are true interior holes in the GT alpha
    (fill_holes(fg) & ~fg) but carry mesh depth - windows/gaps that got filled in.
    coverage_rate: fraction of GT foreground pixels whose z-buffered mesh depth lies
    within +-2 voxels of the GT depth (structure actually reconstructed).
    """
    fills, covers, totals_h, totals_c = [], [], 0, 0
    for frame in gt_frames:
        depth, k, t = frame["depth"], frame["K"], frame["T"]
        fg = frame["foreground"]
        holes = binary_fill_holes(fg) & ~fg
        buffer = rasterize(vertices, k, t, depth.shape)
        buffer = minimum_filter(buffer, size=3)  # close single-pixel splat gaps
        if holes.any():
            fills.append(float(np.isfinite(buffer[holes]).mean()))
            totals_h += int(holes.sum())
        valid = fg & (depth > 0)
        covers.append(float((np.isfinite(buffer[valid]) &
                             (np.abs(buffer[valid] - depth[valid]) <= 2 * voxel)).mean()))
        totals_c += int(valid.sum())
    return {"hole_fill_rate": float(np.mean(fills)), "coverage_rate": float(np.mean(covers)),
            "hole_pixels": totals_h, "coverage_pixels": totals_c}

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsdf_mesh import extract_surface, fuse, write_ply
from verify_scene import umeyama
from diagnostics.pose_depth_ablation import depth_normals, regrid_depth, topology
from diagnostics.resampling_normal_ablation import gt_edge_mask, gt_flat_mask, normal_angle, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, help="capture scene dir with geometry/gt and geometry/omega")
    parser.add_argument("--case", default=None, help="case name; default = capture dir name")
    parser.add_argument("--out", required=True)
    parser.add_argument("--p0-out", default="deliverables/lego_p0")
    parser.add_argument("--fusion-sample", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--exclude-fusion-stems", default="",
                        help="comma-separated views omitted only from fusion; retain all "
                             "cameras for Sim3 and all GT observations for evaluation")
    parser.add_argument("--geometry-dir", default=None,
                        help="geometry root to evaluate; default <capture>/geometry/omega. "
                             "Depth must be camera z-depth on the GT camera grid (800^2)")
    parser.add_argument("--split-sigma", type=float, default=8.0,
                        help="Gaussian sigma (px) splitting low/high frequency depth residual")
    parser.add_argument("--weight-mode", choices=["uniform", "confidence"], default="uniform",
                        help="confidence uses <pred>/confidence/<stem>.npy as TSDF weights, "
                             "regridded like depth and globally rescaled so p99 -> 8 (fuse clips "
                             "at 8; a global rescale is a no-op except at the clip boundary)")
    args = parser.parse_args()
    capture, out = Path(args.capture), Path(args.out)
    case = args.case or capture.name
    gt = capture / "geometry" / "gt"
    pred = Path(args.geometry_dir) if args.geometry_dir else capture / "geometry" / "omega"
    out.mkdir(parents=True, exist_ok=True)

    p0 = json.loads((Path(args.p0_out) / "ablation_report.json").read_text())
    origin = np.asarray(p0["origin"])
    voxel, trunc = float(p0["voxel"]), float(p0["truncation"])
    shape = tuple(p0["shape"])

    records = json.loads((gt / "cameras.json").read_text())["frames"]
    excluded = set(filter(None, args.exclude_fusion_stems.split(",")))
    stems = [Path(f["frame"]).stem for f in records]
    if excluded - set(stems) or len(excluded) == len(stems):
        raise ValueError("fusion exclusions must be known views and retain at least one view")
    predictions = {f["frame"]: f for f in json.loads((pred / "cameras.json").read_text())["frames"]}
    gts = np.asarray([f["T_world_to_cam"] for f in records])
    if args.geometry_dir:
        # External geometry on the GT camera grid: poses are already source-gauge and
        # intrinsics are GT; Sim(3) is the identity and no regrid is applied.
        if len(predictions) != len(records):
            raise SystemExit(f"--geometry-dir must cover exactly the GT frames "
                             f"({len(predictions)} vs {len(records)})")
        scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
        aligned = gts.copy()
    else:
        pts = np.asarray([predictions[f["frame"]]["T_world_to_cam"] for f in records])
        gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
        pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
        scale, rotation, translation = umeyama(pc, gc)
        aligned = pts.copy()
        aligned[:, :3, :3] = pts[:, :3, :3] @ rotation.T
        aligned[:, :3, 3] = scale * pts[:, :3, 3] - aligned[:, :3, :3] @ translation
        np.testing.assert_allclose(-np.einsum("sji,sj->si", aligned[:, :3, :3], aligned[:, :3, 3]),
                                   scale * pc @ rotation.T + translation, atol=5e-6)

    frames = []
    gt_frames = []
    confs = []
    rel_all, ang_all, flat_all, edge_all, scale_low, scale_high = [], [], [], [], [], []
    for index, frame in enumerate(records):
        stem = Path(frame["frame"]).stem
        reference = np.load(gt / "depth" / f"{stem}.npy")
        foreground = np.load(gt / "foreground" / f"{stem}.npy")
        raw = np.load(pred / "depth" / f"{stem}.npy").squeeze() * scale
        k = np.asarray(frame["K_model"])
        pk = np.asarray(predictions[frame["frame"]]["K_model"])
        if args.geometry_dir:
            # External depth lives on the GT camera grid; resize (nearest, protocol
            # default) onto the 512 model grid, then evaluate without regridding.
            if raw.shape != reference.shape:
                import cv2
                raw = cv2.resize(raw.astype(np.float32), (reference.shape[1], reference.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            d = np.where(foreground, raw, 0).astype(np.float32)
        else:
            yy, xx = np.indices(reference.shape, dtype=np.float32)
            u = (xx - k[0, 2]) / k[0, 0] * pk[0, 0] + pk[0, 2]
            v = (yy - k[1, 2]) / k[1, 1] * pk[1, 1] + pk[1, 2]
            d = np.where(foreground, regrid_depth(raw, u, v, "nearest"), 0).astype(np.float32)
        if args.weight_mode == "confidence":
            conf_path = pred / "confidence" / f"{stem}.npy"
            if not conf_path.is_file():
                raise SystemExit(f"--weight-mode confidence but missing {conf_path}")
            conf_raw = np.load(conf_path).squeeze().astype(np.float32)
            if args.geometry_dir:
                if conf_raw.shape != reference.shape:
                    import cv2
                    conf_raw = cv2.resize(conf_raw, (reference.shape[1], reference.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
            else:
                conf_raw = regrid_depth(conf_raw, u, v, "nearest").astype(np.float32)
            confs.append(conf_raw)
        else:
            confs.append(None)
        frames.append({"depth": d, "K": k, "T": aligned[index], "rgb": None})
        gt_frames.append({"depth": reference, "K": k, "T": gts[index],
                          "foreground": foreground})
        valid = (reference > 0) & (d > 0)
        rel_all.append(np.abs(d[valid] - reference[valid]) / reference[valid])
        low_rms, high_rms = scale_split_residual(reference, d, valid, sigma=args.split_sigma)
        scale_low.append(low_rms)
        scale_high.append(high_rms)
        gn, gv = depth_normals(reference, k)
        pn, pv = depth_normals(d, k)
        nv = gv & pv
        angle = normal_angle(gn, pn)
        ang_all.append(angle[nv])
        flat_all.append(angle[gt_flat_mask(gn, gv) & nv])
        edge_all.append(angle[gt_edge_mask(reference) & nv])

    started = time.time()
    keep = [i for i, stem in enumerate(stems) if stem not in excluded]
    frames = [frames[i] for i in keep]
    confs = [confs[i] for i in keep]
    if args.weight_mode == "confidence":
        p99 = float(np.percentile(np.concatenate([c[c > 0] for c in confs if c is not None]), 99))
        confs = [c * (8.0 / max(p99, 1e-9)) if c is not None else None for c in confs]
    field, _, weight = fuse(origin, voxel, shape, trunc, frames, 0, 100, confs,
                            args.weight_mode, 64, 1, sample_mode=args.fusion_sample)
    vertices, faces, normals = extract_surface(field, voxel)
    vertices += origin
    folder = out / case
    folder.mkdir(exist_ok=True)
    stats = {"views": len(frames), "vertices": int(len(vertices)), "faces": int(len(faces)),
             "seconds": round(time.time() - started, 2)}
    if len(faces):
        write_ply(folder / "mesh.ply", vertices, faces, np.full_like(vertices, .65), normals)
        stats.update(topology(faces))
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
        rng = np.random.default_rng(20261102)
        chosen = rng.choice(len(faces), 50000, p=areas / areas.sum())
        a, b = np.sqrt(rng.random(50000)), rng.random(50000)
        bary = np.stack([1 - a, a * (1 - b), a * b], axis=1)
        surface = (triangles[chosen] * bary[..., None]).sum(axis=1)
        np.savez_compressed(folder / "surface_samples.npz", points=surface)
    rel = np.concatenate(rel_all)
    burr = burr_rate(surface, gt_frames, voxel) if len(faces) else 0.0
    structure = structure_metrics(vertices, gt_frames, voxel) if len(faces) else {}
    report = {
        "protocol": "capture-scheme ablation; frozen P0 grid/voxel/truncation; train-only Sim3; "
                    "nearest regrid; GT depth only used for evaluation",
        "capture": str(capture), "case": case,
        "scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist(),
        "voxel": voxel, "truncation": trunc, "origin": origin.tolist(), "shape": list(map(int, shape)),
        "fusion_sample": args.fusion_sample,
        "weight_mode": args.weight_mode,
        "excluded_fusion_stems": sorted(excluded),
        "alignment_views": len(records),
        "depth_on_gt_grid": {"abs_rel": summarize(rel),
                             "normal_deg": summarize(np.concatenate(ang_all)),
                             "normal_deg_flat": summarize(np.concatenate(flat_all)),
                             "normal_deg_edge": summarize(np.concatenate(edge_all)),
                             "residual_lowfreq_rms": float(np.mean(scale_low)),
                             "residual_highfreq_rms": float(np.mean(scale_high)),
                             "split_sigma_px": args.split_sigma},
        "burr_rate": burr,
        "structure": structure,
        "cases": {case: stats},
    }
    observed_src = Path(args.p0_out) / "observed_gt_samples.npy"
    np.save(out / "observed_gt_samples.npy", np.load(observed_src))
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"depth": report["depth_on_gt_grid"], "burr_rate": burr,
                      "structure": structure}, indent=2))
    print(case, stats)


if __name__ == "__main__":
    main()
