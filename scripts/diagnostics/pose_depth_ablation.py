"""P0: cross GT/predicted pose and depth on one fixed GT intrinsic/voxel grid."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsdf_mesh import extract_surface, fuse, unproject, write_ply
from verify_scene import umeyama


def depth_normals(depth, k, radius=2):
    yy, xx = np.indices(depth.shape)
    points = np.stack([(xx-k[0, 2])/k[0, 0]*depth,
                       (yy-k[1, 2])/k[1, 1]*depth, depth], axis=-1)
    valid = minimum_filter((depth > 0).astype(np.uint8), size=2*radius+1, mode="constant") > 0
    dx = np.roll(points, -radius, 1) - np.roll(points, radius, 1)
    dy = np.roll(points, -radius, 0) - np.roll(points, radius, 0)
    normal = np.cross(dx, dy)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    return normal, valid


def topology(faces):
    edges = np.sort(np.vstack([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"boundary_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", default="data/scenes/nerf_lego16/geometry/omega")
    parser.add_argument("--gt", default="data/scenes/nerf_lego_gt16/geometry/gt")
    parser.add_argument("--out", default="deliverables/lego_p0")
    parser.add_argument("--voxel-div", type=int, default=256)
    parser.add_argument("--optimized-geometry", default=None)
    parser.add_argument("--optimized-label", default="pred_pose_pgsr_depth")
    parser.add_argument("--only-optimized", action="store_true")
    args = parser.parse_args()
    pred, gt, out = Path(args.predicted), Path(args.gt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads((gt / "cameras.json").read_text())["frames"]
    predictions = {f["frame"]: f for f in json.loads((pred / "cameras.json").read_text())["frames"]}
    gts = np.asarray([f["T_world_to_cam"] for f in records])
    pts = np.asarray([predictions[f["frame"]]["T_world_to_cam"] for f in records])
    gc = -np.einsum("sji,sj->si", gts[:, :3,:3], gts[:, :3,3])
    pc = -np.einsum("sji,sj->si", pts[:, :3,:3], pts[:, :3,3])
    scale, rotation, translation = umeyama(pc, gc)
    aligned = pts.copy()
    aligned[:, :3,:3] = pts[:, :3,:3] @ rotation.T
    aligned[:, :3,3] = scale*pts[:, :3,3] - aligned[:, :3,:3] @ translation
    np.testing.assert_allclose(-np.einsum("sji,sj->si", aligned[:, :3,:3], aligned[:, :3,3]),
                               scale*pc@rotation.T+translation, atol=5e-6)
    gt_frames, pred_frames, clouds, normal_errors, flat_errors, relative_errors = [], [], [], [], [], []
    error_panels, optimized_frames, optimized_relative, optimized_normals = [], [], [], []
    for index, frame in enumerate(records):
        stem = Path(frame["frame"]).stem
        reference = np.load(gt / "depth" / f"{stem}.npy")
        raw = np.load(pred / "depth" / f"{stem}.npy").squeeze() * scale
        k, pk = np.asarray(frame["K_model"]), np.asarray(predictions[frame["frame"]]["K_model"])
        yy, xx = np.indices(reference.shape, dtype=np.float32)
        # Regrid only K; no per-view depth fitting or pose-dependent depth correction.
        u = (xx-k[0,2])/k[0,0]*pk[0,0]+pk[0,2]
        v = (yy-k[1,2])/k[1,1]*pk[1,1]+pk[1,2]
        d = cv2.remap(raw.astype(np.float32), u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST)
        foreground = np.load(gt / "foreground" / f"{stem}.npy")
        d = np.where(foreground, d, 0).astype(np.float32)
        gt_frames.append({"depth": reference, "K": k, "T": gts[index], "rgb": None})
        pred_frames.append({"depth": d, "K": k, "T": gts[index], "rgb": None})
        rr, cc = np.nonzero(reference > 0)
        clouds.append(unproject(gt_frames[-1], rr, cc))
        valid = (reference > 0) & (d > 0)
        error = np.zeros_like(d)
        error[valid] = np.abs(d[valid]-reference[valid])/reference[valid]
        relative_errors.append(error[valid])
        gn, gv = depth_normals(reference, k)
        pn, pv = depth_normals(d, k)
        nv = gv & pv
        angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(gn*pn, axis=-1)), 0, 1)))
        normal_errors.append(angle[nv])
        flat = nv.copy()
        for axis in (0, 1):
            for delta in (-2, 2):
                flat &= np.sum(gn * np.roll(gn, delta, axis), axis=-1) > np.cos(np.radians(2))
        flat_errors.append(angle[flat])
        if args.optimized_geometry:
            optimized = np.load(Path(args.optimized_geometry) / "depth" / f"{stem}.npy").squeeze() * scale
            optimized = cv2.remap(optimized.astype(np.float32), u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST)
            optimized = np.where(foreground, optimized, 0).astype(np.float32)
            optimized_frames.append({"depth": optimized, "K": k, "T": gts[index], "rgb": None})
            ov = (reference > 0) & (optimized > 0)
            optimized_relative.append(np.abs(optimized[ov]-reference[ov])/reference[ov])
            on, onv = depth_normals(optimized, k)
            oa = np.degrees(np.arccos(np.clip(np.abs(np.sum(gn*on, axis=-1)), 0, 1)))
            optimized_normals.append(oa[gv & onv])
        if index in (0,4,8,12):
            heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(error/.05, 0, 1)*255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            heat[~valid] = 96
            nh = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(angle/45, 0, 1)*255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            nh[~nv] = 96
            error_panels.append((stem, Image.fromarray(heat), Image.fromarray(nh)))
    points = np.concatenate(clouds)
    lo, hi = points.min(0), points.max(0)
    diagonal = np.linalg.norm(hi-lo)
    voxel, trunc = diagonal/args.voxel_div, diagonal/args.voxel_div*4
    origin, upper = lo-trunc*2, hi+trunc*2
    shape = tuple(np.ceil((upper-origin)/voxel).astype(int))
    # Identical ROI, voxel size, truncation, weights, K and foreground in all four cases.
    cases = {"gt_pose_gt_depth": (gts, gt_frames), "gt_pose_pred_depth": (gts, pred_frames),
             "pred_pose_gt_depth": (aligned, gt_frames), "pred_pose_pred_depth": (aligned, pred_frames)}
    if args.optimized_geometry:
        cases[args.optimized_label] = (aligned, optimized_frames)
    report = {"protocol": "diagnostic only; 16 training views, common GT K and foreground; one train-only Sim3, no per-view fit",
              "predicted_depth_regrid": "nearest-neighbour intrinsic-only resampling onto GT K",
              "scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist(),
              "voxel": float(voxel), "truncation": float(trunc), "origin": origin.tolist(), "shape": list(map(int, shape)),
              "cases": {}}
    if args.only_optimized:
        previous = json.loads((out / "ablation_report.json").read_text())
        np.testing.assert_allclose(previous["origin"], origin)
        assert previous["shape"] == report["shape"] and abs(previous["voxel"]-voxel) < 1e-10
        report["cases"] = previous["cases"]
        report["optimized_depth_results"] = previous.get("optimized_depth_results", {})
        if "pgsr_on_fixed_gt_grid" in previous:
            report["pgsr_on_fixed_gt_grid"] = previous["pgsr_on_fixed_gt_grid"]
        cases = {args.optimized_label: cases[args.optimized_label]}
    for name, (poses, source_frames) in cases.items():
        folder = out / name
        folder.mkdir(exist_ok=True)
        frames = [dict(f, T=t) for f, t in zip(source_frames, poses)]
        started = time.time()
        field, _, weight = fuse(origin, voxel, shape, trunc, frames, 0, 100, [None]*len(frames), "uniform", 64, 1)
        vertices, faces, normals = extract_surface(field, voxel)
        vertices += origin
        write_ply(folder / "mesh.ply", vertices, faces, np.full_like(vertices, .65), normals)
        # Surface sampling is fixed-seed and area-weighted, and does not require trimesh.
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)
        rng = np.random.default_rng(20261102)
        selected = rng.choice(len(faces), 50000, p=areas/areas.sum())
        a, b = np.sqrt(rng.random(50000)), rng.random(50000)
        bary = np.stack([1-a, a*(1-b), a*b], axis=1)
        surface = (triangles[selected]*bary[..., None]).sum(axis=1)
        np.savez_compressed(folder / "surface_samples.npz", points=surface)
        stats = {"vertices": len(vertices), "faces": len(faces), **topology(faces), "seconds": round(time.time()-started, 2)}
        report["cases"][name] = stats
        print(name, stats, flush=True)
    reference_rng = np.random.default_rng(20261102)
    np.save(out / "observed_gt_samples.npy", points[reference_rng.choice(len(points), 50000, replace=False)])
    rel, normals, flat = map(np.concatenate, (relative_errors, normal_errors, flat_errors))
    report["depth_on_fixed_gt_grid"] = {"mean_abs_rel": float(rel.mean()), "median_abs_rel": float(np.median(rel)),
                                       "normal_mean_deg": float(normals.mean()), "normal_median_deg": float(np.median(normals)),
                                       "normal_valid_pixels": len(normals), "flat_normal_mean_deg": float(flat.mean()),
                                       "flat_normal_std_deg": float(flat.std()), "flat_valid_pixels": len(flat),
                                       "normal_operator": "5-pixel cross-product stencil; unsigned angle; flat GT normal variation <2 degrees"}
    if args.optimized_geometry:
        od, on = np.concatenate(optimized_relative), np.concatenate(optimized_normals)
        result = {"geometry": args.optimized_geometry, "mean_abs_rel": float(od.mean()), "median_abs_rel": float(np.median(od)),
                  "normal_mean_deg": float(on.mean()), "normal_median_deg": float(np.median(on))}
        report.setdefault("optimized_depth_results", {})[args.optimized_label] = result
        if args.optimized_label == "pred_pose_pgsr_depth":
            report["pgsr_on_fixed_gt_grid"] = result
    canvas = Image.new("RGB", (1024, len(error_panels)*540), (36,36,36))
    draw = ImageDraw.Draw(canvas)
    for i, (stem, dh, nh) in enumerate(error_panels):
        canvas.paste(dh, (0, i*540+28)); canvas.paste(nh, (512, i*540+28))
        draw.text((10,i*540+8), f"{stem}: depth relative error [0, 5%]", fill="white")
        draw.text((522,i*540+8), "Normal unsigned error [0, 45 deg]", fill="white")
    canvas.save(out / "depth_normal_errors.png")
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["depth_on_fixed_gt_grid"], indent=2))


if __name__ == "__main__":
    main()

