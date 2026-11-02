"""P0.5: resampling / normal-operator / fusion diagnosis on the frozen P0 protocol.

Part A (image space): how much of the 32.16-degree mean normal error comes from
regridding Omega depth onto the GT intrinsic grid with nearest-neighbour sampling?
Conditions: nearest vs bilinear regrid, normals computed on the native Omega grid,
an edge-preserving bilateral pre-filter, and the native-grid depth error.

Part B (mesh space): identical GT diagnostic grid / voxel / truncation / weights as
P0, ablating regrid interpolation, TSDF depth sampling, the bilateral filter and a
robust Huber fusion pass. Meshes + surface samples follow the exact P0 procedure so
measure_surfaces.py can evaluate them with --out deliverables/lego_p05.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter, minimum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsdf_mesh import extract_surface, fuse, unproject, write_ply
from verify_scene import umeyama
from diagnostics.pose_depth_ablation import depth_normals, edge_preserving_filter, regrid_depth, topology

PANEL_FRAMES = (0, 4, 8, 12)


def normal_angle(gn, pn):
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(gn * pn, axis=-1)), 0, 1)))


def gt_flat_mask(gn, gv):
    flat = gv.copy()
    for axis in (0, 1):
        for delta in (-2, 2):
            flat &= np.sum(gn * np.roll(gn, delta, axis), axis=-1) > np.cos(np.radians(2))
    return flat


def gt_edge_mask(depth):
    """Pixels near a depth discontinuity or silhouette boundary (thin rods, edges)."""
    valid = depth > 0
    filled_hi = np.where(valid, depth, -np.inf)
    filled_lo = np.where(valid, depth, np.inf)
    local_max = maximum_filter(filled_hi, size=3, mode="constant", cval=-np.inf)
    local_min = minimum_filter(filled_lo, size=3, mode="constant", cval=np.inf)
    spread = np.where(np.isfinite(local_min) & (local_max > -np.inf), local_max - local_min, 0.0)
    boundary = valid & (minimum_filter(valid.astype(np.uint8), size=3, mode="constant") == 0)
    return valid & ((spread > 0.02 * depth) | boundary)


def summarize(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "count": int(values.size)}


def part_a(pred, gt, out, scale, records, predictions, args):
    """Image-space depth/normal diagnosis; no fusion, no meshes."""
    sigmas = [float(s) for s in args.parta_bilateral_sigmas.split(",")]
    condition_names = (["regrid_nearest", "regrid_bilinear", "native_normal", "nearest_radius1"]
                       + [f"bilateral_sc{sigma:g}" for sigma in sigmas])
    conditions = {name: {"rel": [], "angle": [], "flat": [], "edge": []} for name in condition_names}
    native_grid_rel = []
    panels = []
    for index, frame in enumerate(records):
        stem = Path(frame["frame"]).stem
        reference = np.load(gt / "depth" / f"{stem}.npy")
        raw = np.load(pred / "depth" / f"{stem}.npy").squeeze() * scale
        k = np.asarray(frame["K_model"])
        pk = np.asarray(predictions[frame["frame"]]["K_model"])
        yy, xx = np.indices(reference.shape, dtype=np.float32)
        u = (xx - k[0, 2]) / k[0, 0] * pk[0, 0] + pk[0, 2]
        v = (yy - k[1, 2]) / k[1, 1] * pk[1, 1] + pk[1, 2]
        foreground = np.load(gt / "foreground" / f"{stem}.npy")
        gn, gv = depth_normals(reference, k)
        flat_gt = gt_flat_mask(gn, gv)
        edge_gt = gt_edge_mask(reference)

        def measure(d_regridded, radius=2):
            d = np.where(foreground, d_regridded, 0).astype(np.float32)
            valid = (reference > 0) & (d > 0)
            rel = np.abs(d[valid] - reference[valid]) / reference[valid]
            pn, pv = depth_normals(d, k, radius=radius)
            nv = gv & pv
            angle = normal_angle(gn, pn)
            return rel, angle[nv], angle[flat_gt & nv], angle[edge_gt & nv], d, angle, nv

        # A1: current P0 protocol (nearest regrid).
        rel1, ang1, flat1, edge1, d1, map1, nv1 = measure(regrid_depth(raw, u, v, "nearest"))
        # A2: bilinear regrid.
        rel2, ang2, flat2, edge2, _, map2, nv2 = measure(regrid_depth(raw, u, v, "bilinear"))
        # A4: edge-preserving bilateral filters on the native grid, then nearest regrid.
        bilateral_results = {}
        for sigma in sigmas:
            raw_filtered = edge_preserving_filter(raw, args.bilateral_d, sigma,
                                                  args.bilateral_sigma_space)
            rel4, ang4, flat4, edge4, _, map4, nv4 = measure(regrid_depth(raw_filtered, u, v, "nearest"))
            bilateral_results[sigma] = (rel4, ang4, flat4, edge4, map4, nv4)
        # A3: normals computed on the native Omega grid (never resampled), then sampled.
        native_normals, native_valid = depth_normals(raw, pk)
        uf32, vf32 = u.astype(np.float32), v.astype(np.float32)
        sampled = np.stack([cv2.remap(native_normals[..., c].astype(np.float32), uf32, vf32, cv2.INTER_LINEAR)
                            for c in range(3)], axis=-1)
        sampled /= np.maximum(np.linalg.norm(sampled, axis=-1, keepdims=True), 1e-12)
        validity = cv2.remap(native_valid.astype(np.float32), uf32, vf32, cv2.INTER_LINEAR)
        nv3 = gv & (validity > 0.999)
        map3 = normal_angle(gn, sampled)
        ang3, flat3, edge3 = map3[nv3], map3[flat_gt & nv3], map3[edge_gt & nv3]
        # A5: A1 depth with a radius-1 stencil (operator sensitivity).
        _, ang5, flat5, edge5, _, _, _ = measure(regrid_depth(raw, u, v, "nearest"), radius=1)
        frame_values = [("regrid_nearest", (rel1, ang1, flat1, edge1)),
                        ("regrid_bilinear", (rel2, ang2, flat2, edge2)),
                        ("native_normal", (None, ang3, flat3, edge3)),
                        ("nearest_radius1", (None, ang5, flat5, edge5))]
        for sigma, (rel4, ang4, flat4, edge4, _, _) in bilateral_results.items():
            frame_values.append((f"bilateral_sc{sigma:g}", (rel4, ang4, flat4, edge4)))
        for name, values in frame_values:
            rel, ang, flt, edg = values
            if rel is not None:
                conditions[name]["rel"].append(rel)
            conditions[name]["angle"].append(ang)
            conditions[name]["flat"].append(flt)
            conditions[name]["edge"].append(edg)
        # A6: depth error on the native Omega grid (GT depth sampled there instead).
        gy, gx = np.indices(raw.shape, dtype=np.float32)
        gu = (gx - pk[0, 2]) / pk[0, 0] * k[0, 0] + k[0, 2]
        gvv = (gy - pk[1, 2]) / pk[1, 1] * k[1, 1] + k[1, 2]
        ref_on_native = cv2.remap(reference.astype(np.float32), gu.astype(np.float32),
                                  gvv.astype(np.float32), cv2.INTER_LINEAR)
        # Require all four GT taps inside the foreground so silhouette zeros cannot
        # bleed into the sampled reference depth and fabricate huge relative errors.
        fg_weight = cv2.remap(foreground.astype(np.float32), gu.astype(np.float32),
                              gvv.astype(np.float32), cv2.INTER_LINEAR)
        nv6 = (fg_weight > 0.999) & (ref_on_native > 0)
        native_grid_rel.append(np.abs(raw[nv6] - ref_on_native[nv6]) / ref_on_native[nv6])
        if index in PANEL_FRAMES:
            mid_sigma = sigmas[len(sigmas) // 2]
            map4, nv4 = bilateral_results[mid_sigma][4], bilateral_results[mid_sigma][5]
            tiles = []
            for label, heat_map, valid_map, scale_max in (
                    (f"{stem} depth rel err (A1)", map1_placeholder(d1, reference), None, 0.05),
                    ("A1 nearest normal", map1, nv1, 45),
                    ("A2 bilinear normal", map2, nv2, 45),
                    ("A3 native normal", map3, nv3, 45),
                    (f"A4 bilateral sc{mid_sigma:g} normal", map4, nv4, 45)):
                if heat_map is None:
                    heat_map = np.zeros_like(map1)
                clipped = np.uint8(np.clip(heat_map / scale_max, 0, 1) * 255)
                heat = cv2.cvtColor(cv2.applyColorMap(clipped, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                if valid_map is not None:
                    heat[~valid_map] = 96
                tiles.append((label, Image.fromarray(heat).resize((256, 256), Image.NEAREST)))
            panels.append(tiles)
    report = {"protocol": "image-space diagnosis; GT depth used for evaluation only; same 16 training views",
              "scale": scale, "conditions": {}, "native_grid_depth_error": {}}
    for name, buckets in conditions.items():
        entry = {}
        if buckets["rel"]:
            rel = np.concatenate(buckets["rel"])
            entry["depth_abs_rel"] = summarize(rel)
        for key, label in (("angle", "normal_deg"), ("flat", "normal_deg_flat_gt"),
                           ("edge", "normal_deg_edge_gt")):
            values = np.concatenate(buckets[key])
            entry[label] = summarize(values)
        report["conditions"][name] = entry
    report["native_grid_depth_error"] = {"depth_abs_rel": summarize(np.concatenate(native_grid_rel)),
                                         "note": "GT depth bilinearly sampled onto the Omega grid; "
                                                 "the predicted map itself is never resampled"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "resampling_report.json").write_text(json.dumps(report, indent=2))
    if panels:
        cols, rows = len(panels[0]), len(panels)
        canvas = Image.new("RGB", (cols * 256, rows * 284), (36, 36, 36))
        draw = ImageDraw.Draw(canvas)
        for r, tiles in enumerate(panels):
            for c, (label, tile) in enumerate(tiles):
                canvas.paste(tile, (c * 256, r * 284 + 28))
                draw.text((c * 256 + 6, r * 284 + 8), label, fill="white")
        canvas.save(out / "resampling_maps.png")
    print(json.dumps(report, indent=2), flush=True)
    return report


def map1_placeholder(d, reference):
    valid = (reference > 0) & (d > 0)
    error = np.zeros_like(d)
    error[valid] = np.abs(d[valid] - reference[valid]) / reference[valid]
    return error


PART_B_CASES = [
    # name, pose, regrid, fusion sample, bilateral filter, robust fusion
    ("repro_nearest", "pred", "nearest", "nearest", False, False),
    ("bilinear_regrid", "pred", "bilinear", "nearest", False, False),
    ("bilinear_fusion", "pred", "nearest", "bilinear", False, False),
    ("bilinear_both", "pred", "bilinear", "bilinear", False, False),
    ("bilateral", "pred", "nearest", "nearest", True, False),
    ("robust_fusion", "pred", "nearest", "nearest", False, True),
    ("combined", "pred", "bilinear", "bilinear", True, True),
    ("gtpose_bilinear_fusion", "gt", "nearest", "bilinear", False, False),
]


def part_b(pred, gt, out, scale, rotation, translation, records, predictions, gts, aligned, args):
    """Mesh-space ablation on the frozen P0 grid; measure_surfaces.py evaluates the output."""
    clouds, per_frame = [], {}
    for index, frame in enumerate(records):
        stem = Path(frame["frame"]).stem
        reference = np.load(gt / "depth" / f"{stem}.npy")
        raw = np.load(pred / "depth" / f"{stem}.npy").squeeze() * scale
        k = np.asarray(frame["K_model"])
        pk = np.asarray(predictions[frame["frame"]]["K_model"])
        yy, xx = np.indices(reference.shape, dtype=np.float32)
        u = (xx - k[0, 2]) / k[0, 0] * pk[0, 0] + pk[0, 2]
        v = (yy - k[1, 2]) / k[1, 1] * pk[1, 1] + pk[1, 2]
        foreground = np.load(gt / "foreground" / f"{stem}.npy")
        filtered_raw = edge_preserving_filter(raw, args.bilateral_d, args.bilateral_sigma_color,
                                              args.bilateral_sigma_space)
        variants = {}
        for regrid_mode in ("nearest", "bilinear"):
            for filtered in (False, True):
                source = filtered_raw if filtered else raw
                d = regrid_depth(source, u, v, regrid_mode)
                variants[(regrid_mode, filtered)] = np.where(foreground, d, 0).astype(np.float32)
        per_frame[stem] = {"reference": reference, "k": k, "variants": variants}
        rr, cc = np.nonzero(reference > 0)
        clouds.append(unproject({"depth": reference, "K": k, "T": gts[index], "rgb": None}, rr, cc))
    points = np.concatenate(clouds)
    lo, hi = points.min(0), points.max(0)
    diagonal = np.linalg.norm(hi - lo)
    voxel, trunc = diagonal / args.voxel_div, diagonal / args.voxel_div * 4
    origin, upper = lo - trunc * 2, hi + trunc * 2
    shape = tuple(np.ceil((upper - origin) / voxel).astype(int))
    p0_report_path = Path(args.p0_out) / "ablation_report.json"
    if p0_report_path.is_file():
        p0 = json.loads(p0_report_path.read_text())
        np.testing.assert_allclose(origin, p0["origin"], atol=1e-9)
        assert shape == tuple(p0["shape"]) and abs(voxel - p0["voxel"]) < 1e-12, \
            "P0.5 grid must be identical to the frozen P0 diagnostic grid"
    report = {"protocol": "diagnostic only; identical GT grid/voxel/truncation/weights as P0; "
                          "uniform fusion, no consistency filter; GT depth used for evaluation only",
              "scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist(),
              "voxel": float(voxel), "truncation": float(trunc), "origin": origin.tolist(),
              "shape": list(map(int, shape)), "robust_delta": args.robust_delta,
              "bilateral": {"d": args.bilateral_d, "sigma_color": args.bilateral_sigma_color,
                            "sigma_space": args.bilateral_sigma_space},
              "cases": {}}
    selected = set(args.cases.split(",")) if args.cases else None
    for name, pose, regrid_mode, fusion_mode, filtered, robust in PART_B_CASES:
        if selected and name not in selected:
            continue
        poses = gts if pose == "gt" else aligned
        frames = []
        for index, frame in enumerate(records):
            stem = Path(frame["frame"]).stem
            frames.append({"depth": per_frame[stem]["variants"][(regrid_mode, filtered)],
                           "K": per_frame[stem]["k"], "T": poses[index], "rgb": None})
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        started = time.time()
        field, _, weight = fuse(origin, voxel, shape, trunc, frames, 0, 100, [None] * len(frames),
                                "uniform", 64, 1, sample_mode=fusion_mode, robust=robust,
                                robust_delta=args.robust_delta)
        vertices, faces, normals = extract_surface(field, voxel)
        vertices += origin
        stats = {"pose": pose, "regrid": regrid_mode, "fusion_sample": fusion_mode,
                 "bilateral": filtered, "robust": robust,
                 "vertices": int(len(vertices)), "faces": int(len(faces))}
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
        stats["seconds"] = round(time.time() - started, 2)
        report["cases"][name] = stats
        print(name, stats, flush=True)
    observed_src = Path(args.p0_out) / "observed_gt_samples.npy"
    if observed_src.is_file():
        np.save(out / "observed_gt_samples.npy", np.load(observed_src))
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", default="data/scenes/nerf_lego16/geometry/omega")
    parser.add_argument("--gt", default="data/scenes/nerf_lego_gt16/geometry/gt")
    parser.add_argument("--out", default="deliverables/lego_p05")
    parser.add_argument("--p0-out", default="deliverables/lego_p0")
    parser.add_argument("--voxel-div", type=int, default=256)
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    parser.add_argument("--cases", default=None, help="comma-separated Part B case filter")
    parser.add_argument("--bilateral-d", type=int, default=5)
    parser.add_argument("--bilateral-sigma-color", type=float, default=0.02)
    parser.add_argument("--bilateral-sigma-space", type=float, default=3.0)
    parser.add_argument("--parta-bilateral-sigmas", default="0.02,0.05,0.10",
                        help="Part A sweeps these sigma_color values; Part B uses --bilateral-sigma-color")
    parser.add_argument("--robust-delta", type=float, default=0.3)
    args = parser.parse_args()
    pred, gt, out = Path(args.predicted), Path(args.gt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads((gt / "cameras.json").read_text())["frames"]
    predictions = {f["frame"]: f for f in json.loads((pred / "cameras.json").read_text())["frames"]}
    gts = np.asarray([f["T_world_to_cam"] for f in records])
    pts = np.asarray([predictions[f["frame"]]["T_world_to_cam"] for f in records])
    gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
    pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
    scale, rotation, translation = umeyama(pc, gc)
    aligned = pts.copy()
    aligned[:, :3, :3] = pts[:, :3, :3] @ rotation.T
    aligned[:, :3, 3] = scale * pts[:, :3, 3] - aligned[:, :3, :3] @ translation
    np.testing.assert_allclose(-np.einsum("sji,sj->si", aligned[:, :3, :3], aligned[:, :3, 3]),
                               scale * pc @ rotation.T + translation, atol=5e-6)
    if args.part in ("a", "all"):
        part_a(pred, gt, out, scale, records, predictions, args)
    if args.part in ("b", "all"):
        part_b(pred, gt, out, scale, rotation, translation, records, predictions, gts, aligned, args)


if __name__ == "__main__":
    main()
