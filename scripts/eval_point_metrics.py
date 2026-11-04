#!/usr/bin/env python3
"""Internal geometry evaluator: mesh vs GT point set (Lego observed depth or ETH3D scan).

Protocol is frozen in docs/experiments/72H_GEOMETRY.md section 7:
train-only Sim(3) from predicted to GT camera centres, 50k fixed-seed area-weighted
mesh samples, 50k GT samples, accuracy/completeness/Chamfer/F1@1vox/F2@2vox.

    python scripts/eval_point_metrics.py --mesh mesh.ply --gt scan.ply --voxel 0.01 \
        --out surface_metrics.json --pred-cameras cameras.json --gt-poses poses.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "double": "f8", "float32": "f4", "float64": "f8",
}


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_ply(path: Path | str) -> tuple[np.ndarray, np.ndarray | None]:
    """Minimal PLY reader for XYZ vertices and optional triangle faces."""
    path = Path(path)
    with path.open("rb") as f:
        raw = f.readline().decode("ascii", "replace").strip()
        if raw != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        elements = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if not line:
                continue
            if line.startswith("comment") or line.startswith("obj_info"):
                continue
            if line == "end_header":
                break
            tokens = line.split()
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                elements.append({"name": tokens[1], "count": int(tokens[2]), "props": []})
            elif tokens[0] == "property":
                if tokens[1] == "list":
                    elements[-1]["props"].append({
                        "list": True, "count_type": tokens[2], "item_type": tokens[3],
                        "name": tokens[4]})
                else:
                    elements[-1]["props"].append({
                        "list": False, "type": tokens[1], "name": tokens[2]})
        verts = None
        faces = None
        for element in elements:
            if element["name"] == "vertex":
                props = [p for p in element["props"] if not p["list"]]
                names = [p["name"] for p in props]
                if not all(k in names for k in ("x", "y", "z")):
                    raise ValueError(f"{path}: vertex element lacks x/y/z")
                dt = np.dtype([(p["name"], PLY_TYPES[p["type"]]) for p in props])
                if fmt == "ascii":
                    lines = [f.readline().decode("ascii", "replace") for _ in range(element["count"])]
                    verts = np.array([[float(v) for v in ln.split()[:len(names)]] for ln in lines],
                                     dtype=np.float64)
                    verts = np.column_stack([verts[:, names.index(k)] for k in ("x", "y", "z")])
                else:
                    endian = ">" if fmt == "binary_big_endian" else "<"
                    dt = dt.newbyteorder(endian)
                    buf = f.read(dt.itemsize * element["count"])
                    data = np.frombuffer(buf, dtype=dt, count=element["count"])
                    verts = np.column_stack([data[k].astype(np.float64) for k in ("x", "y", "z")])
            elif element["name"] == "face":
                list_prop = next((p for p in element["props"] if p["list"]), None)
                if list_prop is None:
                    continue
                idx_name = list_prop["name"]
                faces = []
                if fmt == "ascii":
                    for _ in range(element["count"]):
                        toks = f.readline().decode("ascii", "replace").split()
                        n = int(toks[0])
                        faces.append([int(x) for x in toks[1:1 + min(n, 3)]])
                else:
                    endian = ">" if fmt == "binary_big_endian" else "<"
                    cdt = np.dtype(PLY_TYPES[list_prop["count_type"]]).newbyteorder(endian)
                    idt = np.dtype(PLY_TYPES[list_prop["item_type"]]).newbyteorder(endian)
                    for _ in range(element["count"]):
                        n = int(np.frombuffer(f.read(cdt.itemsize), dtype=cdt, count=1)[0])
                        buf = f.read(idt.itemsize * n)
                        faces.append(np.frombuffer(buf, dtype=idt, count=n)[:3].tolist())
                faces = np.asarray(faces, dtype=np.int64)
    if verts is None:
        raise ValueError(f"{path}: no vertex element")
    return verts, faces


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        pts = np.load(path)
    else:
        pts, _ = read_ply(path)
    pts = np.asarray(pts, dtype=np.float64)
    pts = np.squeeze(pts)
    if pts.ndim != 2 or pts.shape[1] != 3:
        if pts.ndim == 2 and pts.shape[0] == 3:
            pts = pts.T
        else:
            raise ValueError(f"{path}: expected N x 3 points, got {pts.shape}")
    return pts[np.isfinite(pts).all(axis=1)]


def sample_mesh(verts: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 1e-15
    tri, area = tri[keep], area[keep]
    if len(tri) == 0:
        raise ValueError("mesh has no non-degenerate triangles")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tri), size=count, p=area / area.sum())
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    u = 1.0 - s
    v = s * (1.0 - r2)
    w = s * r2
    return (tri[idx, 0] * u[:, None] + tri[idx, 1] * v[:, None] + tri[idx, 2] * w[:, None])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    u, dvec, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    delta = np.diag([1.0, 1.0, sign])
    rot = u @ delta @ vt
    scale = float((np.diag(delta) @ dvec).sum() /
                  max(((src - mu_s) ** 2).sum() / len(src), 1e-12))
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def alignment(pred_cameras: Path, gt_poses: Path) -> dict:
    frames = json.loads(pred_cameras.read_text())["frames"]
    poses = {Path(p["image"]).name: p for p in json.loads(gt_poses.read_text())}
    src, dst, names = [], [], []
    for frame in frames:
        name = Path(frame["frame"]).name
        if name not in poses:
            continue
        t = np.asarray(frame["T_world_to_cam"], dtype=np.float64)
        pc = -t[:3, :3].T @ t[:3, 3]
        q, tv = np.asarray(poses[name]["qvec"], dtype=np.float64), \
            np.asarray(poses[name]["tvec"], dtype=np.float64)
        rot = quat_to_mat(q)
        gc = -rot.T @ tv
        src.append(pc)
        dst.append(gc)
        names.append(name)
    if len(names) < 3:
        raise ValueError(f"alignment needs >=3 matched cameras, got {len(names)}")
    scale, rot, trans = umeyama(np.asarray(src), np.asarray(dst))
    residual = np.linalg.norm(scale * np.asarray(src) @ rot.T + trans - np.asarray(dst), axis=1)
    span = float(np.linalg.norm(np.asarray(dst).max(0) - np.asarray(dst).min(0)))
    return {"scale": scale, "rotation": rot.tolist(), "translation": trans.tolist(),
            "views": names, "n_views": len(names),
            "centre_rms_relative": float(np.sqrt((residual ** 2).mean()) / max(span, 1e-12))}


def summarize(values: np.ndarray, voxel: float) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "mean_in_voxels": float(values.mean() / voxel)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mesh vs GT point-set geometry metrics")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--voxel", type=float, required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--pred-cameras", default=None, help="predicted cameras.json (optional)")
    parser.add_argument("--gt-poses", default=None, help="ETH3D poses.json (optional)")
    parser.add_argument("--label", default="internal_geometry_eval")
    args = parser.parse_args()

    mesh_path, gt_path, out = Path(args.mesh), Path(args.gt), Path(args.out)
    verts, faces = read_ply(mesh_path)
    if not np.isfinite(verts).all():
        raise SystemExit("mesh vertices contain NaN/Inf")
    if faces is None or len(faces) == 0:
        raise SystemExit("mesh has no triangles")

    voxel_input = float(args.voxel)
    transform = None
    if args.pred_cameras and args.gt_poses:
        transform = alignment(Path(args.pred_cameras), Path(args.gt_poses))
        s = transform["scale"]
        rot = np.asarray(transform["rotation"])
        trans = np.asarray(transform["translation"])
        verts = s * verts @ rot.T + trans
        args.voxel = voxel_input * s
    elif args.pred_cameras or args.gt_poses:
        raise SystemExit("--pred-cameras and --gt-poses must be given together")

    mesh_samples = sample_mesh(verts, faces, args.samples, args.seed)
    gt_all = load_points(gt_path)
    rng = np.random.default_rng(args.seed + 1)
    if len(gt_all) > args.samples:
        gt = gt_all[rng.choice(len(gt_all), size=args.samples, replace=False)]
    else:
        gt = gt_all

    from scipy.spatial import cKDTree
    accuracy = cKDTree(gt).query(mesh_samples, k=1)[0]
    completeness = cKDTree(mesh_samples).query(gt, k=1)[0]
    voxel = float(args.voxel)

    def scores(threshold: float) -> dict:
        precision = float((accuracy <= threshold).mean())
        recall = float((completeness <= threshold).mean())
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
        return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}

    result = {
        "protocol": "internal point-based bidirectional Chamfer; 50k fixed-seed area-weighted "
                    "mesh samples vs 50k fixed-seed GT samples; train-only Sim(3); not an "
                    "official ETH3D result unless the visibility protocol is implemented",
        "label": args.label,
        "voxel_input": voxel_input,
        "voxel": voxel,
        "mesh": str(mesh_path), "mesh_sha256_16": sha16(mesh_path),
        "gt": str(gt_path), "gt_sha256_16": sha16(gt_path),
        "counts": {"mesh_vertices": int(len(verts)), "mesh_triangles": int(len(faces)),
                   "mesh_samples": int(len(mesh_samples)), "gt_points_total": int(len(gt_all)),
                   "gt_samples": int(len(gt))},
        "alignment": None if transform is None else {k: transform[k] for k in
                     ("scale", "rotation", "translation", "n_views", "centre_rms_relative")},
        "bbox": {
            "mesh_min": verts.min(0).tolist(), "mesh_max": verts.max(0).tolist(),
            "gt_min": gt_all.min(0).tolist(), "gt_max": gt_all.max(0).tolist(),
            "gt_diagonal": float(np.linalg.norm(gt_all.max(0) - gt_all.min(0))),
            "centroid_distance_over_gt_diagonal": float(
                np.linalg.norm(verts.mean(0) - gt_all.mean(0)) /
                max(np.linalg.norm(gt_all.max(0) - gt_all.min(0)), 1e-12)),
        },
        "accuracy": summarize(accuracy, voxel),
        "completeness": summarize(completeness, voxel),
        "chamfer": float((accuracy.mean() + completeness.mean()) / 2.0),
        "fscore_1_voxel": scores(voxel),
        "fscore_2_voxel": scores(2 * voxel),
        "outlier_rate_accuracy_gt2vox": float((accuracy > 2 * voxel).mean()),
        "outlier_rate_completeness_gt2vox": float((completeness > 2 * voxel).mean()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("label", "chamfer", "fscore_1_voxel")}, indent=2))


if __name__ == "__main__":
    main()
