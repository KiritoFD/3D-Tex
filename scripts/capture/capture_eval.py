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
    args = parser.parse_args()
    capture, out = Path(args.capture), Path(args.out)
    case = args.case or capture.name
    gt = capture / "geometry" / "gt"
    pred = capture / "geometry" / "omega"
    out.mkdir(parents=True, exist_ok=True)

    p0 = json.loads((Path(args.p0_out) / "ablation_report.json").read_text())
    origin = np.asarray(p0["origin"])
    voxel, trunc = float(p0["voxel"]), float(p0["truncation"])
    shape = tuple(p0["shape"])

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

    frames = []
    rel_all, ang_all, flat_all, edge_all = [], [], [], []
    for index, frame in enumerate(records):
        stem = Path(frame["frame"]).stem
        reference = np.load(gt / "depth" / f"{stem}.npy")
        foreground = np.load(gt / "foreground" / f"{stem}.npy")
        raw = np.load(pred / "depth" / f"{stem}.npy").squeeze() * scale
        k = np.asarray(frame["K_model"])
        pk = np.asarray(predictions[frame["frame"]]["K_model"])
        yy, xx = np.indices(reference.shape, dtype=np.float32)
        u = (xx - k[0, 2]) / k[0, 0] * pk[0, 0] + pk[0, 2]
        v = (yy - k[1, 2]) / k[1, 1] * pk[1, 1] + pk[1, 2]
        d = np.where(foreground, regrid_depth(raw, u, v, "nearest"), 0).astype(np.float32)
        frames.append({"depth": d, "K": k, "T": aligned[index], "rgb": None})
        valid = (reference > 0) & (d > 0)
        rel_all.append(np.abs(d[valid] - reference[valid]) / reference[valid])
        gn, gv = depth_normals(reference, k)
        pn, pv = depth_normals(d, k)
        nv = gv & pv
        angle = normal_angle(gn, pn)
        ang_all.append(angle[nv])
        flat_all.append(angle[gt_flat_mask(gn, gv) & nv])
        edge_all.append(angle[gt_edge_mask(reference) & nv])

    started = time.time()
    field, _, weight = fuse(origin, voxel, shape, trunc, frames, 0, 100, [None] * len(frames),
                            "uniform", 64, 1, sample_mode=args.fusion_sample)
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
    report = {
        "protocol": "capture-scheme ablation; frozen P0 grid/voxel/truncation; train-only Sim3; "
                    "nearest regrid; GT depth only used for evaluation",
        "capture": str(capture), "case": case,
        "scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist(),
        "voxel": voxel, "truncation": trunc, "origin": origin.tolist(), "shape": list(map(int, shape)),
        "fusion_sample": args.fusion_sample,
        "depth_on_gt_grid": {"abs_rel": summarize(rel),
                             "normal_deg": summarize(np.concatenate(ang_all)),
                             "normal_deg_flat": summarize(np.concatenate(flat_all)),
                             "normal_deg_edge": summarize(np.concatenate(edge_all))},
        "cases": {case: stats},
    }
    observed_src = Path(args.p0_out) / "observed_gt_samples.npy"
    np.save(out / "observed_gt_samples.npy", np.load(observed_src))
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["depth_on_gt_grid"], indent=2))
    print(case, stats)


if __name__ == "__main__":
    main()
