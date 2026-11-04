"""CUDA exact surface metrics, triangle-rendered structure metrics and editable tables.

Changing F-score thresholds only requires --phase table. No noise-tolerance
decision is hardcoded. Raw distance arrays remain available for re-analysis.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))


def sample_surface(vertices, faces, count, seed):
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(faces), count, p=areas/areas.sum())
    a, b = np.sqrt(rng.random(count)), rng.random(count)
    bary = np.stack([1-a, a*(1-b), a*b], -1)
    return (tri[ids]*bary[..., None]).sum(1).astype(np.float32), ids


def entries(config, family):
    if family == "rgb":
        rows = json.loads((Path(config["rgb_sweep"]["root"])/"catalog.json").read_text())
        return [dict(row, case=row["scene"], manifest_file="perturbation.json",
                     mesh=Path(row["scene"])/"mesh/mesh_source.ply") for row in rows]
    result = json.loads((Path(config["root"])/"catalog.json").read_text())
    if family == "m2":
        result = [{"name": "initial", "case": None}, {"name": "no_external", "case": None}]+result
    out = Path(config["out"])
    for row in result:
        if row["name"] == "initial":
            row["mesh"] = out/"initial_source.ply"
        elif family == "m2":
            row["mesh"] = out/"m2"/row["name"]/"mesh_m2.ply"
        else:
            row["mesh"] = out/"scan_tsdf"/row["name"]/"mesh.ply"
    return result


def result_folder(config, family):
    return Path(config["rgb_sweep"]["out"] if family == "rgb" else config["out"])/f"{family}_eval"


def evaluate(config, family, only):
    import warp as wp
    from measurement_error.rays import cast, closest
    from ply_to_obj import read_ply
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import medial_axis

    wp.init()
    device = wp.get_device("cuda:0")
    assert device.is_cuda
    master, out = Path(config["master"]), result_folder(config, family)
    out.mkdir(parents=True, exist_ok=True)
    scale = json.loads((master/"scale.json").read_text())["mm_per_source_unit"]
    raw = np.load(master/"source_mesh.npz")
    sv, sf = raw["vertices"], raw["faces"]
    keep = (np.abs(sv[sf]) < 5).all(axis=(1, 2))
    # Original index of every retained source triangle, so per-triangle flags
    # computed on the unfiltered source mesh stay aligned after the far-plane cut.
    original_ids = np.nonzero(keep)[0]
    sf = sf[keep]
    flags_path = ROOT/"outputs/lego_source_edge_flags.npy"
    sharp = None
    if flags_path.exists():
        dihedral = np.load(flags_path)
        if len(dihedral) != len(keep):
            raise ValueError("source edge flags do not match the unfiltered source triangle count")
        sharp = dihedral[original_ids] > config.get("source_sharp_deg", 40.0)
    def mesh_of(vertices, faces):
        return wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device),
                       indices=wp.array(faces.ravel(), dtype=wp.int32, device=device))
    def distances(mesh, points):
        values = wp.empty(len(points), dtype=wp.float32, device=device)
        indices = wp.empty(len(points), dtype=wp.int32, device=device)
        wp.launch(closest, dim=len(points), inputs=[mesh.id, wp.array(points, dtype=wp.vec3, device=device)],
                  outputs=[values, indices], device=device)
        d, ids = values.numpy()*scale, indices.numpy()
        assert (ids >= 0).all() and np.isfinite(d).all()
        return d, ids
    source_mesh = mesh_of(sv, sf)
    count, seed = config["evaluation_samples"], config["evaluation_seed"]
    full, _ = sample_surface(sv, sf, count, seed)
    reference = master/"geometry/gt_ray"
    cameras = json.loads((reference/"cameras.json").read_text())["frames"]
    observed, structures = [], []
    for frame in cameras:
        d = np.load(reference/"depth"/(Path(frame["frame"]).stem+".npy"))
        fg = d > 0
        yy, xx = np.nonzero(fg)
        k, t = np.array(frame["K_model"]), np.array(frame["T_world_to_cam"])
        z = d[yy, xx]
        camera = np.stack([(xx-k[0, 2])/k[0, 0]*z, (yy-k[1, 2])/k[1, 1]*z, z], -1)
        observed.append((camera-t[:3, 3]) @ t[:3, :3])
        skeleton, radius = medial_axis(fg, return_distance=True, rng=seed)
        structures.append((frame, d, binary_fill_holes(fg) & ~fg, skeleton & (radius <= 2)))
    observed = np.concatenate(observed).astype(np.float32)
    observed = observed[np.random.default_rng(seed).choice(len(observed), count, replace=False)]
    # A self-distance check measures numerical floor, not reconstruction accuracy.
    floor, _ = distances(source_mesh, full)
    print("Source self-distance mm quantiles", np.quantile(floor, [.5, .9, .99, .999, 1]).tolist(), flush=True)
    assert float(np.quantile(floor, .999)) < .002
    for entry in entries(config, family):
        if only and entry["name"] not in only:
            continue
        if not entry["mesh"].exists():
            continue
        vertices, faces, _ = read_ply(entry["mesh"])
        vertices, faces = vertices.astype(np.float32), faces.astype(np.int32)
        mesh = mesh_of(vertices, faces)
        points, ids = sample_surface(vertices, faces, count, seed)
        accuracy, source_ids = distances(source_mesh, points)
        completeness, _ = distances(mesh, full)
        seen, _ = distances(mesh, observed)
        def normals(v, f):
            triangles = v[f]
            n = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            return n/np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        angles = np.degrees(np.arccos(np.clip(np.abs((normals(vertices, faces[ids])*
                    normals(sv, sf[source_ids])).sum(-1)), 0, 1)))
        thin_errors, filled, holes = [], 0, 0
        for frame, truth, hole, thin in structures:
            k, t = np.asarray(frame["K_model"]), np.asarray(frame["T_world_to_cam"])
            rotation = t[:3, :3].T.astype(np.float32)
            origin = (-rotation @ t[:3, 3]).astype(np.float32)
            depth = wp.zeros(truth.shape, dtype=wp.float32, device=device)
            wp.launch(cast, dim=truth.shape, inputs=[mesh.id, wp.vec3(*origin), wp.mat33(rotation),
                float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])], outputs=[depth], device=device)
            predicted = depth.numpy()
            filled += int((predicted[hole] > 0).sum())
            holes += int(hole.sum())
            thin_errors.append(np.where(predicted[thin] > 0, np.abs(predicted[thin]-truth[thin])*scale, np.inf))
        target = out/entry["name"]
        target.mkdir(exist_ok=True)
        saved = {"accuracy_mm": accuracy, "completeness_mm": completeness,
                 "observed_mm": seen, "normal_deg": angles,
                 "thin_depth_error_mm": np.concatenate(thin_errors),
                 "source_face_ids": original_ids[source_ids]}
        if sharp is not None:
            saved["normal_flat_deg"] = angles[~sharp[source_ids]]
            saved["normal_edge_deg"] = angles[sharp[source_ids]]
        np.savez_compressed(target/"distances.npz", **saved)
        meta = {"family": family, "case": entry["name"], "mesh_sha256": hashlib.sha256(entry["mesh"].read_bytes()).hexdigest(),
            "scan": json.loads((Path(entry["case"])/entry.get("manifest_file", "manifest.json")).read_text()) if entry["case"] else None,
            "config": config, "mm_per_source_unit": scale, "hole_fill_rate": filled/holes if holes else None,
            "hole_pixels": holes, "thin_proxy": "GT 512px medial-axis pixels with radius <=2px; not semantic rods",
            "normal_split": None if sharp is None else {
                "source_sharp_deg": config.get("source_sharp_deg", 40.0),
                "definition": "nearest source triangle max dihedral above threshold counts as sharp structure",
                "sharp_fraction_of_source": float(sharp.mean()),
                "flat_samples": int((~sharp[source_ids]).sum()), "edge_samples": int(sharp[source_ids].sum())},
            "gpu": device.name, "warp": wp.__version__, "source_self_distance_p999_mm": float(np.quantile(floor, .999)),
            "scope": "calibrated synthetic rig; full-source and observed distances separate; no acceptance decision"}
        omega = Path(entry["case"])/"geometry/omega" if family == "rgb" else Path(config["rgb_run"])/"geometry/omega"
        model = json.loads((omega/"manifest.json").read_text())
        meta["model"] = {key: model[key] for key in ("checkpoint", "aggregator_dtype", "heads_dtype", "seed")}
        meta["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__), Path(__file__).with_name("rays.py"))}
        (target/"metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
        print(entry["name"], "Chamfer full mm", float((accuracy.mean()+completeness.mean())/2), flush=True)


def table(config, family):
    if not config["fscore_thresholds_mm"] or any(t <= 0 for t in config["fscore_thresholds_mm"]):
        raise ValueError("F-score thresholds must be positive millimeters")
    folder = result_folder(config, family)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries(config, family):
        path = folder/entry["name"]
        row = {"case": entry["name"], "status": "pending"}
        if entry["case"]:
            manifest = Path(entry["case"])/entry.get("manifest_file", "manifest.json")
            scan = json.loads(manifest.read_text()) if manifest.exists() else entry
            row.update(noise_mode=scan.get("noise_mode", scan.get("mode")), seed=scan["seed"],
                       target_rms_mm=scan.get("target_rms_mm", scan.get("rms_mm")), realized_rms_mm=scan.get("realized_rms_mm"))
            prediction = Path(entry["case"])/"raw_prediction.json"
            if prediction.exists():
                raw = json.loads(prediction.read_text())
                row.update(vggt_depth_mae_mm=raw["mae_mm"], vggt_depth_rmse_mm=raw["rmse_mm"],
                           vggt_abs_rel=raw["abs_rel"], vggt_normal_deg=raw["normal_deg"],
                           vggt_rpe_rot_deg=raw["rpe_rot_deg"], vggt_camera_rmse_mm=raw["camera_center_rmse_mm"],
                           vggt_valid_fraction=raw["depth_valid_fraction"])
        if (path/"distances.npz").exists():
            d = np.load(path/"distances.npz")
            meta = json.loads((path/"metadata.json").read_text())
            row.update(status="complete", chamfer_full_mm=float((d["accuracy_mm"].mean()+d["completeness_mm"].mean())/2),
                chamfer_observed_mm=float((d["accuracy_mm"].mean()+d["observed_mm"].mean())/2),
                normal_deg=float(d["normal_deg"].mean()), hole_fill=meta["hole_fill_rate"])
            if "normal_flat_deg" in d:
                row.update(normal_flat_deg=float(d["normal_flat_deg"].mean()),
                           normal_edge_deg=float(d["normal_edge_deg"].mean()))
            for threshold in config["fscore_thresholds_mm"]:
                precision = float((d["accuracy_mm"] <= threshold).mean())
                recall = float((d["completeness_mm"] <= threshold).mean())
                seen = float((d["observed_mm"] <= threshold).mean())
                row.update({f"precision@{threshold:g}mm": precision, f"recall@{threshold:g}mm": recall,
                    f"fscore@{threshold:g}mm": 2*precision*recall/max(precision+recall, 1e-12),
                    f"observed_fscore@{threshold:g}mm": 2*precision*seen/max(precision+seen, 1e-12),
                    f"thin_proxy_recall@{threshold:g}mm": float((d["thin_depth_error_mm"] <= threshold).mean())})
        rows.append(row)
    clean = {(r.get("noise_mode"), r.get("seed")): r for r in rows if r.get("target_rms_mm") == 0 and r["status"] == "complete"}
    for row in rows:
        baseline = clean.get((row.get("noise_mode"), row.get("seed")))
        if baseline and row["status"] == "complete":
            row["chamfer_ratio_to_clean"] = row["chamfer_full_mm"]/max(baseline["chamfer_full_mm"], 1e-12)
            for threshold in config["fscore_thresholds_mm"]:
                key = f"fscore@{threshold:g}mm"
                row[f"{key}_ratio_to_clean"] = row[key]/max(baseline[key], 1e-12)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (folder/"results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Error sweep results", "", "Pending rows are unexecuted, not zero. No tolerance verdict is applied.", "",
             "| Case | Status | Input RMS mm | Chamfer full mm | Chamfer observed mm | Normal deg | Hole fill |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row.get(key) for key in ("realized_rms_mm", "chamfer_full_mm", "chamfer_observed_mm", "normal_deg", "hole_fill")]
        lines.append(f"| {row['case']} | {row['status']} | "+" | ".join("—" if v is None else f"{v:.6f}" for v in values)+" |")
    (folder/"results.md").write_text("\n".join(lines)+"\n")
    grouped = []
    modes = config["rgb_sweep"]["modes"] if family == "rgb" else config["noise_modes"]
    levels = config["rgb_sweep"]["rms_mm"] if family == "rgb" else config["noise_rms_mm"]
    for mode in modes:
        for level in levels:
            completed = [r for r in rows if r.get("noise_mode") == mode and r.get("target_rms_mm") == level and r["status"] == "complete"]
            group = {"noise_mode": mode, "target_rms_mm": level, "n_complete": len(completed), "n_expected": len(config["seeds"])}
            for key in fields:
                if key in ("seed", "target_rms_mm"):
                    continue
                values = [r[key] for r in completed if isinstance(r.get(key), (float, int))]
                if values:
                    group[key+"_mean"], group[key+"_std"] = float(np.mean(values)), float(np.std(values))
            grouped.append(group)
    keys = list(dict.fromkeys(k for g in grouped for k in g))
    with (folder/"grouped.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(grouped)
    record = {"family": family, "config": config, "tolerance_verdict": None,
              "raw_distance_sha256": {row["case"]: hashlib.sha256(
                  (folder/row["case"]/"distances.npz").read_bytes()).hexdigest()
                  for row in rows if row["status"] == "complete"}}
    (folder/"table_config.json").write_text(json.dumps(record, indent=2)+"\n")
    print(folder/"results.csv", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/measurement_error.json")
    p.add_argument("--family", choices=["m2", "scan_tsdf", "rgb"], default="m2")
    p.add_argument("--phase", choices=["evaluate", "table", "all"], required=True)
    p.add_argument("--only")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.phase in ("evaluate", "all"):
        evaluate(config, args.family, set(args.only.split(",")) if args.only else None)
    if args.phase in ("table", "all"):
        table(config, args.family)


if __name__ == "__main__":
    main()
