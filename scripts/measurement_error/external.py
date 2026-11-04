"""P1.1 external (camera pose) error injected before TSDF and before M2.

Two injection points are kept separate, because they answer different questions:

  tsdf  -- the registered VGGT poses are perturbed, the cloud is re-fused into a
           new initial mesh, and M2 runs with ideal cameras. This measures how
           much a wrong initial surface can be repaired by the fitting stage.
  m2    -- the frozen initial mesh is untouched; only the differentiable render
           cameras (and the scan look-up) are perturbed.
  joint -- the same perturbation is applied at both stages, to test additivity.

Gauge rule (R0): one Sim(3) registers the VGGT gauge to the GT gauge *before*
any error is injected. The perturbation is added in that gauge and the GT
cameras are never used to re-fit the registration afterwards.

Every distribution draws one unit field per seed and only scales it, so all
levels of one curve share paired samples.

Phases:
  prepare  -- calibration records (and, for tsdf/joint, the re-fused initial mesh)
  run      -- serial M2 optimisation per case with a GPU free-memory check
  evaluate -- Warp surface metrics per position
  collect  -- paired curve tables (mean/std over seeds, equivalence curve)
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(config):
    return config["r1"]["external"]


def out_root(config):
    return Path(config["r1"]["out"])


def names(config, position=None):
    """Case list; the zero level is the frozen R0 baseline and is not re-run.

    Joint cases are only meaningful at the joint position: applying translation
    *and* rotation at the M2 stage alone is not the superposition being tested.
    """
    item = spec(config)
    cases = []
    for level in item["translation_mm"]:
        if level == 0:
            continue
        for distribution in item["distributions"]:
            for seed in item["seeds"]:
                cases.append({"name": f"t{level:g}_{distribution}_s{seed}", "family": "translation",
                              "translation_mm": float(level), "rotation_deg": 0.,
                              "distribution": distribution, "seed": seed})
    for level in item["rotation_deg"]:
        if level == 0:
            continue
        for distribution in item["distributions"]:
            for seed in item["seeds"]:
                cases.append({"name": f"rot{level:g}_{distribution}_s{seed}", "family": "rotation",
                              "translation_mm": 0., "rotation_deg": float(level),
                              "distribution": distribution, "seed": seed})
    for translation, rotation in item["joint_pairs"]:
        for distribution in item["distributions"]:
            for seed in item["seeds"]:
                cases.append({"name": f"j{translation:g}mm_{rotation:g}deg_{distribution}_s{seed}",
                              "family": "joint", "translation_mm": float(translation),
                              "rotation_deg": float(rotation), "distribution": distribution, "seed": seed})
    if position is not None and position != "joint":
        cases = [case for case in cases if case["family"] != "joint"]
    return cases


def zero_reference(config):
    """R0 zero-noise repetitions are byte-identical inputs to the m2 zero case."""
    return Path(config["out"])/"repeatability"/"zero_v2"


def alignment(config):
    """The one frozen Sim(3) and the clean fusion grid, both computed once."""
    from measurement_error.calibration import align_poses
    from tsdf_mesh import load_frame, unproject
    from geometry_masks import load_masks

    rgb = Path(config["rgb_run"])
    resolution = config["observation_resolution"]
    frames = json.loads((rgb/"geometry/omega/cameras.json").read_text())["frames"]
    gt = {Path(f["frame"]).stem: f for f in json.loads(
        (Path(config["master"])/"geometry"/f"gt_ray{resolution}"/"cameras.json").read_text())["frames"]}
    poses = np.asarray([f["T_world_to_cam"] for f in frames])
    gt_poses = np.asarray([gt[Path(f["frame"]).stem]["T_world_to_cam"] for f in frames])
    aligned, registration = align_poses(poses, gt_poses)
    scale = float(registration["scale"])
    mm_per_source_unit = json.loads((Path(config["master"])/"scale.json").read_text())["mm_per_source_unit"]
    masks = load_masks(rgb, frames, "external", rgb/"masks")
    geometry = [load_frame(f, rgb/"geometry/omega", rgb) for f in frames]
    clouds = []
    for view, mask in zip(geometry, masks):
        view["depth"] = np.where(mask, view["depth"], 0)
        rows, cols = np.nonzero(np.isfinite(view["depth"]) & (view["depth"] > 0))
        clouds.append(unproject(view, rows, cols))
    points = np.concatenate(clouds)
    lo, hi = points.min(0), points.max(0)
    voxel = float(np.linalg.norm(hi-lo)/config["tsdf_voxel_divisor"])
    trunc = config["tsdf_truncation_voxels"]*voxel
    lo -= 2*trunc
    hi += 2*trunc
    shape = tuple(np.ceil((hi-lo)/voxel).astype(int))
    # The grid stays axis-aligned in the *predicted* (native) gauge, exactly like
    # native_fusion. A rotation applied to the GT poses would not keep an
    # axis-aligned grid valid, so the injection is mapped back to the native
    # gauge and the extracted vertices are transformed once, as the pipeline does.
    return {"scale": scale, "rotation": registration["rotation"], "translation": registration["translation"],
            "origin_native": lo.tolist(), "voxel_native": voxel, "trunc_native": trunc,
            "shape": list(map(int, shape)), "registration": registration,
            "aligned_poses": aligned.tolist(), "frames": frames,
            "mm_per_source_unit": mm_per_source_unit}


def write_record(config, position, case, reference):
    from measurement_error.calibration import build
    record = build(config, case["translation_mm"], case["rotation_deg"], case["distribution"], case["seed"],
                   calibration_mode="non_ideal")
    injected = record["injected"]
    # The perturbation is applied to the already-registered poses, so the
    # injected RMS *is* the camera RPE against the locked ideal gauge.
    record["gauge"] = {"registration": "locked", "injected_after_registration": True,
                       "gt_used_for_refit": False}
    for stem, frame in record["frames"].items():
        pose = np.asarray(frame["T_world_to_cam"])
        anchor = np.asarray(reference["aligned_poses"][
            next(i for i, f in enumerate(reference["frames"]) if Path(f["frame"]).stem == stem)])
        record["frames"][stem]["camera_rpe_translation_mm"] = float(
            np.linalg.norm(pose[:3, 3]-anchor[:3, 3])*reference["mm_per_source_unit"])
        record["frames"][stem]["camera_rpe_rotation_deg"] = float(
            np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3] @ anchor[:3, :3].T)-1)/2, -1, 1))))
    record["camera_rpe_summary"] = {
        "translation_rms_mm": float(np.sqrt(np.mean([f["camera_rpe_translation_mm"]**2 for f in record["frames"].values()])),
                                    ),
        "rotation_rms_deg": float(np.sqrt(np.mean([f["camera_rpe_rotation_deg"]**2 for f in record["frames"].values()]))),
        "definition": "camera-to-camera RPE against the locked ideal gauge (no GT refit)"}
    path = out_root(config)/"external"/position/"calib"/f"{case['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2)+"\n")
    return path, injected


def fuse_initial(config, case, record, reference):
    """Re-fuse the predicted cloud with perturbed poses, on the frozen clean grid."""
    from tsdf_mesh import load_frame, fuse, extract_surface, write_ply
    from geometry_masks import load_masks

    rgb = Path(config["rgb_run"])
    frames = reference["frames"]
    masks = load_masks(rgb, frames, "external", rgb/"masks")
    scale = reference["scale"]
    rotation = np.asarray(reference["rotation"])
    translation = np.asarray(reference["translation"])
    geometry, confs = [], []
    for frame, mask in zip(frames, masks):
        stem = Path(frame["frame"]).stem
        pose_gt = np.asarray(record["frames"][stem]["T_world_to_cam"], dtype=np.float64)
        # Invert the locked similarity back to the predicted gauge: P = (1/s) P' S.
        pose_native = np.eye(4)
        pose_native[:3, :3] = pose_gt[:3, :3] @ rotation
        pose_native[:3, 3] = (pose_gt[:3, :3] @ translation+pose_gt[:3, 3])/scale
        item = dict(frame)
        item["T_world_to_cam"] = pose_native.tolist()
        view = load_frame(item, rgb/"geometry/omega", rgb)
        view["depth"] = np.where(mask, view["depth"], 0)
        geometry.append(view)
        confs.append(np.load(rgb/"geometry/omega"/"confidence"/(stem+".npy")).squeeze())
    far = float(np.median(np.concatenate([g["depth"][g["depth"] > 0] for g in geometry]))*6)
    started = time.perf_counter()
    field, _, _ = fuse(np.asarray(reference["origin_native"]), reference["voxel_native"],
                       tuple(reference["shape"]), reference["trunc_native"], geometry, 0, far, confs,
                       "confidence", 64, 1)
    vertices, faces, normals = extract_surface(field, reference["voxel_native"])
    vertices += np.asarray(reference["origin_native"])
    vertices = scale*(vertices @ rotation.T)+translation
    out = out_root(config)/"external"/"tsdf"/"initial"/case["name"]
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out/"initial_source.ply", vertices, faces, np.full_like(vertices, .65), normals)
    (out/"fusion.json").write_text(json.dumps({"case": case["name"], "seconds": time.perf_counter()-started,
        "voxel_native": reference["voxel_native"], "trunc_native": reference["trunc_native"],
        "shape": reference["shape"], "vertices": len(vertices), "faces": len(faces),
        "grid": "frozen from the clean fusion; only poses differ",
        "cloud": "predicted depth unprojected with the perturbed registered poses, mapped back to the predicted gauge",
        "mm_per_source_unit": reference["mm_per_source_unit"],
        "calibration": str(out_root(config)/"external"/"tsdf"/"calib"/f"{case['name']}.json"),
        "gt_used_for_refit": False}, indent=2)+"\n")
    print("Fused", case["name"], flush=True)


def prepare(config, only):
    reference = alignment(config)
    directory = out_root(config)/"external"
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"alignment.json").write_text(json.dumps({k: reference[k] for k in
        ("scale", "rotation", "translation", "origin_native", "voxel_native", "trunc_native", "shape")},
        indent=2)+"\n")
    positions = spec(config)["positions"]
    for position in positions:
        for case in names(config, position):
            if only and case["name"] not in only:
                continue
            record, injected = write_record(config, position, case, reference)
            print("Record", position, case["name"], injected["translation_rms_mm"],
                  injected["rotation_rms_deg"], flush=True)
            if position in ("tsdf", "joint"):
                target = directory/"tsdf"/"initial"/case["name"]/"initial_source.ply"
                if not target.exists():
                    fuse_initial(config, case, json.loads(record.read_text()), reference)


def m2_command(config, name, mesh, target, calibration, omega=None):
    optimizer = config["optimizer"]
    init = json.loads((Path(config["out"])/"initialization.json").read_text())
    command = [sys.executable, "scripts/m2_anchor_opt.py", "--mesh", str(mesh), "--out", str(target),
               "--omega", str(omega or (Path(config["rgb_run"])/"geometry/omega")),
               "--gt", str(Path(config["master"])/"geometry"/f"gt_ray{config['observation_resolution']}"),
               "--camera-mode", "diagnostic", "--resolution", str(config["observation_resolution"]),
               "--voxel-size", str(init["voxel_source"]), "--iters", str(optimizer["iters"]),
               "--lr", str(optimizer["lr"]), "--deformation", optimizer["deformation"],
               "--lattice-spacing-voxels", str(optimizer["lattice_spacing_voxels"]),
               "--lambda-prior", str(optimizer["lambda_prior"]),
               "--tolerance-voxels", str(optimizer["tolerance_voxels"]),
               "--pixel-center-offset", ".5", "--perspective-projection",
               "--external-depth", str(zero_scan(config)), "--lambda-external", str(optimizer["lambda_external"]),
               "--seed", str(optimizer.get("seed", 20261103))]
    if calibration:
        command += ["--calibration", str(calibration)]
    return command


def zero_scan(config):
    catalog = json.loads((Path(config["root"])/"catalog.json").read_text())
    for entry in catalog:
        manifest = json.loads((Path(entry["case"])/"manifest.json").read_text())
        if float(manifest["realized_rms_mm"]) == 0.:
            return Path(entry["case"])
    raise ValueError("no zero-error scan case")


def gpu_free_gib():
    values = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    used, total = (float(x) for x in values.split(","))
    return total/1024., (total-used)/1024., used/1024.


def run(config, only):
    for position in spec(config)["positions"]:
        for case in names(config, position):
            if only and case["name"] not in only:
                continue
            target = out_root(config)/"external"/position/"m2"/case["name"]
            if (target/"m2_log.json").exists():
                print("SKIP", position, case["name"], flush=True)
                continue
            total, free, used = gpu_free_gib()
            if free < spec(config)["min_free_gib"]:
                raise SystemExit(f"GPU busy before {position}/{case['name']}: {free:.2f} GiB free of {total:.2f}")
            record = out_root(config)/"external"/position/"calib"/f"{case['name']}.json"
            if position == "m2":
                mesh = Path(config["out"])/"initial_source.ply"
                calibration = record
            elif position == "tsdf":
                mesh = out_root(config)/"external"/"tsdf"/"initial"/case["name"]/"initial_source.ply"
                calibration = None
            else:
                mesh = out_root(config)/"external"/"tsdf"/"initial"/case["name"]/"initial_source.ply"
                calibration = record
            if not mesh.exists():
                raise FileNotFoundError(f"missing {mesh}; run the prepare phase first")
            target.mkdir(parents=True, exist_ok=True)
            command = m2_command(config, case["name"], mesh, target, calibration)
            with (out_root(config)/"external"/position/f"{case['name']}.log").open("w") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            (target/"gpu_before.json").write_text(json.dumps({"total_gib": total, "free_gib": free,
                "used_gib": used, "nvidia_smi": "memory.used,memory.total"}, indent=2)+"\n")
            print("Completed", position, case["name"], flush=True)


def evaluate(config):
    for position in spec(config)["positions"]:
        root = out_root(config)/"external"/position
        catalog = [{"name": case["name"], "case": None} for case in names(config, position)]
        (root/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
        local = dict(config, out=str(root), root=str(root))
        (root/"config.json").write_text(json.dumps(local, indent=2)+"\n")
        subprocess.run([sys.executable, "scripts/measurement_error/evaluate.py", "--config", str(root/"config.json"),
                        "--family", "m2", "--phase", "all"], cwd=ROOT, check=True)


def case_metrics(folder):
    data = np.load(folder/"distances.npz")
    meta = json.loads((folder/"metadata.json").read_text())
    precision = float((data["accuracy_mm"] <= 1.).mean())
    recall = float((data["completeness_mm"] <= 1.).mean())
    seen = float((data["observed_mm"] <= 1.).mean())
    out = {"chamfer_full_mm": float((data["accuracy_mm"].mean()+data["completeness_mm"].mean())/2),
           "chamfer_observed_mm": float((data["accuracy_mm"].mean()+data["observed_mm"].mean())/2),
           "normal_deg": float(data["normal_deg"].mean()),
           "normal_flat_deg": float(data["normal_flat_deg"].mean()) if "normal_flat_deg" in data else None,
           "normal_edge_deg": float(data["normal_edge_deg"].mean()) if "normal_edge_deg" in data else None,
           "precision@1mm": precision, "recall@1mm": recall,
           "observed_fscore@1mm": 2*precision*seen/max(precision+seen, 1e-12),
           "thin_recall@1mm": float((data["thin_depth_error_mm"] <= 1.).mean()),
           "hole_fill": meta["hole_fill_rate"],
           "accuracy_p90_mm": float(np.quantile(data["accuracy_mm"], .9)),
           "accuracy_p99_mm": float(np.quantile(data["accuracy_mm"], .99)),
           "observed_p90_mm": float(np.quantile(data["observed_mm"], .9)),
           "observed_p99_mm": float(np.quantile(data["observed_mm"], .99))}
    return out


def zero_metrics(config):
    """10 R0 repetitions on identical inputs; used as the paired zero control."""
    path = zero_reference(config)/"metrics.json"
    data = json.loads(path.read_text())["summary"]
    keys = {"chamfer_full_mm": "chamfer_full_mm", "chamfer_observed_mm": "chamfer_observed_mm",
            "normal_deg": "normal_deg", "normal_flat_deg": "normal_flat_deg", "normal_edge_deg": "normal_edge_deg",
            "precision@1mm": "precision@1mm", "recall@1mm": "recall@1mm",
            "observed_fscore@1mm": "observed_fscore@1mm", "thin_recall@1mm": "thin_recall@1mm",
            "hole_fill": "hole_fill", "accuracy_p90_mm": "accuracy_p90_mm", "accuracy_p99_mm": "accuracy_p99_mm",
            "observed_p90_mm": "observed_p90_mm", "observed_p99_mm": "observed_p99_mm"}
    return {key: {"mean": data[value]["mean"], "std": data[value]["std"],
                  "min": data[value]["min"], "max": data[value]["max"], "n": 10}
            for key, value in keys.items()}, path


def collect(config):
    zero, zero_path = zero_metrics(config)
    summary = {}
    rows = []
    for position in spec(config)["positions"]:
        root = out_root(config)/"external"/position
        cases = {case["name"]: case for case in names(config, position)}
        for name, case in cases.items():
            folder = root/"m2_eval"/name
            if not (folder/"distances.npz").exists():
                continue
            values = case_metrics(folder)
            record = json.loads((root/"calib"/f"{name}.json").read_text())
            log = json.loads((root/"m2"/name/"m2_log.json").read_text())
            rows.append(dict(values, position=position, name=name, family=case["family"],
                             target_translation_mm=case["translation_mm"], target_rotation_deg=case["rotation_deg"],
                             distribution=case["distribution"], seed=case["seed"],
                             camera_rpe_translation_mm=record["camera_rpe_summary"]["translation_rms_mm"],
                             camera_rpe_rotation_deg=record["camera_rpe_summary"]["rotation_rms_deg"],
                             final_loss=log["iters"][-1]["loss"], seconds=log["seconds"]))
        for position_ in (position,):
            for family in ("translation", "rotation", "joint"):
                for distribution in spec(config)["distributions"]:
                    levels = sorted({(row["target_translation_mm"], row["target_rotation_deg"]) for row in rows
                                     if row["position"] == position_ and row["family"] == family
                                     and row["distribution"] == distribution})
                    for translation, rotation in levels:
                        group = [row for row in rows if row["position"] == position_ and row["family"] == family
                                 and row["distribution"] == distribution
                                 and (row["target_translation_mm"], row["target_rotation_deg"]) == (translation, rotation)]
                        if not group:
                            continue
                        record = {"position": position_, "family": family, "distribution": distribution,
                                  "translation_mm": translation, "rotation_deg": rotation,
                                  "n_complete": len(group), "seeds": sorted(row["seed"] for row in group)}
                        for key in group[0]:
                            if key in ("position", "name", "family", "distribution", "seed",
                                       "target_translation_mm", "target_rotation_deg"):
                                continue
                            values = [row[key] for row in group if row[key] is not None]
                            record[key+"_mean"] = float(np.mean(values))
                            record[key+"_std"] = float(np.std(values))
                        control = zero
                        for key in ("chamfer_observed_mm", "normal_deg", "observed_fscore@1mm", "recall@1mm",
                                    "normal_flat_deg", "normal_edge_deg", "hole_fill", "thin_recall@1mm",
                                    "observed_p90_mm", "observed_p99_mm", "camera_rpe_translation_mm",
                                    "camera_rpe_rotation_deg"):
                            if key+"_mean" not in record or key not in control:
                                continue
                            base = control[key]
                            record["delta_"+key] = record[key+"_mean"]-base["mean"]
                            band = max(2*base["std"], base["max"]-base["min"])
                            record["floor_"+key] = band
                            record["ratio_"+key] = record[key+"_mean"]/base["mean"] if base["mean"] else None
                        summary.setdefault(position_, []).append(record)
    (out_root(config)/"external"/"rows.json").write_text(json.dumps(rows, indent=2)+"\n")
    (out_root(config)/"external"/"summary.json").write_text(json.dumps(
        {"zero_control": zero, "zero_control_source": str(zero_path), "groups": summary,
         "floor_rule": "b = max(2 sigma, range) of the 10 R0 zero repetitions; 2b is the acceptance floor"},
        indent=2)+"\n")
    depth = observed_depth(config)
    (out_root(config)/"external"/"equivalence.json").write_text(json.dumps({
        "mean_observed_depth_mm": depth,
        "per_degree_lateral_mm": float(np.sin(np.deg2rad(1.))*depth),
        "note": "lateral displacement at the object = depth * sin(angle); rotation and translation are "
                "directly comparable only at one depth, which is why the equivalence curve is provided",
        "control_0p1deg_mm": float(np.sin(np.deg2rad(.1))*depth)},
        indent=2)+"\n")
    print(json.dumps({"rows": len(rows), "positions": list(summary)}, indent=2), flush=True)


def landmarks(config):
    """Smallest resolvable error, largest error that still meets the frozen
    criteria, and the plateau position, per curve."""
    summary = json.loads((out_root(config)/"external"/"summary.json").read_text())
    criteria = config["r1"]["criteria"]
    depth = json.loads((out_root(config)/"external"/"equivalence.json").read_text())["mean_observed_depth_mm"]
    curves = summary["groups"]
    table = {}
    for position, records in curves.items():
        for record in records:
            channel = f"{position}/{record['family']}/{record['distribution']}"
            level = record["translation_mm"] if record["family"] == "translation" else record["rotation_deg"]
            entry = table.setdefault(channel, {"position": position, "family": record["family"],
                                               "distribution": record["distribution"], "levels": []})
            entry["levels"].append({
                "level": level,
                "equivalent_lateral_mm": float(np.sin(np.deg2rad(level))*depth) if record["family"] == "rotation"
                else level,
                "chamfer_mean": record["chamfer_observed_mm_mean"],
                "chamfer_std": record["chamfer_observed_mm_std"],
                "chamfer_delta": record["delta_chamfer_observed_mm"],
                "chamfer_floor": record["floor_chamfer_observed_mm"],
                "resolvable": bool(record["delta_chamfer_observed_mm"] > record["floor_chamfer_observed_mm"]),
                "f_mean": record["observed_fscore@1mm_mean"],
                "recall_mean": record["recall@1mm_mean"],
                "normal_mean": record["normal_deg_mean"],
                "normal_flat": record["normal_flat_deg_mean"],
                "normal_edge": record["normal_edge_deg_mean"],
                "hole_fill": record["hole_fill_mean"],
                "thin_recall": record["thin_recall@1mm_mean"],
                "p90": record["observed_p90_mm_mean"], "p99": record["observed_p99_mm_mean"],
                "camera_rpe_translation_mm": record["camera_rpe_translation_mm_mean"],
                "camera_rpe_rotation_deg": record["camera_rpe_rotation_deg_mean"],
                "meets_criteria": bool(
                    record["chamfer_observed_mm_mean"] <= criteria["chamfer_max_mm"]
                    and record["observed_fscore@1mm_mean"] >= criteria["observed_f_min"]
                    and record["recall@1mm_mean"] >= criteria["recall_min"]
                    and record["normal_deg_mean"] <= criteria["normal_max_deg"]
                    and abs(record["hole_fill_mean"]-summary["zero_control"]["hole_fill"]["mean"])
                    <= criteria["hole_fill_abs"]
                    and abs(record["thin_recall@1mm_mean"]-summary["zero_control"]["thin_recall@1mm"]["mean"])
                    <= criteria["thin_recall_abs"])})
            entry["levels"].sort(key=lambda row: row["level"])
        for channel, entry in table.items():
            levels = entry["levels"]
            resolvable = [row["level"] for row in levels if row["resolvable"]]
            passing = [row["level"] for row in levels if row["meets_criteria"]]
            entry["resolvable_from"] = min(resolvable) if resolvable else None
            entry["passes_up_to"] = max(passing) if passing else None
            first, second = levels[0], levels[-1]
            slope = (second["chamfer_mean"]-first["chamfer_mean"])/(second["level"]-first["level"])
            entry["plateau_from"] = None
            for previous, current in zip(levels, levels[1:]):
                local = (current["chamfer_mean"]-previous["chamfer_mean"])/(current["level"]-previous["level"])
                entry["plateau_from"] = current["level"] if local < .2*max(slope, 1e-9) else None
                if entry["plateau_from"]:
                    break
    (out_root(config)/"external"/"landmarks.json").write_text(json.dumps(table, indent=2)+"\n")


def observed_depth(config):
    master = Path(config["master"])
    scale = json.loads((master/"scale.json").read_text())["mm_per_source_unit"]
    depths = []
    for path in sorted((master/"geometry"/"gt_ray").glob("depth/*.npy")):
        value = np.load(path)
        depths.append(value[value > 0])
    return float(np.median(np.concatenate(depths))*scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/measurement_error_v2.json")
    parser.add_argument("--phase", choices=["prepare", "run", "evaluate", "collect", "all"], required=True)
    parser.add_argument("--position", choices=["tsdf", "m2", "joint"])
    parser.add_argument("--only", help="comma separated case names")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.position:
        config["r1"] = dict(config["r1"], external=dict(config["r1"]["external"], positions=[args.position]))
    only = set(args.only.split(",")) if args.only else None
    if args.phase in ("prepare", "all"):
        prepare(config, only)
    if args.phase in ("run", "all"):
        run(config, only)
    if args.phase in ("evaluate", "all"):
        evaluate(config)
    if args.phase in ("collect", "all"):
        collect(config)
        landmarks(config)


if __name__ == "__main__":
    main()
