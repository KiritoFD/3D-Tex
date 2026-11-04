"""P1.2 intrinsic-calibration error measured on the reprojection residual.

Two experiments are kept apart, as the plan requires:

  wrong-K  (this module, phases prepare/run/evaluate/collect)
      the photos are the frozen zero-noise ones; only the calibration used by
      M2 (render camera + scan look-up) is wrong. This is a *wrong calibration*,
      not a distortion of the physical image.

  lens     (phase lens, implemented below)
      the camera is modelled with a Brown-Conrady radial lens: the frozen
      images are warped, VGGT sees the bent geometry, and M2 either corrects
      with the true coefficient or not at all. `px` is the maximum reprojection
      deviation at the image corner.

The primary x-axis is the reprojection residual in observation pixels, with a
px->mm conversion at the current object distance.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))


def spec(config):
    return config["r1"]["intrinsics"]


def out_root(config):
    return Path(spec(config)["out"])


def resolution(config):
    return int(config["observation_resolution"])


def radius_rms_px(width, height):
    """RMS radius from the principal point under a uniform image density."""
    return float(np.sqrt((width**2+height**2)/12))


def focal_factor_rms_px(relative, width, height):
    return abs(relative)*radius_rms_px(width, height)


def lateral_mm_per_px(config):
    return observed_depth(config)/model_focal(config)


def observed_depth(config):
    master = Path(config["master"])
    scale = json.loads((master/"scale.json").read_text())["mm_per_source_unit"]
    depths = [np.load(p) for p in sorted((master/"geometry"/"gt_ray"/"depth").glob("*.npy"))]
    depths = [d[d > 0] for d in depths]
    return float(np.median(np.concatenate(depths))*scale)


def model_focal(config):
    gt = json.loads((Path(config["master"])/"geometry"/"gt_ray"/"cameras.json").read_text())["frames"]
    return float(np.mean([f["K_model"][0][0] for f in gt]))


def names(config):
    item = spec(config)
    cases = []
    for level in item["focal_relative"]:
        if level == 0:
            continue
        cases += [{"name": f"f{level*100:g}pct_fixed_r{rep}", "family": "focal", "mode": "fixed",
                   "level": float(level), "seed": 42, "rep": rep}
                  for rep in range(1, item["fixed_repeats"]+1)]
        cases += [{"name": f"f{level*100:g}pct_perview_s{seed}", "family": "focal", "mode": "per_view",
                   "level": float(level), "seed": seed, "rep": 1} for seed in item["seeds"]]
    for level in item["principal_px"]:
        if level == 0:
            continue
        cases += [{"name": f"p{level:g}px_fixed_r{rep}", "family": "principal", "mode": "fixed",
                   "level": float(level), "seed": 42, "rep": rep}
                  for rep in range(1, item["fixed_repeats"]+1)]
        cases += [{"name": f"p{level:g}px_perview_s{seed}", "family": "principal", "mode": "per_view",
                   "level": float(level), "seed": seed, "rep": 1} for seed in item["seeds"]]
    return cases


def build_record(config, case):
    from measurement_error.calibration import build, perturb_intrinsics, scalar_field
    base = build(config, 0., 0., "independent", case["seed"], calibration_mode="non_ideal")
    stems = list(base["frames"])
    count = len(stems)
    if case["family"] == "focal":
        field = (scalar_field(count, case["seed"]+7, "independent", case["level"])
                 if case["mode"] == "per_view" else np.full(count, case["level"]))
        shifts = np.zeros((count, 2))
    else:
        field = np.zeros(count)
        if case["mode"] == "per_view":
            rng = np.random.default_rng(case["seed"]+8)
            angle = rng.uniform(0, 2*np.pi, size=count)
            shifts = np.stack([np.cos(angle), np.sin(angle)], 1)*case["level"]
        else:
            shifts = np.stack([np.full(count, case["level"]), np.zeros(count)], 1)
    residuals = np.zeros(count)
    for index, stem in enumerate(stems):
        frame = base["frames"][stem]
        k = np.asarray(frame["ideal_K_model"], dtype=np.float64)
        k = perturb_intrinsics(k, float(field[index]), 0.)
        k[0, 2] += shifts[index, 0]
        k[1, 2] += shifts[index, 1]
        frame["K_model"] = k.tolist()
        frame["focal_relative"] = float(field[index])
        frame["principal_px"] = shifts[index].tolist()
        residuals[index] = np.hypot(focal_factor_rms_px(float(field[index]), resolution(config),
                                                        resolution(config)), float(np.hypot(*shifts[index])))
    base["intrinsics"] = {"focal_relative": case["level"] if case["family"] == "focal" else 0.,
                          "principal_px": case["level"] if case["family"] == "principal" else 0.,
                          "mode": case["mode"], "model_width": int(config["model_resolution"]),
                          "realized_focal_relative_rms": float(np.sqrt(np.mean(field**2))),
                          "realized_principal_px_rms": float(np.sqrt(np.mean((shifts**2).sum(1))),
                                                             )
                          if case["family"] == "principal" else 0.,
                          "reprojection_rms_px": float(np.sqrt(np.mean(residuals**2))),
                          "reprojection_definition": "radial residual on the observation grid",
                          "observation_width": resolution(config),
                          "lateral_mm_per_px": lateral_mm_per_px(config)}
    base["gauge"] = {"registration": "locked", "injected_after_registration": True, "gt_used_for_refit": False,
                     "pose_error": "none; intrinsics only"}
    path = out_root(config)/"calib"/f"{case['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2)+"\n")
    return path, residuals


def prepare(config, only):
    for case in names(config):
        if only and case["name"] not in only:
            continue
        path, residuals = build_record(config, case)
        print("Record", case["name"], round(float(np.sqrt(np.mean(residuals**2))), 3), "px", flush=True)


def m2_command(config, mesh, target, calibration):
    from measurement_error.external import m2_command as base
    return base(config, target.name, mesh, target, calibration)


def run(config, only):
    from measurement_error.external import gpu_free_gib
    for case in names(config):
        if only and case["name"] not in only:
            continue
        target = out_root(config)/"m2"/case["name"]
        if (target/"m2_log.json").exists():
            print("SKIP", case["name"], flush=True)
            continue
        total, free, used = gpu_free_gib()
        if free < spec(config).get("min_free_gib", 2.0):
            raise SystemExit(f"GPU busy before {case['name']}: {free:.2f} GiB free of {total:.2f}")
        mesh = Path(config["out"])/"initial_source.ply"
        target.mkdir(parents=True, exist_ok=True)
        command = m2_command(config, mesh, target, out_root(config)/"calib"/f"{case['name']}.json")
        with (out_root(config)/f"{case['name']}.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        (target/"gpu_before.json").write_text(json.dumps({"total_gib": total, "free_gib": free,
            "used_gib": used}, indent=2)+"\n")
        print("Completed", case["name"], flush=True)


def evaluate(config):
    catalog = [{"name": case["name"], "case": None} for case in names(config)]
    out_root(config).mkdir(parents=True, exist_ok=True)
    (out_root(config)/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
    local = dict(config, out=str(out_root(config)), root=str(out_root(config)))
    (out_root(config)/"config.json").write_text(json.dumps(local, indent=2)+"\n")
    subprocess.run([sys.executable, "scripts/measurement_error/evaluate.py", "--config",
                    str(out_root(config)/"config.json"), "--family", "m2", "--phase", "all"], cwd=ROOT, check=True)


def collect(config):
    from measurement_error.external import case_metrics, zero_metrics
    zero, zero_path = zero_metrics(config)
    rows, groups = [], {}
    for case in names(config):
        folder = out_root(config)/"m2_eval"/case["name"]
        if not (folder/"distances.npz").exists():
            continue
        values = case_metrics(folder)
        record = json.loads((out_root(config)/"calib"/f"{case['name']}.json").read_text())
        log = json.loads((out_root(config)/"m2"/case["name"]/"m2_log.json").read_text())
        rows.append(dict(values, name=case["name"], family=case["family"], mode=case["mode"],
                         level=case["level"], seed=case["seed"], rep=case["rep"],
                         reprojection_rms_px=record["intrinsics"]["reprojection_rms_px"],
                         lateral_mm_per_px=record["intrinsics"]["lateral_mm_per_px"],
                         final_loss=log["iters"][-1]["loss"], seconds=log["seconds"]))
    for family in ("focal", "principal"):
        for mode in spec(config)["modes"]:
            for level in sorted({row["level"] for row in rows
                                 if row["family"] == family and row["mode"] == mode}):
                group = [row for row in rows if row["family"] == family and row["mode"] == mode
                         and row["level"] == level]
                if not group:
                    continue
                record = {"family": family, "mode": mode, "level": level, "n_complete": len(group),
                          "seeds": sorted(row["seed"] for row in group),
                          "reprojection_rms_px": group[0]["reprojection_rms_px"],
                          "reprojection_mm": group[0]["reprojection_rms_px"]*group[0]["lateral_mm_per_px"]}
                for key in ("chamfer_observed_mm", "normal_deg", "observed_fscore@1mm", "recall@1mm",
                            "normal_flat_deg", "normal_edge_deg", "hole_fill", "thin_recall@1mm",
                            "observed_p90_mm", "observed_p99_mm"):
                    values = [row[key] for row in group if row[key] is not None]
                    record[key+"_mean"] = float(np.mean(values))
                    record[key+"_std"] = float(np.std(values))
                    base = zero[key]
                    record["delta_"+key] = record[key+"_mean"]-base["mean"]
                    record["floor_"+key] = max(2*base["std"], base["max"]-base["min"])
                    record["ratio_"+key] = record[key+"_mean"]/base["mean"] if base["mean"] else None
                groups.setdefault(f"{family}/{mode}", []).append(record)
    (out_root(config)/"rows.json").write_text(json.dumps(rows, indent=2)+"\n")
    (out_root(config)/"summary.json").write_text(json.dumps(
        {"zero_control": zero, "zero_control_source": str(zero_path), "groups": groups,
         "px_axis": {"observation_px_to_mm": lateral_mm_per_px(config),
                     "model_px_to_mm": observed_depth(config)/model_focal(config)*
                     config["model_resolution"]/resolution(config)},
         "floor_rule": "b = max(2 sigma, range) of the 10 R0 zero repetitions; 2b is the acceptance floor"},
        indent=2)+"\n")
    print(json.dumps({"rows": len(rows), "groups": {k: len(v) for k, v in groups.items()}}, indent=2), flush=True)


def half_diagonal(size):
    return float(np.sqrt(2)*size/2.)


def k1_for_corner_px(size, px):
    """Brown-Conrady k1 whose corner deviation is exactly `px` pixels."""
    return px/half_diagonal(size)


def radial_scale(r, k1):
    return r*(1.+k1*r*r)


def inverse_scale(rd, k1, iterations=6):
    r = rd.copy()
    for _ in range(iterations):
        slope = 1.+3.*k1*r*r
        r = r-(r*(1.+k1*r*r)-rd)/np.where(np.abs(slope) > 1e-6, slope, 1e-6)
    return np.maximum(r, 0.)


def lens_maps(size, k1, forward):
    yy, xx = np.indices((size, size), dtype=np.float64)
    centre, radius = (size-1)/2., half_diagonal(size)
    xn, yn = (xx-centre)/radius, (yy-centre)/radius
    r = np.hypot(xn, yn)
    scaled = radial_scale(r, k1) if forward else inverse_scale(r, k1)
    ratio = np.where(r > 1e-9, scaled/np.where(r > 1e-9, r, 1.), 1.)
    return (centre+radius*ratio*xn).astype(np.float32), (centre+radius*ratio*yn).astype(np.float32)


def warp_lens(rgba, k1, forward):
    import cv2
    u, v = lens_maps(rgba.shape[0], k1, forward)
    return cv2.remap(rgba, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(255, 255, 255, 0))


def lens_names(config):
    return [f"{mode}{level:g}px" for level in spec(config)["distortion_px"] if level > 0
            for mode in ("raw", "fix")]


def lens_prepare(config, only):
    import shutil
    from PIL import Image
    zero = Path(config["rgb_run"])
    root = out_root(config)/"lens"
    size = int(config["model_resolution"])
    order = sorted(path.name for path in (zero/"images").iterdir() if path.suffix.lower() == ".png")
    catalog, roundtrip = [], {}
    for level in spec(config)["distortion_px"]:
        if level == 0:
            continue
        k1 = k1_for_corner_px(size, level)
        for mode in ("raw", "fix"):
            name = f"{mode}{level:g}px"
            if only and name not in only:
                continue
            scene = root/name
            (scene/"images").mkdir(parents=True, exist_ok=True)
            (scene/"masks").mkdir(exist_ok=True)
            if not (scene/"geometry"/"gt").exists():
                (scene/"geometry").mkdir(parents=True, exist_ok=True)
                shutil.copytree(zero/"geometry"/"gt", scene/"geometry"/"gt")
            shutil.copy(zero/"rig.json", scene/"rig.json")
            hashes = {}
            for filename in order:
                ideal = np.asarray(Image.open(zero/"images"/filename).convert("RGBA"), dtype=np.float32)
                if mode == "raw":
                    warped = warp_lens(ideal, k1, forward=False)
                else:
                    warped = warp_lens(warp_lens(ideal, k1, forward=False), k1, forward=True)
                Image.fromarray(np.rint(warped[..., :3]).clip(0, 255).astype(np.uint8)).save(
                    scene/"images"/filename)
                alpha = warped[..., 3]
                Image.fromarray((alpha > 127.5).astype(np.uint8)*255).save(scene/"masks"/filename)
                hashes[filename] = __import__("hashlib").sha256(
                    (scene/"images"/filename).read_bytes()).hexdigest()
            (scene/"perturbation.json").write_text(json.dumps({
                "mode": "lens_distortion", "target_corner_px": level, "k1": k1,
                "correction": "none" if mode == "raw" else "true_k1",
                "input_order": order, "input_sha256": hashes, "seed": 0,
                "definition": "Brown-Conrady radial warp of the frozen ideal pinhole render; "
                              "px is the maximum reprojection deviation at the image corner"}, indent=2)+"\n")
            catalog.append({"name": name, "scene": scene.as_posix(), "mode": mode,
                            "target_corner_px": level, "k1": k1, "seed": 0})
            print("Lens scene", name, flush=True)
        ideal = np.asarray(Image.open(zero/"images"/order[0]).convert("RGBA"), dtype=np.float32)
        back = warp_lens(warp_lens(ideal, k1, forward=False), k1, forward=True)
        roundtrip[f"{level:g}px"] = {"mean_abs": float(np.abs(back-ideal).mean()),
                                     "max_abs": float(np.abs(back-ideal).max()),
                                     "fraction_over_1_lsb": float((np.abs(back-ideal) > 1).mean())}
    (root/"catalog_scenes.json").write_text(json.dumps(catalog, indent=2)+"\n")
    (root/"roundtrip.json").write_text(json.dumps(roundtrip, indent=2)+"\n")
    config_lens = dict(config, rgb_sweep={"root": str(root), "min_free_commit_gib": 3.0})
    (root/"rgb_config.json").write_text(json.dumps(config_lens, indent=2)+"\n")


def lens_run(config, only):
    from measurement_error.external import gpu_free_gib, m2_command
    root = out_root(config)/"lens"
    catalog = json.loads((root/"catalog_scenes.json").read_text())
    for entry in catalog:
        if only and entry["name"] not in only:
            continue
        scene = Path(entry["scene"])
        for phase in ("infer", "fuse"):
            if phase == "infer" and (scene/"geometry"/"omega"/"cameras.json").exists():
                continue
            subprocess.run([sys.executable, "scripts/measurement_error/rgb/run.py", "--config",
                            str(root/"rgb_config.json"), "--phase", phase, "--only", entry["name"]],
                           cwd=ROOT, check=True)
        total, free, used = gpu_free_gib()
        if free < 2.0:
            raise SystemExit(f"GPU busy before lens {entry['name']}: {free:.2f} GiB free")
        target = root/"m2"/entry["name"]
        if (target/"m2_log.json").exists():
            print("SKIP", entry["name"], flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = m2_command(config, entry["name"], scene/"mesh"/"mesh_source.ply", target, None,
                             omega=scene/"geometry"/"omega")
        with (root/f"{entry['name']}.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("Completed lens", entry["name"], flush=True)


def lens_evaluate(config):
    root = out_root(config)/"lens"
    catalog = [{"name": entry["name"], "case": None}
               for entry in json.loads((root/"catalog_scenes.json").read_text())]
    (root/"catalog_m2.json").write_text(json.dumps(catalog, indent=2)+"\n")
    local = dict(config, out=str(root), root=str(root))
    (root/"eval_config.json").write_text(json.dumps(local, indent=2)+"\n")
    # evaluate.py expects its catalog at <root>/catalog.json
    (root/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
    subprocess.run([sys.executable, "scripts/measurement_error/evaluate.py", "--config",
                    str(root/"eval_config.json"), "--family", "m2", "--phase", "all"], cwd=ROOT, check=True)


def lens_collect(config):
    from measurement_error.external import case_metrics, zero_metrics
    root = out_root(config)/"lens"
    zero, zero_path = zero_metrics(config)
    levels = sorted({entry["target_corner_px"] for entry in json.loads(
        (root/"catalog_m2.json").read_text())})
    rows = []
    for level in levels:
        for mode in ("raw", "fix"):
            folder = root/"m2_eval"/f"{mode}{level:g}px"
            if not (folder/"distances.npz").exists():
                continue
            values = case_metrics(folder)
            rows.append(dict(values, mode=mode, corner_px=level,
                             chamfer_ratio=values["chamfer_observed_mm"]/zero["chamfer_observed_mm"]["mean"],
                             fscore_ratio=values["observed_fscore@1mm"]/zero["observed_fscore@1mm"]["mean"]))
    (root/"summary.json").write_text(json.dumps({"zero_control": zero, "zero_control_source": str(zero_path),
        "roundtrip": json.loads((root/"roundtrip.json").read_text()), "rows": rows,
        "note": "raw = distorted photos treated as pinhole (no correction); fix = undistorted with the true k1"},
        indent=2)+"\n")
    print(json.dumps({"lens_rows": len(rows)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/measurement_error_v2.json")
    parser.add_argument("--phase", required=True,
                        choices=["prepare", "run", "evaluate", "collect", "all",
                                 "lens_prepare", "lens_run", "lens_evaluate", "lens_collect"])
    parser.add_argument("--only")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    only = set(args.only.split(",")) if args.only else None
    if args.phase in ("prepare", "all"):
        prepare(config, only)
    if args.phase in ("run", "all"):
        run(config, only)
    if args.phase in ("evaluate", "all"):
        evaluate(config)
    if args.phase in ("collect", "all"):
        collect(config)
    if args.phase in ("lens_prepare", "all"):
        lens_prepare(config, only)
    if args.phase in ("lens_run", "all"):
        lens_run(config, only)
    if args.phase in ("lens_evaluate", "all"):
        lens_evaluate(config)
    if args.phase in ("lens_collect", "all"):
        lens_collect(config)


if __name__ == "__main__":
    main()
