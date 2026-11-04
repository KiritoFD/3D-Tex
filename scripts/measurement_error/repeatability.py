"""Identical-input M2 repetitions: first-divergence localisation and a stable floor.

P0.1 of the acquisition-precision v2 plan. Every repetition is an independent
process with byte-identical mesh, scan, omega export, code and optimizer seed;
only the process instance (and therefore CUDA kernel scheduling) changes.

Phases:
  run      -- execute N repetitions with per-step parameter/gradient hashes
  analyze  -- first divergent step, per-step agreement, vertex and metric distribution
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zero_case(config):
    catalog = json.loads((Path(config["root"])/"catalog.json").read_text())
    for entry in catalog:
        manifest = json.loads((Path(entry["case"])/"manifest.json").read_text())
        if float(manifest["realized_rms_mm"]) == 0.:
            return Path(entry["case"])
    raise ValueError("no zero-noise scan case in the catalog")


def optimizer_command(args, config, scan, mesh, out):
    optimizer = config["optimizer"]
    init = json.loads((Path(args.init).read_text()))
    command = [sys.executable, "scripts/m2_anchor_opt.py", "--mesh", str(mesh), "--out", str(out),
               "--omega", str(Path(config["rgb_run"])/"geometry/omega"),
               "--gt", str(Path(config["master"])/"geometry"/f"gt_ray{config['observation_resolution']}"),
               "--camera-mode", "diagnostic", "--resolution", str(config["observation_resolution"]),
               "--voxel-size", str(init["voxel_source"]), "--iters", str(args.iters or optimizer["iters"]),
               "--lr", str(optimizer["lr"]), "--deformation", optimizer["deformation"],
               "--lattice-spacing-voxels", str(optimizer["lattice_spacing_voxels"]),
               "--lambda-prior", str(optimizer["lambda_prior"]),
               "--tolerance-voxels", str(optimizer["tolerance_voxels"]),
               "--pixel-center-offset", ".5", "--perspective-projection",
               "--external-depth", str(scan), "--lambda-external", str(optimizer["lambda_external"]),
               "--seed", str(optimizer.get("seed", 20261103)), "--print-every", "1"]
    repetition = config.get("repeatability", {})
    command += ["--trace-every", str(repetition.get("trace_every", 1)),
                "--trace-detail-at", str(repetition.get("trace_detail_at", 1))]
    if args.deterministic:
        command.append("--deterministic")
    return command


def run(args):
    config = json.loads(Path(args.config).read_text())
    tag = Path(args.tag_root or Path(config["out"])/"repeatability"/args.tag)
    tag.mkdir(parents=True, exist_ok=True)
    scan = Path(args.scan) if args.scan else zero_case(config)
    mesh = Path(args.mesh)
    omega = Path(config["rgb_run"])/"geometry/omega"
    inputs = {"mesh_sha256": sha_file(mesh), "initialization_sha256": sha_file(args.init),
              "omega_manifest_sha256": sha_file(omega/"manifest.json"),
              "scan_manifest_sha256": sha_file(scan/"manifest.json"),
              "scan_depth_sha256": {p.name: sha_file(p) for p in sorted((scan/"depth").glob("*.npy"))},
              "omega_depth_sha256": {p.name: sha_file(p) for p in sorted((omega/"depth").glob("*.npy"))},
              "code_sha256": {p.name: sha_file(p) for p in sorted((ROOT/"scripts/mesh_opt").glob("*.py"))},
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "deterministic": bool(args.deterministic), "scan": str(scan), "mesh": str(mesh)}
    record = tag/"run.json"
    if record.exists():
        previous = json.loads(record.read_text())
        for key in ("mesh_sha256", "scan_manifest_sha256", "omega_manifest_sha256", "code_sha256"):
            if previous.get(key) != inputs[key]:
                raise ValueError(f"Inputs changed ({key}); use a fresh tag")
    record.write_text(json.dumps(inputs, indent=2)+"\n")
    for index in range(1, args.repeats+1):
        target = tag/"m2"/f"rep{index:02d}"
        if (target/"m2_log.json").exists():
            print("SKIP", target, flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = optimizer_command(args, config, scan, mesh, target)
        command += ["--trace-file", str(target/"trace.jsonl")]
        with (tag/f"rep{index:02d}.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("Completed", target, flush=True)


def traces(tag_root, repeats):
    rows = {}
    for index in range(1, repeats+1):
        path = tag_root/"m2"/f"rep{index:02d}"/"trace.jsonl"
        if not path.exists():
            continue
        rows[index] = {json.loads(line)["iter"]: json.loads(line) for line in path.read_text().splitlines() if line}
    return rows


def analyze(args):
    config = json.loads(Path(args.config).read_text())
    tag = Path(args.tag_root or Path(config["out"])/"repeatability"/args.tag)
    scale = json.loads((Path(config["master"])/"scale.json").read_text())["mm_per_source_unit"]
    recorded = traces(tag, args.repeats)
    if len(recorded) < 2:
        raise ValueError("fewer than two traces; run the repetitions first")
    base = recorded[min(recorded)]
    comparison = {}
    for index, steps in recorded.items():
        if index == min(recorded):
            continue
        first_gradient = first_parameter = first_offset = None
        agreeing = 0
        for step, row in steps.items():
            if step not in base:
                continue
            if first_gradient is None and row["grad_sha256"] != base[step]["grad_sha256"]:
                first_gradient = step
            if first_parameter is None and row["param_sha256"] != base[step]["param_sha256"]:
                first_parameter = step
            if first_offset is None and row["offset_sha256"] != base[step]["offset_sha256"]:
                first_offset = step
            agreeing += int(row["param_sha256"] == base[step]["param_sha256"])
        detail = {}
        for step, row in steps.items():
            if "detail" in row and "detail" in base.get(step, {}):
                mismatched = sorted(key for key, value in row["detail"].items()
                                    if base[step]["detail"].get(key) != value)
                detail[str(step)] = {"keys": len(row["detail"]), "mismatched": mismatched}
        comparison[f"rep{index:02d}"] = {"first_grad_divergence_step": first_gradient,
            "first_parameter_divergence_step": first_parameter,
            "first_offset_divergence_step": first_offset,
            "identical_parameter_steps": agreeing, "compared_steps": len(steps),
            "detail": detail,
            "final_loss": steps[max(steps)]["values"]}
    from ply_to_obj import read_ply
    reference, faces, _ = read_ply(tag/"m2"/f"rep{min(recorded):02d}"/"mesh_m2.ply")
    displacement = {}
    for index in recorded:
        if index == min(recorded):
            continue
        other, other_faces, _ = read_ply(tag/"m2"/f"rep{index:02d}"/"mesh_m2.ply")
        if len(other) != len(reference) or not np.array_equal(other_faces, faces):
            raise ValueError("repetitions changed topology; not comparable")
        distance = np.linalg.norm((other-reference)*scale, axis=1)
        displacement[f"rep{index:02d}"] = {"mean_mm": float(distance.mean()),
            "p50_mm": float(np.quantile(distance, .5)), "p90_mm": float(np.quantile(distance, .9)),
            "p99_mm": float(np.quantile(distance, .99)), "max_mm": float(distance.max())}
    vertices = []
    for index in recorded:
        value, _, _ = read_ply(tag/"m2"/f"rep{index:02d}"/"mesh_m2.ply")
        vertices.append((value-reference)*scale)
    vertices = np.stack(vertices)
    spread = np.linalg.norm(vertices-vertices.mean(0), axis=-1)
    losses = [json.loads((tag/"m2"/f"rep{index:02d}"/"m2_log.json").read_text())["iters"][-1]["loss"]
              for index in recorded]
    payload = {"tag": args.tag, "repeats": len(recorded), "scale_mm_per_source_unit": scale,
        "comparison": comparison, "vertex_displacement_to_first": displacement,
        "vertex_spread_from_mean_mm": {"mean": float(spread.mean()),
            "p50": float(np.quantile(spread, .5)), "p90": float(np.quantile(spread, .9)),
            "p99": float(np.quantile(spread, .99)), "max": float(spread.max())},
        "final_loss": {"values": losses, "mean": float(np.mean(losses)),
            "std": float(np.std(losses)), "min": float(np.min(losses)), "max": float(np.max(losses))},
        "interpretation": "hashes are byte-exact float32; identical parameters mean bit-identical optimisation state"}
    (tag/"repeatability.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({k: payload[k] for k in ("repeats", "vertex_spread_from_mean_mm", "final_loss")}, indent=2), flush=True)


def evaluate(args):
    config = json.loads(Path(args.config).read_text())
    tag = Path(args.tag_root or Path(config["out"])/"repeatability"/args.tag)
    catalog = [{"name": f"rep{index:02d}", "case": None} for index in range(1, args.repeats+1)
               if (tag/"m2"/f"rep{index:02d}"/"mesh_m2.ply").exists()]
    (tag/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
    local = dict(config, out=str(tag), root=str(tag))
    (tag/"config.json").write_text(json.dumps(local, indent=2)+"\n")
    subprocess.run([sys.executable, "scripts/measurement_error/evaluate.py", "--config", str(tag/"config.json"),
                    "--family", "m2", "--phase", "all"], cwd=ROOT, check=True)
    rows = []
    for index in range(1, args.repeats+1):
        folder = tag/"m2_eval"/f"rep{index:02d}"
        if not (folder/"distances.npz").exists():
            continue
        data = np.load(folder/"distances.npz")
        meta = json.loads((folder/"metadata.json").read_text())
        rows.append({"case": f"rep{index:02d}",
            "chamfer_full_mm": float((data["accuracy_mm"].mean()+data["completeness_mm"].mean())/2),
            "chamfer_observed_mm": float((data["accuracy_mm"].mean()+data["observed_mm"].mean())/2),
            "normal_deg": float(data["normal_deg"].mean()),
            "normal_flat_deg": float(data["normal_flat_deg"].mean()) if "normal_flat_deg" in data else None,
            "normal_edge_deg": float(data["normal_edge_deg"].mean()) if "normal_edge_deg" in data else None,
            "hole_fill": meta["hole_fill_rate"],
            "accuracy_p90_mm": float(np.quantile(data["accuracy_mm"], .9)),
            "accuracy_p99_mm": float(np.quantile(data["accuracy_mm"], .99)),
            "observed_p90_mm": float(np.quantile(data["observed_mm"], .9)),
            "observed_p99_mm": float(np.quantile(data["observed_mm"], .99)),
            "precision@1mm": float((data["accuracy_mm"] <= 1.).mean()),
            "recall@1mm": float((data["completeness_mm"] <= 1.).mean()),
            "observed_fscore@1mm": float(2*(data["accuracy_mm"] <= 1.).mean()*(data["observed_mm"] <= 1.).mean()
                /max((data["accuracy_mm"] <= 1.).mean()+(data["observed_mm"] <= 1.).mean(), 1e-12)),
            "thin_recall@1mm": float((data["thin_depth_error_mm"] <= 1.).mean())})
    summary = {}
    if rows:
        keys = [key for key in rows[0] if key != "case" and rows[0][key] is not None]
        for key in keys:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            summary[key] = {"mean": float(values.mean()), "std": float(values.std()),
                            "min": float(values.min()), "max": float(values.max()),
                            "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95))}
    payload = {"tag": args.tag, "rows": rows, "summary": summary,
        "floor": "any difference below these std/range values is not a measurement error effect"}
    (tag/"metrics.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["run", "analyze", "evaluate", "all"], required=True)
    parser.add_argument("--config", default="configs/experiments/measurement_error_v2.json")
    parser.add_argument("--tag", default="zero_v2")
    parser.add_argument("--tag-root", help="explicit output root for a tag outside the v2 output tree")
    parser.add_argument("--scan", help="explicit zero-error scan directory")
    parser.add_argument("--mesh", help="initial mesh for the repetitions")
    parser.add_argument("--init", help="initialization.json matching --mesh")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--iters", type=int, help="override optimizer steps (probe runs only)")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    args.repeats = args.repeats or config.get("repeatability", {}).get("repeats", 10)
    if args.phase in ("run", "all"):
        args.mesh = args.mesh or str(Path(config["out"])/"initial_source.ply")
        args.init = args.init or str(Path(config["out"])/"initialization.json")
        run(args)
    if args.phase in ("analyze", "all"):
        analyze(args)
    if args.phase in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
