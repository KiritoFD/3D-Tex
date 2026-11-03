"""Fixed VGGT initialization plus noisy external-scan anchors, calibrated-rig pilot."""
import argparse
import hashlib
import time
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))


def native_fusion(scene, config):
    import numpy as np
    from geometry_masks import load_masks
    from tsdf_mesh import load_frame, unproject, fuse, extract_surface, write_ply
    omega = scene/"geometry/omega"
    frames = json.loads((omega/"cameras.json").read_text())["frames"]
    masks = load_masks(scene, frames, "external", scene/"masks")
    geometry = [load_frame(f, omega, scene) for f in frames]
    confs, clouds = [], []
    for frame, mask, original in zip(geometry, masks, frames):
        frame["depth"] = np.where(mask, frame["depth"], 0)
        rr, cc = np.nonzero(np.isfinite(frame["depth"]) & (frame["depth"] > 0))
        clouds.append(unproject(frame, rr, cc))
        confs.append(np.load(omega/"confidence"/(Path(original["frame"]).stem+".npy")).squeeze())
    points = np.concatenate(clouds)
    lo, hi = points.min(0), points.max(0)
    voxel = float(np.linalg.norm(hi-lo)/config["tsdf_voxel_divisor"])
    trunc = config["tsdf_truncation_voxels"]*voxel
    lo -= 2*trunc
    hi += 2*trunc
    shape = tuple(np.ceil((hi-lo)/voxel).astype(int))
    far = float(np.median(np.concatenate([g["depth"][g["depth"] > 0] for g in geometry]))*6)
    started = time.perf_counter()
    field, _, _ = fuse(lo, voxel, shape, trunc, geometry, 0, far, confs, "confidence", 64, 1)
    vertices, faces, normals = extract_surface(field, voxel)
    vertices += lo
    out = scene/"mesh"
    out.mkdir(exist_ok=True)
    write_ply(out/"mesh_init.ply", vertices, faces, np.full_like(vertices, .65), normals)
    report = {"scope": "native predicted geometry and annotated masks; GT only downstream evaluation",
        "voxel_native": voxel, "trunc_native": trunc, "shape": list(map(int, shape)),
        "vertices": len(vertices), "faces": len(faces), "seconds": time.perf_counter()-started,
        "camera_source": "predicted", "carving": False, "weight": "confidence clipped 0..8",
        "config": config, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"fusion_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(scene.name, report["vertices"], report["seconds"], flush=True)

def align(scene, gt):
    import numpy as np
    from verify_scene import umeyama
    pred = json.loads((scene/"geometry/omega/cameras.json").read_text())["frames"]
    truth = {f["frame"]: f for f in json.loads((gt/"cameras.json").read_text())["frames"]}
    def centers(records):
        t = np.array([f["T_world_to_cam"] for f in records])
        return -np.einsum("sji,sj->si", t[:, :3, :3], t[:, :3, 3])
    s, r, t = umeyama(centers(pred), centers([truth[f["frame"]] for f in pred]))
    return float(s), r, t

def initialize(config):
    import numpy as np
    from ply_to_obj import read_ply
    from tsdf_mesh import write_ply
    rgb = Path(config["rgb_run"])
    native_fusion(rgb, config)
    resolution = config["observation_resolution"]
    gt = Path(config["master"])/"geometry"/f"gt_ray{resolution}"
    s, r, t = align(rgb, gt)
    vertices, faces, _ = read_ply(rgb/"mesh/mesh_init.ply")
    vertices = s*vertices @ r.T+t
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out/"initial_source.ply", vertices, faces, np.full_like(vertices, .65), np.zeros_like(vertices))
    voxel = json.loads((rgb/"mesh/fusion_report.json").read_text())["voxel_native"]*s
    (out/"initialization.json").write_text(json.dumps({"voxel_source": voxel,
        "scale": s, "rotation": r.tolist(), "translation": t.tolist(),
        "scope": "simulated calibrated rig with ideal registration; not deployment validation",
        "gt_geometry_in_initial_fusion": False, "rgb_run": str(rgb)}, indent=2))
    print("Initialization", voxel, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/measurement_error.json")
    p.add_argument("--phase", choices=["init", "run"], required=True)
    p.add_argument("--only")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.phase == "init":
        initialize(config)
        return
    out = Path(config["out"])
    init = json.loads((out/"initialization.json").read_text())
    entries = [{"name": "no_external", "case": None}] + json.loads((Path(config["root"])/"catalog.json").read_text())
    for entry in entries:
        if args.only and entry["name"] not in args.only.split(","):
            continue
        target = out/"m2"/entry["name"]
        if (target/"m2_log.json").exists():
            saved = json.loads((target/"m2_log.json").read_text())
            expected = dict(config["optimizer"], resolution=config["observation_resolution"])
            expected["lambda_external"] = config["optimizer"]["lambda_external"] if entry["case"] else 0.
            if any(saved["args"][k] != v for k, v in expected.items()):
                raise ValueError(f"Optimizer config differs from completed {target}; choose a fresh out")
            if saved["mesh_sha256"] != hashlib.sha256((out/"initial_source.ply").read_bytes()).hexdigest():
                raise ValueError(f"Initialization changed for {target}")
            if entry["case"]:
                actual = {name: hashlib.sha256((Path(entry["case"])/"depth"/f"{name}.npy").read_bytes()).hexdigest()
                          for name in saved["external_observations"]["depth_sha256"]}
                if actual != saved["external_observations"]["depth_sha256"]:
                    raise ValueError(f"External observations changed for {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "scripts/m2_anchor_opt.py", "--mesh", str(out/"initial_source.ply"),
            "--out", str(target), "--omega", str(Path(config["rgb_run"])/"geometry/omega"),
            "--gt", str(Path(config["master"])/"geometry"/f"gt_ray{config['observation_resolution']}"),
            "--camera-mode", "diagnostic", "--resolution", str(config["observation_resolution"]),
            "--voxel-size", str(init["voxel_source"]), "--iters", str(config["optimizer"]["iters"]),
            "--lr", str(config["optimizer"]["lr"]), "--deformation", config["optimizer"]["deformation"],
            "--lattice-spacing-voxels", str(config["optimizer"]["lattice_spacing_voxels"]),
            "--lambda-prior", str(config["optimizer"]["lambda_prior"]),
            "--tolerance-voxels", str(config["optimizer"]["tolerance_voxels"]),
            "--pixel-center-offset", ".5", "--perspective-projection"]
        if entry["case"]:
            command += ["--external-depth", entry["case"], "--lambda-external", str(config["optimizer"]["lambda_external"])]
        with (target/"run.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("Completed", entry["name"], flush=True)


if __name__ == "__main__":
    main()
