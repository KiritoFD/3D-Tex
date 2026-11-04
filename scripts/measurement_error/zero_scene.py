"""P0.2: one completely consistent zero-error scene (RGB, alpha, GT depth, mesh).

The v1 zero-noise case still used the original NeRF RGB renders, so the images
and the evaluation mesh came from different surfaces. This builder re-renders
the 24 frozen GT views from the frozen source mesh, records SHA256 for every
image, mask, depth map and the mesh, and checks the camera/projection contract
against the exact BVH ground truth.

Phases:
  master  -- copy the immutable source assets into the v2 master and re-export GT depth (512, 256)
  rig     -- write the 24 frozen GT cameras in Blender convention (ideal calibration only)
  audit   -- projection contract, SHA256 manifest, rgb catalog row, perturbation record
  build   -- per-scene GT, VGGT inference, native fusion and the ideal-registration initial mesh
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
ASSETS = ("source_mesh.npz", "stud_vertices.json", "scale.json", "protocol.json",
          "bvh_checks.json", "bvh_checks256.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def master(config):
    source = Path("data/scenes/measurement_precision/master")
    target = Path(config["master"])
    target.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        if not (target/name).exists():
            shutil.copyfile(source/name, target/name)
    for resolution in (config["model_resolution"], config["observation_resolution"]):
        subprocess.run([sys.executable, "scripts/measurement_error/rays.py", "--config", CONFIG_PATH,
                        "--resolution", str(resolution)], cwd=ROOT, check=True)
    manifest = {name: sha(target/name) for name in ASSETS} | {
        f"geometry/{folder}/{path.name}": sha(path) for folder in ("gt_ray", "gt_ray256")
        for path in sorted((target/"geometry"/folder).rglob("*")) if path.is_file()}
    (target/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"master": str(target), "files": len(manifest)}, indent=2), flush=True)


def rig(config):
    meta = json.loads(Path(config["train_meta"]).read_text())
    precision = Path(config["master"])
    model = config["model_resolution"]
    render = config["rgb_sweep"]["resolution"]
    focal = model/(2*np.tan(meta["camera_angle_x"]/2))
    ratio = model/render
    frames = []
    for index in json.loads((precision/"protocol.json").read_text())["indices"]:
        frame = meta["frames"][index]
        k_model = [[focal, 0, (model-1)/2], [0, focal, (model-1)/2], [0, 0, 1]]
        c2w_cv = np.asarray(frame["transform_matrix"]) @ np.diag([1., -1., -1., 1.])
        t_w2c = np.linalg.inv(c2w_cv)
        c2w_blender = np.linalg.inv(t_w2c) @ np.diag([1., -1., -1., 1.])
        k_render = [[focal*render/model, 0, (render-1)/2], [0, focal*render/model, (render-1)/2], [0, 0, 1]]
        frames.append({"file": Path(frame["file_path"]).stem+".png",
                       "frame": Path(frame["file_path"]).stem+".png", "train_index": index,
                       "model_width": model, "model_height": model,
                       "original_width": render, "original_height": render,
                       "K_model": k_model, "K_original": k_render,
                       "pixel_affine_original_to_model": [[ratio, 0, (ratio-1)/2],
                                                          [0, ratio, (ratio-1)/2], [0, 0, 1]],
                       "T_world_to_cam": t_w2c.tolist(), "c2w_blender": c2w_blender.tolist()})
    frames.sort(key=lambda frame: frame["file"])
    scene = Path(config["rgb_run"])
    scene.mkdir(parents=True, exist_ok=True)
    (scene/"rig.json").write_text(json.dumps({"frames": frames, "calibration": "ideal",
        "note": "cameras are the frozen GT rig; no perturbed camera enters this scene"}, indent=2)+"\n")
    print(json.dumps({"scene": str(scene), "frames": len(frames), "render_resolution": render}, indent=2), flush=True)


def audit(config):
    from PIL import Image
    scene = Path(config["rgb_run"])
    precision = Path(config["master"])
    model = config["model_resolution"]
    records = json.loads((scene/"rig.json").read_text())["frames"]
    gt = precision/"geometry"/"gt_ray"
    scale = json.loads((precision/"scale.json").read_text())["mm_per_source_unit"]
    rows = []
    for frame in records:
        stem = Path(frame["file"]).stem
        rgba = np.asarray(Image.open(scene/"images"/frame["file"]).convert("RGBA"), dtype=np.float32)
        alpha = rgba[..., 3]/255.
        truth = np.load(gt/"depth"/f"{stem}.npy")
        foreground = truth > 0
        predicted = alpha > .5
        down = np.asarray(Image.fromarray((predicted*255).astype(np.uint8)).resize((model, model), Image.BOX)) > 127
        union = max(int((down | foreground).sum()), 1)
        k, t = np.asarray(frame["K_model"]), np.asarray(frame["T_world_to_cam"])
        yy, xx = np.nonzero(foreground)
        z = truth[yy, xx]
        camera = np.stack([(xx-k[0, 2])/k[0, 0]*z, (yy-k[1, 2])/k[1, 1]*z, z], -1)
        world = np.linalg.inv(t)[:3, :3] @ camera.T+np.linalg.inv(t)[:3, 3:4]
        back = (t[:3, :3] @ world+t[:3, 3:4])
        u = back[0]/back[2]*k[0, 0]+k[0, 2]
        v = back[1]/back[2]*k[1, 1]+k[1, 2]
        rows.append({"frame": frame["file"], "foreground_pixels": int(foreground.sum()),
            "alpha_pixels_render": int(predicted.sum()), "iou_model_grid": float((down & foreground).sum()/union),
            "disagreement_fraction": float((down ^ foreground).sum()/union),
            "roundtrip_px_max": float(max(np.abs(u-xx).max(), np.abs(v-yy).max())),
            "depth_min_mm": float(z.min()*scale), "depth_max_mm": float(z.max()*scale)})
    summary = {"frames": len(rows),
        "iou_model_grid": {"min": min(r["iou_model_grid"] for r in rows),
                           "mean": float(np.mean([r["iou_model_grid"] for r in rows]))},
        "disagreement_fraction_max": max(r["disagreement_fraction"] for r in rows),
        "roundtrip_px_max": max(r["roundtrip_px_max"] for r in rows),
        "pixel_convention": "integer pixel centres, cx=(W-1)/2, raster adds +0.5 like the v1 protocol"}
    hashes = {"images": {f["file"]: sha(scene/"images"/f["file"]) for f in records},
              "masks": {f["file"]: sha(scene/"masks"/f["file"]) for f in records},
              "gt_depth_model": {f["file"]: sha(gt/"depth"/f"{Path(f['file']).stem}.npy") for f in records},
              "gt_foreground_model": {f["file"]: sha(gt/"foreground"/f"{Path(f['file']).stem}.npy") for f in records},
              "gt_depth_observation": {f["file"]: sha(precision/"geometry"/f"gt_ray{config['observation_resolution']}"
                                                      /"depth"/f"{Path(f['file']).stem}.npy") for f in records},
              "source_mesh": sha(precision/"source_mesh.npz"), "scale": sha(precision/"scale.json"),
              "train_meta": sha(config["train_meta"]), "rig": sha(scene/"rig.json")}
    if (scene/"render_report.json").exists():
        report = json.loads((scene/"render_report.json").read_text())
        hashes["render_report"] = sha(scene/"render_report.json")
        hashes["render_options"] = report["render_options"]
    perturbation = {"mode": "zero", "target_rms_mm": 0., "realized_rms_mm": 0., "seed": 42,
        "input_order": [f["file"] for f in records], "input_sha256": hashes["images"],
        "definition": "no injected error; RGB, alpha and GT depth come from the same frozen source mesh",
        "calibration": "ideal", "config": {k: config[k] for k in ("master", "rgb_run", "model_resolution",
        "observation_resolution", "source")},
        "hashes": hashes, "render_options": hashes.get("render_options", {})}
    (scene/"perturbation.json").write_text(json.dumps(perturbation, indent=2)+"\n")
    (scene/"contract.json").write_text(json.dumps({"summary": summary, "rows": rows, "hashes": hashes}, indent=2)+"\n")
    root = Path(config["rgb_sweep"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    (root/"catalog.json").write_text(json.dumps([{"name": "zero_s42", "scene": scene.as_posix(),
        "mode": "zero", "rms_mm": 0., "seed": 42}], indent=2)+"\n")
    if summary["iou_model_grid"]["min"] < .95 or summary["roundtrip_px_max"] > .5:
        raise ValueError(f"Projection contract failed: {summary}")
    print(json.dumps(summary, indent=2), flush=True)


def build(config):
    """VGGT inference and the ideal-registration initial mesh.

    Per-scene GT depth needs Warp and therefore runs in the WSL venv:
        rgb/prepare.py --phase gt --config <this config>
    """
    for command in ([sys.executable, "scripts/measurement_error/rgb/run.py", "--config", CONFIG_PATH, "--phase", "all"],
                    [sys.executable, "scripts/measurement_error/optimize.py", "--config", CONFIG_PATH, "--phase", "init"]):
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/measurement_error_v2.json")
    parser.add_argument("--phase", choices=["master", "rig", "audit", "build"], required=True)
    args = parser.parse_args()
    CONFIG_PATH = args.config
    config = json.loads(Path(CONFIG_PATH).read_text())
    {"master": master, "rig": rig, "audit": audit, "build": build}[args.phase](config)
