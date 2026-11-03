"""Controlled depth-error data and sensor-only TSDF reference.

This reference isolates scan error; it is not a VGGT+anchor optimisation result.
RGB and all 24 views stay fixed. No resolution experiment is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))


def prepare(config):
    from PIL import Image
    master, root = Path(config["master"]), Path(config["root"])
    res = config.get("observation_resolution", 512)
    truth = master/"geometry"/("gt_ray" if res == 512 else f"gt_ray{res}")
    cameras = json.loads((truth/"cameras.json").read_text())
    scale = json.loads((master/"scale.json").read_text())["mm_per_source_unit"]
    frames = cameras["frames"]
    depths = [np.load(truth/"depth"/(Path(f["frame"]).stem+".npy")) for f in frames]
    rgb = Path(config.get("rgb_run", str(root/"rgb24")))
    (rgb/"images").mkdir(parents=True, exist_ok=True)
    (rgb/"masks").mkdir(exist_ok=True)
    for frame in frames:
        shutil.copyfile(Path(config["rgb_source"])/frame["file"], rgb/"images"/frame["file"])
        with Image.open(rgb/"images"/frame["file"]) as image:
            image.getchannel("A").save(rgb/"masks"/frame["file"])
    hashes = {f["file"]: hashlib.sha256((rgb/"images"/f["file"]).read_bytes()).hexdigest() for f in frames}
    (rgb/"capture_manifest.json").write_text(json.dumps({"label": "fixed_rgb24", "seed": 42,
        "input_order": [f["file"] for f in frames], "input_sha256": hashes,
        "note": "original 800px RGB fixed for all depth-error cases; source mesh mismatch to original RGB is shared"}, indent=2))
    catalog = []
    for mode in config["noise_modes"]:
        if mode not in ("iid", "view_bias"):
            raise ValueError(f"Unsupported depth noise mode: {mode}")
        for sigma in config["noise_rms_mm"]:
            if sigma < 0:
                raise ValueError("Noise RMS must be nonnegative")
            for seed in config["seeds"]:
                name = f"{mode}_rms{sigma:g}_s{seed}"
                case = root/"scans"/name
                (case/"depth").mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(seed)
                errors = [rng.normal(size=d.shape).astype(np.float32) if mode == "iid" else
                          np.full(d.shape, rng.normal(), np.float32) for d in depths]
                n = sum(int((d > 0).sum()) for d in depths)
                norm = np.sqrt(sum(float((e[d > 0].astype(np.float64)**2).sum())
                                   for e, d in zip(errors, depths))/n)
                squares = total = 0.
                for frame, depth, error in zip(frames, depths, errors):
                    noisy = np.where(depth > 0, depth+error*(sigma/scale/norm), 0).astype(np.float32)
                    realized = (noisy[depth > 0].astype(np.float64)-depth[depth > 0])*scale
                    squares += float((realized**2).sum())
                    total += float(realized.sum())
                    np.save(case/"depth"/(Path(frame["frame"]).stem+".npy"), noisy)
                (case/"cameras.json").write_text(json.dumps(cameras, indent=2))
                manifest = {"name": name, "noise_mode": mode, "seed": seed, "target_rms_mm": sigma,
                    "realized_rms_mm": float(np.sqrt(squares/n)), "mean_error_mm": total/n,
                    "definition": "camera-z depth error; iid pixel Gaussian or per-view constant Gaussian bias; globally normalized to target RMS",
                    "mm_per_source_unit": scale, "source_sha256": json.loads((master/"scale.json").read_text())["source_sha256"],
                    "gt_ray_audit": json.loads((truth/"ray_audit.json").read_text()),
                    "input_order": [f["file"] for f in frames], "config": config}
                (case/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                catalog.append({"name": name, "case": case.as_posix()})
    (root/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
    print(f"Prepared {len(catalog)} scan cases; 24 fixed RGB views", flush=True)


def fuse_case(config, entry):
    from tsdf_mesh import load_frame, fuse, extract_surface, write_ply
    case = Path(entry["case"])
    out = Path(config["out"])/"scan_tsdf"/entry["name"]
    if (out/"fusion_report.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    frames = json.loads((case/"cameras.json").read_text())["frames"]
    frames = [load_frame(f, case, case) for f in frames]
    master = Path(config["master"])
    scale = json.loads((master/"scale.json").read_text())["mm_per_source_unit"]
    source = np.load(master/"source_mesh.npz")["vertices"]
    source = source[(np.abs(source) < 5).all(1)]
    voxel = config["voxel_mm"]/scale
    trunc = config["trunc_voxels"]*voxel
    # Fixed grid from the clean source, never from noisy depths.
    origin, hi = source.min(0)-2*trunc, source.max(0)+2*trunc
    shape = tuple(np.ceil((hi-origin)/voxel).astype(int))
    start = time.perf_counter()
    field, _, _ = fuse(origin, voxel, shape, trunc, frames, 0, 10,
                       [None]*len(frames), "uniform", 64, 1)
    vertices, faces, normals = extract_surface(field, voxel)
    vertices += origin
    write_ply(out/"mesh.ply", vertices, faces, np.full_like(vertices, .65), normals)
    result = {"scope": "sensor-only reference, not VGGT+anchors", "case": str(case),
        "voxel_mm": config["voxel_mm"], "trunc_voxels": config["trunc_voxels"],
        "origin": origin.tolist(), "shape": list(map(int, shape)),
        "seconds": time.perf_counter()-start, "vertices": len(vertices), "faces": len(faces)}
    (out/"fusion_report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(entry["name"], result["seconds"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/measurement_error.json")
    p.add_argument("--phase", choices=["prepare", "fuse"], required=True)
    p.add_argument("--only", help="comma separated case names")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.phase == "prepare":
        prepare(config)
    else:
        entries = json.loads((Path(config["root"])/"catalog.json").read_text())
        for entry in entries:
            if args.only and entry["name"] not in args.only.split(","):
                continue
            fuse_case(config, entry)


if __name__ == "__main__":
    main()
