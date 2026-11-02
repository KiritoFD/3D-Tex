"""Prepare Omega-seeded PGSR input, or export its optimized floating-point depth."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare(args):
    from exmesh_adapter import build_colmap
    from tsdf_mesh import load_frame, unproject

    scene, work = Path(args.scene).resolve(), Path(args.workdir).resolve()
    geometry = scene / "geometry/omega"
    filtered = scene / "geometry/filtered_fixed"
    frames = json.loads((geometry / "cameras.json").read_text())["frames"]
    (work / "images").mkdir(parents=True, exist_ok=True)
    points, colours = [], []
    for frame in frames:
        shutil.copy2(scene / "images" / frame["file"], work / "images" / frame["file"])
        frame["_K"] = np.asarray(frame["K_original"])
        frame["_T"] = np.asarray(frame["T_world_to_cam"])
        loaded = load_frame(frame, filtered, scene)
        rr, cc = np.mgrid[0:loaded["depth"].shape[0]:4, 0:loaded["depth"].shape[1]:4]
        rr, cc = rr.ravel(), cc.ravel()
        valid = loaded["depth"][rr, cc] > 0
        rr, cc = rr[valid], cc[valid]
        points.append(unproject(loaded, rr, cc))
        colours.append(loaded["rgb"][rr, cc])
    build_colmap(work / "sparse", frames, frames[0]["original_width"], frames[0]["original_height"])
    xyz, rgb = np.concatenate(points), np.concatenate(colours)
    selected = np.random.default_rng(22).choice(len(xyz), min(30000, len(xyz)), replace=False)
    with (work / "sparse/points3D.txt").open("w") as handle:
        handle.write("# Omega foreground seed points, not GT geometry\n")
        for i, (point, colour) in enumerate(zip(xyz[selected], rgb[selected]), 1):
            handle.write(f"{i} {' '.join(f'{v:.9f}' for v in point)} {' '.join(str(int(v)) for v in colour)} 0\n")
    report = {"scene": str(scene), "frames": [f["frame"] for f in frames], "points": len(selected),
              "camera_and_seed": "raw Omega cameras + filtered Omega depth; no source GT geometry",
              "mask": "dataset alpha, same foreground input as prior reconstruction", "gt_geometry_used": False}
    (work / "input_report.json").write_text(json.dumps(report, indent=2))
    print(report)


def export(args):
    import torch
    import torchvision
    sys.path.insert(0, str(Path(args.pgsr_root).resolve()))
    from argparse import Namespace
    from scene import Scene, GaussianModel
    from scene.app_model import AppModel
    from gaussian_renderer import render

    torch.set_num_threads(4)
    work, model = Path(args.workdir).resolve(), Path(args.model).resolve()
    config = eval((model / "cfg_args").read_text(), {"Namespace": Namespace})
    config.source_path, config.model_path = str(work), str(model)
    config.resolution = 512  # geometry export matches the Omega depth grid
    gaussians = GaussianModel(config.sh_degree)
    scene = Scene(config, gaussians, load_iteration=args.iteration, shuffle=False)
    app = AppModel(); app.load_weights(str(model), args.iteration)
    background = torch.zeros(3, device="cuda")
    pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    original = json.loads((Path(args.scene) / "geometry/omega/cameras.json").read_text())["frames"]
    by_name = {Path(f["frame"]).stem: f for f in original}
    out = Path(args.out).resolve()
    for directory in (out / "depth", out / "renders", out / "confidence"):
        directory.mkdir(parents=True, exist_ok=True)
    exported = {}
    with torch.inference_mode():
        for camera in scene.getTrainCameras():
            result = render(camera, gaussians, pipeline, background, app_model=app, return_plane=True, return_depth_normal=True)
            depth = result["plane_depth"].squeeze().cpu().numpy()
            f = by_name[camera.image_name]
            if depth.shape != (f["model_height"], f["model_width"]):
                raise ValueError(f"Unexpected export depth shape {depth.shape}")
            # Same explicit foreground as Omega; additionally reject unobserved raster pixels.
            mask = np.load(Path(args.scene) / "geometry/filtered_fixed/foreground" / f"{camera.image_name}.npy")
            depth = np.where(mask & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
            np.save(out / "depth" / f"{camera.image_name}.npy", depth)
            np.save(out / "confidence" / f"{camera.image_name}.npy",
                    result["rendered_alpha"].squeeze().cpu().numpy().astype(np.float32))
            torchvision.utils.save_image(result["render"].clamp(0,1), str(out / "renders" / f"{camera.image_name}.png"))
            exported[f["frame"]] = f
            print(f"exported {camera.image_name}", flush=True)
    (out / "cameras.json").write_text(json.dumps({"camera_model": "PINHOLE", "frames": [exported[f["frame"]] for f in original]}, indent=2))
    (out / "manifest.json").write_text(json.dumps({"backend": "PGSR", "iteration": args.iteration,
                                                 "depth": "optimized plane_depth (camera z), foreground masked",
                                                 "confidence": "rasterized alpha coverage, not calibrated depth reliability",
                                                 "gt_geometry_used": False}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "export", "patch"])
    parser.add_argument("--scene", default="data/scenes/nerf_lego16")
    parser.add_argument("--workdir", default="data/scenes/nerf_lego16/pgsr_workdir")
    parser.add_argument("--pgsr-root", default="/root/src/exmesh/PGSR")
    parser.add_argument("--model", default="data/scenes/nerf_lego16/pgsr_workdir/out")
    parser.add_argument("--out", default="data/scenes/nerf_lego16/geometry/pgsr")
    parser.add_argument("--iteration", type=int, default=10000)
    args = parser.parse_args()
    if args.action == "patch":
        path = Path(args.pgsr_root) / "scene/gaussian_model.py"
        source = path.read_text()
        needle = "from pytorch3d.transforms import quaternion_to_matrix"
        if needle in source:
            path.with_suffix(".py.before_3dtex").write_text(source)
            path.write_text(source.replace(needle, "from utils.general_utils import build_rotation as quaternion_to_matrix"))
        renderer = Path(args.pgsr_root) / "gaussian_renderer/__init__.py"
        source = renderer.read_text()
        if '"rendered_alpha": rendered_alpha' not in source:
            renderer.with_suffix(".py.before_3dtex").write_text(source)
            renderer.write_text(source.replace('"rendered_normal": rendered_normal,',
                                               '"rendered_normal": rendered_normal,\n                    "rendered_alpha": rendered_alpha,'))
        print("PGSR uses its existing normalized quaternion rotation helper")
    else:
        (prepare if args.action == "prepare" else export)(args)


if __name__ == "__main__":
    main()
