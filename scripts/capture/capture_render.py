"""Controlled capture-scheme renderer: ring trajectories around the source scene.

Runs inside Blender headless:

    blender -b --disable-autoexec --python scripts/capture/capture_render.py -- \
        --out data/scenes/capture_ring16 --views 16 --name ring16

Per frame: RGBA render (Cycles), exact BVH z-depth, alpha foreground, cameras.json
in the repo schema, and a manifest.json recording every capture parameter so each
ablation group is fully reproducible. Rendering matches the dataset conventions
(800x800, 50mm/36mm sensor, transparent background) and GT depth is produced by
the same world-space BVH first-hit method as diagnostics/lego_gt_depth.py.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.lego_gt_depth import source_bvh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/.cache/nerf-syn/lego_source.blend")
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default=None, help="scheme name recorded in the manifest")
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--elev-deg", type=float, default=39.32925296544009,
                        help="ring elevation; default = mean elevation of the original 16 train views")
    parser.add_argument("--radius", type=float, default=4.031129396750165,
                        help="default = original train camera radius")
    parser.add_argument("--azimuth-offset-deg", type=float, default=0.0)
    parser.add_argument("--second-ring-elev-deg", type=float, default=None,
                        help="optional second ring; views split evenly between the rings")
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--depth-resolution", type=int, default=None,
                        help="GT depth/foreground grid; default = --resolution. Use 512 to match "
                             "the model grid (K_model) exactly like the dataset GT depth")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--engine", default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"],
                        help="EEVEE is the default because Cycles GPU device init crashes on this host "
                             "and CPU path tracing is too slow for batch ablations")
    parser.add_argument("--poses-json", default=None,
                        help="optional JSON list of {stem, c2w_blender(4x4)}; overrides ring parameters")
    parser.add_argument("--pass-mode", choices=["both", "render", "depth"], default="both",
                        help="split rendering and BVH depth into separate Blender runs to halve peak memory")
    parser.add_argument("--lens-mm", type=float, default=50.0)
    parser.add_argument("--sensor-mm", type=float, default=36.0)
    parser.add_argument("--seed", type=int, default=0, help="reserved for future jittered schemes")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    return args


def look_at_blender(position, target=(0.0, 0.0, 0.0)):
    """Camera-to-world in Blender convention (-Z forward, +Y up)."""
    direction = Vector(target) - Vector(position)
    rotation = direction.to_track_quat("-Z", "Y")
    return Matrix.Translation(Vector(position)) @ rotation.to_matrix().to_4x4()


def ring_positions(views, elev_deg, radius, offset_deg):
    elev = math.radians(elev_deg)
    return [(radius * math.cos(elev) * math.cos(math.radians(offset_deg + 360.0 * i / views)),
             radius * math.cos(elev) * math.sin(math.radians(offset_deg + 360.0 * i / views)),
             radius * math.sin(elev)) for i in range(views)]


def render_depth(tree, t_world_to_cam, fx, fy, cx, cy, resolution):
    """Exact world-space BVH first-hit camera z-depth (same method as diagnostics/lego_gt_depth.py)."""
    rot = t_world_to_cam[:3, :3]
    origin = Vector(-rot.T @ t_world_to_cam[:3, 3])
    forward = Vector(rot[2])
    depth = np.zeros((resolution, resolution), np.float32)
    for y in range(resolution):
        for x in range(resolution):
            direction = Vector(rot.T @ np.array([(x - cx) / fx, (y - cy) / fy, 1.0]))
            location, _, _, _ = tree.ray_cast(origin, direction.normalized())
            if location is not None:
                depth[y, x] = (location - origin).dot(forward)
    return depth


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    name = args.name or out.name
    (out / "images").mkdir(parents=True, exist_ok=True)
    gt_root = out / "geometry" / "gt"
    (gt_root / "depth").mkdir(parents=True, exist_ok=True)
    (gt_root / "foreground").mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
    tree = None
    objects = []
    if args.pass_mode in ("both", "depth"):
        tree, objects = source_bvh()

    scene = bpy.context.scene
    if args.pass_mode in ("both", "render"):
        scene.render.engine = args.engine
        if args.engine == "CYCLES":
            scene.cycles.samples = args.samples
            scene.cycles.use_denoising = True
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = "CUDA"
                prefs.get_devices()
                for device in prefs.devices:
                    device.use = device.type == "CUDA"
                scene.cycles.device = "GPU"
                device_note = "cycles CUDA gpu"
            except Exception as exc:  # device setup must never abort the batch; record what happened
                scene.cycles.device = "CPU"
                device_note = f"cycles CPU (gpu setup failed: {exc})"
        else:
            if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = args.samples
            device_note = f"eevee ({args.samples} taa samples)"
    else:
        device_note = "depth-only pass (no render engine initialised)"
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0

    fx = args.lens_mm / args.sensor_mm * args.resolution
    fy = fx
    cx = cy = args.resolution / 2.0
    scale = 512 / args.resolution
    k_original = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    k_model = [[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]]
    affine = [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]

    positions = []
    pose_names = None
    if args.poses_json:
        entries = json.loads(Path(args.poses_json).read_text())
        pose_names = [e["stem"] for e in entries]
        positions = [np.asarray(e["c2w_blender"], dtype=np.float64) for e in entries]
    elif args.second_ring_elev_deg is None:
        positions += ring_positions(args.views, args.elev_deg, args.radius, args.azimuth_offset_deg)
    else:
        first = args.views // 2
        positions += ring_positions(first, args.elev_deg, args.radius, args.azimuth_offset_deg)
        positions += ring_positions(args.views - first, args.second_ring_elev_deg, args.radius,
                                    args.azimuth_offset_deg + 360.0 / args.views)

    cam_data = bpy.data.cameras.new("capture_camera")
    cam_data.lens = args.lens_mm
    cam_data.sensor_width = args.sensor_mm
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.001
    camera = bpy.data.objects.new("capture_camera", cam_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    frames = []
    started = time.time()
    for index, position in enumerate(positions):
        if pose_names is not None:
            stem = pose_names[index]
            c2w = Matrix(np.asarray(position, dtype=np.float64).tolist())
            centre = np.asarray(c2w)[:3, 3]
        else:
            stem = f"{name}_{index:03d}"
            c2w = look_at_blender(position)
            centre = position
        camera.matrix_world = c2w
        image_path = out / "images" / f"{stem}.png"
        scene.render.filepath = str(image_path)
        if args.pass_mode in ("both", "render"):
            bpy.ops.render.render(write_still=True)
        # OpenCV world->camera, matching the dataset conversion used by prepare_scene/heldout.
        t_world_to_cam = np.linalg.inv(np.asarray(c2w) @ np.diag([1.0, -1.0, -1.0, 1.0]))
        depth_range = [0.0, 0.0]
        if args.pass_mode in ("both", "depth"):
            dr = args.depth_resolution or args.resolution
            dscale = dr / args.resolution
            depth = render_depth(tree, t_world_to_cam, fx * dscale, fy * dscale,
                                 cx * dscale, cy * dscale, dr)
            rendered = bpy.data.images.load(str(image_path))
            if dr != args.resolution:
                rendered.scale(dr, dr)
            # bpy image pixels are float RGBA rows from the BOTTOM up; flip to numpy top-down order.
            alpha = np.asarray(rendered.pixels[:], dtype=np.float32)[3::4].reshape(dr, dr)[::-1]
            bpy.data.images.remove(rendered)
            foreground = alpha > 0.5
            depth = np.where(foreground, depth, 0).astype(np.float32)
            np.save(gt_root / "depth" / f"{stem}.npy", depth)
            np.save(gt_root / "foreground" / f"{stem}.npy", foreground)
            depth_range = [float(depth[foreground].min()) if foreground.any() else 0.0,
                           float(depth.max())]
        else:
            foreground = np.zeros((args.resolution, args.resolution), bool)
        radius = float(np.linalg.norm(centre))
        azimuth = math.degrees(math.atan2(centre[1], centre[0]))
        elevation = math.degrees(math.asin(centre[2] / max(radius, 1e-9)))
        frames.append({"frame": f"{stem}.png", "file": f"{stem}.png",
                       "original_width": args.resolution, "original_height": args.resolution,
                       "model_width": 512, "model_height": 512,
                       "K_original": k_original, "K_model": k_model,
                       "T_world_to_cam": t_world_to_cam.tolist(),
                       "pixel_affine_original_to_model": affine,
                       "azimuth_deg": round(azimuth, 3), "elevation_deg": round(elevation, 3),
                       "foreground_pixels": int(foreground.sum()),
                       "depth_range": [float(depth[foreground].min()) if foreground.any() else 0.0,
                                       float(depth.max())]})
        print(f"[capture] {stem} az={azimuth:.1f} el={elevation:.1f} "
              f"fg={foreground.sum()} ({time.time() - started:.1f}s)", flush=True)

    manifest = {
        "scheme": name,
        "params": {"views": args.views, "elev_deg": args.elev_deg,
                   "second_ring_elev_deg": args.second_ring_elev_deg, "radius": args.radius,
                   "azimuth_offset_deg": args.azimuth_offset_deg, "resolution": args.resolution,
                   "depth_resolution": args.depth_resolution,
                   "samples": args.samples, "engine": args.engine, "lens_mm": args.lens_mm,
                   "sensor_mm": args.sensor_mm, "seed": args.seed, "poses_json": args.poses_json,
                   "pass_mode": args.pass_mode},
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "blender": bpy.app.version_string, "device": device_note,
        "look_at": "origin (verified on the original 16 train cameras)",
        "depth_method": "world-space BVH first-hit, float32 camera z-depth, masked by render alpha",
        "render_seconds": round(time.time() - started, 1),
        "objects_renderable": len(objects), "frames": frames,
    }
    (gt_root / "cameras.json").write_text(json.dumps(
        {"camera_model": "PINHOLE", "frames": frames}, indent=2) + "\n")
    (gt_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ok] {len(frames)} frames -> {out} ({device_note})", flush=True)


if __name__ == "__main__":
    main()
