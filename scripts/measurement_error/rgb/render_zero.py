"""Blender: render the 24 frozen GT views from the frozen source mesh (OptiX only).

P0.2 of the acquisition-precision v2 plan. Every view keeps the GT camera, the
source lighting/material/texture of the frozen .blend, resolution, samples and
render seed of the v1 protocol. Geometry is snapshotted and compared against the
immutable source-mesh reference before rendering, so RGB, GT depth and the
evaluation mesh come from one identical mesh.

Requires <scene>/rig.json produced by scripts/measurement_error/zero_scene.py.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def png_sha(path):
    """SHA256 of the decoded 8-bit RGBA pixels, independent of PNG encoder metadata."""
    image = bpy.data.images.load(str(Path(path).resolve()), check_existing=False)
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(image.size[1], image.size[0], 4)
    digest = hashlib.sha256(np.rint(pixels*255).astype(np.uint8).tobytes()).hexdigest()
    bpy.data.images.remove(image)
    return digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/measurement_error_v2.json")
    p.add_argument("--scene", help="defaults to config rgb_run")
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--limit-frames", type=int, default=0)
    p.add_argument("--force", action="store_true")
    # Blender keeps its own flags and the literal "--" in sys.argv; argparse would stop
    # option parsing at that "--" and silently fall back to defaults.
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    args, _ = p.parse_known_args(argv)
    config = json.loads(Path(args.config).read_text())
    spec = config["rgb_sweep"]
    scene_dir = Path(args.scene or config["rgb_run"])
    rig = json.loads((scene_dir/"rig.json").read_text())
    frames = rig["frames"]
    bpy.ops.wm.open_mainfile(filepath=str(Path(config["source"]).resolve()), use_scripts=False)
    render = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    snapshots, vertices, faces, offset = [], [], [], 0
    for ins in dg.object_instances:
        obj = ins.object
        if obj.type != "MESH" or obj.hide_render:
            continue
        mesh = bpy.data.meshes.new_from_object(obj, preserve_all_data_layers=True, depsgraph=dg)
        mesh.calc_loop_triangles()
        matrix = ins.matrix_world.copy()
        xyz = np.array([tuple(matrix @ v.co) for v in mesh.vertices], dtype=np.float32)
        vertices.append(xyz)
        faces.append(np.array([tuple(offset+i for i in f.vertices) for f in mesh.loop_triangles], dtype=np.int32))
        offset += len(xyz)
        snapshots.append((obj.original, mesh, matrix))
    source = np.load(Path(config["master"])/"source_mesh.npz")
    if not np.array_equal(np.concatenate(vertices), source["vertices"]) or not np.array_equal(np.concatenate(faces), source["faces"]):
        raise ValueError("Blender evaluated geometry differs from the frozen source mesh")
    for obj, mesh, matrix in snapshots:
        obj.modifiers.clear()
        obj.constraints.clear()
        obj.parent = None
        obj.data, obj.matrix_world = mesh, matrix
    render.camera.constraints.clear()
    render.camera.parent = None
    render.render.engine = "CYCLES"
    render.cycles.device = "GPU"
    render.cycles.samples = spec["render_samples"]
    render.cycles.seed = spec["render_seed"]
    render.cycles.use_animated_seed = False
    render.cycles.use_adaptive_sampling = False
    render.cycles.use_denoising = False
    render.cycles.use_auto_tile = True
    render.cycles.tile_size = args.tile_size
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "OPTIX"
    devices = [{"name": d.name, "type": d.type, "use": d.use} for d in preferences.devices]
    if not any(d["use"] for d in devices):
        raise RuntimeError("An OptiX GPU is required; CPU fallback is forbidden")
    render.render.resolution_x = render.render.resolution_y = spec["resolution"]
    render.render.resolution_percentage = 100
    render.render.film_transparent = True
    render.render.pixel_aspect_x = render.render.pixel_aspect_y = 1
    render.render.image_settings.file_format = "PNG"
    render.render.image_settings.color_mode = "RGBA"
    render.render.image_settings.color_depth = "8"
    render.camera.data.sensor_fit = "HORIZONTAL"
    render.camera.data.sensor_width = 36
    render.camera.data.shift_x = render.camera.data.shift_y = 0
    render.camera.data.clip_start, render.camera.data.clip_end = .001, 10
    options = {"resolution": spec["resolution"], "samples": spec["render_samples"],
               "seed": spec["render_seed"], "denoising": False, "device": "GPU", "backend": "OPTIX"}
    (scene_dir/"images").mkdir(parents=True, exist_ok=True)
    (scene_dir/"masks").mkdir(exist_ok=True)
    for frame in frames[:args.limit_frames or None]:
        target = scene_dir/"images"/frame["file"]
        if target.exists() and not args.force:
            print("SKIP", frame["file"], flush=True)
            continue
        render.camera.matrix_world = Matrix(frame["c2w_blender"])
        render.camera.data.lens = frame["K_model"][0][0]*36/frame["model_width"]
        render.render.filepath = str(target.resolve())
        started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        seconds = time.perf_counter()-started
        image = bpy.data.images.load(str(target.resolve()), check_existing=False)
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(image.size[1], image.size[0], 4)
        if int((pixels[..., 3] > .5).sum()) < 100:
            raise RuntimeError(f"Empty rendered image: {target}")
        alpha = bpy.data.images.new("mask", width=image.size[0], height=image.size[1], alpha=True)
        mask = np.repeat(pixels[..., 3:4], 4, axis=-1)
        mask[..., 3] = 1
        alpha.pixels[:] = mask.ravel()
        alpha.filepath_raw = str((scene_dir/"masks"/frame["file"]).resolve())
        alpha.file_format = "PNG"
        alpha.save()
        bpy.data.images.remove(alpha)
        bpy.data.images.remove(image)
        rig.setdefault("render_seconds", {})[frame["file"]] = seconds
        rig.setdefault("image_sha256", {})[frame["file"]] = hashlib.sha256(target.read_bytes()).hexdigest()
        rig.setdefault("pixel_sha256", {})[frame["file"]] = png_sha(target)
        (scene_dir/"rig.json").write_text(json.dumps(rig, indent=2))
        print("GPU RGB", frame["file"], round(seconds, 2), flush=True)
    report = {"geometry": "source_bvh viewport-evaluated geometry frozen for RGB",
              "vertices": int(len(np.concatenate(vertices))), "triangles": int(len(np.concatenate(faces))),
              "mesh_sha256": hashlib.sha256((Path(config["master"])/"source_mesh.npz").read_bytes()).hexdigest(),
              "render_options": options, "devices": devices,
              "frames": {frame["file"]: {"seconds": rig.get("render_seconds", {}).get(frame["file"]),
                                         "sha256": rig.get("image_sha256", {}).get(frame["file"]),
                                         "pixel_sha256": rig.get("pixel_sha256", {}).get(frame["file"])}
                         for frame in frames},
              "camera_source": "frozen GT cameras from the master gt_ray export; no jitter",
              "note": "RGB, GT depth and the evaluation mesh all derive from the same frozen source mesh"}
    (scene_dir/"render_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("rendered", len(frames), flush=True)


if __name__ == "__main__":
    main()
