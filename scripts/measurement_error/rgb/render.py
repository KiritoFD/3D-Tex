"""Blender: camera-jitter RGB on OptiX GPU only, fixed source geometry/materials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/measurement_error.json")
    p.add_argument("--only")
    p.add_argument("--limit-frames", type=int, default=0, help="component probe only; inference requires all 24")
    args = p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    config = json.loads(Path(args.config).read_text())
    spec = config["rgb_sweep"]
    bpy.ops.wm.open_mainfile(filepath=str(Path(config["source"]).resolve()), use_scripts=False)
    scene = bpy.context.scene
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
        raise ValueError("GPU RGB evaluated geometry differs from source mesh reference")
    for obj, mesh, matrix in snapshots:
        obj.modifiers.clear()
        obj.constraints.clear()
        obj.parent = None
        obj.data, obj.matrix_world = mesh, matrix
    scene.camera.constraints.clear()
    scene.camera.parent = None
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = spec["render_samples"]
    scene.cycles.seed = spec["render_seed"]
    scene.cycles.use_animated_seed = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "OPTIX"
    devices = [{"name": d.name, "type": d.type, "use": d.use} for d in preferences.devices]
    if not any(d["use"] for d in devices):
        raise RuntimeError("An OptiX GPU is required; CPU fallback is forbidden")
    scene.render.resolution_x = scene.render.resolution_y = spec["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.camera.data.sensor_fit = "HORIZONTAL"
    scene.camera.data.sensor_width = 36
    scene.camera.data.shift_x = scene.camera.data.shift_y = 0
    scene.camera.data.clip_start, scene.camera.data.clip_end = .001, 10
    options = {"resolution": spec["resolution"], "samples": spec["render_samples"],
               "seed": spec["render_seed"], "denoising": False, "device": "GPU", "backend": "OPTIX"}
    only = set(args.only.split(",")) if args.only else None
    rows = json.loads((Path(spec["root"])/"catalog.json").read_text())
    for row in rows:
        if row["mode"] != "camera_jitter" or (only and row["name"] not in only):
            continue
        folder = Path(row["scene"])
        if not (folder/"rig.json").exists():
            continue
        manifest = json.loads((folder/"perturbation.json").read_text())
        if manifest.get("render_options", options) != options:
            raise ValueError("Rendering parameters changed; choose a fresh rgb_sweep.root")
        records = json.loads((folder/"rig.json").read_text())["frames"]
        manifest.update(render_options=options, devices=devices)
        times = manifest.setdefault("render_seconds", {})
        for frame in records[:args.limit_frames or None]:
            target = folder/"images"/frame["file"]
            if not target.exists():
                scene.camera.matrix_world = Matrix(frame["c2w_blender"])
                scene.camera.data.lens = frame["K_model"][0][0]*36/frame["model_width"]
                scene.render.filepath = str(target.resolve())
                started = time.perf_counter()
                bpy.ops.render.render(write_still=True)
                times[frame["file"]] = time.perf_counter()-started
            image = bpy.data.images.load(str(target.resolve()))
            pixels = np.array(image.pixels[:], dtype=np.float32).reshape(image.size[1], image.size[0], 4)
            if int((pixels[..., 3] > .5).sum()) < 100:
                raise RuntimeError(f"Empty rendered image: {target}")
            # Save the exact rendered alpha as an explicit input annotation.
            alpha = bpy.data.images.new("input_annotation", width=image.size[0], height=image.size[1], alpha=True)
            mask = np.repeat(pixels[..., 3:4], 4, axis=-1)
            mask[..., 3] = 1
            alpha.pixels[:] = mask.ravel()
            alpha.filepath_raw = str((folder/"masks"/frame["file"]).resolve())
            alpha.file_format = "PNG"
            alpha.save()
            bpy.data.images.remove(alpha)
            bpy.data.images.remove(image)
            manifest["input_sha256"][frame["file"]] = hashlib.sha256(target.read_bytes()).hexdigest()
            (folder/"perturbation.json").write_text(json.dumps(manifest, indent=2))
            print("GPU RGB", row["name"], frame["file"], flush=True)


if __name__ == "__main__":
    main()
