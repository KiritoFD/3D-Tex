"""Blender background preview from the same four recorded reconstruction cameras.

blender -b --python scripts/render_asset.py -- --asset .../mesh.glb --out ...
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cameras", default="data/scenes/nerf_lego16/geometry/omega/cameras.json")
    parser.add_argument("--clay", action="store_true", help="plain lit geometry, without colour textures")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    asset = Path(args.asset).resolve()
    if asset.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(asset), forward_axis="Y", up_axis="Z")
    elif asset.suffix.lower() == ".ply":
        bpy.ops.wm.ply_import(filepath=str(asset))
    else:
        bpy.ops.import_scene.gltf(filepath=str(asset))
        # Undo glTF import's Y-up conversion to recover pipeline world coordinates.
        for obj in list(bpy.context.scene.objects):
            if obj.parent is None:
                obj.matrix_world = Matrix.Rotation(-math.pi / 2, 4, "X") @ obj.matrix_world
    imported = [{"object": o.name, "vertices": len(o.data.vertices),
                 "faces": len(o.data.polygons), "uv_layers": len(o.data.uv_layers)}
                for o in bpy.context.scene.objects if o.type == "MESH"]
    images = [{"image": im.name, "size": list(im.size), "loaded": im.has_data}
              for im in bpy.data.images if im.source == "FILE"]
    (out / "import_report.json").write_text(json.dumps({"asset": str(asset),
                                                      "meshes": imported, "images": images}, indent=2))
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        image = next((n for n in nodes if n.type == "TEX_IMAGE"), None)
        output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if image and output:
            emission = nodes.new("ShaderNodeEmission")
            mat.node_tree.links.new(image.outputs["Color"], emission.inputs["Color"])
            mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if args.clay:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.color_type = "SINGLE"
        scene.display.shading.single_color = (.65, .65, .65)
        scene.display.shading.light = "MATCAP"
        scene.display.shading.studiolight_rotate_z = 0
        scene.display.shading.studio_light = "basic_grey.exr"
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = False
    scene.cycles.samples = 16 if args.clay else 4
    scene.cycles.device = "CPU"
    scene.world.color = (.12, .12, .12)
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    frames = json.loads(Path(args.cameras).read_text())["frames"]
    for index, fi in enumerate((0, 4, 8, 12)):
        frame = frames[fi]
        data = bpy.data.cameras.new(f"camera_{index}")
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        pose = Matrix(frame["T_world_to_cam"]).inverted()
        camera.matrix_world = pose @ Matrix.Diagonal((1, -1, -1, 1))
        k = frame["K_original"]
        width, height = frame["original_width"], frame["original_height"]
        data.type = "PERSP"
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36
        data.lens = k[0][0] * 36 / width
        data.shift_x = -(k[0][2] - width / 2) / width
        data.shift_y = (k[1][2] - height / 2) / width
        data.clip_start = .001
        scene.camera = camera
        scene.render.resolution_x = 512
        scene.render.resolution_y = round(512 * height / width)
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = 1
        scene.render.pixel_aspect_y = k[0][0] / k[1][1]
        scene.render.filepath = str(out / f"view_{index}.png")
        bpy.ops.render.render(write_still=True)
    scene.camera = bpy.data.objects["camera_2"]
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "SOLID" if args.clay else "MATERIAL"
                if args.clay:
                    area.spaces.active.shading.light = "MATCAP"
                    area.spaces.active.shading.studio_light = "basic_grey.exr"
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "preview.blend"))


if __name__ == "__main__":
    main()
