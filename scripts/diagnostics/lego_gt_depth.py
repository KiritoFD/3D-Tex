"""Blender: exact floating-point z-depth from the public lego source scene.

Run with --disable-autoexec. No embedded source-scene scripts are executed.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    source, scene = Path(args.source).resolve(), Path(args.scene).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices, polygons, objects = [], [], []
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type != "MESH" or obj.hide_render:
            continue
        mesh = obj.to_mesh()
        mesh.calc_loop_triangles()
        offset = len(vertices)
        matrix = instance.matrix_world
        vertices.extend(tuple(matrix @ v.co) for v in mesh.vertices)
        polygons.extend(tuple(offset + i for i in t.vertices) for t in mesh.loop_triangles)
        objects.append({"name": obj.name, "vertices": len(mesh.vertices), "triangles": len(mesh.loop_triangles)})
        obj.to_mesh_clear()
    xyz = np.asarray(vertices)
    print(json.dumps({"source_objects": len(objects), "vertices": len(vertices), "triangles": len(polygons),
                      "bbox": [xyz.min(0).tolist(), xyz.max(0).tolist()] }), flush=True)
    tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)
    del vertices, polygons
    root = scene / "geometry" / "gt"
    frames = json.loads((root / "cameras.json").read_text())["frames"]
    (root / "depth").mkdir(exist_ok=True)
    reports = []
    started = time.time()
    for frame in frames:
        width, height = frame["model_width"], frame["model_height"]
        k, t = np.asarray(frame["K_model"]), np.asarray(frame["T_world_to_cam"])
        rot = t[:3, :3]
        origin = Vector(-rot.T @ t[:3, 3])
        forward = Vector(rot[2])
        depth = np.zeros((height, width), np.float32)
        for y in range(height):
            for x in range(width):
                direction = Vector(rot.T @ np.array([(x-k[0,2])/k[0,0], (y-k[1,2])/k[1,1], 1.]))
                location, _, _, _ = tree.ray_cast(origin, direction.normalized())
                if location is not None:
                    depth[y, x] = (location - origin).dot(forward)
        stem = Path(frame["frame"]).stem
        mask = np.load(root / "foreground" / f"{stem}.npy")
        valid = depth > 0
        iou = float((mask & valid).sum() / max((mask | valid).sum(), 1))
        # Exclude any source-scene props outside the dataset's object foreground.
        depth = np.where(mask, depth, 0).astype(np.float32)
        valid = depth > 0
        np.save(root / "depth" / f"{stem}.npy", depth)
        reports.append({"frame": frame["frame"], "silhouette_iou_to_dataset": iou,
                        "depth_range": [float(depth[valid].min()), float(depth.max())]})
        print(f"[GT] {stem} silhouette IoU={iou:.4f}, {time.time()-started:.1f}s", flush=True)
    report = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "method": "world-space BVH first-hit intersection, float32 camera z-depth; no neural predictions or PNG decoding",
              "objects": objects, "frames": reports, "seconds": time.time()-started}
    (root / "source_report.json").write_text(json.dumps(report, indent=2))
    if min(r["silhouette_iou_to_dataset"] for r in reports) < .95:
        raise RuntimeError("Source scene does not sufficiently match dataset silhouettes; inspect alignment before fusion")


if __name__ == "__main__":
    main()
