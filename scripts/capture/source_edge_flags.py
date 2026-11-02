"""Per-triangle sharp-edge flags of the source scene (thin rods, rims, studs, holes).

A triangle is flagged "sharp" when the largest dihedral angle with an adjacent
triangle exceeds a threshold. The flag array follows the exact triangle ordering
of diagnostics/lego_gt_depth.source_bvh (same depsgraph traversal), which is also
the ordering mathutils.BVHTree.find_nearest indices refer to in
diagnostics/measure_surfaces.py --source-flags.

    blender -b --disable-autoexec --python scripts/capture/source_edge_flags.py --
"""
import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np


def source_triangles():
    """Same traversal as diagnostics.lego_gt_depth.source_bvh: vertices + loop triangles."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices, polygons = [], []
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
        obj.to_mesh_clear()
    return np.asarray(vertices, dtype=np.float64), polygons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/.cache/nerf-syn/lego_source.blend")
    parser.add_argument("--out", default="outputs/lego_source_edge_flags.npy")
    parser.add_argument("--dihedral-deg", type=float, default=25.0)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    source = Path(args.source).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
    started = time.time()
    vertices, polygons = source_triangles()
    normals = np.cross(vertices[[p[1] for p in polygons]] - vertices[[p[0] for p in polygons]],
                       vertices[[p[2] for p in polygons]] - vertices[[p[0] for p in polygons]])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    edge_to_faces = {}
    for face_index, (a, b, c) in enumerate(polygons):
        for edge in ((a, b), (b, c), (c, a)):
            edge_to_faces.setdefault(tuple(sorted(edge)), []).append(face_index)
    max_dihedral = np.zeros(len(polygons))
    for faces in edge_to_faces.values():
        if len(faces) != 2:
            continue
        i, j = faces
        angle = float(np.degrees(np.arccos(np.clip(abs(normals[i] @ normals[j]), 0, 1))))
        max_dihedral[i] = max(max_dihedral[i], angle)
        max_dihedral[j] = max(max_dihedral[j], angle)
    flags = max_dihedral > args.dihedral_deg
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, max_dihedral.astype(np.float32))
    report = {"source": str(source), "triangles": len(polygons),
              "array": "per-triangle max dihedral angle in degrees; threshold applied by the consumer",
              "dihedral_deg_quantiles": {q: float(np.percentile(max_dihedral, float(q)))
                                         for q in (50, 75, 90, 95, 99)},
              "sharp_fraction_at_25deg": float((max_dihedral > 25.0).mean()),
              "sharp_fraction_at_40deg": float((max_dihedral > 40.0).mean()),
              "sharp_fraction_at_60deg": float((max_dihedral > 60.0).mean()),
              "seconds": round(time.time() - started, 1)}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
