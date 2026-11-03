"""Source assets and physical scale. Extraction uses Blender; no RGB rendering.

--phase extract: blender -b --disable-autoexec --python ... -- --phase extract
--phase calibrate: ordinary Python with NumPy/SciPy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def select_views(meta, count):
    poses = np.asarray([f["transform_matrix"] for f in meta["frames"]])
    direction = poses[:, :3, 3]/np.linalg.norm(poses[:, :3, 3], axis=1, keepdims=True)
    chosen = [0]
    while len(chosen) < count:
        distances = np.min(np.linalg.norm(direction[:, None]-direction[chosen], axis=-1), axis=1)
        distances[chosen] = -1
        chosen.append(int(distances.argmax()))
    return sorted(chosen)


def extract(config):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    master = Path(config["master"])
    master.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(config["train_meta"]).read_text())
    chosen = select_views(meta, config["views"])
    contract = {"indices": chosen, "selection": "farthest unit-camera-direction, start train 0",
        "source_sha256": hashlib.sha256(Path(config["source"]).read_bytes()).hexdigest(),
        "meta_sha256": hashlib.sha256(Path(config["train_meta"]).read_bytes()).hexdigest()}
    if (master/"protocol.json").exists():
        existing = json.loads((master/"protocol.json").read_text())
        for key in ("indices", "source_sha256", "meta_sha256"):
            if existing[key] != contract[key]:
                raise ValueError(f"Source contract changed: {key}; use a fresh master directory")
    else:
        (master/"protocol.json").write_text(json.dumps(contract, indent=2))
    bpy.ops.wm.open_mainfile(filepath=str(Path(config["source"]).resolve()), use_scripts=False)
    vertices, faces, studs = [], [], {}
    offset = 0
    mesh_path = master/"source_mesh.npz"
    reuse = mesh_path.exists() and (master/"stud_vertices.json").exists()
    instances = [] if reuse else bpy.context.evaluated_depsgraph_get().object_instances
    for ins in instances:
        obj = ins.object
        if obj.type != "MESH" or obj.hide_render:
            continue
        mesh = obj.to_mesh()
        mesh.calc_loop_triangles()
        matrix = ins.matrix_world.copy()
        xyz = np.array([tuple(matrix @ v.co) for v in mesh.vertices], dtype=np.float32)
        vertices.append(xyz)
        faces.append(np.array([tuple(offset+i for i in f.vertices) for f in mesh.loop_triangles], dtype=np.int32))
        offset += len(xyz)
        if re.match(r"Brick_flat_(02\*04|04\*06|01\*08)(\.|$)", obj.name):
            local = np.array([tuple(v.co) for v in obj.original.data.vertices])
            top = np.abs(local[:, 2]-local[:, 2].max()) < max(np.ptp(local[:, 2])*1e-5, 1e-6)
            studs[obj.name] = {"top_local": local[top].tolist(), "local_bounds": [local.min(0).tolist(), local.max(0).tolist()],
                               "matrix_world": np.asarray(matrix).tolist()}
        obj.to_mesh_clear()
    if reuse:
        cached = np.load(mesh_path)
        vertices, faces = cached["vertices"], cached["faces"]
        studs = json.loads((master/"stud_vertices.json").read_text())
    else:
        vertices, faces = np.concatenate(vertices), np.concatenate(faces)
        np.savez_compressed(mesh_path, vertices=vertices, faces=faces)
    (master/"stud_vertices.json").write_text(json.dumps(studs))
    tree = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)
    for resolution in sorted({config["model_resolution"], config["observation_resolution"]}):
        rng = np.random.default_rng(20261104)
        focal = resolution/(2*np.tan(meta["camera_angle_x"]/2))
        rays = []
        for index in chosen:
            c2w = np.asarray(meta["frames"][index]["transform_matrix"])
            t = np.linalg.inv(c2w @ np.diag([1., -1., -1., 1.]))
            for u, v in rng.integers(0, resolution, (128, 2)):
                direction = t[:3, :3].T @ np.array([(u+.5-resolution/2)/focal, (v+.5-resolution/2)/focal, 1.])
                hit, _, _, _ = tree.ray_cast(Vector(c2w[:3, 3]), Vector(direction).normalized())
                z = 0. if hit is None else float((np.asarray(hit)-c2w[:3, 3]) @ t[2, :3])
                rays.append({"index": index, "u": int(u), "v": int(v), "z": z if 0 < z < 10 else 0.})
        name = "bvh_checks.json" if resolution == 512 else f"bvh_checks{resolution}.json"
        (master/name).write_text(json.dumps({"pixel_center_offset": .5, "rays": rays}))
    print("Extracted immutable source and sparse BVH checks; no rendering", flush=True)


def calibrate(config):
    from scipy.cluster.hierarchy import fclusterdata
    master = Path(config["master"])
    plates, pitches = [], []
    for name, row in json.loads((master/"stud_vertices.json").read_text()).items():
        nx, ny = map(int, re.search(r"_(\d+)\*(\d+)", name).groups())
        top = np.asarray(row["top_local"])
        pitch_est = max(np.ptp(np.asarray(row["local_bounds"]), axis=0)[:2])/max(nx, ny)
        labels = fclusterdata(top[:, :2], pitch_est*.32, criterion="distance", method="single")
        centers = np.array([(top[labels == i].min(0)+top[labels == i].max(0))/2 for i in np.unique(labels)])
        assert len(centers) == nx*ny, name
        matrix = np.asarray(row["matrix_world"])
        centers = centers @ matrix[:3, :3].T+matrix[:3, 3]
        distances = np.linalg.norm(centers[:, None]-centers[None, :], axis=-1)
        distances[distances < 1e-8] = np.inf
        values = distances.min(1).tolist()
        pitches.extend(values)
        plates.append({"object": name, "stud_count": len(centers), "nearest_pitch": values})
    assert len(plates) >= 6
    pitch = float(np.median(pitches))
    deviation = float(np.max(np.abs(np.asarray(pitches)/pitch-1)))
    assert deviation < .001
    scale = config["stud_pitch_mm"]/pitch
    v = np.load(master/"source_mesh.npz")["vertices"]
    v = v[(np.abs(v) < 5).all(1)]
    result = {"basis": "user-selected 8 mm stud pitch; conditional synthetic scale",
        "source_pitch": pitch, "stud_pitch_mm": config["stud_pitch_mm"], "mm_per_source_unit": scale,
        "max_relative_pitch_deviation": deviation, "plate_count": len(plates), "stud_count": len(pitches),
        "plates": plates, "object_extent_mm": (np.ptp(v, axis=0)*scale).tolist(),
        "source_sha256": json.loads((master/"protocol.json").read_text())["source_sha256"]}
    (master/"scale.json").write_text(json.dumps(result, indent=2)+"\n")
    print({k: v for k, v in result.items() if k != "plates"}, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/measurement_error.json")
    p.add_argument("--phase", choices=["extract", "calibrate"], required=True)
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    args = p.parse_args(argv)
    {"extract": extract, "calibrate": calibrate}[args.phase](json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
