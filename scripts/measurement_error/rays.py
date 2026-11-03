"""Exact first-hit source depth on CUDA BVH; validated against Blender BVH rays."""
import json
from pathlib import Path

import numpy as np
import warp as wp


def camera_records(config, master):
    meta = json.loads(Path(config["train_meta"]).read_text())
    res = config["model_resolution"]
    focal = res/(2*np.tan(meta["camera_angle_x"]/2))
    records = []
    for index in json.loads((master/"protocol.json").read_text())["indices"]:
        frame = meta["frames"][index]
        stem = Path(frame["file_path"]).stem
        k = [[focal, 0, (res-1)/2], [0, focal, (res-1)/2], [0, 0, 1]]
        records.append({"frame": stem+".png", "file": stem+".png", "train_index": index,
            "model_width": res, "model_height": res, "original_width": res, "original_height": res,
            "K_model": k, "K_original": k, "pixel_affine_original_to_model": np.eye(3).tolist(),
            "T_world_to_cam": np.linalg.inv(np.asarray(frame["transform_matrix"]) @ np.diag([1., -1., -1., 1.])).tolist()})
    return sorted(records, key=lambda frame: frame["file"])


@wp.func
def segment_distance2(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    edge = b-a
    t = wp.clamp(wp.dot(p-a, edge)/wp.max(wp.dot(edge, edge), wp.float64(1e-30)), wp.float64(0.0), wp.float64(1.0))
    return wp.dot(p-a-t*edge, p-a-t*edge)


@wp.func
def triangle_distance2(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    ab, ac, q = b-a, c-a, p-a
    aa, bb, cc = wp.dot(ab, ab), wp.dot(ab, ac), wp.dot(ac, ac)
    qa, qc = wp.dot(q, ab), wp.dot(q, ac)
    denominator = aa*cc-bb*bb
    if denominator > wp.float64(1e-30):
        u, v = (cc*qa-bb*qc)/denominator, (aa*qc-bb*qa)/denominator
        if u >= wp.float64(0.0) and v >= wp.float64(0.0) and u+v <= wp.float64(1.0):
            normal = wp.cross(ab, ac)
            return wp.dot(q, normal)*wp.dot(q, normal)/wp.dot(normal, normal)
    return wp.min(segment_distance2(p, a, b), wp.min(segment_distance2(p, b, c), segment_distance2(p, c, a)))


@wp.kernel
def closest(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
            distance: wp.array(dtype=wp.float32), index: wp.array(dtype=wp.int32)):
    i = wp.tid()
    query = wp.mesh_query_point(mesh, points[i], 1000.0)
    if query.result:
        position = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        # The built-in point query loses accuracy on tiny source triangles.
        # Its on-triangle point gives an upper bound. Enumerate all overlapping
        # AABBs within that bound and refine the true triangle distances in f64.
        radius = wp.length(points[i]-position)+1.0e-5
        box = wp.mesh_query_aabb(mesh, points[i]-wp.vec3(radius), points[i]+wp.vec3(radius))
        face = int(0)
        best, best_face = wp.float64(1.0e30), int(-1)
        p = wp.vec3d(points[i])
        while wp.mesh_query_next(box, face):
            a = wp.vec3d(wp.mesh_eval_position(mesh, face, 1.0, 0.0))
            b = wp.vec3d(wp.mesh_eval_position(mesh, face, 0.0, 1.0))
            c = wp.vec3d(wp.mesh_eval_position(mesh, face, 0.0, 0.0))
            value = triangle_distance2(p, a, b, c)
            if value < best:
                best, best_face = value, face
        distance[i] = wp.float32(wp.sqrt(wp.max(best, wp.float64(0.0))))
        index[i] = best_face
    else:
        distance[i] = 1.0e6
        index[i] = -1


@wp.kernel
def cast(mesh: wp.uint64, origin: wp.vec3, rotation: wp.mat33,
         fx: float, fy: float, cx: float, cy: float,
         depth: wp.array2d(dtype=wp.float32)):
    v, u = wp.tid()
    direction = wp.normalize(rotation * wp.vec3((float(u)-cx)/fx, (float(v)-cy)/fy, 1.0))
    hit = wp.mesh_query_ray(mesh, origin, direction, 10.0)
    if hit.result:
        forward = rotation * wp.vec3(0.0, 0.0, 1.0)
        depth[v, u] = hit.t * wp.dot(direction, forward)


def export_gt(config, master, records):
    wp.init()
    device = wp.get_device("cuda:0")
    assert device.is_cuda
    source = np.load(master/"source_mesh.npz")
    points = wp.array(source["vertices"], dtype=wp.vec3, device=device)
    indices = wp.array(source["faces"].ravel(), dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=points, indices=indices)
    res = config["model_resolution"]
    gt = master/"geometry"/("gt_ray" if res == 512 else f"gt_ray{res}")
    for folder in ("depth", "foreground"):
        (gt/folder).mkdir(parents=True, exist_ok=True)
    name = "bvh_checks.json" if res == 512 else f"bvh_checks{res}.json"
    reference = json.loads((master/name).read_text())
    assert reference["pixel_center_offset"] == .5
    checks = reference["rays"]
    differences, mismatches = [], []
    for frame in records:
        k, t = np.asarray(frame["K_model"]), np.asarray(frame["T_world_to_cam"])
        rotation = t[:3, :3].T.astype(np.float32)
        origin = (-rotation @ t[:3, 3]).astype(np.float32)
        depth = wp.zeros((res, res), dtype=wp.float32, device=device)
        wp.launch(cast, dim=(res, res), inputs=[mesh.id, wp.vec3(*origin), wp.mat33(rotation),
            float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])], outputs=[depth], device=device)
        values = depth.numpy()
        for row in checks:
            if row["index"] != frame["train_index"]:
                continue
            z = float(values[row["v"], row["u"]])
            if (z > 0) != (row["z"] > 0):
                mismatches.append(row)
            if z > 0 and row["z"] > 0:
                differences.append(abs(z-row["z"]))
        stem = Path(frame["frame"]).stem
        np.save(gt/"depth"/f"{stem}.npy", values)
        np.save(gt/"foreground"/f"{stem}.npy", values > 0)
    scale = json.loads((master/"scale.json").read_text())["mm_per_source_unit"]
    audit = {"method": "CUDA Warp BVH first-hit ray-triangle intersections; no rasterization",
        "warp": wp.__version__, "gpu": device.name, "bvh_rays": len(checks),
        "shared_hits": len(differences), "support_mismatches": len(mismatches),
        "max_error_mm": max(differences)*scale, "mean_error_mm": float(np.mean(differences))*scale,
        "pixel_convention": "integer-index pixel centers, cx=(W-1)/2", "cpu_rendering": False}
    (gt/"ray_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    if max(differences)*scale > .002 or mismatches:
        raise ValueError(f"Ray reference failed 2 micron validation: {audit}")
    (gt/"cameras.json").write_text(json.dumps({"camera_model": "PINHOLE", "frames": records}, indent=2))
    print(audit, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/measurement_error.json")
    parser.add_argument("--resolution", type=int, help="default exports observation and evaluation grids")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    master = Path(config["master"])
    resolutions = [args.resolution] if args.resolution else sorted({config["observation_resolution"], config["model_resolution"]})
    for res in resolutions:
        local = dict(config, model_resolution=res)
        export_gt(local, master, camera_records(local, master))
