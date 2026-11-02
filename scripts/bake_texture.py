#!/usr/bin/env python3
"""A0 stage 3: clean the TSDF mesh, unwrap UVs with xatlas, bake texture, export OBJ/GLB.

This is the step that produces an asset you can double-click in Blender:
    mesh.obj + mesh.mtl + albedo.png      (Wavefront, opens everywhere)
    mesh.glb                              (single file: geometry + UV + texture)
    mesh.ply                              (vertex-coloured, for quick inspection)

Texture baking is *observed* multiview fusion, not generation: every texel gets the
surface point it belongs to, that point is projected into every training camera, and the
visible contributions are combined (filtered-depth occlusion check + incidence-angle
weighted mean). Texels nobody saw stay in `observed_mask.png` = 0 rather than being
invented.

    python3 scripts/bake_texture.py --scene eth3d_pipes --texels 1024
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import OUTPUTS, scene_path


def load_mesh(path: Path):
    import trimesh
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if len(mesh.faces) == 0:
        raise SystemExit(f"{path} has no faces")
    return mesh


def keep_major_components(mesh, min_share: float):
    """Keep components that together make up min_share of triangles; report what was dropped."""
    import trimesh

    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    count = min(len(order), int(np.searchsorted(np.cumsum(sizes[order]), len(mesh.faces) * min_share)) + 1)
    keep = np.isin(labels, order[:count])
    selected = mesh.faces[keep]
    used, inverse = np.unique(selected, return_inverse=True)
    cleaned = trimesh.Trimesh(vertices=mesh.vertices[used], faces=inverse.reshape(-1,3), process=False)
    return cleaned, {"components_in": len(sizes), "components_kept": count,
                    "faces_in": len(mesh.faces), "faces_kept": len(selected),
                    "share_kept": round(float(keep.mean()), 4)}



def unwrap(mesh):
    """xatlas chart segmentation + parameterisation + packing -> per-corner UVs."""
    import xatlas
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    return np.asarray(vmapping), np.asarray(indices), np.asarray(uvs)


def load_cameras(root: Path) -> list[dict]:
    frames = json.loads((root / "cameras.json").read_text())["frames"]
    out = []
    for frame in frames:
        stem = Path(frame["frame"]).stem
        out.append({
            "name": frame["frame"], "stem": stem, "file": frame["file"],
            "K": np.asarray(frame["K_original"], dtype=np.float64),
            "T": np.asarray(frame["T_world_to_cam"], dtype=np.float64),
            "width": frame["original_width"], "height": frame["original_height"],
        })
    return out


def rasterize_uv_triangles(uv_tri: np.ndarray, texels: int):
    """Return (triangle index, texel row, texel col, barycentric) for every covered texel."""
    owner, rows, cols, barys = [], [], [], []
    scaled = uv_tri * texels - 0.5        # texel centres sit at integer + 0.5
    for t, tri in enumerate(scaled):
        u0, u1 = int(np.floor(tri[:, 0].min())), int(np.ceil(tri[:, 0].max()))
        v0, v1 = int(np.floor(tri[:, 1].min())), int(np.ceil(tri[:, 1].max()))
        u0, v0 = max(u0, 0), max(v0, 0)
        u1, v1 = min(u1, texels - 1), min(v1, texels - 1)
        if u1 < u0 or v1 < v0:
            continue
        gu, gv = np.meshgrid(np.arange(u0, u1 + 1, dtype=np.float64),
                             np.arange(v0, v1 + 1, dtype=np.float64))
        px, py = gu.ravel(), gv.ravel()
        x1, y1 = tri[1] - tri[0]
        x2, y2 = tri[2] - tri[0]
        det = x1 * y2 - x2 * y1
        if abs(det) < 1e-12:
            continue
        rx, ry = px - tri[0, 0], py - tri[0, 1]
        b1 = (rx * y2 - ry * x2) / det
        b2 = (x1 * ry - y1 * rx) / det
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
        if not inside.any():
            continue
        owner.append(np.full(int(inside.sum()), t, dtype=np.int64))
        rows.append(py[inside].astype(np.int64))
        cols.append(px[inside].astype(np.int64))
        barys.append(np.stack([b0[inside], b1[inside], b2[inside]], axis=-1))
    if not owner:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 3))
    return np.concatenate(owner), np.concatenate(rows), np.concatenate(cols), np.concatenate(barys)


def bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = np.clip(np.floor(u).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx, wy = u - x0, v - y0
    return (image[y0, x0] * (1 - wx)[..., None] * (1 - wy)[..., None]
            + image[y0, x1] * wx[..., None] * (1 - wy)[..., None]
            + image[y1, x0] * (1 - wx)[..., None] * wy[..., None]
            + image[y1, x1] * wx[..., None] * wy[..., None])


def main() -> None:
    parser = argparse.ArgumentParser(description="UV unwrap + multiview texture bake -> OBJ/GLB")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mesh", default=None, help="default <scene>/mesh/mesh_init.ply")
    parser.add_argument("--out", default=None, help="default <scene>/asset")
    parser.add_argument("--texels", type=int, default=1024, help="atlas resolution (square)")
    parser.add_argument("--min-component-share", type=float, default=0.90,
                        help="keep components until this share of triangles is covered")
    parser.add_argument("--max-image-pyramid", type=int, default=2048,
                        help="downscale source photos to at most this long side for sampling")
    parser.add_argument("--min-views", type=int, default=1,
                        help="texels seen by fewer cameras are left unobserved")
    parser.add_argument("--angle-penalty", type=float, default=0.25)
    parser.add_argument("--filtered", default=None)
    parser.add_argument("--geometry", default=None, help="camera source; defaults to geometry/omega")
    parser.add_argument("--max-faces", type=int, default=100000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    import trimesh
    from PIL import Image

    scene = scene_path(args.scene)
    mesh_path = Path(args.mesh).resolve() if args.mesh else scene / "mesh" / "mesh_init.ply"
    out = Path(args.out).resolve() if args.out else scene / "asset"
    start = time.time()

    mesh = load_mesh(mesh_path)
    mesh, comp_stats = keep_major_components(mesh, args.min_component_share)
    print(f"[clean] {comp_stats['components_kept']}/{comp_stats['components_in']} components, "
          f"{comp_stats['faces_kept']} faces ({comp_stats['share_kept']:.1%})")

    if len(mesh.faces) > args.max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=args.max_faces)
    vmapping, uv_faces, uv_coords = unwrap(mesh)
    vertices = mesh.vertices[vmapping]
    faces = mesh.faces
    print(f"[uv] xatlas: {len(vertices)} uv-verts, {len(faces)} faces, "
          f"uv range [{uv_coords.min():.3f}, {uv_coords.max():.3f}]")
    uv_face_coords = uv_coords[uv_faces]

    geometry = Path(args.geometry) if args.geometry else scene / "geometry" / "omega"
    cameras = load_cameras(geometry)
    images = []
    for cam in cameras:
        with Image.open(scene / "images" / cam["file"]) as im:
            im = im.convert("RGB")
            scale = args.max_image_pyramid / max(im.size)
            if scale < 1:
                im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.BILINEAR)
            images.append(np.asarray(im, dtype=np.float32) / 255.0)

    owner, rows, cols, barys = rasterize_uv_triangles(uv_face_coords, args.texels)
    print(f"[atlas] {len(owner)} covered texels of {args.texels ** 2} "
          f"({len(owner) / args.texels ** 2:.1%})")

    # surface position + normal per texel
    tri = uv_faces[owner]
    world = (vertices[tri[:, 0]] * barys[:, 0:1] + vertices[tri[:, 1]] * barys[:, 1:2]
             + vertices[tri[:, 2]] * barys[:, 2:3])
    e1 = vertices[tri[:, 1]] - vertices[tri[:, 0]]
    e2 = vertices[tri[:, 2]] - vertices[tri[:, 0]]
    normal = np.cross(e1, e2)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)

    acc = np.zeros((len(owner), 3), dtype=np.float64)
    wsum = np.zeros(len(owner), dtype=np.float64)
    hits = np.zeros(len(owner), dtype=np.int32)
    from consistency_filter import bilinear as sample_depth
    filtered = Path(args.filtered) if args.filtered else scene / "geometry" / "filtered"
    raw_frames = json.loads((geometry / "cameras.json").read_text())["frames"]
    affines = {f["frame"]: np.asarray(f["pixel_affine_original_to_model"]) for f in raw_frames}
    for cam, image in zip(cameras, images):
        rot, trans = cam["T"][:3, :3], cam["T"][:3, 3]
        cam_pos = -rot.T @ trans
        cam_xyz = (rot @ world.T).T + trans
        z = cam_xyz[:, 2]
        view = cam_pos - world
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        facing = np.abs((normal * view).sum(axis=1))
        k = cam["K"]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cam_xyz[:, 0] / z * k[0, 0] + k[0, 2]
            v = cam_xyz[:, 1] / z * k[1, 1] + k[1, 2]
        inside = (z > 1e-6) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
        if not inside.any():
            continue
        scale_x = image.shape[1] / cam["width"]
        scale_y = image.shape[0] / cam["height"]
        weight = (facing ** 2 + 1e-3) * (facing > args.angle_penalty)
        depth = np.load(filtered / "depth" / (cam["stem"] + ".npy"))
        affine = affines[cam["name"]]
        mu, mv = u * affine[0, 0] + affine[0, 2], v * affine[1, 1] + affine[1, 2]
        mu = np.nan_to_num(mu, nan=-1, posinf=-1, neginf=-1)
        mv = np.nan_to_num(mv, nan=-1, posinf=-1, neginf=-1)
        measured = sample_depth(depth, mu, mv)
        visible = (measured > 0) & (np.abs(measured - z) < 0.002 + 0.02 * z)
        weight = weight * inside * visible
        u = np.nan_to_num(u, nan=0, posinf=0, neginf=0)
        v = np.nan_to_num(v, nan=0, posinf=0, neginf=0)
        acc += weight[:, None] * bilinear(image, u * scale_x, v * scale_y)
        wsum += weight
        hits += (weight > 0).astype(np.int32)

    observed = (hits >= args.min_views) & (wsum > 0)
    texture = np.zeros((args.texels, args.texels, 3), dtype=np.float32)
    texture[rows[observed], cols[observed]] = (acc[observed] / wsum[observed, None]).astype(np.float32)
    observed_mask = np.zeros((args.texels, args.texels), dtype=np.uint8)
    observed_mask[rows[observed], cols[observed]] = 255

    out.mkdir(parents=True, exist_ok=True)
    dilate(texture, observed_mask > 0, iterations=4)      # gutter: stops bilinear bleeding

    Image.fromarray((np.clip(texture, 0, 1) * 255).astype(np.uint8)).save(out / "albedo.png")
    Image.fromarray(observed_mask).save(out / "observed_mask.png")

    visual = trimesh.visual.TextureVisuals(uv=np.column_stack([uv_coords[:, 0], 1.0 - uv_coords[:, 1]]), image=Image.fromarray(
        (np.clip(texture, 0, 1) * 255).astype(np.uint8)))
    textured = trimesh.Trimesh(vertices=vertices, faces=uv_faces, visual=visual,
                               process=False, validate=False)
    obj_path = out / "mesh.obj"
    textured.export(str(obj_path))
    # Match the GLB's white albedo multiplier instead of trimesh's grey default.
    (out / "material.mtl").write_text(
        "newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 0\nd 1\nillum 1\nmap_Kd albedo.png\n")
    glb_path = out / "mesh.glb"
    from glb_writer import write_textured_glb, verify_glb
    write_textured_glb(glb_path, vertices, uv_faces,
                       np.column_stack([uv_coords[:, 0], 1.0 - uv_coords[:, 1]]),
                       Image.fromarray((np.clip(texture, 0, 1) * 255).astype(np.uint8)))
    print(verify_glb(glb_path))

    stats = {
        "scene": args.scene, "mesh_in": str(mesh_path), **comp_stats,
        "uv_vertices": int(len(vertices)), "faces": int(len(faces)),
        "texels": args.texels, "atlas_coverage": round(len(owner) / args.texels ** 2, 4),
        "observed_texels": int(observed.sum()),
        "observed_fraction": round(float(observed.sum() / max(len(owner), 1)), 4),
        "cameras_used": len(cameras), "seconds": round(time.time() - start, 1),
        "obj": str(obj_path), "glb": str(glb_path),
        "obj_bytes": obj_path.stat().st_size, "glb_bytes": glb_path.stat().st_size,
    }
    (out / "asset_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[bake] {observed.sum()} texels observed ({stats['observed_fraction']:.1%} of covered)")
    print(f"[out] {obj_path} ({stats['obj_bytes'] / 1024:.0f} KiB), "
          f"{glb_path} ({stats['glb_bytes'] / 1024:.0f} KiB), albedo.png")
    if args.json:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        (OUTPUTS / f"{args.scene}_asset.json").write_text(json.dumps(stats, indent=2) + "\n")


def dilate(texture: np.ndarray, mask: np.ndarray, iterations: int) -> None:
    """Bleed colour outward into the gutter so mipmapping does not pull in black."""
    filled = mask.copy()
    for _ in range(iterations):
        neighbours = np.zeros_like(texture)
        weight = np.zeros_like(filled, dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(np.roll(texture * filled[..., None], dy, axis=0), dx, axis=1)
            shifted_mask = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
            neighbours += shifted
            weight += shifted_mask
        grow = (~filled) & (weight > 0)
        texture[grow] = (neighbours[grow] / weight[grow, None])
        filled |= grow


if __name__ == "__main__":
    main()
