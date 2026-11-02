#!/usr/bin/env python3
"""Convert a vertex-coloured TSDF PLY into OBJ/GLB that Blender opens with visible colour.

The TSDF output has colour per *vertex* and no UV, which is what ExMesh wants. Blender does
import PLY vertex colours, but they only show under "Color Attribute" in Material Preview and
are easy to mistake for "no texture". This exporter bakes vertex colours into a small texture
atlas so both OBJ (with MTL + PNG) and GLB carry an actual image.

    python3 scripts/ply_to_obj.py --ply data/scenes/eth3d_kicker/mesh/mesh_init.ply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal ASCII PLY reader for the specific layout our TSDF writer emits."""
    with path.open() as handle:
        head = []
        for line in handle:
            head.append(line.strip())
            if line.strip() == "end_header":
                break
        n_vert = int([h for h in head if h.startswith("element vertex")][0].split()[-1])
        n_face = int([h for h in head if h.startswith("element face")][0].split()[-1])
        has_colour = any("red" in h for h in head)
        has_normals = any(h.startswith("property float nx") for h in head)
        vertex = np.loadtxt(handle, max_rows=n_vert, ndmin=2)
        face_rows = [line.split() for line in handle if line.strip()]
    faces = np.array([[int(v) for v in row[1:4]] for row in face_rows[:n_face]], dtype=np.int64)

    xyz = vertex[:, 0:3]
    offset = 3
    if has_normals:
        offset += 3
    colours = vertex[:, offset:offset + 3] / 255.0 if has_colour else np.full_like(xyz, 0.6)
    return xyz, faces, colours


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex-coloured PLY -> textured OBJ/GLB")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", default=None, help="default <ply dir>/blender")
    parser.add_argument("--texture", type=int, default=2048)
    parser.add_argument("--min-component-share", type=float, default=0.85,
                        help="keep connected components until this share of faces is covered")
    args = parser.parse_args()

    import trimesh
    from PIL import Image

    src = Path(args.ply).resolve()
    out = Path(args.out).resolve() if args.out else src.parent / "blender"
    out.mkdir(parents=True, exist_ok=True)

    xyz, faces, colours = read_ply(src)
    print(f"[in] {src.name}: {len(xyz)} verts, {len(faces)} faces")

    # Keep the largest components but preserve the ORIGINAL vertex indices: rebuilding a
    # per-triangle lookup collapsed a 418k-face mesh into ~1.4k faces, because every triangle
    # of one big connected component shares the same "component" identity.
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False, validate=False)
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency,
                                                      node_count=len(mesh.faces)) \
        if len(mesh.face_adjacency) else np.zeros(len(mesh.faces), dtype=np.int64)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    total = int(sizes.sum())
    keep_labels, running = [], 0
    for label in order:
        keep_labels.append(int(label))
        running += int(sizes[label])
        if running / total >= args.min_component_share:
            break
    keep_faces = np.isin(labels, keep_labels)
    new_faces = faces[keep_faces]
    used = np.unique(new_faces)
    remap = np.full(len(xyz), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_xyz = xyz[used]
    new_col = colours[used]
    new_faces = remap[new_faces]
    meta = {"components_total": int(len(sizes)), "components_kept": len(keep_labels),
            "faces_kept": int(len(new_faces)), "share_kept": round(len(new_faces) / total, 4)}
    print(f"[clean] kept {len(keep_labels)}/{len(sizes)} components "
          f"({meta['share_kept']:.1%} of faces)")
    print(f"[clean] {len(new_xyz)} verts, {len(new_faces)} faces")

    # xatlas gives per-corner UVs; two levels of charting keeps seams reasonable
    import xatlas
    # xatlas returns (vmapping, indices, uvs):
    #   vmapping : per output *vertex* -> index into the input vertex array
    #   uvs      : per output vertex UV
    #   indices  : faces, indexing the OUTPUT vertex array (same space as uvs)
    # So OBJ needs `f <vmapping[i]+1>/<i+1>` and the texcoord array is indexed by the output vertex.
    vmapping, indices, uvs = xatlas.parametrize(new_xyz, new_faces)
    print(f"[uv] {len(uvs)} uv-verts, {len(indices)} faces, "
          f"uv/vert ratio {len(uvs) / max(len(new_xyz), 1):.2f}")

    # bake vertex colours into the atlas: splat each UV vertex, then dilate
    res = args.texture
    atlas = np.zeros((res, res, 3), dtype=np.float32)
    filled = np.zeros((res, res), dtype=bool)
    col = new_col[vmapping] if len(vmapping) == len(uvs) else new_col[np.clip(vmapping, 0, len(new_col) - 1)]
    px = np.clip((uvs[:, 0] * res).astype(np.int64), 0, res - 1)
    py = np.clip((uvs[:, 1] * res).astype(np.int64), 0, res - 1)
    atlas[py, px] = col
    filled[py, px] = True
    for _ in range(6):
        grow = ~filled
        acc = np.zeros_like(atlas)
        weight = np.zeros((res, res), dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            acc += np.roll(np.roll(atlas * filled[..., None], dy, 0), dx, 1)
            weight += np.roll(np.roll(filled, dy, 0), dx, 1).astype(np.float32)
        take = grow & (weight > 0)
        atlas[take] = acc[take] / weight[take, None]
        filled |= take
    coverage = float(filled.mean())
    print(f"[bake] atlas {res}x{res}, {coverage:.1%} filled after dilation")

    image = Image.fromarray((np.clip(atlas, 0, 1) * 255).astype(np.uint8))
    image.save(out / "albedo.png")

    # OBJ needs bottom-origin V; xatlas emits image-space V
    obj_uv = np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=1)
    (out / "mesh.mtl").write_text(
        # Ka/Kd must stay white: Blender multiplies the image by Kd, so a grey Kd makes the
        # whole atlas look dark/black in the viewport. illum 1 keeps it diffuse-only.
        "newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 0.0\nd 1.0\nillum 1\n"
        "map_Kd albedo.png\n")
    with (out / "mesh.obj").open("w") as handle:
        handle.write("mtllib mesh.mtl\n")
        for x, y, z in new_xyz[vmapping]:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in obj_uv:
            handle.write(f"vt {u:.6f} {v:.6f}\n")
        handle.write("usemtl material_0\n")
        # xatlas `indices` already index into `uvs`, so the same number is valid for both the
        # position slot (after remapping) and the texcoord slot.
        for face in indices:
            handle.write("f " + " ".join(f"{a + 1}/{a + 1}" for a in face) + "\n")

    glb = None
    try:
        import glb_writer
        # Expand to one vertex per face corner: glTF needs len(uv) == len(positions).
        corner_uv = obj_uv[indices.reshape(-1)]
        # `indices` index the output (UV) vertex array; `vmapping` maps those back to positions.
        corner_pos = new_xyz[vmapping][indices.reshape(-1)]
        glb = glb_writer.write_textured_glb(
            out / "mesh.glb", corner_pos,
            np.arange(len(corner_pos)).reshape(-1, 3), corner_uv, image)
        check = glb_writer.verify_glb(out / "mesh.glb")
        print(f"[glb] {check}")
        if "TEXCOORD_0" not in check["attributes"] or check["baseColorFactor"] != [1.0, 1.0, 1.0, 1.0]:
            raise RuntimeError(f"GLB material check failed: {check}")
    except Exception as exc:
        print(f"[warn] GLB skipped: {type(exc).__name__}: {exc}")

    stats = {"source": str(src), **meta, "uv_vertices": int(len(uvs)),
             "vertices": int(len(new_xyz)), "faces": int(len(indices)),
             "texture": res, "atlas_coverage": round(coverage, 4),
             "obj_bytes": (out / "mesh.obj").stat().st_size,
             "glb_bytes": glb, "out": str(out)}
    (out / "convert_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[out] {out}/mesh.obj , mesh.mtl , albedo.png" + (", mesh.glb" if glb else ""))


if __name__ == "__main__":
    main()
