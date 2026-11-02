#!/usr/bin/env python3
"""Export an ExMesh checkpoint to Blender-ready OBJ + MTL + PNG, and a single-file GLB.

ExMesh's `auto_mesh_iter_N.ply` only carries vertex colours; the UV mapping, the texture and
the per-corner UV indices live in `point_cloud/iteration_N/model_state_dict.pt`. This reads
those tensors and writes:

    export/mesh.obj, export/mesh.mtl, export/albedo.png   (Blender: File > Import > Wavefront)
    export/mesh.glb                                       (single file, geometry+UV+texture)

V-axis: ExMesh samples its atlas with image-space UVs (v down), while OBJ/glTF use a
bottom-origin V, hence `1 - v`. `--flip-v` exists because guessing this wrong is easy and the
only reliable test is opening the result, so flip it if the texture looks upside down.

    python3 scripts/export_exmesh.py --checkpoint <model_state_dict.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="ExMesh checkpoint -> OBJ/MTL/PNG/GLB")
    parser.add_argument("--checkpoint", required=True, help="model_state_dict.pt")
    parser.add_argument("--out", default=None, help="default <checkpoint dir>/export")
    parser.add_argument("--flip-v", action="store_true", default=True,
                        help="convert image-space V to OBJ bottom-origin (default on)")
    parser.add_argument("--no-flip-v", dest="flip_v", action="store_false")
    parser.add_argument("--texture-name", default="albedo.png")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image

    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve() if args.out else ckpt.parent.parent.parent / "export"
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    need = {"vertices", "faces", "uvs", "uv_indices", "texture"}
    missing = need - set(state)
    if missing:
        raise SystemExit(f"checkpoint lacks {sorted(missing)}; keys are {sorted(state)}")

    verts = state["vertices"].numpy().astype(np.float64)
    faces = state["faces"].numpy().astype(np.int64)
    uvs = state["uvs"].numpy().astype(np.float64)
    uv_idx = state["uv_indices"].numpy().astype(np.int64)
    tex = state["texture"].numpy()
    tex = np.clip(np.transpose(tex, (1, 2, 0)), 0, 1)
    texture = (tex * 255).astype(np.uint8)

    if args.flip_v:
        uvs = np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=1)

    Image.fromarray(texture).save(out / args.texture_name)
    if "texture_mask" in state:
        mask = state["texture_mask"].numpy()
        Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).save(out / "texture_mask.png")

    (out / "mesh.mtl").write_text(
        "# ExMesh output material\n"
        "newmtl material_0\n"
        "Ka 0.000 0.000 0.000\n"
        "Kd 1.000 1.000 1.000\n"      # white so the baked albedo is not tinted down
        "Ks 0.000 0.000 0.000\n"
        "d 1.0\nillum 2\n"
        f"map_Kd {args.texture_name}\n")

    with (out / "mesh.obj").open("w") as handle:
        handle.write("# ExMesh geometry + UV from VGGT-Omega/TSDF init\n")
        handle.write("mtllib mesh.mtl\n")
        for x, y, z in verts:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            handle.write(f"vt {u:.6f} {v:.6f}\n")
        handle.write("usemtl material_0\n")
        for face, uv_face in zip(faces, uv_idx):
            corners = []
            for k in range(3):
                corners.append(f"{face[k] + 1}/{uv_face[k] + 1}")
            handle.write("f " + " ".join(corners) + "\n")

    glb_bytes = None
    try:
        import trimesh
        # trimesh only writes TEXCOORD_0 when the mesh's `visual.uv` length matches its
        # VERTEX count. ExMesh's `uvs` is indexed per corner, so use the OBJ-style layout:
        # expand positions to the per-corner array and keep one UV per new vertex.
        from trimesh.visual.material import SimpleMaterial
        corner_uv = uvs[uv_idx.reshape(-1)]
        corner_pos = verts[faces.reshape(-1)]
        remap = np.repeat(np.arange(len(faces)) * 3, 3)
        new_faces = remap.reshape(-1, 3)
        material = SimpleMaterial(image=Image.fromarray(texture))
        visual = trimesh.visual.TextureVisuals(uv=corner_uv, material=material)
        mesh = trimesh.Trimesh(vertices=corner_pos, faces=new_faces, visual=visual,
                               process=False, validate=False)
        mesh.export(str(out / "mesh.glb"))
        glb_bytes = (out / "mesh.glb").stat().st_size
    except Exception as exc:           # GLB is a convenience; OBJ is the contract
        print(f"[warn] GLB skipped: {type(exc).__name__}: {exc}")

    stats = {
        "checkpoint": str(ckpt), "out": str(out),
        "vertices": int(len(verts)), "faces": int(len(faces)),
        "uv_vertices": int(len(uvs)), "texture": list(texture.shape),
        "flip_v": bool(args.flip_v),
        "obj_bytes": (out / "mesh.obj").stat().st_size,
        "mtl_bytes": (out / "mesh.mtl").stat().st_size,
        "texture_bytes": (out / args.texture_name).stat().st_size,
        "glb_bytes": glb_bytes,
        "uv_range": [float(uvs[:, 0].min()), float(uvs[:, 0].max()),
                     float(uvs[:, 1].min()), float(uvs[:, 1].max())],
        "uv_indices_in_range": bool(uv_idx.min() >= 0 and uv_idx.max() < len(uvs)),
    }
    (out / "export_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[out] {out}")
    print(f"[mesh] {stats['vertices']} verts, {stats['faces']} faces, "
          f"{stats['uv_vertices']} uv-verts, texture {texture.shape[1]}x{texture.shape[0]}")
    print(f"[files] mesh.obj {stats['obj_bytes'] / 1024:.0f} KiB, "
          f"albedo {stats['texture_bytes'] / 1024:.0f} KiB, "
          f"glb {0 if not glb_bytes else glb_bytes / 1024:.0f} KiB")
    if args.json:
        (Path(__file__).resolve().parents[1] / "outputs").mkdir(exist_ok=True)
        (Path(__file__).resolve().parents[1] / "outputs" / "exmesh_export.json").write_text(
            json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
