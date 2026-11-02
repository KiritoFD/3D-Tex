"""Shared GLB writing so both exporters emit identical, Blender-safe material semantics.

trimesh's SimpleMaterial defaults are wrong for a baked albedo atlas:
  * baseColorFactor defaults to a mid grey (0.4), which glTF multiplies with the texture and
    makes every surface look 60% darker - the "texture looks black/dark" symptom in Blender;
  * doubleSided is left false, so back faces (unavoidable in a single-view scan) render black.

This module writes the glTF with baseColorFactor = white and doubleSided = true, and
post-processes trimesh's output to force those two values.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def write_textured_glb(path: Path, positions: np.ndarray, faces: np.ndarray,
                       uvs: np.ndarray, image, name: str = "mesh") -> int:
    """positions/uvs are per-vertex (already expanded), faces index both arrays."""
    import trimesh
    from trimesh.visual.material import SimpleMaterial

    material = SimpleMaterial(image=image, baseColorFactor=[255, 255, 255, 255])
    material.doubleSided = True
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=positions, faces=faces, visual=visual,
                           process=False, validate=False)
    mesh.export(str(path))
    patch_gltf_materials(path)
    return path.stat().st_size


def patch_gltf_materials(path: Path) -> None:
    """Force white baseColorFactor + doubleSided on every material in a .glb."""
    data = bytearray(path.read_bytes())
    # header(12) + first chunk header(8)
    json_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(bytes(data[20:20 + json_len]).decode("utf-8"))
    for material in doc.get("materials", []):
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        material["doubleSided"] = True
        # a baked albedo is unlit-ish: keep metallic off so it is not tinted
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 1.0
    encoded = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)          # chunks must be 4-byte aligned
    rest = bytes(data[20 + json_len:])                    # the BIN chunk, untouched
    total = 12 + 8 + len(encoded) + len(rest)
    out = bytearray(b"glTF" + struct.pack("<I", 2) + struct.pack("<I", total))
    out += struct.pack("<I", len(encoded)) + b"JSON" + encoded
    out += rest
    path.write_bytes(out)


def verify_glb(path: Path) -> dict:
    """Read back the material block so a regression is caught without opening Blender."""
    data = path.read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + json_len].decode("utf-8"))
    prim = doc["meshes"][0]["primitives"][0]
    material = doc["materials"][prim["material"]]
    pbr = material.get("pbrMetallicRoughness", {})
    return {
        "attributes": sorted(prim["attributes"]),
        "has_texture": "baseColorTexture" in pbr,
        "baseColorFactor": pbr.get("baseColorFactor"),
        "double_sided": material.get("doubleSided", False),
        "images": len(doc.get("images", [])),
    }
