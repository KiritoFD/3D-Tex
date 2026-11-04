#!/usr/bin/env python3
"""Freeze one ETH3D P0 grid from the B0 mesh report (no GT inputs)."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256_16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def dir_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    h.update((root / "cameras.json").read_bytes())
    for path in sorted((root / "depth").glob("*.npy")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh-report", required=True)
    p.add_argument("--geometry", required=True, help="B0 omega geometry dir used as ROI")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = json.loads(Path(args.mesh_report).read_text())
    geom = Path(args.geometry)
    grid = {
        "scene": args.scene,
        "source": "B0 raw VGGT depth only",
        "roi_geometry": str(geom.resolve()),
        "roi_geometry_sha256_16": dir_fingerprint(geom),
        "voxel": rep["voxel"],
        "truncation": rep["sdf_trunc"],
        "origin": rep["origin"],
        "shape": rep["shape"],
        "frames": rep["frames_used"],
        "b0_mesh_sha256_16": sha256_16(Path(rep["mesh_ply"]).read_bytes()),
        "voxel_div": 256,
        "trunc_voxels": 4,
    }
    Path(args.out).write_text(json.dumps(grid, indent=2) + "\n")
    print(json.dumps({k: grid[k] for k in ("scene", "voxel", "truncation", "shape")}))


if __name__ == "__main__":
    main()
