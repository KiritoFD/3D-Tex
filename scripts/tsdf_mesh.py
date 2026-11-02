#!/usr/bin/env python3
"""A0 stage 2: fuse consistent depth into a TSDF and extract a triangle mesh (numpy only).

Why not Open3D: `ScalableTSDFVolume.integrate()` silently produced an empty volume in the
Open3D 0.20 build here (even for a synthetic constant-depth plane), and `VoxelBlockGrid`
needs explicit block allocation. A vectorised voxel array is easier to verify and exactly
matches docs/MESH_UV_PIPELINE.md section 3.2:

    sdf = clamp((observed_depth - projected_voxel_z) / truncation, -1, 1)
    F'  = (W*F + w*sdf) / (W + w),     W' = min(W + w, W_max)

Project each voxel into every camera once; integrate only measured truncation bands.
Extract only cells whose eight corners were observed.

    python3 scripts/tsdf_mesh.py --scene eth3d_pipes --voxel-div 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from skimage import measure

from common import OUTPUTS, scene_path


def load_frame(frame: dict, root: Path, scene: Path) -> dict:
    stem = Path(frame["frame"]).stem
    depth = np.load(root / "depth" / f"{stem}.npy")
    depth = depth[..., 0] if depth.ndim == 3 else depth
    depth = depth.astype(np.float32)
    height, width = depth.shape
    k = np.asarray(frame["K_model"], dtype=np.float64).copy()
    if (height, width) != (frame["model_height"], frame["model_width"]):
        k[0] *= width / frame["model_width"]
        k[1] *= height / frame["model_height"]
    t = np.asarray(frame["T_world_to_cam"], dtype=np.float64)
    if t.shape == (3, 4):
        t4 = np.eye(4)
        t4[:3, :] = t
        t = t4
    rgb = None
    image_path = scene / "images" / frame["file"]
    if image_path.is_file():
        from PIL import Image
        with Image.open(image_path) as im:
            import cv2
            affine = np.asarray(frame["pixel_affine_original_to_model"]).copy()
            affine[0] *= width / frame["model_width"]
            affine[1] *= height / frame["model_height"]
            rgb = cv2.warpAffine(np.asarray(im.convert("RGB")), affine[:2], (width, height))
    return {"depth": depth, "K": k, "T": t, "rgb": rgb, "name": frame["frame"]}


def unproject(frame: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    depth, k, t = frame["depth"], frame["K"], frame["T"]
    d = depth[rows, cols].astype(np.float64)
    cam = np.stack([(cols - k[0, 2]) / k[0, 0] * d,
                    (rows - k[1, 2]) / k[1, 1] * d, d], axis=-1)
    return (t[:3, :3].T @ (cam - t[:3, 3]).T).T


def fuse(origin: np.ndarray, voxel: float, shape: tuple[int, int, int], sdf_trunc: float,
         frames: list[dict], near: float, far: float, confs: list[np.ndarray | None],
         weight_mode: str, w_max: float, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one observation per camera/voxel into a dense, chunked grid.

    Only voxels within +-sdf_trunc of an observed surface are updated (the standard TSDF
    band), which is also what keeps memory bounded. Crucially the *mask* is tracked, so
    "unknown" is distinct from "free space": an unseen voxel keeps NaN instead of being
    silently declared outside, which is what was shredding the surface into a lattice.
    """
    # Projective integration: one observation per voxel per frame, independent of
    # local depth sample density. Unknown voxels never contribute a signed distance.
    total = int(np.prod(shape))
    wsum = np.zeros(total)
    fsum = np.zeros(total)
    csum = np.zeros((total, 3))
    for frame, conf in zip(frames, confs):
        depth, k, t = frame["depth"], frame["K"], frame["T"]
        height, width = depth.shape
        for start in range(0, total, 200_000):
            flat = np.arange(start, min(start + 200_000, total))
            idx = np.stack(np.unravel_index(flat, shape), axis=-1)
            pos = origin + (idx + 0.5) * voxel
            cam = pos @ t[:3, :3].T + t[:3, 3]
            z = cam[:, 2]
            positive = z > 1e-6
            u = np.rint(cam[:, 0] / np.maximum(z, 1e-6) * k[0, 0] + k[0, 2]).astype(int)
            v = np.rint(cam[:, 1] / np.maximum(z, 1e-6) * k[1, 1] + k[1, 2]).astype(int)
            inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            flat, z, u, v = flat[inside], z[inside], u[inside], v[inside]
            observed = depth[v, u]
            good = np.isfinite(observed) & (observed > near) & (observed < far)
            good &= np.abs(observed - z) <= sdf_trunc
            flat, z, u, v, observed = flat[good], z[good], u[good], v[good], observed[good]
            w = np.ones(len(flat)) if conf is None or weight_mode == "uniform" else np.clip(conf[v, u], 0.0, 8.0)
            # Batch fusion averages every view before capping the reported support;
            # reaching W_max must not silently discard all subsequent cameras.
            wsum[flat] += w
            fsum[flat] += w * (observed - z) / sdf_trunc
            if frame["rgb"] is not None:
                csum[flat] += w[:, None] * frame["rgb"][v, u] / 255.0
    known = wsum > 0
    field = np.full(total, np.nan, dtype=np.float32)
    field[known] = fsum[known] / wsum[known]
    colour = np.zeros((total, 3), dtype=np.float32)
    colour[known] = csum[known] / wsum[known, None]
    return field.reshape(shape), colour.reshape(*shape, 3), np.minimum(wsum, w_max).reshape(shape)


def extract_surface(tsdf, voxel):
    # Every cell must have eight observed corners. Filling unknown with a positive
    # sign alone invents surfaces along the negative-band/unknown boundary.
    known = np.isfinite(tsdf)
    cells = np.ones(tuple(n - 1 for n in tsdf.shape), dtype=bool)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                cells &= known[x:x+cells.shape[0], y:y+cells.shape[1], z:z+cells.shape[2]]
    field = np.where(known, tsdf, 1.0)
    verts, faces, normals, _ = measure.marching_cubes(field, level=0.0)
    centres = verts[faces].mean(axis=1)
    idx = np.clip(np.floor(centres).astype(int), 0, np.array(cells.shape) - 1)
    faces = faces[cells[idx[:, 0], idx[:, 1], idx[:, 2]]]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return (verts[used] + 0.5) * voxel, remap[faces], normals[used]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse TSDF fusion -> triangle mesh (A0)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--filtered", default=None, help="default <scene>/geometry/filtered")
    parser.add_argument("--out", default=None, help="default <scene>/mesh")
    parser.add_argument("--voxel-div", type=int, default=256)
    parser.add_argument("--trunc-voxels", type=float, default=4.0)
    parser.add_argument("--depth-min", type=float, default=None)
    parser.add_argument("--depth-trunc", type=float, default=None)
    parser.add_argument("--weight-mode", choices=["confidence", "uniform"], default="confidence")
    parser.add_argument("--stride", type=int, default=2, help="legacy CLI option; projective fusion evaluates every voxel")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scene = scene_path(args.scene)
    root = Path(args.filtered).resolve() if args.filtered else scene / "geometry" / "filtered"
    out = Path(args.out).resolve() if args.out else scene / "mesh"
    cameras = json.loads((root / "cameras.json").read_text())["frames"]
    if args.max_frames:
        cameras = cameras[: args.max_frames]

    frames = [load_frame(c, root, scene) for c in cameras]
    valid = np.concatenate([f["depth"][f["depth"] > 0] for f in frames])
    if valid.size < 1000:
        raise SystemExit("Too few valid depth pixels; run consistency_filter.py first.")
    near = args.depth_min if args.depth_min is not None else float(0.0)
    far = args.depth_trunc if args.depth_trunc is not None else float(np.median(valid) * 6.0)
    print(f"[depth] {valid.size} valid px | median {np.median(valid):.3f} | fusing [{near:.3f}, {far:.3f}]")

    confs = []
    for frame in frames:
        path = root.parent / "omega" / "confidence" / f"{Path(frame['name']).stem}.npy"
        c = np.load(path) if path.is_file() else None
        confs.append(None if c is None else (c[..., 0] if c.ndim == 3 else c))

    clouds = []
    for frame in frames:
        rr, cc = np.nonzero((frame["depth"] > near) & (frame["depth"] < far))
        if len(rr):
            clouds.append(unproject(frame, rr, cc))
    pts = np.concatenate(clouds)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    diagonal = float(np.linalg.norm(hi - lo))
    voxel = diagonal / args.voxel_div
    sdf_trunc = voxel * args.trunc_voxels
    lo -= sdf_trunc * 2
    hi += sdf_trunc * 2
    shape = tuple(np.ceil((hi - lo) / voxel).astype(int))
    print(f"[roi] {len(pts)} pts, diag {diagonal:.4f} -> voxel {voxel:.5f}, trunc {sdf_trunc:.5f}, "
          f"grid {list(shape)} ({np.prod(shape) / 1e6:.2f}M voxels)")

    start = time.time()
    tsdf, colour, weight = fuse(lo, voxel, shape, sdf_trunc, frames, near, far, confs,
                                args.weight_mode, 64.0, args.stride)
    observed = weight > 0
    print(f"[tsdf] {observed.sum()} voxels observed in {time.time() - start:.1f}s")

    # Unknown voxels are excluded cell-by-cell by extract_surface.
    observed = weight > 0
    if not ((np.nan_to_num(tsdf, nan=1.0) < 0) & observed).any():
        raise SystemExit("No voxel penetrated the surface: fusion saw only the front side. "
                         "Raise --trunc-voxels or check the depth scale.")
    verts, faces, normals = extract_surface(tsdf, voxel)
    verts += lo
    print(f"[mesh] {len(verts)} verts / {len(faces)} tris")

    vcol = sample_colours(verts - lo, colour, voxel)
    out.mkdir(parents=True, exist_ok=True)
    mesh_path = out / "mesh_init.ply"
    write_ply(mesh_path, verts, faces, vcol, normals)

    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    stats = {
        "scene": args.scene, "frames_used": len(frames), "vertices": int(len(verts)),
        "triangles": int(len(faces)), "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "bbox_min": np.round(verts.min(0), 4).tolist(), "bbox_max": np.round(verts.max(0), 4).tolist(),
        "voxel": round(voxel, 6), "sdf_trunc": round(sdf_trunc, 6), "diagonal": round(diagonal, 4),
        "voxels_observed": int(observed.sum()), "depth_near": round(near, 4),
        "depth_far": round(far, 4), "weight_mode": args.weight_mode,
        "mesh_ply": str(mesh_path), "mesh_ply_bytes": mesh_path.stat().st_size,
    }
    (out / "mesh_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[topology] boundary edges {stats['boundary_edges']}, non-manifold {stats['nonmanifold_edges']}")
    print(f"[out] {mesh_path} ({stats['mesh_ply_bytes'] / 1024:.0f} KiB)")
    if args.json:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        (OUTPUTS / f"{args.scene}_mesh.json").write_text(json.dumps(stats, indent=2) + "\n")


def sample_colours(verts_grid: np.ndarray, colour: np.ndarray, voxel: float) -> np.ndarray:
    idx = np.clip(np.round(verts_grid / voxel - 0.5).astype(np.int64),
                  0, np.array(colour.shape[:3]) - 1)
    sampled = colour[idx[:, 0], idx[:, 1], idx[:, 2]].copy()
    empty = sampled.sum(axis=1) < 1e-6
    sampled[empty] = 0.55
    return sampled


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colours: np.ndarray,
              normals: np.ndarray | None) -> None:
    head = ["ply", "format ascii 1.0", f"element vertex {len(verts)}",
            "property float x", "property float y", "property float z",
            "property float nx", "property float ny", "property float nz",
            "property uchar red", "property uchar green", "property uchar blue",
            f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    rgb = (np.clip(colours, 0, 1) * 255).astype(np.uint8)
    normals = np.zeros_like(verts) if normals is None else normals
    with path.open("w") as handle:
        handle.write("\n".join(head) + "\n")
        for (x, y, z), (nx, ny, nz), (r, g, b) in zip(verts, normals, rgb):
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.4f} {ny:.4f} {nz:.4f} {r} {g} {b}\n")
        for a, b, c in faces:
            handle.write(f"3 {a} {b} {c}\n")


if __name__ == "__main__":
    main()
