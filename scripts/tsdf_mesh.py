#!/usr/bin/env python3
"""A0 stage 2: fuse consistent depth into a TSDF and extract a triangle mesh (numpy only).

Why not Open3D: `ScalableTSDFVolume.integrate()` silently produced an empty volume in the
Open3D 0.20 build here (even for a synthetic constant-depth plane), and `VoxelBlockGrid`
needs explicit block allocation. A vectorised voxel array is easier to verify and exactly
matches docs/MESH_UV_PIPELINE.md section 3.2:

    sdf = clamp((observed_depth - projected_voxel_z) / truncation, -1, 1)
    F'  = (W*F + w*sdf) / (W + w),     W' = min(W + w, W_max)

Strategy: for each frame, unproject depth at stride, splat each sample into the voxels within
one truncation band, and merge with np.add.at over a flat grid. No Python per-voxel loop, so
a ~200x70x130 grid fuses in seconds.

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
        k[0, 2] *= width / frame["model_width"]
        k[1, 2] *= height / frame["model_height"]
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
            rgb = np.asarray(im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
    return {"depth": depth, "K": k, "T": t, "rgb": rgb, "name": frame["frame"]}


def unproject(frame: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    depth, k, t = frame["depth"], frame["K"], frame["T"]
    d = depth[rows, cols].astype(np.float64)
    cam = np.stack([(cols + 0.5 - k[0, 2]) / k[0, 0] * d,
                    (rows + 0.5 - k[1, 2]) / k[1, 1] * d, d], axis=-1)
    return (t[:3, :3].T @ (cam - t[:3, 3]).T).T


def fuse(origin: np.ndarray, voxel: float, shape: tuple[int, int, int], sdf_trunc: float,
         frames: list[dict], near: float, far: float, confs: list[np.ndarray | None],
         weight_mode: str, w_max: float, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse depth into `shape` and fill `band_masks` with the voxels each frame observed.

    Only voxels within +-sdf_trunc of an observed surface are updated (the standard TSDF
    band), which is also what keeps memory bounded. Crucially the *mask* is tracked, so
    "unknown" is distinct from "free space": an unseen voxel keeps NaN instead of being
    silently declared outside, which is what was shredding the surface into a lattice.
    """
    nx, ny, nz = shape
    total = nx * ny * nz
    wsum = np.zeros(total, dtype=np.float64)
    fsum = np.zeros(total, dtype=np.float64)
    csum = np.zeros(total * 3, dtype=np.float64)
    band = int(np.ceil(sdf_trunc / voxel)) + 1
    # The band must be dense enough to form a continuous sheet: at camera distance `d` one
    # voxel spans ~d/f pixels, so a stride larger than that leaves gaps between neighbouring
    # samples and marching cubes returns a lattice instead of a surface.
    for frame, conf in zip(frames, confs):
        depth, k, t = frame["depth"], frame["K"], frame["T"]
        height, width = depth.shape
        rot, trans = t[:3, :3], t[:3, 3]
        fx = float(k[0, 0])
        finite = depth[depth > 0]
        median_depth = float(np.median(finite)) if finite.size else 1.0
        frame_stride = max(1, int(round(median_depth / max(fx, 1e-6) * 0.75)))
        frame_stride = max(1, min(stride, frame_stride))
        rr, cc = np.mgrid[0:height:frame_stride, 0:width:frame_stride]
        rr, cc = rr.ravel(), cc.ravel()
        d = depth[rr, cc]
        sel = (d > near) & (d < far)
        if sel.sum() < 16:
            continue
        rr, cc = rr[sel], cc[sel]
        world = unproject(frame, rr, cc)
        base = np.floor((world - origin) / voxel).astype(np.int64)

        # Chunk the samples: (samples x band^3) grows fast and would otherwise need GBs.
        offs = band_offsets(band)
        chunk = max(1, min(len(rr), 200_000 // max(len(offs), 1)))
        for start_idx in range(0, len(rr), chunk):
            stop = min(start_idx + chunk, len(rr))
            idx = (base[start_idx:stop, None, :] + offs[None, :, :]).reshape(-1, 3)
            ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
            if not ok.any():
                continue
            idx = idx[ok]

            pos = origin + (idx + 0.5) * voxel
            cpos = (rot @ pos.T).T + trans
            z = cpos[:, 2]
            valid = z > 1e-6
            with np.errstate(divide="ignore", invalid="ignore"):
                u = cpos[:, 0] / z * k[0, 0] + k[0, 2]
                v = cpos[:, 1] / z * k[1, 1] + k[1, 2]
            in_img = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not in_img.any():
                continue
            ui = np.clip(np.round(u[in_img]).astype(np.int64), 0, width - 1)
            vi = np.clip(np.round(v[in_img]).astype(np.int64), 0, height - 1)
            observed = depth[vi, ui].astype(np.float64)
            zz = z[in_img]
            # the truncation band: only +-sdf_trunc around the measured surface is meaningful
            good = (observed > near) & (np.abs(zz - observed) <= sdf_trunc + voxel)
            if not good.any():
                continue

            idx_g = idx[in_img][good]
            sdf = np.clip((observed[good] - zz[good]) / sdf_trunc, -1.0, 1.0)
            if weight_mode == "confidence" and conf is not None:
                w = np.clip(conf[vi, ui][good].astype(np.float64), 0.05, 8.0)
            else:
                w = np.ones_like(sdf)
            flat = (idx_g[:, 0] * ny + idx_g[:, 1]) * nz + idx_g[:, 2]
            np.add.at(wsum, flat, w)
            np.add.at(fsum, flat, w * sdf)
            if frame["rgb"] is not None:
                colour = frame["rgb"][vi, ui][good].astype(np.float64) / 255.0
                for c in range(3):
                    np.add.at(csum, flat * 3 + c, w * colour[:, c])
            del idx, pos, cpos, idx_g, flat

    weight = wsum.reshape(shape)
    f = np.full(total, np.nan, dtype=np.float64)      # NaN = never observed (unknown, not outside)
    nz_mask = wsum > 0
    f[nz_mask] = fsum[nz_mask] / wsum[nz_mask]
    tsdf = np.full(shape, np.nan, dtype=np.float32)
    tsdf.reshape(-1)[nz_mask] = f[nz_mask].astype(np.float32)
    colour = np.zeros((total, 3), dtype=np.float32)
    colour[nz_mask] = (csum.reshape(-1, 3)[nz_mask] / wsum[nz_mask, None]).astype(np.float32)
    return tsdf, colour.reshape(*shape, 3), weight


def band_offsets(band: int) -> np.ndarray:
    """Voxel offsets whose Chebyshev distance is <= band (a cube, but ordered by radius)."""
    r = np.arange(-band, band + 1)
    grid = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    order = np.argsort(np.abs(grid).max(axis=1), kind="stable")
    return grid[order]


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
    parser.add_argument("--stride", type=int, default=2, help="depth sampling stride for fusion")
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
    near = args.depth_min if args.depth_min is not None else float(np.percentile(valid, 1))
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
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
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

    # marching cubes needs a finite field. Unknown voxels are filled with an "outside" value but
    # we only *trust* crossings that touch an observed voxel, so unseen space cannot invent a
    # surface. Without this the +1 fill met the -1 band on a lattice of sample points.
    observed = weight > 0
    if not ((np.nan_to_num(tsdf, nan=1.0) < 0) & observed).any():
        raise SystemExit("No voxel penetrated the surface: fusion saw only the front side. "
                         "Raise --trunc-voxels or check the depth scale.")
    field = np.where(observed, tsdf, 1.0).astype(np.float32)
    verts, faces, normals, _ = measure.marching_cubes(field, level=0.0, spacing=(voxel,) * 3)
    # drop triangles that live entirely in the filled (unknown) region
    tri_idx = np.floor(verts / voxel).astype(np.int64)
    tri_idx = np.clip(tri_idx, 0, np.array(observed.shape) - 1)
    touches_observed = observed[tri_idx[:, 0], tri_idx[:, 1], tri_idx[:, 2]]
    faces = faces[touches_observed[faces].any(axis=1)]
    used = np.unique(faces)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts, faces, normals = verts[used], remap[faces], normals[used]
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
