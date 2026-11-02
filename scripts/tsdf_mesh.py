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


def _sample_depth_bilinear(depth: np.ndarray, uf: np.ndarray, vf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample depth at float pixel coords, ignoring non-positive taps.

    A sample is accepted only when the valid taps carry more than half of the total
    bilinear weight, which prevents single-tap extrapolation at silhouette borders.
    """
    height, width = depth.shape
    x0 = np.floor(uf).astype(np.int64)
    y0 = np.floor(vf).astype(np.int64)
    fx = uf - x0
    fy = vf - y0
    wsum = np.zeros(len(uf), dtype=np.float64)
    dsum = np.zeros(len(uf), dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            x = x0 + dx
            y = y0 + dy
            wt = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            d = np.zeros(len(uf), dtype=np.float64)
            d[inside] = depth[y[inside], x[inside]]
            tap_ok = inside & np.isfinite(d) & (d > 0)
            wsum += wt * tap_ok
            dsum += d * wt * tap_ok
    valid = wsum > 0.5
    observed = np.zeros(len(uf), dtype=np.float64)
    observed[valid] = dsum[valid] / wsum[valid]
    return observed, valid


def fuse(origin: np.ndarray, voxel: float, shape: tuple[int, int, int], sdf_trunc: float,
         frames: list[dict], near: float, far: float, confs: list[np.ndarray | None],
         weight_mode: str, w_max: float, stride: int,
         sample_mode: str = "nearest", robust: bool = False,
         robust_delta: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one observation per camera/voxel into a dense, chunked grid.

    Only voxels within +-sdf_trunc of an observed surface are updated (the standard TSDF
    band), which is also what keeps memory bounded. Crucially the *mask* is tracked, so
    "unknown" is distinct from "free space": an unseen voxel keeps NaN instead of being
    silently declared outside, which is what was shredding the surface into a lattice.
    """
    # Projective integration: one observation per voxel per frame, independent of
    # local depth sample density. Unknown voxels never contribute a signed distance.
    total = int(np.prod(shape))

    def project(frame: dict, start: int):
        """Voxel chunk -> per-voxel (flat index, z, observed depth, pixel coords, valid)."""
        depth, k, t = frame["depth"], frame["K"], frame["T"]
        height, width = depth.shape
        flat = np.arange(start, min(start + 200_000, total))
        idx = np.stack(np.unravel_index(flat, shape), axis=-1)
        pos = origin + (idx + 0.5) * voxel
        cam = pos @ t[:3, :3].T + t[:3, 3]
        z = cam[:, 2]
        positive = z > 1e-6
        uf = cam[:, 0] / np.maximum(z, 1e-6) * k[0, 0] + k[0, 2]
        vf = cam[:, 1] / np.maximum(z, 1e-6) * k[1, 1] + k[1, 2]
        u = np.rint(uf).astype(np.int64)
        v = np.rint(vf).astype(np.int64)
        if sample_mode == "bilinear":
            observed, ok = _sample_depth_bilinear(depth, uf, vf)
            valid = positive & ok
        else:
            valid = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            observed = np.zeros(len(flat), dtype=np.float64)
            observed[valid] = depth[v[valid], u[valid]]
        valid &= np.isfinite(observed) & (observed > near) & (observed < far)
        valid &= np.abs(observed - z) <= sdf_trunc
        return flat, z, observed, u, v, valid

    def integrate(frames, confs, estimate=None):
        wsum = np.zeros(total)
        fsum = np.zeros(total)
        csum = np.zeros((total, 3))
        for frame, conf in zip(frames, confs):
            for start in range(0, total, 200_000):
                flat, z, observed, u, v, valid = project(frame, start)
                if estimate is not None:
                    valid &= estimate[flat] > -np.inf
                flat, z, observed, u, v = flat[valid], z[valid], observed[valid], u[valid], v[valid]
                w = np.ones(len(flat)) if conf is None or weight_mode == "uniform" else np.clip(conf[v, u], 0.0, 8.0)
                if estimate is not None:
                    residual = (observed - z) / sdf_trunc - estimate[flat]
                    mag = np.abs(residual)
                    w = w * np.where(mag <= robust_delta, 1.0, robust_delta / np.maximum(mag, 1e-12))
                # Batch fusion averages every view before capping the reported support;
                # reaching W_max must not silently discard all subsequent cameras.
                wsum[flat] += w
                fsum[flat] += w * (observed - z) / sdf_trunc
                if frame["rgb"] is not None:
                    csum[flat] += w[:, None] * frame["rgb"][v, u] / 255.0
        return wsum, fsum, csum

    wsum, fsum, csum = integrate(frames, confs)
    if robust:
        # Second pass: Huber weights around the first-pass estimate, so views that
        # disagree with the current consensus are down-weighted instead of averaged in.
        estimate = np.full(total, -np.inf)
        seen = wsum > 0
        estimate[seen] = fsum[seen] / wsum[seen]
        wsum, fsum, _ = integrate(frames, confs, estimate=estimate)
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
    parser.add_argument("--roi-geometry", default=None,
                        help="derive bounds from these input depths, while fusing --filtered")
    parser.add_argument("--out", default=None, help="default <scene>/mesh")
    parser.add_argument("--voxel-div", type=int, default=256)
    parser.add_argument("--trunc-voxels", type=float, default=4.0)
    parser.add_argument("--depth-min", type=float, default=None)
    parser.add_argument("--depth-trunc", type=float, default=None)
    parser.add_argument("--weight-mode", choices=["confidence", "uniform"], default="confidence")
    parser.add_argument("--sample-mode", choices=["nearest", "bilinear"], default="nearest",
                        help="depth sampling at the projected voxel position; nearest reproduces the original pipeline")
    parser.add_argument("--robust", action="store_true",
                        help="second Huber-weighted fusion pass around the first-pass TSDF estimate")
    parser.add_argument("--robust-delta", type=float, default=0.3,
                        help="Huber delta in truncated-TSDF units for --robust")
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

    roi_frames = frames if args.roi_geometry is None else [
        load_frame(c, Path(args.roi_geometry), scene) for c in cameras]
    clouds = []
    for frame in roi_frames:
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
                                args.weight_mode, 64.0, args.stride,
                                sample_mode=args.sample_mode, robust=args.robust,
                                robust_delta=args.robust_delta)
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
        "sample_mode": args.sample_mode, "robust": bool(args.robust),
        "robust_delta": args.robust_delta if args.robust else None,
        "mesh_ply": str(mesh_path), "mesh_ply_bytes": mesh_path.stat().st_size,
        "roi_geometry": str(args.roi_geometry) if args.roi_geometry else str(root),
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
