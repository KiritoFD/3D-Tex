#!/usr/bin/env python3
"""Render a deterministic synthetic multi-view scene for pipeline smoke tests.

Why: the real ETH3D archives are ~1.3 GiB over a throttled link, but "does
VGGT -> COLMAP sparse -> 3DGS actually run end to end" only needs a handful of images with
parallax. This generator writes those in seconds, offline, with numpy + Pillow only.

The texture lives in *object* space, so it is multi-view consistent and gives the
feed-forward model something real to match on. Depth is interpolated per triangle: for a
planar face the view-space z is linear in screen space, so barycentric z is exact, and a
far-to-near paint order gives a correct z-buffer without any per-pixel bookkeeping.

It is a plumbing fixture, NOT a benchmark: never report its PSNR as reconstruction quality.

    python3 scripts/make_smoke_scene.py --scene smoke_box --frames 6 --size 512
    python3 scripts/run_vggt.py --scene smoke_box --profile tiny
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from common import scene_path

CORNERS = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
                   dtype=np.float64)
# index = xb*4 + yb*2 + zb  (see the comprehension above). Each quad lists 4 corner indices
# walking around one face; winding is irrelevant because occlusion is resolved by the
# far-to-near paint order and lighting uses the explicit normal.
QUADS = [
    ([0, 1, 3, 2], 0, (-1.0, 0.0, 0.0)),   # x = -0.5
    ([4, 6, 7, 5], 1, (1.0, 0.0, 0.0)),    # x = +0.5
    ([0, 1, 5, 4], 2, (0.0, -1.0, 0.0)),   # y = -0.5
    ([2, 3, 7, 6], 3, (0.0, 1.0, 0.0)),    # y = +0.5
    ([0, 4, 6, 2], 4, (0.0, 0.0, -1.0)),   # z = -0.5
    ([1, 5, 7, 3], 5, (0.0, 0.0, 1.0)),    # z = +0.5
]
FACE_COLORS = np.array([
    [0.86, 0.24, 0.24], [0.24, 0.78, 0.36], [0.28, 0.44, 0.90],
    [0.92, 0.80, 0.24], [0.80, 0.36, 0.88], [0.30, 0.82, 0.84],
], dtype=np.float64)
LIGHT = np.array([0.4, 0.7, 0.85], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)


def orbit_camera(index: int, total: int, radius: float, fov_deg: float,
                 size: int) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV-convention camera (x right, y down, z forward) orbiting the origin."""
    azimuth = 2 * math.pi * index / max(total, 1)
    eye = radius * np.array([math.cos(azimuth), math.sin(azimuth), 0.42])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])
    intrinsic = np.eye(3, dtype=np.float64)
    focal = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    intrinsic[0, 0] = intrinsic[1, 1] = focal
    intrinsic[0, 2] = intrinsic[1, 2] = size / 2.0
    return np.hstack([rot, (-rot @ eye)[:, None]]), intrinsic


def project(points: np.ndarray, extr: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """World -> (u, v, z) with a safe positive depth."""
    cam = points @ extr[:, :3].T + extr[:, 3]
    depth = np.maximum(cam[:, 2], 1e-6)
    return np.stack([cam[:, 0] / depth * intr[0, 0] + intr[0, 2],
                     cam[:, 1] / depth * intr[1, 1] + intr[1, 2],
                     cam[:, 2]], axis=-1)


def raster_triangle(screen: np.ndarray, world: np.ndarray, size: int, face_id: int,
                    checker: float, lum: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One triangle -> (flat pixel index, depth, linear RGB) for the pixels it covers."""
    if (screen[:, 2] <= 1e-4).any():
        return np.empty(0, np.int64), np.empty(0), np.empty((0, 3))
    u_lo = max(0, int(math.floor(screen[:, 0].min())))
    u_hi = min(size - 1, int(math.ceil(screen[:, 0].max())))
    v_lo = max(0, int(math.floor(screen[:, 1].min())))
    v_hi = min(size - 1, int(math.ceil(screen[:, 1].max())))
    if u_hi <= u_lo or v_hi <= v_lo:
        return np.empty(0, np.int64), np.empty(0), np.empty((0, 3))

    grid_v, grid_u = np.meshgrid(np.arange(v_lo, v_hi + 1) + 0.5,
                                 np.arange(u_lo, u_hi + 1) + 0.5, indexing="ij")
    pts = np.stack([grid_u.ravel(), grid_v.ravel()], axis=-1).astype(np.float64)

    edge0 = screen[1, :2] - screen[0, :2]
    edge1 = screen[2, :2] - screen[0, :2]
    denom = edge0[0] * edge1[1] - edge0[1] * edge1[0]
    if abs(denom) < 1e-12:
        return np.empty(0, np.int64), np.empty(0), np.empty((0, 3))
    rel = pts - screen[0, :2]
    b1 = (rel[:, 0] * edge1[1] - rel[:, 1] * edge1[0]) / denom
    b2 = (edge0[0] * rel[:, 1] - edge0[1] * rel[:, 0]) / denom
    inside = (b1 >= 0.0) & (b2 >= 0.0) & ((b1 + b2) <= 1.0)
    if not inside.any():
        return np.empty(0, np.int64), np.empty(0), np.empty((0, 3))

    b1, b2 = b1[inside], b2[inside]
    b0 = 1.0 - b1 - b2
    depth = b0 * screen[0, 2] + b1 * screen[1, 2] + b2 * screen[2, 2]
    obj = b0[:, None] * world[0] + b1[:, None] * world[1] + b2[:, None] * world[2]

    cell = np.floor(obj / checker).astype(np.int64).sum(axis=1)
    pattern = np.where(cell % 2 == 0, 1.0, 0.58)
    rgb = np.clip(FACE_COLORS[face_id][None, :] * pattern[:, None] * lum, 0.0, 1.0)

    pu = np.clip(np.floor(pts[inside, 0]).astype(np.int64), 0, size - 1)
    pv = np.clip(np.floor(pts[inside, 1]).astype(np.int64), 0, size - 1)
    return pv * size + pu, depth, rgb


def render_frame(size: int, extr: np.ndarray, intr: np.ndarray, seed: int,
                 checker: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pixels = size * size
    frame = np.tile(0.07 + 0.03 * rng.random(3), (pixels, 1))

    index_parts, depth_parts, rgb_parts = [], [], []
    for quad, face_id, normal in QUADS:
        world_quad = CORNERS[quad]
        screen_quad = project(world_quad, extr, intr)
        lum = 0.28 + 0.72 * max(0.0, float(np.dot(np.array(normal), LIGHT)))
        for tri in ([0, 1, 2], [0, 2, 3]):
            idx, depth, rgb = raster_triangle(screen_quad[tri], world_quad[tri], size,
                                              face_id, checker, lum)
            if idx.size:
                index_parts.append(idx)
                depth_parts.append(depth)
                rgb_parts.append(rgb)
    if not index_parts:
        raise RuntimeError("nothing projected - move the camera closer (see --radius)")

    index = np.concatenate(index_parts)
    depth = np.concatenate(depth_parts)
    rgb = np.concatenate(rgb_parts)
    far_to_near = np.argsort(-depth)                    # last write per pixel is the nearest
    frame[index[far_to_near]] = rgb[far_to_near]
    return (frame.reshape(size, size, 3) * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic multi-view smoke scene (offline)")
    parser.add_argument("--scene", default="smoke_box")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--radius", type=float, default=1.9)
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--checker", type=float, default=0.0625, help="object-space cell size")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.frames < 3:
        parser.error("--frames must be >= 3 (VGGT needs parallax)")
    dest = scene_path(args.scene)
    images = dest / "images"
    if images.exists() and any(images.iterdir()) and not args.force:
        raise SystemExit(f"{images} already has files; pass --force to overwrite")
    images.mkdir(parents=True, exist_ok=True)
    for stale in images.glob("*"):
        stale.unlink()

    names = []
    truth = []
    for index in range(args.frames):
        extr, intr = orbit_camera(index, args.frames, args.radius, args.fov, args.size)
        rgb = render_frame(args.size, extr, intr, args.seed + index, args.checker)
        name = f"smoke_{index:04d}.png"
        Image.fromarray(rgb).save(images / name)
        names.append(name)
        truth.append({"image": name,
                      "cam_from_world": extr.tolist(),
                      "intrinsic": intr.tolist(),
                      "center": (-extr[:, :3].T @ extr[:, 3]).ravel().tolist()})

    (dest / "smoke_manifest.json").write_text(json.dumps({
        "scene": args.scene, "synthetic": True, "benchmark": False, "images": names,
        "size": args.size, "frames": args.frames, "seed": args.seed, "checker": args.checker,
        "fov_deg": args.fov,
        "ground_truth_cameras": truth,
        "warning": "Plumbing fixture only. Never quote PSNR/SSIM from this scene as reconstruction quality.",
    }, indent=2) + "\n")
    print(f"[smoke] wrote {len(names)} images to {images}")


if __name__ == "__main__":
    main()
