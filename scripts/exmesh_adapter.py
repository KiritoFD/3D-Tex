#!/usr/bin/env python3
"""ExMesh scene adapter: VGGT-Omega geometry -> the exact layout ExMesh's train.py expects.

Verified against the official repo (Fan-Treasure/ExMesh):
  scene/__init__.py:54   needs `sparse/`             -> Colmap reader
  scene/dataset_readers.py:251  needs `mesh.ply`     -> our TSDF mesh
  utils/mono_utils.py:25 needs `mono_priors/da3/<img>.npz` with key "depth"
  train.py:158           needs `train_mask/<img>_gtmask.png`

So this script writes:
    <workdir>/images/<name>                 (copied / resized training photos)
    <workdir>/sparse/0/{cameras,images,points3D}.bin    (from Omega K/T)
    <workdir>/mesh.ply                      (copied TSDF triangle mesh)
    <workdir>/mono_priors/da3/<stem>.npz    {"depth": Omega depth on the mesh grid}
    <workdir>/train_mask/<stem>_gtmask.png  (foreground mask from cross-view agreement)

The `da3` directory name is upstream's; the code only reads the NPZ "depth" array, so Omega's
depth drops in without running Depth Anything 3 (noted in the plan supplied by the user).

Depth/pose scale: Omega depth and Omega cameras live in the same non-metric gauge, and the
mesh was built from that same gauge, so all three stay consistent. Nothing here rescales.

    python3 scripts/exmesh_adapter.py --scene eth3d_kicker
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from common import ROOT, scene_path


def build_colmap(sparse: Path, cameras: list[dict], width: int, height: int) -> dict:
    """Write COLMAP *text* model (cameras.txt / images.txt / points3D.txt).

    Written by hand on purpose: pycolmap's Python API changed Image.id -> image_id between 3.10
    and 4.x, and no 3.10 build exists for Python 3.12 on Windows. COLMAP's text format is
    stable across decades and ExMesh's reader (read_write_model.py) handles it directly.
    """
    sparse.mkdir(parents=True, exist_ok=True)
    k = cameras[0]
    lines = ["# Camera list with one line of data per camera:",
             "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
             f"1 PINHOLE {width} {height} {k['_fx']:.6f} {k['_fy']:.6f} {width / 2:.6f} {height / 2:.6f}"]
    (sparse / "cameras.txt").write_text("\n".join(lines) + "\n")

    lines = ["# Image list with two lines of data per image:",
             "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, frame in enumerate(cameras, start=1):
        rot = frame["_T"][:3, :3]
        q = rotation_matrix_to_quaternion(rot)
        t = frame["_T"][:3, 3]
        lines.append(f"{index} {' '.join(f'{v:.9f}' for v in q)} "
                     f"{' '.join(f'{v:.9f}' for v in t)} 1 {frame['file']}")
        lines.append("")            # empty 2D-observation line, as COLMAP writes it
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")
    (sparse / "points3D.txt").write_text(
        "# 3D point list (empty: ExMesh optimises the supplied mesh, not a point cloud)\n")
    return {"images": len(cameras), "camera": "PINHOLE", "width": width, "height": height}


def rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """COLMAP stores (qw, qx, qy, qz) for world->camera rotation, following Eigen's formula."""
    trace = float(np.trace(rot))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def foreground_mask(depth: np.ndarray, mask_path: Path, image_path: Path,
                    far_quantile: float) -> np.ndarray:
    """Object mask = cross-view-agreed depth, minus far background (table/wall)."""
    keep = np.load(mask_path) if mask_path.is_file() else (depth > 0)
    valid = keep & (depth > 0)
    if valid.sum() < 100:
        return valid.astype(np.uint8) * 255
    limit = float(np.quantile(depth[valid], far_quantile))
    return ((valid) & (depth <= limit)).astype(np.uint8) * 255


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ExMesh workdir from Omega geometry")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", default=None, help="default <scene>/exmesh_workdir")
    parser.add_argument("--mesh", default=None, help="default <scene>/mesh/mesh_init.ply")
    parser.add_argument("--filtered", default=None, help="default <scene>/geometry/filtered")
    parser.add_argument("--geometry", default=None, help="default <scene>/geometry/omega")
    parser.add_argument("--image-long-side", type=int, default=1024,
                        help="copy photos at this long side (ExMesh trains at -r 2 by default)")
    parser.add_argument("--far-quantile", type=float, default=0.85,
                        help="mask out depth beyond this within-object quantile (drops table/wall)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    scene = scene_path(args.scene)
    work = Path(args.out).resolve() if args.out else scene / "exmesh_workdir"
    mesh = Path(args.mesh).resolve() if args.mesh else scene / "mesh" / "mesh_init.ply"
    filtered = Path(args.filtered).resolve() if args.filtered else scene / "geometry" / "filtered"
    geometry = Path(args.geometry).resolve() if args.geometry else scene / "geometry" / "omega"
    if not mesh.is_file():
        raise SystemExit(f"mesh missing: {mesh} (run scripts/tsdf_mesh.py first)")

    camera_file = filtered / "cameras.json"
    if not camera_file.is_file():
        raise SystemExit(f"{camera_file} missing (run scripts/consistency_filter.py first)")
    cameras = json.loads(camera_file.read_text())["frames"]

    for sub in ("images", "sparse/0", "mono_priors/da3", "train_mask"):
        (work / sub).mkdir(parents=True, exist_ok=True)

    # one shared intrinsics entry: Omega predicts a centred principal point, and ExMesh's Camera
    # class also assumes a centred one, so exporting a single PINHOLE camera is faithful here.
    sizes = set()
    for frame in cameras:
        stem = Path(frame["frame"]).stem
        depth = np.load(filtered / "depth" / f"{stem}.npy")
        depth = depth[..., 0] if depth.ndim == 3 else depth
        k = np.asarray(frame["K_original"], dtype=np.float64)
        t = np.asarray(frame["T_world_to_cam"], dtype=np.float64)
        frame["_T"] = t
        frame["_fx"] = k[0, 0]          # K_original is already in copied-image pixels
        frame["_fy"] = k[1, 1]
        # depth is annotated in ORIGINAL pixels (K_original), so the NPZ keeps its own shape
        np.savez_compressed(work / "mono_priors" / "da3" / f"{stem}.npz",
                            depth=depth.astype(np.float32))

        source = scene / "images" / frame["file"]
        with Image.open(source) as im:
            im = im.convert("RGB")
            scale = args.image_long_side / max(im.size)
            if scale < 1:
                im = im.resize((int(im.width * scale), int(im.height * scale)),
                               Image.Resampling.LANCZOS)
            target = work / "images" / frame["file"]
            im.save(target, quality=95)
            sizes.add(im.size)

        mask_path = filtered / "mask" / f"{stem}.npy"
        mask = foreground_mask(depth, mask_path, source, args.far_quantile)
        Image.fromarray(mask).resize(depth.shape[::-1], Image.Resampling.NEAREST).save(
            work / "train_mask" / f"{stem}_gtmask.png")

    if len(sizes) > 1:
        raise SystemExit(f"ExMesh needs identical image sizes, got {sorted(sizes)}")
    width, height = sizes.pop()
    colmap_stats = build_colmap(work / "sparse" / "0", cameras, width, height)
    shutil.copy2(mesh, work / "mesh.ply")

    stats = {"scene": args.scene, "workdir": str(work), "images": len(cameras),
             "image_size": [width, height], **colmap_stats,
             "mesh_ply": str(work / "mesh.ply"), "mesh_bytes": (work / "mesh.ply").stat().st_size,
             "mask_far_quantile": args.far_quantile,
             "note": "Omega depth/pose gauge; no rescaling applied anywhere."}
    (work / "adapter_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[out] {work}")
    print(f"[files] images={len(cameras)} camera={width}x{height} "
          f"mesh={stats['mesh_bytes'] / 1024:.0f} KiB depth-npz={len(cameras)} masks={len(cameras)}")
    print("[run] cd ExMesh && python train.py -s " + str(work) + " -m " + str(work / "out"))
    if args.json:
        (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
        (ROOT / "outputs" / f"{args.scene}_exmesh_input.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
