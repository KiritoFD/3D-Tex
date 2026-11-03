"""COLMAP known-pose dense reconstruction (PatchMatch) -> depth/normal npy export.

Builds a COLMAP sparse model from known cameras/poses (empty points3D), runs
image_undistorter + patch_match_stereo, and converts the .bin depth/normal maps
to numpy arrays plus a cameras.json, so capture_eval.py can score PatchMatch
depth through the identical frozen protocol used for VGGT depth.

    .venv/Scripts/python.exe scripts/capture/colmap_mvs.py \
        --images data/scenes/nerf_lego16/images --poses outputs/e2_anchor16_poses.json \
        --workspace outputs/colmap/anchor16 --out data/scenes/nerf_lego16/geometry/patchmatch
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def rotation_to_quaternion(rot):
    """3x3 rotation matrix -> (qw, qx, qy, qz)."""
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    if m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        return (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    if m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        return (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
    return (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s


def write_model(sparse_dir, frames):
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "".join(f"{i + 1} PINHOLE {f['width']} {f['height']} "
                  f"{f['K_original'][0][0]} {f['K_original'][1][1]} "
                  f"{f['K_original'][0][2]} {f['K_original'][1][2]}\n"
                  for i, f in enumerate(frames)))
    lines = ["# Image list with two lines of data per image:\n"
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
    for i, f in enumerate(frames):
        t = np.asarray(f["T_world_to_cam"], dtype=np.float64)
        qw, qx, qy, qz = rotation_to_quaternion(t[:3, :3])
        tx, ty, tz = t[:3, 3]
        lines.append(f"{i + 1} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
                     f"{tx:.12g} {ty:.12g} {tz:.12g} {i + 1} {f['stem']}.png\n\n")
    (sparse_dir / "images.txt").write_text("".join(lines))
    (sparse_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")


def read_bin(path):
    """COLMAP .bin map: ASCII 'width height channels' header then little-endian float32."""
    with open(path, "rb") as handle:
        header = b""
        while len(header.split()) < 3:
            header += handle.read(1)
        width, height, channels = map(int, header.split())
        data = np.frombuffer(handle.read(), dtype=np.float32)
    return data.reshape(height, width, channels)


def run(cmd, log_name, workspace):
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    (workspace / f"{log_name}.log").write_text(
        "$ " + " ".join(map(str, cmd)) + "\n\n" + result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"{log_name} failed ({result.returncode}); see {workspace / f'{log_name}.log'}")
    print(f"[{log_name}] {time.time() - started:.1f}s", flush=True)


def write_stereo_cfg(cfg_path, frames, num_source):
    """Per-reference source image list; empty points3D leaves auto-selection empty."""
    centres = []
    for f in frames:
        t = np.asarray(f["T_world_to_cam"], dtype=np.float64)
        centres.append(-t[:3, :3].T @ t[:3, 3])
    centres = np.asarray(centres)
    lines = ["# Stereo Configuration\n"]
    for i, frame in enumerate(frames):
        dist = np.linalg.norm(centres - centres[i], axis=1)
        order = [j for j in np.argsort(dist, kind="stable") if j != i][:num_source]
        if len(order) >= len(frames) - 1:
            lines.append(f"{frame['stem']}.png\n__all__\n")
        else:
            sources = ", ".join(frames[j]["stem"] + ".png" for j in order)
            lines.append(f"{frame['stem']}.png\n{sources}\n")
    cfg_path.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--poses", required=True, help="aligned poses JSON from export_aligned_poses.py")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True, help="geometry dir for depth/normal/cameras.json")
    parser.add_argument("--colmap-exe", default="data/.cache/colmap/colmap/bin/colmap.exe")
    parser.add_argument("--max-image-size", type=int, default=0, help="PatchMatchStereo cap; 0 = full resolution")
    parser.add_argument("--source-views", type=int, default=10,
                        help="source images per reference for patch-match.cfg (nearest camera centres); "
                             "uses __all__ when >= views-1")
    parser.add_argument("--depth-min", type=float, default=0.0,
                        help="PatchMatchStereo depth_min; required when sparse model has no points")
    parser.add_argument("--depth-max", type=float, default=0.0,
                        help="PatchMatchStereo depth_max; required when sparse model has no points")
    parser.add_argument("--skip-run", action="store_true", help="only convert existing .bin maps")
    args = parser.parse_args()
    images = Path(args.images).resolve()
    payload = json.loads(Path(args.poses).read_text())
    frames = payload["frames"]
    workspace = Path(args.workspace).resolve()
    out = Path(args.out).resolve()
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "normals").mkdir(parents=True, exist_ok=True)
    colmap = Path(args.colmap_exe)
    # Windows .exe paths won't be is_file() under WSL; check via "wsl test" only
    # when running inside WSL, otherwise rely on the host check.
    if sys.platform.startswith("linux"):
        if not Path(str(colmap)).is_file() and not Path(str(colmap).replace("/mnt/g", "G:").replace("/", "\\")).is_file():
            raise SystemExit(f"COLMAP executable missing: {colmap}")
    elif not colmap.is_file():
        raise SystemExit(f"COLMAP executable missing: {colmap}")
    dense = workspace / "dense"
    sparse = workspace / "sparse" / "0"
    if not args.skip_run:
        # Fresh dense workspace (undistorter refuses to overwrite existing images)
        if dense.is_dir():
            import shutil
            shutil.rmtree(dense)
        write_model(sparse, frames)
        run([colmap, "image_undistorter", "--image_path", images, "--input_path", sparse,
             "--output_path", dense, "--output_type", "COLMAP"], "undistort", workspace)
        write_stereo_cfg(dense / "stereo" / "patch-match.cfg", frames, args.source_views)
        patch_args = [colmap, "patch_match_stereo", "--workspace_path", dense,
                      "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"]
        if args.max_image_size:
            patch_args += ["--PatchMatchStereo.max_image_size", str(args.max_image_size)]
        if args.depth_min > 0 and args.depth_max > 0:
            patch_args += ["--PatchMatchStereo.depth_min", str(args.depth_min),
                           "--PatchMatchStereo.depth_max", str(args.depth_max)]
        run(patch_args, "patch_match", workspace)
    depth_dir = dense / "stereo" / "depth_maps"
    normal_dir = dense / "stereo" / "normal_maps"
    cameras = []
    for index, frame in enumerate(frames):
        stem = frame["stem"]
        geometric = depth_dir / f"{stem}.png.geometric.bin"
        photometric = depth_dir / f"{stem}.png.photometric.bin"
        source = geometric if geometric.is_file() else photometric
        if not source.is_file():
            raise SystemExit(f"missing depth map for {stem}: {source}")
        depth = read_bin(source)[..., 0]
        depth[~np.isfinite(depth)] = 0
        depth[depth < 0] = 0
        np.save(out / "depth" / f"{stem}.npy", depth.astype(np.float32))
        normal = read_bin(normal_dir / f"{stem}.png.geometric.bin" if geometric.is_file()
                          else normal_dir / f"{stem}.png.photometric.bin")
        np.save(out / "normals" / f"{stem}.npy", normal.astype(np.float32))
        scale = 512 / frame["width"]
        cameras.append({"frame": f"{stem}.png", "file": f"{stem}.png",
                        "original_width": frame["width"], "original_height": frame["height"],
                        "model_width": frame["width"], "model_height": frame["height"],
                        "K_original": frame["K_original"], "K_model": frame["K_original"],
                        "T_world_to_cam": frame["T_world_to_cam"],
                        "pixel_affine_original_to_model": np.eye(3).tolist()})
    (out / "cameras.json").write_text(json.dumps(
        {"camera_model": "PINHOLE",
         "note": "COLMAP PatchMatch depth (geometric if present else photometric) on the aligned "
                 "predicted poses; z-depth, non-finite/negative zeroed",
         "poses": str(args.poses), "frames": cameras}, indent=2) + "\n")
    print(f"[ok] {len(cameras)} depth/normal maps -> {out}", flush=True)


if __name__ == "__main__":
    main()
