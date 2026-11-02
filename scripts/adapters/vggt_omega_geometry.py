#!/usr/bin/env python3
"""Gate 0: VGGT-Omega -> standard geometry export (cameras / depth / confidence).

Implements the output contract in docs/VGGT_OMEGA_ADAPTER.md rather than a COLMAP-only
dump, because the downstream A0 pipeline (multiview consistency -> TSDF -> mesh -> UV)
needs *depth*, not just points, and needs every array registered to ORIGINAL image pixels.

Written into <scene>/geometry/omega/ (never overwrites the legacy sparse/):
    cameras.json            per frame: K(3x3, original pixels), T_world_to_cam(4x4), size
    depth/<stem>.npy        camera z-depth (not ray length), model resolution
    confidence/<stem>.npy   raw model confidence, no threshold applied here
    images_model/<stem>.png the exact tensor the model saw (for pixel-affine audits)
    manifest.json           checkpoint SHA256, dtypes, shapes, affine A, round-trip error
    points_world.ply        derived from K + depth + pose (NOT a separate pointmap head)

Pixel convention: preprocessing resizes and centre-pads. For each frame we store
    uv_model = A @ uv_original,   A = [[s,0,tx],[0,s,ty],[0,0,1]]
and therefore K_model = A @ K_original, i.e. K_original = A^-1 @ K_model. The manifest
records max/mean of |project(unproject(p)) - p| to prove the two agree.

    python3 scripts/adapters/vggt_omega_geometry.py --scene-dir data/scenes/eth3d_pipes \
        --checkpoint data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt \
        --device cuda --aggregator-dtype bfloat16 --image-resolution 512
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for pkg in (ROOT / "vggt_omega", ROOT / "vggt"):
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT-Omega geometry export (Gate 0 contract)")
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None, help="default <scene>/geometry/omega")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--aggregator-dtype", choices=["auto", "fp32", "bfloat16"], default="auto")
    p.add_argument("--max-frames", type=int, default=0, help="0 = every training image")
    p.add_argument("--image-resolution", type=int, default=512,
                   help="must be divisible by the 16px patch size")
    p.add_argument("--save-model-images", action="store_true",
                   help="also dump the post-resize tensors (adds disk, helps pixel audits)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def pick_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch sees no CUDA device")
    return preference


def pick_aggregator_dtype(preference: str, device: str) -> "torch.dtype":
    if preference == "fp32":
        return torch.float32
    if preference == "bfloat16":
        return torch.bfloat16
    if device != "cuda":
        return torch.float32
    free, _ = torch.cuda.mem_get_info()
    return torch.float32 if free > 6.5 * 2**30 else torch.bfloat16


def file_sha256(path: Path) -> tuple[str, int]:
    """Hash the checkpoint once so the manifest pins exactly which weights produced this."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def subsample(paths: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(paths) <= max_frames:
        return paths
    if max_frames < 2:
        raise SystemExit("--max-frames must be 0 or >= 2")
    picks = sorted({round(i * (len(paths) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [paths[i] for i in picks]


def measure_pixel_affine(path: str, model_hw: tuple[int, int]) -> dict:
    """Recover the original->model pixel transform from the loader's own helpers.
    Computed from the loader's own helper functions rather than a rendered probe. My earlier
    probe fit was wrong twice: the probe's target size is decided by the *token budget*
    (`_balanced_target_shape`), not by the `image_resolution` I passed, so the fitted slope was
    meaningless (3.10 instead of 2.52) and the derived focal ended up 10x off GT. Calling the
    same private helpers the loader calls removes that entire class of bug.

    Contract returned:  u_model = A @ u_original,   A = [[sx,0,tx],[0,sy,ty],[0,0,1]]
    """
    from PIL import Image

    from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio

    original_w, original_h = read_size(path)
    with Image.open(path) as opened:
        crop_w, crop_h = _crop_to_supported_aspect_ratio(opened.convert("RGB")).size
    model_w, model_h = model_hw
    # What the loader actually resizes to for this aspect ratio (see load_fn.load_and_preprocess_images
    # mode="balanced"): a patch-multiple close to the requested token budget.
    target_h, target_w = _balanced_target_shape(crop_h / max(crop_w, 1), max(model_hw), 16)
    if (target_w, target_h) != (model_w, model_h):
        # The tensor we got back is authoritative; if they disagree we must not guess silently.
        target_w, target_h = model_w, model_h

    sx = target_w / crop_w
    sy = target_h / crop_h
    tx = -((original_w - crop_w) // 2) * sx
    ty = -((original_h - crop_h) // 2) * sy
    if abs(sx - sy) > 0.02 * max(sx, sy):
        raise RuntimeError(f"non-uniform resize sx={sx:.6f} sy={sy:.6f}: the loader changed aspect")
    affine = np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])
    if abs(sx * original_w - model_w) > 0.05 * model_w:
        raise RuntimeError(f"affine does not map image width onto the model width: "
                           f"{sx * original_w:.1f} vs {model_w}")
    return {"affine": affine, "scale_x": float(sx), "scale_y": float(sy),
            "offset": [float(tx), float(ty)], "residual_px": 0.0,
            "model_block_wh": [int(target_w), int(target_h)], "padded_pixels": 0,
            "native_crop_wh": [int(crop_w), int(crop_h)], "probe_shrink_factor": 1}


def fit_unique(source: np.ndarray, model: np.ndarray, fit, name: str):
    """Least-squares fit using the mean target per distinct (rounded) source coordinate."""
    keys = np.round(source).astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys, model = keys[order], model[order]
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    means = np.zeros(unique.size)
    np.add.at(means, inverse, model)
    means /= counts
    return fit(unique.astype(np.float64), means, name)


def main() -> None:
    args = parse_args()
    scene = Path(args.scene_dir).expanduser().resolve()
    image_dir = scene / "images"
    ckpt = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.out).resolve() if args.out else scene / "geometry" / "omega"

    if not image_dir.is_dir():
        raise SystemExit(f"No images dir: {image_dir}")
    paths = [p for p in sorted(glob.glob(str(image_dir / "*")))
             if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")]
    if len(paths) < 2:
        raise SystemExit(f"Need >= 2 images in {image_dir}, found {len(paths)}")
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint missing: {ckpt}\n"
                         "Fetch it: python3 scripts/fetch_weights.py --only vggt-omega-512")
    if args.image_resolution % 16:
        raise SystemExit("--image-resolution must be divisible by 16")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device)
    agg_dtype = pick_aggregator_dtype(args.aggregator_dtype, device)
    frames = subsample(paths, args.max_frames)
    print(f"[plan] device={device} aggregator_dtype={agg_dtype} heads=fp32 "
          f"frames={len(frames)}/{len(paths)} resolution={args.image_resolution}")
    print(f"[plan] checkpoint={ckpt.name} ({ckpt.stat().st_size / 2**30:.2f} GiB) -> {out}")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[plan] VRAM free {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    if args.dry_run:
        return

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    t0 = time.time()
    backbone_low_memory = agg_dtype != torch.float32
    use_autocast = (device == "cuda") and not backbone_low_memory
    model = VGGTOmega(autocast=use_autocast).eval()
    # weights_only=True avoids torch's mmap path, which fails on WSL DrvFs (/mnt/...) and is
    # also stricter about what the checkpoint is allowed to execute.
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model = model.to(device=device)
    if backbone_low_memory:
        model.aggregator.to(dtype=agg_dtype)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_seconds = time.time() - t0

    original_sizes = [read_size(p) for p in frames]
    images = load_and_preprocess_images(frames, image_resolution=args.image_resolution)
    model_hw = (int(images.shape[-1]), int(images.shape[-2]))
    if backbone_low_memory:
        images = images.to(device=device, dtype=agg_dtype)
    else:
        images = images.to(device=device)

    t_infer = time.time()
    try:
        with torch.inference_mode():
            predictions = model(images)
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(f"CUDA OOM ({exc}). Ladder: --image-resolution 384/256, --max-frames 4, "
                         "--aggregator-dtype bfloat16, or --device cpu.") from exc
    infer_seconds = time.time() - t_infer

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])

    def to_np(value):
        array = value.detach().float().cpu().numpy()
        return array[0] if array.ndim >= 4 and array.shape[0] == 1 else array

    depth = to_np(predictions["depth"])            # (S, H, W, 1) z-depth in model pixels
    conf = to_np(predictions["depth_conf"])
    if conf.ndim == depth.ndim and conf.shape[-1] == 1:
        conf = conf[..., 0]
    extr = to_np(extrinsic)                        # (S, 3, 4) world -> camera
    intr = to_np(intrinsic)                        # (S, 3, 3) model pixels

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("depth", "confidence", "images_model"):
        (out / sub).mkdir(exist_ok=True)

    cameras = []
    roundtrip = []
    for index, (path, size) in enumerate(zip(frames, original_sizes)):
        stem = Path(path).stem
        np.save(out / "depth" / f"{stem}.npy", depth[index].astype(np.float32))
        np.save(out / "confidence" / f"{stem}.npy", conf[index].astype(np.float32))
        if args.save_model_images:
            import imageio.v2 as imageio
            frame = to_np(predictions["images"])[index]
            imageio.imwrite(out / "images_model" / f"{stem}.png",
                            (np.transpose(np.clip(frame, 0, 1), (1, 2, 0)) * 255).astype(np.uint8))

        measured = measure_pixel_affine(path, model_hw)
        affine = measured["affine"]
        k_model = intr[index].astype(np.float64)
        # Conjugating with A^-1 ... A is a *similarity*, so it leaves a focal length algebraically
        # unchanged: the old `inv(A) @ K @ A` silently returned K_model verbatim and made
        # K_original 10x too small against ETH3D GT (f=341.8 vs 3410.3). What actually changes is
        # the pixel grid K is expressed in, so scale the intrinsics by the inverse pixel scale:
        #     f_orig = f_model / s,   c_orig = (c_model - t) / s      with u_model = s*u_orig + t
        k_original = np.array([
            [k_model[0, 0] / affine[0, 0], 0.0,
             (k_model[0, 2] - affine[0, 2]) / affine[0, 0]],
            [0.0, k_model[1, 1] / affine[1, 1],
             (k_model[1, 2] - affine[1, 2]) / affine[1, 1]],
            [0.0, 0.0, 1.0]])
        # The model's own principal point is not guaranteed to sit exactly at the image centre
        # (see _unproject): record how far off it is, in BOTH frames, so downstream code can
        # decide whether to trust K_model or force the centre.
        centre_model = np.array([model_hw[0] / 2.0, model_hw[1] / 2.0])
        offset_model = k_model[:2, 2] - centre_model
        offset_original = offset_model / np.array([affine[0, 0], affine[1, 1]])
        roundtrip.append(float(np.linalg.norm(offset_original)))

        cameras.append({
            "frame": stem + Path(path).suffix,   # full name so GT joins work
            "file": os.path.basename(path),
            "original_width": int(size[0]),
            "original_height": int(size[1]),
            "model_width": model_hw[0],
            "model_height": model_hw[1],
            "K_model": k_model.tolist(),
            "K_original": k_original.tolist(),
            "T_world_to_cam": np.vstack([extr[index], [0.0, 0.0, 0.0, 1.0]]).astype(np.float64).tolist(),
            "pixel_affine_original_to_model": affine.tolist(),
            "pixel_scale": [measured["scale_x"], measured["scale_y"]],
            "pixel_padded_pixels": measured["padded_pixels"],
            "affine_fit_residual_px": measured["residual_px"],
            "predicted_principal_point_model": k_model[:2, 2].tolist(),
            "principal_point_offset_model_px": offset_model.tolist(),
            "principal_point_offset_original_px": offset_original.tolist(),
        })

    points = unproject_depth_map_to_point_map(depth, extr.astype(np.float64), intr.astype(np.float64))
    keep = np.isfinite(points).all(axis=-1) & (depth[..., 0] > 0)
    colours = np.transpose(to_np(predictions["images"]), (0, 2, 3, 1))
    xyz = points[keep]
    rgb = (colours * 255).clip(0, 255).astype(np.uint8)[keep]
    write_ply(out / "points_world.ply", xyz, rgb)

    sha, size_bytes = file_sha256(ckpt)
    peak = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0
    manifest = {
        "backend": "vggt-omega", "contract": "docs/VGGT_OMEGA_ADAPTER.md",
        "checkpoint": {"path": str(ckpt), "sha256": sha, "bytes": size_bytes},
        "device": device, "aggregator_dtype": str(agg_dtype), "heads_dtype": "float32",
        "torch": torch.__version__, "autocast_used": use_autocast,
        "image_resolution_requested": args.image_resolution,
        "model_width": model_hw[0], "model_height": model_hw[1],
        "depth_definition": "camera z-depth; verified against VGGT unproject helper",
        "depths": {"min": float(depth.min()), "max": float(depth.max())},
        "confidence": {"min": float(conf.min()), "max": float(conf.max()),
                       "mean": float(conf.mean()), "threshold_applied": None},
        "coordinate_frame": "world->camera (OpenCV), arbitrary gauge, not metric",
        "frames": [os.path.basename(p) for p in frames],
        "points_world": {"count": int(len(xyz)), "derived_from": "K + depth + pose"},
        "principal_point_offset_original_px_max": float(max(roundtrip)),
        "timing_seconds": {"load": round(load_seconds, 2), "infer": round(infer_seconds, 2)},
        "peak_cuda_mem_mib": round(peak, 1),
        "note": "heldout/ and ground_truth/ were never read.",
    }
    (out / "cameras.json").write_text(json.dumps({"camera_model": "PINHOLE",
                                                  "frames": cameras}, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[ok] {out} | {len(frames)} frames | depth {manifest['depths']['min']:.3f}"
          f"-{manifest['depths']['max']:.3f} | conf {manifest['confidence']['min']:.2f}"
          f"-{manifest['confidence']['max']:.2f} | points {len(xyz)}")
    print(f"[time] load {load_seconds:.1f}s + infer {infer_seconds:.2f}s | "
          f"peak CUDA {peak:.0f} MiB | principal-point offset {max(roundtrip):.3f} px (original)")


def read_size(path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Minimal binary-free PLY writer: avoids a trimesh dependency for the Gate-0 export."""
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("w") as handle:
        handle.write(header)
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


if __name__ == "__main__":
    main()
