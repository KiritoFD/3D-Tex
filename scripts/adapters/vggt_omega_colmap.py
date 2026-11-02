#!/usr/bin/env python3
"""VGGT-Omega inference -> COLMAP sparse/, sized for small GPUs (and CPU).

Why an adapter instead of calling the upstream demo:
    * facebookresearch/vggt-omega ships only a Gradio demo (demo_gradio.py), which requires
      CUDA, keeps fp32 weights on the GPU (4.3 GiB for this 1.14B-parameter model) and always
      feeds every image it finds. None of that fits a 6-8 GiB laptop GPU.
    * This adapter reuses the upstream model/loader/pose-decode code verbatim and adds the
      three knobs we actually need: --device, --weight-dtype and the frame/resolution budget.
      It also converts the predictions into a COLMAP reconstruction using the vendored
      VGGT exporter (vggt/dependency/np_to_pycolmap.py), so both downstream branches
      (gsplat 3DGS) keep consuming the same sparse/ contract as before.

    python3 scripts/adapters/vggt_omega_colmap.py --scene-dir data/scenes/eth3d_pipes \
        --checkpoint data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt \
        --device cuda --weight-dtype bfloat16 --max-frames 8 --image-resolution 512

Outputs (under <scene-dir>/sparse by default):
    cameras.bin, images.bin, points3D.bin, points.ply, vggt_omega_report.json
Held-out views (scene/heldout/) and ground_truth/ are never read here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

# --- vendored upstream packages: vggt_omega (model) and vggt (COLMAP exporter) -----------
ROOT = Path(__file__).resolve().parents[2]
for pkg in (ROOT / "vggt_omega", ROOT / "vggt"):
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT-Omega -> COLMAP sparse export (low-VRAM aware)")
    p.add_argument("--scene-dir", required=True, help="Scene folder containing images/")
    p.add_argument("--checkpoint", required=True, help="Local VGGT-Omega .pt path")
    p.add_argument("--sparse-dir", default=None, help="Output dir (default <scene-dir>/sparse)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--aggregator-dtype", choices=["auto", "fp32", "bfloat16"], default="auto",
                   help="backbone precision; the heads stay fp32 because upstream promotes "
                        "tokens to fp32 inside camera_head/dense_head. bfloat16 halves the "
                        "1.1B-param backbone on the GPU (4.3 GiB -> 2.2 GiB total)")
    p.add_argument("--max-frames", type=int, default=0, help="Evenly subsample to N frames (0 = all)")
    p.add_argument("--image-resolution", type=int, default=512,
                   help="Token budget side; must be divisible by patch 16 (e.g. 512, 384, 256)")
    p.add_argument("--conf-threshold", type=float, default=1.0,
                   help="Depth confidence cutoff for kept points (upstream demo uses 1.0)")
    p.add_argument("--max-points", type=int, default=100000, help="Cap on points3D written")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Resolve inputs and print the plan only")
    return p.parse_args()


def pick_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch reports no CUDA device")
    return preference


def pick_aggregator_dtype(preference: str, device: str) -> "torch.dtype":
    if preference == "fp32":
        return torch.float32
    if preference == "bfloat16":
        return torch.bfloat16
    if device != "cuda":
        return torch.float32
    free, _ = torch.cuda.mem_get_info()
    # Heads are always fp32 (upstream forces it); the backbone is the part worth shrinking.
    # fp32 backbone needs ~4.3 GiB resident, bf16 ~2.2 GiB. Keep fp32 only when it clearly fits.
    return torch.float32 if free > 6.5 * 2**30 else torch.bfloat16


def subsample(paths: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(paths) <= max_frames:
        return paths
    if max_frames < 2:
        raise SystemExit("--max-frames must be 0 or >= 2")
    idx = [round(i * (len(paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [paths[i] for i in sorted(set(idx))]


def main() -> None:
    args = parse_args()
    scene = Path(args.scene_dir).expanduser().resolve()
    image_dir = scene / "images"
    sparse = Path(args.sparse_dir).resolve() if args.sparse_dir else scene / "sparse"

    if not image_dir.is_dir():
        raise SystemExit(f"No images dir: {image_dir}")
    paths = sorted(glob.glob(os.path.join(str(image_dir), "*")))
    paths = [x for x in paths if os.path.splitext(x)[1].lower() in (".jpg", ".jpeg", ".png")]
    if len(paths) < 2:
        raise SystemExit(f"Need >= 2 images in {image_dir}, found {len(paths)}")
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint missing: {ckpt}\n"
                         f"Fetch it with: python3 scripts/fetch_weights.py --only vggt-omega-512")
    if args.image_resolution % 16:
        raise SystemExit("--image-resolution must be divisible by 16 (VGGT-Omega patch size)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device)
    agg_dtype = pick_aggregator_dtype(args.aggregator_dtype, device)
    frames = subsample(paths, args.max_frames)
    print(f"[plan] device={device} aggregator_dtype={agg_dtype} heads=fp32 "
          f"frames={len(frames)}/{len(paths)} resolution={args.image_resolution} "
          f"conf>={args.conf_threshold} max_points={args.max_points}")
    print(f"[plan] checkpoint={ckpt.name} ({ckpt.stat().st_size / 2**30:.2f} GiB)")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[plan] VRAM free {free / 2**30:.2f} / {total / 2**30:.2f} GiB before start")
    if args.dry_run:
        return

    # --- upstream code, untouched ---------------------------------------------------------
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

    t0 = time.time()
    # Upstream contract: fp32 weights everywhere; the aggregator runs under autocast(bf16) and
    # its heads then promote tokens back to fp32 (camera_head.py:61, dense_head.py:127) and run
    # with autocast disabled. So we must NOT cast the heads. Casting only the aggregator to
    # bf16 - with autocast off, so its own LayerNorms stay bf16 and the heads' .float() calls
    # do the promoting - is what actually halves resident weight memory on a small GPU.
    backbone_low_memory = agg_dtype != torch.float32
    use_upstream_autocast = (device == "cuda") and not backbone_low_memory
    print(f"[load] VGGTOmega(autocast={'on' if use_upstream_autocast else 'off'}) "
          f"+ backbone {agg_dtype} / heads fp32 ...")
    model = VGGTOmega(autocast=use_upstream_autocast).eval()
    # weights_only=True is not just hygiene: torch's default pickle path mmaps the zip archive,
    # and mmap fails on WSL's DrvFs (/mnt/...), which is where this repo lives. This way the
    # checkpoint stays a single copy on the Windows disk.
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model = model.to(device=device)
    if backbone_low_memory:
        model.aggregator.to(dtype=agg_dtype)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"[load] done in {time.time() - t0:.1f}s")

    images = load_and_preprocess_images(frames, image_resolution=args.image_resolution)
    images = images.to(device=device)
    if backbone_low_memory:
        # The backbone is bf16 with autocast off, so its input must already be bf16; the heads
        # receive the fp32 copy they expect via their own .float() promotion.
        images_for_backbone = images.to(dtype=agg_dtype)
    else:
        images_for_backbone = images
    print(f"[input] images {tuple(images.shape)} backbone_input {tuple(images_for_backbone.shape)}")

    try:
        # VGGTOmega.forward() applies the upstream autocast itself; wrapping it again would
        # silently change what the heads see, so we only add inference_mode here.
        with torch.inference_mode():
            predictions = model(images_for_backbone)
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            f"CUDA out of memory during inference ({exc}).\n"
            "Memory ladder, cheapest first: --image-resolution 192/128, --max-frames 4/3, "
            "--aggregator-dtype bfloat16 (halves the backbone weights), or --device cpu.") from exc

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])

    def drop_batch(value):
        """Upstream heads emit (1, S, ...); the numpy exporters want the (S, ...) view."""
        array = value.detach().float().cpu().numpy()
        return array[0] if array.ndim >= 4 and array.shape[0] == 1 else array

    pred = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            pred[key] = drop_batch(value)
    pred["extrinsic"] = drop_batch(extrinsic)
    pred["intrinsic"] = drop_batch(intrinsic)

    # Upstream unproject_depth_map_to_point_map() indexes depth_map[f].squeeze(-1), so it wants
    # the (S, H, W, 1) layout back; keep the trailing channel and take H/W from it.
    depth_full = pred["depth"]
    num_frames, height, width = depth_full.shape[:3]
    conf = pred["depth_conf"][..., 0] if pred["depth_conf"].ndim == 4 else pred["depth_conf"]
    world_points = unproject_depth_map_to_point_map(depth_full, pred["extrinsic"], pred["intrinsic"])

    # --- build the point set in the model frame, then rescale cameras to original pixels ---
    mask = np.isfinite(conf) & (conf >= args.conf_threshold) & np.isfinite(world_points).all(axis=-1)
    keep = randomly_limit_trues(mask, args.max_points)
    points_3d = world_points[keep]
    frames_rgb = np.transpose(pred["images"], (0, 2, 3, 1))[:num_frames]
    rgb = (frames_rgb * 255).clip(0, 255).astype(np.uint8)[keep]
    xyf = create_pixel_coordinate_grid(num_frames, height, width)[keep]
    print(f"[points] kept {len(points_3d)} of {int(mask.sum())} confident pixels")

    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d.astype(np.float32), xyf.astype(np.float32), rgb,
        pred["extrinsic"].astype(np.float64), pred["intrinsic"].astype(np.float64),
        np.array([width, height]), shared_camera=False, camera_type="PINHOLE")

    original_sizes = [read_size(path) for path in frames]
    rename_and_rescale(reconstruction, [os.path.basename(p) for p in frames], original_sizes,
                       (width, height))

    sparse.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse))
    try:
        import trimesh
        trimesh.PointCloud(points_3d, colors=rgb).export(str(sparse / "points.ply"))
    except Exception as exc:  # ply is a convenience artifact, never a gate
        print(f"[warn] points.ply skipped: {type(exc).__name__}: {exc}")

    peak = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0
    report = {
        "backend": "vggt-omega", "checkpoint": str(ckpt), "device": device,
        "aggregator_dtype": str(agg_dtype), "head_dtype": "float32",
        "image_resolution": args.image_resolution,
        "model_input_hw": [int(height), int(width)],
        "frames_used": [os.path.basename(p) for p in frames],
        "frames_total_in_scene": len(paths), "conf_threshold": args.conf_threshold,
        "points_written": int(len(points_3d)), "seconds": round(time.time() - t0, 1),
        "peak_cuda_mem_mib": round(peak, 1), "sparse_dir": str(sparse),
        "cameras": {os.path.basename(p): list(s) for p, s in zip(frames, original_sizes)},
        "note": "Feed-only: heldout/ and ground_truth/ were not read.",
    }
    (sparse / "vggt_omega_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[ok] sparse written to {sparse} | {len(reconstruction.images)} images, "
          f"{len(points_3d)} points | peak CUDA {peak:.0f} MiB | {report['seconds']}s")


# --------------------------------------------------------------------- export helpers ---
def read_size(path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (width, height)


def rename_and_rescale(reconstruction, names: list[str], sizes: list[tuple[int, int]],
                       model_wh: tuple[int, int]) -> None:
    """Map model-resolution cameras back to the real image size and real filenames."""
    mw, mh = model_wh
    for image_id, pyimage in reconstruction.images.items():
        camera = reconstruction.cameras[pyimage.camera_id]
        w, h = sizes[image_id - 1]
        params = camera.params.copy()
        # PINHOLE params: fx, fy, cx, cy -- scale per axis, principal point to the true centre.
        params[0] *= w / mw
        params[1] *= h / mh
        params[2] = w / 2.0
        params[3] = h / 2.0
        camera.params = params
        camera.width, camera.height = int(w), int(h)
        pyimage.name = names[image_id - 1]


if __name__ == "__main__":
    main()
