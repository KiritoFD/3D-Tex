"""Explicit diagnostic/prediction camera and mask contracts for M2."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def remap(array, source_k, target_k, resolution):
    yy, xx = np.indices((resolution, resolution), dtype=np.float64)
    u = np.rint((xx - target_k[0, 2]) / target_k[0, 0] * source_k[0, 0] + source_k[0, 2]).astype(int)
    v = np.rint((yy - target_k[1, 2]) / target_k[1, 1] * source_k[1, 1] + source_k[1, 2]).astype(int)
    valid = (u >= 0) & (u < array.shape[1]) & (v >= 0) & (v < array.shape[0])
    result = np.zeros((resolution, resolution), dtype=np.float32)
    result[valid] = array[v[valid], u[valid]]
    return result


def depth_normals(depth, k, fg, jump=0.05):
    """Camera-frame normals from a z-depth map; robust to depth discontinuities."""
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    point = np.stack([(xx - k[0, 2]) / k[0, 0] * depth,
                      (yy - k[1, 2]) / k[1, 1] * depth, depth], -1)
    dx = np.zeros_like(point)
    dy = np.zeros_like(point)
    dx[:, :-2] = point[:, 2:] - point[:, :-2]
    dy[:-2, :] = point[2:, :] - point[:-2, :]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=-1)
    valid = fg & (depth > 0) & (length > 1e-9)
    valid &= (np.abs(np.roll(depth, -2, axis=1) - depth) <= jump * np.maximum(depth, 1e-6))
    valid &= (np.abs(np.roll(depth, -2, axis=0) - depth) <= jump * np.maximum(depth, 1e-6))
    normal[~valid] = 0
    normal /= np.maximum(length, 1e-12)[..., None]
    flip = (normal * point).sum(-1) > 0
    normal[flip] *= -1
    return normal.astype(np.float32), valid


def load_views(args, device):
    omega = Path(args.omega)
    frames = json.loads((omega / "cameras.json").read_text())["frames"]
    if getattr(args, "max_views", 0):
        take = np.unique(np.linspace(0, len(frames) - 1, args.max_views).round().astype(int))
        frames = [frames[i] for i in take]
    poses = np.asarray([f["T_world_to_cam"] for f in frames])
    scale, rotation, translation = 1., np.eye(3), np.zeros(3)
    gt = None
    calibration_record = None
    if getattr(args, "calibration", None):
        if args.camera_mode != "diagnostic":
            raise ValueError("non-ideal calibration requires the diagnostic rig; modes are never mixed")
        calibration_record = json.loads(Path(args.calibration).read_text())
        if calibration_record["mode"] != "non_ideal" or calibration_record["registration"] != "locked":
            raise ValueError(f"unsupported calibration record: {calibration_record['mode']}/{calibration_record['registration']}")
    if args.camera_mode == "diagnostic":
        from measurement_error.calibration import align_poses
        gt = {Path(f["frame"]).stem: f for f in json.loads(
            (Path(args.gt) / "cameras.json").read_text())["frames"]}
        gts = np.asarray([gt[Path(f["frame"]).stem]["T_world_to_cam"] for f in frames])
        # One ideal Sim(3) fixes the world gauge. Injected calibration error is added
        # afterwards in that gauge and is never re-fitted against the GT cameras.
        poses, registration = align_poses(poses, gts)
        scale, rotation, translation = registration["scale"], np.asarray(registration["rotation"]), np.asarray(registration["translation"])
    elif args.mask_dir is None and args.images is None:
        raise ValueError("prediction mode requires --mask-dir or RGBA --images; no GT fallback")
    views = []
    for frame, pose in zip(frames, poses):
        stem = Path(frame["frame"]).stem
        pk = np.asarray(frame["K_model"])
        k = np.asarray(gt[stem]["K_model"] if gt else pk).copy()
        width = gt[stem].get("model_width", 512) if gt else frame["model_width"]
        height = gt[stem].get("model_height", 512) if gt else frame["model_height"]
        if calibration_record:
            pose = np.asarray(calibration_record["frames"][stem]["T_world_to_cam"])
            k = np.asarray(calibration_record["frames"][stem]["K_model"]).copy()
            width = height = calibration_record["intrinsics"]["model_width"]
        k[0] *= args.resolution / width
        k[1] *= args.resolution / height
        if gt:
            mask = np.load(Path(args.gt) / "foreground" / f"{stem}.npy")
            fg = remap(mask, np.asarray(gt[stem]["K_model"]), k, args.resolution) > .5
        else:
            path = Path(args.mask_dir or args.images) / f"{stem}.png"
            with Image.open(path) as image:
                if args.mask_dir:
                    mask = np.asarray(image.convert("L")) > 127
                else:
                    if "A" not in image.getbands():
                        raise ValueError(f"missing alpha in {path}; supply an external mask")
                    mask = np.asarray(image.getchannel("A")) > 127
            # External masks use original-image coordinates, same contract as RGBA.
            affine = np.asarray(frame["pixel_affine_original_to_model"])
            yy, xx = np.indices((args.resolution, args.resolution), dtype=np.float64)
            model_x = xx * frame["model_width"] / args.resolution
            model_y = yy * frame["model_height"] / args.resolution
            coords = np.stack([model_x, model_y, np.ones_like(xx)], -1) @ np.linalg.inv(affine).T
            u, v = np.rint(coords[..., 0]).astype(int), np.rint(coords[..., 1]).astype(int)
            inside = (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
            fg = np.zeros_like(inside)
            fg[inside] = mask[v[inside], u[inside]]
        depth = remap(np.load(omega / "depth" / f"{stem}.npy").squeeze(), pk, k,
                      args.resolution) * scale
        conf = remap(np.load(omega / "confidence" / f"{stem}.npy").squeeze(), pk, k,
                     args.resolution)
        positive = conf[conf > 0]
        conf = np.clip(conf / max(float(np.percentile(positive, 95)), 1e-6), 0, 1)
        view = {"name": stem, "k": torch.tensor(k, device=device, dtype=torch.float32),
                "t": torch.tensor(pose, device=device, dtype=torch.float32),
                "fg": torch.tensor(fg, device=device),
                "depth": torch.tensor(depth, device=device),
                "confidence": torch.tensor(conf, device=device)}
        if getattr(args, "lambda_normal", 0):
            normal, nvalid = depth_normals(depth, k, fg)
            view["normal"] = torch.tensor(normal, device=device, dtype=torch.float32)
            view["normal_valid"] = torch.tensor(nvalid, device=device)
        views.append(view)
        if getattr(args, "lambda_external", 0):
            if not gt:
                raise ValueError("external scan pilot requires explicit diagnostic calibrated rig")
            external = Path(args.external_depth)
            calibration = {Path(f["frame"]).stem: f for f in json.loads(
                (external/"cameras.json").read_text())["frames"]}
            np.testing.assert_allclose(calibration[stem]["T_world_to_cam"], gt[stem]["T_world_to_cam"], atol=1e-7)
            source_k = np.asarray(calibration[stem]["K_model"])
            raw = np.load(external/"depth"/f"{stem}.npy")
            target = remap(raw, source_k, k, args.resolution)
            views[-1]["external"] = torch.tensor(target, device=device)
        if args.lambda_colmap:
            if not gt:
                raise ValueError("current COLMAP anchors use GT intrinsics; diagnostic mode only")
            from gated_fusion import read_bin
            anchors = {f["stem"]: f for f in json.loads(Path(args.anchor_poses).read_text())["frames"]}
            anchor = anchors[stem]
            np.testing.assert_allclose(anchor["T_world_to_cam"], pose, atol=1e-5)
            raw = read_bin(Path(args.colmap) / f"{stem}.png.geometric.bin")[..., 0]
            target = remap(raw, np.asarray(anchor["K_original"]), k, args.resolution)
            # Geometric-consistency output already gates support; reject non-finite,
            # silhouette pixels, and large disagreements with the initial Omega prior.
            valid = np.isfinite(target) & (target > 0) & fg & (depth > 0)
            valid &= np.abs(target - depth) <= args.anchor_gate_voxels * args.voxel_size
            target[~valid] = 0
            views[-1]["anchor"] = torch.tensor(target, device=device)
    return views, {"scale": float(scale), "rotation": rotation.tolist(),
                   "translation": translation.tolist(), "gt_assisted": gt is not None,
                   "calibration_mode": calibration_record["mode"] if calibration_record else "ideal",
                   "calibration_injected": calibration_record["injected"] if calibration_record else None,
                   "mask_source": "source-GT foreground" if gt else
                                  ("external masks" if args.mask_dir else "dataset RGBA alpha"),
                   "external_scan": args.external_depth if getattr(args, "lambda_external", 0) else None}
