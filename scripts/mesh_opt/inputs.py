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


def load_views(args, device):
    omega = Path(args.omega)
    frames = json.loads((omega / "cameras.json").read_text())["frames"]
    poses = np.asarray([f["T_world_to_cam"] for f in frames])
    scale, rotation, translation = 1., np.eye(3), np.zeros(3)
    gt = None
    if args.camera_mode == "diagnostic":
        from verify_scene import umeyama
        gt = {Path(f["frame"]).stem: f for f in json.loads(
            (Path(args.gt) / "cameras.json").read_text())["frames"]}
        gts = np.asarray([gt[Path(f["frame"]).stem]["T_world_to_cam"] for f in frames])
        pc = -np.einsum("sji,sj->si", poses[:, :3, :3], poses[:, :3, 3])
        gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
        scale, rotation, translation = umeyama(pc, gc)
        poses[:, :3, :3] = poses[:, :3, :3] @ rotation.T
        poses[:, :3, 3] = scale * poses[:, :3, 3] - poses[:, :3, :3] @ translation
    elif args.mask_dir is None and args.images is None:
        raise ValueError("prediction mode requires --mask-dir or RGBA --images; no GT fallback")
    views = []
    for frame, pose in zip(frames, poses):
        stem = Path(frame["frame"]).stem
        pk = np.asarray(frame["K_model"])
        k = np.asarray(gt[stem]["K_model"] if gt else pk).copy()
        width = gt[stem].get("model_width", 512) if gt else frame["model_width"]
        height = gt[stem].get("model_height", 512) if gt else frame["model_height"]
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
        views.append({"name": stem, "k": torch.tensor(k, device=device, dtype=torch.float32),
                      "t": torch.tensor(pose, device=device, dtype=torch.float32),
                      "fg": torch.tensor(fg, device=device),
                      "depth": torch.tensor(depth, device=device),
                      "confidence": torch.tensor(conf, device=device)})
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
                   "mask_source": "source-GT foreground" if gt else
                                  ("external masks" if args.mask_dir else "dataset RGBA alpha"),
                   "external_scan": args.external_depth if getattr(args, "lambda_external", 0) else None}
