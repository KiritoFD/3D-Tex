"""Ideal and non-ideal calibration modes for the diagnostic M2 rig.

Non-ideal calibration injects pose and intrinsic error into the *algorithm's*
camera only. The world gauge is fixed first: predicted poses are registered to
the GT rig with one Sim(3) (the same call the ideal mode uses), and the error is
added afterwards in that gauge. GT cameras are never used to refit the
registration, otherwise a calibration error would be absorbed by the alignment.

Distributions: independent per view, a monotone drift across the view order,
or one shared bias for every view. Every distribution draws one unit field per
seed and only scales it, so different error levels share paired samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rotation_matrix(axis, angle_deg):
    angle = np.deg2rad(angle_deg)
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if angle == 0 or norm < 1e-12:
        return np.eye(3)
    unit = axis/norm
    cross = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross @ cross)


def camera_centers(poses):
    poses = np.asarray(poses, dtype=np.float64)
    return -np.einsum("sji,sj->si", poses[:, :3, :3], poses[:, :3, 3])


def align_poses(poses, gt_poses):
    """Ideal registration of predicted poses to the GT rig; returns the fixed gauge."""
    from verify_scene import umeyama
    poses, gt_poses = np.asarray(poses, dtype=np.float64), np.asarray(gt_poses, dtype=np.float64)
    scale, rotation, translation = umeyama(camera_centers(poses), camera_centers(gt_poses))
    aligned = poses.copy()
    aligned[:, :3, :3] = poses[:, :3, :3] @ rotation.T
    aligned[:, :3, 3] = scale*poses[:, :3, 3]-aligned[:, :3, :3] @ translation
    return aligned, {"scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist()}


def weights(count, distribution):
    if distribution == "independent":
        return np.ones(count)
    if distribution == "bias":
        return np.ones(count)
    if distribution == "drift":
        ramp = np.linspace(0., 2., count)
        return ramp/np.sqrt(np.mean(ramp**2))
    raise ValueError(f"Unsupported calibration distribution: {distribution}")


def unit_field(count, seed, distribution):
    """One direction shared by all views for bias/drift, independent directions otherwise."""
    rng = np.random.default_rng(seed)
    if distribution in ("bias", "drift"):
        return np.repeat(rng.normal(size=(1, 3)), count, 0)
    return rng.normal(size=(count, 3))


def normalize_units(units, weight, target):
    if target <= 0:
        return np.zeros_like(units)
    magnitude = np.sqrt(np.mean(np.sum((weight[:, None]*units)**2, 1)))
    return (weight[:, None]*units)*(target/max(magnitude, 1e-12))


def perturb_poses(poses, translation_mm, rotation_deg, distribution, seed, scale_mm):
    """Rotate about each camera centre and translate the centre, in the fixed GT gauge."""
    count = len(poses)
    translations = normalize_units(unit_field(count, seed, distribution), weights(count, distribution),
                                   translation_mm/scale_mm)
    rotations_deg = normalize_units(unit_field(count, seed+1, distribution), weights(count, distribution),
                                   rotation_deg)
    output, centers = np.asarray(poses, dtype=np.float64).copy(), []
    for index, pose in enumerate(poses):
        rotation_cam = rotation_matrix(rotations_deg[index], np.linalg.norm(rotations_deg[index]))
        inverse = np.linalg.inv(pose)
        center = inverse[:3, 3]+translations[index]
        axis = inverse[:3, :3] @ rotation_cam
        output[index] = np.linalg.inv(np.block([[axis, center[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        centers.append(center)
    measured_translation = float(np.sqrt(np.mean(np.sum((translations*scale_mm)**2, 1))) if count else 0.)
    measured_rotation = float(np.sqrt(np.mean(np.sum(rotations_deg**2, 1))))
    record = {"translation_rms_mm": measured_translation, "rotation_rms_deg": measured_rotation,
              "translation_max_mm": float(np.abs(translations).max()*scale_mm) if count else 0.,
              "rotation_max_deg": float(np.linalg.norm(rotations_deg, axis=1).max()) if count else 0.,
              "distribution": distribution, "seed": seed,
              "definition": "camera-centre translation (world) plus rotation about that centre, in the fixed GT gauge"}
    return output, record


def perturb_intrinsics(k, focal_relative=0., principal_px=0., focal_sign=1.):
    k = np.asarray(k, dtype=np.float64).copy()
    factor = 1.+focal_sign*focal_relative
    k[0, 0] *= factor
    k[1, 1] *= factor
    k[0, 2] += principal_px
    k[1, 2] += principal_px
    return k


def scalar_field(count, seed, distribution, target):
    """Per-view scalar error with a target RMS; one unit field, scaled only."""
    if target == 0:
        return np.zeros(count)
    rng = np.random.default_rng(seed)
    unit = (np.repeat(rng.normal(size=1), count) if distribution in ("bias", "drift")
            else rng.normal(size=count))
    weight = weights(count, distribution)
    magnitude = np.sqrt(np.mean((weight*unit)**2))
    return (weight*unit)*(target/max(magnitude, 1e-12))


def equivalent_shift_px(k, translation_mm, depth_mm):
    """Image-plane displacement of a camera translation at a given depth."""
    return translation_mm*k[0, 0]/depth_mm


def build(config, translation_mm=0., rotation_deg=0., distribution="independent", seed=0,
          focal_relative=0., principal_px=0., calibration_mode="non_ideal", intrinsics_mode="fixed"):
    rgb = Path(config["rgb_run"])
    resolution = config["observation_resolution"]
    predicted = json.loads((rgb/"geometry/omega/cameras.json").read_text())["frames"]
    gt_path = Path(config["master"])/"geometry"/f"gt_ray{resolution}"/"cameras.json"
    shell = {f["frame"]: f for f in json.loads(gt_path.read_text())["frames"]}
    poses = np.asarray([f["T_world_to_cam"] for f in predicted])
    gt_poses = np.asarray([shell[f["frame"]]["T_world_to_cam"] for f in predicted])
    ideal, registration = align_poses(poses, gt_poses)
    aligned = ideal.copy()
    scale = json.loads((Path(config["master"])/"scale.json").read_text())["mm_per_source_unit"]
    if calibration_mode == "ideal":
        if translation_mm or rotation_deg or focal_relative or principal_px:
            raise ValueError("ideal mode must not inject any error; both modes are never mixed")
        injected = {"translation_rms_mm": 0., "rotation_rms_deg": 0., "distribution": "none",
                    "definition": "no injected error; GT registered rig"}
    else:
        aligned, injected = perturb_poses(aligned, translation_mm, rotation_deg, distribution, seed, scale)
    count = len(predicted)
    if intrinsics_mode == "per_view":
        focal = scalar_field(count, seed+7, distribution, focal_relative)
        principal = scalar_field(count, seed+8, distribution, principal_px)
    elif intrinsics_mode == "fixed":
        focal = np.full(count, focal_relative)
        principal = np.full(count, principal_px)
    else:
        raise ValueError(f"Unsupported intrinsics mode: {intrinsics_mode}")
    frames = {}
    for index, (frame, pose, anchor) in enumerate(zip(predicted, aligned, ideal)):
        ideal_k = np.asarray(shell[frame["frame"]]["K_model"], dtype=np.float64)
        frames[Path(frame["frame"]).stem] = {
            "T_world_to_cam": np.asarray(pose).tolist(),
            "K_model": perturb_intrinsics(ideal_k, float(focal[index]), float(principal[index])).tolist(),
            "ideal_pose": np.asarray(anchor).tolist(), "ideal_K_model": ideal_k.tolist(),
            "focal_relative": float(focal[index]), "principal_px": float(principal[index])}
    return {"mode": calibration_mode, "registration": "locked",
            "registration_transform": registration, "injected": injected,
            "intrinsics": {"focal_relative": focal_relative, "principal_px": principal_px,
                           "model_width": int(config["model_resolution"]), "mode": intrinsics_mode,
                           "realized_focal_relative_rms": float(np.sqrt(np.mean(focal**2))),
                           "realized_principal_px_rms": float(np.sqrt(np.mean(principal**2))),
                           "realized_focal_relative_max": float(np.abs(focal).max()),
                           "realized_principal_px_max": float(np.abs(principal).max())},
            "frames": frames, "scale_mm_per_source_unit": scale,
            "note": "GT cameras are evaluation-only; the alignment is computed once in the ideal gauge"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/measurement_error_v2.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibration-mode", choices=["ideal", "non_ideal"], default="non_ideal")
    parser.add_argument("--translation-mm", type=float, default=0.)
    parser.add_argument("--rotation-deg", type=float, default=0.)
    parser.add_argument("--distribution", choices=["independent", "drift", "bias"], default="independent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal-relative", type=float, default=0.)
    parser.add_argument("--principal-px", type=float, default=0.)
    parser.add_argument("--intrinsics-mode", choices=["fixed", "per_view"], default="fixed")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    payload = build(config, args.translation_mm, args.rotation_deg, args.distribution, args.seed,
                    args.focal_relative, args.principal_px, args.calibration_mode, args.intrinsics_mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"mode": payload["mode"], "injected": payload["injected"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
