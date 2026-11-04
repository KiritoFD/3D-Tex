#!/usr/bin/env python3
"""Level 3 controlled run: FlexiCubes (nvdiffrec) from the VGGT mesh.

The VGGT mesh is converted to a signed-distance grid (nearest-surface proxy),
then a FlexiCubes surface is optimised with image supervision:

    L = RGB(foreground) + w_sil * symmetric silhouette
        + w_corr * sparse multi-view correspondence
        + w_dev * FlexiCubes L_dev

Appearance is a learnable 3D colour grid sampled at the dual vertices plus the
degree-2 SH diffuse/specular model from Step 0.  Cameras are fixed GT cameras;
poses and intrinsics are never re-estimated.

Level 3 requires at least two seeds for a repeatability noise floor.  The
script writes one output folder per run and a combined summary.

Run from the repository root:

    PYTHONPATH=scripts /opt/conda/envs/3dtex/bin/python \
        scripts/experiments/step3_flexicubes.py --out outputs/step_baseline_20261105 \
        --run full_sA --seed 20261105 --corr-weight 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/root/deps/nvdiffrec")
import step0_objective as s0
import step0_1_diagnostics as s01
from mesh_opt.geometry import vertex_normals
from geometry.flexicubes import FlexiCubes
from tsdf_mesh import write_ply

SH_KEYS = ("sh_diffuse", "sh_specular", "spec_gain")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--vggt-mesh",
                        default="outputs/geometry_pruning/eval_carve_v35c30_conf/"
                                "carve_v35c30/mesh.ply")
    parser.add_argument("--gt-cameras",
                        default="data/scenes/nerf_lego_gt16/geometry/gt/cameras.json")
    parser.add_argument("--train-images", default="data/scenes/nerf_lego16/images")
    parser.add_argument("--heldout-dir", default="outputs/lego_heldout_val16")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--grid-res", type=int, default=160)
    parser.add_argument("--color-res", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr-sdf", type=float, default=0.01)
    parser.add_argument("--lr-weights", type=float, default=0.01)
    parser.add_argument("--lr-deform", type=float, default=0.001)
    parser.add_argument("--lr-color", type=float, default=0.02)
    parser.add_argument("--lr-light", type=float, default=0.01)
    parser.add_argument("--sil-weight", type=float, default=0.5)
    parser.add_argument("--corr-weight", type=float, default=0.02)
    parser.add_argument("--corr-max-voxels", type=float, default=3.0)
    parser.add_argument("--corr-every", type=int, default=5)
    parser.add_argument("--l-dev-weight", type=float, default=0.1)
    parser.add_argument("--deform-fraction", type=float, default=0.25)
    parser.add_argument("--sdf-init", choices=["proxy", "depth"], default="proxy",
                        help="proxy = nearest-surface signed field with visual-hull fix; "
                             "depth = sign voted from depth rendered off the VGGT mesh")
    parser.add_argument("--voxel", type=float, default=0.01150144350298549)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--corr-file", default=None,
                        help="reuse a correspondence npz built by a previous run")
    return parser.parse_args()


def sample_color(vertices, color, low, high):
    coords = ((vertices - low) / (high - low) * 2 - 1)[None, :, None, None, :]
    return F.grid_sample(color, coords, mode="bilinear", padding_mode="border",
                         align_corners=True)[0, :, :, 0, 0].T.contiguous()


def shade(normal, view_dir, albedo, params):
    normal = F.normalize(normal, dim=-1)
    diffuse = (s0.sh_basis(normal) @ params["sh_diffuse"]).clamp(min=0)
    reflected = 2 * (normal * view_dir).sum(-1, keepdim=True) * normal - view_dir
    specular = (s0.sh_basis(F.normalize(reflected, dim=-1))
                @ params["sh_specular"]).clamp(min=0)
    return albedo * diffuse + params["spec_gain"] * specular


def read_frame_images(camera_json, image_dir):
    payload = json.loads(Path(camera_json).read_text())
    frames = payload["frames"]
    images, intrinsics, poses, masks = [], [], [], []
    for frame in frames:
        image = cv2.imread(str(Path(image_dir) / frame["file"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(Path(image_dir) / frame["file"])
        if image.ndim == 3 and image.shape[2] == 4:
            masks.append(image[..., 3] > 127)
            colour = image[..., :3]
        else:
            masks.append(np.ones(image.shape[:2], dtype=bool))
            colour = image[..., ::-1]
        images.append(cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY))
        intrinsics.append(np.asarray(frame["K_original"], dtype=np.float64))
        poses.append(np.asarray(frame["T_world_to_cam"], dtype=np.float64))
    return images, intrinsics, poses, masks, payload


def sift_triangulation(camera_json, image_dir, voxel, max_per_pair=300):
    """Sparse, multi-view-consistent 3D points from SIFT matches between GT views."""
    grey, intrinsics, poses, masks, _ = read_frame_images(camera_json, image_dir)
    sift = cv2.SIFT_create(nfeatures=2048, contrastThreshold=0.01, edgeThreshold=10)
    keypoints, descriptors = [], []
    for image in grey:
        kp, des = sift.detectAndCompute(image, None)
        keypoints.append(kp)
        descriptors.append(des)
    height, width = grey[0].shape
    pairs = [(i, i + 1) for i in range(len(grey) - 1)]
    pairs += [(i, i + 2) for i in range(len(grey) - 2)]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = 0
    accepted = []
    accepted_errors = []
    for i, j in pairs:
        if descriptors[i] is None or descriptors[j] is None:
            continue
        good = [m for m, n in matcher.knnMatch(descriptors[i], descriptors[j], k=2)
                if m.distance < 0.75 * n.distance][:max_per_pair]
        raw_matches += len(good)
        if len(good) < 8:
            continue
        points_i = np.float32([keypoints[i][m.queryIdx].pt for m in good])
        points_j = np.float32([keypoints[j][m.trainIdx].pt for m in good])
        projection_i = intrinsics[i] @ poses[i][:3]
        projection_j = intrinsics[j] @ poses[j][:3]
        homogeneous = cv2.triangulatePoints(projection_i, projection_j,
                                            points_i.T, points_j.T)
        world = (homogeneous[:3] / homogeneous[3]).T
        for point, pi, pj in zip(world, points_i, points_j):
            if not np.isfinite(point).all():
                continue
            cam_i = poses[i][:3, :3] @ point + poses[i][:3, 3]
            cam_j = poses[j][:3, :3] @ point + poses[j][:3, 3]
            if cam_i[2] <= 0.05 or cam_j[2] <= 0.05:
                continue
            error_i = np.linalg.norm(
                (intrinsics[i] @ (cam_i / cam_i[2]))[:2] - pi) if cam_i[2] > 0 else 1e9
            error_j = np.linalg.norm(
                (intrinsics[j] @ (cam_j / cam_j[2]))[:2] - pj) if cam_j[2] > 0 else 1e9
            if error_i > 3.0 or error_j > 3.0:
                continue
            visible = 0
            for k in range(len(grey)):
                cam = poses[k][:3, :3] @ point + poses[k][:3, 3]
                if cam[2] <= 0.05:
                    continue
                u = intrinsics[k][0, 0] * cam[0] / cam[2] + intrinsics[k][0, 2]
                v = intrinsics[k][1, 1] * cam[1] / cam[2] + intrinsics[k][1, 2]
                if 0 <= u < width and 0 <= v < height and masks[k][int(v), int(u)]:
                    visible += 1
            if visible >= 6:
                accepted.append(point)
                accepted_errors.append(max(error_i, error_j))
    points = np.asarray(accepted, dtype=np.float64) if accepted else np.zeros((0, 3))
    if len(points):
        keep = []
        for index in range(len(points)):
            if all(np.linalg.norm(points[index] - points[other]) > 0.75 * voxel
                   for other in keep):
                keep.append(index)
        points = points[keep]
    stats = {"pairs": len(pairs), "raw_matches": int(raw_matches),
             "triangulated_candidates": int(len(accepted)),
             "accepted_points": int(len(points))}
    if accepted_errors:
        stats["match_reprojection_median_px"] = float(np.median(accepted_errors))
    if len(points):
        per_view = []
        for k in range(len(grey)):
            cam = points @ poses[k][:3, :3].T + poses[k][:3, 3]
            depth = cam[:, 2]
            valid = depth > 0.05
            u = np.zeros(len(points))
            v = np.zeros(len(points))
            u[valid] = intrinsics[k][0, 0] * cam[valid, 0] / depth[valid] \
                + intrinsics[k][0, 2]
            v[valid] = intrinsics[k][1, 1] * cam[valid, 1] / depth[valid] \
                + intrinsics[k][1, 2]
            inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            count = 0
            if inside.any():
                count = int(masks[k][v[inside].astype(int), u[inside].astype(int)].sum())
            per_view.append(count)
        stats["per_view_points"] = per_view
        stats["views_with_20_points"] = int(sum(c >= 20 for c in per_view))
    return points, stats


def correspondence_coverage(points, mesh, voxel):
    if len(points) == 0:
        return {"points": 0, "point_inlier_ratio_2vox": 0.0,
                "surface_coverage_2vox": 0.0, "surface_coverage_1vox": 0.0}
    vertex_tree = cKDTree(mesh["vertices_np"])
    point_distance, _ = vertex_tree.query(points, k=1)
    point_tree = cKDTree(points)
    surface_distance, _ = point_tree.query(mesh["vertices_np"], k=1)
    return {"points": int(len(points)),
            "point_inlier_ratio_2vox": float((point_distance <= 2 * voxel).mean()),
            "point_median_distance_voxel": float(np.median(point_distance) / voxel),
            "surface_coverage_2vox": float((surface_distance <= 2 * voxel).mean()),
            "surface_coverage_1vox": float((surface_distance <= voxel).mean())}


def visual_hull_inside(points, views, resolution):
    """True where a grid point projects inside every GT foreground silhouette."""
    inside = np.ones(len(points), dtype=bool)
    for view in views:
        intrinsic = view["k"].detach().cpu().numpy()
        pose = view["t"].detach().cpu().numpy()
        mask = view["mask"].detach().cpu().numpy()
        cam = points @ pose[:3, :3].T + pose[:3, 3]
        depth = cam[:, 2]
        valid = depth > 0.05
        u = np.zeros(len(points))
        v = np.zeros(len(points))
        u[valid] = intrinsic[0, 0] * cam[valid, 0] / depth[valid] + intrinsic[0, 2]
        v[valid] = intrinsic[1, 1] * cam[valid, 1] / depth[valid] + intrinsic[1, 2]
        inside_view = valid & (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
        hit = np.zeros(len(points), dtype=bool)
        if inside_view.any():
            hit[inside_view] = mask[v[inside_view].astype(int),
                                    u[inside_view].astype(int)]
        inside &= hit
    return inside


def build_sdf_grid(mesh, proxy, grid_res, deform_fraction, views, resolution):
    flexicubes = FlexiCubes(device="cuda")
    grid, cubes = flexicubes.construct_voxel_grid(grid_res)
    vertices = mesh["vertices_np"]
    low, high = vertices.min(0), vertices.max(0)
    center = (low + high) / 2
    scale = float((high - low).max()) * 1.10
    cell = scale / grid_res
    x0 = grid * scale + torch.tensor(center, dtype=torch.float32, device="cuda")
    points = x0.detach().cpu().numpy()
    signed = proxy.signed(points)
    distance = np.abs(signed)
    inside = visual_hull_inside(points, views, resolution)
    # Outside the multi-view visual hull the sign is forced positive; otherwise
    # the nearest-surface normal sign is used.  Without this the non-watertight
    # VGGT mesh produces a closed blob that fills the whole grid.
    signed = np.where(inside, signed, distance)
    signed = np.clip(signed, -3 * cell, 3 * cell)
    sdf = torch.nn.Parameter(torch.tensor(signed, dtype=torch.float32, device="cuda"))
    weights = torch.nn.Parameter(torch.ones(len(cubes), 21, device="cuda"))
    deform = torch.nn.Parameter(torch.zeros_like(x0))
    max_disp = deform_fraction * cell
    bounds = (torch.tensor(center - scale / 2, dtype=torch.float32, device="cuda"),
              torch.tensor(center + scale / 2, dtype=torch.float32, device="cuda"))
    return {"flexicubes": flexicubes, "grid": grid, "cubes": cubes, "x0": x0,
            "sdf": sdf, "weights": weights, "deform": deform, "max_disp": max_disp,
            "scale": scale, "cell": cell, "bounds": bounds, "center": center}


def build_sdf_grid_depth(mesh, proxy, views, grid_res, deform_fraction, resolution):
    """Depth-vote init: sign from depth rendered off the VGGT mesh itself.

    The nearest-surface proxy used by ``build_sdf_grid`` has no reliable sign far
    from the mesh (the VGGT surface is non-watertight), which injects spurious
    level sets.  Here each grid vertex is projected into every fixed GT camera
    and compared with the depth rendered from the same mesh: in front of the
    rendered surface is outside (+), behind it is inside (-).  The magnitude
    still comes from the proxy distance so the band has a usable gradient, and
    vertices outside every silhouette default to outside.
    """
    flexicubes = FlexiCubes(device="cuda")
    grid, cubes = flexicubes.construct_voxel_grid(grid_res)
    vertices = mesh["vertices_np"]
    low, high = vertices.min(0), vertices.max(0)
    center = (low + high) / 2
    scale = float((high - low).max()) * 1.10
    cell = scale / grid_res
    x0 = grid * scale + torch.tensor(center, dtype=torch.float32, device="cuda")
    points = x0.detach().cpu().numpy()
    distance = np.abs(proxy.signed(points))
    renderer = s0.NvdRaster()
    zeros = torch.zeros_like(mesh["vertices"])
    votes = np.zeros(len(points), dtype=np.int32)
    observed = np.zeros(len(points), dtype=bool)
    for view in views:
        depth, alpha, _, _, _ = renderer.render(
            mesh["vertices"], mesh["faces_i32"], zeros, zeros,
            view["k"], view["t"], resolution)
        depth_np = depth.detach().cpu().numpy()
        mask_np = alpha.detach().cpu().numpy() > 0.5
        intrinsic = view["k"].detach().cpu().numpy()
        pose = view["t"].detach().cpu().numpy()
        cam = points @ pose[:3, :3].T + pose[:3, 3]
        z = cam[:, 2]
        valid = z > 0.05
        u = np.zeros(len(points))
        v = np.zeros(len(points))
        u[valid] = intrinsic[0, 0] * cam[valid, 0] / z[valid] + intrinsic[0, 2]
        v[valid] = intrinsic[1, 1] * cam[valid, 1] / z[valid] + intrinsic[1, 2]
        inside_image = valid & (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
        seen = np.zeros(len(points), dtype=bool)
        if inside_image.any():
            rows = v[inside_image].astype(int)
            cols = u[inside_image].astype(int)
            seen[inside_image] = mask_np[rows, cols]
        if seen.any():
            reference = depth_np[v[seen].astype(int), u[seen].astype(int)]
            behind = z[seen] > reference
            votes[seen] += np.where(behind, -1, 1)
            observed |= seen
    signed = np.where(votes < 0, -1.0, 1.0) * distance
    signed = np.clip(signed, -3 * cell, 3 * cell)
    sdf = torch.nn.Parameter(torch.tensor(np.asarray(signed, dtype=np.float32),
                                          device="cuda"))
    weights = torch.nn.Parameter(torch.ones(len(cubes), 21, device="cuda"))
    deform = torch.nn.Parameter(torch.zeros_like(x0))
    max_disp = deform_fraction * cell
    bounds = (torch.tensor(center - scale / 2, dtype=torch.float32, device="cuda"),
              torch.tensor(center + scale / 2, dtype=torch.float32, device="cuda"))
    return {"flexicubes": flexicubes, "grid": grid, "cubes": cubes, "x0": x0,
            "sdf": sdf, "weights": weights, "deform": deform, "max_disp": max_disp,
            "scale": scale, "cell": cell, "bounds": bounds, "center": center,
            "sdf_init": {"mode": "depth", "observed_fraction": float(observed.mean()),
                         "inside_fraction": float((signed < 0).mean())}}


def extract(state):
    x = state["x0"] + state["max_disp"] * torch.tanh(state["deform"])
    weights = state["weights"]
    vertices, faces, l_dev = state["flexicubes"](
        x, state["sdf"], state["cubes"], state["grid_res"],
        weights[:, :12].contiguous(), weights[:, 12:20].contiguous(),
        weights[:, 20], training=False)
    return vertices, faces.to(torch.int32).contiguous(), faces.long(), l_dev


def evaluate_views(renderer, vertices, faces, normals, albedo, params, views, resolution):
    stats = []
    with torch.no_grad():
        for view in views:
            depth, alpha, shading_normal, albedo_pix, view_dir = renderer.render(
                vertices, faces, normals, albedo, view["k"], view["t"], resolution)
            rgb = shade(shading_normal, view_dir, albedo_pix, params) * alpha[..., None]
            stats.append(s0.masked_stats(rgb, alpha, view))
    keys = ("l1", "l1_core", "psnr", "silhouette_iou")
    return {key: float(np.nanmean([item[key] for item in stats])) for key in keys}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    out_root = Path(args.out)
    run_dir = out_root / args.run
    run_dir.mkdir(parents=True, exist_ok=False)
    sampler = s0.GpuPeakSampler()
    sampler.start()
    started = time.time()

    mesh = s0.load_mesh(Path(args.vggt_mesh), device)
    proxy = s01.ProxySurface(Path(args.vggt_mesh), device)
    train = s0.load_views(Path(args.gt_cameras), Path(args.train_images),
                          args.resolution, device)
    if not train["views"]:
        raise ValueError("no train views")
    if args.sdf_init == "depth":
        print(f"[{args.run}] building depth-vote init", flush=True)
        state = build_sdf_grid_depth(mesh, proxy, train["views"], args.grid_res,
                                     args.deform_fraction, args.resolution)
    else:
        state = build_sdf_grid(mesh, proxy, args.grid_res, args.deform_fraction,
                               train["views"], args.resolution)
    state["grid_res"] = args.grid_res

    low, high = state["bounds"]
    cell = state["cell"]
    color = torch.nn.Parameter(
        torch.full((1, 3, args.color_res, args.color_res, args.color_res), 0.5,
                   device=device) + 0.02 * torch.randn(
            1, 3, args.color_res, args.color_res, args.color_res, device=device))
    params = {"sh_diffuse": torch.nn.Parameter(torch.zeros(9, 3, device=device)),
              "sh_specular": torch.nn.Parameter(torch.zeros(9, 3, device=device)),
              "spec_gain": torch.nn.Parameter(torch.tensor(0.1, device=device))}
    with torch.no_grad():
        params["sh_diffuse"][0] = 1.0 / s0.Y00
        state["deform"].add_(torch.randn_like(state["deform"]) * 0.02)
        state["weights"].add_(torch.randn_like(state["weights"]) * 0.02)
        state["sdf"].add_(torch.randn_like(state["sdf"]) * 0.15 * cell)

    heldout_dir = Path(args.heldout_dir)
    heldout_all = s0.load_views(heldout_dir / "cameras.json", heldout_dir / "images",
                               args.resolution, device,
                               split_json=heldout_dir / "split.json")
    count = min(args.heldout_count, len(heldout_all["views"]))
    indices = np.unique(np.linspace(0, len(heldout_all["views"]) - 1,
                                   count).round().astype(int))
    heldout_primary = [heldout_all["views"][i] for i in indices]

    if args.corr_weight <= 0 and not args.corr_file:
        corr_points = np.zeros((0, 3))
        corr_stats = {"disabled": True}
    elif args.corr_file and Path(args.corr_file).exists():
        payload = np.load(args.corr_file)
        corr_points = payload["points"]
        corr_stats = json.loads(payload["stats"].item())
    else:
        corr_points, corr_stats = sift_triangulation(
            args.gt_cameras, args.train_images, args.voxel)
        np.savez_compressed(run_dir / "correspondence.npz", points=corr_points,
                            stats=json.dumps(corr_stats))
    corr_stats.update(correspondence_coverage(corr_points, mesh, args.voxel))
    corr_np = corr_points.astype(np.float32) if len(corr_points) else None
    print(f"[{args.run}] corr {json.dumps(corr_stats)}", flush=True)

    renderer = s0.NvdRaster()
    optimiser = torch.optim.Adam([
        {"params": [state["sdf"]], "lr": args.lr_sdf},
        {"params": [state["deform"]], "lr": args.lr_deform},
        {"params": [state["weights"]], "lr": args.lr_weights},
        {"params": [color], "lr": args.lr_color},
        {"params": [params["sh_diffuse"], params["sh_specular"], params["spec_gain"]],
         "lr": args.lr_light},
    ])

    history = []
    corr_index = corr_target = None
    for iteration in range(args.iters):
        vertices, faces, faces_long, l_dev = extract(state)
        normals = vertex_normals(vertices, faces_long)
        albedo = sample_color(vertices, color, low, high)
        loss = vertices.sum() * 0.0
        for view in train["views"]:
            depth, alpha, shading_normal, albedo_pix, view_dir = renderer.render(
                vertices, faces, normals, albedo, view["k"], view["t"], args.resolution)
            rgb = shade(shading_normal, view_dir, albedo_pix, params) * alpha[..., None]
            loss = loss + s0.photometric_loss(rgb, alpha, view)
            if args.sil_weight:
                loss = loss + args.sil_weight * s0.silhouette_loss(alpha, view)
        loss = loss / len(train["views"]) + args.l_dev_weight * l_dev.mean()
        if args.corr_weight and corr_np is not None and len(corr_np):
            # Rebuild the association every iteration: FlexiCubes changes the
            # vertex count between iterations, so a cached index would go stale.
            with torch.no_grad():
                tree = cKDTree(vertices.detach().cpu().numpy())
                distance, index = tree.query(corr_np, k=1)
                keep = distance <= args.corr_max_voxels * args.voxel
                corr_index = torch.from_numpy(index[keep]).long().to(device)
                corr_target = torch.from_numpy(corr_np[keep]).to(device)
            if corr_index.numel():
                delta = vertices[corr_index] - corr_target
                loss = loss + args.corr_weight * (delta.pow(2).sum(-1).mean()
                                                  / args.voxel ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [state["sdf"], state["deform"], state["weights"], color,
             params["sh_diffuse"], params["sh_specular"], params["spec_gain"]], 1.0)
        optimiser.step()
        with torch.no_grad():
            state["sdf"].clamp_(-3 * cell, 3 * cell)
        if iteration % args.log_every == 0 or iteration == args.iters - 1:
            history.append({"iter": iteration, "loss": float(loss),
                            "faces": int(len(faces)),
                            "vertices": int(len(vertices)),
                            "peak_mib": sampler.delta_mib})
            print(f"[{args.run}] iter {iteration} loss {float(loss):.6f} "
                  f"faces {len(faces)} peak {sampler.delta_mib} MiB", flush=True)

    with torch.no_grad():
        vertices, faces_long, _ = state["flexicubes"](
            state["x0"] + state["max_disp"] * torch.tanh(state["deform"]),
            state["sdf"], state["cubes"], args.grid_res,
            state["weights"][:, :12].contiguous(), state["weights"][:, 12:20].contiguous(),
            state["weights"][:, 20], training=False)
        faces = faces_long.long()
        faces_i32 = faces.to(torch.int32).contiguous()
        normals = vertex_normals(vertices, faces)
        albedo = sample_color(vertices, color, low, high)
        train_stats = evaluate_views(renderer, vertices, faces_i32, normals, albedo,
                                     params, train["views"], args.resolution)
        heldout_stats = evaluate_views(renderer, vertices, faces_i32, normals, albedo,
                                       params, heldout_primary, args.resolution)
        heldout_all_stats = evaluate_views(renderer, vertices, faces_i32, normals, albedo,
                                           params, heldout_all["views"], args.resolution)
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    write_ply(run_dir / "mesh.ply", vertices_np, faces_np,
              np.full_like(vertices_np, 0.65), normals.detach().cpu().numpy())
    sampler.stop()
    seconds = time.time() - started
    summary = {
        "run": args.run, "seed": args.seed, "status": "ok",
        "vggt_mesh": args.vggt_mesh, "vggt_mesh_sha256_16": mesh["sha256_16"],
        "views": len(train["views"]), "resolution": args.resolution,
        "grid_res": args.grid_res, "color_res": args.color_res,
        "iters": args.iters, "seconds": seconds,
        "sdf_init": {"mode": args.sdf_init, **(state.get("sdf_init") or {})},
        "peak_vram_compute_mib": int(torch.cuda.max_memory_allocated() // 2**20),
        "peak_vram_smi_delta_mib": int(sampler.delta_mib),
        "gpu_used_before_mib": int(sampler.idle_mib),
        "final_faces": int(len(faces)), "final_vertices": int(len(vertices)),
        "mesh_sha256_16": s0.sha256_16(run_dir / "mesh.ply"),
        "correspondence": corr_stats,
        "losses": {k: float(v) for k, v in {
            "sil_weight": args.sil_weight, "corr_weight": args.corr_weight,
            "l_dev_weight": args.l_dev_weight}.items()},
        "train_render": train_stats, "heldout_primary_render": heldout_stats,
        "heldout_all_render": heldout_all_stats, "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "run", "seconds", "final_faces", "peak_vram_compute_mib",
        "heldout_primary_render", "heldout_all_render")}, indent=2))


if __name__ == "__main__":
    main()
