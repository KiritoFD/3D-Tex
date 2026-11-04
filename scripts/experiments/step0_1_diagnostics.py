"""Step 0.1 diagnostics for the image-driven surface plan.

P1  local Gaussian bumps on silhouette / inner depth edge / flat regions
P2  automatic differentiation vs central finite difference for the scalar
    deformation parameter a of V_i(a) = V_i^0 + a * w_i * n_i
P3  appearance freedom: frozen appearance, SH-only adaptation, low-rank albedo
P4  supervision ablation on silhouette / inner depth edge / flat seeds:
    RGB, symmetric silhouette, combined, proxy correspondence
P5  local recovery with 20-50 low-dimensional optimisation steps

Step 0 (``step0_objective.py``) and its 80% NO-GO result stay untouched.  This
script writes only to a new output directory.

Run from the repository root:

    PYTHONPATH=scripts /opt/conda/envs/3dtex/bin/python \
        scripts/experiments/step0_1_diagnostics.py --out outputs/step0_1_20261105
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step0_objective as s0
from mesh_opt.geometry import vertex_normals


def parse_list(text, cast=float):
    return [cast(item) for item in str(text).split(",") if item.strip()]


class ProxySurface:
    """Outward-oriented GT-proxy vertex normals for a local signed distance.

    This is a nearest-surface proxy: the signed distance is the projection of
    the offset to the nearest GT-proxy vertex onto that vertex normal.  It is
    not a nearest-vertex distance, which was the defect of the Step 0 metric.
    """

    def __init__(self, path, device):
        mesh = s0.load_mesh(Path(path), device)
        self.vertices = mesh["vertices"].detach().cpu().numpy().astype(np.float64)
        normals = mesh["normals"].detach().cpu().numpy().astype(np.float64)
        center = self.vertices.mean(0)
        if float((((self.vertices - center) * normals).sum(1)).sum()) < 0:
            normals = -normals
        self.normals = normals
        self.tree = cKDTree(self.vertices)

    def signed(self, points):
        _, index = self.tree.query(points, k=1)
        offset = points - self.vertices[index]
        return (offset * self.normals[index]).sum(-1)


def projection(vertices, view, res):
    k, t = view["k"], view["t"]
    cam = vertices @ t[:3, :3].transpose(0, 1) + t[:3, 3]
    z = cam[..., 2].clamp(min=1e-6)
    u = k[0, 0] * cam[..., 0] / z + k[0, 2]
    v = k[1, 1] * cam[..., 1] / z + k[1, 2]
    return u, v, z


def shade_with(normal, view_dir, albedo, params):
    normal = F.normalize(normal, dim=-1)
    diffuse = (s0.sh_basis(normal) @ params["sh_diffuse"]).clamp(min=0)
    reflected = 2 * (normal * view_dir).sum(-1, keepdim=True) * normal - view_dir
    specular = (s0.sh_basis(F.normalize(reflected, dim=-1)) @ params["sh_specular"]).clamp(min=0)
    return albedo * diffuse + params["spec_gain"] * specular


def render_params(renderer, vertices, mesh, normals, albedo, params, view, res):
    depth, alpha, shading_normal, albedo_pix, view_dir = renderer.render(
        vertices, mesh["faces_i32"], normals, albedo, view["k"], view["t"], res)
    color = shade_with(shading_normal, view_dir, albedo_pix, params)
    return color * alpha[..., None], alpha, depth


def view_loss(rgb, alpha, depth, view, kind, sil_weight, vertices, base, weight, res):
    if kind == "rgb":
        return s0.photometric_loss(rgb, alpha, view)
    if kind == "sil":
        return s0.silhouette_loss(alpha, view)
    if kind == "combo":
        return (s0.photometric_loss(rgb, alpha, view)
                + sil_weight * s0.silhouette_loss(alpha, view))
    if kind == "corr":
        return correspondence_view_loss(vertices, base, weight, view, res)
    raise ValueError(f"unknown loss kind {kind}")


def correspondence_view_loss(vertices, base, weight, view, res):
    """Reprojection correspondence on the bump support (proxy target = base)."""
    selected = weight > 0.2
    if not bool(selected.any()):
        return vertices.sum() * 0.0
    u0, v0, z0 = projection(base[selected], view, res)
    u1, v1, z1 = projection(vertices[selected], view, res)
    valid = (z0 > 1e-6) & (z1 > 1e-6)
    if not bool(valid.any()):
        return vertices.sum() * 0.0
    error = ((u1[valid] - u0[valid]) ** 2 + (v1[valid] - v0[valid]) ** 2)
    return error.mean() / (res * res)


def total_loss(vertices, mesh, params, albedo, views, kind, sil_weight,
               base, weight, renderer, res):
    normals = vertex_normals(vertices, mesh["faces"])
    total = 0
    for view in views:
        rgb, alpha, _ = render_params(renderer, vertices, mesh, normals, albedo,
                                      params, view, res)
        total = total + view_loss(rgb, alpha, None, view, kind, sil_weight,
                                  vertices, base, weight, res)
    return total / len(views)


def deform(base, normals, weight, a_vox, voxel):
    return base + (a_vox * voxel) * (weight[:, None] * normals)


def bump_weight(base, seed_vertex, sigma_vox, voxel):
    distance = ((base - base[seed_vertex]) ** 2).sum(-1)
    weight = torch.exp(-distance / (2 * (sigma_vox * voxel) ** 2))
    return weight / weight.max().clamp(min=1e-12)


def boundary_masks(depth, alpha, res):
    alpha_np = alpha.detach().cpu().numpy()
    depth_np = depth.detach().cpu().numpy()
    foreground = (alpha_np > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    boundary = ((cv2.dilate(foreground, kernel) - foreground)
                | (foreground - cv2.erode(foreground, kernel))) > 0
    valid = alpha_np > 0.1
    depth_clean = np.where(valid, depth_np, 0.0).astype(np.float32)
    gx = cv2.Sobel(depth_clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_clean, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    interior = cv2.erode(foreground, np.ones((9, 9), np.uint8)) > 0
    threshold = np.quantile(magnitude[interior], 0.95) if interior.any() else 0.0
    edge = (magnitude >= max(float(threshold), 1e-9)) & interior
    flat_interior = cv2.erode(foreground, np.ones((15, 15), np.uint8)) > 0
    edge_near = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    flat = flat_interior & ~edge_near
    return boundary, edge, flat


def select_seeds(mesh, views, params, albedo, renderer, args, curvature):
    base = mesh["vertices"]
    normals = mesh["normals"]
    device = base.device
    resolution = args.resolution
    low_curvature = curvature <= torch.quantile(curvature, 0.30)
    visibility = torch.zeros(len(base), dtype=torch.long, device=device)
    candidates = {"silhouette": set(), "inner_depth_edge": set(), "flat": set()}
    for view in views:
        with torch.no_grad():
            depth, alpha, _, _, _ = renderer.render(
                base, mesh["faces_i32"], normals, albedo, view["k"], view["t"], resolution)
        boundary, edge, flat = boundary_masks(depth, alpha, resolution)
        u, v, z = projection(base, view, resolution)
        ui = u.detach().cpu().numpy()
        vi = v.detach().cpu().numpy()
        zi = z.detach().cpu().numpy()
        depth_np = depth.detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()
        inside = ((ui >= 0) & (ui < resolution) & (vi >= 0) & (vi < resolution)
                  & np.isfinite(ui) & np.isfinite(vi) & (zi > 0))
        index = np.zeros(len(base), dtype=np.int64)
        index[inside] = vi[inside].astype(np.int64) * resolution + ui[inside].astype(np.int64)
        sampled_depth = np.zeros(len(base), dtype=np.float32)
        sampled_alpha = np.zeros(len(base), dtype=np.float32)
        sampled_depth[inside] = depth_np.ravel()[index[inside]]
        sampled_alpha[inside] = alpha_np.ravel()[index[inside]]
        visible_np = (inside & (sampled_alpha > 0.9)
                      & (np.abs(zi - sampled_depth) <= 3.0 * args.voxel))
        visible = torch.from_numpy(visible_np).to(device)
        visibility += visible.long()
        for name, mask in (("silhouette", boundary), ("inner_depth_edge", edge),
                           ("flat", flat)):
            host = np.zeros(len(base), dtype=bool)
            host[inside] = mask[vi[inside].astype(np.int64), ui[inside].astype(np.int64)]
            hit = visible_np & host
            if name == "flat":
                hit &= low_curvature.detach().cpu().numpy()
            candidates[name].update(np.flatnonzero(hit).tolist())
    seeds = {}
    min_separation = 25.0 * args.voxel
    for name, ids in candidates.items():
        ordered = sorted(ids, key=lambda i: (-int(visibility[i].item()), i))
        chosen = []
        base_np = base.detach().cpu().numpy()
        for vertex in ordered:
            if all(np.linalg.norm(base_np[vertex] - base_np[other]) >= min_separation
                   for other in chosen):
                chosen.append(int(vertex))
            if len(chosen) >= args.seeds_per_region:
                break
        seeds[name] = [{"vertex": int(vertex),
                        "position": [float(x) for x in base_np[vertex]],
                        "visible_views": int(visibility[vertex].item()),
                        "curvature": float(curvature[vertex].item())} for vertex in chosen]
    return seeds


def evaluate_loss(vertices, mesh, params, albedo, views, kind, args, renderer,
                  base, weight, params_albedo=None):
    return float(total_loss(vertices, mesh, params, params_albedo if params_albedo is not None else albedo,
                            views, kind, args.sil_weight, base, weight,
                            renderer, args.resolution))


def deformation_case(seed, base, normals, mesh, params, albedo, views, args,
                     renderer, region, delta, sign, kind="combo", proxy=None):
    weight = bump_weight(base, seed["vertex"], args.bump_sigma_voxels, args.voxel)
    a0 = sign * delta
    started = time.time()
    loss_zero = evaluate_loss(deform(base, normals, weight, 0.0, args.voxel),
                              mesh, params, albedo, views, kind, args, renderer,
                              base, weight)
    a = torch.tensor(a0, device=base.device, requires_grad=True)
    vertices = deform(base, normals, weight, a, args.voxel)
    loss = total_loss(vertices, mesh, params, albedo, views, kind,
                      args.sil_weight, base, weight, renderer, args.resolution)
    gradient = torch.autograd.grad(loss, a)[0]
    a_plus = a0 + args.fd_step
    a_minus = a0 - args.fd_step
    loss_plus = evaluate_loss(deform(base, normals, weight, a_plus, args.voxel),
                              mesh, params, albedo, views, kind, args, renderer,
                              base, weight)
    loss_minus = evaluate_loss(deform(base, normals, weight, a_minus, args.voxel),
                               mesh, params, albedo, views, kind, args, renderer,
                               base, weight)
    finite_difference = (loss_plus - loss_minus) / (2 * args.fd_step)
    ad_value = float(gradient.item())
    correct = bool(ad_value * a0 > 0)
    denominator = max(abs(finite_difference), 1e-9)
    proxy_sd = proxy_sd_slope = None
    correct_proxy = None
    if proxy is not None:
        selected = weight > 0.2
        weights_np = weight[selected].detach().cpu().numpy().astype(np.float64)
        sd_values = []
        for offset in (args.fd_step, -args.fd_step):
            points = deform(base, normals, weight, a0 + offset, args.voxel)[selected]
            distances = proxy.signed(points.detach().cpu().numpy().astype(np.float64))
            sd_values.append(float(np.average(distances, weights=weights_np)))
        proxy_sd = float(np.average(
            proxy.signed(deform(base, normals, weight, a0, args.voxel)[selected]
                         .detach().cpu().numpy().astype(np.float64)), weights=weights_np))
        proxy_sd_slope = (sd_values[0] - sd_values[1]) / (2 * args.fd_step)
        correct_proxy = bool(ad_value * (proxy_sd * proxy_sd_slope) > 0)
    return {
        "mesh": mesh["path"], "mesh_name": mesh.get("label", ""),
        "mesh_sha256_16": mesh["sha256_16"],
        "region": region, "seed_vertex": seed["vertex"],
        "seed_visible_views": seed["visible_views"], "seed_curvature": seed["curvature"],
        "delta_voxels": delta, "sign": sign, "loss_kind": kind,
        "views": len(views), "resolution": args.resolution, "iters": 1,
        "peak_vram_torch_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "seconds": round(time.time() - started, 1),
        "loss_zero": loss_zero, "loss_at_delta": float(loss.item()),
        "delta_loss": float(loss.item()) - loss_zero,
        "ad_dlda": ad_value, "fd_dlda": float(finite_difference),
        "fd_magnitude": float((loss_plus - loss_minus) / 2),
        "fd_relative_error": abs(ad_value - finite_difference) / denominator,
        "fd_sign_agree": bool(ad_value * finite_difference > 0),
        "correct_direction": correct,
        "proxy_signed_distance": proxy_sd,
        "proxy_sd_slope": proxy_sd_slope,
        "correct_toward_proxy": correct_proxy,
        "valid": True,
    }


def aggregate_cases(cases):
    ad = np.array([case["ad_dlda"] for case in cases], dtype=np.float64)
    fd = np.array([case["fd_dlda"] for case in cases], dtype=np.float64)
    return {
        "cases": len(cases),
        "correct_ratio": float(np.mean([case["correct_direction"] for case in cases])),
        "fd_sign_agreement": float(np.mean([case["fd_sign_agree"] for case in cases])),
        "fd_relative_error_median": float(np.median(
            [case["fd_relative_error"] for case in cases])),
        "ad_dlda_median": float(np.median(np.abs(ad))),
        "fd_dlda_median": float(np.median(np.abs(fd))),
        "delta_loss_median": float(np.median([case["delta_loss"] for case in cases])),
        "positive_delta_loss_ratio": float(np.mean(
            [case["delta_loss"] > 0 for case in cases])),
        "toward_proxy_ratio": float(np.mean(
            [case["correct_toward_proxy"] for case in cases
             if case.get("correct_toward_proxy") is not None])
        ) if any(case.get("correct_toward_proxy") is not None for case in cases) else None,
    }


def make_basis(base, rank=8, seed=0):
    generator = torch.Generator(device=base.device).manual_seed(seed)
    directions = torch.randn(3, rank, generator=generator, device=base.device)
    centered = base - base.mean(0, keepdim=True)
    scale = centered.std(0, keepdim=True).clamp(min=1e-6)
    return torch.sin((centered / scale) @ directions)


def adapt_params(mesh, params, albedo, vertices, views, mode, args, renderer):
    base = mesh["vertices"]
    trainable = {key: params[key].clone().detach().requires_grad_(True)
                 for key in ("sh_diffuse", "sh_specular", "spec_gain")}
    if mode == "frozen":
        return {key: value.detach() for key, value in params.items()}, albedo, None, None
    basis = make_basis(base, rank=args.lowrank_rank) if mode == "lowrank" else None
    coeff = torch.zeros(args.lowrank_rank, 3, device=base.device, requires_grad=True) \
        if mode == "lowrank" else None
    parameters = list(trainable.values()) + ([coeff] if coeff is not None else [])
    optimizer = torch.optim.Adam(parameters, lr=args.adapt_lr)
    initial_loss = None
    for iteration in range(args.adapt_iters):
        optimizer.zero_grad()
        if coeff is not None:
            current = albedo + basis @ coeff
        else:
            current = albedo
        normals = vertex_normals(vertices, mesh["faces"])
        total = 0
        for view in views:
            rgb, alpha, _ = render_params(renderer, vertices, mesh, normals, current,
                                          trainable, view, args.resolution)
            total = total + s0.photometric_loss(rgb, alpha, view) / len(views)
        if initial_loss is None:
            initial_loss = float(total.item())
        total.backward()
        optimizer.step()
    for key in trainable:
        trainable[key] = trainable[key].detach()
    final_albedo = (albedo + basis @ coeff).detach() if coeff is not None \
        else albedo
    with torch.no_grad():
        final_normals = vertex_normals(vertices, mesh["faces"])
        final_loss = 0.0
        for view in views:
            rgb, alpha, _ = render_params(renderer, vertices, mesh, final_normals,
                                          final_albedo, trainable, view, args.resolution)
            final_loss += float(s0.photometric_loss(rgb, alpha, view).item()) / len(views)
    return trainable, final_albedo, initial_loss, final_loss


def sift_stats(views, mesh, params, albedo, renderer, args):
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher.create(cv2.NORM_L2)
    per_view = []
    with torch.no_grad():
        for view in views:
            rgb, alpha, _ = render_params(renderer, mesh["vertices"], mesh,
                                          mesh["normals"], albedo, params, view,
                                          args.resolution)
            rendered = (rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            target = (view["rgb"].detach().cpu().numpy() * 255).astype(np.uint8)
            gray_a = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            key_a, desc_a = detector.detectAndCompute(gray_a, None)
            key_b, desc_b = detector.detectAndCompute(gray_b, None)
            if desc_a is None or desc_b is None:
                per_view.append({"view": view["name"], "matches": 0, "coverage": 0.0})
                continue
            pairs = matcher.knnMatch(desc_b, desc_a, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2
                    and pair[0].distance < 0.75 * pair[1].distance]
            cells = set()
            for match in good:
                x, y = key_b[match.queryIdx].pt
                if 0 <= int(y) < args.resolution and 0 <= int(x) < args.resolution:
                    if bool(view["mask"][int(y), int(x)]):
                        cells.add((int(y) // (args.resolution // 8),
                                   int(x) // (args.resolution // 8)))
            per_view.append({"view": view["name"], "matches": len(good),
                             "coverage": len(cells) / 64.0})
    return {
        "per_view": per_view,
        "matches_median": float(np.median([row["matches"] for row in per_view])),
        "coverage_median": float(np.median([row["coverage"] for row in per_view])),
        "sparse_note": "SIFT matches on Lego are sparse; no dense correspondence available",
    }


def recovery_case(seed, base, normals, mesh, params, albedo, views, args,
                  renderer, kind):
    weight = bump_weight(base, seed["vertex"], args.bump_sigma_voxels, args.voxel)
    a = torch.tensor(args.recover_start_voxels, device=base.device, requires_grad=True)
    optimizer = torch.optim.Adam([a], lr=args.recover_lr)
    trajectory = []
    started = time.time()
    losses = []
    for iteration in range(args.recover_iters):
        optimizer.zero_grad()
        vertices = deform(base, normals, weight, a, args.voxel)
        loss = total_loss(vertices, mesh, params, albedo, views, kind,
                          args.sil_weight, base, weight, renderer, args.resolution)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (iteration + 1) % 5 == 0 or iteration + 1 == args.recover_iters:
            trajectory.append({"iter": iteration + 1, "a_voxels": float(a.item()),
                               "loss": float(loss.item())})
    final = float(a.item())
    initial = float(args.recover_start_voxels)
    return {
        "mesh": mesh["path"], "mesh_name": mesh.get("label", ""),
        "mesh_sha256_16": mesh["sha256_16"],
        "loss_kind": kind, "seed_vertex": seed["vertex"], "seed_visible_views": seed["visible_views"],
        "views": len(views), "resolution": args.resolution,
        "iters": args.recover_iters, "optimizer": f"Adam lr {args.recover_lr} voxel",
        "peak_vram_torch_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "seconds": round(time.time() - started, 1),
        "final_loss": losses[-1], "initial_loss": losses[0],
        "a_start_voxels": initial, "a_final_voxels": final,
        "abs_reduction_ratio": (abs(initial) - abs(final)) / abs(initial),
        "trajectory": trajectory, "valid": True,
    }


def aggregate_recovery(cases):
    return {
        "cases": len(cases),
        "final_abs_median": float(np.median([abs(case["a_final_voxels"]) for case in cases]),
        ),
        "reduction_median": float(np.median([case["abs_reduction_ratio"] for case in cases]),
        ),
        "loss_decrease_ratio": float(np.mean([
            (case["initial_loss"] - case["final_loss"]) / max(case["initial_loss"], 1e-12)
            for case in cases]),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--gt-mesh", default="data/scenes/nerf_lego_gt16/mesh/mesh_init.ply")
    parser.add_argument("--vggt-mesh", default="outputs/geometry_pruning/"
                        "eval_carve_v35c30_conf/carve_v35c30/mesh.ply")
    parser.add_argument("--gt-cameras", default="data/scenes/nerf_lego_gt16/geometry/gt/cameras.json")
    parser.add_argument("--train-images", default="data/scenes/nerf_lego16/images")
    parser.add_argument("--appearance-gt", default="outputs/step0_20261105_r2/0a/gt/appearance.pt")
    parser.add_argument("--appearance-vggt", default="outputs/step0_20261105_r2/0a/vggt/appearance.pt")
    parser.add_argument("--meshes", default="gt,vggt")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--voxel", type=float, default=0.01150144350298549)
    parser.add_argument("--deltas", default="0.5,1.0")
    parser.add_argument("--fd-step", type=float, default=0.1)
    parser.add_argument("--seeds-per-region", type=int, default=3)
    parser.add_argument("--bump-sigma-voxels", type=float, default=4.0)
    parser.add_argument("--sil-weight", type=float, default=0.5)
    parser.add_argument("--adapt-iters", type=int, default=30)
    parser.add_argument("--adapt-lr", type=float, default=0.01)
    parser.add_argument("--lowrank-rank", type=int, default=8)
    parser.add_argument("--recover-iters", type=int, default=30)
    parser.add_argument("--recover-lr", type=float, default=0.1)
    parser.add_argument("--recover-start-voxels", type=float, default=1.0)
    parser.add_argument("--stages", default="p1,p2,p3,p4,p5")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / "summary.json").exists():
        raise SystemExit(f"{out_root}/summary.json exists; refusing to overwrite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    stages = {item.strip() for item in args.stages.split(",") if item.strip()}
    deltas = parse_list(args.deltas)
    mesh_names = [item.strip() for item in args.meshes.split(",") if item.strip()]
    appearance_paths = {"gt": args.appearance_gt, "vggt": args.appearance_vggt}
    mesh_paths = {"gt": args.gt_mesh, "vggt": args.vggt_mesh}
    mesh_labels = {"gt": "GT-depth TSDF proxy (frozen reference mesh)",
                   "vggt": "VGGT/TSDF carved mesh"}
    views = s0.load_views(Path(args.gt_cameras), Path(args.train_images),
                          args.resolution, device)["views"]
    renderer = s0.NvdRaster()
    proxy = ProxySurface(args.gt_mesh, device)
    checked_out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True, check=False).stdout.strip()
    summary = {
        "reference": {
            "gt_mesh": str(args.gt_mesh), "gt_mesh_label": mesh_labels["gt"],
            "proxy_note": "GT mesh is a frozen TSDF built from GT depth, not an analytic asset",
            "vggt_mesh": str(args.vggt_mesh), "vggt_mesh_label": mesh_labels["vggt"],
            "views": len(views), "resolution": args.resolution,
            "voxel": args.voxel,
        },
        "gpu_before": checked_out,
        "runs": {}, "invalid_runs": [], "gates": {},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=False).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                                            capture_output=True, text=True,
                                            check=False).stdout.strip()),
    }

    def run_case(key, function, *call_args, **call_kwargs):
        torch.cuda.reset_peak_memory_stats()
        try:
            record = function(*call_args, **call_kwargs)
            summary["runs"][key] = record
            print(f"[{key}] " + json.dumps({k: v for k, v in record.items()
                                            if k not in ("trajectory",)})[:1200],
                  flush=True)
            return record
        except Exception as error:  # noqa: BLE001
            summary["invalid_runs"].append({
                "run": key, "reason": repr(error),
                "traceback": traceback.format_exc()})
            print(f"[{key}] INVALID {error!r}", flush=True)
            return None

    for name in mesh_names:
        mesh = s0.load_mesh(Path(mesh_paths[name]), device)
        mesh["label"] = name
        state = torch.load(appearance_paths[name], map_location="cpu")
        params = {key: value.to(device) for key, value in state.items()}
        albedo = params["albedo"]
        curvature = s0.curvature_mask(mesh)
        seeds = select_seeds(mesh, views, params, albedo, renderer, args, curvature)
        summary.setdefault("seeds", {})[name] = seeds
        print(f"[{name}] seeds " + json.dumps({k: len(v) for k, v in seeds.items()}),
              flush=True)
        if not seeds.get("silhouette"):
            summary["invalid_runs"].append({
                "run": f"{name}_seeds", "reason": "no silhouette seeds found"})

        if "p2" in stages:
            for region in ("silhouette", "inner_depth_edge", "flat"):
                if not seeds.get(region):
                    continue
                for sign in (1.0, -1.0):
                    seed = seeds[region][0]
                    run_case(f"p2_{name}_{region}_{'pos' if sign > 0 else 'neg'}",
                             deformation_case, seed, mesh["vertices"], mesh["normals"],
                             mesh, params, albedo, views, args, renderer, region,
                             1.0, sign, proxy=proxy)

        if "p1" in stages:
            for region in ("silhouette", "inner_depth_edge", "flat"):
                for delta in deltas:
                    for sign in (1.0, -1.0):
                        for seed in seeds.get(region, []):
                            tag = "pos" if sign > 0 else "neg"
                            run_case(
                                f"p1_{name}_{region}_d{delta:g}_{tag}_v{seed['vertex']}",
                                deformation_case, seed, mesh["vertices"], mesh["normals"],
                                mesh, params, albedo, views, args, renderer, region,
                                delta, sign, proxy=proxy)

        if "p3" in stages:
            seed = (seeds.get("silhouette") or seeds.get("inner_depth_edge") or [None])[0]
            if seed is not None:
                weight = bump_weight(mesh["vertices"], seed["vertex"],
                                     args.bump_sigma_voxels, args.voxel)
                perturbed = deform(mesh["vertices"], mesh["normals"], weight,
                                   1.0, args.voxel)
                for mode in ("frozen", "sh", "lowrank"):
                    started = time.time()
                    torch.cuda.reset_peak_memory_stats()
                    adapted, ad_albedo, initial_loss, final_loss = adapt_params(
                        mesh, params, albedo, perturbed, views, mode, args, renderer)
                    if mode == "frozen":
                        with torch.no_grad():
                            initial_loss = float(total_loss(
                                perturbed, mesh, adapted, ad_albedo, views, "rgb",
                                args.sil_weight, mesh["vertices"], weight,
                                renderer, args.resolution).item())
                        final_loss = initial_loss
                    a = torch.tensor(1.0, device=device, requires_grad=True)
                    vertices = deform(mesh["vertices"], mesh["normals"], weight, a, args.voxel)
                    loss = total_loss(vertices, mesh, adapted, ad_albedo, views, "combo",
                                      args.sil_weight, mesh["vertices"], weight,
                                      renderer, args.resolution)
                    gradient = float(torch.autograd.grad(loss, a)[0].item())
                    record = {
                        "mesh": mesh["path"], "mesh_sha256_16": mesh["sha256_16"],
                        "mode": mode, "seed_vertex": seed["vertex"],
                        "views": len(views), "resolution": args.resolution,
                        "iters": args.adapt_iters if mode != "frozen" else 0,
                        "peak_vram_torch_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        "seconds": round(time.time() - started, 1),
                        "rgb_loss_before_adapt": initial_loss,
                        "rgb_loss_after_adapt": final_loss,
                        "combo_loss_at_delta": float(loss.item()),
                        "ad_dlda": gradient, "correct_direction": bool(gradient > 0),
                        "valid": True,
                    }
                    run_case(f"p3_{name}_{mode}", lambda record=record: record)

        if "p4" in stages:
            for region in ("silhouette", "inner_depth_edge", "flat"):
                for seed in seeds.get(region, [])[:2]:
                    for kind in ("rgb", "sil", "combo", "corr"):
                        for sign in (1.0, -1.0):
                            tag = "pos" if sign > 0 else "neg"
                            run_case(f"p4_{name}_{region}_{kind}_{tag}_v{seed['vertex']}",
                                     deformation_case, seed,
                                     mesh["vertices"], mesh["normals"], mesh, params, albedo,
                                     views, args, renderer, region, 1.0, sign, kind,
                                     proxy=proxy)
            summary.setdefault("sift", {})[name] = sift_stats(
                views, mesh, params, albedo, renderer, args)
            print(f"[{name}] sift {json.dumps(summary['sift'][name])}",
                  flush=True)

        if "p5" in stages:
            for seed in seeds.get("silhouette", [])[:2]:
                for kind in ("rgb", "sil", "combo", "corr"):
                    run_case(f"p5_{name}_{kind}_v{seed['vertex']}", recovery_case,
                             seed, mesh["vertices"], mesh["normals"], mesh, params, albedo,
                             views, args, renderer, kind)

    p1_cases = [record for key, record in summary["runs"].items() if key.startswith("p1_")]
    p2_cases = [record for key, record in summary["runs"].items() if key.startswith("p2_")]
    p3_cases = [record for key, record in summary["runs"].items() if key.startswith("p3_")]
    p4_cases = [record for key, record in summary["runs"].items() if key.startswith("p4_")]
    p5_cases = [record for key, record in summary["runs"].items() if key.startswith("p5_")]

    summary["p1"] = {}
    for case in p1_cases:
        key = f"{case['mesh_name']}_{case['region']}_d{case['delta_voxels']:g}"
        summary["p1"].setdefault(key, []).append(case)
    summary["p1"] = {key: aggregate_cases(value) for key, value in summary["p1"].items()}

    summary["p2"] = aggregate_cases(p2_cases) if p2_cases else None
    summary["p2_detail"] = p2_cases
    summary["p4"] = {}
    summary["p4_by_region"] = {}
    for case in p4_cases:
        summary["p4"].setdefault(case["loss_kind"], []).append(case)
        summary["p4_by_region"].setdefault(
            f"{case['mesh_name']}_{case['region']}_{case['loss_kind']}", []).append(case)
    summary["p4"] = {key: aggregate_cases(value) for key, value in summary["p4"].items()}
    summary["p4_by_region"] = {key: aggregate_cases(value)
                               for key, value in summary["p4_by_region"].items()}
    summary["p5"] = {}
    for case in p5_cases:
        summary["p5"].setdefault(case["loss_kind"], []).append(case)
    summary["p5"] = {key: aggregate_recovery(value) for key, value in summary["p5"].items()}

    p2_pass = bool(summary["p2"] and summary["p2"]["fd_sign_agreement"] >= 0.8
                   and summary["p2"]["fd_relative_error_median"] <= 0.5
                   and (summary["p2"].get("toward_proxy_ratio") or 0.0) >= 0.8)
    combo_gt_cases = [case for case in p5_cases
                      if case["mesh_name"] == "gt" and case["loss_kind"] == "combo"]
    sil_gt_cases = [case for case in p5_cases
                    if case["mesh_name"] == "gt" and case["loss_kind"] == "sil"]
    combo_gt = aggregate_recovery(combo_gt_cases) if combo_gt_cases else None
    sil_gt = aggregate_recovery(sil_gt_cases) if sil_gt_cases else None
    supervision_pass = bool(
        combo_gt and combo_gt["final_abs_median"] <= 0.2
        and combo_gt["reduction_median"] >= 0.8)
    summary["gates"] = {
        "level1_gradient_correctness": {
            "pass": p2_pass,
            "criterion": ("AD vs FD sign agreement >= 0.8, median relative error <= 0.5, "
                          "and toward-GT-proxy direction ratio >= 0.8"),
            "numbers": summary["p2"],
            "note": "VGGT uses the GT-proxy nearest-surface signed distance, not nearest vertex",
        },
        "level2_supervision_effectiveness": {
            "pass": supervision_pass,
            "criterion": ("on the GT proxy (clean target), the full combined supervision "
                          "recovers >= 80% of the synthetic bump and ends <= 0.2 voxel; "
                          "silhouette-only is reported as a coverage-bound diagnostic"),
            "numbers": {"combo_gt": combo_gt, "silhouette_gt": sil_gt,
                        "combo_all": summary["p5"].get("combo")},
            "silhouette_note": ("silhouette constrains only the visible outline; on the "
                                "GT proxy it does not recover an interior bump by itself"),
        },
        "level3_real_reconstruction": {"pass": None, "status": "not_run",
                                        "reason": "gated by level 2; no VGGT-mesh optimisation was run"},
        "level4_final_task": {"pass": None, "status": "not_run",
                              "reason": "gated by level 3"},
    }
    summary["later_pass_note"] = ("Step 0.1 uses the symmetric silhouette loss; Step 0 numbers "
                                  "used foreground-only RGB and are not comparable.")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[summary] " + json.dumps(summary["gates"], indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
