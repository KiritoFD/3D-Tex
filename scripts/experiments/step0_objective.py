#!/usr/bin/env python3
"""Step 0 objective-validity diagnostics for the image-driven surface plan.

0a fixes a mesh (GT-depth TSDF proxy or the VGGT/TSDF P0 mesh) and optimises a
factored appearance model only: per-vertex albedo, degree-2 SH diffuse
irradiance and a degree-2 SH specular environment.  Both meshes are rendered
with the same 16 Lego cameras and the same images; appearance is fitted on the
16 train views and scored on 4 held-out views.

0b keeps the appearance model frozen, perturbs vertices along their normals by
+/-0.5/1/2 voxels, and asks whether the photometric gradient points back to the
reference surface (GT mesh restoration, and VGGT->GT direction). 0c repeats the
same statistics using only highlight pixels.

Run from the repository root:

    PYTHONPATH=scripts /opt/conda/envs/3dtex/bin/python \
        scripts/experiments/step0_objective.py --out outputs/step0_20261105
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from mesh_opt.geometry import regularizer_graph, vertex_normals
from ply_to_obj import read_ply

Y00 = 0.28209479177387814


def sha256_16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def json_sha_16(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class GpuPeakSampler(threading.Thread):
    """External nvidia-smi memory.used peak, minus the idle value."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.idle_mib = self._used()
        self.peak_mib = self.idle_mib
        self._stop_event = threading.Event()
        self.samples = 0

    def _used(self):
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        return int(out)

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.peak_mib = max(self.peak_mib, self._used())
                self.samples += 1
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    @property
    def delta_mib(self):
        return self.peak_mib - self.idle_mib


class NvdRaster:
    """nvdiffrast renderer with world-space normals and per-vertex albedo."""

    def __init__(self, near=0.1, far=10.0):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext()
        self.near, self.far = near, far

    def render(self, vertices, faces_i32, normals, albedo, k, t, res):
        cam = (t[:3, :3] @ vertices.transpose(-1, -2)).transpose(-1, -2) + t[:3, 3]
        z = cam[..., 2].clamp(min=1e-6)
        x_ndc = (2 * k[0, 0] / res) * cam[..., 0] / z + (2 * k[0, 2] / res - 1)
        y_ndc = (2 * k[1, 1] / res) * cam[..., 1] / z + (2 * k[1, 2] / res - 1)
        z_ndc = (2 * z - self.near - self.far) / (self.far - self.near)
        clip = torch.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], -1).contiguous()
        rast, _ = self.dr.rasterize(self.glctx, clip.unsqueeze(0), faces_i32,
                                    resolution=[res, res])
        channels = [cam[..., 2:3], torch.ones_like(cam[..., 2:3]), normals, albedo]
        attr = torch.cat(channels, -1).contiguous()
        col, _ = self.dr.interpolate(attr.unsqueeze(0), rast, faces_i32)
        col = self.dr.antialias(col, rast, clip.unsqueeze(0), faces_i32)[0]
        alpha = col[..., 1].clamp(0, 1)
        alpha_safe = alpha.clamp(min=1e-3)
        depth = torch.where(alpha > 1e-6, col[..., 0] / alpha_safe,
                            torch.zeros_like(alpha))
        shading_normal = F.normalize(col[..., 2:5] / alpha_safe[..., None], dim=-1)
        albedo_pix = col[..., 5:8] / alpha_safe[..., None]
        # World-space positions and view directions.
        rows, cols = torch.meshgrid(torch.arange(res, device=depth.device),
                                    torch.arange(res, device=depth.device), indexing="ij")
        point_cam = torch.stack([(cols - k[0, 2]) / k[0, 0] * depth,
                                 (rows - k[1, 2]) / k[1, 1] * depth, depth], -1)
        rotation, translation = t[:3, :3], t[:3, 3]
        point_world = (point_cam - translation) @ rotation
        center_world = -translation @ rotation
        view_dir = F.normalize(point_world - center_world, dim=-1)
        # Orient interpolated normals toward the camera; winding is not trusted.
        facing = torch.sign((shading_normal * view_dir).sum(-1, keepdim=True) + 1e-12)
        shading_normal = shading_normal * facing
        return depth, alpha, shading_normal, albedo_pix, view_dir


def sh_basis(direction):
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    return torch.stack([
        0.28209479177387814 * torch.ones_like(x),
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        -1.0925484305920792 * y * z,
        0.9461746957575601 * z * z - 0.31539156525252005,
        -1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y),
    ], -1)


class Appearance(torch.nn.Module):
    """Per-vertex albedo + global SH diffuse irradiance + SH specular envmap."""

    def __init__(self, vertices):
        super().__init__()
        self.albedo = torch.nn.Parameter(torch.full_like(vertices, 0.5))
        self.sh_diffuse = torch.nn.Parameter(torch.zeros(9, 3, device=vertices.device))
        self.sh_specular = torch.nn.Parameter(torch.zeros(9, 3, device=vertices.device))
        self.spec_gain = torch.nn.Parameter(torch.tensor(0.1, device=vertices.device))
        with torch.no_grad():
            self.sh_diffuse[0] = 1.0 / Y00

    def shade(self, normal, view_dir, albedo):
        normal = F.normalize(normal, dim=-1)
        diffuse = (sh_basis(normal) @ self.sh_diffuse).clamp(min=0)
        reflected = 2 * (normal * view_dir).sum(-1, keepdim=True) * normal - view_dir
        specular = (sh_basis(F.normalize(reflected, dim=-1)) @ self.sh_specular).clamp(min=0)
        return albedo * diffuse + self.spec_gain * specular


def load_rgba(path: Path, resolution: int, device):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGBA").resize(
            (resolution, resolution), Image.BILINEAR), dtype=np.float32) / 255.0
    rgb = torch.from_numpy(array[..., :3].copy()).to(device)
    mask = torch.from_numpy(array[..., 3] > 0.5).to(device)
    return rgb, mask


def load_views(camera_json: Path, image_dir: Path, resolution: int, device, names=None,
               split_json: Path | None = None):
    payload = json.loads(Path(camera_json).read_text())
    frames = payload["frames"]
    if split_json is not None and Path(split_json).exists():
        split = json.loads(Path(split_json).read_text())
        wanted = {f"val_r_{i}.png" for i in split["indices"]}
        frames = [f for f in frames if f["file"] in wanted]
    views = []
    for frame in frames:
        if names is not None and Path(frame["file"]).stem not in names:
            continue
        stem = Path(frame["file"]).stem
        scale = resolution / float(frame["original_width"])
        k = torch.tensor(np.asarray(frame["K_original"], dtype=np.float64) * scale,
                         dtype=torch.float32, device=device)
        t = torch.tensor(np.asarray(frame["T_world_to_cam"], dtype=np.float64),
                         dtype=torch.float32, device=device)
        rgb, mask = load_rgba(Path(image_dir) / frame["file"], resolution, device)
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        values = luminance[mask]
        if values.numel() >= 8:
            threshold = max(0.60, float(torch.quantile(values, 0.90)))
        else:
            threshold = 0.8
        highlight = mask & (luminance >= threshold)
        strict = mask & (rgb.max(-1).values >= 0.95) & (luminance >= 0.80) \
            & (rgb.min(-1).values >= 0.55)
        views.append({
            "name": stem, "k": k, "t": t, "rgb": rgb, "mask": mask,
            "highlight": highlight, "strict_highlight": strict,
            "highlight_threshold": float(threshold),
            "luminance_mean": float(values.mean()) if values.numel() else 0.0,
        })
    return {"views": views, "camera_model": payload.get("camera_model"),
            "source": payload.get("source") or payload.get("protocol")}


def load_mesh(path: Path, device):
    vertices_np, faces_np, _ = read_ply(Path(path))
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int64, device=device)
    return {"path": str(path), "sha256_16": sha256_16(Path(path)),
            "vertices_np": vertices_np, "faces_np": faces_np,
            "vertices": vertices, "faces": faces,
            "faces_i32": faces.to(torch.int32).contiguous(),
            "normals": vertex_normals(vertices, faces).detach()}


def masked_stats(rgb, alpha, view):
    target, mask = view["rgb"], view["mask"]
    diff = (rgb - target).abs().mean(-1)
    full = float(diff[mask].mean()) if mask.any() else float("nan")
    core_mask = mask & (alpha.detach() > 0.95)
    core = float(diff[core_mask].mean()) if core_mask.any() else float("nan")
    mse = ((rgb - target) ** 2).mean(-1)[mask].mean()
    psnr = float(10 * torch.log10(1.0 / mse.clamp(min=1e-12))) if mask.any() else float("nan")
    rendered_mask = alpha.detach() > 0.5
    union = (rendered_mask | mask).sum()
    iou = float((rendered_mask & mask).sum()) / float(union) if union > 0 else float("nan")
    return {"l1": full, "l1_core": core, "psnr": psnr, "silhouette_iou": iou}


def photometric_loss(rgb, alpha, view, region="fg"):
    target, mask = view["rgb"], view["mask"]
    if region == "highlight":
        mask = view["highlight"]
    diff = (rgb - target).abs().mean(-1)
    return diff[mask].mean() if mask.any() else rgb.sum() * 0.0


def render_color(renderer, mesh_vertices, faces_i32, normals, albedo, appearance, view, res):
    depth, alpha, shading_normal, albedo_pix, view_dir = renderer.render(
        mesh_vertices, faces_i32, normals, albedo, view["k"], view["t"], res)
    color = appearance.shade(shading_normal, view_dir, albedo_pix)
    return color * alpha[..., None], alpha


def evaluate(mesh, appearance, views, renderer, res):
    records = []
    with torch.no_grad():
        for view in views:
            rgb, alpha = render_color(renderer, mesh["vertices"], mesh["faces_i32"],
                                      mesh["normals"], appearance.albedo, appearance,
                                      view, res)
            stats = masked_stats(rgb, alpha, view)
            if view["highlight"].any():
                hd = (rgb - view["rgb"]).abs().mean(-1)
                stats["l1_highlight"] = float(hd[view["highlight"]].mean())
                hmse = ((rgb - view["rgb"]) ** 2).mean(-1)[view["highlight"]].mean()
                stats["psnr_highlight"] = float(10 * torch.log10(1.0 / hmse.clamp(min=1e-12)))
            records.append({"view": view["name"], **stats,
                            "highlight_fraction": float(view["highlight"].float().mean())})
    keys = set(records[0]) - {"view"}
    for record in records[1:]:
        keys &= set(record)
    mean = {key: float(np.nanmean([r[key] for r in records])) for key in keys}
    return {"per_view": records, "mean": mean}


def direction_stats(gradient, target_direction, extra_valid=None):
    magnitude = gradient.norm(dim=1)
    target_norm = target_direction.norm(dim=1)
    valid = (magnitude > 1e-9) & (target_norm > 1e-6)
    if extra_valid is not None:
        valid &= extra_valid
    if not valid.any():
        return {"evaluable_vertices": 0, "correct_ratio": None, "cosine_mean": None}
    projection = (gradient * target_direction).sum(1)
    cosine = (-projection / (magnitude * target_norm).clamp(min=1e-12))[valid]
    selected_magnitude = magnitude[valid]
    result = {"evaluable_vertices": int(valid.sum()),
              "correct_ratio": float((cosine > 0).float().mean()),
              "cosine_mean": float(cosine.mean()),
              "cosine_median": float(cosine.median()),
              "cosine_p10": float(torch.quantile(cosine, 0.10)),
              "gradient_median": float(selected_magnitude.median()),
              "projection_fraction_mean": float(
                  (projection[valid].abs() / selected_magnitude.clamp(min=1e-12)).mean())}
    for quantile, name in ((0.50, "top50"), (0.25, "top25")):
        threshold = torch.quantile(selected_magnitude, quantile)
        subset = cosine[selected_magnitude >= threshold]
        result[f"correct_ratio_{name}"] = float((subset > 0).float().mean())
    return result


def albedo_smoothness(albedo, edges, degree, weights):
    neighbor = torch.zeros_like(albedo)
    neighbor.index_add_(0, edges[:, 0], albedo[edges[:, 1]])
    laplacian = albedo - neighbor / degree.clamp(min=1)[:, None]
    return (weights[:, None] * laplacian.abs()).mean()


def train_appearance(name, mesh, train_views, heldout_all_views, heldout_primary,
                     args, out_dir, renderer):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Appearance(mesh["vertices"])
    edges, degree, crease_weights = regularizer_graph(mesh["vertices"], mesh["faces"])
    groups = [
        {"params": [model.albedo], "lr": args.lr_albedo},
        {"params": [model.sh_diffuse, model.sh_specular, model.spec_gain],
         "lr": args.lr_light},
    ]
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=1e-4)
    sampler = GpuPeakSampler()
    sampler.start()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history, heldout_history = [], []
    final_heldout, last_full = None, float("nan")
    stop_reason = "max_iters"
    for iteration in range(args.iters):
        optimizer.zero_grad()
        full = core = iou = 0.0
        for view in train_views:
            rgb, alpha = render_color(renderer, mesh["vertices"], mesh["faces_i32"],
                                      mesh["normals"], model.albedo, model, view,
                                      args.resolution)
            loss = photometric_loss(rgb, alpha, view) / len(train_views)
            loss.backward()
            stats = masked_stats(rgb.detach(), alpha.detach(), view)
            full += stats["l1"] / len(train_views)
            core += stats["l1_core"] / len(train_views)
            iou += stats["silhouette_iou"] / len(train_views)
        if args.lambda_albedo:
            smooth = albedo_smoothness(model.albedo, edges, degree, crease_weights)
            (args.lambda_albedo * smooth).backward()
        else:
            smooth = torch.zeros(())
        if not torch.isfinite(model.albedo.grad).all():
            raise FloatingPointError(f"nonfinite albedo gradient at iteration {iteration + 1}")
        optimizer.step()
        scheduler.step()
        last_full = full
        if (iteration + 1) % 25 == 0 or iteration == 0:
            row = {"iter": iteration + 1, "train_l1": full, "train_l1_core": core,
                   "silhouette_iou": iou, "albedo_smooth": float(smooth),
                   "seconds": round(time.time() - started, 1),
                   "lr": optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(f"[{name}] " + json.dumps(row), flush=True)
        if (iteration + 1) % args.heldout_every == 0:
            final_heldout = evaluate(mesh, model, heldout_primary, renderer,
                                     args.resolution)
            heldout_history.append({"iter": iteration + 1, **final_heldout["mean"]})
            print(f"[{name}] heldout@{(iteration + 1)}: " +
                  json.dumps(final_heldout["mean"]), flush=True)
        if iteration + 1 >= args.min_iters and (iteration + 1) % 200 == 0:
            recent = [r["train_l1"] for r in history[-8:]]
            if len(recent) == 8 and (max(recent) - min(recent)) / max(recent) < 0.002:
                stop_reason = "converged"
                break
    final_heldout = evaluate(mesh, model, heldout_primary, renderer, args.resolution)
    heldout_all_result = evaluate(mesh, model, heldout_all_views, renderer,
                                  args.resolution)
    sampler.stop()
    seconds = round(time.time() - started, 1)
    torch.save({"albedo": model.albedo.detach().cpu(),
                "sh_diffuse": model.sh_diffuse.detach().cpu(),
                "sh_specular": model.sh_specular.detach().cpu(),
                "spec_gain": model.spec_gain.detach().cpu()},
               out_dir / "appearance.pt")
    manifest = {
        "name": name, "valid": True, "mesh": mesh["path"],
        "mesh_sha256_16": mesh["sha256_16"], "vertices": int(len(mesh["vertices"])),
        "faces": int(len(mesh["faces"])),
        "views_train": [v["name"] for v in train_views],
        "views_heldout_primary": [v["name"] for v in heldout_primary],
        "views_heldout_all": [v["name"] for v in heldout_all_views],
        "resolution": args.resolution, "iters": args.iters,
        "completed_iters": iteration + 1, "stopped": stop_reason,
        "optimizer": "Adam cosine", "lr_albedo": args.lr_albedo,
        "lr_light": args.lr_light, "lambda_albedo": args.lambda_albedo,
        "init": "albedo 0.5, SH DC 1/Y00, spec_gain 0.1",
        "appearance_params": {"albedo": list(model.albedo.shape),
                              "sh_diffuse": [9, 3], "sh_specular": [9, 3],
                              "spec_gain": 1},
        "peak_vram_torch_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "nvidia_smi_idle_mib": sampler.idle_mib,
        "nvidia_smi_peak_mib": sampler.peak_mib,
        "nvidia_smi_delta_mib": sampler.delta_mib,
        "seconds": seconds,
        "final_train_l1": last_full,
        "final_train_l1_core": history[-1]["train_l1_core"] if history else None,
        "final_train_silhouette_iou": history[-1]["silhouette_iou"] if history else None,
        "heldout_primary": final_heldout["mean"],
        "heldout_all": heldout_all_result["mean"],
        "history": history,
        "heldout_history": heldout_history,
        "heldout_curve": final_heldout["per_view"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{name}] done {seconds}s, torch peak "
          f"{manifest['peak_vram_torch_mib']} MiB, nvidia delta "
          f"{manifest['nvidia_smi_delta_mib']} MiB", flush=True)
    return model, manifest


def appearance_from_checkpoint(path: Path):
    state = torch.load(path, map_location="cpu")
    return state


def make_appearance_tensors(state, mesh, device):
    class Frozen:
        pass
    frozen = Frozen()
    frozen.albedo = state["albedo"].to(device)
    frozen.sh_diffuse = state["sh_diffuse"].to(device)
    frozen.sh_specular = state["sh_specular"].to(device)
    frozen.spec_gain = state["spec_gain"].to(device)

    def shade(normal, view_dir, albedo):
        normal = F.normalize(normal, dim=-1)
        diffuse = (sh_basis(normal) @ frozen.sh_diffuse).clamp(min=0)
        reflected = 2 * (normal * view_dir).sum(-1, keepdim=True) * normal - view_dir
        specular = (sh_basis(F.normalize(reflected, dim=-1)) @ frozen.sh_specular).clamp(min=0)
        return albedo * diffuse + frozen.spec_gain * specular
    frozen.shade = shade
    return frozen


def curvature_mask(mesh):
    tri = mesh["vertices"][mesh["faces"]]
    face_normal = F.normalize(torch.cross(tri[:, 1] - tri[:, 0],
                                          tri[:, 2] - tri[:, 0], dim=1), dim=1)
    disagreement = (1 - (face_normal * mesh["normals"][mesh["faces"]]).sum(-1)).clamp(0, 2)
    crease = torch.zeros(len(mesh["vertices"]), device=mesh["vertices"].device)
    for corner in range(3):
        crease.index_add_(0, mesh["faces"][:, corner], disagreement)
    return crease / 3.0


def gradient_case(mesh, appearance_state, train_views, delta_voxels, sign, args,
                  renderer, gt_tree, gt_vertices):
    voxel = args.voxel
    base = mesh["vertices"]
    normals = mesh["normals"]
    crease = curvature_mask(mesh)
    curved = crease > torch.median(crease)
    perturbed = (base + sign * delta_voxels * voxel * normals).detach()
    vertices = perturbed.clone().requires_grad_(True)
    appearance = make_appearance_tensors(appearance_state, mesh, base.device)
    sampler = GpuPeakSampler()
    sampler.start()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    gradient_full = torch.zeros_like(vertices)
    gradient_high = torch.zeros_like(vertices)
    highlight_views = 0
    for view in train_views:
        normals_t = vertex_normals(vertices, mesh["faces"])
        rgb, alpha = render_color(renderer, vertices, mesh["faces_i32"], normals_t,
                                  appearance.albedo, appearance, view, args.resolution)
        loss_full = photometric_loss(rgb, alpha, view)
        has_high = bool(view["highlight"].any())
        grad_full = torch.autograd.grad(loss_full, vertices, retain_graph=has_high)[0]
        gradient_full += grad_full
        if has_high:
            loss_high = photometric_loss(rgb, alpha, view, region="highlight")
            gradient_high += torch.autograd.grad(loss_high, vertices)[0]
            highlight_views += 1
    sampler.stop()
    seconds = round(time.time() - started, 2)
    restore_direction = -sign * normals
    full_stats = direction_stats(gradient_full, restore_direction)
    high_stats = direction_stats(gradient_high, restore_direction)
    curved_stats = direction_stats(gradient_full, restore_direction, curved)
    record = {
        "mesh": mesh["path"], "mesh_sha256_16": mesh["sha256_16"],
        "delta_voxels": delta_voxels, "sign": sign,
        "views": len(train_views), "resolution": args.resolution,
        "iters": 1, "peak_vram_torch_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "nvidia_smi_idle_mib": sampler.idle_mib, "nvidia_smi_peak_mib": sampler.peak_mib,
        "nvidia_smi_delta_mib": sampler.delta_mib, "seconds": seconds,
        "restore_toward_base": full_stats,
        "restore_toward_base_highlight_pixels": high_stats,
        "restore_toward_base_curved_vertices": curved_stats,
        "highlight_views": highlight_views,
        "final_loss": float(photometric_loss_sum(train_views, renderer, vertices,
                                                 mesh, appearance, args)),
        "valid": True,
    }
    if gt_tree is not None:
        points = vertices.detach().cpu().numpy()
        distance, index = gt_tree.query(points)
        target = torch.tensor(gt_vertices[index] - points, dtype=torch.float32,
                              device=base.device)
        farther = torch.tensor(distance > 0.25 * voxel, device=base.device)
        record["toward_gt_surface"] = direction_stats(gradient_full, target,
                                                      farther)
        record["toward_gt_surface_curved_vertices"] = direction_stats(
            gradient_full, target, farther & curved)
        record["toward_gt_surface_highlight_pixels"] = direction_stats(
            gradient_high, target, farther)
        record["distance_to_gt_voxels"] = {
            "mean": float(distance.mean() / voxel),
            "median": float(np.median(distance) / voxel),
            "p90": float(np.quantile(distance, 0.90) / voxel),
        }
        record["target_gt_note"] = "nearest GT proxy vertex; excludes vertices closer than 0.25 voxel"
    return record


def photometric_loss_sum(views, renderer, vertices, mesh, appearance, args):
    total = 0.0
    with torch.no_grad():
        for view in views:
            normals_t = vertex_normals(vertices, mesh["faces"])
            rgb, alpha = render_color(renderer, vertices, mesh["faces_i32"], normals_t,
                                      appearance.albedo, appearance, view, args.resolution)
            total += float(photometric_loss(rgb, alpha, view))
    return total / max(len(views), 1)


def restore_case(mesh, appearance_state, train_views, delta_voxels, sign, args,
                 renderer, gt_tree, gt_vertices):
    voxel = args.voxel
    base = mesh["vertices"]
    perturbed = (base + sign * delta_voxels * voxel * mesh["normals"]).detach()
    vertices = perturbed.clone().requires_grad_(True)
    appearance = make_appearance_tensors(appearance_state, mesh, base.device)
    optimizer = torch.optim.Adam([vertices], lr=args.restore_lr_voxels * voxel)
    base_points = base.detach().cpu().numpy()
    base_tree = cKDTree(base_points)
    sampler = GpuPeakSampler()
    sampler.start()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    curve = []

    def measure(label):
        points = vertices.detach().cpu().numpy()
        to_base = base_tree.query(points)[0]
        entry = {"label": label, "mean_to_base_voxels": float(to_base.mean() / voxel)}
        if gt_tree is not None:
            to_gt = gt_tree.query(points)[0]
            entry["mean_to_gt_voxels"] = float(to_gt.mean() / voxel)
            entry["median_to_gt_voxels"] = float(np.median(to_gt) / voxel)
        curve.append(entry)

    measure("start")
    for iteration in range(args.restore_iters):
        optimizer.zero_grad()
        for view in train_views:
            normals_t = vertex_normals(vertices, mesh["faces"])
            rgb, alpha = render_color(renderer, vertices, mesh["faces_i32"], normals_t,
                                      appearance.albedo, appearance, view, args.resolution)
            loss = photometric_loss(rgb, alpha, view) / len(train_views)
            loss.backward()
        optimizer.step()
        if (iteration + 1) % 5 == 0 or iteration + 1 == args.restore_iters:
            measure(iteration + 1)
    sampler.stop()
    decreases_base = curve[0]["mean_to_base_voxels"] - curve[-1]["mean_to_base_voxels"]
    valid = bool(np.isfinite(curve[-1]["mean_to_base_voxels"]))
    return {
        "mesh": mesh["path"], "mesh_sha256_16": mesh["sha256_16"],
        "delta_voxels": delta_voxels, "sign": sign,
        "views": len(train_views), "resolution": args.resolution,
        "iters": args.restore_iters + 1, "optimizer": f"Adam lr {args.restore_lr_voxels} voxel",
        "peak_vram_torch_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "nvidia_smi_idle_mib": sampler.idle_mib, "nvidia_smi_peak_mib": sampler.peak_mib,
        "nvidia_smi_delta_mib": sampler.delta_mib,
        "seconds": round(time.time() - started, 2),
        "curve": curve, "base_error_decrease_voxels": float(decreases_base),
        "valid": valid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--gt-mesh", default="data/scenes/nerf_lego_gt16/mesh/mesh_init.ply")
    parser.add_argument("--vggt-mesh", default="outputs/geometry_pruning/"
                        "eval_carve_v35c30_conf/carve_v35c30/mesh.ply")
    parser.add_argument("--gt-cameras", default="data/scenes/nerf_lego_gt16/geometry/gt/cameras.json")
    parser.add_argument("--train-images", default="data/scenes/nerf_lego16/images")
    parser.add_argument("--heldout-dir", default="outputs/lego_heldout_val16")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--min-iters", type=int, default=800)
    parser.add_argument("--heldout-every", type=int, default=500)
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--lr-albedo", type=float, default=0.05)
    parser.add_argument("--lr-light", type=float, default=0.02)
    parser.add_argument("--lambda-albedo", type=float, default=0.0,
                        help="crease-weighted albedo Laplacian regularizer")
    parser.add_argument("--voxel", type=float, default=0.01150144350298549)
    parser.add_argument("--deltas", default="0.5,1.0,2.0")
    parser.add_argument("--restore-iters", type=int, default=0,
                        help="perturbation restoration steps; 0 disables (0b gate uses directions)")
    parser.add_argument("--restore-lr-voxels", type=float, default=0.10)
    parser.add_argument("--restore-deltas", default="1.0")
    parser.add_argument("--stages", default="0a,0b,0c")
    parser.add_argument("--meshes", default="gt,vggt", help="gt,vggt subset for smoke runs")
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    out_root = Path(args.out)
    if (out_root / "summary.json").exists():
        raise FileExistsError(f"completed step0 output exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    selected = {s.strip() for s in args.meshes.split(",") if s.strip()}

    gt_mesh = load_mesh(Path(args.gt_mesh), device)
    vggt_mesh = load_mesh(Path(args.vggt_mesh), device)
    if gt_mesh["vertices"].shape == vggt_mesh["vertices"].shape:
        raise ValueError("GT and VGGT meshes must be distinct")
    train = load_views(Path(args.gt_cameras), Path(args.train_images),
                       args.resolution, device)
    heldout_all = load_views(Path(args.heldout_dir) / "cameras.json",
                             Path(args.heldout_dir) / "images", args.resolution, device,
                             split_json=Path(args.heldout_dir) / "split.json")
    count = min(args.heldout_count, len(heldout_all["views"]))
    indices = np.unique(np.linspace(0, len(heldout_all["views"]) - 1,
                                    count).round().astype(int))
    heldout_primary = {"views": [heldout_all["views"][i] for i in indices]}
    if not train["views"]:
        raise ValueError("no train views loaded")
    print(f"[data] train={len(train['views'])} heldout_primary="
          f"{[v['name'] for v in heldout_primary['views']]} "
          f"heldout_all={len(heldout_all['views'])}", flush=True)

    # Mesh diagonal, used for near/far bounds.
    all_points = torch.cat([gt_mesh["vertices"], vggt_mesh["vertices"]])
    diagonal = float((all_points.amax(0) - all_points.amin(0)).norm())
    renderer = NvdRaster(near=diagonal * 0.01, far=diagonal * 5)

    summary = {
        "schema": "3dtex-step0-objective-validity-v1",
        "date": "2026-11-05",
        "out": str(out_root),
        "scene": "Lego nerf_lego16",
        "resolution": args.resolution,
        "voxel": args.voxel,
        "seed": args.seed,
        "train_views": [v["name"] for v in train["views"]],
        "heldout_primary": [v["name"] for v in heldout_primary["views"]],
        "heldout_all": [v["name"] for v in heldout_all["views"]],
        "highlight_definition": "foreground luminance >= max(0.60, view p90)",
        "strict_highlight_definition": "fg & maxRGB>=0.95 & Y>=0.80 & minRGB>=0.55",
        "reference": {
            "gt_mesh": {"path": str(args.gt_mesh), "sha256_16": gt_mesh["sha256_16"],
                        "vertices": int(len(gt_mesh["vertices"]))},
            "gt_mesh_is_proxy": True,
            "proxy_note": "The repository has no analytic GT mesh. The reference is a "
                          "frozen TSDF mesh built from the 16 rendered GT depth maps "
                          "(data/scenes/nerf_lego_gt16/mesh/mesh_init.ply), i.e. a "
                          "GT-depth proxy, not the original Lego asset.",
            "vggt_mesh": {"path": str(args.vggt_mesh), "sha256_16": vggt_mesh["sha256_16"],
                          "vertices": int(len(vggt_mesh["vertices"]))},
            "cameras": str(args.gt_cameras),
            "camera_note": "Both meshes are already in the source/GT gauge (the VGGT "
                           "mesh was built after the train-only Sim(3)); GT train "
                           "cameras are used for both arms so the comparison isolates "
                           "appearance. Held-out cameras and images were never used in "
                           "reconstruction or alignment.",
        },
        "runs": {},
        "invalid_runs": [],
    }

    if "0a" in stages:
        for name, mesh in ((n, m) for n, m in (("gt", gt_mesh), ("vggt", vggt_mesh))
                           if n in selected):
            try:
                _, manifest = train_appearance(name, mesh, train["views"],
                                               heldout_all["views"],
                                               heldout_primary["views"], args,
                                               out_root / "0a" / name, renderer)
                summary["runs"][f"0a_{name}"] = manifest
            except Exception as error:
                summary["invalid_runs"].append(
                    {"run": f"0a_{name}", "reason": repr(error),
                     "traceback": traceback.format_exc()})

    if any(stage in stages for stage in ("0b", "0c")):
        gt_tree = cKDTree(gt_mesh["vertices_np"])
        for name, mesh in ((n, m) for n, m in (("gt", gt_mesh), ("vggt", vggt_mesh))
                           if n in selected):
            state_path = out_root / "0a" / name / "appearance.pt"
            if not state_path.exists():
                summary["invalid_runs"].append(
                    {"run": f"0b_{name}", "reason": "missing appearance.pt from 0a"})
                continue
            state = appearance_from_checkpoint(state_path)
            for delta in deltas:
                for sign in (1, -1):
                    label = f"0b_{name}_d{delta:g}_{'pos' if sign > 0 else 'neg'}"
                    try:
                        record = gradient_case(
                            mesh, state, train["views"], delta, sign, args, renderer,
                            None if name == "gt" else gt_tree,
                            gt_mesh["vertices_np"])
                        record["run"] = label
                        summary["runs"][label] = record
                        target_key = ("toward_gt_surface" if name == "vggt"
                                      else "restore_toward_base")
                        print(f"[{label}] {target_key}: "
                              f"{json.dumps(record[target_key])}", flush=True)
                    except Exception as error:
                        summary["invalid_runs"].append(
                            {"run": label, "reason": repr(error),
                             "traceback": traceback.format_exc()})
            if args.restore_iters > 0:
                for delta in [float(x) for x in args.restore_deltas.split(",")]:
                    for sign in (1, -1):
                        label = f"restore_{name}_d{delta:g}_{'pos' if sign > 0 else 'neg'}"
                        try:
                            record = restore_case(
                                mesh, state, train["views"], delta, sign, args,
                                renderer, None if name == "gt" else gt_tree,
                                gt_mesh["vertices_np"])
                            record["run"] = label
                            summary["runs"][label] = record
                            print(f"[{label}] " + json.dumps(record["curve"][-1]),
                                  flush=True)
                        except Exception as error:
                            summary["invalid_runs"].append(
                                {"run": label, "reason": repr(error),
                                 "traceback": traceback.format_exc()})

    # Gates.
    gates = {}
    gt_run = summary["runs"].get("0a_gt")
    vggt_run = summary["runs"].get("0a_vggt")
    if gt_run and vggt_run:
        gt_psnr = gt_run["heldout_primary"]["psnr"]
        vggt_psnr = vggt_run["heldout_primary"]["psnr"]
        gt_core = gt_run["heldout_primary"]["l1_core"]
        vggt_core = vggt_run["heldout_primary"]["l1_core"]
        gates["0a"] = {
            "gt_heldout_psnr": gt_psnr, "vggt_heldout_psnr": vggt_psnr,
            "psnr_margin_db": gt_psnr - vggt_psnr,
            "gt_heldout_l1_core": gt_core, "vggt_heldout_l1_core": vggt_core,
            "gt_train_l1": gt_run["final_train_l1"],
            "vggt_train_l1": vggt_run["final_train_l1"],
            "pass": bool(gt_psnr - vggt_psnr >= 1.0),
            "rule": "GT-mesh held-out PSNR exceeds VGGT-mesh by >= 1 dB",
        }
        gates["0a"]["gt_silhouette_iou"] = gt_run["heldout_primary"]["silhouette_iou"]
        gates["0a"]["vggt_silhouette_iou"] = vggt_run["heldout_primary"]["silhouette_iou"]
    cases = {key: value for key, value in summary["runs"].items()
             if key.startswith("0b_")}
    if cases:
        ratios = {key: value["toward_gt_surface"]["correct_ratio"]
                  for key, value in cases.items() if key.startswith("0b_vggt")}
        gt_ratios = {key: value["restore_toward_base"]["correct_ratio"]
                     for key, value in cases.items() if key.startswith("0b_gt")}
        high_ratios = {key: value["toward_gt_surface_highlight_pixels"]["correct_ratio"]
                       for key, value in cases.items() if key.startswith("0b_vggt")}
        gt_high_ratios = {key: value["restore_toward_base_highlight_pixels"]["correct_ratio"]
                          for key, value in cases.items() if key.startswith("0b_gt")}
        gates["0b"] = {
            "vggt_toward_gt_ratios": ratios,
            "gt_restore_ratios": gt_ratios,
            "vggt_highlight_ratios": high_ratios,
            "gt_highlight_ratios": gt_high_ratios,
            "pass": bool(all(v is not None and v >= 0.80 for v in ratios.values())
                         and all(v is not None and v >= 0.80 for v in gt_ratios.values())),
            "rule": "VGGT-mesh gradient points toward the GT proxy and GT-mesh "
                    "perturbations restore toward the base, >=80% for every delta/sign",
        }
        def finite_mean(values):
            values = [v for v in values if v is not None and np.isfinite(v)]
            return float(np.mean(values)) if values else float("nan")

        gates["0c"] = {
            "gt_highlight_cosine_mean": finite_mean(
                [v["restore_toward_base_highlight_pixels"]["cosine_mean"]
                 for v in cases.values() if v["run"].startswith("0b_gt")]),
            "vggt_highlight_cosine_mean": finite_mean(
                [v["toward_gt_surface_highlight_pixels"]["cosine_mean"]
                 for v in cases.values() if v["run"].startswith("0b_vggt")]),
            "gt_highlight_correct_ratio": finite_mean(
                [v["restore_toward_base_highlight_pixels"]["correct_ratio"]
                 for v in cases.values() if v["run"].startswith("0b_gt")]),
            "vggt_highlight_correct_ratio": finite_mean(
                [v["toward_gt_surface_highlight_pixels"]["correct_ratio"]
                 for v in cases.values() if v["run"].startswith("0b_vggt")]),
            "pass": None,
            "rule": "highlight gradients must not be systematically reversed "
                    "(cosine mean > 0 and correctness not below 50%)",
        }
        c0 = gates["0c"]["gt_highlight_cosine_mean"]
        c1 = gates["0c"]["vggt_highlight_cosine_mean"]
        r0 = gates["0c"]["gt_highlight_correct_ratio"]
        r1 = gates["0c"]["vggt_highlight_correct_ratio"]
        gates["0c"]["pass"] = bool(np.isfinite(c0) and np.isfinite(c1)
                                   and np.isfinite(r0) and np.isfinite(r1)
                                   and c0 > 0 and c1 > 0 and r0 >= 0.5 and r1 >= 0.5)
    if all(key in gates for key in ("0a", "0b", "0c")):
        gates["overall"] = "GO" if all(gates[key]["pass"] for key in
                                       ("0a", "0b", "0c")) else "NO_GO"
    summary["gates"] = gates
    summary["git"] = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"],
                                              text=True).strip()),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[summary] " + json.dumps({"gates": gates, "invalid": summary["invalid_runs"]}),
          flush=True)


if __name__ == "__main__":
    main()
