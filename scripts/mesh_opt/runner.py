"""Local direct-mesh experiments. Fixed topology; geometric supervision only."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from mesh_opt.geometry import (displacement_loss, regularizer_graph, tolerance_loss,
                               make_lattice, lattice_offsets)
from mesh_opt.inputs import load_views


def main():
    from mesh_opt.render import NvdRenderer
    from ply_to_obj import read_ply
    from tsdf_mesh import write_ply
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--omega", default="data/scenes/nerf_lego16/geometry/omega")
    parser.add_argument("--gt", default="data/scenes/nerf_lego_gt16/geometry/gt")
    parser.add_argument("--camera-mode", choices=["diagnostic", "prediction"], required=True,
                        help="mesh must be in source gauge for diagnostic, Omega gauge for prediction")
    parser.add_argument("--mask-dir", help="external original-image-coordinate PNG masks")
    parser.add_argument("--images", help="RGBA images; alpha is an annotated mask, not inferred")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=.05, help="Adam displacement step in voxels")
    parser.add_argument("--voxel-size", type=float, required=True, help="voxel in the input mesh gauge")
    parser.add_argument("--lambda-sil", type=float, default=.3)
    parser.add_argument("--lambda-fs", type=float, default=.5)
    parser.add_argument("--lambda-reg", type=float, default=.1)
    parser.add_argument("--lambda-prior", type=float, default=0., help="tolerance depth prior weight")
    parser.add_argument("--tolerance-voxels", type=float, default=2.)
    parser.add_argument("--lambda-colmap", type=float, default=0.)
    parser.add_argument("--external-depth", help="controlled external scan with cameras.json and depth/")
    parser.add_argument("--lambda-external", type=float, default=0.)
    parser.add_argument("--pixel-center-offset", type=float, default=0.,
                        help="use 0.5 for integer-index camera coordinates on raster pixel centers")
    parser.add_argument("--perspective-projection", action="store_true")
    parser.add_argument("--colmap", default="outputs/colmap/sweep/anchor16/cost2_nc2")
    parser.add_argument("--anchor-poses", default="outputs/e2_anchor16_poses.json")
    parser.add_argument("--anchor-gate-voxels", type=float, default=4.)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--deformation", choices=["vertex", "lattice"], default="vertex")
    parser.add_argument("--lattice-spacing-voxels", type=float, default=8.)
    args = parser.parse_args()
    if args.voxel_size <= 0 or args.resolution % 16:
        raise ValueError("positive voxel-size and resolution divisible by 16 required")
    if args.lambda_external and not args.external_depth:
        raise ValueError("--lambda-external requires --external-depth")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "m2_log.json").exists():
        raise FileExistsError(f"completed run exists: {out}; choose a fresh output directory")
    vertices_np, faces_np, _ = read_ply(Path(args.mesh))
    initial = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int64, device=device)
    faces_i32 = faces.to(torch.int32).contiguous()
    # Optimise dimensionless voxel offsets; equal step semantics in either gauge.
    if args.deformation == "lattice":
        parameter, grid = make_lattice(initial, args.voxel_size, args.lattice_spacing_voxels)
        get_offsets = lambda: lattice_offsets(parameter, grid)
    else:
        parameter = torch.zeros_like(initial, requires_grad=True)
        get_offsets = lambda: parameter
    with torch.no_grad():
        edges, degree, weights = regularizer_graph(initial, faces)
    views, provenance = load_views(args, device)
    depths = torch.cat([v["depth"][v["depth"] > 0] for v in views])
    renderer = NvdRenderer(near=float(depths.min()) * .2, far=float(depths.max()) * 3,
                           perspective=args.perspective_projection)
    opt = torch.optim.Adam([parameter], lr=args.lr)
    omega_manifest = json.loads((Path(args.omega) / "manifest.json").read_text())
    log = {"args": vars(args), "provenance": provenance,
           "working_tree_diff_sha256": hashlib.sha256(subprocess.check_output(
               ["git", "diff", "HEAD", "--", "scripts", "tests"], stderr=subprocess.DEVNULL)).hexdigest(),
           "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "mesh_sha256": hashlib.sha256(Path(args.mesh).read_bytes()).hexdigest(),
           "model": omega_manifest.get("checkpoint"),
           "precision": {"optimisation": "float32", "initial_geometry": omega_manifest.get("aggregator_dtype")},
           "inputs": [v["name"] for v in views], "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(), "renderer": "nvdiffrast",
           "parameter_shape": list(parameter.shape), "iters": []}
    if args.lambda_external:
        external = Path(args.external_depth)
        log["external_observations"] = {
            "manifest": json.loads((external/"manifest.json").read_text()),
            "depth_sha256": {v["name"]: hashlib.sha256(
                (external/"depth"/f"{v['name']}.npy").read_bytes()).hexdigest() for v in views}}
    (out / "manifest.json").write_text(json.dumps(log, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for it in range(args.iters):
        opt.zero_grad()
        offsets = get_offsets()
        model_vertices = initial + args.voxel_size * offsets
        # Accumulate image gradients on one vertex leaf, then backpropagate once
        # through the deformation map. This avoids 16 retained grid-sampling graphs.
        vertices = model_vertices.detach().requires_grad_(True)
        values = dict(sil=0., fs=0., prior=0., colmap=0., external=0.)
        for view in views:
            raster_k = view["k"].clone()
            raster_k[:2, 2] += args.pixel_center_offset
            depth, alpha = renderer.render_frame(vertices, faces_i32, raster_k, view["t"], args.resolution)
            fg = view["fg"]
            terms = {"sil": (alpha - fg.float()).abs().mean(),
                     "fs": (alpha[~fg] ** 2).mean()}
            if args.lambda_prior:
                terms["prior"] = tolerance_loss(depth, view["depth"], view["confidence"],
                    fg & (view["depth"] > 0) & (alpha.detach() > .99),
                    args.tolerance_voxels * args.voxel_size, args.voxel_size)
            if args.lambda_colmap:
                terms["colmap"] = tolerance_loss(depth, view["anchor"], view["confidence"],
                    (view["anchor"] > 0) & (alpha.detach() > .99), 0., args.voxel_size)
            if args.lambda_external:
                terms["external"] = tolerance_loss(depth, view["external"], torch.ones_like(depth),
                    (view["external"] > 0) & (alpha.detach() > .99), 0., args.voxel_size)
            loss = sum(getattr(args, f"lambda_{key}") * val for key, val in terms.items()) / len(views)
            loss.backward()
            for key, val in terms.items():
                values[key] += float(val.detach()) / len(views)
        reg = displacement_loss(offsets, edges, degree, weights, 1.)
        torch.autograd.backward([model_vertices, args.lambda_reg * reg],
                                [vertices.grad, None])
        values["reg"] = float(reg.detach())
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"nonfinite vertex gradient at iteration {it + 1}")
        opt.step()
        if it == 0 or (it + 1) % args.print_every == 0 or it + 1 == args.iters:
            current = get_offsets().detach()
            delta = current.norm(dim=1)
            row = {"iter": it + 1, **values,
                   "loss": sum(getattr(args, f"lambda_{k}") * val for k, val in values.items()),
                   "delta_mean_voxels": float(delta.mean()), "delta_p90_voxels": float(delta.quantile(.9)),
                   "delta_max_voxels": float(delta.max()),
                   "mean_shift_voxels": current.mean(0).tolist()}
            log["iters"].append(row)
            print(json.dumps(row), flush=True)
    final = (initial + args.voxel_size * get_offsets().detach()).cpu().numpy()
    write_ply(out / "mesh_m2.ply", final, faces_np, np.full_like(final, .65), np.zeros_like(final))
    log["seconds"] = round(time.time() - started, 2)
    log["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    (out / "m2_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"[ok] {out}: {log['seconds']}s, {log['peak_vram_gb']} GiB", flush=True)
