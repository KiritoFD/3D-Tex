"""Operator-level determinism probes for the M2 optimizer (WSL CUDA + nvdiffrast).

This does not run the optimiser. It isolates the individual CUDA operations the
optimiser depends on and reports whether two repetitions of the *same* call on
the *same* input return bit-identical results. Hashes are byte-exact SHA256 of
float32 values.

Phases:
  ops    --operator isolation (index_add, grid_sample, raster, antialias, backward)
  report --rewrite report.json from the last ops run
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))

import numpy as np
import torch


def sha(tensor):
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def record(name, first, second, note=None):
    same = sha(first) == sha(second)
    difference = float((first.detach()-second.detach()).abs().max())
    row = {"probe": name, "repeat_identical": same, "max_abs_diff": difference,
           "sha256": [sha(first), sha(second)]}
    if note:
        row["note"] = note
    print(json.dumps(row), flush=True)
    return row


def probe_index_add(device, size=1_000_000, rows=100_000, seed=20261103):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    index = torch.randint(0, size, (rows,), generator=generator).to(device)
    value = torch.randn(rows, 3, generator=generator).to(device)
    buffer = torch.zeros(size, 3, device=device)
    buffer.index_add_(0, index, value)
    first = buffer
    buffer = torch.zeros(size, 3, device=device)
    buffer.index_add_(0, index, value)
    second = buffer
    rows_out = [record("index_add_forward", first, second,
                       "used by regularizer_graph and displacement_loss every step")]
    # Deterministic replacement candidate: sort by target index, cumulative sum, segment diff.
    def sorted_segments(index, value, size):
        order = torch.argsort(index, stable=True)
        targets = index[order]
        sorted_values = value[order]
        total = sorted_values.cumsum(0)
        boundaries = torch.nonzero(targets[1:] != targets[:-1]).squeeze(-1)
        ends = torch.cat([boundaries, torch.tensor([len(targets)-1], device=device)])
        shifted = torch.cat([torch.zeros(1, value.shape[1], device=device), total[ends[:-1]]], 0)
        segments = total[ends]-shifted
        out = torch.zeros(size, value.shape[1], device=device)
        out.index_copy_(0, targets[ends], segments)
        return out
    deterministic = sorted_segments(index, value, size)
    deterministic2 = sorted_segments(index, value, size)
    check = record("index_add_sorted_cumsum", deterministic, deterministic2,
                   "order-independent summation replacement candidate")
    check["max_abs_diff_vs_index_add"] = float((deterministic-first).abs().max())
    rows_out.append(check)
    return rows_out


def probe_grid_sample(device, seed=20261103):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    field = torch.randn(1, 3, 16, 16, 16, generator=generator).to(device).requires_grad_(True)
    grid = (torch.rand(4096, 1, 1, 3, generator=generator).to(device)*1.9-0.95).detach().unsqueeze(0)
    forward = []
    backward = []
    for _ in range(2):
        sample = torch.nn.functional.grid_sample(field, grid, mode="bilinear",
                                                 padding_mode="border", align_corners=True)
        forward.append(sample.detach().clone())
        grad, = torch.autograd.grad(sample.sum(), field, retain_graph=False)
        backward.append(grad.detach().clone())
    forward = record("grid_sample_forward", forward[0], forward[1],
                     "lattice_offsets in mesh_opt.geometry")
    backward = record("grid_sample_backward", backward[0], backward[1],
                      "documented nondeterministic CUDA scatter-add backward")
    return [forward, backward]


def probe_render(device, mesh_path, gt_path, resolution=256, seed=20261103):
    import nvdiffrast.torch as dr
    from ply_to_obj import read_ply
    vertices_np, faces_np, _ = read_ply(Path(mesh_path))
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int32, device=device)
    frames = json.loads((Path(gt_path)/"cameras.json").read_text())["frames"]
    frame = frames[0]
    k = torch.tensor(np.asarray(frame["K_model"]), dtype=torch.float32, device=device)
    t = torch.tensor(np.asarray(frame["T_world_to_cam"]), dtype=torch.float32, device=device)
    width = frame.get("model_width", 512)
    k[0] *= resolution/width
    k[1] *= resolution/width
    k[:2, 2] += .5
    glctx = dr.RasterizeCudaContext()
    near, far = 0.2, 30.

    def rasterize(v):
        cam = (t[:3, :3] @ v.transpose(-1, -2)).transpose(-1, -2)+t[:3, 3]
        z = cam[..., 2].clamp(min=1e-6)
        x_ndc = (2*k[0, 0]/resolution)*cam[..., 0]/z+(2*k[0, 2]/resolution-1)
        y_ndc = (2*k[1, 1]/resolution)*cam[..., 1]/z+(2*k[1, 2]/resolution-1)
        z_ndc = ((far+near)/(far-near)-(2*far*near)/(far-near)/z)
        clip = torch.stack([x_ndc*z, y_ndc*z, z_ndc*z, z], -1).contiguous()
        rast, _ = dr.rasterize(glctx, clip.unsqueeze(0), faces, resolution=[resolution, resolution])
        attr = torch.cat([cam[..., 2:3], torch.ones_like(cam[..., 2:3])], -1).contiguous()
        col, _ = dr.interpolate(attr.unsqueeze(0), rast, faces)
        return col, rast, clip

    # forward: full render (rasterize + interpolate + antialias)
    def full(v):
        col, rast, clip = rasterize(v)
        col = dr.antialias(col, rast, clip.unsqueeze(0), faces)
        alpha = col[0, ..., 1].clamp(0, 1)
        depth = torch.where(alpha > 1e-6, col[0, ..., 0]/alpha.clamp(min=1e-6), torch.zeros_like(alpha))
        return depth, alpha

    with torch.no_grad():
        first = full(vertices)
        second = full(vertices)
        raster_only = [rasterize(vertices)[0].detach() for _ in range(2)]
    rows = [record("rasterize_interpolate_forward", raster_only[0], raster_only[1],
                   "nvdiffrast rasterize+interpolate; no antialias"),
            record("render_depth_forward", first[0], second[0], "includes antialias"),
            record("render_alpha_forward", first[1], second[1], "includes antialias")]

    # backward: same leaf values rebuilt from a fixed copy, so only op order can differ
    base = vertices.detach().clone()
    grads = []
    for _ in range(2):
        leaf = base.clone().requires_grad_(True)
        depth, alpha = full(leaf)
        loss = (depth*alpha).sum()
        grad, = torch.autograd.grad(loss, leaf)
        grads.append(grad.detach().clone())
    rows.append(record("render_backward", grads[0], grads[1],
                       "antialias + interpolate backward; atomic accumulation suspect"))

    # antialias-free backward isolates whether antialias is the source
    grads_no_aa = []
    for _ in range(2):
        leaf = base.clone().requires_grad_(True)
        col, rast, clip = rasterize(leaf)
        alpha = col[0, ..., 1].clamp(0, 1)
        depth = torch.where(alpha > 1e-6, col[0, ..., 0]/alpha.clamp(min=1e-6), torch.zeros_like(alpha))
        grad, = torch.autograd.grad((depth*alpha).sum(), leaf)
        grads_no_aa.append(grad.detach().clone())
    rows.append(record("render_backward_without_antialias", grads_no_aa[0], grads_no_aa[1],
                       "same loss and depth formula, antialias removed"))
    return rows


def probe_blas(device, seed=20261103):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(1024, 1024, generator=generator).to(device)
    b = torch.randn(1024, 1024, generator=generator).to(device)
    first = a @ b
    second = a @ b
    return [record("cublas_matmul", first, second, "cuBLAS reduction order under fixed workspace")]


def probe_adam(device, seed=20261103):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameter = torch.randn(50_000, 3, generator=generator).to(device)
    grad = torch.randn(50_000, 3, generator=generator).to(device)
    results = []
    for _ in range(2):
        value = parameter.clone().requires_grad_(True)
        opt = torch.optim.Adam([value], lr=.05)
        value.grad = grad.clone()
        opt.step()
        results.append(value.detach().clone())
    return [record("adam_step", results[0], results[1], "elementwise update; expected deterministic")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["ops", "report"], default="ops")
    parser.add_argument("--config", default="configs/experiments/measurement_error_v2.json")
    parser.add_argument("--mesh", default="outputs/measurement_error_mm/initial_source.ply")
    parser.add_argument("--gt", default="data/scenes/measurement_precision/master/geometry/gt_ray256")
    parser.add_argument("--out", default="outputs/measurement_error_v2/repeatability/probe.json")
    args = parser.parse_args()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "report":
        print(target)
        return
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    rows = []
    rows += probe_index_add(device)
    rows += probe_grid_sample(device)
    rows += probe_blas(device)
    rows += probe_adam(device)
    rows += probe_render(device, args.mesh, args.gt)
    payload = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
               "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "probes": rows,
               "summary": {"nondeterministic": [r["probe"] for r in rows if not r["repeat_identical"]],
                           "deterministic": [r["probe"] for r in rows if r["repeat_identical"]]}}
    target.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
