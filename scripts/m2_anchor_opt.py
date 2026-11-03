#!/usr/bin/env python3
"""M2 anchor run: vertex-position optimisation of the v35c30+conf baseline mesh
with GEOMETRIC supervision only (zero RGB / zero photometric terms anywhere).

Supervision (all data already produced):
  L_colmap   hard depth anchors: COLMAP geometric-consistency-gated pixels
             (cost2_nc2 level, ~3.7% coverage) - render depth must match the
             gated depth along the same ray;
  L_crop     structure-region anchors: mixed-trajectory close-up depth
             (roofwin/wheels x 16, poses/depth from mixed_fusion output), only
             inside the crop foreground;
  L_sil      silhouette: soft alpha vs object mask (GT foreground protocol
             mask, the same one capture_eval applies);
  L_fs       free-space: soft alpha must vanish along background rays
             (attacks the burr cloud);
  L_reg      edge-aware displacement regulariser: L1 laplacian of the vertex
             displacement, down-weighted at initial creases so true edges can
             move freely; plus a tiny L2 prior against global drift/inflation.

Optimisation: Adam on vertex positions (no topology change). Displacement
statistics are logged to prove there is no global inflate/shrink mode.

    .venv/Scripts/python.exe scripts/m2_anchor_opt.py \
        --mesh outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply \
        --out outputs/m2/run_main --iters 400
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))
from ply_to_obj import read_ply  # noqa: E402
from verify_scene import umeyama  # noqa: E402
from tsdf_mesh import write_ply  # noqa: E402


# ---------------- differentiable soft rasterizer (from the M1 spike) ----------------
def project(verts3d, k, t):
    cam = (t[:3, :3] @ verts3d.transpose(-1, -2)).transpose(-1, -2) + t[:3, 3]
    z = cam[..., 2]
    u = cam[..., 0] / z.clamp(min=1e-6) * k[0, 0] + k[0, 2]
    v = cam[..., 1] / z.clamp(min=1e-6) * k[1, 1] + k[1, 2]
    return torch.stack([u, v], -1), z


def build_buckets(uv, z, width, height, tile, sigma, cache_path=None, max_span=8):
    """Tile -> candidate triangle indices. Long sliver triangles can span
    thousands of tiles and explode host RAM, so triangles whose tile span
    exceeds max_span are kept in a global list under key (-1, -1) that is
    appended to every tile's candidates by pixels_by_tile."""
    if cache_path and Path(cache_path).is_file():
        blob = np.load(cache_path)
        return {tuple(int(x) for x in kk.split("_")): torch.from_numpy(blob[kk]).to(uv.device)
                for kk in blob.files}
    with torch.no_grad():
        lo = uv.min(dim=1).values - sigma
        hi = uv.max(dim=1).values + sigma
        in_front = (z > 1e-6).any(dim=1)
        tile_lo = torch.floor(lo / tile).long().clamp(0, width // tile - 1)
        tile_hi = torch.floor(hi / tile).long().clamp(0, height // tile - 1)
        span = tile_hi - tile_lo  # (F,2)
        large = in_front & (span.amax(dim=1) > max_span)
        buckets = {(-1, -1): torch.nonzero(large).squeeze(-1).cpu().numpy().astype(np.int64)}
        for i in torch.nonzero(in_front & ~large).squeeze(-1).tolist():
            for ty in range(tile_lo[i, 1].item(), tile_hi[i, 1].item() + 1):
                for tx in range(tile_lo[i, 0].item(), tile_hi[i, 0].item() + 1):
                    buckets.setdefault((tx, ty), []).append(i)
        # nearest-first ordering: with the max_cand cap in soft_render, the kept
        # subset must be the closest triangles, not an arbitrary index range
        zmin = z.min(dim=1).values.cpu().numpy()
        out = {}
        for k, v in buckets.items():
            v = np.asarray(v, dtype=np.int64)
            out[k] = v[np.argsort(zmin[v], kind="stable")]
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **{f"{k[0]}_{k[1]}": v for k, v in out.items()})
    return {k: torch.from_numpy(v).to(uv.device) for k, v in out.items()}


def soft_render(vertices, faces, buckets, k, t, pixels, sigma=1.5, max_cand=256,
                temperature=30.0):
    """Soft depth + alpha at arbitrary pixels (differentiable in vertices)."""
    tri = vertices[faces]
    uv, z = project(tri, k, t)
    depth_out = torch.zeros(len(pixels), device=vertices.device)
    alpha_out = torch.zeros(len(pixels), device=vertices.device)
    for key, sel_list in pixels_by_tile(pixels, buckets).items():
        cand, sel = sel_list
        if len(cand) > max_cand:
            cand = cand[:max_cand]
        p = pixels[sel]
        uv_c = uv[cand]
        z_c = z[cand]
        v0, v1, v2 = uv_c[:, 0], uv_c[:, 1], uv_c[:, 2]
        den = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
        w0 = ((v1[:, 1] - v2[:, 1]) * (p[:, None, 0] - v2[None, :, 0])
              + (v2[:, 0] - v1[:, 0]) * (p[:, None, 1] - v2[None, :, 1])) / den
        w1 = ((v2[:, 1] - v0[:, 1]) * (p[:, None, 0] - v2[None, :, 0])
              + (v0[:, 0] - v2[:, 0]) * (p[:, None, 1] - v2[None, :, 1])) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        d_edge = torch.stack([w0, w1, w2], -1).clamp(max=0.0).abs().amax(dim=-1)
        weight = torch.where(inside, torch.ones_like(d_edge),
                             torch.exp(-((d_edge * 20.0 * sigma / 1.5) ** 2)))
        z_b = w0 * z_c[None, :, 0] + w1 * z_c[None, :, 1] + w2 * z_c[None, :, 2]
        score = weight / z_b.clamp(min=1e-6)
        attn = torch.softmax(score * temperature, dim=1)
        depth_out[sel] = (attn * z_b).sum(dim=1)
        alpha_out[sel] = weight.amax(dim=1)
    return depth_out, alpha_out


def pixels_by_tile(pixels, buckets):
    """Group pixel indices by their tile, returning {tile: (cand, sel)} lists.
    Large-span triangles live under key (-1, -1) and are appended everywhere."""
    groups = {}
    glob = buckets.get((-1, -1))
    with torch.no_grad():
        tile = (pixels / 16).long()
        keys, inv = torch.unique(tile, dim=0, return_inverse=True)
        for gi, key in enumerate(keys.tolist()):
            sel = torch.nonzero(inv == gi).squeeze(-1)
            cand = buckets.get(tuple(key))
            if glob is not None and len(glob):
                cand = glob if cand is None else torch.cat([cand, glob])
            if cand is not None and len(sel):
                groups[tuple(key)] = (cand, sel)
    return groups


class NvdRenderer:
    """nvdiffrast full-frame soft renderer (WSL acceleration path).

    render_frame returns (depth, alpha) images; losses gather the pixels they
    need. Gradients flow to vertex positions through barycentrics (depth) and
    through antialias coverage (alpha)."""

    def __init__(self, near=1.0, far=10.0):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext()
        self.near, self.far = near, far

    def render_frame(self, vertices, faces_i32, k, t, res):
        cam = (t[:3, :3] @ vertices.transpose(-1, -2)).transpose(-1, -2) + t[:3, 3]
        z = cam[..., 2].clamp(min=1e-6)
        w = z
        # nvdiffrast NDC: x right, y DOWN (verified vs GT foreground, IoU 0.947)
        x_ndc = (2 * k[0, 0] / res) * cam[..., 0] / z + (2 * k[0, 2] / res - 1)
        y_ndc = (2 * k[1, 1] / res) * cam[..., 1] / z + (2 * k[1, 2] / res - 1)
        z_ndc = (2 * z - self.near - self.far) / (self.far - self.near)
        clip = torch.stack([x_ndc * w, y_ndc * w, z_ndc * w, w], -1).contiguous()
        rast, _ = self.dr.rasterize(self.glctx, clip.unsqueeze(0), faces_i32,
                                    resolution=[res, res])
        attr = torch.cat([cam[..., 2:3], torch.ones_like(cam[..., 2:3])], -1).contiguous()
        col, _ = self.dr.interpolate(attr.unsqueeze(0), rast, faces_i32)
        col = self.dr.antialias(col, rast, clip.unsqueeze(0), faces_i32)
        return col[0, ..., 0], col[0, ..., 1].clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pixel-scale", type=float, default=0.5, help="0.5 = 256^2, 1.0 = 512^2")
    parser.add_argument("--sample-fg", type=int, default=3072)
    parser.add_argument("--sample-bg", type=int, default=1536)
    parser.add_argument("--lambda-colmap", type=float, default=1.0)
    parser.add_argument("--lambda-crop", type=float, default=1.0)
    parser.add_argument("--lambda-sil", type=float, default=0.3)
    parser.add_argument("--lambda-fs", type=float, default=0.5)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--lambda-prior", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--bucket-tile", type=int, default=16)
    parser.add_argument("--renderer", choices=["bucket", "nvd"], default="bucket",
                        help="bucket = pure-torch fallback (Windows); nvd = nvdiffrast (WSL)")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ps = args.pixel_scale
    HW = int(512 * ps)

    # ---- mesh (source gauge)
    vertices_np, faces_np, _ = read_ply(Path(args.mesh))
    vertices0 = torch.from_numpy(vertices_np).float().to(device)
    faces = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    vertices = vertices0.clone().requires_grad_(True)

    # ---- cameras: aligned predicted poses of anchor16 (frozen train-only Sim3)
    omega_dir = ROOT / "data/scenes/nerf_lego16/geometry/omega"
    gt_dir = ROOT / "data/scenes/nerf_lego_gt16/geometry/gt"
    om_frames = json.loads((omega_dir / "cameras.json").read_text())["frames"]
    gt_frames = json.loads((gt_dir / "cameras.json").read_text())["frames"]
    pts = np.asarray([f["T_world_to_cam"] for f in om_frames])
    gts = np.asarray([f["T_world_to_cam"] for f in gt_frames])
    gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
    pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
    scale, rot, trans = umeyama(pc, gc)
    aligned = pts.copy()
    aligned[:, :3, :3] = pts[:, :3, :3] @ rot.T
    aligned[:, :3, 3] = scale * pts[:, :3, 3] - aligned[:, :3, :3] @ trans
    names = [Path(f["frame"]).stem for f in gt_frames]
    n_views = len(gt_frames)

    views = []
    for i, f in enumerate(gt_frames):
        k = np.asarray(f["K_model"], dtype=np.float64).copy()
        k[0] *= ps
        k[1] *= ps
        fg_np = np.load(gt_dir / "foreground" / f"{names[i]}.npy").astype(np.float32)
        pool = max(1, fg_np.shape[0] // HW)
        fg_t = (F.max_pool2d(torch.from_numpy(fg_np)[None, None], pool, stride=pool)[0, 0] > 0.5).to(device)
        views.append({
            "name": names[i],
            "k": torch.from_numpy(k).float().to(device),
            "t": torch.from_numpy(aligned[i]).float().to(device),
            "fg": fg_t,
        })

    # ---- colmap gate anchors (cost2_nc2, 800^2 -> GT 512^2 nearest -> working res)
    import re

    def read_bin(path):
        with open(path, "rb") as handle:
            header = handle.read(32)
            m = re.match(rb"(\d+)&(\d+)&(\d+)&", header)
            w, h, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            handle.seek(m.end())
            data = np.frombuffer(handle.read(), dtype=np.float32)
        return data.reshape(h, w, c)

    colmap_dir = ROOT / "outputs/colmap/sweep/anchor16/cost2_nc2"
    for i, v in enumerate(views):
        stem = v["name"]
        cm = read_bin(colmap_dir / f"{stem}.png.geometric.bin")[..., 0]
        cm512 = cm[::2, ::2] if cm.shape[0] == 1024 else \
            torch.nn.functional.interpolate(
                torch.from_numpy(cm)[None, None], size=(512, 512), mode="nearest")[0, 0].numpy()
        rows, cols = np.nonzero(np.isfinite(cm512) & (cm512 > 0))
        px = torch.stack([torch.from_numpy(cols.astype(np.float32)),
                          torch.from_numpy(rows.astype(np.float32))], -1) * ps + ps / 2
        zt = torch.from_numpy(cm512[rows, cols]).float().to(device)
        views[i]["anchors_colmap"] = (px.to(device), zt)

    # ---- crop anchors (mixed_fusion output: refined poses + corrected depth)
    fus_geom = ROOT / "outputs/mixed_fusion/geom"
    fus_cameras = json.loads((fus_geom / "cameras.json").read_text())["frames"]
    merged_gt = ROOT / "outputs/mixed_fusion/gt"
    crop_views = []
    for f in fus_cameras:
        stem = Path(f["frame"]).stem
        if stem in names:
            continue
        k = np.asarray(f["K_model"], dtype=np.float64).copy()
        k[0] *= ps
        k[1] *= ps
        d = np.load(fus_geom / "depth" / f"{stem}.npy").squeeze().astype(np.float32)
        if ps != 1.0:
            d = torch.nn.functional.interpolate(
                torch.from_numpy(d)[None, None], scale_factor=ps, mode="nearest")[0, 0].numpy()
        fg = np.load(merged_gt / "foreground" / f"{stem}.npy").astype(np.float32)
        fg = torch.nn.functional.interpolate(
            torch.from_numpy(fg)[None, None], size=d.shape, mode="nearest")[0, 0].numpy() > 0.5
        rows, cols = np.nonzero(fg & (d > 0))
        px = torch.stack([torch.from_numpy(cols.astype(np.float32)),
                          torch.from_numpy(rows.astype(np.float32))], -1) + 0.5
        crop_views.append({
            "name": stem,
            "k": torch.from_numpy(k).float().to(device),
            "t": torch.from_numpy(np.asarray(f["T_world_to_cam"])).float().to(device),
            "px": px.to(device),
            "z": torch.from_numpy(d[rows, cols]).float().to(device),
        })

    # ---- tile buckets (cached on disk; bucket renderer only) ----
    if args.renderer == "bucket":
        bucket_dir = out / "buckets"
        for i, v in enumerate(views):
            uv0, z0 = project(vertices0[faces], v["k"], v["t"])
            cache = bucket_dir / f"view_{i}_{HW}.npz"
            views[i]["buckets"] = build_buckets(uv0, z0, HW, HW, args.bucket_tile, 1.5, cache)
        for v in crop_views:
            uv0, z0 = project(vertices0[faces], v["k"], v["t"])
            cache = bucket_dir / f"crop_{v['name']}_{HW}.npz"
            v["buckets"] = build_buckets(uv0, z0, HW, HW, args.bucket_tile, 1.5, cache)
    else:
        nvd = NvdRenderer()
        faces_i32 = faces.to(torch.int32).contiguous()

    # ---- neighbour structure for the regulariser (torch_sparse-free)
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.cat([edges, edges.flip(1)], dim=0)
    # initial crease weight: small mean-curvature => full regularisation, crease => less
    with torch.no_grad():
        v0f = vertices0[faces]
        fnorm = torch.cross(v0f[:, 1] - v0f[:, 0], v0f[:, 2] - v0f[:, 0], dim=1)
        fnorm = fnorm / fnorm.norm(dim=1, keepdim=True).clamp(min=1e-9)
        vnorm = torch.zeros_like(vertices0)
        vnorm.index_add_(0, faces[:, 0], fnorm)
        vnorm.index_add_(0, faces[:, 1], fnorm)
        vnorm.index_add_(0, faces[:, 2], fnorm)
        vnorm = vnorm / vnorm.norm(dim=1, keepdim=True).clamp(min=1e-9)
        cos_nn = (fnorm[faces[:, 0]] * fnorm[faces[:, 1]]).sum(-1).clamp(-1, 1)
        crease_v = (1 - cos_nn).clamp(min=0)
        edge_w = torch.ones(len(vertices0), device=device)
        edge_w.index_add_(0, faces[:, 0], crease_v)
        deg = torch.zeros(len(vertices0), device=device).index_add_(0, faces[:, 0], torch.ones_like(crease_v))
        crease_score = (edge_w / deg.clamp(min=1)).clamp(0, 2)
        reg_w = torch.exp(-crease_score * 4.0)  # 1 at flat, ~0 at creases

    nbr_sum = torch.zeros(len(vertices0), 3, device=device)
    deg = torch.zeros(len(vertices0), device=device)
    nbr_sum.index_add_(0, edges[:, 0], vertices0[edges[:, 1]])
    deg.index_add_(0, edges[:, 0], torch.ones(len(edges), device=device))

    opt = torch.optim.Adam([vertices], lr=args.lr)
    log = {"args": vars(args), "iters": []}
    started = time.time()
    ckpt_dir = out / "ckpt"
    for it in range(args.iters):
        opt.zero_grad()
        delta = vertices - vertices0
        losses = {}
        if args.renderer == "nvd":
            # full-frame renders; backward per view to bound autograd memory
            l_col = torch.zeros((), device=device)
            n_col = 0
            l_sil = torch.zeros((), device=device)
            l_fs = torch.zeros((), device=device)
            for i, v in enumerate(views):
                dep, alpha = nvd.render_frame(vertices, faces_i32, v["k"], v["t"], HW)
                parts = []
                px, zt = v["anchors_colmap"]
                if len(px):
                    pi = px.long()
                    zr = dep[pi[:, 1], pi[:, 0]]
                    rel = (zr - zt) / zt.clamp(min=1e-6)
                    lc = F.huber_loss(rel, torch.zeros_like(rel), delta=0.02)
                    parts.append(args.lambda_colmap * lc)
                    l_col = l_col + lc.detach() * len(px)
                    n_col += len(px)
                lsil = (alpha - v["fg"].float()).abs().mean()
                lfs = (alpha[~v["fg"]] ** 2).mean()
                parts.append(args.lambda_sil * lsil + args.lambda_fs * lfs)
                l_sil = l_sil + lsil.detach()
                l_fs = l_fs + lfs.detach()
                torch.autograd.backward(sum(parts))
            l_crop = torch.zeros((), device=device)
            n_crop = 0
            for v in crop_views:
                dep, _ = nvd.render_frame(vertices, faces_i32, v["k"], v["t"], HW)
                sel = torch.randint(0, len(v["px"]), (min(1024, len(v["px"])),), device=device)
                pi = v["px"][sel].long()
                zt = v["z"][sel]
                zr = dep[pi[:, 1], pi[:, 0]]
                rel = (zr - zt) / zt.clamp(min=1e-6)
                lc = F.huber_loss(rel, torch.zeros_like(rel), delta=0.02)
                (args.lambda_crop * lc).backward()
                l_crop = l_crop + lc.detach() * len(pi)
                n_crop += len(pi)
            losses["colmap"] = l_col / max(n_col, 1)
            losses["crop"] = l_crop / max(n_crop, 1)
            losses["sil"] = l_sil / len(views)
            losses["fs"] = l_fs / len(views)
            lap_num = torch.zeros(len(vertices0), 3, device=device)
            lap_num.index_add_(0, edges[:, 0], delta[edges[:, 1]])
            lap = delta - lap_num / deg.clamp(min=1).unsqueeze(-1)
            losses["reg"] = (reg_w.unsqueeze(-1) * lap.abs()).mean()
            losses["prior"] = (delta ** 2).mean()
            ((args.lambda_reg * losses["reg"] + args.lambda_prior * losses["prior"])).backward()
            total = (args.lambda_colmap * losses["colmap"] + args.lambda_crop * losses["crop"]
                     + args.lambda_sil * losses["sil"] + args.lambda_fs * losses["fs"]
                     + args.lambda_reg * losses["reg"] + args.lambda_prior * losses["prior"])
        else:
            # colmap anchors
            l = torch.zeros((), device=device)
            n = 0
            for i, v in enumerate(views):
                px, zt = v["anchors_colmap"]
                if len(px) == 0:
                    continue
                zr, _ = soft_render(vertices, faces, v["buckets"], v["k"], v["t"], px)
                rel = (zr - zt) / zt.clamp(min=1e-6)
                l = l + F.huber_loss(rel, torch.zeros_like(rel), delta=0.02) * len(px)
                n += len(px)
            losses["colmap"] = l / max(n, 1)
            # crop anchors (subsample per crop view)
            l = torch.zeros((), device=device)
            n = 0
            for v in crop_views:
                sel = torch.randint(0, len(v["px"]), (min(1024, len(v["px"])),), device=device)
                px, zt = v["px"][sel], v["z"][sel]
                zr, _ = soft_render(vertices, faces, v["buckets"], v["k"], v["t"], px)
                rel = (zr - zt) / zt.clamp(min=1e-6)
                l = l + F.huber_loss(rel, torch.zeros_like(rel), delta=0.02) * len(px)
                n += len(px)
            losses["crop"] = l / max(n, 1)
            # silhouette + free-space
            l_sil = torch.zeros((), device=device)
            l_fs = torch.zeros((), device=device)
            n = 0
            for i, v in enumerate(views):
                fg_idx = torch.nonzero(v["fg"], as_tuple=True)
                bg_idx = torch.nonzero(~v["fg"], as_tuple=True)
                fg_sel = torch.randint(0, len(fg_idx[0]), (args.sample_fg,), device=device)
                bg_sel = torch.randint(0, len(bg_idx[0]), (args.sample_bg,), device=device)
                pyx_fg = torch.stack([fg_idx[1][fg_sel], fg_idx[0][fg_sel]], -1).float() + 0.5
                pyx_bg = torch.stack([bg_idx[1][bg_sel], bg_idx[0][bg_sel]], -1).float() + 0.5
                px = torch.cat([pyx_fg, pyx_bg], 0)
                _, a = soft_render(vertices, faces, v["buckets"], v["k"], v["t"], px)
                target = torch.cat([torch.ones(len(pyx_fg), device=device),
                                    torch.zeros(len(pyx_bg), device=device)])
                l_sil = l_sil + (a - target).abs().mean()
                l_fs = l_fs + (a[len(pyx_fg):] ** 2).mean()
                n += 1
            losses["sil"] = l_sil / n
            losses["fs"] = l_fs / n
            # regulariser: edge-aware L1 laplacian of displacement + tiny prior
            lap_num = torch.zeros(len(vertices0), 3, device=device)
            lap_num.index_add_(0, edges[:, 0], delta[edges[:, 1]])
            lap = delta - lap_num / deg.clamp(min=1).unsqueeze(-1)
            losses["reg"] = (reg_w.unsqueeze(-1) * lap.abs()).mean()
            losses["prior"] = (delta ** 2).mean()
            total = (args.lambda_colmap * losses["colmap"] + args.lambda_crop * losses["crop"]
                     + args.lambda_sil * losses["sil"] + args.lambda_fs * losses["fs"]
                     + args.lambda_reg * losses["reg"] + args.lambda_prior * losses["prior"])
            total.backward()
        opt.step()
        if (it + 1) % args.print_every == 0 or it == 0:
            with torch.no_grad():
                d = delta.norm(dim=1)
                entry = {"iter": it + 1, "loss": float(total),
                         **{k: float(v) for k, v in losses.items()},
                         "sec_per_iter": round((time.time() - started) / (it + 1), 2),
                         "delta_mean": float(d.mean()), "delta_p50": float(d.quantile(0.5)),
                         "delta_p90": float(d.quantile(0.9)), "delta_max": float(d.max()),
                         "mean_shift": delta.mean(dim=0).tolist()}
                log["iters"].append(entry)
                print(json.dumps(entry), flush=True)
        if (it + 1) % args.ckpt_every == 0:
            with torch.no_grad():
                ckpt_dir.mkdir(exist_ok=True)
                cur = vertices.detach().cpu().numpy()
                write_ply(ckpt_dir / f"mesh_iter{it + 1:04d}.ply", cur, faces_np,
                          np.full_like(cur, 0.65), np.zeros_like(cur))
                (out / "m2_log_partial.json").write_text(json.dumps(log, indent=2) + "\n")

    mesh_out = out / "mesh_m2.ply"
    with torch.no_grad():
        final = vertices.cpu().numpy()
        fnorm_np = np.zeros_like(final)
    write_ply(mesh_out, final, faces_np, np.full_like(final, 0.65), fnorm_np)
    log["seconds"] = round(time.time() - started, 1)
    log["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "m2_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"[ok] mesh -> {mesh_out} ({log['seconds']}s, peak {log['peak_vram_gb']} GiB)")


if __name__ == "__main__":
    main()
