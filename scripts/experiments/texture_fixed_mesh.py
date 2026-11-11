#!/usr/bin/env python3
"""固定 Mesh 的多视图纹理重建基线（MVS-Texturing 简化版）。

几何与位姿全部冻结，只做纹理：
    UV 展开 (xatlas) -> 逐 texel 反投影 -> 视锥/朝向/自遮挡判定
    -> 视角选择与融合 -> 接缝统计 -> 纹理化渲染 heldout 视角并评测。

三种纹理策略（共享同一套可见性判定，仅视角选择/融合不同）：
    naive_nearest : 每 texel 取相机中心最近的视角硬投影，无朝向/遮挡判定（朴素基线）
    best_view     : 每 texel 取入射角最优的单个可见视角（最接近 MVS-Texturing 的 label 选择）
    weighted_mean : 所有可见视角按 cos^2 加权平均（多视角融合）

说明：本实现用“加权平均 + gutter 膨胀”代替 MVS-Texturing 的全局 label 优化与泊松融合，
属于简化版，报告中必须显式标注；不做任何几何改动（不简化、不删连通块）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

SCENE = Path("/root/3dtex/data/scenes/nerf_lego_gt16")
HELDOUT = Path("/root/3dtex/outputs/lego_heldout_val16")
OUT_ROOT = Path("/root/3dtex/outputs/texture_fixed_mesh_20261105")

MESHES = {
    "vggt": Path("/root/3dtex/outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply"),
    "proxy": Path("/root/3dtex/data/scenes/nerf_lego_gt16/mesh/mesh_init.ply"),
}

VOXEL = 0.011501

# 运行时由 --scene / --heldout / --mesh-path / --voxel 覆盖（默认值保持上一条工作线的行为）
SCENE_DIR = SCENE
HELDOUT_DIR = HELDOUT
MESH_PATHS = dict(MESHES)


# ---------------------------------------------------------------- IO / 几何

def load_cameras(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for f in data["frames"]:
        out.append({
            "frame": f["frame"],
            "file": f.get("file", f["frame"]),
            "K": np.asarray(f["K_original"], np.float64),
            "T": np.asarray(f["T_world_to_cam"], np.float64),
            "W": int(f["original_width"]),
            "H": int(f["original_height"]),
        })
    return out


def load_images(cameras: list[dict], root: Path, device):
    rgbs, alphas = [], []
    for c in cameras:
        from PIL import Image
        with Image.open(root / c["file"]) as im:
            arr = np.asarray(im.convert("RGBA"), np.float32) / 255.0
        if arr.shape[0] != c["H"] or arr.shape[1] != c["W"]:
            raise RuntimeError(f"image size mismatch for {c['file']}: {arr.shape}")
        rgbs.append(torch.from_numpy(arr[..., :3].transpose(2, 0, 1)).to(device))
        alphas.append(torch.from_numpy(arr[..., 3]).to(device))
    return torch.stack(rgbs), torch.stack(alphas)


def load_mesh(path: Path, cache: Path):
    if cache.exists():
        d = np.load(cache)
        return d["vertices"].astype(np.float32), d["faces"].astype(np.int32)
    import trimesh

    t0 = time.time()
    mesh = trimesh.load(str(path), force="mesh", process=False)
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces, np.int32)
    print(f"[mesh] {path.name}: {len(v)} verts, {len(f)} faces, load {time.time() - t0:.1f}s")
    if not np.isfinite(v).all():
        raise SystemExit("mesh has NaN/Inf vertices")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, vertices=v, faces=f)
    return v, f


def unwrap_uv(verts: np.ndarray, faces: np.ndarray, cache: Path):
    if cache.exists():
        d = np.load(cache)
        return d["vmapping"], d["uv_faces"], d["uvs"]
    import xatlas

    t0 = time.time()
    vmapping, indices, uvs = xatlas.parametrize(verts, faces)
    vmapping = np.asarray(vmapping, np.int32)
    uv_faces = np.asarray(indices, np.int32)
    uvs = np.asarray(uvs, np.float32)
    print(f"[uv] xatlas: {len(uvs)} uv-verts, {len(uv_faces)} faces, {time.time() - t0:.1f}s, "
          f"range [{uvs.min():.4f}, {uvs.max():.4f}]")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, vmapping=vmapping, uv_faces=uv_faces, uvs=uvs)
    return vmapping, uv_faces, uvs


def vertex_charts(n: int, faces: np.ndarray):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    a = faces[:, [0, 1, 2]].reshape(-1)
    b = faces[:, [1, 2, 0]].reshape(-1)
    g = sp.coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)).tocsr()
    ncomp, labels = connected_components(g, directed=False)
    return labels.astype(np.int32), int(ncomp)


def vertex_normals(verts: np.ndarray, faces: np.ndarray):
    V = torch.from_numpy(verts).cuda().float()
    Ft = torch.from_numpy(faces).cuda().long()
    e1 = V[Ft[:, 1]] - V[Ft[:, 0]]
    e2 = V[Ft[:, 2]] - V[Ft[:, 0]]
    fn = torch.cross(e1, e2, dim=-1)
    vn = torch.zeros_like(V)
    for k in range(3):
        vn.index_add_(0, Ft[:, k], fn)
    vn = vn / torch.clamp(vn.norm(dim=-1, keepdim=True), min=1e-12)
    return vn


# ---------------------------------------------------------------- 光栅化

def uv_raster(ctx, uvs: np.ndarray, uv_faces: np.ndarray, res: int):
    """UV 三角形光栅化到 atlas；row = v 索引，top-left 原点。"""
    pos = np.zeros((len(uvs), 4), np.float32)
    pos[:, 0] = uvs[:, 0] * 2.0 - 1.0
    pos[:, 1] = uvs[:, 1] * 2.0 - 1.0
    pos[:, 3] = 1.0
    pos_t = torch.from_numpy(pos).cuda()
    tri_t = torch.from_numpy(uv_faces).cuda().to(torch.int32)
    rast, _ = dr.rasterize(ctx, pos_t[None], tri_t, resolution=[res, res])
    return rast


def build_clip(verts: np.ndarray, T: np.ndarray, K: np.ndarray, W: int, H: int,
               near: float = 0.001, far: float = 100.0):
    R, t = T[:3, :3], T[:3, 3]
    cam = verts @ R.T + t
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
    xc = (2.0 * fx / W) * x + (2.0 * cx / W - 1.0) * z
    yc = (2.0 * fy / H) * y + (2.0 * cy / H - 1.0) * z
    a = (far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    zc = a * z + b
    clip = np.stack([xc, yc, zc, z], axis=1).astype(np.float32)
    return clip, z.astype(np.float32)


def camera_raster(ctx, clip: np.ndarray, cam_z: np.ndarray, faces_t, W: int, H: int):
    clip_t = torch.from_numpy(clip).cuda()
    rast, _ = dr.rasterize(ctx, clip_t[None], faces_t, resolution=[H, W])
    z_t = torch.from_numpy(cam_z).cuda().reshape(1, -1, 1)
    assert z_t.shape[1] == len(clip), (z_t.shape, len(clip))
    zbuf, _ = dr.interpolate(z_t, rast, faces_t)
    return rast, zbuf[0, :, :, 0]


def sample_rgb(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor, W: int, H: int):
    gx = 2.0 * u / (W - 1) - 1.0
    gy = 2.0 * v / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
    out = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out[0, :, 0, :].T.contiguous()


def sample_map(mp: torch.Tensor, uv_px: torch.Tensor):
    """mp: [H,W]；uv_px: [...,2]；最近邻。"""
    gx = uv_px[..., 0] * 2.0 - 1.0
    gy = uv_px[..., 1] * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(mp[None, None], grid, mode="nearest", padding_mode="border",
                        align_corners=True)
    return out[0, 0]


# ---------------------------------------------------------------- 烘焙

def bake(ctx, uvs, uv_faces, uv_verts, verts, faces, cameras, images, res,
         min_cos=0.2, occ_tol_vox=2.0, chart_of_vert=None):
    t0 = time.time()
    rast = uv_raster(ctx, uvs, uv_faces, res)
    cov = rast[0, :, :, 3] > 0
    rows, cols = torch.nonzero(cov, as_tuple=True)
    tid = rast[0, :, :, 3][rows, cols].long() - 1
    bary = torch.stack([rast[0, :, :, 0][rows, cols],
                        rast[0, :, :, 1][rows, cols]], dim=-1)
    bary = torch.cat([bary, 1.0 - bary.sum(dim=-1, keepdim=True)], dim=-1)
    fverts = torch.from_numpy(uv_verts[uv_faces[tid.cpu().numpy()]]).cuda()
    world = (fverts * bary[:, :, None]).sum(dim=1)
    e1 = fverts[:, 1] - fverts[:, 0]
    e2 = fverts[:, 2] - fverts[:, 0]
    nrm = torch.cross(e1, e2, dim=-1)
    nrm = nrm / torch.clamp(nrm.norm(dim=-1, keepdim=True), min=1e-12)
    N = world.shape[0]
    print(f"[atlas] res {res}: {N} covered texels ({N / res ** 2:.3f} of atlas), "
          f"elapsed {time.time() - t0:.1f}s")

    acc = torch.zeros((N, 3), dtype=torch.float32, device="cuda")
    wsum = torch.zeros(N, dtype=torch.float32, device="cuda")
    best_w = torch.zeros(N, dtype=torch.float32, device="cuda")
    best_col = torch.zeros((N, 3), dtype=torch.float32, device="cuda")
    n_inside = torch.zeros(N, dtype=torch.int32, device="cuda")
    n_facing = torch.zeros(N, dtype=torch.int32, device="cuda")
    n_visible = torch.zeros(N, dtype=torch.int32, device="cuda")
    nearest_d = torch.full((N,), 1e10, dtype=torch.float32, device="cuda")
    nearest_i = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    faces_t = torch.from_numpy(faces).cuda().int()
    per_cam = []
    for ci, (cam, img) in enumerate(zip(cameras, images)):
        T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
        R = torch.from_numpy(T[:3, :3].T.copy()).cuda().float()
        t = torch.from_numpy(T[:3, 3].copy()).cuda().float()
        cam_xyz = world @ R + t
        z = cam_xyz[:, 2]
        u = cam_xyz[:, 0] / z * float(K[0, 0]) + float(K[0, 2])
        v = cam_xyz[:, 1] / z * float(K[1, 1]) + float(K[1, 2])
        inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        cam_pos = -T[:3, :3].T @ T[:3, 3]
        view = torch.from_numpy(cam_pos).cuda().float() - world
        view = view / torch.clamp(view.norm(dim=-1, keepdim=True), min=1e-12)
        cosang = (nrm * view).sum(dim=-1)
        facing = cosang > min_cos

        clip, camz = build_clip(verts, T, K, W, H)
        _, zbuf = camera_raster(ctx, clip, camz, faces_t, W, H)
        ui = torch.clamp(u, 0, W - 1).long()
        vi = torch.clamp(v, 0, H - 1).long()
        zi = zbuf[vi, ui]
        tol = occ_tol_vox * VOXEL + 0.01 * z
        occluded = (zi > 0) & ((z - zi) > tol)
        visible = inside & facing & ~occluded

        n_inside += inside.int()
        n_facing += (inside & facing).int()
        n_visible += visible.int()

        d = (world - torch.from_numpy(cam_pos).cuda().float()).norm(dim=-1)
        upd = d < nearest_d
        nearest_d = torch.where(upd, d, nearest_d)
        nearest_i = torch.where(upd, torch.full_like(nearest_i, ci), nearest_i)

        wgt = torch.where(visible, torch.clamp(cosang, min=0.0) ** 2,
                          torch.zeros_like(cosang))
        col = sample_rgb(img, torch.nan_to_num(u, nan=0.0), torch.nan_to_num(v, nan=0.0), W, H)
        acc += col * wgt[:, None]
        wsum += wgt
        better = wgt > best_w
        best_w = torch.where(better, wgt, best_w)
        best_col = torch.where(better[:, None], col, best_col)

        per_cam.append({
            "frame": cam["frame"],
            "visible_texels": int(visible.sum()),
            "visible_share": round(float(visible.float().mean()), 4),
            "inside_share": round(float(inside.float().mean()), 4),
        })
        print(f"  [cam {ci:02d}] {cam['frame']}: visible {int(visible.sum())} "
              f"({float(visible.float().mean()):.3f})")

    naive = torch.zeros((N, 3), dtype=torch.float32, device="cuda")
    naive_ok = torch.zeros(N, dtype=torch.bool, device="cuda")
    for ci, (cam, img) in enumerate(zip(cameras, images)):
        sel = nearest_i == ci
        if not bool(sel.any()):
            continue
        T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
        R = torch.from_numpy(T[:3, :3].T.copy()).cuda().float()
        t = torch.from_numpy(T[:3, 3].copy()).cuda().float()
        sub = world[sel]
        cam_xyz = sub @ R + t
        z = cam_xyz[:, 2]
        u = cam_xyz[:, 0] / z * float(K[0, 0]) + float(K[0, 2])
        v = cam_xyz[:, 1] / z * float(K[1, 1]) + float(K[1, 2])
        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        col = sample_rgb(img, torch.nan_to_num(u, nan=0.0), torch.nan_to_num(v, nan=0.0), W, H)
        naive[sel] = col * ok[:, None]
        naive_ok[sel] = ok
        print(f"  [naive {ci:02d}] assigned {int(sel.sum())}, valid {int(ok.sum())}")

    hist = {
        "0": int((n_visible == 0).sum()),
        "1": int((n_visible == 1).sum()),
        "2_3": int(((n_visible >= 2) & (n_visible <= 3)).sum()),
        "4_plus": int((n_visible >= 4).sum()),
    }
    stats = {
        "atlas_res": res,
        "covered_texels": int(N),
        "atlas_coverage": round(float(N / res ** 2), 4),
        "mean_visible_views": round(float(n_visible.float().mean()), 3),
        "visible_hist": hist,
        "invisible_reason": {
            "no_frustum": int((n_inside == 0).sum()),
            "backface_only": int(((n_inside > 0) & (n_facing == 0)).sum()),
            "occluded_only": int(((n_facing > 0) & (n_visible == 0)).sum()),
        },
        "naive_filled_share": round(float(naive_ok.float().mean()), 4),
        "per_camera": per_cam,
        "minutes": round((time.time() - t0) / 60.0, 2),
    }
    return {
        "rows": rows, "cols": cols, "tid": tid, "world": world, "normal": nrm,
        "res": res, "stats": stats, "vis_count": n_visible,
        "tex": {
            "naive_nearest": naive,
            "best_view": best_col,
            "weighted_mean": torch.where(wsum[:, None] > 0,
                                         acc / torch.clamp(wsum, min=1e-12)[:, None],
                                         torch.zeros_like(acc)),
        },
        "filled": {
            "naive_nearest": naive_ok,
            "best_view": best_w > 0,
            "weighted_mean": wsum > 0,
        },
    }


def build_atlas(rows, cols, res, values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    if values.ndim == 2:
        out = np.zeros((res, res, 3), np.float32)
        out[rows.cpu().numpy(), cols.cpu().numpy()] = values.astype(np.float32)
    else:
        out = np.zeros((res, res), np.float32)
        out[rows.cpu().numpy(), cols.cpu().numpy()] = values.astype(np.float32)
    return out


def dilate(texture, mask, iterations=4):
    filled = mask.copy()
    for _ in range(iterations):
        nb = np.zeros_like(texture)
        wgt = np.zeros_like(filled, dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sh = np.roll(np.roll(texture * filled[..., None], dy, axis=0), dx, axis=1)
            shm = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
            nb += sh
            wgt += shm
        grow = (~filled) & (wgt > 0)
        texture[grow] = nb[grow] / wgt[grow, None]
        filled |= grow
    return texture


def seam_stats(rows, cols, res, world, normal, chart_vert, uv_faces, tid, texture,
               rng_seed=20261105):
    """接缝色差：chart 边缘 texel 中，3D 距离 < 1.2 voxel、法线一致、chart 不同的一对 texel。"""
    from scipy.spatial import cKDTree

    rows_np = rows.cpu().numpy()
    cols_np = cols.cpu().numpy()
    chart_map = np.full((res, res), -1, np.int64)
    chart_map[rows_np, cols_np] = chart_vert[uv_faces[tid.cpu().numpy()][:, 0]]
    covered = chart_map >= 0
    nb_diff = np.zeros((res, res), bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        sh = np.roll(np.roll(covered, dy, axis=0), dx, axis=1)
        nb_diff |= covered & (~sh)
    by, bx = np.nonzero(nb_diff)
    if len(by) == 0:
        return {"pairs": 0}
    pos = world.detach().cpu().numpy()
    nrm = normal.detach().cpu().numpy()
    owner = np.full(res * res, -1, np.int64)
    owner[rows_np * res + cols_np] = np.arange(len(rows_np))
    bflat = by * res + bx
    cov_idx = owner[bflat]
    pts, colors, nrs = pos[cov_idx], texture[by, bx], nrm[cov_idx]
    if len(pts) > 200000:
        rng = np.random.default_rng(rng_seed)
        pick = rng.choice(len(pts), 200000, replace=False)
        pts, colors, nrs, bflat = pts[pick], colors[pick], nrs[pick], bflat[pick]
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=1.2 * VOXEL, output_type="ndarray")
    if len(pairs) == 0:
        return {"pairs": 0, "boundary_texels": int(len(pts))}
    chart_of_b = chart_map.ravel()[bflat]
    keep = chart_of_b[pairs[:, 0]] != chart_of_b[pairs[:, 1]]
    pairs = pairs[keep]
    if len(pairs) == 0:
        return {"pairs": 0, "boundary_texels": int(len(pts))}
    ndot = (nrs[pairs[:, 0]] * nrs[pairs[:, 1]]).sum(axis=1)
    pairs = pairs[ndot > 0.5]
    if len(pairs) == 0:
        return {"pairs": 0, "boundary_texels": int(len(pts))}
    d = np.abs(colors[pairs[:, 0]] - colors[pairs[:, 1]])
    dist = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1) / VOXEL
    return {
        "pairs": int(len(pairs)),
        "boundary_texels": int(len(pts)),
        "mean_abs_rgb": round(float(d.mean()), 5),
        "p95_abs_rgb": round(float(np.percentile(d, 95)), 5),
        "mean_pair_dist_vox": round(float(dist.mean()), 3),
    }


# ---------------------------------------------------------------- 顶点色基线

def bake_vertex_colors(ctx, verts, faces, cameras, images, vn, min_cos=0.2, occ_tol_vox=2.0):
    V = torch.from_numpy(verts).cuda().float()
    N = V.shape[0]
    acc = torch.zeros((N, 3), dtype=torch.float32, device="cuda")
    wsum = torch.zeros(N, dtype=torch.float32, device="cuda")
    faces_t = torch.from_numpy(faces).cuda().int()
    for cam, img in zip(cameras, images):
        T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
        R = torch.from_numpy(T[:3, :3].T.copy()).cuda().float()
        t = torch.from_numpy(T[:3, 3].copy()).cuda().float()
        cam_xyz = V @ R + t
        z = cam_xyz[:, 2]
        u = cam_xyz[:, 0] / z * float(K[0, 0]) + float(K[0, 2])
        v = cam_xyz[:, 1] / z * float(K[1, 1]) + float(K[1, 2])
        inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cam_pos = torch.from_numpy(-T[:3, :3].T @ T[:3, 3]).cuda().float()
        view = cam_pos[None] - V
        view = view / torch.clamp(view.norm(dim=-1, keepdim=True), min=1e-12)
        cosang = (vn * view).sum(dim=-1)
        facing = cosang > min_cos
        clip, camz = build_clip(verts, T, K, W, H)
        _, zbuf = camera_raster(ctx, clip, camz, faces_t, W, H)
        ui = torch.clamp(u, 0, W - 1).long()
        vi = torch.clamp(v, 0, H - 1).long()
        zi = zbuf[vi, ui]
        tol = occ_tol_vox * VOXEL + 0.01 * z
        visible = inside & facing & ~((zi > 0) & ((z - zi) > tol))
        wgt = torch.where(visible, torch.clamp(cosang, min=0.0) ** 2,
                          torch.zeros_like(cosang))
        col = sample_rgb(img, torch.nan_to_num(u, nan=0.0), torch.nan_to_num(v, nan=0.0), W, H)
        acc += col * wgt[:, None]
        wsum += wgt
    colors = torch.where(wsum[:, None] > 0,
                         acc / torch.clamp(wsum, min=1e-12)[:, None],
                         torch.zeros_like(acc))
    return colors


# ---------------------------------------------------------------- 评测

def psnr(pred: np.ndarray, gt: np.ndarray):
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(pred: np.ndarray, gt: np.ndarray):
    from skimage.metrics import structural_similarity as ssim_fn
    return float(ssim_fn(gt, pred, data_range=1.0, channel_axis=2))


def bbox_crop(fg: np.ndarray, pad=8):
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        return slice(0, fg.shape[0]), slice(0, fg.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, fg.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, fg.shape[1])
    return slice(y0, y1), slice(x0, x1)


def sample_uv(tex_chw: torch.Tensor, uv_px: torch.Tensor, W: int, H: int):
    gx = uv_px[..., 0] * 2.0 - 1.0
    gy = uv_px[..., 1] * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(tex_chw[None], grid, mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out[0].permute(1, 2, 0)


def render_and_eval(ctx, verts, faces, heldout_cams, heldout_imgs, heldout_alphas,
                    variants, vcolor_uv_colors, uv_attr, vis_map, faces_t, device):
    metrics = {k: [] for k in variants}
    renders = {}
    for ci, (cam, gt_img, gt_alpha) in enumerate(zip(heldout_cams, heldout_imgs, heldout_alphas)):
        T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
        clip, camz = build_clip(verts, T, K, W, H)
        rast, _ = camera_raster(ctx, clip, camz, faces_t, W, H)
        cov = rast[0, :, :, 3] > 0
        upix = dr.interpolate(uv_attr, rast, faces_t)[0][0]
        vis_px = sample_map(vis_map, upix)

        gt = gt_img.detach().cpu().numpy().transpose(1, 2, 0)
        fg_gt = gt_alpha.detach().cpu().numpy() > 0.5
        cov_np = cov.detach().cpu().numpy()
        vis_np = vis_px.detach().cpu().numpy()

        iou = float((fg_gt & cov_np).sum() / max((fg_gt | cov_np).sum(), 1))
        holes = fg_gt & (~cov_np)
        extra = cov_np & (~fg_gt)
        crop = bbox_crop(fg_gt | cov_np)

        for name, kind in variants.items():
            if kind == "vertex":
                pred_t = sample_vertex(vcolor_uv_colors, upix)
            else:
                pred_t = sample_uv(kind, upix, W, H)
            pred = pred_t.detach().cpu().numpy()
            pred = pred * cov_np[..., None]

            inter = fg_gt & cov_np
            seen = inter & (vis_np >= 1)
            unseen = inter & (vis_np < 1)
            m = {
                "psnr_full": psnr(pred, gt),
                "psnr_fg": psnr(pred[fg_gt], gt[fg_gt]),
                "psnr_inter": psnr(pred[inter], gt[inter]),
                "psnr_seen": psnr(pred[seen], gt[seen]) if seen.sum() > 0 else None,
                "psnr_unseen": psnr(pred[unseen], gt[unseen]) if unseen.sum() > 0 else None,
                "ssim_full": ssim(pred, gt),
                "ssim_crop": ssim(pred[crop], gt[crop]),
                "iou": iou,
                "fg_pixels": int(fg_gt.sum()),
                "inter_pixels": int(inter.sum()),
                "seen_pixels": int(seen.sum()),
                "unseen_pixels": int(unseen.sum()),
                "holes_pixels": int(holes.sum()),
                "holes_share": round(float(holes.sum() / max(fg_gt.sum(), 1)), 4),
                "extra_share": round(float(extra.sum() / max(fg_gt.sum(), 1)), 4),
                "mean_vis_in_fg": round(float(vis_np[inter].mean()), 3) if inter.sum() else None,
                "mean_abs_err": round(float(np.abs(pred - gt).mean()), 5),
            }
            metrics[name].append(m)
            renders.setdefault(name, []).append(pred)

    agg = {}
    for name, rows in metrics.items():
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
        agg[name] = {}
        for k in keys:
            vals = [r[k] for r in rows if r[k] is not None]
            agg[name][k] = round(float(np.mean(vals)), 4)
            agg[name][k + "_std"] = round(float(np.std(vals)), 4)
    return metrics, agg, renders


# ---------------------------------------------------------------- 主流程

def main():
    global SCENE_DIR, HELDOUT_DIR, MESH_PATHS, VOXEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", default="vggt")
    ap.add_argument("--atlas", default="2048")
    ap.add_argument("--variants", default="naive_nearest,best_view,weighted_mean,vertex_color")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--min-cos", type=float, default=0.2)
    ap.add_argument("--occ-tol-vox", type=float, default=2.0)
    ap.add_argument("--max-cams", type=int, default=0, help="冒烟用：限制训练相机数")
    ap.add_argument("--cam-indices", default="",
                    help="逗号分隔的训练相机索引（如 0,4,8）；与 --max-cams 互斥。"
                         "单参考视角扫描（Tex-T1 多视图条件准备）用它指定唯一参考相机")
    ap.add_argument("--max-heldout", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--check-train", type=int, default=0, help="在训练视角做投影自检，取前 N 个")
    ap.add_argument("--oracle", action="store_true",
                    help="几何上界：每个 heldout 视角只用它自己烘焙纹理再渲染该视角")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scene", default=str(SCENE),
                    help="训练场景根目录（需 geometry/gt/cameras.json 与 images/）")
    ap.add_argument("--heldout", default=str(HELDOUT),
                    help="heldout 根目录（需 cameras.json 与 images/，RGBA）")
    ap.add_argument("--mesh-path", action="append", default=[],
                    help="覆盖网格路径，形如 name=path（可重复）")
    ap.add_argument("--voxel", type=float, default=VOXEL,
                    help="几何体素尺度，用于遮挡容差与接缝邻域（默认 Lego 值）")
    args = ap.parse_args()

    SCENE_DIR = Path(args.scene)
    HELDOUT_DIR = Path(args.heldout)
    for spec in args.mesh_path:
        name, path = spec.split("=", 1)
        MESH_PATHS[name] = Path(path)
    VOXEL = args.voxel

    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = dr.RasterizeCudaContext()

    train_cams = load_cameras(SCENE_DIR / "geometry" / "gt" / "cameras.json")
    train_imgs, _ = load_images(train_cams, SCENE_DIR / "images", device)
    held_cams = load_cameras(HELDOUT_DIR / "cameras.json")
    held_imgs, held_alphas = load_images(held_cams, HELDOUT_DIR / "images", device)
    if args.max_cams and args.cam_indices:
        raise ValueError("--max-cams 与 --cam-indices 不能同时给出（选择规则会含混）")
    if args.cam_indices:
        idx = [int(x) for x in str(args.cam_indices).split(",") if x.strip()]
        missing = [i for i in idx if not (0 <= i < len(train_cams))]
        if missing:
            raise ValueError(f"--cam-indices 越界: {missing}（共 {len(train_cams)} 个训练相机）")
        train_cams = [train_cams[i] for i in idx]
        train_imgs = [train_imgs[i] for i in idx]
    if args.max_cams:
        train_cams, train_imgs = train_cams[:args.max_cams], train_imgs[:args.max_cams]
    if args.max_heldout:
        held_cams = held_cams[:args.max_heldout]
        held_imgs, held_alphas = held_imgs[:args.max_heldout], held_alphas[:args.max_heldout]
    print(f"[setup] train cams {len(train_cams)}, heldout cams {len(held_cams)}")

    resolutions = [int(x) for x in str(args.atlas).split(",")]
    variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    summary = {"train_cameras": len(train_cams),
               "train_cam_frames": [c["frame"] for c in train_cams],
               "heldout_cameras": len(held_cams),
               "meshes": {}, "runs": []}

    for mesh_name in [x.strip() for x in args.meshes.split(",") if x.strip()]:
        mesh_path = MESH_PATHS[mesh_name]
        cache_dir = out_dir / "cache"
        verts, faces = load_mesh(mesh_path, cache_dir / f"{mesh_name}.npz")
        vmapping, uv_faces, uvs = unwrap_uv(verts, faces, cache_dir / f"{mesh_name}_uv.npz")
        uv_verts = verts[vmapping]
        chart_vert, n_chart = vertex_charts(len(uvs), uv_faces)
        vn = vertex_normals(verts, faces)
        uv_attr = torch.from_numpy(uvs).cuda().float()[None]
        faces_t = torch.from_numpy(faces).cuda().int()
        faces_uv_t = torch.from_numpy(uv_faces).cuda().int()

        t0 = time.time()
        vcolors = bake_vertex_colors(ctx, verts, faces, train_cams, train_imgs, vn,
                                     args.min_cos, args.occ_tol_vox)
        vcolors_uv = vcolors[vmapping]
        vc_time = time.time() - t0
        print(f"[vcolor] {len(vcolors)} vertices, {vc_time:.1f}s")

        mesh_rec = {
            "mesh_path": str(mesh_path),
            "vertices": int(len(verts)), "faces": int(len(faces)),
            "uv_vertices": int(len(uvs)), "charts": n_chart,
            "vcolor_seconds": round(vc_time, 1),
            "resolutions": {},
        }

        for res in resolutions:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            baked = bake(ctx, uvs, uv_faces, uv_verts, verts, faces,
                         train_cams, train_imgs, res,
                         args.min_cos, args.occ_tol_vox, chart_vert)
            res_rec = {"bake_stats": baked["stats"]}

            variants = {}
            atlas_sizes = {}
            for name in variant_names:
                if name == "vertex_color":
                    continue
                if name not in baked["tex"]:
                    raise SystemExit(f"unknown variant {name}")
                tex = build_atlas(baked["rows"], baked["cols"], res, baked["tex"][name])
                filled = build_atlas(baked["rows"], baked["cols"], res,
                                     baked["filled"][name].float())
                tex = dilate(tex, filled > 0.5, iterations=4)
                variants[name] = tex
                atlas_sizes[name] = {
                    "filled_share_of_covered": round(float(baked["filled"][name].float().mean()), 4),
                }
                from PIL import Image
                Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8)).save(
                    out_dir / f"{mesh_name}_atlas{res}_{name}.png")
            res_rec["atlas_filled"] = atlas_sizes

            seam = seam_stats(baked["rows"], baked["cols"], res, baked["world"],
                              baked["normal"], chart_vert, uv_faces, baked["tid"],
                              variants["weighted_mean"]
                              if "weighted_mean" in variants else list(variants.values())[0])
            res_rec["seam"] = seam
            vis_map = build_atlas(baked["rows"], baked["cols"], res, baked["vis_count"])
            vis_map_t = torch.from_numpy(vis_map).cuda()
            print(f"[seam] {seam}")

            if not args.no_render:
                # 渲染 variants：纹理 + 顶点色
                render_variants = {}
                if "vertex_color" in variant_names:
                    vc_uv = vcolors_uv
                    render_variants["vertex_color"] = ("vertex", vc_uv)
                for name, tex in variants.items():
                    render_variants[name] = ("texture", torch.from_numpy(tex).cuda().permute(2, 0, 1).contiguous())

                metrics, agg, renders = eval_all(ctx, uv_verts, uv_faces, held_cams, held_imgs,
                                                 held_alphas, render_variants, uv_attr,
                                                 vis_map_t, faces_uv_t)
                res_rec["metrics"] = agg
                res_rec["per_view"] = {k: metrics[k] for k in metrics}
                if args.oracle:
                    oracle, oracle_per_view = eval_oracle(ctx, uvs, uv_faces, uv_verts, verts,
                                                          faces, held_cams, held_imgs, held_alphas,
                                                          res, args.min_cos, args.occ_tol_vox,
                                                          uv_attr, faces_uv_t)
                    res_rec["oracle_geometry_ceiling"] = oracle
                    res_rec["oracle_per_view"] = oracle_per_view
                    print(f"[oracle] {oracle}")
                # 保存若干视角的渲染对比图
                save_renders(out_dir, f"{mesh_name}_atlas{res}", held_imgs, renders, held_alphas, held_cams)
                if args.check_train:
                    tr_metrics, tr_agg, _ = eval_all(ctx, uv_verts, uv_faces,
                                                     train_cams[:args.check_train],
                                                     train_imgs[:args.check_train],
                                                     torch.stack([torch.ones_like(train_imgs[0, 0])] * args.check_train),
                                                     render_variants, uv_attr, vis_map_t, faces_uv_t)
                    res_rec["train_check"] = tr_agg
                    print("[train-check]", {k: v.get("psnr_full") for k, v in tr_agg.items()})
            res_rec["seconds"] = round(time.time() - t0, 1)
            res_rec["peak_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            mesh_rec["resolutions"][str(res)] = res_rec
            summary["runs"].append({"mesh": mesh_name, "atlas": res,
                                    "seconds": res_rec["seconds"]})
            print(f"[done] {mesh_name} atlas {res}: {res_rec['seconds']}s")

        summary["meshes"][mesh_name] = mesh_rec

    summary["min_cos"] = args.min_cos
    summary["occ_tol_vox"] = args.occ_tol_vox
    summary["simplified_note"] = "xatlas UV + 逐 texel 可见性/遮挡 + 加权平均/gutter 膨胀；未实现全局 label 优化与泊松融合"
    (out_dir / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[out] {out_dir / ('summary' + args.tag + '.json')}")


def eval_all(ctx, uv_verts, uv_faces, cams, imgs, alphas, variants, uv_attr, vis_map, faces_uv_t):
    """注意：属性（UV / 顶点色）按 uv 顶点索引，因此光栅化也用 uv 网格（几何等价）。"""
    metrics = {k: [] for k in variants}
    renders = {k: [] for k in variants}
    for cam, gt_img, gt_alpha in zip(cams, imgs, alphas):
        T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
        clip, camz = build_clip(uv_verts, T, K, W, H)
        rast, _ = camera_raster(ctx, clip, camz, faces_uv_t, W, H)
        cov = rast[0, :, :, 3] > 0
        upix = dr.interpolate(uv_attr, rast, faces_uv_t)[0][0]
        vis_px = sample_map(vis_map, upix)

        gt = gt_img.detach().cpu().numpy().transpose(1, 2, 0)
        fg_gt = gt_alpha.detach().cpu().numpy() > 0.5
        cov_np = cov.detach().cpu().numpy()
        vis_np = vis_px.detach().cpu().numpy()

        from scipy.ndimage import binary_erosion
        inner = binary_erosion(fg_gt, iterations=8)
        ring = fg_gt & (~inner)

        inter = fg_gt & cov_np
        seen = inter & (vis_np >= 1)
        unseen = inter & (vis_np < 1)
        holes = fg_gt & (~cov_np)
        extra = cov_np & (~fg_gt)
        crop = bbox_crop(fg_gt | cov_np)
        iou = float(inter.sum() / max((fg_gt | cov_np).sum(), 1))

        for name, (kind, val) in variants.items():
            if kind == "texture":
                pred_t = sample_uv(val, upix, W, H)
            else:
                pred_t = dr.interpolate(val[None], rast, faces_uv_t)[0][0]
            pred = pred_t.detach().cpu().numpy() * cov_np[..., None]
            m = {
                "psnr_full": round(psnr(pred, gt), 3),
                "psnr_fg": round(psnr(pred[fg_gt], gt[fg_gt]), 3) if fg_gt.sum() else None,
                "psnr_inter": round(psnr(pred[inter], gt[inter]), 3) if inter.sum() else None,
                "psnr_seen": round(psnr(pred[seen], gt[seen]), 3) if seen.sum() else None,
                "psnr_unseen": round(psnr(pred[unseen], gt[unseen]), 3) if unseen.sum() else None,
                "ssim_full": round(ssim(pred, gt), 4),
                "ssim_crop": round(ssim(pred[crop], gt[crop]), 4),
                "iou": round(iou, 4),
                "fg_pixels": int(fg_gt.sum()),
                "inter_pixels": int(inter.sum()),
                "seen_pixels": int(seen.sum()),
                "unseen_pixels": int(unseen.sum()),
                "holes_pixels": int(holes.sum()),
                "holes_share": round(float(holes.sum() / max(fg_gt.sum(), 1)), 4),
                "extra_share": round(float(extra.sum() / max(fg_gt.sum(), 1)), 4),
                "mean_vis_in_fg": round(float(vis_np[inter].mean()), 3) if inter.sum() else None,
                "mean_abs_err": round(float(np.abs(pred - gt).mean()), 5),
                "psnr_inner": round(psnr(pred[inner], gt[inner]), 3) if inner.sum() else None,
                "psnr_ring": round(psnr(pred[ring], gt[ring]), 3) if ring.sum() else None,
                "inner_share": round(float(inner.sum() / max(fg_gt.sum(), 1)), 4),
            }
            metrics[name].append(m)
            renders[name].append(pred)

    agg = {}
    for name, rows in metrics.items():
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
        agg[name] = {}
        for k in keys:
            vals = [r[k] for r in rows if r[k] is not None]
            agg[name][k] = round(float(np.mean(vals)), 4)
            agg[name][k + "_std"] = round(float(np.std(vals)), 4)
    return metrics, agg, renders


def eval_oracle(ctx, uvs, uv_faces, uv_verts, verts, faces, cams, imgs, alphas, res,
                min_cos, occ_tol_vox, uv_attr, faces_uv_t):
    """几何上界（oracle）：每个 heldout 视角只用它自己烘焙纹理，再以同一视角渲染。

    该变体的残差只可能来自“固定 Mesh 几何 + 固定位姿 + 投影模型”，
    因此可与多视角纹理结果对比，把误差拆成“几何上限”与“纹理整合损失”。
    """
    from scipy.ndimage import binary_erosion
    rows_out = []
    for cam, gt_img, gt_alpha in zip(cams, imgs, alphas):
        baked = bake(ctx, uvs, uv_faces, uv_verts, verts, faces, [cam], [gt_img], res,
                     min_cos, occ_tol_vox)
        tex = build_atlas(baked["rows"], baked["cols"], res, baked["tex"]["best_view"])
        filled = build_atlas(baked["rows"], baked["cols"], res,
                             baked["filled"]["best_view"].float())
        tex = dilate(tex, filled > 0.5, iterations=4)
        tex_t = torch.from_numpy(tex).cuda().permute(2, 0, 1).contiguous()

        T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
        clip, camz = build_clip(uv_verts, T, K, W, H)
        rast, _ = camera_raster(ctx, clip, camz, faces_uv_t, W, H)
        cov = rast[0, :, :, 3] > 0
        upix = dr.interpolate(uv_attr, rast, faces_uv_t)[0][0]
        cov_np = cov.detach().cpu().numpy()
        pred = sample_uv(tex_t, upix, W, H).detach().cpu().numpy() * cov_np[..., None]

        gt = gt_img.detach().cpu().numpy().transpose(1, 2, 0)
        fg_gt = gt_alpha.detach().cpu().numpy() > 0.5
        inter = fg_gt & cov_np
        inner = binary_erosion(fg_gt, iterations=8)
        rows_out.append({
            "frame": cam["frame"],
            "psnr_full": round(psnr(pred, gt), 3),
            "psnr_fg": round(psnr(pred[fg_gt], gt[fg_gt]), 3),
            "psnr_inter": round(psnr(pred[inter], gt[inter]), 3) if inter.sum() else None,
            "psnr_inner": round(psnr(pred[inner], gt[inner]), 3) if inner.sum() else None,
            "ssim_full": round(ssim(pred, gt), 4),
            "iou": round(float(inter.sum() / max((fg_gt | cov_np).sum(), 1)), 4),
            "covered_texels": int(len(baked["rows"])),
            "visible_texels": int(baked["vis_count"].sum()),
        })
    keys = [k for k, v in rows_out[0].items() if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows_out if r[k] is not None]
        agg[k] = round(float(np.mean(vals)), 4)
        agg[k + "_std"] = round(float(np.std(vals)), 4)
    return agg, rows_out


def save_renders(out_dir: Path, prefix: str, gt_imgs, renders, alphas, cams, n=4):
    from PIL import Image
    d = out_dir / "renders"
    d.mkdir(parents=True, exist_ok=True)
    names = list(renders.keys())
    for i in range(min(n, len(cams))):
        gt = (gt_imgs[i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cols = [gt]
        for nm in names:
            cols.append((np.clip(renders[nm][i], 0, 1) * 255).astype(np.uint8))
        gap = np.full((gt.shape[0], 4, 3), 255, np.uint8)
        row = cols[0]
        for c in cols[1:]:
            row = np.concatenate([row, gap, c], axis=1)
        Image.fromarray(row).save(d / f"{prefix}_{Path(cams[i]['frame']).stem}.png")


if __name__ == "__main__":
    main()
