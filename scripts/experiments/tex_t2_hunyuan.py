#!/usr/bin/env python3
"""Tex-T2: Hunyuan3D-2.1 Paint (纹理生成) 在固定 Mesh 上的基线与四轴评测。

只做纹理生成（Paint），几何与位姿全部冻结：
  * 输入 Mesh：GT-depth 代理 Mesh / VGGT Mesh（顶点数与 T0 一致）
  * 参考图像：固定单张 GT 训练视图（默认 r_0.png），与 T1 保持同口径
  * 复用 scripts/experiments/texture_fixed_mesh.py 的相机、光栅化与采样设置

四轴指标：
  1) seen-view PSNR / SSIM / LPIPS（参考视图本身）
  2) unseen-view PSNR / SSIM / LPIPS + 跨视角一致性（16 heldout）
  3) 真实 3D seam-pair 色差（Hunyuan 自带 atlas 的 chart 边界）
  4) PBR 材质真实性：albedo/metallic/roughness 在固定光照下重渲误差

产物目录：outputs/tex_t2_hunyuan_20261105/
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

ROOT = Path("/root/3dtex")
HY3D_ROOT = ROOT / "third_party" / "hunyuan3d"
HY3D_PAINT = HY3D_ROOT / "hy3dpaint"
STUB = ROOT / "third_party" / "stubs"
SCENE = ROOT / "data" / "scenes" / "nerf_lego_gt16"
HELDOUT = ROOT / "outputs" / "lego_heldout_val16"
T0_ROOT = ROOT / "outputs" / "texture_fixed_mesh_20261105"
OUT_ROOT = ROOT / "outputs" / "tex_t2_hunyuan_20261105"

MESHES = {
    "proxy": SCENE / "mesh" / "mesh_init.ply",
    "vggt": ROOT / "outputs" / "geometry_pruning" / "eval_carve_v35c30_conf" / "carve_v35c30" / "mesh.ply",
}

VOXEL = 0.011501
REF_FRAME = "r_0.png"

# 固定光照（世界坐标，Z 轴向上）：主光在前上方，补光在另一侧
LIGHT_KEY = np.array([0.35, 0.75, 0.55], np.float32)
LIGHT_FILL = np.array([-0.55, -0.35, 0.45], np.float32)


# ---------------------------------------------------------------- 通用 IO

def json_dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回把单位向量 a 转到单位向量 b 的最小旋转矩阵。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        # 近似反向：绕任意正交轴转 180°
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def load_cameras(path: Path):
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


def cam_center(cam):
    T = cam["T"]
    return (-T[:3, :3].T @ T[:3, 3]).astype(np.float64)


def load_rgba(cams, root: Path):
    from PIL import Image
    rgbs, alphas = [], []
    for c in cams:
        with Image.open(root / c["file"]) as im:
            arr = np.asarray(im.convert("RGBA"), np.float32) / 255.0
        rgbs.append(arr[..., :3])
        alphas.append(arr[..., 3])
    return rgbs, alphas


# ---------------------------------------------------------------- Paint

def canonicalize_mesh(mesh_path: Path, ref_cam, out_dir: Path, tag: str):
    """把参考相机方向旋转到 Hunyuan 的 canonical 正面 (+Y, elev=0)。

    只做可逆刚体旋转，不改拓扑/缩放；paint 结束后用 R^T 逆变换回原世界系。
    """
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    V = np.asarray(mesh.vertices, np.float32)
    F = np.asarray(mesh.faces, np.int32)
    center = (V.max(0) + V.min(0)) / 2.0
    u_ref = cam_center(ref_cam) - center
    u_ref /= np.linalg.norm(u_ref)
    R = rotation_from_to(u_ref, np.array([0.0, 1.0, 0.0]))
    Vc = (R @ (V - center).T).T + center
    out = out_dir / f"{tag}_canonical.ply"
    trimesh.Trimesh(vertices=Vc, faces=F, process=False).export(str(out))
    meta = {
        "ref_frame": ref_cam["frame"],
        "cam_center": cam_center(ref_cam).round(6).tolist(),
        "mesh_center": center.round(6).tolist(),
        "rotation": np.round(R, 8).tolist(),
        "rot_deg": round(float(np.degrees(np.arccos(np.clip(
            np.dot(u_ref, np.array([0.0, 1.0, 0.0])), -1, 1)))), 3),
        "canonical_path": str(out),
    }
    return out, R, center, meta


def inverse_world_obj(src_obj: Path, R: np.ndarray, center: np.ndarray, dst_obj: Path):
    """把 canonical 帧的 textured OBJ 顶点逆变换回原世界系，保留 UV/MTL/贴图。"""
    import shutil
    dst_obj.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in Path(src_obj).read_text().splitlines():
        if line.startswith("v "):
            p = np.array([float(x) for x in line.split()[1:4]], np.float64)
            p = R.T @ (p - center) + center
            lines.append("v %.6f %.6f %.6f" % tuple(p))
        else:
            lines.append(line)
    dst_obj.write_text("\n".join(lines) + "\n")
    base_src = Path(src_obj).with_suffix("")
    base_dst = Path(dst_obj).with_suffix("")
    # Hunyuan 的 metallic/roughness 按 mesh_utils 默认写成 .jpg，diffuse 为 .jpg，mtl 同名
    for suffix in (".jpg", ".jpeg", ".png", ".mtl",
                   "_metallic.jpg", "_metallic.png", "_roughness.jpg", "_roughness.png"):
        s = Path(str(base_src) + suffix)
        if s.exists():
            shutil.copy2(s, Path(str(base_dst) + suffix))
    return dst_obj


def geometry_check(src_ply: Path, out_obj: Path) -> dict:
    """验证 paint 前后几何未被修改（xatlas 只重排顶点，不移动表面）。"""
    import trimesh
    from scipy.spatial import cKDTree
    a = trimesh.load(str(src_ply), force="mesh", process=False)
    b = trimesh.load(str(out_obj), force="mesh", process=False)
    va = np.asarray(a.vertices, np.float64)
    vb = np.asarray(b.vertices, np.float64)
    tree = cKDTree(va)
    d, _ = tree.query(vb, k=1, workers=-1)
    return {
        "src_vertices": int(len(va)),
        "out_vertices": int(len(vb)),
        "face_count_equal": bool(len(a.faces) == len(b.faces)),
        "bbox_delta_max": round(float(np.abs(a.bounds - b.bounds).max()), 7),
        "nearest_vertex_mean": round(float(d.mean()), 7),
        "nearest_vertex_p99": round(float(np.percentile(d, 99)), 7),
        "nearest_vertex_max": round(float(d.max()), 7),
    }


def run_paint(args, mesh_name: str, out_dir: Path):
    import torch
    # 让 hy3dpaint 的 cfg 相对路径可用
    os.chdir(HY3D_ROOT)
    sys.path.insert(0, str(STUB))
    sys.path.insert(0, str(HY3D_PAINT))
    from utils.torchvision_fix import apply_fix
    apply_fix()
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    train_cams = load_cameras(SCENE / "geometry" / "gt" / "cameras.json")
    ref_cam = [c for c in train_cams if c["frame"] == args.ref_frame]
    if not ref_cam:
        raise SystemExit(f"reference frame {args.ref_frame} not found")
    ref_cam = ref_cam[0]

    t0 = time.time()
    canon_path, R, center, cmeta = canonicalize_mesh(MESHES[mesh_name], ref_cam, out_dir, mesh_name)
    print(f"[canon] {mesh_name}: rot {cmeta['rot_deg']} deg -> {canon_path}", flush=True)

    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.multiview_cfg_path = str(HY3D_PAINT / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.realesrgan_ckpt_path = str(HY3D_PAINT / "ckpt" / "RealESRGAN_x4plus.pth")
    if Path(args.paint_model_path).is_dir():
        # multiview_utils 会把 snapshot_download 的返回值再拼上 hunyuan3d-paintpbr-v2-1/
        conf.multiview_pretrained_path = str(Path(args.paint_model_path).resolve().parent)
    if Path(args.dino_path).is_dir():
        conf.dino_ckpt_path = str(Path(args.dino_path).resolve())
    conf.render_size = args.render_size
    conf.texture_size = args.texture_size
    conf.merge_method = "fast"

    # 本地权重目录：直接让 snapshot_download 原样返回，避免联网与 HF API 限流
    import huggingface_hub
    _orig_sd = huggingface_hub.snapshot_download

    def _sd_local(repo_id, *a, **k):
        if os.path.isdir(str(repo_id)):
            return str(repo_id)
        return _orig_sd(repo_id, *a, **k)

    huggingface_hub.snapshot_download = _sd_local

    if Path(args.dino_path).is_dir():
        import transformers
        _orig = transformers.AutoModel.from_pretrained

        def _patched(name, *a, **k):
            if "dinov2" in str(name):
                name = args.dino_path
            return _orig(name, *a, **k)

        transformers.AutoModel.from_pretrained = _patched

    if args.cpu_offload:
        # 低显存模式：pipeline 先整载到 CPU，再手动只把 unet/vae/text_encoder 上卡
        import diffusers
        diffusers.DiffusionPipeline.to = lambda self, *a, **k: self
    if args.dino_cpu:
        try:
            if str(HY3D_PAINT) not in sys.path:
                sys.path.insert(0, str(HY3D_PAINT))
            from hunyuanpaintpbr.unet.modules import Dino_v2 as _Dino
            _Dino.to = lambda self, *a, **k: self
            print("[paint] Dino_v2 固定在 CPU（float32）", flush=True)
        except Exception as e:
            print(f"[paint] dino cpu patch failed: {type(e).__name__}: {e}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    paint = Hunyuan3DPaintPipeline(conf)
    print("[paint] models loaded", flush=True)
    if args.cpu_offload:
        inner = paint.models["multiview_model"].pipeline
        # image_encoder 在本 pipeline 中未被调用，直接释放
        if getattr(inner, "image_encoder", None) is not None:
            del inner.image_encoder
            gc.collect()
        for _n in ("unet", "vae", "text_encoder"):
            _m = getattr(inner, _n, None)
            if _m is not None:
                _m.to("cuda")
        # 文本编码只在 prompt 准备阶段用一次，用完立即退回 CPU，给 UNet 让出显存
        _ep = inner.encode_prompt
        _te = inner.text_encoder

        def _ep_lowmem(*a, **k):
            _te.to("cuda")
            out = _ep(*a, **k)
            _te.to("cpu")
            torch.cuda.empty_cache()
            return out

        inner.encode_prompt = _ep_lowmem
        print(f"[paint] low-mem placement enabled, device={inner.device}", flush=True)
    if args.dino_cpu and hasattr(paint.models["multiview_model"], "dino_v2"):
        mv = paint.models["multiview_model"]
        dino = mv.dino_v2.cpu().float()
        def _dino_cpu(x, _m=dino):
            if torch.is_tensor(x):
                x = x.cpu().float()
            out = _m(x)
            return out.half().cuda()
        mv.dino_v2 = _dino_cpu
        print("[paint] dino moved to CPU", flush=True)
    ref_img = SCENE / "images" / ref_cam["file"]
    canon_obj = out_dir / f"{mesh_name}_painted_canonical.obj"
    paint(
        mesh_path=str(canon_path),
        image_path=str(ref_img),
        output_mesh_path=str(canon_obj),
        use_remesh=False,
        save_glb=False,
    )
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    obj_world = inverse_world_obj(canon_obj, R, center, out_dir / f"{mesh_name}_painted.obj")
    gcheck = geometry_check(MESHES[mesh_name], obj_world)
    del paint
    gc.collect()
    torch.cuda.empty_cache()

    rec = {
        "mesh": mesh_name,
        "mesh_path": str(MESHES[mesh_name]),
        "reference_frame": ref_cam["frame"],
        "reference_image": str(ref_img),
        "resolution": args.resolution,
        "max_num_view": args.max_num_view,
        "steps": 15,
        "scheduler": "UniPC",
        "guidance_scale": 3.0,
        "render_size": args.render_size,
        "texture_size": args.texture_size,
        "atlas_saved": args.texture_size // 2,
        "use_remesh": False,
        "save_glb": False,
        "canonicalization": cmeta,
        "geometry_check": gcheck,
        "output_obj": str(obj_world),
        "seconds": round(time.time() - t0, 1),
        "peak_gpu_mib": round(peak, 1),
        "valid": True,
    }
    return rec


# ---------------------------------------------------------------- 渲染与采样

def make_ctx():
    import nvdiffrast.torch as dr
    return dr.RasterizeCudaContext()


def render_maps(ctx, uv_verts, uv_faces, uv_attr, cam, tex_chw, mr_chw=None):
    """返回 cov / albedo / world pos / world normal / metallic / roughness（像素级）。

    uv_verts / uv_faces 同时用于光栅化与位置/法线插值（uv 网格与几何等价）。
    """
    import torch
    import torch.nn.functional as F
    import nvdiffrast.torch as dr

    T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
    clip, camz = tfm.build_clip(uv_verts, T, K, W, H)
    rast, _ = tfm.camera_raster(ctx, clip, camz, uv_faces, W, H)
    cov = rast[0, :, :, 3] > 0
    upix = dr.interpolate(uv_attr, rast, uv_faces)[0][0]
    albedo = tfm.sample_uv(tex_chw, upix, W, H)

    pos = dr.interpolate(torch.from_numpy(uv_verts).cuda().float()[None], rast, uv_faces)[0][0]
    nrm = dr.interpolate(tfm.vertex_normals(uv_verts, uv_faces.cpu().numpy())[None], rast, uv_faces)[0][0]

    metal = rough = None
    if mr_chw is not None:
        gx = upix[..., 0] * 2.0 - 1.0
        gy = upix[..., 1] * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None]
        metal = F.grid_sample(mr_chw[0:1][None], grid, mode="bilinear", padding_mode="border",
                              align_corners=True)[0, 0].unsqueeze(-1)
        rough = F.grid_sample(mr_chw[1:2][None], grid, mode="bilinear", padding_mode="border",
                              align_corners=True)[0, 0].unsqueeze(-1)
    return {"cov": cov, "albedo": albedo, "pos": pos, "normal": nrm,
            "metallic": metal, "roughness": rough}


def shade_pbr(albedo, normal, pos, cam_pos, metallic, roughness, device="cuda"):
    """固定主光 + 补光 + 环境光的简化 Cook-Torrance，用于 PBR 重渲误差。"""
    import torch
    import torch.nn.functional as F

    N = F.normalize(normal, dim=-1)
    V = F.normalize(cam_pos[None, None, :] - pos, dim=-1)
    if metallic is None:
        metallic = torch.zeros_like(albedo[..., :1])
    if roughness is None:
        roughness = torch.full_like(albedo[..., :1], 0.9)
    rough = torch.clamp(roughness, 0.05, 1.0)
    F0 = 0.04 * (1.0 - metallic) + albedo * metallic

    out = albedo * 0.22  # ambient
    for ldir, power in ((LIGHT_KEY, 1.35), (LIGHT_FILL, 0.45)):
        L = torch.tensor(ldir, device=device, dtype=torch.float32)
        L = F.normalize(L, dim=0)[None, None, :]
        Hh = F.normalize(L + V, dim=-1)
        ndl = torch.clamp((N * L).sum(-1, keepdim=True), 0.0, 1.0)
        ndv = torch.clamp((N * V).sum(-1, keepdim=True), 1e-3, 1.0)
        ndh = torch.clamp((N * Hh).sum(-1, keepdim=True), 0.0, 1.0)
        vdh = torch.clamp((V * Hh).sum(-1, keepdim=True), 0.0, 1.0)
        a = rough ** 2
        a2 = a * a
        D = a2 / (np.pi * (ndh ** 2 * (a2 - 1.0) + 1.0) ** 2 + 1e-8)
        k = (rough + 1.0) ** 2 / 8.0
        G = (ndl / (ndl * (1.0 - k) + k + 1e-8)) * (ndv / (ndv * (1.0 - k) + k + 1e-8))
        fres = F0 + (1.0 - F0) * (1.0 - vdh) ** 5
        spec = D * G * fres / (4.0 * ndl * ndv + 1e-6)
        diffuse = albedo * (1.0 - metallic) * ndl / np.pi
        out = out + power * (diffuse + spec)
    return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------- 指标

def psnr_np(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def ssim_np(pred, gt):
    from skimage.metrics import structural_similarity as ssim_fn
    return float(ssim_fn(gt, pred, data_range=1.0, channel_axis=2))


def bbox_crop(fg, pad=8):
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        return slice(0, fg.shape[0]), slice(0, fg.shape[1])
    return (slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, fg.shape[0])),
            slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, fg.shape[1])))


def lpips_fn(model, pred, gt):
    import torch
    crop = bbox_crop((gt.max(2) > 0.02) | (pred.max(2) > 0.02))
    a = torch.from_numpy(pred[crop]).permute(2, 0, 1)[None].cuda() * 2 - 1
    b = torch.from_numpy(gt[crop]).permute(2, 0, 1)[None].cuda() * 2 - 1
    with torch.no_grad():
        return float(model(a, b).item())


def eval_views(views, cams, gts, alphas, lpips_model, method, split):
    rows = []
    for v, cam, gt, alpha in zip(views, cams, gts, alphas):
        pred = v["albedo"].detach().cpu().numpy() * v["cov"].detach().cpu().numpy()[..., None]
        fg = alpha > 0.5
        cov = v["cov"].detach().cpu().numpy()
        inter = fg & cov
        crop = bbox_crop(fg | cov)
        m = {
            "frame": cam["frame"],
            "psnr_full": round(psnr_np(pred, gt), 3),
            "psnr_fg": round(psnr_np(pred[fg], gt[fg]), 3) if fg.sum() else None,
            "psnr_inter": round(psnr_np(pred[inter], gt[inter]), 3) if inter.sum() else None,
            "ssim_full": round(ssim_np(pred, gt), 4),
            "ssim_crop": round(ssim_np(pred[crop], gt[crop]), 4),
            "lpips_crop": round(lpips_fn(lpips_model, pred, gt), 4),
            "iou": round(float(inter.sum() / max((fg | cov).sum(), 1)), 4),
            "holes_share": round(float((fg & ~cov).sum() / max(fg.sum(), 1)), 4),
            "extra_share": round(float((cov & ~fg).sum() / max(fg.sum(), 1)), 4),
            "fg_pixels": int(fg.sum()),
        }
        rows.append(m)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        agg[k] = round(float(np.mean(vals)), 4)
        agg[k + "_std"] = round(float(np.std(vals)), 4)
    return rows, agg


def cross_view_consistency(verts, faces, cams, views, n_points=30000, seed=20261105):
    """在网格表面采样点，投影到各 heldout 视角，比较同一 3D 点渲染色的跨视角差异。"""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    probs = areas / areas.sum()
    idx = rng.choice(len(faces), n_points, p=probs)
    u = rng.random(n_points, dtype=np.float32)
    v = rng.random(n_points, dtype=np.float32)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    tri_s = tri[idx]
    pts = tri_s[:, 0] + u[:, None] * (tri_s[:, 1] - tri_s[:, 0]) + v[:, None] * (tri_s[:, 2] - tri_s[:, 0])

    samples, valids = [], []
    imgs = [vv["albedo"].detach().cpu().numpy() for vv in views]
    for cam, img in zip(cams, imgs):
        T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
        xyz = pts @ T[:3, :3].T + T[:3, 3]
        z = xyz[:, 2]
        uu = xyz[:, 0] / np.maximum(z, 1e-8) * K[0, 0] + K[0, 2]
        vv = xyz[:, 1] / np.maximum(z, 1e-8) * K[1, 1] + K[1, 2]
        ok = (z > 1e-6) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        col = np.zeros((len(pts), 3), np.float32)
        if ok.any():
            gx = 2.0 * uu[ok] / (W - 1) - 1.0
            gy = 2.0 * vv[ok] / (H - 1) - 1.0
            grid = torch.stack([torch.from_numpy(gx), torch.from_numpy(gy)], -1)[None, None].float().cuda()
            im = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda()
            res = F.grid_sample(im, grid, mode="bilinear", padding_mode="border", align_corners=True)
            col[ok] = res[0].permute(1, 2, 0)[0].cpu().numpy()
        samples.append(col)
        valids.append(ok)
    diffs = []
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            both = valids[i] & valids[j]
            if both.sum() < 50:
                continue
            d = np.abs(samples[i][both] - samples[j][both]).mean(1)
            diffs.append(d)
    if not diffs:
        return {"pairs": 0}
    d = np.concatenate(diffs)
    return {
        "surface_points": int(len(pts)),
        "pairs": int(len(diffs)),
        "pair_samples": int(len(d)),
        "mean_abs_rgb": round(float(d.mean()), 5),
        "p95_abs_rgb": round(float(np.percentile(d, 95)), 5),
    }


def seam_for_mesh(verts, faces, uvs, atlas1024, ctx):
    """Hunyuan 自带 atlas：xatlas 式 chart 边界 + 3D 近邻配对色差（与 T0 同口径）。"""
    import torch
    import nvdiffrast.torch as dr

    uvs = np.asarray(uvs, np.float32)
    uv_faces = np.asarray(faces, np.int32)
    uvs4 = np.zeros((len(uvs), 4), np.float32)
    uvs4[:, 0] = uvs[:, 0] * 2.0 - 1.0
    uvs4[:, 1] = uvs[:, 1] * 2.0 - 1.0
    uvs4[:, 3] = 1.0
    rast, _ = dr.rasterize(ctx, torch.from_numpy(uvs4)[None].cuda(),
                           torch.from_numpy(uv_faces).cuda().to(torch.int32), resolution=[1024, 1024])
    cov = rast[0, :, :, 3] > 0
    rows, cols = torch.nonzero(cov, as_tuple=True)
    tid = rast[0, :, :, 3][rows, cols].long() - 1
    bary = torch.stack([rast[0, :, :, 0][rows, cols], rast[0, :, :, 1][rows, cols]], -1)
    bary = torch.cat([bary, 1.0 - bary.sum(-1, keepdim=True)], -1)
    fverts = torch.from_numpy(verts[uv_faces[tid.cpu().numpy()]]).cuda()
    world = (fverts * bary[:, :, None]).sum(1)
    e1 = fverts[:, 1] - fverts[:, 0]
    e2 = fverts[:, 2] - fverts[:, 0]
    nrm = torch.cross(e1, e2, dim=-1)
    nrm = nrm / torch.clamp(nrm.norm(dim=-1, keepdim=True), min=1e-12)
    chart_vert, _ = tfm.vertex_charts(len(uvs), uv_faces)
    atlas_hwc = atlas1024.permute(1, 2, 0).detach().cpu().numpy()
    return tfm.seam_stats(rows, cols, 1024, world, nrm, chart_vert, uv_faces, tid, atlas_hwc)


def load_texture_chw(path: Path, dev="cuda"):
    import torch
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).to(dev)


def load_gray_chw(path: Path, dev="cuda"):
    import torch
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), np.float32) / 255.0
    return torch.from_numpy(arr[None])


def resize_atlas(tex_chw, size=1024):
    import torch
    import torch.nn.functional as F
    return F.interpolate(tex_chw[None], size=(size, size), mode="bilinear",
                         align_corners=True)[0]


# ---------------------------------------------------------------- eval 主流程

def eval_mesh(args, mesh_name: str, out_dir: Path, lpips_model):
    import torch
    import trimesh

    obj = out_dir / f"{mesh_name}_painted.obj"
    if not obj.exists():
        return None, {"valid": False, "reason": "painted obj missing"}
    mesh = trimesh.load(str(obj), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.int32)
    uvs = np.asarray(mesh.visual.uv, np.float32)
    # OBJ 的 vt.v=0 在图像底部；tfm 约定 v=0 即图像第 0 行，需翻转
    uvs[:, 1] = 1.0 - uvs[:, 1]
    uv_attr = torch.from_numpy(uvs).cuda().float()[None]
    faces_t = torch.from_numpy(faces).cuda().int()

    tex = load_texture_chw(out_dir / f"{mesh_name}_painted.jpg")
    mr = None
    mpath = out_dir / f"{mesh_name}_painted_metallic.png"
    if not mpath.exists():
        mpath = out_dir / f"{mesh_name}_painted_metallic.jpg"
    rpath = out_dir / f"{mesh_name}_painted_roughness.png"
    if not rpath.exists():
        rpath = out_dir / f"{mesh_name}_painted_roughness.jpg"
    if mpath.exists() and rpath.exists():
        mr = torch.cat([load_gray_chw(mpath), load_gray_chw(rpath)], 0).cuda()

    train_cams = load_cameras(SCENE / "geometry" / "gt" / "cameras.json")
    ref_cam = [c for c in train_cams if c["frame"] == args.ref_frame][0]
    ref_rgb, ref_alpha = load_rgba([ref_cam], SCENE / "images")
    held_cams = load_cameras(HELDOUT / "cameras.json")
    if args.max_heldout:
        held_cams = held_cams[:args.max_heldout]
    held_rgb, held_alpha = load_rgba(held_cams, HELDOUT / "images")

    ctx = make_ctx()
    cam_pos_ref = torch.from_numpy(cam_center(ref_cam)).cuda().float()

    def render_all(cams, tex_chw, mr_chw):
        views = []
        for cam in cams:
            r = render_maps(ctx, verts, faces_t, uv_attr, cam, tex_chw, mr_chw)
            views.append(r)
        return views

    # ---- albedo（unlit）
    seen_views = render_all([ref_cam], tex, None)
    seen_rows, seen_agg = eval_views(seen_views, [ref_cam], ref_rgb, ref_alpha, lpips_model, "hunyuan", "seen")
    unseen_views = render_all(held_cams, tex, None)
    unseen_rows, unseen_agg = eval_views(unseen_views, held_cams, held_rgb, held_alpha, lpips_model, "hunyuan", "unseen")
    consistency = cross_view_consistency(verts, faces, held_cams, unseen_views, args.consistency_points)

    # ---- PBR 重渲（固定光照）
    def shade_views(cams, tex_chw, mr_chw):
        out = []
        for cam in cams:
            r = render_maps(ctx, verts, faces_t, uv_attr, cam, tex_chw, mr_chw)
            pos = r["pos"]
            shaded = shade_pbr(r["albedo"], r["normal"], pos,
                               torch.from_numpy(cam_center(cam)).cuda().float(),
                               r["metallic"], r["roughness"])
            r["albedo"] = shaded
            out.append(r)
        return out

    pbr_seen = shade_views([ref_cam], tex, mr)
    _, pbr_seen_agg = eval_views(pbr_seen, [ref_cam], ref_rgb, ref_alpha, lpips_model, "hunyuan_pbr", "seen")
    pbr_unseen = shade_views(held_cams, tex, mr)
    _, pbr_unseen_agg = eval_views(pbr_unseen, held_cams, held_rgb, held_alpha, lpips_model, "hunyuan_pbr", "unseen")

    # ---- PBR 材质统计（前景）
    mvals, rvals = [], []
    for v in pbr_seen + pbr_unseen:
        cov = v["cov"].detach().cpu().numpy()
        if v["metallic"] is not None:
            mvals.append(v["metallic"].detach().cpu().numpy()[cov])
            rvals.append(v["roughness"].detach().cpu().numpy()[cov])
    mat = {}
    if mvals:
        m = np.concatenate(mvals)
        r = np.concatenate(rvals)
        mat = {
            "metallic_mean": round(float(m.mean()), 4),
            "metallic_p90": round(float(np.percentile(m, 90)), 4),
            "metallic_gt_half": round(float((m > 0.5).mean()), 4),
            "roughness_mean": round(float(r.mean()), 4),
            "roughness_p10": round(float(np.percentile(r, 10)), 4),
            "roughness_lt_0_3": round(float((r < 0.3).mean()), 4),
        }

    # ---- seam
    atlas1024 = resize_atlas(tex, 1024)
    seam = seam_for_mesh(verts, faces, uvs, atlas1024, ctx)

    # ---- T0 对照（同一相机/渲染器/着色器）
    t0 = {}
    try:
        d = np.load(T0_ROOT / "cache" / f"{mesh_name}.npz")
        du = np.load(T0_ROOT / "cache" / f"{mesh_name}_uv.npz")
        v0 = d["vertices"].astype(np.float32)
        f0 = d["faces"].astype(np.int32)
        vmapping = du["vmapping"].astype(np.int32)
        uv_faces0 = du["uv_faces"].astype(np.int32)
        uv0 = du["uvs"].astype(np.float32)
        uv_verts0 = v0[vmapping]
        uv_attr0 = torch.from_numpy(uv0).cuda().float()[None]
        uv_faces0_t = torch.from_numpy(uv_faces0).cuda().int()
        tex0 = load_texture_chw(T0_ROOT / f"{mesh_name}_atlas1024_weighted_mean.png")

        def render0(cam):
            return render_maps(ctx, uv_verts0, uv_faces0_t, uv_attr0, cam, tex0, None)

        s0 = [render0(ref_cam)]
        _, s0_agg = eval_views(s0, [ref_cam], ref_rgb, ref_alpha, lpips_model, "t0", "seen")
        u0 = [render0(c) for c in held_cams]
        _, u0_agg = eval_views(u0, held_cams, held_rgb, held_alpha, lpips_model, "t0", "unseen")
        p0s = []
        for r in s0:
            r["albedo"] = shade_pbr(r["albedo"], r["normal"], r["pos"],
                                    torch.from_numpy(cam_center(ref_cam)).cuda().float(), None, None)
            p0s.append(r)
        _, p0s_agg = eval_views(p0s, [ref_cam], ref_rgb, ref_alpha, lpips_model, "t0_pbr", "seen")
        p0u = []
        for c in held_cams:
            r = render0(c)
            r["albedo"] = shade_pbr(r["albedo"], r["normal"], r["pos"],
                                    torch.from_numpy(cam_center(c)).cuda().float(), None, None)
            p0u.append(r)
        _, p0u_agg = eval_views(p0u, held_cams, held_rgb, held_alpha, lpips_model, "t0_pbr", "unseen")
        cons0 = cross_view_consistency(v0, f0, held_cams, u0, args.consistency_points)
        t0 = {
            "seen_albedo": s0_agg, "unseen_albedo": u0_agg,
            "seen_pbr_default_material": p0s_agg, "unseen_pbr_default_material": p0u_agg,
            "consistency": cons0,
            "atlas": str(T0_ROOT / f"{mesh_name}_atlas1024_weighted_mean.png"),
        }
    except Exception as e:
        t0 = {"error": f"{type(e).__name__}: {e}"}

    rec = {
        "mesh": mesh_name,
        "mesh_path": str(MESHES[mesh_name]),
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "atlas_saved": int(tex.shape[-1]),
        "reference_frame": ref_cam["frame"],
        "seen": {"albedo": seen_agg, "pbr": pbr_seen_agg, "rows": seen_rows},
        "unseen": {"albedo": unseen_agg, "pbr": pbr_unseen_agg, "rows": unseen_rows},
        "cross_view_consistency": consistency,
        "seam": seam,
        "material": mat,
        "t0_reference": t0,
        "valid": True,
    }
    return rec, None


def save_render_grid(out_dir: Path, mesh_name: str, tag: str, gts, preds, alphas, cams, n=3):
    from PIL import Image
    d = out_dir / "renders"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(cams))):
        gt = (gts[i] * 255).astype(np.uint8)
        pr = (np.clip(preds[i], 0, 1) * 255).astype(np.uint8)
        gap = np.full((gt.shape[0], 4, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([gt, gap, pr], 1)).save(
            d / f"{mesh_name}_{tag}_{Path(cams[i]['frame']).stem}.png")


# ---------------------------------------------------------------- main

def main():
    global tfm
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["paint", "eval", "all"])
    ap.add_argument("--meshes", default="proxy,vggt")
    ap.add_argument("--ref-frame", default=REF_FRAME)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--max-num-view", type=int, default=6)
    ap.add_argument("--render-size", type=int, default=1024)
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-heldout", type=int, default=0)
    ap.add_argument("--consistency-points", type=int, default=30000)
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--dino-cpu", action="store_true")
    ap.add_argument("--paint-model-path", default="/root/hunyuan_weights/hunyuan3d-paintpbr-v2-1")
    ap.add_argument("--dino-path", default="/root/hunyuan_weights/dinov2-giant")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
    sys.path.insert(0, str(STUB))
    import texture_fixed_mesh as tfm

    summary = {"args": vars(args), "runs": {}, "invalid": {}}

    if args.stage in ("paint", "all"):
        for name in [x.strip() for x in args.meshes.split(",") if x.strip()]:
            print(f"[paint] === {name} ===", flush=True)
            try:
                rec = run_paint(args, name, out_dir)
                summary["runs"][name] = rec
                json_dump(out_dir / f"paint_{name}{args.tag}.json", rec)
            except Exception as e:
                tb = traceback.format_exc()
                summary["invalid"][name] = {"stage": "paint", "error": f"{type(e).__name__}: {e}"}
                (out_dir / f"paint_{name}{args.tag}.err.txt").write_text(tb)
                print(tb, flush=True)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                gc.collect()

    if args.stage in ("eval", "all"):
        import torch
        from lpips import LPIPS
        lpips_model = LPIPS(net="alex", verbose=False).cuda().eval()
        for name in [x.strip() for x in args.meshes.split(",") if x.strip()]:
            print(f"[eval] === {name} ===", flush=True)
            try:
                rec, err = eval_mesh(args, name, out_dir, lpips_model)
                if rec is not None:
                    summary["runs"].setdefault(name, {}).update(rec)
                    json_dump(out_dir / f"metrics_{name}{args.tag}.json", rec)
            except Exception as e:
                tb = traceback.format_exc()
                summary["invalid"].setdefault(name, {})["eval"] = f"{type(e).__name__}: {e}"
                (out_dir / f"metrics_{name}{args.tag}.err.txt").write_text(tb)
                print(tb, flush=True)
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    json_dump(out_dir / f"summary{args.tag}.json", summary)
    print(f"[out] {out_dir / ('summary' + args.tag + '.json')}", flush=True)
    if summary["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
