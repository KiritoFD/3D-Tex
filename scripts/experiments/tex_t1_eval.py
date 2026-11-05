#!/usr/bin/env python3
"""Tex-T1 评测：把 SUPRA-Tex 导出的 GLB 放回世界坐标，复用 T0 渲染/评测口径出四轴指标。

四轴：
  1) 参考保真：seen-view PSNR / LPIPS（seen = 该 texel 在参考视角可见）
  2) 未见区域补全：unseen-view PSNR / LPIPS + 跨视角一致性
  3) 表面连续性：真实 3D seam-pair 色差（T0 seam_stats 口径）
  4) 材质真实性：albedo 保真、roughness/metallic 统计、固定光照重渲误差
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/root/3dtex/scripts/experiments")
import texture_fixed_mesh as T0  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402

OUT = Path("/root/3dtex/outputs/tex_t1_supra_20261105")


def load_glb(path: Path):
    import trimesh

    obj = trimesh.load(str(path), process=False)
    if isinstance(obj, trimesh.Scene):
        # glTF 的节点常带变换（如 Y-up 旋转）；必须应用 graph 变换后再取几何
        parts = []
        for node in obj.graph.nodes_geometry:
            T, gname = obj.graph[node]
            g = obj.geometry[gname].copy()
            g.apply_transform(T)
            if getattr(g.visual, "uv", None) is not None and g.faces is not None and len(g.faces):
                parts.append(g)
        if not parts:
            parts = [obj.dump(concatenate=True)]
        obj = max(parts, key=lambda m: len(m.faces))
    v = np.asarray(obj.vertices, np.float32)
    f = np.asarray(obj.faces, np.int32)
    uv = np.asarray(obj.visual.uv, np.float32)
    mat = obj.visual.material
    img = None
    for attr in ("baseColorTexture", "image"):
        cand = getattr(mat, attr, None)
        if cand is not None:
            img = cand
            break
    if isinstance(img, list):
        img = img[0]
    tex = np.asarray(img.convert("RGB"), np.float32) / 255.0 if img is not None else None
    mr_img = getattr(mat, "metallicRoughnessTexture", None)
    mr = np.asarray(mr_img.convert("RGB"), np.float32) / 255.0 if mr_img is not None else None
    meta = {
        "material": type(mat).__name__,
        "metallic_factor": float(getattr(mat, "metallicFactor", 0.0) or 0.0),
        "roughness_factor": float(getattr(mat, "roughnessFactor", 1.0) or 0.0),
        "texture_res": int(tex.shape[0]) if tex is not None else 0,
        "has_mr": mr is not None,
    }
    return v, f, uv, tex, mr, meta


def to_world(p: np.ndarray, meta: dict) -> np.ndarray:
    """GLB 顶点 -> 世界坐标。

    SUPRA 导出的 GLB 与规范系相差一个绕 Y 轴 180° 的刚体变换
    （实测最优：p_canon = [-x, y, -z]，chamfer 0.0049 规范单位 ≈ 1.2cm，
    来自 mesh_export.py 的 Z-up -> Y-up 轴交换 + 场景节点变换），
    这里先还原到 SUPRA 规范系，再做规范相机逆变换。
    """
    p = np.stack([-p[:, 0], p[:, 1], -p[:, 2]], axis=1)
    s = float(meta["scale_s"])
    dc = float(meta["depth_center"])
    T = np.asarray(meta["T_world_to_cam"], np.float64)
    R, t = T[:3, :3], T[:3, 3]
    q = np.stack([p[:, 0] / s, -p[:, 1] / s, dc - p[:, 2] / s], axis=1)
    return (q - t) @ R


def load_t0_atlas(out_dir: Path, mesh: str, atlas_res: int):
    """T0 产物：世界坐标 mesh + xatlas UV 缓存 + atlas PNG（用于同一口径下补算 LPIPS/一致性）。"""
    data = np.load(out_dir / "cache" / f"{mesh}_uv.npz")
    uvs = np.asarray(data["uvs"], np.float32)
    uv_faces = np.asarray(data["uv_faces"], np.int32)
    vmapping = np.asarray(data["vmapping"], np.int32)
    verts, faces = T0.load_mesh(T0.MESH_PATHS[mesh], out_dir / "cache" / f"{mesh}.npz")
    uv_verts = np.asarray(verts[vmapping], np.float32)
    tex_path = out_dir / f"{mesh}_atlas{atlas_res}_weighted_mean.png"
    from PIL import Image
    tex = np.asarray(Image.open(tex_path).convert("RGB"), np.float32) / 255.0
    meta = {"mesh_path": str(T0.MESH_PATHS[mesh]), "vertices": int(len(verts)),
            "faces": int(len(faces)), "crop_side": 0, "camera_angle_deg": 0.0, "scale_s": 1.0}
    gmeta = {"material": "T0_atlas_png", "metallic_factor": None, "roughness_factor": None,
             "texture_res": int(tex.shape[0]), "has_mr": False}
    return np.asarray(verts, np.float32), faces, uvs, uv_faces, uv_verts, tex, None, meta, gmeta


def resize_rgb(arr: np.ndarray, res: int) -> np.ndarray:
    if arr.shape[0] == res:
        return arr
    from PIL import Image
    return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((res, res), Image.LANCZOS),
                      np.float32) / 255.0


def lpips_on_mask(model, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None):
    if mask is not None:
        ys, xs = np.nonzero(mask)
        if len(ys) == 0 or (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) < 4096:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pred, gt, m = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        pred, gt = pred * m[..., None], gt * m[..., None]
    if min(pred.shape[:2]) < 8:
        return None
    a = torch.from_numpy(pred.transpose(2, 0, 1).copy()).cuda()[None] * 2 - 1
    b = torch.from_numpy(gt.transpose(2, 0, 1).copy()).cuda()[None] * 2 - 1
    with torch.no_grad():
        return round(float(model(a, b).item()), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", action="append", default=[], help="name=path，name 需含 proxy/vggt")
    ap.add_argument("--out", default=str(OUT / "eval"))
    ap.add_argument("--atlas-res", type=int, default=0, help="0 = 用纹理分辨率")
    ap.add_argument("--max-heldout", type=int, default=0)
    ap.add_argument("--t0-dir", default=str(OUT / "t0_single_view"), help="T0 单视角产物目录（t0:<mesh> 规格用）")
    ap.add_argument("--t0-atlas-res", type=int, default=2048)
    ap.add_argument("--save-renders", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = dr.RasterizeCudaContext()
    held_cams = T0.load_cameras(T0.HELDOUT_DIR / "cameras.json")
    held_imgs, held_alphas = T0.load_images(held_cams, T0.HELDOUT_DIR / "images", "cuda")
    if args.max_heldout:
        held_cams = held_cams[:args.max_heldout]
        held_imgs, held_alphas = held_imgs[:args.max_heldout], held_alphas[:args.max_heldout]

    ref_cam = T0.load_cameras(T0.SCENE_DIR / "geometry" / "gt" / "cameras.json")[0]
    ref_rgbs, _ = T0.load_images([ref_cam], T0.SCENE_DIR / "images", "cuda")
    ref_img = ref_rgbs[0]

    import lpips
    lp_model = lpips.LPIPS(net="alex").cuda().eval()

    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
    for spec in args.glb:
        if spec.startswith("t0:"):
            name, mesh_name = spec, spec.split(":", 1)[1]
            verts_w, faces, uvs, uv_faces, uv_verts, tex, mr, meta, gmeta = load_t0_atlas(
                Path(args.t0_dir), mesh_name, args.t0_atlas_res)
            glb_path = meta["mesh_path"]
        else:
            name, glb_path = spec.split("=", 1)
            mesh_name = "proxy" if "proxy" in name else "vggt"
            meta = json.loads((OUT / "inputs" / f"{mesh_name}_v0" / "meta.json").read_text())

            verts, faces, uvs, tex, mr, gmeta = load_glb(Path(glb_path))
            if tex is None:
                raise SystemExit(f"{glb_path}: GLB 没有 baseColorTexture，无法评测")
            if len(uvs) != len(verts) or int(faces.max()) >= len(uvs):
                raise SystemExit(f"{glb_path}: UV 顶点数({len(uvs)}) 与 3D 顶点数({len(verts)}) 不一致，"
                                 "需要显式 uv->3d 映射")
            verts_w = to_world(verts, meta).astype(np.float32)
            # mesh_export.py 写 GLB 前做了 uvs[:,1] = 1 - V；trimesh 读 GLB 不再翻转，
            # 而本评测的 atlas 行号约定是 V 向下，故这里翻回烘焙时的约定。
            uvs = np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=1).astype(np.float32)
            uv_faces = faces
            uv_verts = verts_w
        uv_attr = torch.from_numpy(uvs).cuda().float()[None]
        faces_uv_t = torch.from_numpy(uv_faces).cuda().int()
        chart_vert, n_chart = T0.vertex_charts(len(uvs), uv_faces)

        res = args.atlas_res or int(tex.shape[0])
        baked = T0.bake(ctx, uvs, uv_faces, uv_verts, verts_w, faces, [ref_cam], [ref_img],
                        res, 0.2, 2.0, chart_vert)
        vis_map = torch.from_numpy(T0.build_atlas(baked["rows"], baked["cols"], res,
                                                  baked["vis_count"])).cuda()

        tex_t = torch.from_numpy(tex).cuda().permute(2, 0, 1).contiguous()
        # 法线按 UV 顶点计算（GLB 的 UV 顶点数可能多于/少于 3D 顶点）
        fv_t = T0.vertex_normals(uv_verts, uv_faces)

        seam = T0.seam_stats(baked["rows"], baked["cols"], res, baked["world"], baked["normal"],
                             chart_vert, uv_faces, baked["tid"], resize_rgb(tex, res))

        per_view, renders = [], []
        for ci, (cam, gt_img_t, gt_alpha_t) in enumerate(zip(held_cams, held_imgs, held_alphas)):
            T, K, W, H = cam["T"], cam["K"], cam["W"], cam["H"]
            clip, camz = T0.build_clip(uv_verts, T, K, W, H)
            rast, _ = T0.camera_raster(ctx, clip, camz, faces_uv_t, W, H)
            cov = (rast[0, :, :, 3] > 0)
            upix = dr.interpolate(uv_attr, rast, faces_uv_t)[0][0]
            vis_px = T0.sample_map(vis_map, upix)
            pred = T0.sample_uv(tex_t, upix, W, H)
            cov_np = cov.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy() * cov_np[..., None]

            gt = gt_img_t.detach().cpu().numpy().transpose(1, 2, 0)
            fg = gt_alpha_t.detach().cpu().numpy() > 0.5
            vis_np = vis_px.detach().cpu().numpy()
            inter = fg & cov_np
            seen = inter & (vis_np >= 1)
            unseen = inter & (vis_np < 1)
            from scipy.ndimage import binary_erosion
            inner = binary_erosion(fg, iterations=8)

            m = {
                "view": ci, "frame": cam["frame"],
                "psnr_full": round(T0.psnr(pred_np, gt), 3),
                "psnr_fg": round(T0.psnr(pred_np[fg], gt[fg]), 3) if fg.sum() else None,
                "psnr_seen": round(T0.psnr(pred_np[seen], gt[seen]), 3) if seen.sum() else None,
                "psnr_unseen": round(T0.psnr(pred_np[unseen], gt[unseen]), 3) if unseen.sum() else None,
                "ssim_full": round(T0.ssim(pred_np, gt), 4),
                "psnr_inner": round(T0.psnr(pred_np[inner], gt[inner]), 3) if inner.sum() else None,
                "lpips_fg": lpips_on_mask(lp_model, pred_np, gt, fg),
                "lpips_seen": lpips_on_mask(lp_model, pred_np, gt, seen),
                "lpips_unseen": lpips_on_mask(lp_model, pred_np, gt, unseen),
                "iou": round(float(inter.sum() / max((fg | cov_np).sum(), 1)), 4),
                "seen_pixels": int(seen.sum()), "unseen_pixels": int(unseen.sum()),
            }
            per_view.append(m)
            renders.append(pred_np)
            if args.save_renders:
                rdir = out_dir / "renders"
                rdir.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                Image.fromarray((pred_np * 255).clip(0, 255).astype(np.uint8)).save(
                    rdir / f"{name}_v{ci:02d}_pred.png")
                Image.fromarray((gt * 255).clip(0, 255).astype(np.uint8)).save(
                    rdir / f"{name}_v{ci:02d}_gt.png")
            print(f"  [view {ci:02d}] psnr_fg={m['psnr_fg']} seen={m['psnr_seen']} "
                  f"unseen={m['psnr_unseen']} lpips={m['lpips_fg']}")

        agg = {}
        for k in ["psnr_full", "psnr_fg", "psnr_seen", "psnr_unseen", "ssim_full", "psnr_inner",
                  "lpips_fg", "lpips_seen", "lpips_unseen", "iou"]:
            vals = [r[k] for r in per_view if r[k] is not None]
            if vals:
                agg[k] = round(float(np.mean(vals)), 4)
                agg[k + "_std"] = round(float(np.std(vals)), 4)

        # ---- 跨视角一致性 ----
        cons = []
        for ia, ib in [(0, 1), (0, 8), (4, 12)]:
            if ia >= len(renders) or ib >= len(renders):
                continue
            W, H = held_cams[ia]["W"], held_cams[ia]["H"]
            clip_a, camz_a = T0.build_clip(uv_verts, held_cams[ia]["T"], held_cams[ia]["K"], W, H)
            rast_a, _ = T0.camera_raster(ctx, clip_a, camz_a, faces_uv_t, W, H)
            cov_a = (rast_a[0, :, :, 3] > 0)
            va = dr.interpolate(torch.from_numpy(uv_verts).cuda()[None], rast_a, faces_uv_t)[0][0]
            na = F.normalize(dr.interpolate(fv_t[None], rast_a, faces_uv_t)[0][0], dim=-1)
            va_np, na_np = va.detach().cpu().numpy(), na.detach().cpu().numpy()

            Tb = held_cams[ib]["T"]
            qb = va_np @ np.asarray(Tb[:3, :3], np.float32).T + np.asarray(Tb[:3, 3], np.float32)
            zb = qb[..., 2]
            ub = qb[..., 0] / np.where(zb > 1e-6, zb, 1e-6) * float(held_cams[ib]["K"][0, 0]) + float(held_cams[ib]["K"][0, 2])
            vb = qb[..., 1] / np.where(zb > 1e-6, zb, 1e-6) * float(held_cams[ib]["K"][1, 1]) + float(held_cams[ib]["K"][1, 2])
            ok = cov_a.detach().cpu().numpy() & (zb > 1e-6) & (ub >= 0) & (ub < W) & (vb >= 0) & (vb < H)

            clip_b, camz_b = T0.build_clip(uv_verts, Tb, held_cams[ib]["K"], W, H)
            rast_b, zbuf_b = T0.camera_raster(ctx, clip_b, camz_b, faces_uv_t, W, H)
            cov_b_np = (rast_b[0, :, :, 3] > 0).detach().cpu().numpy()
            zbuf_np = zbuf_b.detach().cpu().numpy()
            ui = np.clip(np.round(ub).astype(int), 0, W - 1)
            vi = np.clip(np.round(vb).astype(int), 0, H - 1)
            ok &= cov_b_np[vi, ui]
            ok &= np.abs(zbuf_np[vi, ui] - zb) < (2.0 * T0.VOXEL + 0.01 * np.abs(zb))
            nb = dr.interpolate(fv_t[None], rast_b, faces_uv_t)[0][0].detach().cpu().numpy()
            ok &= (na_np * nb[vi, ui]).sum(-1) > 0.5
            ok &= renders[ia].sum(-1) > 0
            diff = np.abs(renders[ia][ok] - renders[ib][vi[ok], ui[ok]])
            cons.append({
                "pair": [ia, ib], "pixels": int(ok.sum()),
                "mean_abs_rgb": round(float(diff.mean()), 5) if ok.sum() else None,
                "coverage": round(float(ok.sum() / max(int(cov_a.sum()), 1)), 4),
            })
            print(f"  [consistency {ia}-{ib}] pixels={int(ok.sum())} mad={cons[-1]['mean_abs_rgb']}")

        # ---- 材质轴 ----
        mat = dict(gmeta)
        if mr is not None:
            resized_mr = resize_rgb(mr, res)
            mrr = resized_mr[baked["rows"].cpu().numpy(), baked["cols"].cpu().numpy()]
            mat["roughness_mean"] = round(float(mrr[:, 1].mean()), 4)
            mat["roughness_std"] = round(float(mrr[:, 1].std()), 4)
            # glTF 标准：R 未用、G=roughness、B=metallic
            mat["metallic_mean"] = round(float(mrr[:, 2].mean()), 4)
            mat["metallic_std"] = round(float(mrr[:, 2].std()), 4)

        T, K, W, H = ref_cam["T"], ref_cam["K"], ref_cam["W"], ref_cam["H"]
        clip, camz = T0.build_clip(uv_verts, T, K, W, H)
        rast, _ = T0.camera_raster(ctx, clip, camz, faces_uv_t, W, H)
        cov = (rast[0, :, :, 3] > 0)
        upix = dr.interpolate(uv_attr, rast, faces_uv_t)[0][0]
        alb = T0.sample_uv(tex_t, upix, W, H)
        nor = F.normalize(dr.interpolate(fv_t[None], rast, faces_uv_t)[0][0], dim=-1)
        ldir = torch.tensor([0.35, 0.55, 0.76], device="cuda", dtype=torch.float32)
        ldir = ldir / ldir.norm()
        lam = torch.clamp((nor * ldir).sum(-1), min=0.0)
        shade = alb * (0.30 + 0.65 * lam[..., None])
        cov_np = cov.detach().cpu().numpy()
        ref_np = ref_img.detach().cpu().numpy().transpose(1, 2, 0)
        if cov_np.sum():
            mat["albedo_psnr_vs_ref"] = round(T0.psnr(
                alb.detach().cpu().numpy()[cov_np], ref_np[cov_np]), 3)
            mat["relight_psnr_vs_ref"] = round(T0.psnr(
                shade.detach().cpu().numpy()[cov_np], ref_np[cov_np]), 3)
        mat["relight_note"] = "固定方向光 albedo*(0.30+0.65 n·l)，与参考图比，仅作自一致性参考"

        rec = {
            "name": name, "glb": str(glb_path),
            "mesh_meta": {k: meta[k] for k in ("mesh_path", "vertices", "faces", "crop_side",
                                               "camera_angle_deg", "scale_s")},
            "glb_meta": {**gmeta, "vertices": int(len(verts_w)), "faces": int(len(faces)),
                         "uv_vertices": int(len(uvs)), "charts": n_chart},
            "zero_vis_texels_share": round(float((baked["vis_count"] == 0).float().mean().item()), 4),
            "bake_stats": baked["stats"],
            "aggregate": agg, "per_view": per_view, "seam": seam,
            "cross_view_consistency": cons, "material": mat,
            "reference_view": 0,
        }
        summary[name] = rec
        (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"[{name}] agg={ {k: v for k, v in agg.items() if not k.endswith('_std')} }")
        print(f"[{name}] seam={seam}")
        print(f"[{name}] material={mat}")
        torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[out] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
