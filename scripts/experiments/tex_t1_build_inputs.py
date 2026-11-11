#!/usr/bin/env python3
"""Tex-T1：把固定 Mesh + GT 相机转成 SUPRA-Tex texturing 的规范输入包。

规范相机约定（见 third_party/supra_tex/supratex/inference/pipeline.py::_camera）：
    相机位于 (0, 0, distance)，看向原点，+Y 朝上，FOV 为 camera_angle_x。
本脚本对参考视角做「以主点为中心的正方形裁剪」，因此规范投影与原相机逐像素一致：
    p = s * [qx, -qy, -qz + dc]        (q = R_wc @ v_world + t_wc)
    camera_angle_x = 2 atan(N / (2 fx)),  distance = s * dc
其中 s 取使 mesh 完整落入 [-0.5, 0.5]^3 占位网格的最大尺度。

输出（每个 mesh × 每个参考视角一份）：
    geometry.pt, camera.json, condition_rgba.png, albedo.png, mra.png, meta.json

MRA 三种模式（--mra-mode，默认 neutral，不影响既有产物复现）：
    neutral  roughness=0.5、metallic=0 的中性先验（与 2026-11-05 轮一致）
    preset   常数 roughness/metallic（--roughness/--metallic），用于受控常数对照
    image    roughness 由参考图局部对比度启发式导出（合成先验，非真实材质测量）；
             metallic 只能取常数（图像不携带金属度信息）
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

SCENE = Path("/root/3dtex/data/scenes/nerf_lego_gt16")
MESHES = {
    "proxy": SCENE / "mesh" / "mesh_init.ply",
    "vggt": Path("/root/3dtex/outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply"),
}
OUT_ROOT = Path("/root/3dtex/outputs/tex_t1_supra_20261105/inputs")
SUPRA_RES = 1024


def load_camera(path: Path, index: int) -> dict:
    data = json.loads(path.read_text())
    f = data["frames"][index]
    return {
        "frame": f["frame"],
        "file": f.get("file", f["frame"]),
        "K": np.asarray(f["K_original"], np.float64),
        "T": np.asarray(f["T_world_to_cam"], np.float64),
        "W": int(f["original_width"]),
        "H": int(f["original_height"]),
        "index": index,
    }


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces, np.int32)
    if not np.isfinite(v).all():
        raise SystemExit(f"mesh has NaN/Inf vertices: {path}")
    return v, f


def load_reference_image(path: Path, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回 (RGB float32[H,W,3], alpha 0/255[H,W])；训练图无 alpha 时用 GT 深度兜底。"""
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), np.float32) / 255.0
    if rgba.shape[:2] != (cam["H"], cam["W"]):
        raise RuntimeError(f"image shape mismatch: {rgba.shape} vs {cam['H'], cam['W']}")
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if alpha.min() < 0.999:  # 图像自带 alpha
        mask = (alpha >= 0.5).astype(np.uint8) * 255
        src = "image_alpha"
    else:
        depth = None
        for cand in (path.parent.parent / "geometry" / "gt" / "depth" / (path.stem + ".npy"),
                     path.parent.parent / "geometry" / "gt" / "depth" / (path.stem + ".png")):
            if cand.is_file():
                depth = np.asarray(Image.open(cand), np.float32) if cand.suffix == ".png" else np.load(cand)
                if depth.ndim == 3:
                    depth = depth[..., 0]
                break
        if depth is not None and depth.shape == (cam["H"], cam["W"]):
            mask = (np.isfinite(depth) & (depth > 0)).astype(np.uint8) * 255
            src = "gt_depth"
        else:
            lum = rgb.mean(axis=-1)
            mask = (lum > 0.02).astype(np.uint8) * 255
            src = "luminance_threshold"
        if src == "luminance_threshold":
            print(f"[warn] {path.name}: 无 alpha/深度，使用亮度阈值前台（{mask.mean() / 255:.3f}）")
    return rgb, mask


def build_mra(mode: str, rgb_p: np.ndarray, a_bin: np.ndarray,
              roughness: float, metallic: float, contrast_k: int) -> tuple[np.ndarray, dict]:
    """按模式合成 MRA（R=metallic，G=roughness，B=alpha 覆盖）。返回 (mra, 记录)。"""
    if mode == "neutral":
        rough_map = np.full(rgb_p.shape[:2], 0.5, np.float32)
        metal_map = np.zeros(rgb_p.shape[:2], np.float32)
        rec = {"mode": "neutral", "note": "中性先验：roughness=0.5, metallic=0"}
    elif mode == "preset":
        rough_map = np.full(rgb_p.shape[:2], float(roughness), np.float32)
        metal_map = np.full(rgb_p.shape[:2], float(metallic), np.float32)
        rec = {"mode": "preset", "roughness": float(roughness), "metallic": float(metallic)}
    elif mode == "image":
        from scipy.ndimage import uniform_filter
        lum = rgb_p @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        m1 = uniform_filter(lum, size=contrast_k)
        m2 = uniform_filter(lum * lum, size=contrast_k)
        contrast = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
        fg = a_bin > 0
        if fg.sum() < 100:
            raise SystemExit("image MRA 模式需要足够的前景像素")
        lo, hi = np.percentile(contrast[fg], [2, 98])
        rough_map = np.clip((contrast - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)
        metal_map = np.full(rgb_p.shape[:2], float(metallic), np.float32)
        rec = {"mode": "image", "contrast_window": contrast_k,
               "contrast_p2": round(float(lo), 4), "contrast_p98": round(float(hi), 4),
               "roughness_range": [round(float(rough_map.min()), 4), round(float(rough_map.max()), 4)],
               "metallic": float(metallic),
               "note": "roughness=参考图局部对比度归一化（合成先验，非真实材质测量）"}
    else:
        raise SystemExit(f"未知 mra-mode: {mode}")
    mra = np.zeros(rgb_p.shape[:2] + (3,), np.uint8)
    mra[..., 0] = np.clip(metal_map * 255, 0, 255).astype(np.uint8)
    mra[..., 1] = np.clip(rough_map * 255, 0, 255).astype(np.uint8)
    mra[..., 2] = a_bin  # alpha 覆盖
    rec["roughness_mean"] = round(float(mra[..., 1].mean() / 255.0), 4)
    rec["roughness_std"] = round(float(mra[..., 1].std() / 255.0), 4)
    rec["metallic_mean"] = round(float(mra[..., 0].mean() / 255.0), 4)
    return mra, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--ref-index", type=int, default=0)
    ap.add_argument("--views", default="", help="逗号分隔的参考视角列表（如 0,4,8,12）；给出时覆盖 --ref-index")
    ap.add_argument("--mesh", action="append", default=[], help="name=path，可重复")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--pad", type=float, default=0.10, help="裁剪相对前景的留白比例")
    ap.add_argument("--mra-mode", choices=("neutral", "preset", "image"), default="neutral")
    ap.add_argument("--roughness", type=float, default=0.5, help="preset 模式的常数 roughness；image 模式的 metallic 用 --metallic")
    ap.add_argument("--metallic", type=float, default=0.0)
    ap.add_argument("--mra-contrast-k", type=int, default=9, help="image 模式的局部对比度窗口")
    args = ap.parse_args()

    scene = Path(args.scene)
    meshes = dict(MESHES) if scene == SCENE else {}
    for spec in args.mesh:
        name, path = spec.split("=", 1)
        meshes[name] = Path(path)
    out_root = Path(args.out)
    views = [int(x) for x in args.views.split(",") if x.strip()] or [args.ref_index]

    reports = {}
    for ref_index in views:
        reports[ref_index] = build_one(scene, meshes, out_root, ref_index, args)
    if len(reports) == 1:
        report = next(iter(reports.values()))
    else:
        report = {"multi_view": True,
                  "views": {str(k): v for k, v in reports.items()}}
    (out_root / "input_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[out] {out_root / 'input_report.json'}")


def build_one(scene: Path, meshes: dict, out_root: Path, ref_index: int, args) -> dict:
    cam = load_camera(scene / "geometry" / "gt" / "cameras.json", ref_index)
    ref_img_path = scene / "images" / cam["file"]
    rgb, alpha = load_reference_image(ref_img_path, cam)
    K, T, W, H = cam["K"], cam["T"], cam["W"], cam["H"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    report = {"ref_index": ref_index, "ref_frame": cam["frame"], "ref_image": str(ref_img_path),
              "mra": args.mra_mode, "K": K.tolist(), "T": T.tolist(), "W": W, "H": H, "meshes": {}}

    for name, mesh_path in meshes.items():
        verts, faces = load_mesh(mesh_path)
        q = verts @ T[:3, :3].T + T[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            xp = fx * q[:, 0] / q[:, 2]
            yp = fy * q[:, 1] / q[:, 2]
        valid = q[:, 2] > 1e-6
        reach = max(np.percentile(np.abs(xp[valid]), 99.9), np.percentile(np.abs(yp[valid]), 99.9))
        N = int(2.0 * math.ceil(reach * (1.0 + args.pad)))
        N = max(64, min(N, int(min(2 * cx, 2 * cy, W, H))))
        if N % 2:
            N -= 1
        crop = (int(round(cx - N / 2)), int(round(cy - N / 2)), int(round(cx + N / 2)), int(round(cy + N / 2)))

        x0, y0, x1, y1 = crop
        rgb_c = rgb[y0:y1, x0:x1]
        a_c = alpha[y0:y1, x0:x1]
        rgb_p = np.asarray(Image.fromarray((rgb_c * 255).astype(np.uint8)).resize((SUPRA_RES, SUPRA_RES), Image.LANCZOS), np.float32)
        a_p = np.asarray(Image.fromarray(a_c).resize((SUPRA_RES, SUPRA_RES), Image.NEAREST))
        a_bin = (a_p >= 128).astype(np.uint8) * 255
        cond = np.dstack([rgb_p.astype(np.uint8), a_bin])
        albedo = rgb_p.astype(np.uint8)
        mra, mra_rec = build_mra(args.mra_mode, rgb_p, a_bin, args.roughness, args.metallic,
                                 args.mra_contrast_k)
        # 居中裁剪后主点位于裁剪中心，需确认
        if abs(cx - x0) - N / 2 > 1e-3 or abs(cy - y0) - N / 2 > 1e-3:
            print(f"[warn] {name}: 裁剪未完全以主点为中心（dx={cx - x0 - N / 2:.2f}, dy={cy - y0 - N / 2:.2f}）")

        # 规范坐标
        dc = 0.5 * (q[:, 2].min() + q[:, 2].max())  # 物体中心深度（决定物体 z 中心为 0）
        u = np.stack([q[:, 0], -q[:, 1], -(q[:, 2] - dc)], axis=1)
        s = 0.49 / float(np.abs(u).max())
        p = (u * s).astype(np.float32)
        dist = s * float(dc)
        fov = 2.0 * math.atan(N / (2.0 * fx))

        out_dir = out_root / f"{name}_v{ref_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save({"verts": torch.from_numpy(p), "faces": torch.from_numpy(faces.astype(np.int64))},
                   out_dir / "geometry.pt")
        (out_dir / "camera.json").write_text(json.dumps({
            "camera_angle_x": fov, "distance": dist, "mesh_scale": 1.0,
            "image_width": SUPRA_RES, "image_height": SUPRA_RES,
            "source": f"tex_t1 GT camera frame {cam['frame']} (index {ref_index})",
            "crop_xyxy": [x0, y0, x1, y1], "crop_side": N,
        }, indent=2))
        Image.fromarray(cond).save(out_dir / "condition_rgba.png")
        Image.fromarray(albedo).save(out_dir / "albedo.png")
        Image.fromarray(mra).save(out_dir / "mra.png")

        # 自检：把规范 mesh 按 camera.json 投影到 1024，与裁剪后的前景比 IoU
        f1024 = fx * SUPRA_RES / N
        depth = dist - p[:, 2]
        uu = f1024 * p[:, 0] / depth + SUPRA_RES / 2
        vv = SUPRA_RES / 2 - f1024 * p[:, 1] / depth
        ui = np.clip(np.round(uu).astype(int), 0, SUPRA_RES - 1)
        vi = np.clip(np.round(vv).astype(int), 0, SUPRA_RES - 1)
        splat = np.zeros((SUPRA_RES, SUPRA_RES), bool)
        splat[vi, ui] = True
        from scipy.ndimage import binary_dilation, binary_fill_holes
        splat = binary_fill_holes(binary_dilation(splat, iterations=8))
        fg = a_bin > 0
        iou = float((splat & fg).sum() / max((splat | fg).sum(), 1))
        mesh_rec = {
            "mesh_path": str(mesh_path), "vertices": int(len(verts)), "faces": int(len(faces)),
            "crop_xyxy": [x0, y0, x1, y1], "crop_side": N,
            "camera_angle_x": fov, "camera_angle_deg": math.degrees(fov),
            "distance": dist, "scale_s": s, "depth_center": float(dc),
            "bbox_min": p.min(0).tolist(), "bbox_max": p.max(0).tolist(),
            "projection_silhouette_iou_vs_image_fg": round(iou, 4),
            "alpha_source_share": round(float((a_bin > 0).mean()), 4),
            "mra": mra_rec,
            "T_world_to_cam": T.tolist(),
        }
        report["meshes"][name] = mesh_rec
        (out_dir / "meta.json").write_text(json.dumps(mesh_rec, indent=2, ensure_ascii=False))
        print(f"[{name}] N={N} fov={math.degrees(fov):.2f}deg dist={dist:.4f} s={s:.4f} "
              f"IoU(proj,f g)={iou:.3f} -> {out_dir}")
    return report


if __name__ == "__main__":
    main()
