#!/usr/bin/env python3
"""在现有 Lego 网格上合成「可控高频纹理」样本（固定几何 / 固定 GT 相机）。

用途：只改变图像内容的空间频率，不改 Mesh、UV、相机与纹理算法。
    - 影像 = 对 Mesh 顶点世界坐标直接求值的 3D 程序化纹理（value noise）
    - 训练 / heldout 影像由同一 Mesh + 同一相机渲染，因此几何误差恒为零
    - 该样本用于验证「高频内容下的算法行为」，不代表真实文物 / 复杂 PBR 材质

    python3 scripts/experiments/synth_hfreq_scene.py --wavelength 0.008 \
        --out outputs/texture_hfreq_20261105/scenes/hf_w008
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

LEGO_SCENE = Path("/root/3dtex/data/scenes/nerf_lego_gt16")
LEGO_HELDOUT = Path("/root/3dtex/outputs/lego_heldout_val16")
MESH_CACHE = Path("/root/3dtex/outputs/texture_fixed_mesh_20261105/cache/proxy.npz")


def hash01(ix, iy, iz, seed):
    """整数格点 -> [0,1) 的稳定伪随机（三通道）。"""
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    r = (h & 0xFFFFFF) / float(0x1000000)
    g = ((h >> 8) & 0xFFFFFF) / float(0x1000000)
    b = ((h >> 16) & 0xFFFFFF) / float(0x1000000)
    return np.stack([r, g, b], axis=-1)


def value_noise(P, lam, seed):
    """P: [N,3] 世界坐标；lam: 特征波长（米）。三线性插值 + smoothstep。"""
    q = P / lam
    i0 = np.floor(q).astype(np.int64)
    f = q - i0
    u = f * f * (3.0 - 2.0 * f)
    out = np.zeros((len(P), 3), np.float32)
    for dx in (0, 1):
        wx = u[:, 0] if dx else (1.0 - u[:, 0])
        for dy in (0, 1):
            wy = u[:, 1] if dy else (1.0 - u[:, 1])
            for dz in (0, 1):
                wz = u[:, 2] if dz else (1.0 - u[:, 2])
                w = (wx * wy * wz)[:, None]
                out += w * hash01(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
    return out


def make_pattern(P, lam, seed=7):
    base = 0.5 + 0.5 * value_noise(P, 0.40, seed)          # 低频底色（≈0.4 m）
    hf = value_noise(P, lam, seed + 11)                    # 目标高频
    col = 0.30 * base + 0.70 * hf
    return np.clip(col, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavelength", type=float, required=True, help="高频特征波长（米）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-cams", default=str(LEGO_SCENE / "geometry" / "gt" / "cameras.json"))
    ap.add_argument("--heldout-dir", default=str(LEGO_HELDOUT))
    ap.add_argument("--mesh-cache", default=str(MESH_CACHE))
    args = ap.parse_args()

    import torch
    import nvdiffrast.torch as dr
    from PIL import Image

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "geometry" / "gt").mkdir(parents=True, exist_ok=True)
    (out / "heldout" / "images").mkdir(parents=True, exist_ok=True)

    d = np.load(args.mesh_cache)
    verts = d["vertices"].astype(np.float32)
    faces = d["faces"].astype(np.int32)
    print(f"[mesh] {len(verts)} verts / {len(faces)} faces")

    ctx = dr.RasterizeCudaContext()
    v_t = torch.from_numpy(verts).cuda()
    f_t = torch.from_numpy(faces).cuda().int()
    w_t = torch.from_numpy(verts).cuda()[None]

    # 复用主脚本的相机数学（同一套投影，避免另一套实现引入偏差）
    spec = importlib.util.spec_from_file_location(
        "tfm", "/root/3dtex/scripts/experiments/texture_fixed_mesh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def render(cam):
        clip, camz = mod.build_clip(verts, cam["T"], cam["K"], cam["W"], cam["H"])
        rast, _ = mod.camera_raster(ctx, clip, camz, f_t, cam["W"], cam["H"])
        wpos = dr.interpolate(w_t, rast, f_t)[0][0]           # [H,W,3]
        cov = rast[0, :, :, 3] > 0
        P = wpos.detach().cpu().numpy().reshape(-1, 3)
        col = make_pattern(P, args.wavelength, args.seed).reshape(cam["H"], cam["W"], 3)
        cov_np = cov.detach().cpu().numpy()
        col[~cov_np] = 0.0
        return col, cov_np

    def cams_of(path, root_images):
        data = json.loads(Path(path).read_text())
        out_list = []
        for f in data["frames"]:
            out_list.append({
                "frame": f["frame"], "file": f.get("file", f["frame"]),
                "K": np.asarray(f["K_original"], np.float64),
                "T": np.asarray(f["T_world_to_cam"], np.float64),
                "W": int(f["original_width"]), "H": int(f["original_height"]),
                "root": root_images,
            })
        return out_list

    train = cams_of(args.train_cams, LEGO_SCENE / "images")
    held = cams_of(Path(args.heldout_dir) / "cameras.json", Path(args.heldout_dir) / "images")
    shutil.copy(args.train_cams, out / "geometry" / "gt" / "cameras.json")
    shutil.copy(Path(args.heldout_dir) / "cameras.json", out / "heldout" / "cameras.json")

    for group, cams in (("train", train), ("heldout", held)):
        for i, cam in enumerate(cams):
            col, cov = render(cam)
            rgba = np.zeros((cam["H"], cam["W"], 4), np.uint8)
            rgba[..., :3] = (col * 255).astype(np.uint8)
            rgba[..., 3] = (cov * 255).astype(np.uint8)
            if group == "train":
                dst = out / "images" / cam["file"]
            else:
                dst = out / "heldout" / "images" / cam["file"]
            Image.fromarray(rgba).save(dst)
            if i % 8 == 0:
                print(f"[{group} {i:02d}] {dst.name} cover {float(cov.mean()):.3f}")

    (out / "meta.json").write_text(json.dumps({
        "synthetic": True,
        "warning": "只在固定 Mesh 上验证高频内容下的算法行为，不代表真实文物或复杂 PBR 材质",
        "wavelength_m": args.wavelength,
        "mesh_cache": args.mesh_cache,
        "pattern": "3D value noise, base 0.4 m + target wavelength, 70% HF amplitude",
        "train_cameras": len(train), "heldout_cameras": len(held),
    }, indent=2, ensure_ascii=False))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
