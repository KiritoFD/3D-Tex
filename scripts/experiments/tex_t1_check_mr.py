#!/usr/bin/env python3
"""检查 GLB 的 baseColor / metallicRoughness 贴图通道统计（诊断 bf16 旁路臂材质退化）。"""
import numpy as np
import trimesh

p = "/root/3dtex/outputs/tex_t1_supra_20261105/glb_bf16/supra_proxy_v0_s42_bf16.glb"
o = trimesh.load(p, process=False)
if isinstance(o, trimesh.Scene):
    o = o.dump(concatenate=True)
m = o.visual.material
for nm, img in (("baseColor", m.baseColorTexture), ("metallicRoughness", m.metallicRoughnessTexture)):
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    print(nm, a.shape)
    for i, ch in enumerate("RGB"):
        x = a[..., i]
        print(f"  {ch}: min {x.min():.3f} max {x.max():.3f} mean {x.mean():.4f} std {x.std():.4f}")
