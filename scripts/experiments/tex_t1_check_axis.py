#!/usr/bin/env python3
"""在 SUPRA 规范系内搜索 GLB 与输入 Mesh 的最优轴向映射（只诊断，不改产物）。"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/3dtex/scripts/experiments")
import texture_fixed_mesh as T0  # noqa: E402

OUT = "/root/3dtex/outputs/tex_t1_supra_20261105"
glb, mesh = sys.argv[1], sys.argv[2]
meta = json.loads(open(f"{OUT}/inputs/{mesh}_v0/meta.json").read())
s, dc = float(meta["scale_s"]), float(meta["depth_center"])
Tm = np.asarray(meta["T_world_to_cam"], np.float64)

obj = trimesh.load(glb, process=False)
if isinstance(obj, trimesh.Scene):
    obj = obj.dump(concatenate=True)
gv = np.asarray(obj.vertices, np.float64)
verts, _ = T0.load_mesh(T0.MESH_PATHS[mesh], Path(f"{OUT}/t0_single_view/cache/{mesh}.npz"))

# 输入 world -> SUPRA 规范系
cam = np.asarray(verts, np.float64) @ Tm[:3, :3].T + Tm[:3, 3]
p_in = np.stack([s * cam[:, 0], -s * cam[:, 1], -s * (cam[:, 2] - dc)], axis=1)
print(f"input canonical bbox {np.round(p_in.min(0), 4)} .. {np.round(p_in.max(0), 4)}")
print(f"glb   raw     bbox {np.round(gv.min(0), 4)} .. {np.round(gv.max(0), 4)}")
print(f"input canonical extent {np.round(p_in.max(0) - p_in.min(0), 4)}")
print(f"glb   raw     extent {np.round(gv.max(0) - gv.min(0), 4)}")

tree_in = cKDTree(p_in[::7])
best = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
        R = np.zeros((3, 3))
        for i, (p, sn) in enumerate(zip(perm, sg)):
            R[i, p] = sn
        if np.linalg.det(R) < 0:
            continue
        g = gv @ R.T
        d, _ = tree_in.query(g[::3])
        best.append((float(d.mean()), perm, sg))
best.sort()
print("top-5 axis candidates (mean chamfer in canonical units):")
for item in best[:5]:
    print(f"  {item[0]:.5f}  perm={item[1]} signs={item[2]}")
