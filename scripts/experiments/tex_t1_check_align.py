#!/usr/bin/env python3
"""检查 SUPRA GLB 与输入 Mesh 的世界坐标对齐：还原导出器的 Z-up->Y-up 轴交换后量 chamfer。"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/3dtex/scripts/experiments")
import texture_fixed_mesh as T0  # noqa: E402

OUT = "/root/3dtex/outputs/tex_t1_supra_20261105"
glb = sys.argv[1]
mesh = sys.argv[2]


def to_world(p, meta, swap):
    if swap:
        p = np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)
    s = float(meta["scale_s"])
    dc = float(meta["depth_center"])
    Tm = np.asarray(meta["T_world_to_cam"], np.float64)
    q = np.stack([p[:, 0] / s, -p[:, 1] / s, dc - p[:, 2] / s], axis=1)
    return (q - Tm[:3, 3]) @ Tm[:3, :3]


meta = json.loads(open(f"{OUT}/inputs/{mesh}_v0/meta.json").read())
obj = trimesh.load(glb, process=False)
if isinstance(obj, trimesh.Scene):
    obj = obj.dump(concatenate=True)
gv = np.asarray(obj.vertices, np.float32)
verts, faces = T0.load_mesh(T0.MESH_PATHS[mesh], Path(f"{OUT}/t0_single_view/cache/{mesh}.npz"))

for tag, swap in (("raw", False), ("swap_inverse", True)):
    w = to_world(gv, meta, swap)
    print(f"[{tag}] glb world bbox  {np.round(w.min(0), 3)} .. {np.round(w.max(0), 3)}")
    print(f"[{tag}] orig world bbox {np.round(verts.min(0), 3)} .. {np.round(verts.max(0), 3)}")
    d, _ = cKDTree(w[::3]).query(verts[::7])
    print(f"[{tag}] orig->glb nearest: mean {d.mean():.4f} m, p95 {np.percentile(d, 95):.4f} m")
    d2, _ = cKDTree(verts[::7]).query(w[::3])
    print(f"[{tag}] glb->orig nearest: mean {d2.mean():.4f} m, p95 {np.percentile(d2, 95):.4f} m")
