# 交付物：ETH3D kicker 重建结果

本目录是**生成产物**（已 git-ignore，不进版本库）。数据来源：ETH3D `kicker` 真实照片，
12 张训练视图（25 张中，8 GiB 显存只放得下 12 张）。

## 推荐先打开这个

### `kicker/` —— ExMesh 优化 600 步后的带 UV 资产（最好的一个）

| 文件 | 说明 |
|---|---|
| `mesh.obj` + `mesh.mtl` + `albedo.png` | Blender: `File → Import → Wavefront (.obj)`，三个文件要在同一目录 |
| `mesh.glb` | Blender: `File → Import → glTF 2.0 (.glb)`，单文件、贴图已内嵌 |
| `render_4views.png` | 四个正交视角的离线渲染，先看这个判断好不好 |
| `texture_mask.png` | 纹理有效区（白色=有真实观测，黑色=未观测/补洞导致，不应视为重建结果） |

规模：182,913 顶点 / 348,491 面，纹理 2560×2560。

### `kicker_tsdf/` —— 未经 ExMesh 的 TSDF 原始网格（基线对照）

193,460 顶点 / 372,798 面，纹理 2048×2048。颜色来自 TSDF 顶点色 baked 到图集，
细节明显不如 `kicker/`，用来对比 ExMesh 到底带来了什么。

## Blender 里的注意事项

1. **贴图可能偏暗或没显示**：导入后切到 `Material Preview` / `Rendered` 视图；
   MTL 里已设 `Kd 1 1 1`，如果仍看不到贴图，检查 Image Texture 节点是否指向 `albedo.png`。
2. **UV 上下可能颠倒**：导出时按 OBJ 惯例做了 `1-v`。如果贴图明显上下反了，
   重新导出时加 `--no-flip-v`（`scripts/export_exmesh.py`）。
3. **不是闭合物体**：这是单面扫描，背面/未观测区域是洞（边界边 3.3 万条）。
   想要水密模型需要额外的补洞或对称先验，当前没有做。
4. **尺度非米制**：VGGT-Ω 输出是任意 gauge（本场景 Sim(3) 缩放约 3.28）。需要真实尺寸时
   必须用已知参考物或 GT 标定。

## 重新生成

```bash
python3 scripts/adapters/vggt_omega_geometry.py --scene-dir data/scenes/eth3d_kicker \
    --checkpoint data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt \
    --device cuda --aggregator-dtype bfloat16 --image-resolution 512 --max-frames 12
python3 scripts/consistency_filter.py --scene eth3d_kicker --min-agree 1 --tau-rel 0.12 --max-frames 12
python3 scripts/tsdf_mesh.py --scene eth3d_kicker --voxel-div 256 --trunc-voxels 4 --stride 2
python3 scripts/ply_to_obj.py --ply data/scenes/eth3d_kicker/mesh/mesh_init.ply   # TSDF 直出
python3 scripts/exmesh_adapter.py --scene eth3d_kicker
# 在 ExMesh 目录训练，然后：
python3 scripts/export_exmesh.py --checkpoint <...>/out/point_cloud/iteration_N/model_state_dict.pt
```

## 已知问题

- ExMesh 只跑了 600 步（官方 10,000 步），细节远未收敛。
- 只有 12/25 个视角，覆盖不均导致部分区域纹理拉花。
- 没有跑 ETH3D 官方几何评估（需要 scan 对齐协议），所以**没有 Chamfer/F-score 数字**。
