# GT 深度 → TSDF 对照（2026-11-02）

`lego_gt_tsdf/` 使用公开 lego Blender 源场景射线求交生成浮点 GT 深度，固定为相同 16 视角、512²、TSDF 256 分辨率；未过 ExMesh、未做表面平滑。

- `mesh.glb` / `mesh.obj` + `material.mtl` + `albedo.png`：10 万面、2048² 彩色资产。
- `mesh_raw.ply`：25.1 万面原始 TSDF 几何。
- `preview.blend`：默认显示贴图。
- `raw_geometry/preview.blend`：无贴图原始几何。
- `comparison.png`：原图 / VGGT→TSDF / GT→TSDF，彩色与原始几何两行。

源场景轮廓与数据集平均 IoU 0.9868，存在小幅差异。GT 对照更规整，但细结构和可见性缺口仍在。实验记录见 [GT_TSDF_DIAGNOSTIC.md](../docs/GT_TSDF_DIAGNOSTIC.md)。

---

# 修正版 lego（2026-11-02）

最新产物位于 `lego_fixed/`，保留旧版 `lego/` 供比较：

- `lego_fixed/mesh.glb`：10,000 步 ExMesh，102,849 顶点 / 201,153 面，1024² 纹理。
- `lego_fixed/preview.blend`：已设好四台相机与无光照贴图材质的 Blender 预览。
- `lego_fixed/render_4views.png`：修正版四视角。
- `lego_fixed/comparison.png`：原图 / 旧版 700 步 / 修正版 10,000 步，同相机比较。
- `lego_fixed/comparison_report.json`：四个训练视角的轮廓 IoU 0.712→0.964，前景 PSNR 16.92→19.17 dB；不是 heldout benchmark。
- `lego_fixed_tsdf/`：修正融合与烘焙后的独立 CPU 基线，2048² 观测贴图。

本次使用数据集 RGBA alpha 作为 **GT 前景 mask**。背景外壳已去除，细杆、孔洞与表面细节仍不准确，不能称为完整高质量资产。完整修复记录见 [PIPELINE_FIXES.md](../docs/PIPELINE_FIXES.md)。

以下为旧版交付记录。

---

# 交付物：重建结果（2026-11-02）

本目录是**生成产物**，已 git-ignore，不进版本库。每个子目录都能直接用 Blender 打开。

问题排查与已知缺陷见 [docs/ISSUES_2026_11_02.md](../docs/ISSUES_2026_11_02.md)。

## 三个版本

| 目录 | 内容 | 规模 | 说明 |
|---|---|---|---|
| `lego/` | NeRF-Synthetic lego，**ExMesh 700 步** | 5.0 万顶点 / 9.8 万面，1024² | 目前最好的单物体资产 |
| `lego_tsdf/` | 同一场景，**未过 ExMesh** | 6.1 万顶点 / 12 万面，2048² | 基线，用于对比 ExMesh 的作用 |
| `kicker/` | ETH3D kicker，ExMesh 600 步 | 18.3 万顶点 / 34.8 万面，2560² | 大型室内场景，不是单物体 |
| `kicker_tsdf/` | ETH3D kicker，未过 ExMesh | 19.3 万顶点 / 37.3 万面，2048² | 同上，基线 |

每个目录含：

- `mesh.glb` — 单文件（几何 + UV + 内嵌纹理），`File → Import → glTF 2.0 (.glb)`
- `mesh.obj` + `mesh.mtl` + `albedo.png` — 三者需在同一目录，`File → Import → Wavefront (.obj)`
- `render_4views.png` — 四个正交视角的离线渲染，**先看这个判断好坏**

## Blender 注意事项

1. **必须切到 Material Preview / Rendered 视图**，Solid 视图不显示贴图。
2. 材质已设 `Kd/Ka = 1 1 1`、`illum 1`，`baseColorFactor = [1,1,1,1]`、`doubleSided = true`
   —— 这些是为了避免"贴图看起来全黑"，已通过回读 GLB 校验。
3. **UV 方向**：导出时按 OBJ 惯例做了 `1-v`。若贴图上下颠倒，用
   `scripts/export_exmesh.py --no-flip-v` 重新导出。
4. **不是闭合物体**：单面扫描，背面是洞（边界边 3.3–3.6 万条）。
5. **尺度非米制**：VGGT-Ω 输出是任意 gauge（kicker 的 Sim(3) 缩放约 3.28）。

## 质量现状（务必阅读）

当前产物**只有"管线跑通"的意义，不是可用资产**：

- 物体外面套着一层**灰色外壳**——背景被 TSDF 当成表面融合了，这是主要瓶颈；
- ExMesh 只跑了 600–700 步（官方 10,000）；
- 没有官方几何指标（ETH3D 的 scan 对齐协议尚未接入）。

## 重新生成

```bash
# 1. 几何（GPU）
python3 scripts/adapters/vggt_omega_geometry.py --scene-dir data/scenes/nerf_lego16 \
    --checkpoint data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt \
    --device cuda --aggregator-dtype bfloat16 --image-resolution 512

# 2. 一致性过滤（CPU；--conf-threshold 用于压制背景）
python3 scripts/consistency_filter.py --scene nerf_lego16 --min-agree 1 --tau-rel 0.12 --conf-threshold 10

# 3. TSDF → mesh
python3 scripts/tsdf_mesh.py --scene nerf_lego16 --voxel-div 256 --trunc-voxels 4 --stride 2

# 4a. 直接导出（基线）
python3 scripts/ply_to_obj.py --ply data/scenes/nerf_lego16/mesh/mesh_simplified.ply

# 4b. ExMesh（需先在其仓库打补丁，见 docs/EXMESH_ADAPTER.md）
python3 scripts/exmesh_adapter.py --scene nerf_lego16
EXMESH_TEX=1024 python3 train.py -s <abs>/exmesh_workdir -m <abs>/exmesh_workdir/out -r 2
python3 scripts/export_exmesh.py --checkpoint <...>/model_state_dict.pt
```

> 宿主机上权重加载必须走 `scripts/torch_stream_loader.py` 流式读取：torch 2.13 在已驻留
> 4.6 GiB 权重后再构造模型会 access violation（线程数不是原因，详见
> [docs/ISSUES_2026_11_02.md](../docs/ISSUES_2026_11_02.md) §7）。
