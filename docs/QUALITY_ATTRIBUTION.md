# 质量归因：VGGT-Ω 预测不准，还是我们后处理不行？（2026-11-02）

## 结论先行

**主要是后处理的问题，VGGT-Ω 的预测是准的。**

用 ETH3D `kicker`（真实照片 25 张，取其中 12 张训练视图）逐级对比真值后发现：

| 环节 | 指标 | 结论 |
|---|---|---|
| VGGT-Ω 相机位姿 | Sim(3) 对齐后中心误差 **0.59%** 场景尺度、旋转 **0.57°** 均值 / 1.21° 最大 | ✅ 很准 |
| VGGT-Ω 内参 | 修正换算后 `fx = 3401.7`，ETH3D 真值 `3410.34` | ✅ **误差 −0.25%** |
| VGGT-Ω 深度 | 跨视图一致率中位数 **0.819**（|Δz| ≤ 5%·z） | ✅ 自洽 |
| 我们的相机/深度导出 | 修复前 `K_original.f = 341.8`（真值 3410.3） | ❌ **差 10 倍** |
| 我们的 TSDF 融合 | `ScalableTSDFVolume` 输出空网格；自写版本曾把未知区当成"外部" | ❌ 密集 bug |
| 我们的 UV 烘焙 | xatlas 在破碎网格上给每个面单独成岛，纹理成噪点 | ❌ 上游输入太差 |

一句话：**模型给的几何足够好，是我们把像素坐标、体素符号、UV 参数化几处写错，把小误差放大成了"烂网格"。**

## 1. 怎么把两者分离开

关键在于**每一级都对着 ETH3D 官方真值量化**，而不是只看最终网格好不好看。ETH3D 场景自带：

- `ground_truth/calibration/cameras.txt`：官方 **PINHOLE** 内参（6211×4137，`fx=3410.34`）
- `ground_truth/calibration/poses.json`：官方相机位姿
- `ground_truth/scan_eval/*.ply`：激光扫描几何

### 1.1 位姿误差（与后处理无关）

只用训练视图估计 Sim(3)，再报误差：

```
Sim3 scale 3.2778 | camera centre err mean 0.0200 m (0.59% of 7.23 m span)
rotation 0.57 deg mean / 1.21 deg max
```

VGGT-Ω 输出的坐标系是**任意 gauge、非米制**（scale 3.28 就是证据），这符合 `docs/VGGT_OMEGA_ADAPTER.md` 的预期。位姿精度本身没有问题。

### 1.2 内参误差（这里暴露了我们的第一个 bug）

我们记录的换算公式是

```
u_model = A @ u_original          A = [[sx,0,tx],[0,sy,ty],[0,0,1]]
K_original = A^-1 @ K_model @ A
```

**这个共轭变换是错的。** `A` 是相似变换，`A^-1 K A` 在代数上**原样返回 `K_model`**——焦点、主点全不变。于是导出的 `K_original` 等于 `K_model`：

```
K_model    f = 341.76   (model 宽 624)
K_original f = 341.76   (原图宽 6211)   ← 应是 3410.3
```

焦距差了 **10 倍**。但由于我们**同时**把这个错误 K 用在反投影上，深度和相机"自洽"地一起错了，**在单视图里看不出来**，只有和真值比才暴露。

正确写法是按像素网格缩放内参（`K` 表达在哪个像素系里）：

```python
k_original = [[f_model / sx, 0, (cx_model - tx) / sx],
              [0, f_model / sy, (cy_model - ty) / sy],
              [0, 0, 1]]
```

修正后：`f = 3401.7` vs 真值 `3410.34`，**误差 −0.25%**。这也反过来证明 **VGGT-Ω 的内参预测非常准**。

### 1.3 像素仿射本身也错了两次

第一版用"坐标编码探针图"去反推 `A`，踩了两个坑：

1. **探针目标尺寸用错了参数。** `load_and_preprocess_images` 的 `balanced` 模式按 **token 预算** 决定目标宽高（`_balanced_target_shape`），不是我传入的 `image_resolution`。传 624 得到的张量是 **768×512**，拟合出的斜率 3.10 实际应为 2.52。
2. **缩略图坐标恢复时重复缩放。**

两次都表现为"看起来能跑，数字差一个常数倍"。最终改成**直接调用 loader 自己的 `_crop_to_supported_aspect_ratio` + `_balanced_target_shape`** 解析计算，并加断言 `sx * W ≈ model_width`，错就直接报错而不是静默出错误结果。

## 2. 后处理里的其它 bug（与模型无关）

### 2.1 Open3D TSDF 给空网格

`open3d 0.20` 的 `ScalableTSDFVolume.integrate()` 在本机**即使对合成等深度平面也返回空网格**（`extract_triangle_mesh()` 0 个三角形）。`UniformTSDFVolume` 正常，`VoxelBlockGrid` 需要显式 block 分配。为避免绑定到某个 Open3D 版本的行为，TSDF 改为 `scripts/tsdf_mesh.py` 里的自实现（约 100 行 numpy + `skimage.measure.marching_cubes`），公式严格按 `MESH_UV_PIPELINE.md` §3.2。

### 2.2 未观测体素被当"外部" → 网格碎成格子

自实现的第一版把未观测体素填 `+1`（外部）。由于 `sdf = clamp((observed−z)/trunc, −1, +1)` 对**表面之前的整个空间**都返回 `+1`，自由空间和未观测空间无法区分，结果表面被切成一格格孤立面片：

```
修复前：非流形边 50,994，网格呈"帘幕/格子"状
修复后：只更新 ±trunc 带内体素，未观测区保留 NaN（未知 ≠ 外部），并在 marching cubes 后剔除
        全部落在未观测区的三角面
        非流形边 0
```

### 2.3 深度一致性滤波的单位错配

`consistency_filter.py` 原本用 `K_original` 反投影**模型分辨率**的深度图。在 1.2 的 bug 存在时两者恰好相等，所以测不出来；内参一修好就立刻崩（一致率 0，只剩 2 对）。

修正为统一用 `K_model`（深度图所在网格），这才是定义深度的那组内参。

## 3. 修复前后的量化对比

同一场景 `eth3d_kicker`、同样 12 帧、同样 TSDF 参数：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 跨视图深度一致率（中位数，\|Δz\|≤5%z） | 0.168 | **0.819** |
| 通过的视图对 | 92 | **105** |
| 深度保留像素比例 | 16.2% | **92.0%** |
| 网格三角面 | 386,930（帘幕状） | 418,865（实心曲面） |
| 非流形边 | 50,994 | **0** |
| ExMesh UV 图 | 噪点 | 可辨识的 kicker 纹理岛 |

一致率从 0.168 到 0.819 这个变化最能说明问题：如果 VGGT-Ω 的深度本身不准，修好后处理参数不可能让"不同视角看到的同一表面"突然对齐到 82%。

## 4. 仍然存在的限制（不要当成已解决）

1. **只有 12/25 帧**：25 帧在 8 GiB 卡上 OOM（峰值约 4.2 GiB 权重 + 激活），低显存档位下取 `--max-frames 12`。
2. **边界边 33,375**：TSDF 网格不是闭合曲面，物体背面/未观测区域是洞。这是"只观测到有限视角"的物理限制，也是 `MESH_UV_PIPELINE.md` 要求单独标记的区域。
3. **ExMesh 只跑了 200 步**（官方 10,000 步、`-r 2`、2560² 纹理）。当前产物只能证明管线通，**不能作为质量结论**。
4. **没有跑几何评价指标**：ETH3D 需要官方 scan 对齐/可见性协议才能报 Chamfer/F-score，本仓库还没实现 -> 目前所有"几何好坏"的判断都基于可视检查和自洽性，不是 benchmark 数字。
5. **scale 非米制**：所有尺寸都是预测 gauge，跨方法比较必须先做 train-only Sim(3)。

## 5. 复现

```bash
# 1. VGGT-Ω 几何导出（GPU；含内参与像素仿射）
python3 scripts/adapters/vggt_omega_geometry.py --scene-dir data/scenes/eth3d_kicker \
    --checkpoint data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt \
    --device cuda --aggregator-dtype bfloat16 --image-resolution 512 --max-frames 12

# 2. 对真值检查（CPU）
python3 scripts/verify_omega_geometry.py --scene eth3d_kicker

# 3. 多视图一致性 + TSDF + mesh
python3 scripts/consistency_filter.py --scene eth3d_kicker --min-agree 1 --tau-rel 0.12 --max-frames 12
python3 scripts/tsdf_mesh.py --scene eth3d_kicker --voxel-div 256 --trunc-voxels 4 --stride 2

# 4. ExMesh 输入 + 训练（见 docs/EXMESH_ADAPTER.md）
python3 scripts/exmesh_adapter.py --scene eth3d_kicker
# 5. 导出 Blender 资产
python3 scripts/export_exmesh.py --checkpoint <...>/model_state_dict.pt
```

## 6. 教训

三类 bug 的共同点是**"自洽但错"**：错误的 K 配上错误的深度、未观测区当成外部、仿射重复缩放，单独看每个模块都"能跑"，只有对着真值才暴露。因此本仓库后续所有几何改动都要求：

- 有真值时**必须**在 `verify_omega_geometry.py` 里加断言（如 `sx*W ≈ model_width`、`K_original.f ≈ GT`）；
- 没有真值时用**已知几何的合成场景**（`scripts/make_smoke_scene.py`）验证解析式；
- 文档里区分"已验证"与"计划中"，不把"跑通"写成"效果好"。
