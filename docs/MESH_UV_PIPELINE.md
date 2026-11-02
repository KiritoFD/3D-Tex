> **2026-11-02 implementation update:** The Omega/TSDF/ExMesh path exists. Historical pending-status text below describes the original plan. See [PIPELINE_FIXES.md](PIPELINE_FIXES.md) for current corrections, reproduced artifacts and remaining quality limits.

# 从 VGGT-Ω 到 Mesh + UV：无 Gaussian 主线（2026-11-01）

**决策：不训练 3DGS / 2DGS，不把 Gaussian 当最终物体表面，也不把它作为默认中间阶段。**

这是工程实施方案，**不是已经跑通的代码**。当前仓库仍只有旧版 VGGT 的 COLMAP 导出及旧版 2D-SuGaR 参考代码；VGGT-Ω、新 TSDF、UV 烘焙、Mesh 优化适配器需要实现并在 GPU 上验收。[研究选型](SOTA_2026.md) | [Ω 接口](VGGT_OMEGA_ADAPTER.md)。

## 1. 推荐的分层图

```text
train-view RGB (12--25 images for ETH3D examples)
    |
    v
VGGT-Ω 1B 512 / 416 ----> K_i, T_cw_i, depth_i, depth_conf_i
    |                              |
    |                       frame/camera/depth alignment & checks
    |                              |
    +--------------------> robust multiview depth consistency
                                   |
                      sparse TSDF / volumetric fusion (CPU first)
                                   |
                         marching cubes / cleanup
                                   |
                            mesh_init.ply
                                   |
             +---------------------+------------------------+
             |                                              |
     A: direct mesh optimizer                        B: GeoSVR (voxel)
     ExMesh adapted to TSDF init                     voxel optimization
     topology split/merge + UV                      TSDF mesh extraction
             |                                              |
       mesh_refined + UV                            mesh_refined -> xatlas
             +---------------------+------------------------+
                                   |
                       UV unwrapping (xatlas, if needed)
                                   |
                          z-buffer multiview RGB fusion
                                   |
                       visibility-aware texture refinement
                                   |
                    seam consistency + missing-area handling
                                   |
                         OBJ/MTL/PNG, GLB (+ optional PBR)
```

A **最贴近最终资产**，但需改造 ExMesh 官方初始化；B **不使用 Gaussian 几何主表征**、已有可执行训练代码，不过是 voxel -> mesh，不能宣称直接 Mesh 优化。零训练基线是 Ω → TSDF → xatlas → 多视图烘焙。MeshSplatting 可以作为第三条“triangle-based”对照，不进入默认路径。**任何方案都不能保证单靠没有观测到的照片重建真实背面；这种区域必须单独标记。**

## 2. VGGT-Ω 的标准输出契约

统一以**原始图片像素坐标**保存：
- `images/train/*.jpg`、`cameras.json`：每帧的 `K(3x3)`、`T_world_to_cam(4x4)`、原图尺寸、预处理变换；
- `depth/<frame>.npy`：与对应相机内参配准的**相机 z 深度**（非 ray length；须检查模型的输出定义）；
- `confidence/<frame>.npy`：模型深度置信度/不确定性原值，不强行假设越大越好，先用重投影误差标定方向；
- `masks/<frame>.png`：物体/有效深度区域；记录是否用 GT 掩码（评估时单列）；
- `points_world.ply`：由 K + depth + pose *推导*，不是 VGGT-Ω 独立 PointMap 头；
- `manifest.json`：模型 checkpoint SHA、图像顺序、像素变换、坐标系、推理 dtype、版本。

**重要事实：** VGGT-Ω 的官方示例返回 `predictions["pose_enc"]`、`depth`、`depth_conf`；不像旧 VGGT 需要直接读取一个 PointMap head。[官方代码](https://github.com/facebookresearch/vggt-omega)。

设模型提供的 `T_cw=[R|t]` 是世界到相机，像素 `p=[u,v,1]`，相机 z 深度 `d`：

```text
X_cam   = d * inv(K) @ p
X_world = R.T @ (X_cam - t)
```

**禁止**直接将 512/416 模型输入空间的 K 与原始大图/裁剪图混合使用。显式保存 resize、pad、crop 的像素仿射矩阵，使用 `K_original = A^{-1} K_model`（若 `A` 定义为 original→model）。执行 `project(unproject(pixel))` round-trip，平均误差应接近浮点精度。

VGGT-Ω 输出通常是单一重建坐标规范（gauge），**不承诺真实米制尺度**。跨方法比较前用仅训练视图估算 Sim(3) 对齐；不能用 heldout/激光 GT 调参或选择模型。

## 3. 无学习的可靠几何基线：多视图深度 -> TSDF -> Mesh

### 3.1 输入过滤：比增加网络更重要

1. 排除背景/天空、无穷远、弱纹理反射、高亮、无效深度、外扩像素。
2. 在每对有重叠的视图 `i,j`，把 `X_i(u)` 投影到 j；检查像素落在可见区域、z>0，且 `|z_projected - D_j(v)| <= τ_abs + τ_rel * z`。可用局部梯度/置信度决定阈值。
3. 深度置信度需要按**跨视图一致性分桶校准**：不要凭单一默认阈值判断好坏；对不一致像素降权/剔除。
4. 稀疏/薄壁物体需允许 open surface；TSDF 平滑可能抹去薄片、孔洞和尖锐边缘，且相机漂移会造成双层面。

### 3.2 TSDF 实现

初版直接使用 [Open3D ScalableTSDFVolume](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html) CPU 后端；有多视图权重需求时改成 `t.geometry.VoxelBlockGrid` 或自定义稀疏哈希 TSDF。

每个体素距离场更新（符号随实现文档统一）：

```text
sdf = clamp((observed_depth - projected_voxel_z) / truncation, -1, +1)
F'  = (W*F + w*sdf)/(W+w)
W'  = min(W+w, W_max)
```

基线先用固定权重，再消融 `w = calibrated_depth_reliability * visibility * max(0,n·view)^γ`。体素大小以 ROI 的对角线 D 为参考：粗 `D/256`、中 `D/512`、细 `D/1024`，`truncation≈3–5 voxel` *只作搜索初值*；DTU 与 ETH3D 场景尺度不能直接共享米制阈值。

导出 `mesh_init.ply`，记录 vertices/faces、连通分量数、非流形边、边界边、退化面、覆盖率。最大分量清理只能在“单物体”场景使用，不应默默删去器物把手等真实多组件。

### 3.3 为什么要 mesh optimization

TSDF 融合直接是**可运行且可复现**的第一版，但过度平滑、模糊尖角和漏掉未见表面。细化策略分开对比：
- **ExMesh 式显式优化**：初始 `mesh_init.ply` → 可微三角形光栅化 → 顶点/UV 纹理双路优化 + 动态分裂/合并，目标是最终可编辑 Mesh。**官方 ExMesh README 默认 PGSR→Mesh；替换 TSDF 为我们的改造，未经论文验证**。[源码](https://github.com/Fan-Treasure/ExMesh)。
- **GeoSVR 式稀疏体素优化**：COLMAP cameras/images + 外部深度约束 → sparse voxel optimization → TSDF/mesh extract；更强几何指标但会有单独 UV 步骤。[源码](https://github.com/Fictionarry/GeoSVR)。避免“借用 depth priors”的比较偏差，改成 Ω 的深度需做分支实验。
- **MeshSplatting**：从三角形 soup → restricted Delaunay → 连通不透明三角网格；不是 3D Gaussian 主线，但用专用 triangle splatting rasterizer，**默认不导出可复用 UV 纹理图**，须另加参数化与烘焙。[源码](https://github.com/meshsplatting/mesh-splatting)。

### 3.4 优化损失（建议，不是现有代码）

```text
L = λ_rgb * robust_photo(render(mesh,T), RGB)
  + λ_sil * silhouette_loss
  + λ_depth * robust_depth(render_depth, filtered_Omega_depth)
  + λ_normal * normal_consistency
  + λ_lap * mesh_laplacian
  + λ_edge * edge_length_regularization
```

光度重建与 depth 约束必须分离，并标记**因深度相机共用同一个 Ω 预测而非独立 GT**。优化前先冻结 UV 然后几何、几何稳定后冻结 Mesh 优化材质，避免 shape/lighting/texture 互相代偿。不能把 mesh triangle soup 的顶点色当高质量纹理。

## 4. UV Atlas：与几何独立，面-角 UV 而非每顶点独占

1. 稳定的三角面拓扑 -> [xatlas](https://github.com/jpcy/xatlas) chart segmentation / parameterization / packing；或保留 ExMesh 动态维护的 UV。
2. 存储 `vertices[N,3]`、`faces[M,3]`、`uv_coordinates[K,2]`、`face_uv[M,3]`、chart ID、每 chart 的 gutter/mip pad；seam 上一个 3D 顶点可对应多个 UV 顶点，这是**正常**的。
3. 查验 UV finite / [0,1] / 面积 >0 / 覆盖率 / 非授权的 charts 重叠 / 重叠三角形 raster test / 每 chart 不小于目标 texel/mip padding。
4. 产物 `mesh.obj`+`mesh.mtl`+`albedo.png`，以及 `asset.glb`；材质贴图与 OBJ 的 vt/面-角拓扑要一致。UV 解法不同，atlas 的逐像素 MSE 不能直接跨方法比较。

## 5. 纹理：先观测重建，最后生成

**第一阶段：严谨的多视图投影融合**

对 UV 每个有效 texel：由 face/barycentric 求表面 `X,n`；投影到所有训练相机；做 z-buffer 一致性/遮挡检查；高分辨率 RGB 采样；融合权重使用可见性、拍摄角、投影 texel footprint、清晰度、曝光一致性。强烈建议先做 weighted median / IRLS Huber，对高光和跨视角颜色偏差有鲁棒性。

保留 `texture_observed.png`、`observed_mask.png`、`weight_sum.exr`、`source_view_id.png`，**不能直接把没有观测到的纹理当作真实重建**。深度来自低分辨率 Ω，纹理颜色从原始高分辨率 RGB 取样，只需要正确追踪内参变换。

**第二阶段：接缝与曝光一致性**

从 Mesh 相邻三角形的共享 3D 边识别跨 chart seam pair，构造等距采样点 `(t_left,t_right)`：
```text
L_seam = mean_pairs rho(T(uv_left)-T(uv_right))
L_photo = mean_visible rho(render(T, mesh, camera_i)-I_i)
```
可以先做 seam strip blending + atlas gutter dilation，再用固定 Mesh 的 differentiable renderer 优化 UV 纹理，且保证观测区域的 high-frequency detail 不被全图平滑抹去。

**第三阶段：未观察区域补全（可选）**

CVPR 2026 的 **MV2UV** 针对 UV / 多视图联合条件；**MatLat** 针对预训练图像生成模型 PBR 纹理和 VAE locality；**NaTex/Lafite** 是更重的原生 3D latent texture 路线。不要混淆生成质量 FID 与实拍纹理还原误差。保留 repaired/generated mask 与纯观测纹理并列输出。

**PBR 进阶**：ExMesh++ (arXiv 2026) 先冻结 mesh+UV，再优化 UV base-color/roughness/normal（可选 metallic）与 environment illumination；支持重光照，但论文当前是**预印本**，不能声称本仓库已有完整实现。[论文](https://arxiv.org/abs/2608.24109)。

## 6. 数据和评估（验收门槛）

- `ETH3D pipes/relief/kicker`：继续作为下载、姿态、多视图深度/网格融合的 **integration smoke test**；它们不是干净的单个文物 GT UV 数据。
- `DTU`：几何 benchmark。对齐 DTU 官方 calibration/masks/evaluation protocol；不能混用不同作者脚本生成的平均 CD 来宣称统一排名。
- `已知 textured mesh 的小资产`（例如允许使用的 GSO / 自制 Blender 小器物）：UV 完整性、渲染保真、接缝统计、不可见区域补全；必要时用固定 GT Mesh 剥离几何误差。
- 泄露禁止：VGGT-Ω 和后续优化只用 train images；独立 heldout 所用 GT camera 必须经 train-only Sim(3) 映射到预测坐标；绝不以 test 来挑参数。

| 轴 | 必须报告 |
|---|---|
| 几何 | CD、F-score @ 明确阈值、normal consistency、完整率、薄结构/孔洞与连通分量 |
| UV | UV valid face%、岛数、重叠%、有效 texel coverage、seam pair 数、gutter |
| 纹理 | heldout PSNR/SSIM/LPIPS、seam deltaE/RGB、high-frequency recovery、已观测 vs 生成掩码 |
| 资源 | 每阶段 wall time、GPU 峰值和主机 RAM，记录模型/数据分辨率 |

**独立三个可比消融**：
A0 = Ω depth + TSDF + xatlas + bake（0 场景训练）。
A1 = A0 mesh -> **改造版** ExMesh 顶点/拓扑优化 + xatlas/bake。
A2 = Ω cameras + GeoSVR voxel optimization -> mesh -> xatlas/bake。
仅在 A0/A1/A2 固定几何之后，再比较 UV 纹理算法。

## 7. 资源与工程实现顺序

- **CPU first**：synthetic 立方体/薄片/多 chart UV，project-unproject、TSDF 提取、xatlas 参数化、z-buffer/occlusion、seam pairing；小样本 ETH3D 复查。这个阶段没有 1B 推理。
- **一张 4090 24GB**：Ω 25 帧官方 A100 512 checkpoint peak **7.80GB**（不能当成 4090 实测）；GeoSVR 论文附录 11–14GB 的工作负载；ExMesh 论文 DTU reported 4.5GB（**PGSR 初始化，不是 Ω→TSDF 改造版**）。每阶段顺序执行不需要多卡合并显存。
- **两张 4090**用于 A1/A2 并行消融，而非主线功能要求。
- **Gate 0**：完成新的 `run_omega.py` 与标准 depth/camera exporter；**当前没有**。
- **Gate 1**：CPU baseline A0 生成真实 `mesh.obj`+`albedo.png`，接缝/UV 单测通过。
- **Gate 2**：GPU A1/A2 兼容/性能验证，再决定是否移植 ExMesh 或 GeoSVR 必要代码（不引 submodule）。
- **Gate 3**：固定 mesh 的 UV/接缝优化和 Direct UV VAE 实验；不混合优化目标。

主要文献：[VGGT-Ω](https://github.com/facebookresearch/vggt-omega)、[Open3D TSDF](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html)、[ExMesh](https://arxiv.org/html/2606.07288)、[GeoSVR](https://arxiv.org/html/2509.18090)、[MeshSplatting](https://meshsplatting.github.io/)、[xatlas](https://github.com/jpcy/xatlas)、[ExMesh++](https://arxiv.org/abs/2608.24109)。