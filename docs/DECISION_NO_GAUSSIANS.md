# ADR: 2026-11-01 — 3D-Tex 主线不使用 3D Gaussian

状态：**已批准架构方向，工程实现未完成。**

## 决策

- 删除根目录 vendored `3dgs/` 及主线 `run_3dgs.py`、`recon-3dgs` 环境/依赖配置，不再训练 gsplat 作为独立分支。
- 保留 `2dsugar/` 作为**冻结历史研究参考代码**，不在默认运行、安装或 SOTA 推荐主线中调用。它内部依赖 2DGS/3DGS；“留存源代码”不等于“工程采用 Gaussian”。
- `vggt/` 当前是旧版 VGGT；下一代将引入平级的 `vggt_omega/`，以相机、深度、置信度对齐输出为核心，**不假装已经迁移**。
- 目标表示唯一：三角 Mesh + 标准 per-corner UV + 有效性/观测掩码 + RGB/PBR 贴图。
- A0 = 深度融合 TSDF + xatlas + 多视图纹理；A1 = 改造 ExMesh；A2 = GeoSVR 对照。

## 为什么

3DGS 是快速可微**体渲染/外观表示**，对三角网格表面、watertight/非流形、UV/材质编辑不提供直接约束；它的优秀 PSNR 不代表 Mesh 拓扑正确。我们不需要先优化百万 Gaussian 再丢掉它们，只为获得表面。

并不意味着所有 splatting 研究都无效：MeshSplatting 采用三角形栅格化，并非 3D Gaussian；ExMesh 以 PGSR 初始化也说明在一些场景 Gaussian 可提供有用 bootstrap。这个决策针对本项目**主流程和维护成本**，不是“Gaussian 在任何场景都不可用”的物理结论。

## 已知代价和风险

TSDF 平滑薄结构、受 pose/depth bias 影响，无法凭空恢复未观察面；A1 直接用 TSDF 初始化而不是 ExMesh 官方 PGSR，没有保证成功。GeoSVR 精度强但输出需要额外 UV；真实照片下材质/光照耦合不可直接当 albedo。

## 迁移和验证

现存 `run_all.sh` 变成**显式失败的旧入口**，禁止静默跑旧路线；历史代码可从 Git history 获取。新 pipeline 的具体步骤和 gate 定义见 [MESH_UV_PIPELINE.md](MESH_UV_PIPELINE.md)。待 Ω/TSDF/UV 真正验收后才能提供新的 end-to-end 一键命令。
