# 72 小时几何研究计划（单卡 RTX 4090）

更新：2026-11-05 · **研究方向变更**（见「变更说明」）；2026-11-04 的阶段记录保留为历史并标为 superseded  
远端：`ssh 4090`（`/root/3dtex`）  
代码版本：`a702e33f7edb85c8d1d920de948f3857464407f4`（2026-11-04 的数据对应 `851065bd`）  
配置：[configs/experiments/72h.json](../../configs/experiments/72h.json)

## 变更说明（2026-11-05）

**结论：主线从“VGGT 深度/法线矫正”改为“以 VGGT-Ω 粗几何为初始化的 image-driven surface reconstruction（图像驱动表面重建）”。**

- **改了什么：** VGGT-Ω 只提供粗相机、尺度、轮廓与初始几何，**不再作为监督目标**；用原始多视图 RGB 的可微渲染直接优化表面，允许几何显著偏离 VGGT。阶段 1 用 nvdiffrec/FlexiCubes 与 ICLR Mesh Splatting 恢复准确表面（允许拓扑变化），阶段 2 用 ExMesh + 自适应 remesh 做显式精修与 UV/纹理；反光场景引入 Factored-NeuS 式材质/光照分解。
- **为什么：** P1 与阶段 3 已证明逐像素 photometric depth + VGGT 深度/法线先验会把外观误差解释成几何误差，并把 VGGT 的平滑偏差与细节缺失一起保留。
- **移除：** URGT 与 HyDen 不再作为主线、也不再作为附录候选；以此为先验的 A1/A2 深度—法线矫正路线一并删除。
- **保留（不改写历史）：** 阶段 2 冻结协议与全部测量数字不变；本文件的旧章节整体 superseded，新主线见 §10。
- **说明：** 本文件是随配置分发的计划快照；阶段 2/3 的完整执行记录（§10–§11）在 `docs/experiments/72H_GEOMETRY.md`，其 §12 与本文 §10 对应。
- **不变：** 不训练/微调 VGGT-Ω；不下载 DTU 或全量 ETH3D；`data/`、`outputs/` 只读。

## 0. 结论（2026-11-04，superseded by §10）

- **阶段 1 主链路通过。** VGGT-Ω 1B-512、nvdiffrast、ExMesh CUDA 扩展、Geometry Evaluation 均已在 4090 上跑通；Lego 基线在 4090 重建后 F1 = **0.6376**（4070 参考 0.6315），过 0.62 门槛。
- **4090 实测：** 16 / 32 / 64 视角推理峰值 **3.99 / 4.49 / 6.34 GB**，推理耗时 **0.97 / 1.77 / 4.10 s**（约 15.6–18.1 视角/s），64 视角远低于 24 GB。
- **M2 重复性地板已重建：** 10 次同输入重跑，Observed Chamfer = **0.009414 ± 0.000039**（CV 0.41%），F1 = **0.7365 ± 0.00127**；10 个 mesh 文件均不相同，说明地板来自真实的运行时非确定性，不是缓存复用。
- **旧 4070 地板不再沿用。** 相同输入在 4090 上重算的 VGGT 输出与旧缓存存在 bf16 非确定性差异，且 4090 上基线 F1 已变化，因此后续跨场景对比一律使用本轮 4090 地板。
- **阶段 1 尾巴已闭环：** SSH 于 `2026-11-04 22:22:53 +08:00` 恢复（`PORT_OPEN=True`、`SSH_OK`）；PyTorch3D `0.7.9` import 与 `quaternion_to_matrix` CUDA 前向通过；ExMesh `build_rotation` 前向通过，`simple_knn._C`、`diff_plane_rasterization` 均可 import。
- **基线显存补采样完成：** nvidia-smi 总占用峰值 8269 MiB（空闲 1721 MiB，含其他租户），扣除空闲后 **delta 6548 MiB**，两次重跑数值完全一致；脚本内 torch 报告峰值 3993 MiB。两者差异来自 CUDA context/权重流式加载等非张量暂存，报告以更保守的 nvidia-smi delta 作为资源上限。
- **未闭环项：** tinycudann / torch_scatter 按要求留到阶段 5；ETH3D 官方可见性协议未实现，阶段 2 的 ETH3D 数字一律标注为**内部几何评测**。

## 1. 远端与资源

| 项 | 实测 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24564 MiB |
| 驱动 / CUDA | 590.48 / 13.1 |
| 主机 | 容器 `7a9405e8fb76`，128 vCPU，251 GB RAM |
| 磁盘 | `/` 1.8T，已用 91%，余约 158 GB |
| 其他占用 | 其他用户 gradio 推理进程约 1.7 GB；未杀任何进程 |
| 并发 | 所有长任务用 tmux；日志落盘 |

远端工作区：

```text
/root/3dtex                  # 代码 + 最小数据 + 输出
/root/deps/nvdiffrast        # 源码构建
/root/deps/ExMesh            # CUDA 扩展
/root/deps/pytorch3d         # 源码构建（0.7.9，已验证）
/root/deps/blender/blender-3.3.9-linux-x64/blender
```

## 2. 环境迁移与验证

| 组件 | 状态 | 验证 |
|---|---|---|
| Conda `3dtex` | 通过 | Python 3.11；`torch 2.6.0+cu124`、`torchvision 0.21.0`；CUDA toolkit 12.4 |
| VGGT-Ω 1B-512 | 通过 | 16 视角完整导出 depth / confidence / cameras / points；权重 sha256 前 16 位 `c02da418b18bb01d` 与本地一致 |
| nvdiffrast | 通过 | 源码构建；CUDA rasterize 冒烟测试通过 |
| ExMesh CUDA | 通过 | `simple_knn._C`、`diff_plane_rasterization` import 通过；`utils.general_utils.build_rotation` 四元数前向 `(1,3,3)` 和 3.0 通过 |
| PyTorch3D | 通过 | 源码构建 `0.7.9`；`quaternion_to_matrix` CUDA 前向 `(1,3,3)` 和 3.0 通过 |
| Geometry Evaluation | 通过 | Blender 3.3.9 + 冻结 P0 网格；参考 mesh 重评测与历史指标一致 |
| Mesh Splatting 依赖 | 按计划延期 | tinycudann、torch_scatter 未安装（阶段 5） |

几何评测复现精度（同一参考 mesh）：

| 指标 | 历史值 | 4090 重测 | 差值 |
|---|---|---|---|
| Observed Chamfer | 0.012579542 | 0.012579542 | 9.6e-12 |
| F1 @1 voxel | 0.631527 | 0.631527 | 0.0 |

评测协议：冻结 P0 网格（voxel `0.01150144350298549`），50k 固定种子面积加权表面采样，GT 观测深度点仅用于 completeness；Sim(3) 只用训练视角估计（`alignment_views = 16`），不参与指标。

## 3. VGGT-Ω 推理迁移与显存/速度

命令：

```bash
python scripts/adapters/vggt_omega_geometry.py \
  --scene-dir <scene> \
  --checkpoint data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt \
  --out <out> --device cuda --aggregator-dtype bfloat16 --image-resolution 512
```

| Run | 输入 | batch（视角） | 峰值显存 | 推理耗时 | 速度（视角/s） | 过闸 |
|---|---|---|---|---|---|---|
| `vggt16` | nerf_lego16 | 16 | 3993 MiB | 0.97 s | 16.5 | 通过 |
| `vggt32` | capture_set32 | 32 | 4488 MiB | 1.77 s | 18.1 | 通过 |
| `vggt64` | capture_set64 | 64 | 6335 MiB | 4.10 s | 15.6 | 通过 |

说明：

- 三次加载权重约 20–24 s；表中的速度只计推理，不含加载。
- 16 视角使用 Lego 16 视角集，32/64 使用 `capture_set32/64`；三者的视角集合不完全嵌套，但批量规模和显存/吞吐趋势可直接用于阶段 2 的显存预算。
- 官方 A100 参考（25 张约 7.8 GB、100 张约 13.37 GB）与本轮 4090（16 张 3.99 GB、64 张 6.34 GB）不冲突：本轮使用 bf16 aggregator 且只做前向导出。

## 4. Lego 基线（4090 重建）

流程：4090 新 VGGT 输出 → `geometry_pruning`（`vote-frac 0.35`, `confirm-max 0.3`）→ train-only Sim(3) → confidence 加权 TSDF → 冻结 P0 网格 → Blender 表面评测。

| Run | 输入视角 | 指标 | 结果 | 门槛 | 判定 |
|---|---|---|---|---|---|
| `baseline_v35c30_conf_4090` | 16 | F1 @1 voxel | **0.6376** | ≥ 0.62 | 通过 |
| | | Precision / Recall | 0.5845 / 0.7012 | — | — |
| | | F2 @2 voxel | 0.8456 | — | — |
| | | Observed Chamfer | **0.012488** | ≤ 0.0135 | 通过 |
| | | 法向平均误差 | 50.10° | — | — |
| | | 深度 abs-rel（GT 网格上） | 1.234% | — | — |

与 4070 参考对比：

| 指标 | 4070 参考 | 4090 重建 | 变化 |
|---|---|---|---|
| F1 @1 voxel | 0.6315 | 0.6376 | +0.0061 |
| Observed Chamfer | 0.012580 | 0.012488 | −0.000092 |
| 法向平均误差 | 50.44° | 50.10° | −0.34° |

峰值显存（补采样，`outputs/72h/peak_sample/`）：

| 阶段 | nvidia-smi 总占用峰值 | 空闲基线 | delta | 脚本内 torch 峰值 | 采样 |
|---|---|---|---|---|---|
| VGGT-Ω 16 视角推理 | 8269 MiB | 1721 MiB | **6548 MiB** | 3993 MiB | 30 ms 轮询，两次重跑完全一致 |
| pruning v35c30 | 2339 MiB | 1721 MiB | **618 MiB** | — | 30 ms 轮询，耗时 0.6 s |

采样方式：外部进程每 30 ms 执行 `nvidia-smi --query-gpu=memory.used`，取峰值减去运行前空闲值，扣除其他租户常驻的 1721 MiB。容器内 `nvidia-smi --query-compute-apps` 不返回子进程行（`PROC_SEEN=0`），因此采用总占用 delta；两次 VGGT 采样 delta 同为 6548 MiB，说明窗口内无其他租户波动。资源表以更保守的 6548 MiB 作为基线 GPU 上限。吞吐：融合 + 评测约 1.7 min（CPU 主导），不涉及训练 sps。

## 5. M2 重复性地板（4090）

固定输入：`outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply`（sha256 前 16 位 `ecf994efd40a5d9c`）。  
固定配置：voxel `0.01150144350298549`、300 iters、lr 0.05、`lambda-prior 0.1`、`tolerance-voxels 1.0`、seed `20261103`、diagnostic 相机、256 分辨率。  
`deterministic=false`，即允许真实运行时非确定性暴露。

| Run | Observed Chamfer | F1 @1 voxel | Precision | Recall | 耗时 | it/s | 峰值显存 |
|---|---|---|---|---|---|---|---|
| run01 | 0.009421 | 0.7357 | 0.6852 | 0.7943 | 38.25 s | 7.84 | 0.159 GB |
| run02 | 0.009354 | 0.7387 | 0.6904 | 0.7943 | 33.40 s | 8.98 | 0.159 GB |
| run03 | 0.009420 | 0.7369 | 0.6878 | 0.7935 | 38.08 s | 7.88 | 0.159 GB |
| run04 | 0.009431 | 0.7354 | 0.6854 | 0.7932 | 35.28 s | 8.50 | 0.159 GB |
| run05 | 0.009411 | 0.7367 | 0.6869 | 0.7942 | 37.51 s | 8.00 | 0.159 GB |
| run06 | 0.009404 | 0.7365 | 0.6875 | 0.7930 | 35.99 s | 8.34 | 0.159 GB |
| run07 | 0.009375 | 0.7384 | 0.6894 | 0.7948 | 34.40 s | 8.72 | 0.159 GB |
| run08 | 0.009396 | 0.7366 | 0.6861 | 0.7951 | 38.13 s | 7.87 | 0.159 GB |
| run09 | 0.009434 | 0.7354 | 0.6857 | 0.7928 | 37.76 s | 7.94 | 0.159 GB |
| run10 | 0.009498 | 0.7348 | 0.6836 | 0.7942 | 37.17 s | 8.07 | 0.159 GB |

汇总：

| 统计量 | Observed Chamfer | F1 @1 voxel |
|---|---|---|
| 均值 | **0.009414** | **0.7365** |
| 样本标准差 | **0.0000387** | **0.00127** |
| 最小值 | 0.009354 | 0.7348 |
| 最大值 | 0.009498 | 0.7387 |
| 极差 | 0.000144 | 0.00393 |

门槛与判定：

| 门槛 | 结果 | 判定 |
|---|---|---|
| 重复次数 ≥ 10 | 10 | 通过 |
| Chamfer CV ≤ 1% | 0.41% | 通过 |
| F1 std ≤ 0.005 | 0.00127 | 通过 |
| 10 个 mesh 非同一文件 | 10/10 唯一 sha256 | 通过（真实非确定性） |

M2 相对本轮基线：Chamfer 从 0.012488 降到 0.009414（**−24.6%**），F1 从 0.6376 升到 0.7365（**+15.5%**）。后续阶段 3 的 G1/G2 对照必须以 **0.009414 ± 0.000039** 作为 M2 地板，不得沿用 4070 数值。

## 6. 已完成 / 未完成

已完成：

1. `ssh 4090` 连通性、GPU、磁盘、远端最小代码/数据/权重同步。
2. VGGT-Ω 1B-512 推理迁移与 16 视角导出验证。
3. 16 / 32 / 64 视角显存与速度实测。
4. nvdiffrast、ExMesh 两个 CUDA 扩展、Blender Geometry Evaluation 验证。
5. Lego 基线在 4090 上从新 VGGT 输出重建并过闸。
6. M2 10 次重复性地板重建。
7. 统一产物：`outputs/72h/summary.json`、运行日志、评测 JSON。
8. SSH 恢复后补验 PyTorch3D / ExMesh 前向，并补齐基线 GPU 峰值采样。

未完成 / 阻塞：

| 项 | 原因 | 影响 | 处理 |
|---|---|---|---|
| tinycudann / torch_scatter | 按计划延期 | 不影响阶段 2–4 | 阶段 5 前安装 |
| ETH3D | 阶段 2 才启动 | 当前无 ETH3D 数值 | 官方可见性协议未实现，一律标注“内部几何评测” |
| ETH3D 扫描指标器 | 仓库没有现成的 scan→mesh 评测脚本 | 阶段 2 需要新增/验证 | 先冻结协议，阶段 2 首日实现并做合成自检 |

SSH 中断为远端隧道瞬时不可达（`Connection timed out`），不代表任务失败；恢复探测记录见第 0 节，所有阶段 1 结果已落盘并同步回本地 `outputs/72h/`。

## 7. 阶段 2 评测协议（冻结于 2026-11-04）

冻结原则：本节写定后，阶段 2 执行期间不得修改视角拆分、基线定义、P0 网格口径、指标或门槛。任何偏离作为新 arm 追加记录，不覆盖既有结果；无效 run 必须写明原因。

### 7.1 场景与视角清单

| 场景 | 重建视角（train-only） | 持有视角（不参与重建与几何指标） | GT 几何 | 指标标注 |
|---|---:|---:|---|---|
| Lego `nerf_lego16` | 16 | 0 | 同 16 视角渲染 GT depth（observed） | 内部 P0 协议 |
| ETH3D `pipes` | 12 | 2 | 官方 laser `scan_eval` | 内部几何评测 |
| ETH3D `relief` | 25 | 6 | 官方 laser `scan_eval` | 内部几何评测 |
| ETH3D `kicker` | 25 | 6 | 官方 laser `scan_eval` | 内部几何评测 |

- ETH3D 合计 **62 重建视角 + 14 持有视角**；持有视角只保留给阶段 6 的外观/novel-view，绝不进入 VGGT、TSDF、网格优化或几何指标。
- 所有 GT（相机位姿、激光扫描、渲染 depth、foreground mask）只在评测脚本读取；重建输入只允许数据集原始图像与 VGGT 输出。
- Lego 的 16 视角同时用于重建和 observed GT depth 评测，这是阶段 1 既有 P0 协议；GT depth 不回流重建。
- ETH3D 若未实现官方可见性协议，Completeness 使用全扫描，数值偏保守，所有结论标注“内部几何评测”，不与官方榜单比较。

### 7.2 基线定义

| 臂 | 定义 | 使用 GT | 状态 |
|---|---|---|---|
| **B0** | 原始 VGGT depth/confidence → train-only Sim(3) → TSDF：uniform 权重、nearest 采样、无过滤、无 carve | 否 | 阶段 2 新跑 |
| **B1** | `consistency_filter`（跨视角一致性，no masks，`tau_abs = 1%` 相机跨度、`tau_rel = 0.05`、`min_agree = 2`、无 confidence 截断）→ confidence 加权 TSDF（nearest） | 否 | 阶段 2 新跑 |
| **B1\*** | Lego 遗留 `v35c30`：GT foreground mask 的 free-space carving + confidence 融合（阶段 1 基线） | **是**（GT foreground mask） | 仅诊断，不参与跨场景门槛 |

- B0 与 B1 使用同一份 VGGT 输出、同一 P0 网格、同一 train-only Sim(3)、同一采样与评测脚本，唯一差异是过滤与权重。
- B1\* 只保留 Lego，与 B1 并列展示并显式标注 GT-assisted；**不得把 B1\* 数字当作 B1**。
- 阶段 1 的 F1 0.6376 / Chamfer 0.012488 属于 B1\*；阶段 2 Lego 的 B0/B1 是新结果，门槛按 7.5 重新判定。

融合命令（两场景路径不同，但都只输入 VGGT 输出）：

```bash
# ---- ETH3D: B0 / B1 on the scene-level frozen grid ----
python scripts/tsdf_mesh.py --scene eth3d_<name> --filtered <out>/geometry/omega \
  --roi-geometry <out>/geometry/omega --out <out>/mesh_b0 \
  --voxel-div 256 --trunc-voxels 4 --weight-mode uniform --sample-mode nearest

python scripts/consistency_filter.py --scene eth3d_<name> \
  --geometry <out>/geometry/omega --out <out>/geometry/filtered \
  --mask-source none --tau-rel 0.05 --min-agree 2

python scripts/tsdf_mesh.py --scene eth3d_<name> --filtered <out>/geometry/filtered \
  --roi-geometry <out>/geometry/omega --out <out>/mesh_b1 \
  --voxel-div 256 --trunc-voxels 4 --weight-mode confidence --sample-mode nearest

# ---- Lego: capture_eval fuses into the existing frozen deliverables/lego_p0 grid ----
python scripts/capture/capture_eval.py --capture <lego_arm_dir> --out <out> \
  --p0-out deliverables/lego_p0 --fusion-sample nearest --weight-mode uniform   # B0
python scripts/capture/capture_eval.py --capture <lego_arm_dir> --out <out> \
  --p0-out deliverables/lego_p0 --fusion-sample nearest --weight-mode confidence # B1
```

Lego 的 `<lego_arm_dir>/geometry/omega` 指向该臂 depth（B0=原始 omega，B1=`filtered` + `confidence/` 软链指回同源 `omega/confidence/`），`geometry/gt` 软链到 `nerf_lego_gt16/geometry/gt`，B1 复算后仍用 `measure_surfaces` 保留阶段 1 的三角面指标。软链只作权重读取，不改变过滤结果。

### 7.3 P0 网格冻结口径

- **Lego：** 直接复用阶段 1 冻结网格 `deliverables/lego_p0`：voxel `0.01150144350298549`、origin `[-0.7297994798693271, -1.2320406083074635, -0.43665042425509004]`、shape `[127, 215, 134]`、truncation `0.04600577401194196`。B0/B1/B1\* 全部 assert 该值，不得重算。
- **ETH3D：** 每个场景以 B0 为唯一来源冻结一次：
  1. 只读 B0 的 VGGT depth，不使用 GT 相机或扫描；
  2. 以 `--roi-geometry <B0 omega> --voxel-div 256 --trunc-voxels 4` 推导 voxel/origin/shape；
  3. 写入 `outputs/72h/p0_grids.json`：`scene, voxel, truncation, origin, shape, roi_geometry_sha256, b0_mesh_sha256`；origin 定义为 `lo = pts.min - 2*trunc`，阶段 2 首日只读推导落盘，不改融合代码；
  4. B1、阶段 3 的 G 臂一律传 `--roi-geometry <B0 omega>`，保证同一网格；
  5. 若某臂的 voxel/shape 与冻结值不一致，该 run 判无效并重跑，禁止各臂自行拟合网格。
- **禁止 GT-derived 网格**：不得用 GT 相机反投影 bbox 或扫描 bbox 定网格；ETH3D 扫描只用于评测。

### 7.4 指标与评测器

统一评测器（阶段 2 首日实现并先通过合成自检，写入 `scripts/`）：

- 输入：预测 mesh、GT 点集、P0 voxel；输出 `surface_metrics.json`。
- 对齐：仅用训练视角相机中心估计 Sim(3)（Lego 用 GT 相机，ETH3D 用官方 `cameras.txt`）；scale 同时作用于深度与 mesh；不做逐视角 scale/shift。若 ETH3D 校准位姿与扫描不在同一 gauge，允许一次基于训练视角的 Sim(3)，仍标注 internal。
- 采样：mesh 侧 50k 固定种子面积加权表面采样；GT 侧 50k 固定种子采样（Lego 用 `observed_gt_samples.npy`，ETH3D 用 `scan_eval/*.ply`）。
- 指标：
  - **Accuracy** = mesh 采样点到 GT 点集的平均最近距离；
  - **Completeness** = GT 采样点到 mesh 采样点的平均最近距离；
  - **Chamfer** = 两者对称平均；
  - **F1 @1 voxel** = precision（accuracy ≤ 1 voxel）与 recall（completeness ≤ 1 voxel）的调和平均；同时输出 F2 @2 voxel；
  - 阈值换算：Lego 的 P0 网格已在 GT gauge，直接使用 1 voxel；ETH3D 的冻结网格在预测 gauge，阈值取 `P0 voxel × Sim(3) scale`，并把 scale 写入 `surface_metrics.json`；
  - Lego 额外保留 Blender `measure_surfaces` 的三角面精确指标与法向误差，作为与阶段 1 的连续性对照。
- 输出命名：`outputs/72h/<scene>_<arm>/surface_metrics.json`，必须引用 P0 网格哈希与 GT 采样哈希。

### 7.5 门槛

每个 run 的有效性门槛（无效即判 fail 并写明原因）：

| 项 | 门槛 |
|---|---|
| VGGT manifest | 视角数、checkpoint sha256、depth 无 NaN/Inf |
| mesh | 存在；顶点 > 1000、三角面 > 0；顶点无 NaN/Inf |
| 网格一致性 | voxel/shape 与冻结值一致 |
| 资源 | 单场景单臂 ≤ 30 min；nvidia-smi 峰值 delta ≤ 10 GB |
| 记录 | batch（视角数）、峰值显存、sps、过闸判定、日志路径齐全 |

B1 相对 B0 的判定：

| 项 | 门槛 |
|---|---|
| 单场景“改善” | Chamfer 相对下降 ≥ 2% **且** F1 绝对提升 ≥ 0.005 |
| 跨场景过闸 | ≥ 2/3 ETH3D 场景改善 **且** Lego 改善 |
| 退化保护 | 任一场景 Chamfer 退化 > 5% 或 F1 下降 > 0.02 即整体不过闸 |
| Lego 绝对地板 | B1 Chamfer ≤ 0.0135 且 F1 ≥ 0.62（沿用阶段 1 门槛，不因 B1 去 GT 而放宽；未达即记录为负结果） |

ETH3D 不设事前绝对门槛：首个 B0 数值是描述性基线，阶段 3 起引用时在同一协议下复现；只执行上述相对门槛。

### 7.6 ETH3D 数据与下载检查

- `python scripts/download_eth3d.py --dry-run` 通过：3 个场景、6 个官方 URL，官方近似压缩体积 **1.30 GB**。
- 本地三个场景已就绪；manifest 拆分：pipes `12/2`、relief `25/6`、kicker `25/6`。
- 不下载 DTU，不拉全量 ETH3D；只允许 `--scenes pipes relief kicker`。
- 下载器自带 7z 完整性校验、SHA256 归档哈希与确定性拆分，拆分在 VGGT 之前完成。

## 8. 下一步（阶段 2 执行顺序）

1. 实现并自检统一扫描/点集指标器（合成 mesh + 已知偏移，误差应接近注入值）。
2. 冻结 ETH3D B0 网格到 `outputs/72h/p0_grids.json`，确认 Lego 复用 `deliverables/lego_p0`。
3. 依次跑 Lego B0/B1、ETH3D B0/B1；全部 tmux + 日志落盘，按 7.5 记录与判定。
4. B1 过闸后进入阶段 3（G1/G2），以阶段 2 各场景 B1 数值为对照；不覆盖阶段 1/2 记录。
5. 若 ETH3D 指标异常，先检查对齐 gauge 与扫描单位，再判定 run 无效；不得静默删除。

## 9. 产物索引

| 产物 | 本地 | 远端 |
|---|---|---|
| 汇总 JSON | `outputs/72h/summary.json` | `/root/3dtex/outputs/72h/summary.json` |
| VGGT manifests | `outputs/72h/vggt_*/manifest.json` | `/root/3dtex/outputs/72h/vggt_*/manifest.json` |
| 日志 | `outputs/72h/logs/` | `/root/3dtex/outputs/72h/logs/` |
| 基线指标 | `outputs/72h/baseline/surface_metrics.json` | `/root/3dtex/outputs/72h/baseline/eval_v35c30_conf/` |
| M2 10 次指标 | `outputs/72h/m2_floor/surface_metrics.json` | `/root/3dtex/outputs/72h/m2_floor/eval/surface_metrics.json` |
| M2 10 个 mesh | 未回传（体积大） | `/root/3dtex/outputs/72h/m2_floor/run01..run10/mesh_m2.ply` |
| 4090 VGGT 输出 | 未回传（体积大） | `/root/3dtex/outputs/72h/vggt_lego16/`、`vggt_set32/`、`vggt_set64/` |
| 峰值采样日志 | `outputs/72h/logs/peak_sample.log` | `/root/3dtex/outputs/72h/logs/peak_sample.log` |
| 峰值采样输出 | 未回传（体积大） | `/root/3dtex/outputs/72h/peak_sample/` |
| 阶段 2 协议 | 本文件 §7 + `configs/experiments/72h.json` `phase2_protocol` | 同步 |

关键输入哈希（sha256 前 16 位）：

| 输入 | 哈希 |
|---|---|
| VGGT-Ω 1B-512 权重 | `c02da418b18bb01d` |
| M2 初始 mesh | `ecf994efd40a5d9c` |
| Lego omega cameras | `ca0e334cd3335f8f` |
| Lego GT cameras | `a175631fab8c0264` |
| 冻结 P0 网格报告 | `267d256959724420` |
| source edge flags | `1feff905b1b3b8f3` |
| Lego 首帧图像 `r_0.png` | `90d5de8283795410` |

## 10. 新主线：image-driven surface reconstruction（2026-11-05 起）

完整版见 `docs/experiments/72H_GEOMETRY.md` §12（含阶段 2/3 历史记录）。

### 10.1 立场

- VGGT-Ω 只负责粗相机、尺度、轮廓与初始几何，**不是监督目标**；允许优化后的表面显著偏离 VGGT。
- 优化目标是逆渲染 `M*, A, L = argmin L(Render(M, A, L), I_1..N) + λ R(M; M_VGGT)`，`R` 是弱初始化正则，不是逐像素深度拟合。
- 反光物体的亮斑与颜色不只由几何决定；必须用 Factored-NeuS 式材质/光照分解，否则 RGB 损失会把几何误差解释掉。
- 允许拓扑变化：错误三角面应能被移除并重建，而不是只做顶点位移。

### 10.2 两阶段

| 阶段 | 方法 | 输出 | 不做 |
|---|---|---|---|
| Stage 1 表面恢复 | nvdiffrec / FlexiCubes（SDF/DMTet）与 ICLR Mesh Splatting（多层软化 + 重网格） | Ω 粗几何 + RGB → 准确表面 | 不做 UV；不追求少三角面；不引入 VGGT 深度监督 |
| Stage 2 显式精修 | ExMesh + 自适应 remesh + 法线精修 + UV/纹理烘焙 | 准确表面 + RGB → OBJ/GLB/UV/texture | 不重新引入强 VGGT depth loss |

Stage 1 监督：`λ_s L_silhouette + λ_c L_correspondence + λ_r L_render + λ_g L_geometry`。

### 10.3 第一阶段实验（Lego16，固定 Ω 相机与同一组 16 张 RGB）

**Step 0 优化目标有效性测试（先于 A/B/C，未过则不启动对照）**

| 子测试 | 做法 | 通过条件（暂定，Step 0 后冻结） |
|---|---|---|
| 0a 外观拟合 | 固定 GT Mesh / VGGT Mesh，只优化受约束外观；16 训练视角 + 4 heldout | heldout 上 GT-Mesh 误差明显低于 VGGT Mesh |
| 0b 已知扰动 | 对 GT Mesh 施加 ±0.5/1/2 voxel 法线偏移、平滑、填洞，再优化几何 | 梯度方向与“指向 GT”一致率 ≥80%，误差单调下降 |
| 0c 反光诊断 | 亮斑/高光区单独统计 0a、0b | 亮斑区出现反向梯度即判外观模型失败，先上材质/光照分解 |

**A/B/C 对照**

| 实验 | 初始化 → 方法 | 目的 |
|---|---|---|
| A | Ω TSDF → ExMesh | 显式 Mesh 方法在固定 Ω 相机下的真实能力 |
| B | Ω TSDF → Mesh Splatting | 多层可微优化 + DMTet 拓扑变化能否推正错误表面 |
| C | Ω SDF → nvdiffrec / FlexiCubes | 联合表面/材质/光照能否分离几何与外观 |

### 10.4 验收（沿用阶段 2 冻结协议，不改历史数字）

| 项 | 门槛 | 现状参考 |
|---|---|---|
| F1 @1 voxel | ≥ 0.75 | 参考 Mesh 0.632；M2 0.7365 |
| 法线平均误差 | < 50.44° | 4070 参考 50.44°；M2/G2 48.7–48.8° |
| Sharp Recall | ≥ 0.694 | 冻结值 0.694；B0 0.7166 |
| 孔洞 / 薄结构 | 可视质量改善 | deletion-only 路线只能开孔 |
| Heldout 渲染 | 不退化 | 需先补 held-out 协议 |
| 资源 | ≤30 min/臂、≤10 GB、记录齐全 | 沿用阶段 2 §7.5 |

### 10.5 历史边界

- 阶段 2 冻结协议与阶段 2/3 测量数字全部保留，不改写。
- 旧 G1/G2/G3 属于 VGGT 深度—法线矫正路线，已 superseded，仅作“深度先验方向封死”的证据。
- URGT/HyDen 不进主线、不进附录；A1/A2 深度—法线先验矫正取消。
