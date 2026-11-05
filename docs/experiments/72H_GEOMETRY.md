# 72 小时几何研究计划（单卡 RTX 4090）

更新：2026-11-05 · **研究方向变更**（见「变更说明」）；2026-11-04 的阶段 1–3 记录保留为历史并标为 superseded  
远端：`ssh 4090`（`/root/3dtex`）  
代码版本：`a702e33f7edb85c8d1d920de948f3857464407f4`（2026-11-04 的数据对应 `851065bd`）  
配置：[configs/experiments/72h.json](../../configs/experiments/72h.json)

## 变更说明（2026-11-05）

**结论：主线从“VGGT 深度/法线矫正”改为“以 VGGT-Ω 粗几何为初始化的 image-driven surface reconstruction（图像驱动表面重建）”。**

- **改了什么：** VGGT-Ω 只提供粗相机、尺度、轮廓与初始几何，**不再作为监督目标**；后续用原始多视图 RGB 的可微渲染直接优化表面，允许几何显著偏离 VGGT。阶段 1 用 nvdiffrec/FlexiCubes 与 ICLR Mesh Splatting 从粗几何恢复准确表面（允许拓扑变化），阶段 2 用 ExMesh + 自适应 remesh 做显式精修与 UV/纹理；反光场景引入 Factored-NeuS 式的材质/光照分解。
- **为什么：** P1 与 §11 已证明，逐像素 photometric depth（含 ZNCC/SSIM + 深度/法线先验）在反光弱纹理场景下会把外观误差解释成几何误差；以 VGGT 深度为目标还会把它的平滑偏差、错误连接和细节缺失一起保留。继续沿深度矫正方向无法越过 0.632 / 0.737 这组参考。
- **移除：** URGT 与 HyDen 不再作为主线，也不再作为附录候选；此前以它们为先验的 A1/A2 深度—法线矫正路线一并删除。`docs/experiments/GEOMETRY_OPTIMIZATION.md` 的 route 表已移除 URGT 条目。
- **保留（不改写历史）：** §7 冻结协议、§10/§11 的全部测量数字与判定原样保留；§0–§11 是 2026-11-04 的历史记录，整体标为 superseded，由 §12 的新主线取代。
- **不变：** 不训练/微调 VGGT-Ω；不下载 DTU 或全量 ETH3D；`data/`、`outputs/` 只读。

## 0. 结论（2026-11-04，superseded by §12）

- **阶段 1 主链路通过。** VGGT-Ω 1B-512、nvdiffrast、ExMesh CUDA 扩展、Geometry Evaluation 均已在 4090 上跑通；Lego 基线在 4090 重建后 F1 = **0.6376**（4070 参考 0.6315），过 0.62 门槛。
- **4090 实测：** 16 / 32 / 64 视角推理峰值 **3.99 / 4.49 / 6.34 GB**，推理耗时 **0.97 / 1.77 / 4.10 s**（约 15.6–18.1 视角/s），64 视角远低于 24 GB。
- **M2 重复性地板已重建：** 10 次同输入重跑，Observed Chamfer = **0.009414 ± 0.000039**（CV 0.41%），F1 = **0.7365 ± 0.00127**；10 个 mesh 文件均不相同，说明地板来自真实的运行时非确定性，不是缓存复用。
- **旧 4070 地板不再沿用。** 相同输入在 4090 上重算的 VGGT 输出与旧缓存存在 bf16 非确定性差异，且 4090 上基线 F1 已变化，因此后续跨场景对比一律使用本轮 4090 地板。
- **阶段 1 尾巴已闭环：** SSH 于 `2026-11-04 22:22:53 +08:00` 恢复（`PORT_OPEN=True`、`SSH_OK`）；PyTorch3D `0.7.9` import 与 `quaternion_to_matrix` CUDA 前向通过；ExMesh `build_rotation` 前向通过，`simple_knn._C`、`diff_plane_rasterization` 均可 import。
- **基线显存补采样完成：** nvidia-smi 总占用峰值 8269 MiB（空闲 1721 MiB，含其他租户），扣除空闲后 **delta 6548 MiB**，两次重跑数值完全一致；脚本内 torch 报告峰值 3993 MiB。两者差异来自 CUDA context/权重流式加载等非张量暂存，报告以更保守的 nvidia-smi delta 作为资源上限。
- **未闭环项：** tinycudann / torch_scatter 按要求留到阶段 5；ETH3D 官方可见性协议未实现，阶段 2 的 ETH3D 数字一律标注为**内部几何评测**。
- **阶段 2 结论（2026-11-04）：** 统一点集指标器自检通过；Lego B1 过绝对地板（Chamfer 0.012655 ≤ 0.0135、F1 0.6271 ≥ 0.62）但相对改善仅 −0.84% Chamfer，低于 2% 门槛且落在 M2 噪声内；ETH3D 三场景中仅 relief 过闸（Chamfer −31.2%、F1 +0.0644），pipes 退化 +109.3%，kicker Chamfer −5.8% 但 F1 基本不变 → **1/3 改善 + 任一退化 >5%，阶段 2 整体不过闸**。阶段 3 只能以 B0 为对照做受控探索。

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

**评测器与自检（2026-11-04）：** `scripts/eval_point_metrics.py`（自动读 PLY/npy、train-only Sim(3)、双向 KD-tree 最近邻）。合成自检结果：

| 用例 | 注入 | 实测 |
|---|---|---|
| 平面整体偏移 `+0.01` | accuracy/completeness 0.01；F1@0.02=1，F1@0.005=0 | accuracy 0.010676、completeness 0.010754、Chamfer 0.010715；F1 1.0 / 0.0 |
| 已知 Sim(3)，`scale=1.3` + 17° 旋转 + 平移 | 尺度应精确恢复，Chamfer 应接近采样噪声 | scale 1.3000000000000005、Chamfer 0.002737、F1 1.0 |

自检结论：偏移量、F-score 阈值响应与 Sim(3) 恢复均通过，可作为阶段 2 统一指标器。

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

## 10. 阶段 2 执行结果（2026-11-04）

结论：**指标器自检通过；Lego B1 过绝对地板但未过相对门槛；ETH3D 三场景全部跑完，仅 relief 单场景过闸（1/3），pipes 出现 +109% 退化。阶段 2 整体不过闸，阶段 3 只能以 B0 为对照做受控探索。**

### 10.1 统一指标器

- 脚本：`scripts/eval_point_metrics.py`（自写 PLY 解析、npy/PLY GT、面积加权 50k 采样、train-only UMEYAMA Sim(3)、cKDTree 双向最近邻）。
- 合成自检（§7.4 表）通过：已知偏移 0.01 → Chamfer 0.010715；已知 Sim(3) `scale=1.3` + 17° 恢复 scale 1.3000000000000005，Chamfer 0.002737。
- 修复记录：`read_ply()` 增加 `Path | str` 归一化，ETH3D 批量拼接 GT 时不再因字符串路径失败。
- ETH3D 全部输出 `label=internal_geometry_eval`，统计口径为全量扫描点集，**无官方可见性协议，不可与官方榜单比较**。

### 10.2 逐 run 记录

batch = 1 scene/前向（VGGT 单序列）；峰值显存 = `nvidia-smi` 总占用 30 ms 轮询减空闲基线 1721 MiB；sps = 视角数 / VGGT infer 秒。

| Run | 视角 | batch | 峰值 delta (MiB) | sps | 网格 voxel / shape | Chamfer | F1@1vox | 过闸 |
|---|---|---|---|---|---|---|---|---|
| Lego B0 | 16 | 1 | 6548 | 16.49 | 0.01150144 / [127,215,134]（冻结） | 0.018928 | 0.319234 | 基线 |
| Lego B1 | 16 | 1 | 0（CPU 阶段） | — | 同上 | 0.018769 | 0.329733 | 相对：**未过**（Chamfer −0.84% < 2%） |
| pipes B0 | 12 | 1 | 6698 | 14.46 | 0.02887965 / [256,57,96] | 0.166632 | 0.146610 | 基线 |
| pipes B1 | 12 | 1 | 0（CPU 阶段） | — | 同上 | 0.348758 | 0.136827 | **退化 +109.3%** |
| relief B0 | 25 | 1 | 7288 | 17.24 | 0.04572080 / [131,187,170] | 1.085581 | 0.153491 | 基线 |
| relief B1 | 25 | 1 | 0（CPU 阶段） | — | 同上 | 0.747220 | 0.217912 | **过**（−31.2%，F1 +0.0644） |
| kicker B0 | 25 | 1 | 7288 | 17.61 | 0.01825780 / [213,60,174] | 0.443895 | 0.022397 | 基线 |
| kicker B1 | 25 | 1 | 0（CPU 阶段） | — | 同上 | 0.418255 | 0.022145 | **未过**（Chamfer −5.8% 但 F1 −0.00025） |

- 全部 run 资源门槛通过（单场景单臂远低于 30 min，峰值 delta 均 ≤ 7.3 GB ≤ 10 GB）。
- 无效 run：relief B1 第一次尝试用过滤后深度中位数推 far（5.156），导致 voxel 0.04536 / shape [129,187,171]，`GRID_MATCH` 断言失败；已删除该 arm，改用 B0 冻结的 `depth_near/depth_far` 重跑（`p2_eth3d_v3.log`）。pipes 首次运行在 GT 拼接时因字符串路径报错（已修 `read_ply`），未产生指标。
- 峰值采样：pipes 的 delta 来自同输入隔离复跑探针（`peak_vggt_eth3d_pipes.log`）；relief/kicker 为实际运行采样。

### 10.3 门槛判定（冻结规则）

| 判定项 | 结果 | 结论 |
|---|---|---|
| Lego 绝对地板（triangle 协议） | Chamfer 0.012655 ≤ 0.0135；F1 0.6271 ≥ 0.62 | 通过 |
| Lego 相对改善（point 协议） | Chamfer −0.84%（要求 ≥2%）；F1 +0.0105 | 未过 |
| ETH3D 单场景改善 | relief 过；pipes、kicker 未过 | 1/3 |
| 跨场景 | ≥2/3 ETH3D **且** Lego 改善 | 未过 |
| 退化保护 | pipes Chamfer +109.3% | 未过 |

补充说明：

1. **Lego 两个协议符号相反**（point −0.84%，triangle +0.11%），且两者都小于 M2 重复性地板（Chamfer 1σ = 0.41% 相对），因此 Lego 的 B0/B1 差异不可区分于运行时噪声；B1 的收益不成立。
2. **pipes 的 B1 是覆盖率损失**：仅保留 73.1% 像素，mesh 从 40978 顶点降到 15464，completeness 从 0.104 恶化到 0.474，不是精度提升。
3. **kicker 的 F1@1vox 退化**：50k 采样误差地板约 0.57 m ≈ 9 倍 1-voxel 阈值（0.0628 m），F1 在两臂都只有 0.022 量级，该场景的 F1 条件不具判别力；按冻结协议记为未过，未事后调参。
4. **ETH3D 绝对值只作内部排序**：relief/kicker 的 B0 误差地板为 7–24 voxel（512 分辨率 VGGT 深度 + 17–68 m 大场景），Sim(3) 相机中心 RMS 0.28%–0.74% 正常。

### 10.4 阶段 3 判断

- 阶段 3 **可以启动，但不能以 B1 作为默认起点**：B1 只在 relief 一个场景成立，且在 pipes 造成大幅退化；跨场景门槛未过。
- 建议阶段 3 的 G1/G2 以 **B0** 为对照臂，并把 relief 作为唯一已知可复现 B1 收益的验证场景；pipes/kicker 需先解决覆盖率与阈值退化问题，再讨论一致性过滤。
- 不修改已冻结的 §7 协议；若阶段 3 需要新 arm，按 §7 追加记录，不覆盖本表。

产物索引补充：

| 产物 | 本地 | 远端 |
|---|---|---|
| 阶段 2 run 记录 | `outputs/72h/phase2/run_records.json` | `/root/3dtex/outputs/72h/phase2/run_records.json` |
| ETH3D P0 网格冻结 | `outputs/72h/p0_grids.json` | `/root/3dtex/outputs/72h/p0_grids.json` |
| ETH3D B0/B1 指标 | `outputs/72h/phase2/eth3d_*/{b0,b1}/point_metrics.json` | `/root/3dtex/outputs/72h/phase2/eth3d_*/{b0,b1}/eval/point_metrics.json` |
| Lego B0/B1 指标 | `outputs/72h/phase2/lego_b*/{point,surface}_metrics.json` | `/root/3dtex/outputs/72h/phase2/lego_b*/eval*/` |
| 阶段 2 日志 | `outputs/72h/logs/p2_lego.log`、`p2_eth3d_v2.log`、`p2_eth3d_v3.log`、`peak_vggt_eth3d_*.log` | 同名 |

## 11. 阶段 3 执行结果（2026-11-04）

结论：**ETH3D 的评测口径已校准到可用（pipes/kicker 的 Chamfer 可信，1-vox F1 不可用）；kicker 的「0.57 m 采样地板」是口径误记，真实采样地板 0.0245 m，0.57 m 来自方法侧的背景壳层。G1 未超噪声且损伤锐边；G2 相对 B0 大幅改善（−21.2% Chamfer / +0.079 F1）但相对 4090 M2 地板仍差 5.9%，且 sharp recall 低于 B0；G3 代理全面退化，止损。阶段 3 不产生可替换 M2 的新地板。**

### 11.1 限时校准（≤1.5 h）：ROI / 单位 / Sim(3) / 采样地板

脚本 `scripts/diagnostics/eth3d_protocol_calibration.py`，只改评测口径（ROI、采样密度、阈值单位），不调方法参数；每个场景对 B0/B1 重评测一次。

| 场景 | GT 点数 | GT 对角线 | 相机跨度 | Sim(3) scale | 相机中心 RMS | 50k 采样地板 | 1M 采样地板 | 观测 ROI 占比 |
|---|---|---|---|---|---|---|---|---|
| kicker | 44.5 M | 15.14 m | 7.08 m | 3.4414 | 0.0233 m = 0.37 vox | **0.0245 m** | 0.00566 m | 17.96% |
| pipes | 11.5 M | 17.59 m | 2.65 m | 2.6688 | 0.0073 m = 0.095 vox | **0.0180 m** | 0.00381 m | 30.95% |

| 场景 / 臂 | 顶点 | 表面积 | Chamfer（全扫描） | Chamfer（ROI） | Accuracy | >2 vox 离群 | 结论 |
|---|---|---|---|---|---|---|---|
| kicker B0 | 64 286 | 15.38 m² | 0.43027 | 0.46524 | 0.5481 m = 8.72 vox | 89.1% | 方法问题：原始 VGGT 背景壳层 |
| kicker B1 | 45 046 | 11.39 m² | 0.40275 | 0.43720 | 0.4793 m = 7.63 vox | 88.3% | 同上，过滤仅削掉部分背景 |
| pipes B0 | 40 978 | 22.59 m² | 0.1498 | 0.1491 | 0.1991 m = 2.58 vox | 30.8% | 基线 |
| pipes B1 | 15 464 | 8.90 m² | 0.3354 | 0.3555 | 0.2065 m = 2.68 vox | 32.6% | 方法问题：覆盖率崩塌（面积 −60.6%） |

口径修正结论：

1. **§10.3 的「kicker 50k 采样地板 0.57 m」是误记。** 50k 采样的真实地板为 0.0245 m（< 1 vox），0.57 m 是网格 accuracy 均值，来自 89% 网格采样点 >2 vox 的背景壳层。kicker 的失败是方法问题，不是评测问题。
2. **pipes B1 的退化不是评测口径造成的。** Sim(3) 残差 0.0073 m、采样地板 0.018 m 均正常；B1 把表面积从 22.59 m² 砍到 8.90 m²，Chamfer(ROI) +138%，是方法侧的覆盖率损失。
3. **ETH3D 不可用 1-vox F1 做闸门。** 没有官方可见性协议，69%–82% 的扫描点不在任何训练视锥内；ROI 内 1-vox recall 仅 0.015–0.031。阶段 3 之后 ETH3D 以 **Chamfer + accuracy 离群率** 为准，F1 仅作描述。
4. ROI 定义为「至少落入一个训练相机像平面内的扫描点」∩「网格包围盒 + 3 vox」；两种 ROI 下 B0/B1 相对结论符号一致。

### 11.2 G1 / G2（Lego，B0 为对照臂，三角面协议）

冻结口径：`voxel=0.01150144`，参考 `outputs/geometry_pruning/eval_carve_v35c30_conf`；M2 地板 = Chamfer **0.009414 ± 0.000039**、F1 0.7357。batch 为每迭代参与视角数（梯度逐视角平均）；sps = 迭代/秒；峰值显存 = nvidia-smi 30 ms 轮询减空闲 1721 MiB。

| Run | 迭代/视角 | batch | 峰值 delta (MiB) | sps (it/s) | Chamfer | F1 | Recall | Sharp Recall | 法线误差 | Hole Fill | Coverage | 过闸 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0（对照） | 0 / 16 | 16 | — | — | 0.0126409 | 0.6260 | 0.7189 | 0.7166 | 51.20° | 0.9836 | 0.4306 | 基线 |
| G1 Lattice4 | 300 / 16 | 16 | 686 | 9.0 | 0.0125401 | 0.6292 | 0.6584 | 0.6224 | 50.18° | 0.9656 | 0.4718 | **未过** |
| G1 Lattice4 rep2 | 300 / 16 | 16 | 686 | 8.9 | 0.012519 | 0.6271 | 0.6561 | 0.6148 | 50.37° | 0.9659 | 0.4706 | **未过** |
| G1 Lattice8 | 300 / 16 | 16 | 684 | 9.5 | 0.012839 | 0.6118 | 0.6177 | 0.5747 | 50.30° | 0.9692 | 0.4584 | **未过** |
| G2 Lattice4+normal+prior | 300 / 16 | 16 | 768 | 4.9 | 0.009966 | 0.7050 | 0.7071 | 0.6812 | 48.82° | 0.8963 | 0.5744 | 对 B0 过；对地板不过 |
| G2 rep2 | 300 / 16 | 16 | 768 | 4.9 | 0.009982 | 0.7055 | 0.7082 | 0.6765 | 48.68° | 0.8935 | 0.5727 | 同上 |
| G3 代理（非 URGT） | 150 / 4 | 4 | 832 | 14.9 | 0.011875 | 0.5989 | 0.5705 | 0.5993 | 49.13° | 0.8722 | 0.4451 | **止损** |

- G1 Lattice4：Chamfer −0.79%、F1 +0.0032，均落在 M2 重复噪声量级（Chamfer σ 0.41% 相对、F1 σ ≤0.005），且 **Sharp Recall −0.094**；不计收益。Lattice8 全面退化。
- G2：相对 B0 **Chamfer −21.2%、F1 +0.0790、法线 −2.38°、Sharp Accuracy −0.59 vox**，重复间差 0.000016（远小于收益）；但相对 M2 地板 **Chamfer +5.86%、F1 −0.0307**，且 Sharp Recall 0.681 < B0 0.717 < 地板 0.780。即 G2 只是把 B0 拉回接近地板，未建立新地板，并触及「不得牺牲细结构」硬门槛。
- 未参与优化视角轮廓质量：本阶段 Lego 两个臂都用了全部 16 视角，无 held-out 划分，记 N/A；阶段 4 需先补 held-out 协议再逐组报告。
- G2 的 `prior` 项收敛在 ~0.93 vox（Huber 超出量），说明 VGGT 弱先验与初始网格仍存在系统性偏差，不是靠权重调平的问题。

### 11.3 G3：局部法线矫正代理止损（旧路线，已停用）

> 2026-11-05 注：本节属于已 superseded 的 VGGT 深度—法线矫正路线；相关先验路线（含 URGT/HyDen 一类）已从计划中移除，此处仅保留历史记录。

- 仓库与远端（`/root`、`/root/deps`）均无 URGT 代码或权重，且本窗口不做网络拉取；G3 的 URGT 本体记为 **BLOCKED**，不伪造结果。
- 可执行的最小代理：从 G2 网格出发，4 视角、512 分辨率、Lattice2、法线权重 0.3、弱先验，150 迭代（`lego_g3_proxy`）。
- 结果：Chamfer 0.011875（较 G2 **+19.2%**）、F1 0.5989（−0.106）、Sharp Recall 0.5993（−0.082）、法线 49.13°（+0.31°）。4 视角高分辨率局部修正以覆盖率换取局部精度，整体退化。
- **判定：止损，不接完整多视图，不接入流水线。** relief 侧 G3 同样 blocked（ETH3D 无 runner 所需的逐视角 GT depth/foreground 契约，需先补 GT 渲染）。

### 11.4 阶段 4 ExMesh E0–E3 判断

- 阶段 3 未产生优于 4090 M2 地板（0.009414）的几何臂；G2 仅证明「法线一致性 + 弱先验」能把 B0 拉近地板，并在 ETH3D 上暴露背景壳层与可见性缺口。
- 因此 **不建议按计划全量启动 E0–E3**：E0（ExMesh 资产/环境/协议自检）可启动；E1–E3 应等两个前置项完成后再开：
  1. kicker/pipes 的背景壳层抑制（confidence/free-space 约束），否则 F-score 只会在错误区域优化；
  2. ETH3D 可见性掩码协议（训练视锥 + 遮挡测试），否则跨场景闸门不可信。
- 若强行启动 E1–E3，预期只会在 Lego 上重复 0.619–0.62 量级，无法证明相对 M2 地板的增益。

产物：

| 产物 | 本地 | 远端 |
|---|---|---|
| 阶段 3 run 记录 | `outputs/72h/phase3/run_records.json` | `/root/3dtex/outputs/72h/phase3/run_records.json` |
| 校准 JSON | `outputs/72h/phase3/calibration/{kicker,pipes}.json` | `/root/3dtex/outputs/72h/phase3/calibration/` |
| Lego 三角面指标 | `outputs/72h/phase3/lego_eval*/surface_metrics.json` | `/root/3dtex/outputs/72h/phase3/lego_eval*/` |
| 日志 | `outputs/72h/logs/p3_*.log` | `/root/3dtex/outputs/72h/logs/` |
| 新增脚本 | `scripts/diagnostics/eth3d_protocol_calibration.py`、`scripts/experiments/p3_g1g2_lego.sh`、`scripts/mesh_opt/{render,inputs,geometry,runner}.py` | 同步至远端同路径 |

## 12. 新主线：image-driven surface reconstruction（2026-11-05 起）

### 12.1 立场

- **VGGT-Ω 是初始化器，不是目标。** 只提供粗相机、尺度、轮廓和粗几何；允许优化后的表面显著偏离 VGGT 的深度与法线。
- **目标是逆渲染，不是深度回归：** `M*, A, L = argmin L(Render(M, A, L), I_1..N) + λ R(M; M_VGGT)`，其中 `R` 是以 VGGT 几何为初始化的弱正则，不是逐像素深度拟合。
- **反光场景必须分解外观。** Lego 的亮斑与颜色不只由几何决定；需要 Factored-NeuS 式的材质/光照建模，否则 RGB 损失会把几何误差解释掉。
- **允许拓扑变化。** 错误三角面应能被移除并重建正确表面；只做顶点位移的显式优化不足以恢复孔洞与薄结构。

### 12.2 两阶段架构

| 阶段 | 目标 | 方法 | 输入 → 输出 | 明确不做 |
|---|---|---|---|---|
| Stage 1 表面恢复 | 从粗几何恢复准确表面，允许拓扑变化 | nvdiffrec / FlexiCubes（SDF / DMTet）与 ICLR Mesh Splatting（多层软化 + 重网格） | Ω 粗几何（TSDF / SDF）+ 原始 RGB → 准确表面 | 不做 UV；不追求少三角面；**不引入 VGGT 深度监督** |
| Stage 2 显式精修 | 高质量显式 Mesh 与资产化 | ExMesh + 自适应 remesh（细分/简化）+ 法线精修 + UV / 纹理烘焙 | Stage 1 表面 + 原始 RGB → OBJ / GLB / UV / texture | **不重新引入强 VGGT depth loss**，避免把已纠正的几何拉回去 |

Stage 1 监督组合：`L1 = λ_s L_silhouette + λ_c L_correspondence + λ_r L_render + λ_g L_geometry`。

区域降级原则：轮廓约束消除多余实体；可靠特征匹配提供三维对应；纹理充分区域用 RGB 细节；亮斑/弱纹理区依赖受约束逆渲染与几何正则；完全无信息区域只保留几何先验，不用 RGB 硬推。

### 12.3 第一阶段实验（全部在 Lego16，固定 Ω 相机与同一组 16 张 RGB）

**Step 0：优化目标有效性测试（先做，早于 A/B/C）**

目的：确认渲染损失真的偏好正确几何，而不是让外观模型把几何错误解释掉。

| 子测试 | 做法 | 通过条件（暂定，Step 0 结束后冻结） |
|---|---|---|
| 0a 外观拟合 | 分别固定 GT Mesh 与 VGGT/TSDF Mesh，只优化外观（受约束材质/光照）；同 16 训练视角，另留 4 个 heldout | 训练视角都能拟合到低损失；**heldout 上 GT-Mesh 的误差必须明显低于 VGGT Mesh**（方向正确；暂定 ≥1 dB PSNR 或等量 Chamfer 优势） |
| 0b 已知扰动 | 对 GT Mesh 施加已知扰动（沿法线 ±0.5 / 1 / 2 voxel、平滑、填洞），再从扰动网格做几何优化 | 优化方向应把扰动推回 GT：扰动区梯度与“指向 GT”方向的一致率 ≥80%，且恢复后误差单调下降 |
| 0c 反光诊断 | 在亮斑/高光区单独统计 0a、0b | 若外观模型在亮斑区给出反向梯度，该外观模型判失败，必须先上材质/光照分解 |

0a / 0b 未通过时 A/B/C 一律不启动 —— 否则会重复 P1 的「外观拟合越好、几何越错」。

**A/B/C 三方对照（固定条件：同 16 视角、同 Ω 相机且不重估位姿、同初始化来源、同评测脚本）**

| 实验 | 初始化 → 方法 | 目的 |
|---|---|---|
| A | Ω TSDF → ExMesh | 现有显式 Mesh 方法在固定 Ω 相机下的真实能力（历史 ExMesh 10k F≈0.619 作对照） |
| B | Ω TSDF → Mesh Splatting | 多层表面可微优化 + DMTet 拓扑变化，能否把 VGGT 的错误表面推正 |
| C | Ω SDF → nvdiffrec / FlexiCubes | 联合表面、材质、光照优化，能否在反光场景下分离几何与外观 |

每臂必须记录 batch（视角数）、峰值显存、sps、过闸判定与日志路径；新增 arm 只追加记录，不改历史行。

### 12.4 验收口径（沿用 §7 冻结协议，不改历史数字）

| 项 | 门槛 | 现状参考（保持不变） |
|---|---|---|
| F1 @1 voxel | **≥ 0.75** | 参考 Mesh 0.632；M2 0.7365（地板 0.737） |
| 法线平均误差 | **< 50.44°** | 4070 参考 50.44°；4090 基线 50.10°；M2/G2 48.7–48.8° |
| Sharp Recall | **≥ 0.694** | 冻结值 0.694；B0 0.7166；G2 0.6812 未达 |
| 孔洞 / 薄结构 | 可视质量确实改善（窗口孔洞、细杆、薄片） | 以往 deletion-only 路线只能开孔、不能重建薄结构 |
| Heldout 渲染 | 不退化 | 需先补 held-out 协议（当前 Lego 16 视角全部参与优化） |
| 资源 | 单臂 ≤30 min、峰值 delta ≤10 GB、记录齐全 | 沿用 §7.5 |

`F1 ≥ 0.75` 高于 M2 地板 0.7365，是**阶段性目标**，不是论文声称的效果；达不到就记录为负结果，不事后调参。

### 12.5 与历史记录的边界

- §7 冻结协议继续作为唯一评测口径；§10 / §11 的数字、判定与无效 run 记录全部保留。
- §11.2 的 G1/G2 与 §11.3 的 G3 代理属于旧的 VGGT 深度—法线矫正路线，已 superseded：其结论只用于说明「深度先验方向封死」，不再作为新主线的起点臂。
- A1/A2（以 URGT/HyDen 一类先验做深度—法线矫正）取消，不再列入候选；URGT/HyDen 不进入主线，也不进入附录。

### 12.6 Step 0 结果：优化目标有效性（2026-11-05，NO_GO）

**判定：0a 通过，0b 失败，0c 未出现系统性反向但信号接近随机。总体 NO_GO，A/B/C 全部未启动。**
依据 §12.3 的冻结口径：0b 要求方向一致率 ≥80%，实测为 52%–59%；0a 的 GT/VGGT heldout 优势为 +4.23 dB，通过 ≥1 dB 门槛。

**参照说明（代理参照，不是解析 GT 资产）**：仓库没有原始 Lego GT mesh。本测试的 “GT Mesh” 是 `data/scenes/nerf_lego_gt16/mesh/mesh_init.ply`，即由同 16 张 GT 渲染深度构建的**冻结 TSDF 网格**（140,623 顶点，sha256_16 `e4006a146e8ae5d6`），仅作为 GT 深度的代理参照；VGGT Mesh 为 `outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply`（202,660 顶点，sha256_16 `ecf994efd40a5d9c`）。两者均在 source/GT gauge，两个臂都使用同一组 GT train 相机；heldout 相机与图像从未参与重建或对齐。

**运行记录**（脚本 `scripts/experiments/step0_objective.py`；输出 `outputs/step0_20261105_r2/`；日志 `logs/step0_20261105_r2.log`；GPU 空闲基线 2309 MiB，含其他租户）：

| run | 训练/heldout 视角 | 分辨率 | 迭代 | 初始化 | train L1 | heldout PSNR / L1-core | 峰值显存（torch / smi delta） | 耗时 |
|---|---|---|---|---|---|---|---|---|
| 0a GT | 16 / 4（全部 16） | 256 | 800（3000 上限，早停收敛） | albedo 0.5、SH DC 1/Y00、spec 0.1 | 0.03121 | **19.861 dB / 0.04987** | 155.7 MiB / 98 MiB | 186.1 s |
| 0a VGGT | 16 / 4（全部 16） | 256 | 800（3000 上限，早停收敛） | 同上 | 0.04882 | **15.632 dB / 0.08451** | 201.9 MiB / 72 MiB | 188.2 s |

0a 全部 16 个 heldout 的平均同样成立（GT 19.551 dB / VGGT 15.435 dB）。silhouette IoU 为 0.9721 与 0.9462，说明 VGGT Mesh 的轮廓更差；训练视角上 GT 也更好（0.0312 vs 0.0488），因此这不是“GT 只赢在 heldout”的过拟合假象。外观表示为 per-vertex albedo + degree-2 SH 漫反射 irradiance + degree-2 SH 镜面 envmap + 标量 spec_gain，λ_albedo = 0.05（该权重在 pilot 中选定，存在轻微 heldout 选择偏差）。

**0b 已知扰动（冻结外观，只回传几何梯度）**：

| 扰动 | GT 拉回 base 的方向一致率 | GT 到 base 的平均位移（voxel） | VGGT 指向 GT 代理的方向一致率 | VGGT 梯度在 GT 方向上的投影占比 |
|---|---|---|---|---|
| ±0.5 voxel | 0.592 / 0.591 | 0.918 → 0.842 | 0.535 / 0.524 | 1.09% / 1.06% |
| ±1 voxel | 0.561 / 0.567 | 0.921 → 0.853 | 0.527 / 0.522 | 1.14% / 1.07% |
| ±2 voxel | 0.531 / 0.525 | — | 0.523 / 0.519 | 1.39% / 1.20% |

- 所有档位的方向一致率在 **51.9%–59.2%**，远低于 80%；曲率较高的顶点子集更差（0.505–0.586）。
- VGGT 的梯度几乎与“指向 GT 表面”的方向正交（投影占比约 **1%**）；GT 臂虽沿法线方向有梯度（投影占比约 65%），但正负号接近对半（cosine mean 0.037–0.136）。
- 恢复曲线不是单调下降：GT δ=1 从 0.918 降到 0.842（第 25 步）后又回升到 0.848；VGGT δ=1 负向甚至从 0.837 降到 0.808 后回升到 0.859，最终误差大于起点（`base_error_decrease_voxels = -0.021`）。
- 扰动后图像损失随 δ 单调升高（GT 0.0532 → 0.1233；VGGT 0.0846 → 0.1389），说明损失**能察觉**扰动，但梯度方向不足以把表面推回；per-vertex albedo 可以在平滑区解释掉几何误差。

**0c 亮斑/高光诊断**（前景亮度 ≥ max(0.60, 每视角 p90)；严格高光另按 maxRGB≥0.95 & Y≥0.80 & minRGB≥0.55）：

| 臂 | 高光区 cosine mean | 高光区方向一致率 |
|---|---|---|
| GT | +0.0365 | 0.5238 |
| VGGT | +0.0380 | 0.5267 |

高光区没有出现系统性反向梯度，因此**按 0c 的判据外观模型不算失败**；但一致率仅略高于 50%，不能据此认为外观模型已具备可用的几何分辨力。

**结论与建议改动**（不启动 A/B/C）：

1. **失败点**：0b 的“均匀法线偏移”在平滑/弱纹理区是孔径问题，沿表面切向的滑动不改变图像；per-vertex albedo 与 SH 光照共同提供了足够自由度吸收残余。16 个近正面视角进一步放大了这一退化。
2. **先改测试再做 A/B/C**：(a) 扰动改为**局部高斯凸起/凹坑**，并限制在 silhouette、深度不连续与高曲率区域（可见 ≥3 视角）；(b) 方向统计只在该局部 patch 与多视角重叠区计算；(c) 把外观自由度收窄（低秩 albedo / 平滑先验 / 固定 albedo 只优化光照）以降低“几何误差被外观吸收”的能力。
3. **若收紧后仍 <80%**：说明单靠 RGB 渲染损失在 Lego16 上不足以驱动几何，Stage 1 必须同时引入 silhouette/correspondence 项（§12.2 已有 `λ_s`、`λ_c`），且不应把 0b 的 80% 门槛作为唯一过闸条件；该负结果应保留，不事后调参。
4. **无效 run 记录**：第一轮 `outputs/step0_20261105/` 的 0a 有效（GT/VGGT heldout 19.855/15.633 dB），但其 12 个 0b run 全部因 `curvature_mask` 的面法线与顶点法线平均形状错误（`RuntimeError: size of tensor a (251156) must match tensor b (3)`）而无效；修复为对每个面的 3 个顶点法线取均值后，`_r2` 的 12 个 0b run 全部有效。冒烟目录 `outputs/step0_smoke_fix/` 仅用于验证修复，不作为结果。

**资源**：三个子测试合计约 7–8 min；单 run 最长 188.2 s；峰值 delta 最大 98 MiB，远低于 §12.4 的 10 GB 上限。所有长任务在远端 tmux 中运行并落盘。

### 12.7 Step 0.1 诊断：局部形变、梯度校验与监督消融（2026-11-05）

**定位**：§12.6 的 80% NO_GO 结论与全部数字原样保留，本节是**新增的 Step 0.1**，不覆盖、不改写旧结论。Step 0.1 放弃“用局部测试作总闸门”的方案 1，改用局部标量形变参数与四级验收阶梯。

**参照口径**：仍使用 §12.6 的两个网格与同一组 16 张 GT train 视图（256×256，voxel `0.01150144350298549`）。GT Mesh 是 GT 深度的冻结 TSDF 代理（sha256_16 `e4006a146e8ae5d6`），不是解析 GT 资产；VGGT Mesh sha256_16 `ecf994efd40a5d9c`。VGGT 的方向判定改用 **GT 代理表面的符号距离**（最近 GT 表面点 + 该点法线投影），不再使用最近顶点。

#### 12.7.1 代码修正（先修再做）

1. **判据改为局部形变参数 a**。`V_i(a) = V_i^0 + a w_i n_i`，`w_i` 为 σ=4 voxel 的高斯 bump，做 `∂L/∂a` 是否指向 `a=0` 的检验，替换逐顶点方向一致率（相邻顶点对同一三角面的影响高耦合，不适合判断整体可行性）。
2. **AD 与中心差分校验**。`h=0.2 voxel`，先做该项；未通过时只查可微光栅化、遮挡切换与梯度实现，不加几何正则。
3. **`curvature_mask()` 归一化修正**。把原来“相邻面法线差累加后统一除 3”改为按每个顶点实际关联面数归一化（`crease / count`），并重算高曲率子集。受影响范围：§12.6 的“高曲率子集更差（0.505–0.586）”一句基于旧归一化，只能作为旧口径历史记录；Step 0.1 的 flat 种子改用修正后的低曲率 30% 分位。
4. **`photometric_loss()` 补对称 silhouette 项**。新增 `silhouette_loss = |alpha - GT mask|`（全图，含 Mesh 溢出 GT 轮廓的外部像素）与 `silhouette_weight` 参数；Step 0.1 的 combo = 前景 RGB + 0.5×对称 silhouette。§12.6 的 Step 0 结果来自**前景 RGB 旧版损失**，两版损失不可混用比较。
5. **对应关系口径**：Lego 纹理稀疏，SIFT 中位匹配数 GT 139.5 / VGGT 90.5，覆盖率（8×8 网格）0.398 / 0.344；因此不构造图像特征密集对应。Step 0.1 的 `corr` 项是**代理对应**：把 bump 支撑集投影回 bump 前的 base 表面，明确不是图像特征匹配，只作为几何控制的对照。

#### 12.7.2 必须写明的历史解释更正

- **更正 A**：§12.6 用“per-vertex albedo 与 SH 光照提供自由度吸收残余”解释 0b 失败是**错误归因**。0b 期间外观参数是冻结的（只回传几何梯度），albedo 不可能吸收几何误差；0a 的高自由度外观只是**可能**削弱可辨识性，与 0b 的结论是两件事。
- **更正 B**：§12.6 的 VGGT→GT 方向使用 `cKDTree` 最近**顶点**，两网格采样与拓扑不同，因此“梯度在 GT 方向投影仅 1%”不可靠；该数字不得作为“梯度与 GT 表面正交”的证据。Step 0.1 改用最近表面点符号距离，P2 的 `toward_proxy_ratio = 0.9167` 才是可用于判断的数字。
- **更正 C**：0b 测的是沿顶点法线位移，不是切向滑动；不能把失败主因归到“切向孔径歧义”。0b 只能确认**几何梯度不可辨识**。
- **更正 D**：0c 只能说明“高光区未观察到明显整体反向偏置”，不能证明镜面/材质建模正确。

#### 12.7.3 P1 局部形变：哪些区域真的存在几何信息

combo 监督、6 cases/格（3 seeds × ±1 方向）；`correct` 指 AD 指向 a=0，`toward_proxy` 指该方向在 GT 代理符号距离下也朝表面收敛：

| mesh | 区域 | δ=0.5 correct / toward | δ=1 correct / toward |
|---|---|---|---|
| GT 代理 | silhouette | 1.000 / 1.000 | 1.000 / 1.000 |
| GT 代理 | inner depth edge | 1.000 / 1.000 | 1.000 / 1.000 |
| GT 代理 | flat | 1.000 / 1.000 | 0.833 / 0.833 |
| VGGT | silhouette | 1.000 / 0.833 | 1.000 / 1.000 |
| VGGT | inner depth edge | 1.000 / **0.667** | 1.000 / **0.667** |
| VGGT | flat | 1.000 / 0.833 | 1.000 / 1.000 |

结论：三个区域都存在可下降的几何梯度，轮廓区最稳定；GT 代理上全部指向正确表面。VGGT 的内轮廓/深度边缘有 2/6 个种子的局部损失极小点**不在 GT 表面上**（toward 0.667），说明 VGGT 表面的局部最优与真实表面仍有偏差。该差异不构成“梯度错误”，但说明 VGGT 起始几何的可恢复性受限。

#### 12.7.4 P2 梯度校验（最先做，Level 1）

12 cases（两 mesh × 三区域 × ±1 方向，combo，`h=0.2 voxel`）：

| 指标 | 数值 |
|---|---|
| AD 指向 a=0 的比例 | **1.000** |
| AD 与中心差分符号一致率 | **1.000** |
| 相对误差中位数 | **0.4785** |
| 指向 GT 代理的比例 | **0.9167** |
| 扰动后损失上升比例 | 0.9167 |

**Level 1 梯度正确性通过**。相对误差中位数 0.478 仍偏高，原因是 silhouette 项在 bump 支撑外的贡献接近噪声级；符号一致率为 1.0，说明方向没有实现错误，因此不需要通过加几何正则来“修方向”。

#### 12.7.5 P3 外观自由度是否削弱几何信息

1 个内轮廓种子 × 30 步 Adam（lr 0.01），损失口径统一为**同一 RGB 前景损失**的适应前/后：

| mesh | 模式 | RGB 适应前 | RGB 适应后 | ∂L/∂a (a=+1) | 方向正确 |
|---|---|---|---|---|---|
| GT | frozen | 0.031463 | 0.031463 | 3.12e-5 | 是 |
| GT | SH-only | 0.031463 | 0.031487 | 3.13e-5 | 是 |
| GT | low-rank albedo + SH | 0.031463 | 0.031792 | 3.10e-5 | 是 |
| VGGT | frozen | 0.048928 | 0.048928 | 6.07e-5 | 是 |
| VGGT | SH-only | 0.048928 | 0.048978 | 6.13e-5 | 是 |
| VGGT | low-rank albedo + SH | 0.048928 | 0.049050 | 6.28e-5 | 是 |

结论：在 ±1 voxel 的局部 bump 上，30 步外观适应**没有降低** RGB 损失（略升 ≤1%），几何梯度大小变化 ≤3%，三模式全部指向 a=0。**本测试未观察到“材质模型吸收几何误差”**。限制：只测了一个种子、30 步、单一低秩基底，不能证明任意外观模型在所有尺度上都不吸收；0a 的结论也不因此被推翻。

#### 12.7.6 P4 监督消融：每种监督各自解决什么

24 cases/种（两 mesh × 三区域 × 2 seeds × ±1 方向，δ=1）：

| 监督 | correct | FD 符号一致 | toward proxy | 相对误差中位数 |
|---|---|---|---|---|
| RGB（前景） | 1.000 | 1.000 | 0.958 | 0.399 |
| Silhouette（对称） | 0.792 | 0.958 | 0.833 | 0.762 |
| RGB + 0.5×Sil | 1.000 | 1.000 | 0.958 | 0.529 |
| Correspondence（代理） | 1.000 | 1.000 | 0.958 | 2.3e-5 |

区域边界：GT 轮廓种子 4/4、GT 内轮廓 sil 3/4、GT 平坦区 sil 3/4、VGGT 轮廓 sil **2/4**、VGGT 内轮廓 sil 3/4、VGGT 平坦区 sil 4/4。Silhouette 只能约束**可见轮廓**，对轮廓以外/内部 bump 的判别力有限；Correspondence 只能约束**可靠匹配区**，这里的匹配区就是 bump 支撑集本身，因此它接近 100% 是构造决定的，不能当作“图像对应可解”的证据。

#### 12.7.7 P5 局部恢复：几何能否被持续推回

50 步 Adam（lr 0.03，a 从 +1.0 voxel 起），2 个轮廓种子/mesh：

| mesh | 监督 | 最终 a（voxel） | 收敛比例 |
|---|---|---|---|
| GT | combo | **0.0158 / 0.0540** | **0.984 / 0.946** |
| GT | RGB | -0.0554 / 0.0428 | 0.945 / 0.957 |
| GT | silhouette | 0.3031 / 0.3908 | 0.697 / 0.609 |
| GT | corr（代理） | -0.0232 / -0.0229 | 0.977 |
| VGGT | combo | **-0.2086 / -0.0088** | 0.791 / 0.991 |
| VGGT | RGB | -0.1979 / 0.0226 | 0.802 / 0.977 |
| VGGT | silhouette | -0.5746 / -0.3807 | 0.425 / 0.619 |
| VGGT | corr（代理） | -0.0230 / -0.0229 | 0.977 |

聚合中位数：combo 0.0349 voxel / 0.965；RGB 0.0464 / 0.954；corr 0.0229 / 0.977；silhouette 0.3857 / 0.614。

结论：在**已知 GT 代理目标**上，完整 combo 监督把 1 voxel 的合成凸起恢复到 0.035 voxel（96.5%），几何可持续下降；silhouette 单项只能恢复到 0.35–0.39 voxel，符合“只约束可见轮廓”的边界。VGGT 的 combo 在 2 个种子中有 1 个过冲到 -0.209 voxel（越过起点 0.2 voxel），说明监督能把几何移动，但 VGGT 起始表面本身不是正确表面，不能据此判定收敛到真值。

#### 12.7.8 四级闸门判定

| 级别 | 判定 | 依据 |
|---|---|---|
| Level 1 梯度正确性 | **PASS** | AD/FD 符号一致 1.000，相对误差中位 0.4785，指向代理 0.9167 |
| Level 2 监督有效性 | **PASS** | GT 代理上 combo 恢复 96.5%、终值 0.0349 voxel（阈值 ≥80%、≤0.2） |
| Level 3 真实重建有效性 | **not_run** | 需要完整方法从 VGGT Mesh 出发并带正则/表面参数化；P5 的 VGGT 局部恢复目标未知且出现 0.2 voxel 过冲，不能替代该闸门 |
| Level 4 最终任务有效性 | **not_run** | 依赖 Level 3 的新 Mesh |

silhouette 单项不达标（终值 0.347 voxel）作为**覆盖边界的诊断**保留，不再作为 Level 2 的否决条件；Level 2 的判据是“完整监督能降低已知几何误差”。

#### 12.7.9 Baseline 选择（只做一个，不开 A/B/C）

选择 **nvdiffrec（DMTet/FlexiCubes + nvdiffrast）**。理由：仓库已有 nvdiffrast 光栅化与 SH/albedo 分解可直接复用，而它是两个候选里唯一自带表面参数化与几何正则（可微拓扑、Laplacian、法线一致性）的方法，正好补上 Step 0.1 暴露的“缺少表面参数化/正则”缺口；Mesh Splatting 的 splat 管线与现有渲染栈不兼容，需要重写，风险更高。只做一个小规模对照，不并行开多项。

#### 12.7.10 运行记录与资源

- 输出：`outputs/step0_1_20261105_final2/`（`summary.json` sha256_16 `34ceb8304b1d74ec`）；日志 `logs/step0_1_20261105_final2.log`。
- 规模：202 个有效 run（P1 72、P2 12、P3 6、P4 96、P5 16），**无效 run 0**；视图 16、分辨率 256；P1/P2/P4 每 case 1 次 AD + 2 次中心差分（各 16 视角），P3 30 步，P5 50 步。
- 资源：开跑前 GPU 2235 / 24564 MiB（另一条纹理基线不在本容器可见进程内，未杀任何进程）；torch 峰值 856.8 MiB；202 run 计算耗时合计 144.6 s，端到端墙钟 2 min 33 s（05:46:20 → 05:48:53）。
- 过程 run（不作为结论）：`outputs/step0_1_20261105_r2/` 的 P3 为旧损失口径（适应前 RGB / 适应后 combo，不可比）、P4 只覆盖轮廓种子；`outputs/step0_1_20261105_final/` 的 P3 low-rank 模式对 albedo 做了 `[0,1]` clamp，使适应前 RGB 基线偏移 0.001（0.03248 vs 0.03146），三模式不可比。二者均已由 `final2` 替代；其余 P1/P2/P4/P5 数字在两版中一致。

### 12.8 Level 3：真实重建有效性（nvdiffrec / FlexiCubes，2026-11-05）

#### 12.8.1 设置与口径

- **起点**：`outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply`（VGGT Mesh，202,660 顶点 / 348,875 面，sha256_16 `ecf994efd40a5d9c`）。
- **相机**：`data/scenes/nerf_lego_gt16/geometry/gt/cameras.json` 的 16 张 GT 相机，位姿与内参全程固定，**不重估位姿**。
- **监督**：前景 RGB L1 + 0.5×对称 silhouette `|alpha - GT mask|` + 可靠 correspondence + FlexiCubes `L_dev`。外观为 64³ 可学习 RGB 网格（在 dual 顶点双线性采样）+ 2 阶 SH diffuse/specular + `spec_gain`。
- **对应关系**：SIFT（ratio 0.75，相邻/隔帧 29 对，1,281 原始匹配）→ `cv2.triangulatePoints`（GT K/T）→ 重投影与多视图一致性过滤；217 点、16/16 视图 ≥20 点、2 voxel 内点率 0.770。
- **实际资源**：分辨率 256、grid 160、100 迭代（`--iters 100`）；256 grid 另做一臂（见 §12.8.7）。
- **对照**：起点 VGGT Mesh 的原始指标直接复用 Level 2 已记录的数字（§12.4 「参考 Mesh」与 `outputs/geometry_pruning/eval_carve_v35c30_conf/surface_metrics.json`），**不重算、不改写**。

#### 12.8.2 判定：**Level 3 NOT PASS**

| 指标 | 基线（VGGT Mesh） | full_sA | full_sB | 两 seed 重复区间 | 与基线差 |
|---|---|---|---|---|---|
| F@1voxel | **0.6315** | 0.4915 | 0.4943 | 0.4929 ± 0.0014 | **−0.1386（≈99× 半区间）** |
| Chamfer (m) | **0.012580** | 0.026650 | 0.026158 | 0.026404 ± 0.000246 | **+0.013824（2.10×）** |
| 法线角 mean / median (°) | **50.44 / 52.98** | 53.61 / 57.97 | 53.76 / 58.14 | 53.69 ± 0.08 / 58.06 ± 0.09 | **+3.25°** |
| Sharp Recall | **0.6935** | 0.8231 | 0.8191 | 0.8211 ± 0.0020 | +0.1276 |
| Sharp Precision | **0.3395** | 0.1761 | 0.1743 | 0.1752 ± 0.0009 | **−0.1643** |
| accuracy mean（voxel） | **1.350** | 3.990 | 3.905 | 3.948 ± 0.043 | +2.60 |
| observed completeness（voxel） | 0.837 | 0.644 | 0.644 | 0.644 | −0.193 |
| 锐边区 accuracy（voxel） | **2.070** | 4.952 | 4.818 | 4.885 ± 0.067 | **+2.82** |
| burr rate | **0.0162** | 0.6603 | 0.6564 | 0.658 | **+0.642** |
| 边界边 / 连通分量 | 55,397 / 3,781 | 12 / 5 | 22 / 5 | — | 闭合/合并 |
| 表面积（m²） | 30.69 | 90.95 | 92.16 | — | **≈3.0×** |
| 面数 | 348,875 | 618,374 | 624,912 | — | 1.8× |

**判定依据**：F@1voxel 与 Chamfer 的退化幅度分别是两 seed 重复半区间的 99 倍和 56 倍，远超噪声门限；改善方向完全缺失，因此不是“在噪声内无法判定”，而是**确定的退化**。按 §12.4 的门槛：`F1 ≥ 0.75` 不满足（0.493）、`normal ≤ 50.44°` 不满足（53.69°）、`sharp_recall ≥ 0.694` 数值上满足（0.8211）但锐边精度/毛刺同时崩塌，触发“锐边退化即不通过”的显式条件。

#### 12.8.3 重复性噪声门限（2 seeds）

同一配置仅换 seed（20261105 / 20261106），每个 run 100 迭代：

| 指标 | sA | sB | 范围 | 半区间 |
|---|---|---|---|---|
| F@1voxel | 0.4915 | 0.4943 | 0.0028 | ±0.0014 |
| Chamfer (m) | 0.026650 | 0.026158 | 0.000492 | ±0.000246 |
| 法线角 mean (°) | 53.61 | 53.76 | 0.15 | ±0.075 |
| 法线角 median (°) | 57.97 | 58.14 | 0.17 | ±0.085 |
| Sharp Recall | 0.8231 | 0.8191 | 0.0040 | ±0.0020 |
| Sharp Precision | 0.1761 | 0.1743 | 0.0018 | ±0.0009 |
| 锐边区 accuracy（voxel） | 4.952 | 4.818 | 0.134 | ±0.067 |
| 面数 | 618,374 | 624,912 | 6,538 | ±3,269 |

两 seed 的波动很小（F1 半区间 0.0014），**任何小于 0.003 的 F1 变化都不能称为改善**。本实验所有需要“改善”的结论都按该门限判定；`corr` 消融的相对差 0.0023 就落在这个门限内（见 §12.8.6）。

#### 12.8.4 锐边结论：**退化（显式否决条件触发）**

- Sharp Recall 0.6935 → 0.8211（+0.13），但 Sharp Precision 0.3395 → 0.1752（**几乎减半**），burr 0.0162 → 0.658（**约 40 倍**），锐边区 accuracy 2.07 → 4.89 voxel。
- 解释：优化后的网格把 GT 的锐边“覆盖到了”（recall 上升来自面积极度膨胀与孔洞闭合），但产出的是大量**伪锐边/毛刺**，真正的锐边位置误差反而更大。这在 §12.4 的显式条件（“锐边退化即不通过”）下**单独构成不通过**。
- **冲突说明**：几何上确有局部改善——observed completeness 0.837 → 0.644 voxel、边界边 55,397 → 12、连通分量 3,781 → 5（孔洞与碎片被闭合）；同时 accuracy 1.35 → 3.95 voxel、Chamfer 2.1×、sharp precision 减半。即“补洞/连通性改善”与“锐边与位置精度退化”同时成立，二者不能相加为“净改善”。按用户给出的判定顺序，**锐边退化优先，Level 3 不通过**。

#### 12.8.5 初始化单独就已经退化（关键归因证据）

`init_sA` 是**同一 SDF 初始化只做 1 步、且关闭 silhouette 与 correspondence**的提取结果（等价于“把 VGGT Mesh 转成 FlexiCubes 的 SDF 再提取”）：

| 指标 | 基线 | init_sA（无监督，1 步） | full_sA（100 步） |
|---|---|---|---|
| F@1voxel | 0.6315 | **0.4821** | 0.4915 |
| Chamfer (m) | 0.012580 | 0.026159 | 0.026650 |
| accuracy（voxel） | 1.350 | 3.738 | 3.990 |
| 表面积 | 30.69 | 62.82 | 90.95 |
| 边界边 / 分量 | 55,397 / 3,781 | 136 / 10 | 12 / 5 |

初始化提取**单独**就吃掉了几乎全部指标（F1 −0.149、Chamfer 2.08×、面积 2.05×），100 步优化在其上只再变化 F1 +0.0094/+0.0122。原因是 VGGT Mesh 非水密（55,397 条边界边、3,781 个分量），最近表面符号场在远处不可靠，被 `±3 cell` 截断后形成额外等值面，提取出的表面本身就有约 2 倍面积。

#### 12.8.6 监督各项的实际贡献

| 对比 | F@1voxel | Chamfer (m) | accuracy（voxel） | 结论 |
|---|---|---|---|---|
| full_sA（RGB + 0.5 sil + corr） | 0.4915 | 0.026650 | 3.990 | 参照 |
| nocorr_sA（去 correspondence） | 0.4892 | 0.026714 | 3.991 | **差 0.0023 < 两 seed 范围 0.0028，无可测贡献** |
| nosil_sA（去 silhouette） | 0.3822 | 0.047994 | 7.669 | **silhouette 是主要支撑**（F1 −0.109，accuracy 近 2×） |
| ldev_sA（`L_dev` 权重 0.1 → 1.0） | 0.5084 | 0.025996 | 3.858 | 正则只能减缓膨胀（面积 79.7 vs 90.9），仍远低于基线 |
| g256_sA（grid 160 → 256） | 0.4304 | 0.032163 | 5.099 | 更细网格**没有**修复问题，反而更差、面数 2.64M |

- **correspondence 处于弱激活状态**：217 个点，`surface_coverage_2vox = 0.0122`、`surface_coverage_1vox = 0.0018`（即 1.2% 的网格表面在 2 voxel 内有对应点）。该项在本场景**不足以承担几何约束**，这与 §12.7.5 记录的“Lego 纹理稀疏”一致，不能据此说明“图像对应无用”，只能说明本场景下不可用。
- **silhouette 的作用方向是“膨胀”**：去掉 silhouette 后 accuracy 7.67 voxel、silhouette IoU 0.641；加上后回到 3.99 voxel / 0.927。它压住了外溢，但没有把表面拉回真实位置。
- **RGB 项在像素级被外观网格吸收**：16 视图 + 64³ 颜色网格可以把前景涂满，因此 RGB 不能提供足够的几何约束；这与 §12.7.5 P3 的“外观适应不削弱局部梯度”并不矛盾——P3 测的是 ±1 voxel 局部扰动，这里是整面自由形变。

#### 12.8.7 资源开销

| run | 迭代 | 秒 | torch 峰值 (MiB) | SMI delta (MiB) | 面数 | heldout primary PSNR / IoU |
|---|---|---|---|---|---|---|
| full_sA | 100 | 109.0 | 3625 | 5743 | 618,374 | 17.00 / 0.930 |
| full_sB | 100 | 108.8 | 3630 | 5743 | 624,912 | 16.81 / 0.928 |
| nocorr_sA | 100 | 100.5 | 3618 | 5743 | 615,578 | 16.92 / 0.932 |
| nosil_sA | 100 | 113.0 | 3864 | 5751 | 850,274 | 14.84 / 0.669 |
| init_sA | 1 | 84.4 | 2889 | 5089 | 444,382 | 11.94 / 0.933（外观仅 1 步，不可比） |
| g256_sA | 100 | 387.4 | 13269 | 18158 | 2,642,494 | 15.23 / 0.868 |
| ldev_sA | 100 | 108.4 | 3558 | 5414 | 543,916 | 17.42 / 0.938 |

- 4 个正式 run 端到端 06:18:30 → 06:25:51（7 min 21 s）；开跑前 GPU 已用 2,671 MiB（他人进程 PID 106129/573758，**未杀**），全程未超过 `24,564 MiB` 的 75%。
- 几何评价用 `/root/deps/blender/blender-3.3.9-linux-x64/blender`：默认的 `/usr/local/bin/blender` 是 2.93，无法打开 3.0 写出的 `lego_source.blend`（`File written by a newer Blender`），这是工具链限制，不影响指标口径。
- **资源不是本次失败的原因**：256 grid 只用了 13.3 GB torch 峰值（18.2 GB SMI delta）就跑通了，且结果更差；100 迭代的算力开销约 110 s/run，远低于 §12.4 的 30 min 上限。

#### 12.8.8 无效 run（保留日志，不丢弃）

| run | 现象 | 原因 | 处理 |
|---|---|---|---|
| full_sA（首次） | 第 2 次 backward `CUDA device-side assert` | 缓存的 correspondence 近邻索引在 FlexiCubes 顶点数变化后越界 | 改为每轮重建 KDTree 关联；3 迭代验证通过后重跑 |
| tsdf_sA（第 1 次） | `index out of bounds` 断言 | SDF 长度 `res³` 与 FlexiCubes 顶点 `(res+1)³` 不一致 | 改为在顶点处插值；日志 `outputs/step_baseline_20261105/invalid/tsdf_sA_attempt1_sdf_shape_mismatch.log` |
| tsdf_init_sA | 提取面 139.3 m²（输入 30.7），heldout IoU 0.366 | TSDF 融合在开放边界/无观测区按最近已知体素填符号，产生额外薄片 | 判为**初始化实现无效**，保留 |
| tsdf_sA（第 2 次） | 面 228.9 m²，IoU 0.373 | 同上 | 判为无效，保留 |
| depthinit_sA | `inside_fraction = 0.459`、IoU 0.425 | 渲染深度符号投票在轮廓外无法投票，自由空间被判为内部 | 判为**初始化实现无效**，保留 |

有效 run 7 个（init_sA、full_sA、full_sB、nocorr_sA、nosil_sA、g256_sA、ldev_sA）。

#### 12.8.9 归因：实现 / 方法 / 资源

- **实现（主因）**：SDF 初始化不成立。非水密网格的符号场不可靠，三条独立初始化路线（最近表面 + visual hull、TSDF 融合、渲染深度投票）全部失败，提取面面积为输入的 2.0–4.5 倍。仅此一项就足以让 F1 从 0.6315 掉到 0.48。这属于**实现/工具链限制**，不是方法本身的判定。
- **方法（次因，方向性证据）**：在自身初始化上，优化把 F1 从 0.4821 提升到 0.4915/0.4943（+0.0094/+0.0122，约 7–9× 噪声半区间），说明监督确实能推几何；但同一优化把 accuracy 从 3.74 推到 3.99 voxel、面积从 62.8 涨到 90.9，即**RGB + 对称 silhouette 的目标偏好“胖表面”**。`L_dev` ×10 把面积压回 79.7、F1 升到 0.5084，仍只有基线的 80%。因此方法层面存在“目标函数对膨胀不敏感、correspondence 又太稀疏无法钉住表面”的结构性问题，但**在本实验里无法与初始化缺陷分离**，不能宣判方法死刑。
- **资源（非瓶颈）**：grid 256、100 迭代均可跑；时间/显存都没有挡住实验。

**结论**：Level 3 **不通过**；主要原因是实现（初始化），方法层面只得到“目标偏好膨胀、correspondence 弱激活”的方向性证据，需先修初始化才能对方法本身下结论。按 §12.4/§12.7.8 的阶梯，Level 4 不启动。

#### 12.8.10 与 Level 4 的衔接（heldout 渲染）

| run | heldout primary（4 视图） | heldout all（16 视图） |
|---|---|---|
| full_sA | PSNR 16.999，L1 0.08729，silhouette IoU 0.92999 | PSNR 17.241，L1 0.08443，IoU 0.91895 |
| full_sB | PSNR 16.814，L1 0.08935，IoU 0.92827 | — |
| ldev_sA | PSNR 17.425，L1 0.08155，IoU 0.93750 | — |

这些数字**包含训练后的外观**，Level 2 只有几何指标、没有渲染基线，因此不能与历史数字比较，只作为 Level 4 的参考起点。注意：silhouette IoU 高达 0.93–0.94，而几何精度仅 4 voxel 偏差——说明该渲染指标不能替代几何闸门。

### 12.9 初始化专项：代理 Mesh 0/100 步对照（2026-11-05）

目标：在**不改初始化路线、不换方法**的前提下，把 0 步（纯初始化提取）与 100 步（完整优化）分开测，用“相对输入质量的保留率”判断上一轮 Level 3 的失败到底落在哪一环。

§12.8 的全部结论与数字保持原样，本节另记。

#### 12.9.1 口径修正：旧的 `init_sA` 作废

旧 `init_sA` = SDF 初始化 + **1 步 Adam + 随机扰动**（`sdf ±0.15 cell`、`weights ±0.02`、`deform ±0.02`），还带一个 RGB 步，**不是纯初始化**。它的数字（F1 0.4821、面积 62.82，见 §12.8.5/§12.8.7）保留为历史记录，但不再被当作“0 步”。

新口径 `--extract-only`：不扰动、不建优化器、不进训练循环、不做渲染评价，直接 `FlexiCubes(training=False)` 导出初始化等值面（`iters = 0`，`mode = extract_only`）。

#### 12.9.2 三处实现修复（`scripts/experiments/step3_flexicubes.py`）

1. **`--extract-only`**：如上；summary 显式写 `mode`/`iters`，外观与 heldout 渲染不参与（未训练的颜色场没有意义）。
2. **符号场**：`ProxySurface.signed()` 退出本线。新 `TriangleSDF` 的**幅度**是点到最近三角面的距离（KDTree 在三角形重心上取 k=8 短名单，再做 Ericson 精确点-三角最近点），**符号**取最近三角面的朝外法线点积。最近点落在**开放边**（只被一个三角形使用的边）上时，该体素标为不可靠；不可靠或“观察到背景”的体素一律取外部，其余保留三角面法线符号。**不做任何水密化**。
3. **可见性三态**：`visual_hull_inside()` 拆成 `carved` / `observed` / `unknown`。落入图像且投到背景 = 该相机确实看见空空间，可雕刻；落入图像且投到前景 = 可能是遮挡，不删；不在视锥 = **未观察**，保留法线符号、不删除。`unknown_fraction` 与 `carved_fraction` 写入 summary。

已知失效情形（写清，不掩盖）：开放边界处的最近点落边 → 不可靠 → 强制外部，会在轮廓/开口附近侵蚀内部，并迫使 FlexiCubes 在开口处补一圈闭合“墙”；两张输入都非水密（VGGT 55,397 边界边 / 3,781 分量，代理 30,964 / 1,152），因此该失效在两行都会发生。量化见 §12.9.6。

#### 12.9.3 输入基线与 2×2 对照（同相机、同分辨率 256、grid 160、同一 evaluator）

输入基线（同一 `mesh_opt/evaluate.py` + Blender 3.3.9，冻结 P0 协议）：

| 输入 | F@1voxel | Chamfer (m) | 法线 mean (°) | Sharp P / R | 表面积 (m²) | 分量 | 边界边 |
|---|---|---|---|---|---|---|---|
| VGGT Mesh | 0.6315 | 0.012580 | 50.44 | 0.3395 / 0.6935 | 30.69 | 3,781 | 55,397 |
| GT-depth 代理 Mesh | 0.9358 | 0.003859 | 38.94 | 0.7688 / 0.9404 | 22.85 | 1,152 | 30,964 |

0 步与 100 步（每格均为新跑；VGGT 100 步与代理 100 步用同一份新初始化代码，避免跨初始化比较）：

| 输入 | 步数 | F@1voxel | Chamfer (m) | 法线 (°) | Sharp P / R | 面积 (m²) | 分量 | 边界边 | 新面→输入距离 p50 / p90 / max（voxel） |
|---|---|---|---|---|---|---|---|---|---|
| VGGT | 0（extract-only） | **0.6075** | 0.020025 | 51.16 | 0.3881 / 0.6458 | 47.73 | 1 | 0 | 1.17 / 11.64 / 28.9 |
| VGGT | 100（完整优化） | 0.5843 | 0.020247 | 54.11 | 0.2563 / 0.8357 | 84.39 | 3 | 0 | 1.73 / 8.88 / 29.1 |
| GT-depth 代理 | 0（extract-only） | **0.8056** | 0.016142 | 48.60 | 0.6945 / 0.9357 | 44.43 | 4 | 0 | 1.23 / 12.23 / 27.3 |
| GT-depth 代理 | 100（完整优化） | 0.6647 | 0.016629 | 53.67 | 0.3608 / 0.8556 | 75.82 | 2 | 0 | 1.80 / 8.81 / 27.0 |

参考（旧初始化口径，仅供对照，不参与判定）：`full_sA`（VGGT + 旧 SDF + 100 步）F1 0.4915、Chamfer 0.026650、面积 90.95、p50/p90/max 1.82 / 9.36 / 75.0 voxel。

#### 12.9.4 保留率与相对输入的 delta（验收口径）

相对输入（不用“代理 F1 ≥ 0.7”这类绝对阈值；噪声门限取 §12.8.3：F1 ±0.0014、Chamfer ±0.000246、法线 ±0.075°、Sharp Recall ±0.0020）：

| 格子 | F1 保留率 | ΔF1 | ΔChamfer（相对输入） | Δ法线 | Sharp P / R 保留率 | 面积倍数 |
|---|---|---|---|---|---|---|
| VGGT 0 步 | **96.2%** | −0.0240 | **+59.2%**（1.59×） | +0.72° | 114.3% / 93.1% | 1.56× |
| VGGT 100 步 | 92.5% | −0.0472 | +60.9%（1.61×） | +3.67° | 75.5% / 120.5% | 2.75× |
| 代理 0 步 | **86.1%** | −0.1302 | **+318%**（4.18×） | +9.66° | 90.3% / 99.5% | 1.94× |
| 代理 100 步 | 71.0% | −0.2711 | +331%（4.31×） | +14.73° | 46.9% / 91.0% | 3.32× |

- 全部退化幅度都远超噪声（最小的一项 VGGT 0→100 步 F1 −0.0232 仍是 16× 半区间），没有“噪声内无法判定”的格子。
- **代理 0 步不是无损**：F1 保留 86% 看着正常，但 Chamfer 是输入的 4.18×、面积 1.94×、13.6% 的新表面离输入 >10 voxel。说明 SDF 转换本身对“本来很准”的输入也有明显损伤。

#### 12.9.5 责任归属：落在**第 3 行**（代理 0 步正常、100 步崩）

| 判定行 | 与本轮数据对照 | 结论 |
|---|---|---|
| 1. 两 Mesh 0 步都崩 → SDF 转换共性 | 两 0 步 F1 保留 96% / 86%，未崩；但面积 1.56× / 1.94×、Chamfer 1.59× / 4.18×、~13% 面积离输入 >10 voxel | **存在共性问题，但不是本次主因** |
| 2. 代理 0 步正常、VGGT 0 步崩 | VGGT 0 步 F1 保留 96.2%、Chamfer +59%，比代理 0 步（+318%）**相对损伤更小** | **排除** |
| 3. 代理 0 步正常、100 步崩 → 目标/正则/参数化 | 代理 0.8056 → 0.6647（−0.1409，100× 噪声）、Sharp P 0.6945 → 0.3608（−48%）、面积 44.4 → 75.8（+71%）；VGGT 同样 0.6075 → 0.5843、Sharp P 0.3881 → 0.2563 | **成立（主因）** |
| 4. 代理全程稳定、VGGT 仍失败 | 代理 100 步 F1 −17.5%（相对自身 0 步）、Sharp P 近半 | **排除** |

证据链：

- **100 步是共同的损伤来源**：两行的 F1 都下降、Sharp Precision 都下降、面积都膨胀；代理行损失更大是因为它的起点更好，而不是因为输入更差。100 步对代理把 precision 从 0.7043 打到 0.5507，而 recall 只从 0.9408 掉到 0.8384 —— 表面被推离真实位置、扩大覆盖，而不是“优化不足”。
- **初始化实现已不再是致命项**：同一份代码，纯 0 步 VGGT F1 0.6075（旧 `init_sA` 0.4821，+0.125）、面积 47.73（旧 62.82，−24%）、最大离群 28.9 voxel（旧 95.3），且 1 个连通分量、0 边界边。§12.8.9 的“实现为主因”结论**需要改写**：初始化实现的致命缺陷已被修复，剩余主因是目标/正则/参数化。
- **第 4 行的“方法可工作”不能成立**：代理行在 100 步后同样显著退化，说明不是“初始化鲁棒性不足”单独造成的。

按要求，下一步应做的是**分离 RGB、Silhouette 与颜色场自由度**（例如：冻结颜色只优化几何、关闭 RGB 只留 sil+corr、缩小颜色网格/去掉 specular），而不是继续改初始化。

#### 12.9.6 三处修复的实际效果

**(a) `--extract-only` 与旧 `init_sA`（同一 VGGT 输入、g160）**

| 指标 | `init_sA`（1 步 + 扰动，历史） | `vggt_e0`（纯 0 步） | 变化 |
|---|---|---|---|
| F@1voxel | 0.4821 | **0.6075** | +0.1254 |
| Chamfer (m) | 0.026159 | **0.020025** | −23.4% |
| 表面积 (m²) | 62.82 | **47.73** | −24.0% |
| 面数 | 444,382 | 302,544 | −31.9% |
| 连通分量 | 10 | **1** | 合并 |
| 边界边 | 136 | 0 | 闭合 |
| 新面→输入 p50 / p90 / max（voxel） | 1.03 / 9.96 / **95.3** | 1.17 / 11.64 / **28.9** | 中位 +13%，p90 +17%，**最大 −70%** |

两者差异同时包含“去掉 1 步 Adam + 扰动”和“换符号场”，不能完全拆分；可以确定的是：**旧的 1 步扰动本身就在破坏几何**（RGB 一步就把 F1 从 0.6075 量级压到 0.482）。

**(b) 符号场改动前后**

| 指标 | 旧（最近顶点法线 + 全视角前景判定） | 新（三角面距离 + 边界感知法线符号 + 三态可见性） |
|---|---|---|
| 0 步面积（VGGT） | 62.82 | **47.73** |
| 新面→输入 max | 95.3 voxel | **28.9 voxel** |
| 连通分量 / 边界边 | 10 / 136 | **1 / 0** |
| 新面→输入 p90 | 9.96 voxel | 11.64 voxel（略差） |

p90 变差的机制：边界感知把开放边上的体素判为外部，FlexiCubes 必须在开口处闭合，形成一圈“闭合墙”；这些墙在 75–95 分位表现为离输入面 5–16 voxel 的闭合面。**收益是没有离群薄片（max −70%）、连通性正常；代价是开口处多一圈墙**。这是非水密输入下“准确可贴图表面”和“闭合体积”之间的取舍，没有用强制水密化掩盖。

**(c) 三态可见性（g160，4,173,281 体素）**

| 输入 | unknown | carved | hull（未雕刻） | 不可靠（全域 / carved 内 / hull 内） | carve 翻转的“法线内部” |
|---|---|---|---|---|---|
| VGGT | 0.079% | 91.65% | 8.35% | 55.9% / 58.4% / **29.0%** | 30.2% |
| 代理 | 0.102% | 91.72% | 8.28% | 41.2% / 42.8% / **24.2%** | 16.3% |

- `unknown` 在 16 视角下可以忽略（0.08–0.10%），旧版“未进视锥即外部”的实际影响很小；真正决定符号质量的是 **hull 内 24–29% 的不可靠体素**（最近点落在开放边上），它们被默认取外部，正是开口处侵蚀与“闭合墙”的来源。
- 30.2% / 16.3% 的体素仅靠法线符号会被判为内部、但被 carving 纠正为外部——这就是旧版不做 carve 时面积翻倍的主因。

#### 12.9.7 资源开销

| run | 步数 | 秒 | torch 峰值 (MiB) | SMI delta (MiB) | 面数 | heldout primary PSNR / IoU |
|---|---|---|---|---|---|---|
| vggt_e0 | 0 | 160.8 | 1,847 | 2,819 | 302,544 | —（未渲染） |
| proxy_e0 | 0 | 146.0 | 1,842 | 3,111 | 257,476 | —（未渲染） |
| vggt_e100 | 100 | 190.5 | 3,570 | 7,072 | 571,032 | 17.99 / 0.9475 |
| proxy_e100 | 100 | 175.4 | 3,423 | 6,205 | 501,268 | 19.10 / 0.9592 |
| （对照）full_sA 旧 SDF | 100 | 109.0 | 3,625 | 5,743 | 618,374 | 17.00 / 0.9300 |

- 0 步 run 的耗时主要在建 KDTree + 4.17M 体素的三角面距离查询与 16 视角可见性判定（146–161 s 的绝大部分），FlexiCubes 提取与导出只占很小一部分。
- GPU 开跑前已用 2,245–2,671 MiB（他人进程，**未杀**）；正式 run 的 torch 峰值 ≤3.6 GB，SMI delta 峰值 7.07 GB，未触及 24,564 MiB 的 30%。**资源不是瓶颈**。
- 评测：`mesh_opt/evaluate.py` + `/root/deps/blender/blender-3.3.9-linux-x64/blender`，5 个 mesh 合计约 50 s。

#### 12.9.8 无效 run 与口径说明

| run | 状态 | 说明 |
|---|---|---|
| smoke_proxy_g64 / smoke_vggt_g64 | 冒烟，不进对照表 | grid 64 用于验证 `--extract-only` 代码路径与耗时（14.7 s / 14.8 s），不产生表内数字 |
| `init_sA` | **作废为历史** | 1 步 Adam + 随机扰动，不是纯初始化；数字仍在 §12.8.5 |
| 本轮正式 run | 4/4 有效 | vggt_e0、proxy_e0、vggt_e100、proxy_e100；无失败 run |

#### 12.9.9 边界声明

- 本线是**自建 FlexiCubes harness**：只借用 nvdiffrec 的表面提取模块（`geometry/flexicubes.py`），外观、监督、正则、评测都是本仓库实现，**不是 nvdiffrec 官方训练方案的复现**。因此结论不得表述为“nvdiffrec 方法失败”。
- 不对非水密、有真实开口或未观察区域的 Mesh 强制转成水密实体；目标是准确的可贴图表面，不是位置错误的封闭体积。
- Step 0 / Step 0.1 / Level 3 的旧结论与数字一律保留（§12.6–§12.8），本节只新增。

#### 12.9.10 结论

1. 纯初始化（0 步）不再是主要损伤源：VGGT 保留 96% F1、代理保留 86% F1；旧 `init_sA` 的 0.4821 有一部分来自那 1 步扰动 + 优化，不全是 SDF 转换的错。
2. 100 步优化是**共同**的损伤源：两行 F1、Sharp Precision、面积都朝坏的方向走，且代理行（起点最好）损失最大。责任归属第 3 行。
3. 仍有共性残留：两 0 步面积 1.56× / 1.94×、代理 Chamfer 4.18×，hull 内 24–29% 体素符号不可靠。需要局部可靠符号（GWN 或边界带内的符号扩散）才能继续压掉开口处的“墙”。
4. 下一步先做 §12.9.5 的监督分离（RGB / silhouette / 颜色场自由度），再决定是否继续投入符号场。

### 12.10 grid 分辨率专项：160 vs 256（2026-11-05）

目标：判断 §12.9 的损伤（面积膨胀、Chamfer 退化）里有多少来自 **grid 160 的量化上限**、有多少来自 **100 步的优化目标**。

§12.8、§12.9 的全部结论与数字保持原样，本节只新增。

#### 12.10.1 为什么怀疑分辨率

冻结协议里评测体素是 `0.01150144350298549`，而 FlexiCubes 的格元是 `scale / grid_res`（`scale` = 输入 bbox 最长边 × 1.10）：

| 输入 | scale (m) | grid 160 格元 | 折合评测体素 | grid 256 格元 | 折合评测体素 |
|---|---|---|---|---|---|
| GT-depth 代理 | 2.5826 | 0.016141 | **1.403 voxel** | 0.010088 | **0.877 voxel** |
| VGGT | 2.5683 | 0.016052 | **1.396 voxel** | 0.010032 | **0.872 voxel** |

g160 的格元**比评测体素（1 voxel）还粗 40%**：即使等值面完全正确，单是采样定位就可能带来约 ±0.7 voxel 的偏差，足以解释 F1@1vox 与 Chamfer 的相当一部分损失。256 的格元首次细于评测体素（0.87 voxel），因此这是一次真正的“去掉量化上限”的测试。

#### 12.10.2 代码改动与等价性

`construct_voxel_grid` 在 GPU 上要做 8 角/立方体的 `torch.unique`，res 256 的瞬时显存需求约 7–8 GB，在他人占卡时反复 OOM（见 §12.10.5）。改为：`grid_res >= 200` 时用 `FlexiCubes(device="cpu")` 在主机上构造同一份格子，再 `grid/cubes.to("cuda")`；`grid_res <= 160` 保留原 GPU 路径，**§12.9 的全部数字不受影响、不需重跑**。

等价性自检（服务器实测）：res 16/32/64/128 上 CPU 与 GPU 构造的 `cubes` 完全相同（`torch.equal`）；`grid` 最大坐标差 `5.96e-8`（float32 eps 量级，折合 `1.5e-5` 个格元），差值来自主机与设备的除法舍入，不影响几何。

#### 12.10.3 160 vs 256 对照（同相机、同分辨率 256、同一 evaluator、`--extract-only`）

| 输入 | grid | F@1voxel | Chamfer (m) | 法线 (°) | Sharp P / R | 面积 (m²) | 分量 | 面数 | p50 / p90 / max（voxel） |
|---|---|---|---|---|---|---|---|---|---|---|
| 代理（输入本身） | — | 0.9358 | 0.003859 | 38.94 | 0.7688 / 0.9404 | 22.85 | 1,152 | 251,156 | — |
| 代理 0 步 | 160 | 0.8056 | 0.016142 | 48.60 | 0.6945 / 0.9357 | 44.43 | 4 | 257,476 | 1.23 / 12.23 / 27.3 |
| 代理 0 步 | **256** | **0.8052** | **0.015712** | 49.23 | 0.6522 / 0.9505 | **50.26** | 5 | **741,016** | 1.41 / 11.96 / 27.33 |
| VGGT（输入本身） | — | 0.6315 | 0.012580 | 50.44 | 0.3395 / 0.6935 | 30.69 | 3,781 | 348,875 | — |
| VGGT 0 步 | 160 | 0.6075 | 0.020025 | 51.16 | 0.3881 / 0.6458 | 47.73 | 1 | 302,544 | 1.17 / 11.64 / 28.9 |
| VGGT 0 步 | **256** | **0.6072** | **0.020060** | 52.12 | 0.3544 / 0.6928 | **55.32** | 10 | **877,852** | 1.27 / 11.08 / 29.14 |

保留率（相对各自输入，口径同 §12.9.4）：

| 格子 | F1 保留率 | Chamfer / 输入 | 面积倍数 | Sharp P / R 保留率 |
|---|---|---|---|---|
| 代理 0 步 g160 | 86.1% | 4.18× | 1.94× | 90.3% / 99.5% |
| 代理 0 步 g256 | **86.0%** | **4.07×** | **2.20×** | 84.8% / 101.1% |
| VGGT 0 步 g160 | 96.2% | 1.59× | 1.56× | 114.3% / 93.1% |
| VGGT 0 步 g256 | **96.2%** | **1.59×** | **1.80×** | 104.4% / 99.9% |

256−160 的具体差值：

| 输入 | ΔF1 | ΔChamfer (m) | Δ面积 | Δ面数 | Δ分量 |
|---|---|---|---|---|---|
| 代理 | −0.00033 | **−0.00043**（−2.7%） | **+5.83（+13.1%）** | +483,540（2.88×） | +1 |
| VGGT | −0.00028 | **+0.000035**（+0.2%，变差） | **+7.59（+15.9%）** | +575,308（2.90×） | +9 |

#### 12.10.4 判定：**不是分辨率问题**

按“相对输入的保留率”比较（噪声门限仍取 §12.8.3：F1 ±0.0014、Chamfer ±0.000246）：

1. **F1**：两输入的 ΔF1 都是 −0.0003，**低于噪声半区间**，等于没动。g160→g256 把面数提到 2.9 倍、格元细了 1.6 倍，F@1voxel 一点没变。
2. **Chamfer**：代理 −0.00043（约为 Chamfer 噪声的 1.75 倍，可判定为真实但微小），占“输入→0 步”总退化量（0.012283）的 **3.5%**；VGGT 反而 +0.000035（变差）。两输入合计：**≥96% 的 Chamfer 退化与分辨率无关**。
3. **面积**：细网格下膨胀**更大**（代理 1.94×→2.20×，VGGT 1.56×→1.80×）。分辨率不但没有缓解膨胀，反而让它更明显——因为细网格能解析出更多“闭合墙”面。
4. **Sharp Precision**：两输入都下降（0.6945→0.6522、0.3881→0.3544），即细网格让锐边更差，不是更好。
5. **到输入的距离分布**：p90 基本不动（12.23→11.96、11.64→11.08），>10 voxel 的面积占比不降（13.6%→13.4%、12.6%→11.7%）；中位反而略升（1.23→1.41、1.17→1.27）。

结论：**面积膨胀与 Chamfer 退化都不是网格分辨率的量化上限造成的**。g160 的格元确实粗于评测体素（1.40 voxel），但把它细到 0.88 voxel 之后指标没有实质变化，说明误差来源不是“格元太粗”，而是 SDF 初始化与 100 步优化把表面推离了输入（§12.9.5 的第 3 行）。

附带说明：代理 0 步的 Chamfer（0.016142）恰好等于 g160 格元（0.016141），到 5 位有效数字重合；在 256 上这个关系不再成立（0.015712 ≈ 1.56 个 g256 格元）。本报告不把它当作因果证据，只记录为观察。

#### 12.10.5 未做的格子与无效 run

| 格子 | 状态 | 原因 |
|---|---|---|
| VGGT 0 步 g256 | **已补** | 严格按条件（“代理 256 的 Chamfer/面积保留率明显改善”）并不成立；但这一格是显存受限时的最低方案（只需 0 步，约 20 min），且能确认“两输入趋势是否一致”，故仍补跑。结果**一致**：F1 不动、面积更膨胀 |
| 代理 100 步 g256 | **未做：显存不足** | 0 步 g256 的 SMI delta 已达 **6,995 MiB**，而当时可用窗口仅 **7,523 MiB**（他人常驻 15,314 + 1,698 MiB，未杀）。100 步还要加 Adam 状态：`weights` 16.7M×21×4 B = 1.40 GB ×2、`sdf` 0.068 GB ×2、`deform` 0.20 GB ×2 ≈ **+3.3 GB**，合计约 10.3 GB > 7.5 GB 可用。按“时间允许且显存够”的前置条件判定为不可行，未启动 |

无效 run（全部为 OOM，均发生在 `construct_voxel_grid` 内，未产生任何 mesh）：

| 次数 | 位置 | 原因 |
|---|---|---|
| 3 | 手动尝试（`torch.round` / `torch.unique`） | 他人进程 15.3 GB + 1.66 GB 常驻，空闲 7.5 GB |
| 16 | `wait_g256.sh` 轮询重试（每 60 s，14:41–15:00，空闲窗口 6.4–7.5 GB） | 同上；每次都在 `torch.unique` 申请 1,024 MiB 时失败（已分配 6.28 GiB，仅剩 71 MiB） |

典型报错（原文，来自重试日志）：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB.
GPU 0 has a total capacity of 23.52 GiB of which 71.44 MiB is free.
Process 106129 has 1.66 GiB memory in use. Process 880404 has 14.96 GiB memory in use.
Process 918059 has 6.80 GiB memory in use. Of the allocated memory 6.28 GiB is allocated by PyTorch ...
```

说明：这 19 次失败的重试日志写在 `logs/proxy_e0_g256.log`，成功重跑复用了同一路径，旧 trace 已被覆盖；`logs/wait_g256.log` 的 20 行轮询记录（空闲显存时间序列）仍在。改用 CPU 构造格子后，同一台机器、同一可用窗口（7,523 MiB）一次通过。

#### 12.10.6 资源开销

| run | grid | 秒 | torch 峰值 (MiB) | SMI delta (MiB) | 面数 | mesh sha256_16 |
|---|---|---|---|---|---|---|
| proxy_e0_g256 | 256 | **1,108.1** | 6,455 | 6,995 | 741,016 | `5088aeae288fc9f0` |
| vggt_e0_g256 | 256 | **1,182.1** | 6,460 | 6,977 | 877,852 | `cabbc2886545a2f8` |
| （对照）proxy_e0 g160 | 160 | 146.0 | 1,842 | 3,111 | 257,476 | `11d09040d4335787` |
| （对照）vggt_e0 g160 | 160 | 160.8 | 1,847 | 2,819 | 302,544 | — |

- 耗时从 146–161 s 涨到 1,108–1,182 s（7.5×），**几乎全部来自 CPU 侧的 17M 体素三角面距离查询与 `torch.unique`**（grid 构造、16 视角可见性判定、FlexiCubes 提取各占很小一部分）。GPU 只在最后搬运格子与提取等值面。
- 显存：0 步 g256 的 SMI delta 6,995 MiB 已吃掉当时可用窗口（7,523 MiB）的 93%，这是 100 步无法启动的直接原因。
- 评测：`mesh_opt/evaluate.py` + Blender 3.3.9，两个 mesh 约 20 s。

#### 12.10.7 边界声明

- 本格只测 **0 步（纯初始化提取）**，不含 100 步；因此它能回答“初始化/量化上限是不是主因”，**不能**直接回答“100 步的膨胀在细网格下是否仍然存在”——后者因显存不可行而未测，结论中已按此限定表述。
- 256 的格子由 CPU 构造，与 GPU 构造在 `cubes` 上完全一致、在坐标上有 ≤`1.5e-5` 格元的浮点差；因此 g256 的数字与 g160 在严格意义上不是同一浮点路径，差异远小于任何被测效应。
- g160 全部数字未重跑、未改动；§12.8/§12.9 结论保持原样。

#### 12.10.8 结论

1. **分辨率不是瓶颈**：g256 把格元从 1.40 评测体素降到 0.88 体素、面数 ×2.9，F@1voxel 两输入都只变 −0.0003（噪声内），Chamfer 至多改善 2.7%，面积膨胀反而增加 13–16%。
2. **两输入趋势一致**：代理与 VGGT 在 256 上都表现为“F1 不动、面积更膨胀、Sharp Precision 更差”，因此不是某一输入的特例。
3. 面积膨胀与 Chamfer 退化的来源应回到 **SDF 初始化的开口处闭合墙** 与 **100 步优化目标**（§12.9.5 第 3 行），按原计划继续做 §12.9.5 的监督消融（RGB / silhouette / 颜色场自由度分离），不再在网格分辨率上投入。
4. 100 步 g256 属于“显存不足未做”，不是“做过且失败”；若之后卡上有 ≥10 GB 空闲窗口，可补这一格以把“100 步膨胀是否随分辨率变化”从推断变成实测。

### 12.11 监督消融：是谁把表面推胖（2026-11-05）

目标：§12.10 已排除 grid 分辨率的量化上限，按 §12.9.5 决策表第 3 行继续做监督分离——把 RGB、对称 silhouette、correspondence 与**颜色场自由度**逐项拆开，回答“面积膨胀与 Chamfer 退化分别由哪一项贡献”。

§12.8、§12.9、§12.10 的全部结论与数字保持原样，本节只新增。

#### 12.11.1 设置与代码开关

- 起点：GT-depth 代理 Mesh（140,623 顶点 / 251,156 面，sha256_16 `e4006a146e8ae5d6`）；grid 160、256 渲染分辨率、16 张 GT 相机与位姿、64³ 颜色网格 + SH diffuse/specular、`L_dev` 0.1、seed 20261105，与 §12.9 的 `proxy_e100` 完全一致，只动监督项与颜色自由度。
- 新开关（`scripts/experiments/step3_flexicubes.py`）：`--rgb-weight`（0 关闭 RGB）、`--sil-weight`、`--corr-weight`、`--albedo-mode {grid,fixed,lowrank}`、`--color-rank`。默认值仍是 §12.9 的 full 配置，可原样复现。
- `fixed`：albedo 冻结在 0.5 灰，只优化 SH/光照；`lowrank`：albedo 用 rank-`r` 的 CP 分解（rank 4 约 780 参数、rank 16 约 3,120 参数，对比 64³×3 = 786,432）。
- 评测：`mesh_opt/evaluate.py` + Blender 3.3.9，冻结 P0 协议，未改任何口径；区域分解用 `TriangleSDF` 按“输出面心最近输入点”分类（`opening` = 最近点落在输入开放边上、符号不可靠；`near` ≤3 vox；`mid` 3–10 vox；`far` >10 vox）。

#### 12.11.2 主因排序（先给结论）

以 0 步（§12.9 `proxy_e0`）为基准，面积倍数 1.944×、F1 0.8056、Chamfer 0.016142：

| 名次 | 来源 | 面积增量（相对 0 步） | F1（相对 0 步） | Sharp P | 判定 |
|---|---|---|---|---|---|
| 1 | **RGB 光度项** | **+1.636×**（119% 的 full 增量） | −0.181 | 0.310 | 主因，且**独占**膨胀 |
| 2 | 颜色场自由度不足 | +2.037 ~ +3.165×（rank-16 / rank-4 / frozen） | −0.296 / −0.303 / −0.348 | 0.208 / 0.196 / 0.151 | **反向**：限制颜色让几何更胖 |
| 3 | silhouette | +0.060×（4%） | −0.019 | 0.744 | 基本是空转，不抑制也不膨胀 |
| 4 | correspondence | −0.069×（−5%） | −0.028 | 0.775 | 弱激活，几乎不改变几何 |

- **RGB 光度项是唯一把表面推胖的项**。只开 RGB（`rgb_only`）就得到 3.580×，比 full（3.318×）还胖；把 RGB 关掉（`sil_only`、`corr_only`）面积回到 2.004× / 1.875×，即 0 步水平。按增量算：full 的 +1.374× 里，RGB 单项就贡献 +1.636×。
- **颜色自由度不是帮凶，是减压阀**。自由度越小越胖，且单调：64³ 网格 3.318× → rank-16 3.981× → rank-4 4.284× → 冻结 5.109×；F1 同步从 0.6647 掉到 0.4578。也就是说，颜色场能解释外观时，几何不必去解释；颜色一旦被冻住/压扁，同一份 RGB 残差只能由几何去背。
- **silhouette 与 correspondence 都不背这个锅**。`sil_only` 的总损失（此时只剩 silhouette 一项）从 0.0038 降到 0.0019，但面积几乎不动（2.004× vs 1.944×）——初始化（carving）之后，轮廓本来就已经被满足，这一项没有多少可动的空间。
- correspondence 只作参考：本轮对应点是 217 个、`surface_coverage_2vox` 1.9%，历史上就被标为弱激活；`rgb_sil`（3.325×）与 full（3.318×）在噪声内相同，说明去掉它不影响膨胀。

#### 12.11.3 各配置对照表（g160，除注明外 100 步）

| 配置 | RGB / sil / corr | F1 | ΔF1 vs 0 步 | Chamfer (m) | Cham×输入 | ΔChamfer vs 0 步 | 法线 (°) | Sharp P / R | 面积 (m²) | 面积倍数 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 步（§12.9 基准） | — | **0.8056** | — | **0.016142** | 4.183 | — | 48.60 | 0.6945 / 0.9357 | 22.213 | 1.944× |
| full（§12.9 `proxy_e100`） | 1 / 0.5 / 0.02 | 0.6647 | −0.1409 | 0.016629 | 4.309 | +0.000487 | 53.67 | 0.3608 / 0.8556 | 37.909 | 3.318× |
| rgb_only | 1 / 0 / 0 | 0.6243 | −0.1813 | 0.017626 | 4.568 | +0.001484 | 54.19 | 0.3100 / 0.8386 | 40.904 | 3.580× |
| sil_only | 0 / 0.5 / 0 | **0.7865** | −0.0191 | **0.016051** | 4.160 | −0.000091 | 50.53 | 0.7435 / 0.8664 | 22.902 | 2.004× |
| corr_only | 0 / 0 / 0.02 | 0.7775 | −0.0281 | 0.017187 | 4.454 | +0.001045 | 51.66 | **0.7747** / 0.7812 | 21.417 | 1.875× |
| rgb_sil | 1 / 0.5 / 0 | 0.6655 | −0.1401 | 0.016675 | 4.321 | +0.000533 | 53.66 | 0.3631 / 0.8657 | 37.989 | 3.325× |
| fixed albedo | 1 / 0.5 / 0.02 | 0.4578 | −0.3478 | 0.026915 | 6.975 | +0.010773 | 56.52 | 0.1512 / 0.7495 | 58.375 | **5.109×** |
| lowrank-16 albedo | 1 / 0.5 / 0.02 | 0.5100 | −0.2956 | 0.022618 | 5.861 | +0.006476 | 55.17 | 0.2078 / 0.7538 | 45.480 | 3.981× |
| lowrank-4 albedo | 1 / 0.5 / 0.02 | 0.5029 | −0.3027 | 0.022841 | 5.919 | +0.006699 | 55.38 | 0.1963 / 0.7300 | 48.947 | 4.284× |
| rgb_only, 25 步 | 1 / 0 / 0 | 0.6316 | — | 0.017019 | 4.410 | — | 53.63 | 0.4027 / 0.7451 | 32.911 | 2.880× |
| rgb_only, 50 步 | 1 / 0 / 0 | 0.6418 | — | 0.016792 | 4.351 | — | 53.65 | 0.3729 / 0.8051 | 35.545 | 3.111× |
| full, 25 步 | 1 / 0.5 / 0.02 | 0.6665 | — | 0.016632 | 4.310 | — | 53.23 | 0.4587 / 0.7830 | 31.879 | 2.790× |
| full, 50 步 | 1 / 0.5 / 0.02 | 0.6788 | — | 0.016222 | 4.204 | — | 53.12 | 0.4196 / 0.8245 | 33.902 | 2.967× |

噪声门限沿用 §12.8.3（2 seeds）：F1 ±0.0014、Chamfer ±0.000246、法线 ±0.075°、Sharp Recall ±0.0020；Sharp Precision 未单独测种子噪声，但本轮全部变化的量级（0.049–0.543）比任何候选噪声大一个数量级以上。

读法与结论：

1. 除 `sil_only` 的 ΔF1（−0.0191，13.6× 半区间）外，所有 ΔF1 都在 20–250× 噪声之外；`sil_only` / `corr_only` 的 Chamfer 变化在噪声内（−0.000091 = 0.37×、+0.001045 = 4.2×）。
2. Chamfer 的退化同样是 RGB 主导：`rgb_only` 比 0 步差 +0.001484（6.0× 噪声），full 只差 +0.000487（2.0×）；关掉 RGB 后`sil_only` 甚至比 0 步略好。注意 0 步本身已经把 Chamfer 打到输入的 4.18×（§12.9 的初始化残留），本轮讨论的是**优化引入的那一部分**。
3. 覆盖（Recall）几乎不随配置变化（0.73–0.89），掉的是 **Precision**（0.50–0.77 vs 0.7043）与 **Sharp P**（0.151–0.775 vs 0.6945）：表面被推得比真实位置更外、更钝，而不是“没覆盖到”。

#### 12.11.4 颜色场自由度：不是帮凶，是减压阀

同一条自由度轴（其余设置完全相同）：

| 颜色场 | 参数量 | 面积倍数 | F1 | Sharp P | Chamfer (m) | heldout PSNR |
|---|---|---|---|---|---|---|
| 64³ 网格（full） | 786,432 | **3.318×** | **0.6647** | **0.3608** | **0.016629** | **18.83 / 19.10**（`rgb_sil` / full） |
| CP rank 16 | ≈3,120 | 3.981× | 0.5100 | 0.2078 | 0.022618 | 15.25 |
| CP rank 4 | ≈780 | 4.284× | 0.5029 | 0.1963 | 0.022841 | 15.26 |
| 冻结（0.5 灰） | 0 | 5.109× | 0.4578 | 0.1512 | 0.026915 | 13.16 |

- 面积随自由度**单调**下降，F1 随自由度单调上升 → 颜色自由度**在帮着把几何推瘦**，不是推胖。这与“颜色场抢走了几何该解释的东西”的直觉相反：更自由的颜色反而让几何更接近输入。
- 需要写明的**混淆**：`fixed` 不是“冻结到真实 albedo”，而是冻结在 0.5 灰，光度目标从一开始就不可能被满足——它与 full 的起点相同（总 loss 0.2934），但 full 能降到 0.0527，`fixed` 只降到 0.1703、rank-4/16 只降到 0.1224/0.1222。也就是说这三格的 RGB 残差**根本没被解释掉**，剩下的差额只能由几何去背。因此它们只能证明“颜色失信 → 几何背锅”，**不能**单独证明“自由度少就必然膨胀”；rank-4/16 两格是同一目标下的纯自由度对照，方向仍然一致。
- 这一组也说明 full 的 3.318× **不是**颜色场“太自由”造成的：把自由度降到 1/250，面积不是缩小而是涨到 3.98–5.11×。

#### 12.11.5 步数曲线：目标本身偏好胖面，且后期继续漂移

| 步数 | full 面积倍数 | full F1 | full Sharp P | rgb_only 面积倍数 | rgb_only F1 | rgb_only Sharp P |
|---|---|---|---|---|---|---|
| 0 | 1.944× | 0.8056 | 0.6945 | 1.944× | 0.8056 | 0.6945 |
| 25 | 2.790× | 0.6665 | 0.4587 | 2.880× | 0.6316 | 0.4027 |
| 50 | 2.967× | 0.6788 | 0.4196 | 3.111× | 0.6418 | 0.3729 |
| 100 | 3.318× | 0.6647 | 0.3608 | 3.580× | 0.6243 | 0.3100 |

- 面积在两条曲线上都**单调上升**（full 1.94 → 2.79 → 2.97 → 3.32；rgb_only 1.94 → 2.88 → 3.11 → 3.58），Sharp P 单调下降。
- 但**不是纯后期漂移**：25 步已经到 2.79–2.88×，即 100 步值的 84–80%。膨胀在早期就发生，之后只是缓慢继续。这支持“目标本身偏好更大的表面积（更多面能解释更多像素）”而不是“训练后期跑飞”。
- F1 在 50 步见顶（0.6788 / 0.6418），100 步回落 0.014（10× 噪声）；即“多跑 50 步”既不改善几何，也不改善锐边。

#### 12.11.6 面积增量落在哪些区域

面积用真实三角形面积（`0.5·Σ|cross|`，见 12.11.9 的口径更正）；输入面积 11.426 m²。区域按输出面心到输入表面的最近距离分类，`opening` 表示最近点落在输入的开放边上（符号不可靠）：

| 配置 | 总面积 | opening | near (≤3 vox) | mid (3–10 vox) | far (>10 vox) |
|---|---|---|---|---|---|
| 0 步 | 22.213 | 4.62（20.8%） | 12.04（54.2%） | 3.11（14.0%） | 2.46（11.1%） |
| full 100 步 | 37.909 | 8.30（21.9%） | 20.51（54.1%） | 6.56（17.3%） | 2.50（6.6%） |
| rgb_only | 40.904 | 9.85（24.1%） | 21.11（51.6%） | 7.28（17.8%） | 2.68（6.6%） |
| sil_only | 22.902 | 4.12（18.0%） | 13.42（58.6%） | 2.89（12.6%） | 2.47（10.8%） |
| corr_only | 21.417 | 3.39（15.8%） | 12.68（59.2%） | 2.87（13.4%） | 2.49（11.6%） |
| rgb_sil | 37.989 | 8.37（22.0%） | 20.44（53.8%） | 6.65（17.5%） | 2.56（6.7%） |
| fixed albedo | 58.375 | 13.51（23.1%） | 21.37（36.6%） | 18.10（31.0%） | 5.41（9.3%） |
| lowrank-16 | 45.480 | 11.90（26.2%） | 19.47（42.8%） | 11.01（24.2%） | 3.11（6.8%） |
| lowrank-4 | 48.947 | 12.82（26.2%） | 20.32（41.5%） | 12.63（25.8%） | 3.21（6.6%） |

相对 0 步的增量（m²）：

| 配置 | Δ 总面积 | Δ opening | Δ near | Δ mid | Δ far |
|---|---|---|---|---|---|
| full 100 步 | +15.696 | **+3.68** | **+8.47** | +3.45 | +0.04 |
| rgb_only | +18.691 | +5.23 | +9.07 | +4.17 | +0.22 |
| sil_only | +0.689 | −0.50 | +1.38 | −0.22 | +0.01 |
| corr_only | −0.796 | −1.23 | +0.64 | −0.24 | +0.03 |
| fixed albedo | **+36.162** | +8.89 | +9.33 | **+14.99** | +2.95 |

- 增量**贴着输入表面**：full 的 +15.7 m² 里 8.47 落在 `near`（≤3 voxel 的壳层）、3.68 落在开口区、3.45 落在 3–10 voxel，离输入 >10 voxel 的 `far` 几乎不变（+0.04）。所以膨胀是“沿表面外扩的一层壳”＋“开口处闭合墙加高”，不是“飞出远处的大薄片”。
- `far` 占比在所有 100 步配置里都从 11.1% 降到 6.6–6.8%（面积绝对值不动），说明优化没有新增离群面，只是把已有的近表面加厚。
- 冻结/低秩颜色时，`mid` 从 3.11 m² 涨到 11.0–18.1 m²（full 的 3.5–4.8 倍），是最能区分“颜色失信 → 几何背锅”的区域特征：颜色解释不了的中间距离表面被推向更远的壳层。
- 口径限制：`opening` 是“最近点落在输入开放边上”，占输出的 16–26%，意味着这部分面积的距离/符号都不可靠（输入本身有 30,964 条开放边）；`near` 与 `far` 的判定是可靠的。

#### 12.11.7 冲突单列：RGB 同时是“唯一的对外观负责的项”和“唯一的膨胀来源”

| 配置 | 几何（F1 / Sharp P / 面积倍数） | 外观（train PSNR / heldout primary PSNR） |
|---|---|---|
| rgb_only | 0.6243 / 0.3100 / 3.580× | 21.58 / **17.95** |
| rgb_sil | 0.6655 / 0.3631 / 3.325× | 21.89 / **18.83** |
| full 50 步 | 0.6788 / 0.4196 / 2.967× | 21.69 / **19.07** |
| sil_only | 0.7865 / 0.7435 / 2.004× | 10.66 / 11.41（颜色场从未被监督，**PSNR 无意义**） |
| corr_only | 0.7775 / 0.7747 / 1.875× | 10.27 / 11.05（同上） |
| fixed albedo | 0.4578 / 0.1512 / 5.109× | 14.06 / 13.16 |
| lowrank-4 | 0.5029 / 0.1963 / 4.284× | 16.64 / 15.26 |

- 几何最好的两格（`sil_only`、`corr_only`）**没有可用于外观的颜色**：RGB 权重为 0 时颜色网格拿不到梯度，PSNR 约 11 dB，只是“几何没被推动”的空转结果，不能作为“去掉 RGB 的方案”。所以“去掉 RGB 换取几何”在本评测里是空的。
- 既要外观又要几何的可行配置里，`rgb_sil`（18.83 dB / F1 0.6655 / 3.325×）与 full 的 `50 步`（19.07 dB / 0.6788 / 2.967×）是最不坏的组合；`full 100 步`多花的 50 步换来 PSNR +0.04 dB、F1 −0.014、面积 +0.35×，是净损失。
- 结论：**冲突是真实的**——RGB 既是唯一能产出外观的项，也是唯一把几何推胖的项。本轮**没有任何配置**同时保住几何不退化并产出可用外观；能做的只是把 RGB 的步数截断在 50 步以少付一点几何代价。

#### 12.11.8 资源、GPU 排队与无效 run

| run | 步数 | 秒 | torch 峰值 (MiB) | SMI delta (MiB) | 面数 | mesh sha256_16 |
|---|---|---|---|---|---|---|
| rgb_only | 100 | 193.1 | 3,503 | 5,109 | 541,736 | `7075f350132e82cc` |
| sil_only | 100 | 175.8 | 3,250 | 4,527 | 280,672 | `ae62030ee23f76a0` |
| corr_only | 100 | 176.5 | 3,240 | 4,211 | 261,348 | `123a6e7fed991f25` |
| rgb_sil | 100 | 192.5 | 3,400 | 5,003 | 502,312 | `4ff1e9cda48e3fc6` |
| fixed_albedo | 100 | 204.9 | 3,751 | 5,277 | 771,628 | `4d56ba87a4b773ec` |
| lowrank-4 | 100 | 195.1 | 3,619 | 5,293 | 647,636 | `16b851669e7e3e8d` |
| lowrank-16 | 100 | 191.5 | 3,576 | 5,255 | 607,156 | `396b46a98a2a3988` |
| rgb_only 25 步 | 25 | 158.1 | 3,314 | 4,903 | 427,500 | `eb022a47a29af94e` |
| rgb_only 50 步 | 50 | 169.2 | 3,357 | 4,971 | 465,812 | `a22f5c338705bb4a` |
| full 25 步 | 25 | 155.9 | 3,302 | 4,905 | 414,060 | `3915808e7b06ab5f` |
| full 50 步 | 50 | 174.9 | 3,338 | 5,384 | 445,540 | `a1603a6e069147cd` |

- 单 run 155.9–204.9 s，其中 140 s 左右是固定的 SDF 建格 + 16 视角可见性 + KDTree 初始化，**每步优化只占 0.3–0.6 s**；因此 25 步与 100 步的耗时只差 40 s。11 个 run 合计约 35 min GPU；评测 11 个 mesh 约 2 min（CPU）。
- 显存：torch 峰值 3.24–3.75 GB，SMI delta 4.21–5.38 GB，远低于 24 GB。**资源不是瓶颈**。
- **GPU 排队（如实记录）**：16:31 启动时卡上有他人常驻 15,314 + 1,698 MiB（free 7,043 MiB）。前两个 run（`lowrank-4`、`fixed_albedo`）在 16:49–16:54 跑完。之后纹理线（`scripts/experiments/tex_t1_run.sh`）取得 `/root/3dtex/.locks/gpu4090.lock`，并在**持锁状态下**等 ≥10,000–10,500 MiB 空闲（该窗口始终只有 7,043 MiB），因此从 17:18 排队到 **18:08:05** 才拿到锁，等待 **1,465 s**（含此前一轮约 50 min 的排队）。全程未杀任何进程、未与推理任务并发抢卡；拿到锁后仍先检查 `free ≥ 6,300 MiB` 才开跑，若显存被别人占住则立刻释放锁再排队（避免自己变成持锁阻塞者）。18:08 之后纹理线改为轮询、每轮只短暂持锁，剩余 8 个 run 的排队时间降到 0–45 s。
  期间我为避免 `flock -w 3600` 到期丢 run，把排队的 battery 从“单次长等待”改成“3 s 轮询 + 拿到锁先验显存、不足立刻释放”，前两个 run 的产物与 `mesh.ply` 被复用（`SKIP`），未重跑。
- **无效 run：无**。11 个 run 全部 `EXIT=0` 且产出 `mesh.ply`；没有 OOM、没有中断。被替换的只是排队脚本本身（v1/v2 是两个 battery 包装脚本），不是科学 run。

#### 12.11.9 口径更正（只新增说明，不改旧数字）

1. **面积口径**：§12.9 / §12.10 表里的“面积 (m²)”用的是 `Σ|cross|`（等于真实面积的 2 倍），本节起用 `0.5·Σ|cross|`（真实面积）。因此**同一网格的绝对面积值本节是旧表的一半**（例如 `proxy_e0` 旧表 44.43 → 本节 22.213），所有倍数、比值不受影响。§12.9 / §12.10 的数字原样保留，不做改写。
2. **连通分量口径**：§12.9 的“分量”列（`proxy_e0` = 4、`proxy_e100` = 2）来自 `/tmp/init2x2_stats.py` 的 union-find，其中用 `index // 3` 取面编号，而 `index` 是 `3·N` 行拼接后的行号——这个映射会把面编号算错（正确应为 `index % N`），结果是过并，旧分量数偏小。本节改用正确的“共享边邻接 + 面编号 `% N`”重算，并对旧锚点也重算了一遍作参考（不同工具的邻接定义会有差异，这里只保证表内 11 个 run 是同一定义）：

| 网格 | 旧“分量”（有映射错误，不改） | 正确分量（共享边邻接） | 其中 ≥50 面 | 最大分量面数 |
|---|---|---|---|---|
| 代理输入 | 1 | **1,211** | 9 | 244,308 |
| `proxy_e0` | 4 | **660** | 66 | 241,976 |
| `proxy_e100` | 2 | **1,438** | 68 | 473,252 |

   本节所有 run 的“正确分量”都是 500–2,950 量级，但**每一格都只有 1 个 ≥1% 面数的分量**（最大分量占 92–94%），其余是小数目的碎片。边界边全部为 0（输出闭合）。因此“分量数”在本轮不能作为“几何塌缩/炸开”的判据；真正的判据是最大分量占比（全部正常）与面积/距离分布。

#### 12.11.10 边界声明

- 本线仍是**自建 FlexiCubes harness**，不是 nvdiffrec 官方训练方案的复现；结论不得表述为“nvdiffrec 方法失败”。
- 几何指标全部相对 **GT-depth 代理 Mesh**（由 16 张 GT 相机渲染的 GT 深度生成）评测，不是官方 benchmark，也不与 DTU/ETH3D 榜单比较。
- 所有格子只跑 1 个 seed（噪声门限取自 §12.8.3 的 2-seed 估计）；单 seed 的判定一律对照该门限，不另设新门限。
- `sil_only` / `corr_only` 的 PSNR 不可解释（颜色场无监督），只作为“RGB 项缺席时几何不动”的对照，不参与外观比较。
- `fixed albedo` 是“冻结在 0.5 灰”，不是“冻结到真实 albedo”，存在目标不可达的混淆，已在 12.11.4 写明。
- 未做的对照：`--rgb-weight` 的中间值（0.1 / 0.25 / 0.5）与 silhouette 的中间值未扫；`--color-res` 空间分辨率轴（16³ / 32³）未扫。若要判断“RGB 权重能不能买到折中”，需要这两条轴，本轮只给了 1.0 与 0 两端。

#### 12.11.11 结论

1. **主因是 RGB 光度项**：只开 RGB 就复现并超过 full 的膨胀（3.580× vs 3.318×），关掉 RGB 面积回到 0 步水平（2.004× / 1.875×）。面积增量 119% 来自 RGB，silhouette 贡献 4%，correspondence 贡献 −5%。
2. **颜色自由度不是推胖的原因，是减压阀**：自由度 786k → 3k → 780 → 0 参数，面积 3.318× → 3.981× → 4.284× → 5.109×，F1 0.6647 → 0.4578。限制颜色会把同一份 RGB 残差赶到几何上。
3. **膨胀位置**：+15.7 m² 增量里 8.47 在 `near`（≤3 voxel 壳层）、3.68 在开口闭合区、3.45 在 3–10 voxel，`far` 几乎不动 → 是“沿表面加厚 + 开口加墙”，不是离群飞面。
4. **步数**：面积随步数单调上升，25 步已达 100 步的 80–84%，属“目标偏好胖面 + 缓慢持续漂移”；F1 在 50 步见顶，100 步净损失。
5. **冲突无法在本轮内解决**：RGB 是唯一产出外观的项，也是唯一推胖的项；几何最好的两格没有外观。下一步要判断是否存在折中，只能扫 `--rgb-weight` 中间值与 `--color-res`，而不是回退到分辨率或初始化。

### 12.12 取舍曲线：面积膨胀 vs 外观（2026-11-05）

目标：§12.11 判定“RGB 既是唯一产出外观的项、也是唯一推胖的项”后，本节扫 `--rgb-weight` 与 `--color-res` 两条轴，回答“有没有一个点，再降 RGB 权重 / 再压颜色分辨率时，外观掉得比几何赚得多”。不回头碰分辨率（§12.10 已排除）与初始化（§12.9 已冻结）。

§12.8–§12.11 的全部结论与数字保持原样，本节只新增。

#### 12.12.1 设置

- 起点与 §12.9 `proxy_e100` 完全一致：GT-depth 代理 Mesh（sha256_16 `e4006a146e8ae5d6`）、grid 160、256 渲染分辨率、16 张 GT 相机、64³ 颜色网格 + SH diffuse/specular、`L_dev` 0.1、seed 20261105、100 步；correspondence 复用 `outputs/step_init_20261105/proxy_e100/correspondence.npz`（217 点，`surface_coverage_2vox` 1.9%，弱激活）。
- 两轴**单独扫**，不做初始交叉网格：
  - 轴 1：`--rgb-weight` ∈ {0.1, 0.25, 0.5}，其余保持 full（rgb_w 1.0 的 `full100` 作对照点，`full50` 作步数参照）。
  - 轴 2：`--color-res` ∈ {16³, 32³}，`--rgb-weight` 保持 1.0，64³ 的 `full100` 作对照。
- **额外探针（超出原设计，用于定位悬崖）**：轴 1 在 0.02 处出现拐点后，补扫 `rgb-weight` = 0.005 / 0.01，并在 0.02 处做一次 2×2 交叉确认（rgb_w {0.02, 1.0} × color-res {16, 64}，只新增 `rgbw002_c16` 一格，其余三格复用 `rgbw002`、`cres16`、`full100`）。规则第 3 条要求“单独扫出明显拐点才做交叉”，本轮的拐点正是由这次交叉确认的。
- 终止点：`rgb-weight = 0` 的 `sil_only`（§12.11）作为“颜色场完全失去监督”的端点参照，其 PSNR 无意义，只用于展示断崖位置。

#### 12.12.2 结论（先给）

1. **拐点存在，位置是 `rgb-weight = 0.02`**（color-res 64，100 步）：面积 2.224×、heldout PSNR **20.27**（全部 12 个有外观读数的格子里最高）、F1 0.7636、Chamfer 0.015698。在 0.02 之上（0.05→1.0），降权重让几何与外观**同时**变好；在 0.02 之下（0.01→0.005），几何继续变好而外观开始变差。拐点就卡在这两个区间之间。
2. **拐点很浅**：0.02 → 0.005 只掉 **0.32 dB**（20.27 → 19.95），换来面积 −0.169×（2.224 → 2.055）、F1 +0.0210（0.7636 → 0.7846）、Sharp P +0.0897（0.6254 → 0.7151）。按单位面积折算，这一段是 **−0.189 dB / 0.1×**；而右侧 1.0→0.02 是 **+0.107 dB / 0.1×**——同一方向（降面积）在右侧赚钱、在左侧亏钱，这就是拐点的定义。
3. **颜色分辨率轴没有拐点**：16³ → 32³ → 64³ 在几何与外观上是**双输/双赢**关系（面积 3.579× → 3.415× → 3.318×，PSNR 17.10 → 18.38 → 19.10）。压颜色分辨率不构成取舍点——它只是同时伤害两边。
4. **断崖在 0.005 → 0**：`sil_only` 面积 1.944×（回到 0 步）、F1 0.7865，但 PSNR 11.41——颜色场拿不到梯度。这不是“折中”，是失效。
5. 因此**不能写“在该目标形式下不存在折中点”**：折中点存在（0.02 附近），但它的成因不是“光度项与几何天然对立”，而是**权重失配 + 缺少几何侧正则**。详见 12.12.8。
6. **目前最不坏的可行配置 = `rgbw002`**（rgb_w 0.02 / color-res 64 / 100 步）。它与 0 步基准的差距见 12.12.7：几何仍明显差于初始化（F1 −0.0420，30× 噪声），只是“少伤害 + 最好外观”，不代表方法已经有效。

#### 12.12.3 扫参表（g160，除注明外 100 步；sil 0.5 / corr 0.02）

| run | rgb_w | color_res | 步数 | F1 | ΔF1 vs 0 步 | Chamfer (m) | Cham×输入 | 法线 (°) | Sharp P / R | 面积 (m²) | 面积倍数 | heldout PSNR | train PSNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 步（基准） | — | — | 0 | **0.8056** | — | **0.016142** | 4.183 | 48.60 | 0.6945 / 0.9357 | 22.213 | 1.944× | — | — |
| rgbw0005 | 0.005 | 64 | 100 | 0.7846 | −0.0210 | 0.015862 | 4.111 | 50.06 | 0.7151 / 0.8744 | 23.477 | 2.055× | 19.95 | 22.26 |
| rgbw001 | 0.01 | 64 | 100 | 0.7775 | −0.0281 | 0.015745 | 4.080 | 50.22 | 0.6918 / 0.8769 | 24.138 | 2.113× | 20.12 | 22.57 |
| **rgbw002** | **0.02** | **64** | **100** | **0.7636** | −0.0420 | **0.015698** | **4.068** | 50.39 | 0.6254 / 0.8679 | 25.416 | **2.224×** | **20.27** | 22.72 |
| rgbw005 | 0.05 | 64 | 100 | 0.7333 | −0.0723 | 0.015795 | 4.093 | 51.64 | 0.5178 / 0.8729 | 28.250 | 2.473× | 20.00 | 22.68 |
| rgbw010 | 0.10 | 64 | 100 | 0.7122 | −0.0934 | 0.016170 | 4.190 | 52.61 | 0.4713 / 0.8726 | 30.908 | 2.705× | 19.76 | 22.41 |
| rgbw025 | 0.25 | 64 | 100 | 0.6944 | −0.1112 | 0.016340 | 4.234 | 52.82 | 0.4176 / 0.8733 | 33.601 | 2.941× | 19.68 | 22.35 |
| rgbw050 | 0.50 | 64 | 100 | 0.6797 | −0.1259 | 0.016428 | 4.257 | 53.01 | 0.3836 / 0.8646 | 35.952 | 3.147× | 19.25 | 22.04 |
| full100（对照） | 1.00 | 64 | 100 | 0.6647 | −0.1409 | 0.016629 | 4.309 | 53.67 | 0.3608 / 0.8556 | 37.909 | 3.318× | 19.10 | 21.88 |
| full50（参照） | 1.00 | 64 | 50 | 0.6788 | −0.1268 | 0.016222 | 4.204 | 53.12 | 0.4196 / 0.8245 | 33.902 | 2.967× | 19.07 | 21.69 |
| rgbw002_c16 | 0.02 | 16 | 100 | 0.6934 | −0.1122 | 0.017303 | 4.484 | 51.37 | 0.4834 / 0.8058 | 26.840 | 2.349× | 18.31 | 19.62 |
| cres16 | 1.00 | 16 | 100 | 0.5786 | −0.2270 | 0.019382 | 5.023 | 54.14 | 0.2742 / 0.7617 | 40.891 | 3.579× | 17.10 | 18.91 |
| cres32 | 1.00 | 32 | 100 | 0.6335 | −0.1721 | 0.017779 | 4.607 | 53.26 | 0.3380 / 0.8134 | 39.021 | 3.415× | 18.38 | 20.92 |

噪声门限沿用 §12.8.3（2 seeds）：F1 ±0.0014、Chamfer ±0.000246、法线 ±0.075°、Sharp Recall ±0.0020。**PSNR 没有 2-seed 噪声估计**（§12.8.3 未包含），因此 0.1–0.2 dB 级的差异一律不作显著性声明，只报数值与单调方向。

读法：

1. `rgb-weight` 轴：F1 在 0.005–0.02 之间每档变化 0.0071–0.0139（5–10× 噪声），是真实效应；面积 2.055× → 3.318× 单调；PSNR 在 20.27 处见顶后回落。
2. Chamfer 的**名义最优就在 0.02**（0.015698，比 0 步还低 0.000444 = 1.8× 噪声），0.005 / 0.01 反而略升（+0.000164 / +0.000047，均在 1× 噪声内）——即再往低权重走，Chamfer 不再获益，只有 F1 / Sharp P 继续改善。
3. `rgbw002_c16` 是 2×2 里唯一新格：说明在颜色容量被压到 16³ 时，把 rgb 权重降到 0.02 仍然**同时**改善几何（3.579× → 2.349×）与外观（17.10 → 18.31 dB）。两轴是**可加的主效应**，没有交互（见 12.12.5）。

#### 12.12.4 取舍曲线：面积× vs heldout PSNR

按面积升序排列（即把 12 格连成一条曲线；`sil_only` 只作端点参照，不参与斜率拟合）：

| 点 | 面积× | heldout PSNR | F1 | Sharp P | Chamfer | 向更小面积走一步：ΔPSNR / Δ面积 = 斜率 |
|---|---|---|---|---|---|---|
| `sil_only`（rgb 0） | 1.944× | 11.41（**无意义**：颜色无监督） | 0.7865 | 0.7435 | 0.016051 | —（断崖，不是折中） |
| rgbw0005 | 2.055× | 19.95 | 0.7846 | 0.7151 | 0.015862 | → 1.944×：−8.54 dB（失效） |
| rgbw001 | 2.113× | 20.12 | 0.7775 | 0.6918 | 0.015745 | → 2.055×：−0.17 dB / −0.058× = **−0.29 dB / 0.1×** |
| **rgbw002（峰）** | **2.224×** | **20.27** | 0.7636 | 0.6254 | **0.015698** | → 2.113×：−0.15 dB / −0.111× = **−0.14 dB / 0.1×（左侧）** |
| rgbw005 | 2.473× | 20.00 | 0.7333 | 0.5178 | 0.015795 | → 2.224×：**+0.27 dB（+0.108 / 0.1×，右侧）** |
| rgbw010 | 2.705× | 19.76 | 0.7122 | 0.4713 | 0.016170 | → 2.473×：+0.24 dB（+0.103 / 0.1×） |
| rgbw025 | 2.941× | 19.68 | 0.6944 | 0.4176 | 0.016340 | → 2.705×：+0.08 dB（+0.034 / 0.1×） |
| rgbw050 | 3.147× | 19.25 | 0.6797 | 0.3836 | 0.016428 | → 2.941×：+0.43 dB（+0.209 / 0.1×） |
| full100 | 3.318× | 19.10 | 0.6647 | 0.3608 | 0.016629 | → 3.147×：+0.15 dB（+0.088 / 0.1×） |
| cres32（rgb 1.0） | 3.415× | 18.38 | 0.6335 | 0.3380 | 0.017779 | → 3.318×：+0.72 dB（+0.74 / 0.1×，颜色轴） |
| cres16（rgb 1.0） | 3.579× | 17.10 | 0.5786 | 0.2742 | 0.019382 | → 3.415×：+1.28 dB（+0.78 / 0.1×，颜色轴） |

曲线形状（示意，横轴 0.1× 一格、纵轴 0.5 dB 一格；纵轴数值按格取整）：

```text
held PSNR
 20.5 ┤      ★0.02
 20.0 ┤ ●0.005/0.01     ●0.05
 19.5 ┤                        ●0.10  ●0.25
 19.0 ┤                                  ●0.50    ●1.00
 18.5 ┤                                              ●res32
 17.0 ┤                                                      ●res16
      └┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──
      2.0    2.25   2.5    2.75   3.0    3.25   3.5   (面积×)
```

读法：

- 曲线是**单峰**的：从左（小面积、高外观）到右（大面积、低外观）递减，但在最左端（2.055×/2.113×）掉头向下——那一段才是真正的取舍区。
- **峰点 = 2.224× / 20.27**，即 `rgbw002`。它不是“几何与外观都最好”（几何最好的是 `rgbw0005` 与 `sil_only`），而是**外观最好**，同时几何优于 full100 与 full50。
- 两条轴的斜率差 7 倍：颜色轴每换回 0.1× 面积要付 **0.77 dB**，权重轴（右侧）只要 **0.107 dB**。外观的瓶颈主要在**颜色容量**、不在几何。

#### 12.12.5 2×2 交叉确认

| 面积× / PSNR / F1 | color-res 64 | color-res 16 | 颜色轴主效应 |
|---|---|---|---|
| **rgb_w 1.0** | 3.318× / 19.10 / 0.6647 | 3.579× / 17.10 / 0.5786 | 面积 −0.261×、PSNR +2.00 |
| **rgb_w 0.02** | 2.224× / 20.27 / 0.7636 | 2.349× / 18.31 / 0.6934 | 面积 −0.125×、PSNR +1.96 |
| 权重轴主效应 | 面积 −1.094×、PSNR +1.17 | 面积 −1.230×、PSNR +1.21 | — |

- 权重轴的主效应在两种颜色分辨率下几乎相同（PSNR +1.17 vs +1.21，F1 +0.0989 vs +0.1152），**没有交互**：拐点在 64³ 上成立，在 16³ 上也成立。
- 颜色轴的主效应也稳定（PSNR +2.00 vs +1.96），但在低权重下换到的面积更少（−0.125× vs −0.261×）——说明颜色容量不足造成的膨胀，可以被“降权重”抵消掉一部分，且两者叠加后仍优于 full。
- 交叉结论：**拐点不是 64³ 的特例**；`rgbw002` 仍是 2×2 里的最优点（面积最小、PSNR 最高的格）。

#### 12.12.6 区域分解：低权重下增量只剩“贴面壳层”

面积用真实三角形面积（`0.5·Σ|cross|`）；输入面积 11.426 m²；分类同 §12.11（`opening` = 最近输入点落在开放边上、符号不可靠；`near` ≤3 vox；`mid` 3–10 vox；`far` >10 vox）。

| 配置 | 总面积 | opening | near | mid | far | Δ vs 0 步（opening / near / mid / far） |
|---|---|---|---|---|---|---|
| 0 步 | 22.213 | 4.62 | 12.04 | 3.11 | 2.46 | — |
| rgbw0005 | 23.477 | 4.51 | 13.67 | 2.86 | 2.43 | −0.11 / +1.63 / −0.25 / −0.03 |
| rgbw001 | 24.138 | 4.76 | 14.07 | 2.88 | 2.42 | +0.14 / +2.03 / −0.23 / −0.04 |
| rgbw002 | 25.416 | 5.03 | 15.01 | 2.97 | 2.41 | +0.41 / +2.97 / −0.14 / −0.05 |
| rgbw005 | 28.250 | 5.57 | 16.55 | 3.71 | 2.42 | +0.95 / +4.51 / +0.60 / −0.04 |
| rgbw010 | 30.908 | 6.14 | 17.86 | 4.46 | 2.44 | +1.52 / +5.82 / +1.35 / −0.02 |
| rgbw025 | 33.601 | 6.87 | 18.97 | 5.28 | 2.48 | +2.25 / +6.93 / +2.17 / +0.02 |
| rgbw050 | 35.952 | 7.67 | 19.84 | 5.95 | 2.49 | +3.05 / +7.80 / +2.84 / +0.03 |
| full100 | 37.909 | 8.31 | 20.53 | 6.57 | 2.50 | +3.68 / +8.47 / +3.45 / +0.04 |
| rgbw002_c16 | 26.840 | 5.04 | 15.47 | 3.90 | 2.43 | +0.42 / +3.43 / +0.79 / −0.03 |
| cres32 | 39.021 | 8.71 | 19.96 | 7.73 | 2.62 | +4.09 / +7.92 / +4.62 / +0.16 |
| cres16 | 40.891 | 9.35 | 19.81 | 8.99 | 2.74 | +4.73 / +7.77 / +5.88 / +0.28 |

- `rgb-weight` 决定的是**加厚壳层的厚度**：0.02 时增量 +3.20 m² 里 2.97 在 `near`（≤3 voxel），`mid`/`far` 基本不动；1.0 时同一份增量在 `near` +8.47、`opening` +3.68、`mid` +3.45。所有配置的 `far` 都近 0，说明没有新增离群薄片。
- `color-res` 决定的是**中距壳层与开口区**：16³ 把增量推到 `mid` +5.88、`opening` +4.73（两者合计 64% 的增量），是“颜色容量不足 → 中间距离表面被推出去”的特征；32³ 处于中间。
- `rgbw002` 的 `opening` 只增 +0.41 m²，是全部 100 步配置里最接近 0 步的；`sil_only` 的 `opening` 反而 −0.50（关掉 RGB 后开口闭合墙不再长高）。

#### 12.12.7 “最不坏配置”与 0 步基准的差距

`rgbw002`（rgb_w 0.02 / color-res 64 / g160 / 100 步 / sil 0.5 / corr 0.02）：

| 指标 | 0 步基准 | rgbw002 | 差距 | 折合噪声（§12.8.3） |
|---|---|---|---|---|
| F1 @1 voxel | 0.8056 | 0.7636 | **−0.0420** | 30× |
| Chamfer (m) | 0.016142 | 0.015698 | **−0.000444**（名义更好） | 1.8× |
| 法线 (°) | 48.60 | 50.39 | +1.79 | 24× |
| Sharp P | 0.6945 | 0.6254 | −0.0691 | —（未测种子噪声） |
| Sharp R | 0.9357 | 0.8679 | −0.0678 | 34× |
| 面积倍数 | 1.944× | 2.224× | **+0.280×（+14.4%）** | — |
| heldout PSNR | —（0 步颜色未训练） | 20.27 | — | — |

相对 full100 与 full50：

| 对照 | ΔF1 | ΔChamfer | Δ面积 | Δheldout PSNR |
|---|---|---|---|---|
| vs full100（rgb 1.0 / 100 步） | **+0.0989**（71×） | −0.000931（3.8×） | **−1.094×** | **+1.17 dB** |
| vs full50（rgb 1.0 / 50 步） | **+0.0848**（61×） | −0.000524（2.1×） | **−0.743×** | **+1.20 dB** |

必须说清楚的三点：

1. **`rgbw002` 在 F1 上仍明显差于 0 步**（30× 噪声），所以它是“最不坏”，不是“恢复”。优化在本轮**从未超过初始化的 F1**；面积也在 0 步之上 0.280×。唯一名义上优于 0 步的是 Chamfer（1.8× 噪声，属边缘）。
2. 它**同时**优于 full100 与 full50 的几何和外观——说明 full 的 1.0 权重与 100 步这两项本身就是净损失，而不是“用几何换外观”的代价。
3. 它的几何本质是“少动”：步数 100 但 rgb 权重极低，表面基本保留输入形状，颜色场单独拟合外观。因此它不能证明“可微渲染能改善几何”，只能证明“在当前监督形式下，别让 RGB 主导几何是必需的”。

#### 12.12.8 根因判断：折中点是“权重失配 + 正则缺失”的产物，不是光度项形式的固有矛盾

必须敢写的那句话：**“在该目标形式下不存在折中点”这个说法在本轮数据上不成立，也不应该为了交差去写。** 理由与根因如下：

1. **折中点确实存在，而且已经定位**：`rgb-weight = 0.02`（面积 2.224×、heldout PSNR 20.27）。它的左侧（0.02→0.005）才是“几何赚、外观亏”，斜率 −0.189 dB / 0.1×；右侧（0.02→1.0）是“两边都赚”，斜率 +0.107 dB / 0.1×。所以不是“几何与外观单调对立”，而是**单峰**。
2. **0.02→1.0 段的“两边都赚”说明 1.0 不是取舍点，而是过度加权的操作点**。证据：把 RGB 权重从 1.0 降到 0.02，面积 −1.094×、F1 +0.0989、PSNR +1.17 dB、Sharp P +0.2646。若两项真在对立，不可能三项同时朝有利方向走。
3. **根因是几何侧正则缺失（结构性），权重失配只是它的放大器（可调）**：
   - `L = rgb_w·RGB + sil_w·silhouette + corr_w·corr + 0.1·L_dev`。`L_dev` 约束的是**相对初始形变的偏移幅度**，不对表面积、曲率、法线一致性做任何惩罚。于是“沿表面加一层壳”是降低 RGB 残差最便宜的路径（更多面覆盖更多像素，L1 下降而无需对齐），目标函数里没有任何项与之竞争。
   - 光度项本身偏“覆盖型”而非“对齐型”（逐像素 RGB L1，无可见性/多视一致性约束），在反光塑料上大量残差是颜色场（64³ albedo + SH diffuse/specular）解释不了的高频/镜面成分；这些残差只能由几何去背——§12.11.4 的 rank-4/fixed 三格与本节 16³/32³ 两格都是同一个机制的正向证据（颜色容量越差，几何越胖）。
   - 权重上 `rgb 1.0` vs `sil 0.5` 让 RGB 残差在总损失里绝对主导，把上述结构性缺陷放大到 3.318×。
4. **因此正确的说法是**：折中点的位置（0.02）由**权重比例**决定，折中点的“深度”（0.32 dB）由**颜色容量**决定；而膨胀之所以会发生，是因为**缺少面积/曲率/法线正则**，不是因为“光度项只能靠膨胀来降低误差”。光度项形式（L1 RGB）是可修的（改 SSIM/LPIPS 或加可见性加权），正则缺失是不可通过调参绕过的。
5. **这是因果判断，不是已证结论**（必须写明）：要证伪“加正则即可消除膨胀”，需要再做一组“面积/曲率项”或“patch-based 光度项”的对照；本轮没有做，因此本节只把“权重失配 + 正则缺失”作为**最符合现有 12 格数据的解释**，而非定论。

#### 12.12.9 资源、GPU 锁与无效 run

| run | 步数 | 秒 | torch 峰值 (MiB) | SMI delta (MiB) | 面数 | mesh sha256_16 |
|---|---|---|---|---|---|---|
| rgbw010 | 100 | 201.2 | 3,292 | 4,903 | 402,744 | `39feb33464197fd2` |
| rgbw025 | 100 | 201.9 | 3,336 | 4,931 | 442,872 | `e3380fc015e44e30` |
| rgbw050 | 100 | 202.6 | 3,371 | 5,099 | 475,168 | `e8de278288dc41f1` |
| cres16 | 100 | 201.9 | 3,510 | 5,113 | 553,164 | `619a555b9465c429` |
| cres32 | 100 | 202.0 | 3,409 | 4,949 | 519,436 | `93403141254d22e0` |
| rgbw005 | 100 | 201.8 | 3,251 | 4,871 | 364,768 | `6f0f8f0613a08a27` |
| rgbw002 | 100 | 200.4 | 3,230 | 4,883 | 320,724 | `c903a16173fbaaef` |
| rgbw001 | 100 | 200.9 | 3,230 | 6,870 | 299,696 | `2026fe8351a153e6` |
| rgbw0005 | 100 | 200.1 | 3,230 | 4,881 | 288,964 | `494dc9d002a37a55` |
| rgbw002_c16 | 100 | 201.0 | 3,220 | 4,885 | 342,620 | `797a732bc7fabe67` |
| full100（复用） | 100 | 175.4 | 3,423 | 6,205 | 501,268 | `c8cecd09e3426bd6` |
| full50（复用） | 50 | 174.9 | 3,338 | 5,384 | 445,540 | `a1603a6e069147cd` |

- 10 个新 run 合计 **2,013.8 s ≈ 33.6 min** GPU（单 run 200.1–202.6 s，其中约 140 s 是固定的 SDF 建格 + 16 视角可见性 + KDTree 初始化）；评测 10 个 mesh 约 100 s、区域分解 2 轮约 4 min（均为 CPU/Blender，与 GPU 无冲突）。
- **GPU 锁**：全部 10 次 `LOCK_OK waited=0s free=7895 MiB`。纹理线（`scripts/experiments/tex_t1_run.sh`）仍在轮询等 10.5 GB，但**未持锁**，卡上 free 一直是 7,895 MiB ≥ 本线 6,300 MiB 门槛；每次取锁失败即立刻 `flock -u`、3 s 后重试，未把锁握在手里等待，未杀任何进程，未并发跑两个 run。无 OOM、无反压。
- `rgbw001` 的 SMI delta 6,870 MiB 明显高于同组（4,871–4,883），而其 torch 峰值与其他格完全相同（3,230 MiB）；这是 30 ms 采样窗口内其他租户波动造成的读数，不影响本线显存结论（本线真实占用约 3.2 GB compute + context）。
- **无效 run**：**科学 run 无**——10 个 run 全部 `EXIT=0`、全部产出 `mesh.ply` 与 `summary.json`。有 1 次**评测包装脚本**失败（`evaluate.py` 因多传了一个空参数报 `unrecognized arguments`），未产生任何 partial 产物，修正后重跑成功、旧格数据未被覆盖；不构成无效科学 run。
- 步数参照 `full50` 与对照 `full100` 直接复用 §12.9/§12.11 的产物，未重跑、未改动。

#### 12.12.10 边界声明

- 本线仍是**自建 FlexiCubes harness**，不是 nvdiffrec 官方训练方案；不得表述为“nvdiffrec 无法避免膨胀”。
- 几何指标全部相对 **GT-depth 代理 Mesh** 评测，不是官方 benchmark。
- 所有格子只跑 **1 个 seed**；门限沿用 §12.8.3 的 2-seed 估计。**PSNR 没有 2-seed 噪声估计**，因此 12.12.4 里 0.15–0.32 dB 的差异只作观测值报告，不作显著性声明；几何侧（F1 / 面积 / Sharp P）的单调性远超声门限，可作结论。
- **`rgb-weight = 0` 的 PSNR 无意义**（颜色场从未被监督），只用来标出断崖位置，不参与曲线拟合。
- 面积口径为 `0.5·Σ|cross|`（真实面积）；§12.9/§12.10 的 `Σ|cross|` 数字原样保留（12.11.9 已说明）。
- `color-res` 只扫了 16³/32³/64³，**未测 96³/128³**；`rgb-weight` 只扫到 0.005，**未再细分 0.005–0.02**。曲线的两端（颜色容量上限、权重 0.005 以下的滚降）尚未闭合。
- `rgbw002_c16` 是为了确认拐点而补的 2×2 单格，不是完整的交叉网格；其余三格复用既有产物，其评测与区域分解均在**同一口径**下重算，未混用旧评测文件。

#### 12.12.11 结论

1. **拐点 = `rgb-weight 0.02`（面积 2.224× / heldout PSNR 20.27）**。在 0.02–1.0 之间，降权重让几何与外观**同时**变好（面积 −1.094×、F1 +0.0989、PSNR +1.17 dB、Sharp P +0.2646），因此 full 的 1.0 是过度加权的操作点，不是取舍点。
2. **真正的取舍区很窄**：0.02 → 0.005 只掉 0.32 dB，换来面积 −0.169×、F1 +0.0210、Sharp P +0.0897；Chamfer 在 0.02 已到名义最优，再降不再获益。
3. **颜色分辨率轴没有折中**：16³ → 64³ 双输/双赢（面积 3.579× → 3.318×，PSNR 17.10 → 19.10），压分辨率同时伤害几何与外观；每单位面积的 dB 效率是权重轴的 7 倍，说明外观瓶颈主要在颜色容量。
4. **断崖在 0.005 → 0**：`sil_only` 面积回到 1.944× 但 PSNR 11.41，颜色场无监督——这是失效，不是折中。
5. **最不坏可行配置 = `rgbw002`**：F1 0.7636、Chamfer 0.015698、Sharp P 0.6254、面积 2.224×、heldout PSNR 20.27。相对 0 步仍是“几何差 0.0420 F1 / 面积多 0.280×”，相对 full100 则是三维全胜（F1 +0.0989、面积 −1.094×、PSNR +1.17 dB）。
6. **根因：权重失配（可调）+ 几何侧正则缺失（结构性）+ 颜色容量不足（可调）**；不是光度项形式本身必然要求膨胀。该判断需要“面积/曲率正则”或“patch-based 光度项”的对照才能升级为定论。

### 12.13 几何正则对照：面积项与曲率项（2026-11-05）

目标：把 §12.12.8 标为“待验证根因”的那句话（“加壳是降低 RGB L1 的最便宜路径，因为 `L_dev` 只约束偏移幅度、不约束面积/曲率”）升级或推翻。§12.8–§12.12 原样保留，本节只新增。

#### 12.13.1 设置

- 起点与 §12.12 一致：GT-depth 代理 Mesh（`e4006a146e8ae5d6`）、g160、100 步、16 GT 相机、256 渲染、64³ 颜色 + SH、`L_dev` 0.1、`sil` 0.5、`corr` 0.02、seed 20261105，correspondence 复用 `outputs/step_init_20261105/proxy_e100/correspondence.npz`。
- 新增两项（`step3_flexicubes.py`）：
  - **面积项**：`λ_area · (S/S0 − 1)²`，`S = Σ 三角形真实面积`，`S0` = 训练前同一次 `extract(state)` 的面积（22.478 m²，含 §12.9 的扰动）。
  - **曲率项**：`λ_curv · B/B0`，`B = Σ_{共享边} |e| · (1 − cos θ)`（离散弯曲能量），`B0 = 716.2`。
- 四格主扫，全部 `rgb_w = 1.0`：`areg001`(λ_area 0.01)、`areg010`(0.10)、`creg001`(λ_curv 0.01)、`creg010`(0.10)。λ 跨一个量级，用实际项值（`reg.area_x`、`reg.curv_rel`）标定，不按名义权重。
- 条件补格：面积项在 λ=0.10 时把面积压到 2.010×（0 步为 1.944×）且 PSNR 19.37 ≥ 19 → 触发第 5 格 **`r002_areg`**（`rgb_w 0.02 + λ_area 0.10`）。共 5 格，未做无效追加。

#### 12.13.2 结论：根因“升级一半、推翻一半”，几何改善被否定

1. **升级（面积部分已验证）**。在 `rgb_w=1.0` 下单独加面积项：`3.318× → 2.010×`（λ 0.10），同时 **PSNR 19.10 → 19.37**、Chamfer 0.016629 → **0.015202**（5.8× 噪声）、法线 53.67° → **47.56°**。壳变薄没有付出外观代价，反而更好。→ “膨胀由缺少面积约束造成”成立；光度项形式本身不必然要求膨胀。
2. **推翻（曲率不是同一机制）**。单独加曲率项不做修复，只做收缩/平滑：面积被压到 `1.704×`（λ 0.01）甚至 `1.465×`（λ 0.10），**低于 0 步的 1.944×**；Chamfer 恶化到 0.017062 / 0.018067；Sharp Recall 掉到 0.8065 / 0.7606。→ “曲率缺失”不是膨胀的主因；用弯曲能量压面积会把表面缩进 GT 内部。
3. **几何质量没有被正则改善**。5 格的 F1 最高 **0.7842**（`r002_areg`），仍低于 0 步的 0.8056；Sharp Recall 全部低于 0 步 0.9357。面积项能把壳压回初始化水平，但压回去之后也就停在初始化水平——**壳可被压制，几何没有因此变好**。
4. **没有任何配置优于 0 步基准**（F1 0.8056 / Chamfer 0.016142）：F1 无一格超过；Chamfer 有 4 格名义更好（0.0152–0.0157，1.8–5.8× 噪声），但都不伴随 F1 / Sharp Recall 的改善。见 12.13.4。
5. 因此判定：**该目标形式在本 harness（自建 FlexiCubes + 本次 SDF 初始化）内收口**——优化能把损坏降到接近初始化，但无法产出优于初始化的几何。不再追加扫参。

#### 12.13.3 正则格表（g160，100 步）

| run | rgb_w | λ_area | λ_curv | 面积× | 面积 m² | F1 | P | R | Chamfer | 法线 | Sharp P / R | held PSNR | train PSNR | 面数 | 秒 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 步（基准） | — | — | — | 1.944× | 22.213 | **0.8056** | 0.7043 | 0.9408 | 0.016142 | 48.60 | 0.6945 / 0.9357 | — | — | — | — |
| full100（无正则） | 1.00 | 0 | 0 | 3.318× | 37.909 | 0.6647 | 0.5507 | 0.8384 | 0.016629 | 53.67 | 0.3608 / 0.8556 | 19.10 | 21.88 | 501,268 | 175.4 |
| areg001 | 1.00 | 0.01 | 0 | 2.216× | 25.321 | 0.6624 | 0.5707 | 0.7891 | 0.015696 | 48.63 | 0.3898 / 0.8108 | 19.22 | 22.25 | 339,384 | 182.5 |
| areg010 | 1.00 | 0.10 | 0 | **2.010×** | 22.964 | 0.6800 | 0.6058 | 0.7748 | **0.015202** | **47.56** | 0.4420 / 0.7982 | 19.37 | 22.39 | 305,120 | 201.4 |
| creg001 | 1.00 | 0 | 0.01 | 1.704× | 19.469 | 0.7332 | 0.6740 | 0.8039 | 0.017062 | 44.47 | 0.6423 / 0.8065 | **19.90** | 23.16 | 229,472 | 200.2 |
| creg010 | 1.00 | 0 | 0.10 | 1.465× | 16.743 | 0.7501 | 0.6998 | 0.8082 | 0.018067 | 43.96 | **0.7447** / 0.7606 | 18.49 | 20.60 | 181,432 | 197.2 |
| **r002_areg** | **0.02** | 0.10 | 0 | 1.968× | 22.490 | **0.7842** | **0.7077** | 0.8792 | 0.015697 | 48.91 | **0.7015 / 0.8711** | 20.19 | 22.81 | 287,708 | 200.4 |

实际正则项终值：`areg001` 面积 1.126 S0；`areg010` 1.021 S0；`creg001` B/B0 = 0.426；`creg010` B/B0 = 0.140；`r002_areg` 1.000 S0。

读法：

1. **面积项单调可控**：λ 0.01 → S/S0 1.126，λ 0.10 → 1.021；λ=0.10 的几何几乎回到 S0（1.944×），而面数从 501k 降到 305k。λ 已到“能把壳完全压住”的量级，无需再往上扫。
2. **`areg001` 是“半压”**：面积 2.216× 与 `rgbw002`（2.224×）几乎相同，但 F1 0.6624 远低于 `rgbw002` 的 0.7636——说明“面积相同”不等于“几何相同”，`rgb_w=1.0` 的残差仍把表面推向错误位置，只是被面积项挡住了体积增长。
3. **曲率项效果是收缩而非校正**：面积 1.704× / 1.465×，低于初始化，Chamfer 反而变差。`creg010` 的 Sharp P 0.7447 是全部 100 步配置里最高（比 0 步 0.6945 还高），但这是“表面被磨平后锐边占比上升”的假象，Sharp Recall 同时掉到 0.7606。

#### 12.13.4 与 0 步基准、`rgbw002` 的对比

相对 0 步（面积 1.944×、F1 0.8056、Chamfer 0.016142、SharpP 0.6945、SharpR 0.9357）：

| 配置 | Δ面积 | ΔF1 | ΔChamfer（负=更好） | ΔSharp P | ΔSharp R | Δheld PSNR |
|---|---|---|---|---|---|---|
| `areg010` | +0.066× | −0.1256 | **−0.000940**（3.8×） | −0.2525 | −0.1375 | — |
| `creg001` | −0.240× | −0.0724 | +0.000920（更差） | −0.0522 | −0.1292 | — |
| `creg010` | −0.479× | −0.0555 | +0.001925（更差） | +0.0502 | −0.1751 | — |
| **`r002_areg`** | **+0.024×（+1.2%）** | **−0.0214（15× 噪声）** | **−0.000445（1.8×）** | **+0.0070** | −0.0646（32×） | — |

相对 `rgbw002`（2.224× / 0.7636 / 0.015698 / 0.6254 / 0.8679 / 20.27）：

| 配置 | Δ面积 | ΔF1 | ΔChamfer | ΔSharp P | Δheld PSNR |
|---|---|---|---|---|---|
| `r002_areg` | **−0.256×** | **+0.0206（15×）** | −0.000001 | **+0.0761** | −0.08 dB（未测噪声） |
| `areg010` | −0.214× | −0.0836 | **−0.000496（2×）** | −0.1834 | −0.90 dB |

`r002_areg` 是唯一在几何与外观上同时不劣于 `rgbw002` 的格：面积回到初始化水平、F1 +0.0206、Sharp P +0.0761，PSNR 只低 0.08 dB（远小于 §12.12 里 0.15–0.32 dB 的可观测差异，且无 2-seed 估计）。它没有超过 0 步，因为 F1 差 0.0214、Sharp R 差 0.0646。

注意 `r002_areg` 的 F1 损失**几乎全在 recall**：precision 0.7077（略高于基准 0.7043），recall 0.8792（基准 0.9408）。也就是说它不再“多盖”，代价是漏掉 6% 的观测表面；这正是“壳被压制”的直接后果，不是新的几何错误。

#### 12.13.5 区域分解

| 配置 | opening | near | mid | far | Δ vs 0 步（opening / near / mid / far） |
|---|---|---|---|---|---|
| 0 步 | 4.62 | 12.04 | 3.11 | 2.46 | — |
| full100 | 8.31 | 20.53 | 6.57 | 2.50 | +3.68 / +8.47 / +3.45 / +0.04 |
| areg001 | 4.12 | 15.82 | 3.93 | 1.45 | −0.50 / +3.78 / +0.82 / **−1.01** |
| areg010 | 3.57 | 14.68 | 3.34 | 1.37 | −1.05 / +2.64 / +0.23 / **−1.09** |
| creg001 | 3.81 | 11.53 | 2.15 | 1.98 | −0.81 / −0.51 / −0.96 / −0.48 |
| creg010 | 3.08 | 9.77 | 1.91 | 1.98 | −1.54 / −2.27 / −1.20 / −0.48 |
| r002_areg | 4.17 | 13.30 | 2.65 | 2.37 | −0.45 / +1.26 / −0.46 / −0.09 |

- **面积项优先削 `far`**：full100 的 `far` 已经只有 2.50（近 0 步），但 `areg001`/`areg010` 把它压到 **1.45 / 1.37**——低于 0 步 1.01–1.09 m²。这部分是初始化里距离输入 >10 voxel 的薄片/尾部，被面积项当成“不产生像素收益的体积”削掉了。**这是一处过度删除**：0 步已在 `far` 只有 2.46，基准里这些面属于 GT 表面的一部分（Completeness），删掉会伤 recall（见 12.13.4 的 recall 下降）。
- `areg010` 同时把 `opening` 从 4.62 压到 3.57（−1.05），说明 λ=0.10 已强到会削开口区的真实表面。
- 曲率项的删除是**面状**的：`near` 从 12.04 掉到 11.53 / 9.77，`mid` 3.11 → 1.91，是整体收缩而非某一区域。
- `r002_areg` 的分布最接近 0 步（−0.45 / +1.26 / −0.46 / −0.09），总量 +0.277 m²。它是“面积项把 full 的 `near` 壳（+8.47）压回到 +1.26”的结果。

#### 12.13.6 为什么“壳压回去了，几何却没变好”

1. **目标函数的几何最优点就在初始化附近**。所有让表面多动的配置（`rgb_w=1.0` 的 full100、颜色容量不足的 `cres16/32`、曲率项把面磨平后的 `creg010`）F1 都低于 0 步；唯一接近 0 步的 `r002_areg` 也是“几乎不动”。即：**优化没有把表面推向 GT，只是被约束在初始表面附近**。
2. **`L_dev` 不是没用，而是方向不对**。它约束相对初始形变的偏移幅度，属于“锚定”而不是“形状先验”；在没有形状先验时，光度残差只能靠加壳来消化（full100），或在锚定足够强时保持不动（`r002_areg`）。
3. **面积项是“刹车”，不是“方向盘”**。它能把 3.318× 拉回 2.010×，但不会把面推到正确位置；`areg001`（面积 2.216×、F1 0.6624）与 `rgbw002`（面积 2.224×、F1 0.7636）面积几乎相同、F1 差 0.1012（72× 噪声），说明两者的差别在位置而非体积。
4. **曲率项方向错误**：弯曲能量最小的形状是球/平面，压 `B/B0` 会把表面收缩（1.465×）并磨平锐边（Sharp R 0.7606），这与“恢复真实表面”背道而驰。
5. **必须点明的竞争解释**：本次初始化是 **GT-depth 代理 Mesh**，其 F1 已经贴着评测来源（0.8056）。也就是说“优化无法超过初始化”有一部分是**结构性天花板**，不能单独归因于目标形式。要区分这两者，必须换 VGGT 初始化重跑（§12.8 Level 3 做过，但受不同混淆项影响），本轮不做。

#### 12.13.7 资源与无效 run

| run | 秒 | torch 峰值 (MiB) | SMI delta (MiB) | 面数 | mesh sha256_16 |
|---|---|---|---|---|---|
| areg001 | 182.5 | 3,255 | 4,905 | 339,384 | `56229368a74dc55e` |
| areg010 | 201.4 | 3,252 | 4,801 | 305,120 | `3c3200f6d2afd31f` |
| creg001 | 200.2 | 3,262 | 4,901 | 229,472 | `8e5d38b7927779b3` |
| creg010 | 197.2 | 3,266 | 4,761 | 181,432 | `4ab9ef2dec663524` |
| r002_areg | 200.4 | 3,253 | 5,001 | 287,708 | `11b3fe963797bdd5` |

- 5 个 run 合计 **981.7 s ≈ 16.4 min** GPU；评测 5 个 mesh ≈ 55 s、区域分解 ≈ 3 min（CPU/Blender，与 GPU 不冲突）。
- **GPU 锁**：5 次全部 `LOCK_OK waited=0s free=7895 MiB`；短持锁轮询，取锁失败立即释放，未杀进程、未并发、无 OOM。纹理线仍在轮询 10.5 GB 但始终未持锁。
- **无效 run**：无。5 个 run 全部 `EXIT=0`，全部产出 `mesh.ply` / `summary.json`。有一次取锁前 tmux 重定向失败（日志父目录尚未创建，`No such file or directory`），未启动 Python、无产物，补建目录后正常启动——不构成无效科学 run。
- 曲率项每步多一次 3F 边排序（约 1.5M key）；`creg` 两格耗时 200.2 / 197.2 s，与无正则格（200.4 s）无显著差异，未造成显存增量（torch 峰值 3,262–3,266 MiB）。

#### 12.13.8 边界声明

- 结论只限**自建 FlexiCubes harness + 本次 GT-depth 代理 SDF 初始化**；不得表述为“nvdiffrec / ExMesh 无法重建几何”。
- 代理初始化与评测 GT 同源（GT depth），F1 天花板被人为抬高；“无法超过初始化”不能全归因于目标形式，见 12.13.6 第 5 条。
- 面积项 `S` 用真实三角形面积（`0.5·Σ|cross|`）；`S0` 含 §12.9 的扰动，因此 `S/S0` 与“相对输入面积”不是同一分母（0 步 = 1.944× 输入 = 0.988 S0）。
- 曲率项是 `Σ|e|(1−cosθ)` 的**总弯曲能量**，不是逐顶点 `|κ|`；它对尺度敏感，已用 `B0` 归一化，但 λ 的物理意义仍依赖网格密度。
- 每格 1 个 seed；PSNR 无 2-seed 噪声估计。
- 只做了 `rgb_w = 1.0` 的正则扫参 + 1 格组合；未做 `rgb_w` 与 λ 的完整交叉。

#### 12.13.9 结论

1. **根因“升级一半”**：面积正则独立把 full 的 3.318× 压到 2.010×，同时 PSNR +0.27 dB、Chamfer −5.8× 噪声、法线 −6.1°。→ “膨胀源于缺少面积约束”得到验证，不是光度项形式的必然。
2. **根因“推翻一半”**：曲率正则不是同一机制，它把面积压到 1.70×/1.47×（低于初始化），Chamfer 恶化，Sharp Recall 掉到 0.76–0.81 → 过平滑/收缩，不能作为修复项。
3. **几何没有被改善**：5 格 F1 最高 0.7842 ≤ 0.79，全部低于 0 步 0.8056；Sharp Recall 全部低于 0.9357。**壳可被压制，但几何没有因此变好。**
4. **本轮没有任何配置优于 0 步基准**：F1 无一格超过 0.8056；Chamfer 有 4 格名义超过 0.016142（0.0152–0.0157，1.8–5.8× 噪声），但均不伴随 F1 改善，不作为“几何改善”的证据。
5. **几何线在本 harness 内收口**：目标函数的几何最优点就是初始表面；在该目标形式下继续扫参不会产生优于初始化的几何。要突破必须改目标形式（patch-based / 可见性加权光度项、形状先验）或改表示/初始化，且需先排除“代理初始化天花板”这一混淆项。

### 12.14 竞争解释排除：VGGT 初始化上的同一格（2026-11-05）

§12.13.6 第 5 条把“代理初始化与评测 GT 同源、F1 天花板被抬高”列为竞争解释。本节只做一格格子来排除它：把 `areg010` 的配置**原样平移到 VGGT 初始化**上。§12.13 的数字一律不改。

#### 12.14.1 判定口径（预注册，跑之前写下）

对照点**只用同一个 VGGT 输入的 0 步**（`outputs/step_init_20261105/vggt_e0`：F1 **0.607472**、Chamfer **0.020024646**、面积 **1.556×**（47.726/30.685，两者同口径）、法线 **51.158°**、Sharp P/R **0.388136 / 0.645848**），不用代理 0 步。噪声门限沿用 §12.8.3：F1 ±0.0014、Chamfer ±0.000246、法线 ±0.075°。

- **A｜F1 > 0.607472 + 0.0014（≥0.6089）**：目标形式本身能改善几何 → “无法超过初始化”是代理同源造成的假象，**几何线不能收口**，据此重开完整对比。
- **B｜F1 持平（|ΔF1| ≤ 0.0014）或更低，但面积/Chamfer/法线任一项优于 0 步**（面积向 1.0× 收敛 ≥0.1×、或 Chamfer 改善 > 0.000246、或法线改善 > 0.075°）：**“目标能约束形状但未能提升 F1”**，仍是收口，但把限制条件写进结论。
- **C｜全面劣于 0 步**（F1 ≤ 0.6059 且面积/Chamfer/法线均无改善）：**“该目标形式在 VGGT 初始化上同样无收益”**，收口判断升级为对**目标形式**成立，而不只是对代理初始化成立。

预注册时的预期（诚实记录）：面积正则的作用是压缩膨胀，不产生定位能力，因此更可能是 B；若落在 A 则必须重开对比。

#### 12.14.2 设置（与 `areg010` 完全一致，只换初始化）

- 输入：VGGT Mesh `outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply`（sha256_16 `ecf994efd40a5d9c`），不是代理 Mesh。
- g160、100 步、渲染 256、16 GT 相机、64³ 颜色 + SH、`rgb_w 1.0`、`sil 0.5`、`corr 0.02`、`L_dev 0.1`、`λ_area 0.10`、`λ_curv 0`、seed 20261105、冻结 voxel。
- correspondence 必须在该网格上重建（`corr_index` 是顶点索引，不能复用代理的 npz），因此不传 `--corr-file`。
- 产物 `outputs/step_reg_vggt_20261105/areg010_vggt/`，日志同目录 `logs/`。



