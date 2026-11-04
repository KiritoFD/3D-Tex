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

