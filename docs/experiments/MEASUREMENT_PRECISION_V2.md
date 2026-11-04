# 采集精度测量 v2 — R0 前置（重复性地板与零误差场景）

本轮只做方案 R0（P0.1 / P0.2 / 冻结新零噪声基线），v1 的
[MEASUREMENT_TOLERANCE.md](MEASUREMENT_TOLERANCE.md) 与其产物全部保留作对照，v2 写入独立目录
`outputs/measurement_error_v2/`、`data/scenes/measurement_error_v2/`。

## 0. 结论先行

1. **M2 不能确定化**：`torch.use_deterministic_algorithms(True)` 在第一个反传直接抛
   `grid_sampler_3d_backward_cuda does not have a deterministic implementation`，因此 P0.1 走
   "建立稳定重复性分布 + 明确只能报告大于地板的退化"这条路，而不是逐字节复现。
2. **首次分叉点就在第 1 步**：10 次同输入/同 seed 运行，梯度、参数、lattice offset 的 float32 哈希
   **没有一步相同**（`identical_parameter_steps = 0/300`，`first_*_divergence_step = 1`）；
   第 1 步 24 个 depth + 24 个 alpha 渲染输出（48 个哈希）全部不一致。原因已定位到算子级：
   nvdiffrast antialias 的前/反向 + 多个 `index_add` 类原子累加，逐像素 1 ulp 级差异被 Adam 放大。
3. **重复性地板（v2 零噪声 10 次）**：可观察 Chamfer **1.0799 ± 0.0059 mm**（2b 验收地板
   0.0443 mm = 4.10%），全表面 Chamfer 2.0891 ± 0.0173 mm，法线 51.177 ± 0.123°（地板 0.82°），
   F@1mm(observed) 0.6310 ± 0.0019（地板 1.90%），终损 3.0797 ± 0.0119；顶点相对均值散布
   mean/p90/p99/max = 1.372 / 2.580 / 4.565 / 18.532 mm。
   **结论：小于 4.1%（Chamfer）/ 0.82°（法线）/ 1.9%（F）的差异不可解读，不能报亚毫米容差。**
4. **零误差场景**：几何侧逐字节一致 —— 源 Mesh 与 24 张 GT 深度均与冻结清单 SHA256 相同，
   冻结扫描在 rms=0 时输出的深度与 GT 深度**逐字节相等**；RGB 侧 22/24 帧不是逐字节一致，
   但全部 24 帧 alpha 完全相同，RGB 只有 **107 / 15,360,000 像素（7.0e-6）差 1 个 8-bit 量化级**，
   最大差值 1 LSB。即：**几何零误差成立，RGB 只存在量化边界抖动**。
5. **新零噪声基线已冻结**（§3），R1/R2 的所有比值与门限都以它为准。

## 1. P0.1 重复性

### 1.1 等价条件（每次重复之间唯一变量 = 进程实例）

同一 `initial_source.ply`（SHA `75f78dff…3927`）、同一 `initialization.json`
（`6c30f37b…aee`）、同一 omega 导出清单（`053af974…a2c1`）、同一零噪声扫描
（`8242795a…53e7`，24 个 `depth/r_*.npy` 逐个哈希相同）、同一代码（`scripts/mesh_opt/*.py`
五个文件哈希）、同一 commit（`671a111`）、同一优化器 seed `20261103`。`run.json` 在开跑前
写入并在后续 rep 中校验输入未变，任何输入漂移会直接报错而不是静默继续。

**优化器重复噪声与输入扰动种子是两条独立通道**：重复性实验只改进程，扫描 seed 固定 42；
后续 R1/R2 的输入扰动只改扫描噪声 seed（42/43/44/…），优化器 seed 恒为 20261103。

### 1.2 算子级探针（GPU 串行，`determinism_probe.py`）

torch 2.14.1+cu126 / RTX 4070 Laptop，同一 tensor 在同一进程内连续两次前向或反传：

| 探针 | 两次完全相同 | 最大绝对差 | 说明 |
|---|---|---|---|
| `index_add_forward` | ✗ | 2.38e-07 | 图正则 `regularizer_graph`、位移项每步都用 |
| `index_add_sorted_cumsum` | ✓ | 0.0 | 顺序无关求和替代（与 `index_add_` 差 4.93e-05） |
| `grid_sample_forward` | ✓ | 0.0 | `lattice_offsets` |
| `grid_sample_backward` | ✗ | 7.15e-07 | 官方文档标注的非确定性 scatter-add 反传 |
| `cublas_matmul` | ✓ | 0.0 | 固定 workspace 下约简顺序稳定 |
| `adam_step` | ✓ | 0.0 | 逐元素更新，本应确定 |
| `rasterize_interpolate_forward`（无 antialias） | ✓ | 0.0 | nvdiffrast 前向本身确定 |
| `render_depth_forward`（含 antialias） | ✗ | 4.53e-06 | antialias 引起 |
| `render_alpha_forward`（含 antialias） | ✗ | 2.98e-08 | antialias 引起 |
| `render_backward`（含 antialias） | ✗ | 6.10e-05 | antialias + 插值反传的原子累加 |
| `render_backward_without_antialias` | ✗ | 4.77e-07 | 去掉 antialias 仍不确定 → 原子累加 |

即：**前向光栅化本身是确定的，不确定的只有 antialias 前/反向与 scatter-add 反传**。
把 antialias 拿掉能把反传误差从 6.1e-05 降到 4.8e-07，但仍不为零。
`sorted-cumsum` 替代 `index_add_` 可以做到完全确定，但会改动图正则的数值（4.93e-05），
等于更换基线，R0 不做这个改动，留给后续单独评估。

### 1.3 端到端分叉定位

`repeatability.py` 用子进程逐个起 300 步优化，`--trace-every 1` 记录 **每一步** 的
`grad_sha256 / param_sha256 / offset_sha256`（float32 字节级 SHA256），第 1 步额外记录
24 视图 depth/alpha 的逐视图哈希。

| 观察 | 结果 |
|---|---|
| 首次梯度分叉 | **第 1 步**（9 个对比 rep 全部） |
| 首次参数分叉 | **第 1 步** |
| 首次 offset 分叉 | **第 1 步** |
| 300 步中参数一致的步数 | **0 / 300** |
| 第 1 步渲染输出不一致 | 48 / 48（24 depth + 24 alpha） |

因为第 1 步前的状态逐字节相同，所以分叉点在**第一次前向/反传**，与优化器、累积步数无关。

### 1.4 确定性模式的失败原文

```
RuntimeError: grid_sampler_3d_backward_cuda does not have a deterministic implementation,
but you set 'torch.use_deterministic_algorithms(True)'
```

`--deterministic` 已实现为 `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
`torch.use_deterministic_algorithms(True)` + cudnn deterministic / benchmark=False，但只要 lattice
形变还在用 `grid_sample` 就走不通。日志：`outputs/measurement_error_v2/repeatability/v1_det/rep01.log`。

### 1.5 稳定重复性分布（v2 零噪声 10 次，tag `zero_v2`）

地板定义沿用 v1：`b = max(2σ, 全距)`，验收地板 `2b`（单向劣化不超过 2 倍重复性波动）。

| 指标 | 均值 | σ | min | max | p05 | p95 | 2b 地板 | 相对 2b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 可观察 Chamfer mm | **1.0799** | 0.0059 | 1.0664 | 1.0886 | 1.0711 | 1.0879 | **0.0443** | 4.10% |
| 全表面 Chamfer mm | 2.0891 | 0.0173 | 2.0549 | 2.1171 | 2.0588 | 2.1091 | 0.1244 | 5.96% |
| 法线（总）° | 51.177 | 0.1225 | 51.012 | 51.421 | 51.015 | 51.372 | 0.8177 | 1.60% |
| 法线（平）° | 50.058 | 0.1461 | 49.899 | 50.318 | 49.904 | 50.300 | 0.8374 | 1.67% |
| 法线（边）° | 54.663 | 0.2323 | 54.234 | 55.000 | 54.276 | 54.963 | 1.5319 | 2.80% |
| F@1mm（observed） | 0.6310 | 0.0019 | 0.6289 | 0.6349 | 0.6290 | 0.6341 | 0.0120 | 1.90% |
| Precision@1mm | 0.6506 | 0.0034 | 0.6457 | 0.6567 | 0.6462 | 0.6555 | 0.0219 | 3.37% |
| Recall@1mm | 0.2874 | 0.0016 | 0.2849 | 0.2906 | 0.2852 | 0.2898 | 0.0114 | 3.95% |
| thin_recall@1mm | 0.1452 | 0.0048 | 0.1362 | 0.1523 | 0.1377 | 0.1508 | 0.0323 | 22.2% |
| 可观察 P90 mm | 1.8811 | 0.0058 | 1.8709 | 1.8899 | 1.8726 | 1.8887 | 0.0380 | 2.02% |
| 可观察 P99 mm | 3.4680 | 0.0432 | 3.4050 | 3.5642 | 3.4138 | 3.5351 | 0.3185 | 9.18% |
| 全表面 P90 mm | 3.1131 | 0.0429 | 3.0365 | 3.1715 | 3.0513 | 3.1695 | 0.2701 | 8.68% |
| 全表面 P99 mm | 8.2104 | 0.1258 | 8.0613 | 8.4491 | 8.0625 | 8.4256 | 0.7756 | 9.45% |
| hole_fill | 0.5814 | 0.0084 | 0.5637 | 0.5990 | 0.5689 | 0.5931 | 0.0705 | 12.1% |

顶点侧（同拓扑，不做任何对齐）：

| 量 | mean | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| 相对 10 次均值散布 mm | **1.372** | 1.156 | 2.580 | 4.565 | 18.532 |
| 相对 rep01 位移 mm（9 个 rep 的范围） | 1.999–2.070 | 1.688–1.726 | 3.778–3.930 | 6.567–7.118 | 13.954–19.495 |

终损 3.0797 ± 0.0119（3.0620–3.0980）。**终损的标准差 0.39% 比多数被测档位的损失变化还大**，
所以 R1/R2 一律用表面指标判定，不用损失比较档位。

### 1.6 解读与硬限制

- 地板以上的差距才可报：Chamfer ≥ 4.10%、法线 ≥ 0.82°、F ≥ 1.90%、hole_fill ≥ 12.1%。
  亚毫米（0.1 mm 级）的 Chamfer 差在 M2 上**物理上不可分辨**。
- `thin_recall` 的 2b 高达 22%（0.0323 绝对），只能报"没有结构性退化"，不能报小幅改善。
- 顶点 max 位移 18.5 mm 说明少数细窄/薄结构位置的**拓扑级**抖动（不是测量误差），R1/R2 报
  薄结构结论时必须附 hole_fill / thin_recall，不能只看 Chamfer。

## 2. P0.2 零误差场景

### 2.1 冻结清单与 SHA256

全部从同一份 Blender 源 Mesh 派生：`source_mesh.npz` 顶点/面在渲染前被 `np.array_equal`
校验等于 master 冻结副本，Blender 里的 modifier / constraint / parent 全部清空后再渲。

| 产物 | 文件级 SHA256 |
|---|---|
| `master/source_mesh.npz` | `703b24c3c183e385d851307a92d8ae78eea1f12fb0a95c3014b537b13991ef81` |
| `master/source_manifest.json`（24 个 GT 深度 + 相机 + 尺度/协议） | `0eb2189bd78da067e05357e455859aa9d16bfc479cb7f07e5a77fb18fc87f3d1` |
| `master/geometry/gt_ray/cameras.json` | `eadc8e9c698c3d6accdd7c7546dfa9e81db2c1a218ab95f7ca6cf5fe284fa388` |
| `rgb/zero/perturbation.json`（逐图输入 SHA） | `ce9893499e250c930a9df08f530aa41e0d86d5467cccef3f3fb55fd53efa8f2a` |
| `rgb/zero/contract.json` | `2be8b0bfb6718ad51b229f1dd3bd797c9a33b0f149809c50cdee492fcce09764` |
| `rgb/zero/rig.json`（24 相机 + 逐帧渲染 SHA） | `a087bf36fdc773251ad0c4116b74098da567329a70734c1df1e41a8991385a43` |
| `scans/…/iid_rms0_s42/manifest.json` | `8242795ad4cb61e8962cee1d76a5ec14063422af7112d98463c8ba5834bf53e7` |

- 尺度：`mm_per_source_unit = 125.83285964739473`（按 8 mm 砖距标定），物距 363–655 mm。
- 冻结参数：24 个 GT 相机、800² 分辨率、Cycles OptiX、256 samples、seed 42、无降噪、tile 128、
  `film_transparent`；每帧约 2.6–7.6 s，24 帧合计 ≈100 s。
- **几何零误差验证**：rms=0 的扫描深度与 `master/geometry/gt_ray` 的 24 个 `r_*.npy`
  逐字节相等（回归测试 `test_zero_rms_scan_reproduces_gt_depth_bytes` 直接比 SHA）。

### 2.2 RGB 重渲染的复现性（第二次全量重渲）

同一 `.blend` + 同一帧参数 + 同一 seed，在新目录全量重渲 24 帧后与首轮比较（解码后的 8-bit RGBA）：

| 量 | 结果 |
|---|---|
| 逐帧解码像素 SHA 完全相同 | 2 / 24 |
| RGB 差异像素数 | **107 / 15,360,000**（7.0e-6） |
| 单像素最大差 | **1 LSB**（所有帧 ≤ 1） |
| alpha 差异像素数 | **0**（24 帧全部逐字节相同） |
| 前景像素数 | 每帧完全相同（如 `r_0` 149,945） |

即 RGB 只在 8-bit 量化边界翻 1 个等级，掩膜与几何完全不变。**这解释了 v1 中"B 类（重渲 RGB）
零噪声对照自带 0.1283 mm Chamfer 带"的来源**：重渲不可字节复现，所以 RGB 分支的零噪声控制组
不能借用几何分支的窄带，必须自带 5 seed 控制带。

### 2.3 相机/投影契约

`contract.json`（24 帧，模型分辨率 512、渲染 800）：

| 检查 | 结果 |
|---|---|
| alpha 下采样到 512 与 GT 前景 IoU | min **0.9843** / mean **0.9943** |
| 前景不一致率 | max **1.57%** |
| 深度反投影往返误差 | max **2.27e-13 px** |
| 像素约定 | 整数像素中心，`cx=(W-1)/2`，光栅化 `+0.5`（与 v1 协议一致） |
| 内参换算 | `K_model` 按 512、`K_original` 按 800，`pixel_affine_original_to_model = [[ratio,0,(ratio-1)/2]]` |

IoU < 0.95 或往返 > 0.5 px 时 `zero_scene.py --phase audit` 直接抛错，不会静默产出。

### 2.4 理想 / 非理想标定模式（R1 的接口，本轮只建不用）

`calibration.py` 把两件事彻底分开：

- **ideal**（默认，本轮与 v1 全量数据都用它）：GT 位姿注册世界，任何非零注入直接
  `raise ValueError`。
- **non_ideal**：先做**一次**固定世界注册（`align_poses`，与 `verify_scene.umeyama` 位级一致，
  实测 max diff 0.0），把该 registration 写成 `"locked"`，**之后才注入**平移/旋转/内参误差；
  下游 `mesh_opt/inputs.py` 只用注入后的 `T_world_to_cam`，**不会再用 GT 相机重新拟合 Sim(3)**。
  prediction-only 路径带 `--calibration` 直接报错。

三种误差定义在代码里分开并各自有单位测试：图像平面定位位移 px、相机中心平移 mm、
轴向深度/整图共享偏差 mm。`0.1° @470 mm = 0.8206 mm` 的换算已被测试钉死。

## 3. 冻结的新零噪声基线（与 v1 对照）

| 指标 | v1（6 次，`671a111` 前） | v2（10 次，本轮） | 差 |
|---|---:|---:|---:|
| 可观察 Chamfer mm | 1.0833 ± 0.0043 | **1.0799 ± 0.0059** | −0.31% |
| 全表面 Chamfer mm | 2.0680 ± 0.0175 | 2.0891 ± 0.0173 | +1.02% |
| 法线（总）° | 51.796 | 51.177 ± 0.123 | −0.62° |
| 法线 平/边 ° | 50.88 / 54.66 | 50.058 / 54.663 | 平 −0.82° |
| P / R @1mm | 0.653 / 0.295 | 0.6506 / 0.2874 | R −0.008 |
| F@1mm = 2PR/(P+R) | 0.4061 ± 0.0024 | 0.3987 | −1.8% |
| F@1mm（observed 口径） | — | 0.6310 ± 0.0019 | 新增 |
| hole_fill | 0.6333 ± 0.0082 | 0.5814 ± 0.0084 | −8.2% |
| 顶点相对位移 mean mm | 2.14（3 rep） | 2.00（10 rep） | — |
| 终损 | 3.407–3.437 | 3.080 ± 0.012 | 不可比 |

- **两种 F 必须写明口径**：`fscore@1mm = 2PR/(P+R)`（含未观测的 GT 完整性，数值偏低），
  `observed_fscore@1mm`（用"观测点到 GT ≤1 mm"的比例，更接近设备侧表面重合率）。
  v1 文档用的是前者；本文件 §4 的门限用后者，跨版本比较时只能同口径比。
- 终损不可跨基线比较：v2 的 RGB/omega 是从同一 Mesh 重渲的一致版本，v1 的 omega 来自另一套
  渲染，损失绝对水平没有可比性。
- hole_fill 下降 8.2%（0.633→0.581）超出了本轮的 2b 地板（12.1%）以内，属"落在地板内"，不作结论。

## 4. 先冻结的判据（R1/R2 直接套用，不再看数据后调整）

以 §1.5 的地板与 §3 的基线为准，**任何档位在跑之前就按这张表判**：

| 指标 | 门限（相对 v2 零噪声基线） | 绝对值 | 是否高于地板 |
|---|---|---:|---|
| 可观察 Chamfer | ≤ 1.10× | ≤ 1.1879 mm | 是（地板 4.10%） |
| 全表面 Chamfer | ≤ 1.10× | ≤ 2.2980 mm | 是（地板 5.96%） |
| F@1mm（observed） | ≥ 0.95× | ≥ 0.5994 | 是（地板 1.90%） |
| Recall@1mm | 绝对降 ≤ 0.02 | ≥ 0.2674 | 是（地板 0.0114） |
| Precision@1mm | ≥ 0.95×（附加） | ≥ 0.6181 | 是（地板 0.0219） |
| 法线（总） | 增加 ≤ 2° | ≤ 53.177° | 是（地板 0.82°） |
| hole_fill | 无结构性退化（≤ 2b） | 变化 ≤ 0.0705 | — |
| thin_recall@1mm | 无结构性退化（≤ 2b） | 变化 ≤ 0.0323 | — |

规则：

1. **单向判罚**：只有"变差"方向计入（Chamfer/法线增大、F/Recall 减小）。
2. **每档先 5 seed**；落在门限 ± 地板以内（临界档）的加到 ≥10 seed。
3. **成对样本**：同一种随机误差跨档位复用同一组随机场（`index` 只缩放，不重抽），
   避免档位间差异被随机种子差异污染。
4. 任何小于 2b 地板的差异**不得**被解释为采集精度效应，只能记为"未分辨"。

## 5. 工程改动与测试

| 文件 | 作用 |
|---|---|
| `configs/experiments/measurement_error_v2.json` | v2 唯一配置入口；不覆盖 v1 产物；**故意不含 `rgb_source`**，避免重渲覆盖已冻结的零误差图像 |
| `scripts/measurement_error/determinism_probe.py` | 算子级确定性探针（§1.2） |
| `scripts/measurement_error/repeatability.py` | 10 次重复的 run / analyze / evaluate，逐步入参哈希与首次分叉定位 |
| `scripts/measurement_error/zero_scene.py` | master 冻结、24 GT 相机、合同审计、SHA 清单、零噪声扫描与初始化 |
| `scripts/measurement_error/rgb/render_zero.py` | Blender/OptiX 重渲 24 视图；修复 Blender 5.2 下 `--` 被 argparse 吞掉的参数透传 bug |
| `scripts/measurement_error/calibration.py` | ideal / non_ideal 标定、锁定注册、平移/旋转/内参注入与单位换算 |
| `scripts/mesh_opt/runner.py` | `--trace-file/--trace-every/--trace-detail-at/--deterministic/--calibration`，每步 grad/param/offset 哈希 |
| `scripts/mesh_opt/inputs.py` | 诊断分支复用 `align_poses`；non_ideal 必须 `registration="locked"`；prediction 模式拒绝 GT 回退 |
| `scripts/measurement_error/scan.py` | `rgb_source` 改为可选，已预渲染的零误差 RGB 直接复用 |
| `tests/test_measurement_error.py` | 新增 8 个用例（零噪声逐字节、成对/单位/锁定注册） |

测试（9 个用例全绿，`.venv/Scripts/python.exe tests/test_measurement_error.py`）覆盖：
零扰动逐字节还原、rms=1 的实现值 1±0.02、成对样本严格 2 倍、bias 全视图同向、drift 单调、
旋转不动光心、mm/px/焦距/主点单位换算、`0.1°@470mm=0.8206mm`、
注入误差后经理想注册仍保留 2 mm、ideal 模式注入即报错、prediction 模式拒绝标定文件。

## 6. 复现

```powershell
# 1) master 冻结 + 24 GT 相机 + 合同审计 + 零噪声扫描（Windows venv 无 warp，GT 深度用 WSL）
python scripts/measurement_error/zero_scene.py --config configs/experiments/measurement_error_v2.json --phase master
python scripts/measurement_error/zero_scene.py --config configs/experiments/measurement_error_v2.json --phase rig
python scripts/measurement_error/zero_scene.py --config configs/experiments/measurement_error_v2.json --phase audit
wsl -e /root/venvs/mesh-splatting/bin/python scripts/measurement_error/rays.py --phase all --config configs/experiments/measurement_error_v2.json

# 2) 24 帧 RGB（Blender 5.2.4，OptiX；PowerShell 直接拼参数会把 "--" 吃掉，用 Start-Process 传数组）
$exe = "<blender>/blender.exe"
$args = @('-b','--python','scripts/measurement_error/rgb/render_zero.py','--',
          '--config','configs/experiments/measurement_error_v2.json')
Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory . -NoNewWindow -Wait

# 3) 重复性与地板（WSL，torch 2.14.1+cu126）
wsl -e /root/venvs/mesh-splatting/bin/python scripts/measurement_error/determinism_probe.py
wsl -e /root/venvs/mesh-splatting/bin/python scripts/measurement_error/repeatability.py --phase all --tag zero_v2 --repeats 10
```

产物：`outputs/measurement_error_v2/repeatability/zero_v2/{repeatability.json, metrics.json}`；
探针 `repeatability/probe.json`；重渲对照 `outputs/measurement_error_v2/repro_render/`。

## 7. 遗留缺口与限制

1. **确定性无法开启**，除非弃用 lattice `grid_sample` 或换用确定性 antialias；本轮不换实现（会移动基线）。
2. **RGB 不是逐字节可复现**（1 LSB / 7e-6 像素）。RGB 分支的零噪声控制带必须每批自带 ≥5 seed，
   不能借用几何分支的窄带。
3. 地板是本机（RTX 4070 Laptop / torch 2.14.1+cu126 / 驱动当前版本）的；换硬件、换驱动、
   换精度（bf16 VGGT）都要重测地板。
4. 10 次重复只能给出 4.1% / 0.82° / 1.9% 这一级的分辨率；方案要求的"1 mm 是否可接受"这类
   判据，必须靠单档全表面指标 + 多 seed，不能靠单次 Chamfer 的小数位。
5. 本轮未做任何 P1/P2 扫描；R1 的外参/内参曲线按 §4 的门限在下一批执行。
