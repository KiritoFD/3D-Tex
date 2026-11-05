# 阶段 2：ExMesh 自适应 Remesh —— 复现准备与预注册（2026-11-05）

> 本文件只记录**不吃 GPU** 的准备结果和**事先写死**的评测口径。
> 本轮没有启动任何训练/优化，文中**不包含任何未实测的 ExMesh 效果数字**。

## 0. 准备到什么程度、GPU 部分还需什么

**已完成（全部 CPU / 编译）：**

1. 上游 pin：`third_party/exmesh/` 存 pristine 源码（26 MiB / 1633 文件），commit 与 LICENSE 已记录。
2. 独立环境 `/root/conda_prefix/exmesh`：python 3.11.13 + torch 2.6.0+cu124，三个 CUDA 扩展
   （simple-knn / diff-plane-rasterization / nvdiffrast）已编译并 import 通过，**未做任何 GPU 计算验证**。
3. 两份输入：GT-depth 代理 Mesh 与 VGGT Mesh，各留 source gauge 与 Omega gauge 两个版本，
   sha256_16、非水密状态、连通分量、边界边、面积，以及 `evaluate.py` 口径的基线指标均已复算。
4. 预注册评测口径与通过门槛（§6），评测路径已实测可复现基线（§4.4）。

**尚未开始（GPU）：** ExMesh 一次前向/优化都没跑。写作时（2026-11-05 21:0x）GPU 已用
`18,636 MiB / 24,564 MiB`、空闲 `5,448 MiB`，**不满足 §7.2 的 ≥12 GiB 启动门槛**，故未启动。
开跑前还有 3 项 CPU 侧适配与校验，见 §8。

---

## 1. 上游代码 pin

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/Fan-Treasure/ExMesh |
| commit | `2cecbe513a0e2e021cd5cab509761ddc458ee526` |
| commit 时间 | 2026-10-28 16:38:40 +0800 |
| tree | `84e8b03d390117de3a849f0704d9e877529d462d` |
| License | Apache-2.0（`third_party/exmesh/LICENSE.md`，10,801 B） |
| vendored 路径 | `third_party/exmesh/`（26 MiB，1633 个文件，**不含 `.git`**） |
| 上游文件改动 | **0 处**（用 `git archive` 从上述 commit 导出，未就地编辑） |
| 我们的补丁 | `third_party/exmesh_patches/0001-train-mask-resize.patch`，`git apply --check -p1` 通过 |

机器可读 pin：`third_party/exmesh_pin.json`。

说明：

- `/root/deps/ExMesh`（仓库外）是带 `train.py` 修改的工作副本，仅作对照，不作为交付。
- 补丁 0001 来自 `docs/EXMESH_ADAPTER.md` §2.2：`loadCam()` 把图像缩到
  `-r` 分辨率，但 `train.py` 读 `train_mask` 时用原图尺寸，导致 mask 与渲染尺寸不一致。补丁
  `PATCH(3D-Tex)` 只加一行 NEAREST resize，使用时打补丁，不写进 vendored 源码。
- 旧文档说"上游 simple-knn / diff-plane-rasterization 需补 `<cfloat>` / `<cstdint>`"：那是 GCC 13
  的行为。本机 host 编译器是 conda-forge GCC 9.5，**不需要**该补丁，未添加。

---

## 2. 独立环境

| 项 | 值 |
|---|---|
| 前缀 | `/root/conda_prefix/exmesh`（7.1 GiB，未污染 `3dtex` / `supra-tex` / `hy3d_paint`） |
| Python | 3.11.13 |
| torch | 2.6.0+cu124（CUDA runtime 12.4） |
| torchvision | 0.21.0+cu124 |
| numpy | 1.26.4（首次安装拉到 2.4.6，已固定回 1.26.4 以匹配 3dtex 侧的旧 API 假设） |
| CUDA toolkit | 12.4.131，`/opt/conda/envs/nvcc124`（`nvcc 12.4`） |
| host 编译器 | conda-forge gcc 9.5.0（`/opt/conda/envs/nvcc124/bin/x86_64-conda-linux-gnu-gcc`） |

编译约束：`CUDA_VISIBLE_DEVICES=''` + `TORCH_CUDA_ARCH_LIST=8.9` + `MAX_JOBS=8`，
全程**未创建 CUDA context、未占显存**。系统 gcc 7.5 编译 torch 2.6 头文件会因 C++17 失败，
因此显式指定 conda-forge GCC 9.5 作为 host compiler（这是本轮唯一一次编译踩坑）。

| 扩展 | 来源 | wheel sha256_16 | 大小 | 编译日志 |
|---|---|---|---|---|
| `simple_knn 0.0.0` | ExMesh `2cecbe5` 的 `PGSR/submodules/simple-knn` | `2f6e10deec774303` | 428,650 B | `outputs/exmesh_prep_20261105/logs/build_simple_knn.log` |
| `diff_plane_rasterization 0.0.0` | ExMesh `2cecbe5` 的 `PGSR/submodules/diff-plane-rasterization` | `d374b476501ff617` | 470,969 B | `logs/build_dpr.log` |
| `nvdiffrast 0.4.0` | `/root/deps/nvdiffrast` @ `253ac4f`（与 `3dtex` 同源） | `f80a5bda82650f4e` | 720,397 B | `logs/build_nvdiffrast3.log` |

wheel 存放于 `/root/build/wheels/`，用 `pip install --no-deps` 装入新前缀。

**import smoke（`CUDA_VISIBLE_DEVICES=''`，不占用 GPU）：**

- `torch.cuda.is_available() == False`、`device_count() == 0`（预期，因为显式禁用了设备）；
- `nvdiffrast.torch`、`simple_knn._C`、`diff_plane_rasterization._C`、`open3d`、`xatlas`、`trimesh`、
  `skimage`、`lpips`、`plyfile`、`imageio`、`cv2`、`tensorboard` 全部 import 成功；
- 在上游目录执行 `import train` 成功（打印 `EXMESH_IMPORT_OK`）。

**未验证**：CUDA kernel 的数值正确性、渲染结果、显存峰值——这些必须等 GPU 阶段，本轮不做。

---

## 3. 依赖对照表

上游 `requirements.txt` 只有 8 行（open3d / plyfile / opencv-python / lpips / trimesh / xatlas /
imageio / scikit-image），但代码实际 import 更多；下表按**实际 import 路径**核对。

| 依赖 | 来源 | 状态 | 说明 |
|---|---|---|---|
| torch / torchvision | 代码必需 | ✅ 2.6.0+cu124 / 0.21.0+cu124 | 与 4090 现有驱动匹配 |
| numpy | 代码必需 | ✅ 1.26.4 | 固定版本 |
| nvdiffrast | `renderer/__init__.py`、`utils/graphics_utils.py` | ✅ 0.4.0 源码编译 | 训练渲染必需 |
| simple-knn | `scene/` 初始化 | ✅ 0.0.0 | 上游子模块，已编译 |
| diff-plane-rasterization | `renderer/` 光栅化 | ✅ 0.0.0 | 上游子模块，已编译 |
| open3d | requirements | ✅ 0.18.0 | |
| plyfile | requirements | ✅ 1.1.3 | |
| opencv-python | requirements | ✅ `opencv-python-headless 4.11.0` | 用 headless 避免 GUI 依赖 |
| lpips | requirements | ✅ 0.1.4 | 仓库另有内置 `lpipsPyTorch/` |
| trimesh | requirements | ✅ 5.2.0 | |
| xatlas | requirements | ✅ 0.0.11 | UV 展开 |
| imageio | requirements | ✅ 2.38.1 | |
| scikit-image | requirements | ✅ 0.26.0 | |
| scipy | 代码 import | ✅ 1.17.1 | |
| tqdm / tensorboard | 代码 import | ✅ 4.70.1 / 2.21.0 | |
| Pillow | 代码 import | ✅ | |
| **pytorch3d** | 仅 `PGSR/scene/gaussian_model.py` | ⚠️ 未装 | ExMesh 训练路径不 import PGSR；只在我们想跑 PGSR 基线时才需要 |
| **depth_anything_3** | 仅 `scripts/da3_dtu_colmap.py` | ⚠️ 未装 | 数据准备脚本；本项目复用 VGGT-Ω / GT 深度，不跑它 |

结论：**ExMesh 训练路径无缺失依赖**；两项未装依赖都不在当前执行路径上，属于"缺失但不阻塞"。

---

## 4. 输入 Mesh 与可用性统计

### 4.1 文件与哈希（全部在 `outputs/exmesh_prep_20261105/inputs/`）

| 文件 | 含义 | sha256_16 | 字节 |
|---|---|---|---|
| `proxy/mesh.ply` | GT-depth 冻结 TSDF 代理 Mesh（source gauge） | `e4006a146e8ae5d6` | 14,200,684 |
| `vggt/mesh.ply` | VGGT Mesh（source gauge） | `ecf994efd40a5d9c` | 20,754,399 |
| `proxy_omega/mesh.ply` | 上者经冻结 Sim3 逆变换到 Omega gauge | `8ac26e48d5e360dd` | 24,861,692 |
| `vggt_omega/mesh.ply` | 上者经冻结 Sim3 逆变换到 Omega gauge | `d784d2bab141b76e` | 36,215,890 |

来源路径（两处都核对过）：

- 代理 Mesh：`data/scenes/nerf_lego_gt16/mesh/mesh_init.ply`（251,156 面，与 §4.4 的 F1 0.9358 一致）。
- VGGT Mesh：任务书写的是 `outputs/geometry_pruning/eval_carve_v35c30c_conf/carve_v35c30c/mesh.ply`，
  **实际磁盘路径是** `outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply`
  （`c30c` 是笔误）。已确认该文件与 `eval_carve_v35c30_conf/ablation_report.json` 的
  `case=carve_v35c30` 对应，sha 与参考目录下的 mesh 完全一致。

### 4.2 拓扑统计（CPU，trimesh + scipy，脚本 `scripts/exmesh_prep/input_stats.py`）

| 指标 | 代理 Mesh | VGGT Mesh | 原 G0 记录 |
|---|---|---|---|
| 顶点 / 面 | 140,623 / 251,156 | 202,660 / 348,875 | 251,156 面 ✓ |
| watertight | **否** | **否** | 非水密 ✓ |
| 连通分量（面邻接，主口径） | 1,211 | 3,995 | — |
| 连通分量（顶点连通，参考） | 1,144 | 3,793 | 3,781（定义差异，见下） |
| 边界边 | 30,964 | **55,397** | 55,397 ✓ |
| 非流形边（>2 面共用） | 0 | 0 | — |
| 真实三角面积 (m²) | 11.4254 | 15.3427 | — |
| 叉积和口径面积 (m²) | 22.8508 | **30.6855** | 30.69（≈2× 真实面积） |
| 欧拉数 V−E+F | −437 | 524 | — |
| bbox (min → max) | (−0.670, −1.169, −0.340) → (0.656, 1.179, 1.009) | (−0.678, −1.180, −0.362) → (0.668, 1.155, 1.014) | — |

口径差异说明（写进报告时不能混用）：

- 表面积：`mesh_opt/evaluate.py` 的采样权重用 `|cross|` 的和（少一个 1/2），所以历史 G0 的
  "面积 30.69 m²" 是**真实面积的 2 倍**。本文按真实面积 15.3427 设门槛，同时附叉积和口径，
  两者比值一致。
- 连通分量：3,781 与本次 3,793 的差来自定义（是否计入仅被顶点共享的面片、是否剔除未引用顶点）。
  主口径固定为**面邻接**（3,995），并同时报顶点连通值。
- 边界边、面数、非水密三维与 G0 完全一致，说明输入没有被替换或悄悄改动。

### 4.3 用 `evaluate.py` 复算的输入基线

命令（CPU + Blender 3.3.9，无 GPU 依赖）：

```bash
cd /root/3dtex
BL=/root/deps/blender/blender-3.3.9-linux-x64/blender
/root/conda_prefix/exmesh/bin/python scripts/mesh_opt/evaluate.py \
  --out outputs/exmesh_prep_20261105/eval_proxy \
  --meshes proxy=outputs/exmesh_prep_20261105/inputs/proxy/mesh.ply --blender $BL
# VGGT 的 source-gauge 版直接评测会得到 F1=0（见 4.4），正式口径走 *_omega + --mesh-is-omega
```

| 指标 | 代理 Mesh | VGGT Mesh | G0/任务书 |
|---|---|---|---|
| F@1vox | **0.9358** | **0.6315** | 0.9358 / 0.6315 ✓ |
| precision | 0.9190 | 0.5808 | — |
| observed recall | 0.9531 | 0.6920 | — |
| accuracy mean (m) | 0.004141 | 0.015527 | — |
| completeness mean (m) | 0.003577 | 0.009631 | — |
| Chamfer (m) | **0.003859** | **0.012579** | 0.003859 / 0.012580 ✓ |
| 法线 mean (deg) | 38.94 | 50.38 | 50.44（差 0.06°，以本次复算为准） |
| Sharp precision@1vox | 0.7688 | **0.3381** | 0.3395（差 0.0014，以本次复算为准） |
| Sharp recall@1vox | 0.9404 | 0.6935 | 0.6935 ✓ |
| accuracy >2 vox 离群率 | 0.71% | 23.14% | — |

日志：`outputs/exmesh_prep_20261105/logs/eval_proxy.log`、`logs/eval_vggt2.log`，
指标 JSON 在 `outputs/exmesh_prep_20261105/eval_*/surface_metrics.json`。

### 4.4 尺度/gauge 实测结论（重要）

- 直接对 **source gauge** 的 VGGT Mesh 加 `--mesh-is-omega` → **F1 = 0**（位移到 GT 之外），
  说明传入的 mesh 已经是 Sim3 之后的结果，**不能再乘一次**。
- 用冻结 Sim3 逆变换得到 `vggt_omega/mesh.ply`，再用 `--mesh-is-omega` 评测 → F1 **0.6314**、
  Chamfer 0.012579、法线 50.38°，与 4.3 的 source 值一致（F1 差 0.0001 来自 ASCII PLY 的
  6 位小数舍入，约 1e-6 m，可忽略）。
- 代理 Mesh 逆变换后同样回到 F1 **0.9358**、Chamfer 0.003859。

因此 §6 的正式评测将**统一走 Omega gauge**（ExMesh 输出也是 Omega gauge），
并固定 `--mesh-is-omega`；source gauge 文件作为可核对的原件保留。

---

## 5. 我们的适配点：PGSR 初始化 → VGGT/TSDF Mesh

### 5.1 官方前提 vs 我们的替换

- 官方 README：先用 PGSR 跑 5000 步得到 mesh 作为 ExMesh 初始化，再进行自适应 remesh + 纹理。
- 我们的替换：**直接从原始 Mesh 初始化**（不经 PGSR、也不再经自建 FlexiCubes harness）。
  两份输入分别是 VGGT-Ω→TSDF Mesh（G0）与 GT-depth 冻结 TSDF 代理 Mesh。
- 因此本报告只能表述为"**把上游的自适应 remesh 模块接到我们的 TSDF Mesh 初始化上**"，
  **不得表述为"官方方法完整复现"**；官方结果依赖 PGSR 初始化，不在本轮对照范围内。

### 5.2 gauge 决策（写死）

- 上游 `scripts/exmesh_adapter.py` 按 **Omega gauge** 设计（"Ω 深度、Ω 相机、Ω 融合网格同一非米制 gauge，
  不做任何缩放"，见 `docs/EXMESH_ADAPTER.md` §3）。
- 但两份 Mesh 在磁盘上只有 **source gauge**（评测坐标系）。因此：
  - VGGT 侧：相机/深度用原生 Omega 值；Mesh 用 `vggt_omega/mesh.ply`（逆 Sim3）。
  - 代理侧：GT 相机、GT 深度、代理 Mesh 都在 source gauge，全部按同一 Sim3 逆变换到 Omega：
    `z_omega = z_src / s`，`T_omega = T_src @ [[Rᵀ/s, −Rᵀ t/s], [0,0,0,1]]`，Mesh 按本节的逆变换公式。
- 冻结 Sim3（`ablation_report.json`，只用于相机/深度/网格对齐，不参与评测拟合）：
  `s = 4.723245823229695`，
  `R = [[-0.999907,-0.003463,0.013170],[-0.013603,0.298304,-0.954374],[-0.000624,-0.954465,-0.298323]]`，
  `t = [-0.051168, 3.836517, 1.203765]`。
- Mesh 变换：`p_src = s·R·p_omega + t` ⇔ `p_omega = (p_src − t)·R / s`（R 正交，转置即逆）。
  本轮 `scripts/exmesh_prep/to_omega.py` 已按此实现并验证（§4.4）。
- **风险**：任何一处漏缩放都会破坏深度先验与网格的对齐；开跑前必须做 §8 的对齐 spot-check。

### 5.3 输入契约（从上游代码读出）

`-s` 目录需要：

```text
<workdir>/
├── images/                      # 16 张，-r 后 400×400（原图 800×800 RGBA）
├── sparse/0/{cameras.txt,images.txt,points3D.txt}   # COLMAP 文本，world→camera
├── mesh.ply                     # 我们的 TSDF Mesh（Omega gauge）
├── mono_priors/da3/<img>.npz    # {"depth": ...}，不跑 Depth Anything 3
└── train_mask/<img>_gtmask.png  # 前景 mask（本实验用 GT 前景，属监督信息，已在纹理线声明）
```

- 图像：`data/scenes/nerf_lego16/images` 与 `data/scenes/nerf_lego_gt16/images` 的 16 张完全相同
  （`r_0.png` sha256 一致，800×800 RGBA），所以两个 workdir 只差相机/深度/mesh 的来源。
- 相机：VGGT 用 `data/scenes/nerf_lego16/geometry/omega/cameras.json`（含 `K_original`、`T_world_to_cam`、
  `pixel_affine_original_to_model`）；代理用 `data/scenes/nerf_lego_gt16/geometry/gt/cameras.json`。
- Mask：`data/scenes/nerf_lego_gt16/geometry/gt/foreground/*.npy`，写盘为 `*_gtmask.png`；
  上游 `filtered/` 目录不存在，适配时用显式路径传 GT 前景，不靠默认。
- UV/纹理：ExMesh 自己生成 UV（xatlas）与 2560² 纹理，输入 Mesh 不需要 UV。

### 5.4 不改上游

所有适配都在仓库侧脚本/补丁完成，`third_party/exmesh/` 保持 pristine：

- 补丁 0001：`train_mask` 随 `-r` 缩放（必需）。
- 若需要多 seed（§6.3）：追加补丁 0002 暴露 `--seed`，仍不改上游文件。

---

## 6. 预注册评测口径（冻结，开跑前不得改）

### 6.1 冻结常量

- 体素：`voxel = 0.01150144350298549 m`（`ablation_report.json`，全流程沿用，不因 Mesh 变化重算）。
- 评测器：`scripts/mesh_opt/evaluate.py` → `scripts/capture/capture_eval.py` + Blender 3.3.9
  `scripts/diagnostics/measure_surfaces.py`（CPU）。
- GT：冻结的 16 帧 GT 深度、GT 相机、`observed_gt_samples.npy`、`lego_source_edge_flags.npy`、`lego_source.blend`。
- 输出 Mesh 一律在 Omega gauge → 评测命令必须带 `--mesh-is-omega`（§4.4 已验证）。

### 6.2 指标定义与聚合

| 指标 | 定义 | 工具/字段 | 聚合 |
|---|---|---|---|
| F@1vox | precision = 输出面采样点距 GT 点 ≤1 vox 的比例；observed recall 同理 | `fscore_1_voxel` | F/P/R 单值 |
| Chamfer | accuracy 与 observed completeness 的均值 | `symmetric_mean_distance` | 单值 (m) |
| 法线误差 | 输出面法线与 GT 法线夹角 | `unsigned_normal_mean_deg` / median | 单值 (deg) |
| Sharp P/R | 源边（>40°）区域的 1 vox 精度/召回 | `structure_split.precision_sharp@1vox` / `recall_sharp@1vox` | 单值 |
| 面积 | 三角面积和（真实面积；另报叉积和 2× 口径） | trimesh / numpy | 单值 (m²) |
| 连通分量 | 面邻接连通分量（主）；顶点连通（参考） | `connected_component_labels` / scipy | 整数 |
| 边界边 | 只被 1 个面使用的边 | 边计数 | 整数 |
| Remesh 保真距离 | 输出面按面积加权采样 50k 点 → 到输入 Mesh 表面的最近距离 | trimesh `ProximityQuery` | p50 / p90 / max（m 与 vox 双单位），并报 >4 vox 占比 |

Remesh 距离的采样与随机数固定：`numpy.random.default_rng(0)`，面积权重按三角形面积；
输入 Mesh 取该 run 对应的输入（`vggt_omega` 或 `proxy_omega`），不跨输入比较。

### 6.3 噪声估计

- 上游 `utils/general_utils.py:148-150` 把 `random` / `numpy` / `torch` 全部固定为 seed 0，**没有 `--seed` CLI**。
- 预注册：用补丁 0002 暴露 `--seed`，同一配置跑 **2 个 seed（20261105 / 20261106）**；
  若补丁未实施，则退化为同 seed 重复 2 次，并在报告中写明"σ 只含执行噪声、不含初始化扰动"这一退化。
- σ 取重复 run 的样本标准差；门槛中的 `2σ` 均指此值。σ 无法计算时**不得声称任何改善**。

### 6.4 通过门槛（数值写死，按输入分别判）

| 编号 | 判据 | VGGT 输入 | 代理输入 |
|---|---|---|---|
| T1-F1 | F1 ≥ 输入 F1 − max(0.01, 2σ) | ≥ 0.6215（σ=0 时） | ≥ 0.9258 |
| T1-Comp | 面邻接连通分量 ≤ 输入的 10% | ≤ 399 | ≤ 121 |
| T1-Bnd | 边界边 ≤ 输入的 10% | ≤ 5,539 | ≤ 3,096 |
| T2-SharpP | Sharp precision ≥ 输入 − max(0.02, 2σ) | ≥ 0.3181 | ≥ 0.7488 |
| T2-SharpR | Sharp recall ≥ 输入 − max(0.02, 2σ) | ≥ 0.6735 | ≥ 0.9204 |
| T3-Area | 真实面积 ≤ 1.5× 输入 | ≤ 23.014 m² | ≤ 17.138 m² |
| T4-Normal | 法线 mean ≤ 输入 + max(3°, 2σ) | ≤ 53.38° | ≤ 41.94° |
| T5-Chamfer | Chamfer ≤ 输入 + max(0.001 m, 2σ) | ≤ 0.013579 | ≤ 0.004859 |
| T6-Remesh | 新面→输入面距离 p50 ≤ 0.5 vox、p90 ≤ 2 vox、max ≤ 8 vox、>4 vox 占比 ≤ 1% | 同（与输入无关） | 同 |

判定规则：

- **通过（"自适应 remesh 在保住 F1 的前提下减少连通分量与边界边"成立）**：T1 三项 + T2 + T3 + T4 + T5 + T6 全部满足。
- T1-F1 未过但 T1-Comp/Bnd 过：判"以保真度换拓扑"，**不通过**。
- T1-F1 过但分量/边界边未下降 90%：判"remesh 无效"，**不通过**。
- 任一 run 出现 OOM / NaN / 崩溃 / 超时：记为无效 run，**必须列原因，不得丢弃**；无效 run 不计入 σ。
- 不接受事后追加或替换指标；如需新增指标，只能作为"辅助观察"另行列，不参与判定。

### 6.5 产物位置

- 本轮：`outputs/exmesh_prep_20261105/`（inputs / eval_* / logs / input_stats.json）；
  复现用的两个脚本已入库到 `scripts/exmesh_prep/`。
- GPU 阶段（尚未创建）：`outputs/exmesh_gpu_20261106/<run>/`（每 run 独立目录，不覆盖任何既有 outputs）。

---

## 7. 尚未开始的 GPU 部分

### 7.1 命令模板（等 §8 完成后执行）

```bash
cd /root/3dtex/third_party/exmesh            # 或打补丁后的 scratch 副本
/root/conda_prefix/exmesh/bin/python train.py \
  -s /root/3dtex/outputs/exmesh_gpu_20261106/<run>/workdir \
  -m /root/3dtex/outputs/exmesh_gpu_20261106/<run>/out \
  -r 2 > /root/3dtex/outputs/exmesh_gpu_20261106/<run>/train.log 2>&1
```

默认参数（上游 `arguments/__init__.py`）：`iterations=10,000`、`split_from_iter=1000`、
`split_until_iter=7000`、`split_size=24.0`、`max_shapes=200,000`、`texture_resolution=(2560,2560)`。
`-r 2` → 400×400（原图 800×800）。

### 7.2 显存估算（**估算，非实测**）

- 上游 200 步冒烟在 8 GiB 卡（4070 Laptop）上跑通，mesh 规模 214k 顶点 / 386k 面，与本输入的
  202k / 349k 同量级。
- 本轮 16 视角 400×400、2560² 纹理、`max_shapes` 200k，估算峰值 **6–12 GiB**。
- **启动门槛：锁内检测 `memory.free ≥ 12 GiB`**；跑起来后记录 `torch.cuda.max_memory_allocated`
  与 `nvidia-smi` 采样峰值，实测值回填报告。估算不得写成实测。

### 7.3 排队规则（共享 GPU）

- 锁：`/root/3dtex/.locks/gpu4090.lock`，`flock -n` 短持锁。
- 持锁后立刻查 `nvidia-smi --query-gpu=memory.free --format=csv,noheader`：
  `≥ 12 GiB` 才启动；不足则**立即放锁**，`sleep 3` 后重试，**不握锁等待**。
- 不杀、不重启他人进程；不并发多个 run；显存紧张时只降低分辨率（`-r 3`）并重跑，不做深度调优。

### 7.4 运行后评测（CPU）

1. `evaluate.py --mesh-is-omega` 生成 `surface_metrics.json`（每 run 一份）。
2. 拓扑统计（`input_stats.py` 同口径）与 Remesh 距离（§6.2）各出一份 JSON。
3. 汇总表：4 个 run（2 输入 × 2 seed）逐项对照 §6.4，标出未通过的项，并给重复波动区间。

### 7.5 已知风险

- `split_size=24` 是像素域阈值，400×400 下的实际分裂强度未实测。
- 输入非水密（55,397 条边界边、3,995 个分量）可能让 split/merge 更激进，也可能产生更多碎片。
- 若深度先验与 Mesh 的 gauge 对齐有误，ExMesh 会把深度当强监督，误差会被"吸收"进几何。
- 纹理 2560² 在 400×400 下可能不是显存主项，但仍需实测。

---

## 8. 开跑前还剩的 CPU 待办（都不占 GPU）

1. **workdir 构造**：按 §5.2 生成 `vggt_omega` 与 `proxy_omega` 两个 workdir
   （相机/深度按同一 Sim3 变换，mask 用 GT 前景），复用 `scripts/exmesh_adapter.py`；
   该脚本当前是 Omega 原生路径，新增的只是"从 source gauge 输入生成 Omega 相机/深度"的预处理，
   不改上游代码。
2. **对齐校验**（纯 CPU，跑前必须过）：16 帧齐全、图像/深度/mask 尺寸一致、主点居中、
   `mesh.ply` 可被上游 `fetchMesh` 读取、渲染深度与先验深度在若干像素上的符号/量级 spot-check。
3. **run 编排**：2 输入 × 2 seed = 4 个 run 的锁、日志、峰值显存与指标落盘；补丁 0002（seed）如需实施在此完成。

---

## 9. 复现命令附录

```bash
# 1) 独立环境
/opt/conda/bin/conda create -p /root/conda_prefix/exmesh python=3.11 -y
/root/conda_prefix/exmesh/bin/pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
/root/conda_prefix/exmesh/bin/pip install 'numpy==1.26.4' plyfile open3d opencv-python-headless \
  lpips trimesh xatlas imageio scikit-image scipy tqdm tensorboard ninja

# 2) 三个 CUDA 扩展（只编译，不创建 CUDA context）
export CUDA_HOME=/opt/conda/envs/nvcc124
export TORCH_CUDA_ARCH_LIST=8.9 CUDA_VISIBLE_DEVICES='' MAX_JOBS=8
export CC=$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc CXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++
/root/conda_prefix/exmesh/bin/pip wheel --no-build-isolation --no-deps -w /root/build/wheels \
  /root/build/exmesh_ext/simple-knn
/root/conda_prefix/exmesh/bin/pip wheel --no-build-isolation --no-deps -w /root/build/wheels \
  /root/build/exmesh_ext/diff-plane-rasterization
/root/conda_prefix/exmesh/bin/pip wheel --no-build-isolation --no-deps -w /root/build/wheels \
  /root/build/exmesh_ext/nvdiffrast
/root/conda_prefix/exmesh/bin/pip install --no-deps /root/build/wheels/*.whl

# 3) 输入
cp data/scenes/nerf_lego_gt16/mesh/mesh_init.ply \
   outputs/exmesh_prep_20261105/inputs/proxy/mesh.ply
cp outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply \
   outputs/exmesh_prep_20261105/inputs/vggt/mesh.ply
/root/conda_prefix/exmesh/bin/python scripts/exmesh_prep/to_omega.py \
  outputs/geometry_pruning/eval_carve_v35c30_conf/ablation_report.json \
  outputs/exmesh_prep_20261105/inputs/vggt/mesh.ply  outputs/exmesh_prep_20261105/inputs/vggt_omega/mesh.ply \
  outputs/exmesh_prep_20261105/inputs/proxy/mesh.ply outputs/exmesh_prep_20261105/inputs/proxy_omega/mesh.ply
```

---

## 10. 本轮明确未做的事

- 没有启动 ExMesh 训练、优化、渲染或任何 GPU 计算；CUDA 扩展只编译 + import。
- 没有对 ExMesh 的最终几何/纹理质量做任何测量，因此不写任何效果数字。
- 没有改动 `docs/experiments/72H_GEOMETRY.md`、`configs/experiments/72h.json`、`docs/experiments/TEX_*.md`、
  `outputs/step_*`、`outputs/tex_t*`。
