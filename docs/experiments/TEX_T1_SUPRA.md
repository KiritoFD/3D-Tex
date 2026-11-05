# Tex-T1：SUPRA-Tex 固定 Mesh 纹理生成推理（2026-11-05）

第三条工作线：**几何与位姿冻结，只用一张参考图做生成式纹理补全**，对标 T0 的
单视角投影融合基线。目标是回答"生成式纹理模型在固定 Mesh 上能带来多大收益、离 T0 多远"。

> 本线**未改动** `scripts/experiments/step3_flexicubes.py`、`scripts/experiments/texture_fixed_mesh.py`、
> `docs/experiments/72H_GEOMETRY.md`、`docs/experiments/TEXTURE_HFREQ.md`；T0 的渲染/评测口径由
> `texture_fixed_mesh.py` 直接 import 复用。`TEXTURE_HFREQ.md` 的分辨率阈值正在被另一条线修正，
> 本文不引用其任何结论。

## 0. 结论

**一句话：在"同一张参考图 + 同一 Mesh + 同一相机/渲染口径"下，SUPRA-Tex（bf16 受限规格）相对
T0 单视角投影融合，未见区域补全提升约 +7.2 dB PSNR、LPIPS 降低 56%，接缝色差降低 37%；
可见区域只提升 +1.37 dB。收益集中在"没被参考视角看到的区域"，不是在可见区域的纹理精度。**

| 轴 | 指标 | T0 单视角（proxy） | SUPRA-Tex（proxy, bf16） | 差值 |
|---|---|---|---|---|
| 1 参考保真 | seen-view PSNR | 17.80 | **19.17** | +1.37 dB |
| 1 参考保真 | seen-view LPIPS | 0.0774 | **0.0639** | −0.0135 |
| 2 未见补全 | unseen-view PSNR | 6.38 | **13.59** | **+7.21 dB** |
| 2 未见补全 | unseen-view LPIPS | 0.360 | **0.159** | **−56%** |
| 2 未见补全 | 跨视角一致性 MAD（0↔1 / 0↔8 / 4↔12） | 0.0320 / 0.0690 / 0.0761 | **0.0162 / 0.0276 / 0.0343** | ≈ −50% |
| 3 表面连续 | seam-pair 平均 \|ΔRGB\| / p95 | 0.0556 / 0.392 | **0.0348 / 0.133** | −37% / −66% |
| 4 材质真实 | albedo vs 参考图 PSNR | 22.57 | **25.12** | +2.55 dB |
| 4 材质真实 | 固定光照重渲 vs 参考图 PSNR | 17.61 | **18.44** | +0.83 dB |

必须一起读的四条限定：

1. **只有 bf16 旁路臂跑完**。fp32（8.38 GB 权重）在本轮 90 min 排队窗口内始终拿不到
   ≥10.5 GB 空闲显存，没有产出。因此上表是**受限规格（bf16 权重）**数字，
   不能与 fp32 基线直接对比，也不能当成"SUPRA-Tex 的最终基线"。
2. **同一张参考图**：两条线都用视角索引 0（`r_0.png`，GT `cameras.json` 的第 0 个 frame）
   作为唯一条件。T0 是"把这张图按可见性投影到 atlas"，SUPRA 是"以这张图为条件做生成"。
   没有把"更多条件视角"带来的收益混进来。
3. **几何不是严格"零改动"（proxy 行）**：SUPRA 的导出路径默认 `remesh=True`（Dual Contouring 重建 +
   decimate），proxy 从 140,623 顶点变成 410,054 顶点 / 484,950 面。形状与位姿未变
   （对齐后 chamfer 0.0049 规范单位 ≈ 1.2 cm 均值；渲染轮廓 IoU 0.968，与 T0 的 0.968 一致），
   但拓扑被重生成。这是**方法固有的几何改动**，不是我们改的。
4. **B1（VGGT 行）已补上，但与 proxy 行规格不同**：VGGT 在 `texture-size=1024` +
   **不重网格化**（原始 202,660 顶点 / 348,875 面直接做 UV 展开，跳过 remesh 与 simplify）下跑通，
   见 §5 的三次尝试。因此 vggt 与 proxy 两行**不是同一套导出配置**，只能各自与自己的 T0 行对比；
   vggt 的绝对数值低于 proxy 不能直接归因于 Mesh 质量或模型结构。

## 1. 口径

- 输入 Mesh（未改几何、未重估位姿）：
  - proxy：`data/scenes/nerf_lego_gt16/mesh/mesh_init.ply`（140,623 顶点，251,156 面）
  - vggt：`outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply`（202,660 顶点，348,875 面）
- 参考视图：索引 **0**，`data/scenes/nerf_lego_gt16/images/r_0.png`；两条线同一张。
- 渲染/评测：直接 `import texture_fixed_mesh as T0`，复用
  `build_clip` / `camera_raster` / `bake` / `build_atlas` / `sample_uv` / `sample_map` /
  `vertex_charts` / `vertex_normals` / `seam_stats` / `psnr` / `ssim`，atlas 2048，
  heldout 16 视角（`outputs/lego_heldout_val16`），`VOXEL=0.0115`，`min_cos=0.2`，遮挡容差 `2 voxel + 1% z`。
- 四轴定义：
  1. **参考保真**：与参考视角（frame 0）比较，分 seen（该 texel 在参考视角可见）/ unseen 两类像素；
     PSNR 在对应像素集上算，LPIPS 用 AlexNet，先按前景 bbox 裁剪、背景置零。
  2. **未见补全**：heldout 16 视角上同样分 seen/unseen；"一致性"取三对视角的重叠像素，
     渲染 RGB 的 MAD（|pred_u − pred_v| 均值，不含 GT）。
  3. **表面连续**：真实 3D seam-pair——在 atlas 里取 chart 边界两侧相邻 texel 对（各 4 px 内），
     反查 3D 位置与法线，要求两侧 3D 点距离 ≤ 2 voxel（`mean_pair_dist_vox≈0.76`），
     统计两侧 RGB 的 `|ΔRGB|` 均值 / p95，最多 200,000 对。
  4. **材质真实**：GLB 含 PBR。给出 albedo 贴图渲染 vs 参考图的 PSNR、roughness/metallic 统计，
     以及固定方向光重渲（`albedo*(0.30+0.65 n·l)`）vs 参考图的 PSNR（仅自一致性参考，不是物理正确性）。

## 2. 四轴指标（完整）

heldout 16 视角、atlas 2048。`*_std` 为 per-view 标准差。

| 指标 | T0 proxy | SUPRA proxy (bf16, 2048/remesh) | T0 vggt | SUPRA vggt (bf16, 1024/不重网格) |
|---|---|---|---|---|
| psnr_full | 13.057 ± 2.650 | 19.181 ± 2.037 | 12.896 ± 2.541 | 16.802 ± 1.956 |
| psnr_fg | 8.075 ± 2.629 | 14.643 ± 2.253 | 7.909 ± 2.511 | 12.167 ± 2.088 |
| psnr_seen | 17.798 ± 1.914 | 19.171 ± 1.437 | 15.537 ± 1.871 | 16.571 ± 1.642 |
| psnr_unseen | 6.381 ± 1.869 | 13.591 ± 1.675 | 6.479 ± 1.817 | 11.490 ± 1.779 |
| ssim_full | 0.7199 ± 0.0473 | 0.7920 ± 0.0332 | 0.7028 ± 0.0431 | 0.7329 ± 0.0356 |
| psnr_inner（前景腐蚀 8 px） | 8.147 ± 2.804 | 14.739 ± 2.316 | 7.999 ± 2.700 | 12.434 ± 2.262 |
| lpips_fg | 0.4012 ± 0.0837 | 0.2142 ± 0.0439 | 0.4311 ± 0.0764 | 0.3121 ± 0.0433 |
| lpips_seen | 0.0774 ± 0.0166 | 0.0639 ± 0.0161 | 0.1110 ± 0.0249 | 0.0922 ± 0.0256 |
| lpips_unseen | 0.3600 ± 0.0925 | 0.1589 ± 0.0539 | 0.3723 ± 0.0824 | 0.2244 ± 0.0512 |
| silhouette IoU | 0.9676 ± 0.0100 | 0.9681 ± 0.0073 | 0.9362 ± 0.0249 | 0.9363 ± 0.0249 |
| 一致性 MAD 0↔1 | 0.0320 | 0.0162 | 0.0290 | 0.0179 |
| 一致性 MAD 0↔8 | 0.0690 | 0.0276 | 0.0595 | 0.0250 |
| 一致性 MAD 4↔12 | 0.0761 | 0.0343 | 0.0744 | 0.0381 |
| seam pairs | 544,086 | 336,178 | 286,948 | 260,419 |
| seam mean \|ΔRGB\| | 0.0556 | 0.0348 | 0.0533 | 0.0330 |
| seam p95 \|ΔRGB\| | 0.3922 | 0.1333 | 0.3804 | 0.1255 |
| albedo PSNR vs ref | 22.573 | 25.119 | 19.578 | 22.275 |
| relight PSNR vs ref | 17.607 | 18.444 | 16.385 | 16.982 |
| roughness mean/std | —（无 MR） | 0.4977 / 0.0122 | — | 0.4979 / 0.0123 |
| metallic mean/std | —（无 MR） | 0.0000 / 0.0001 | — | 0.0000 / 0.0000 |

关于 proxy 行需要注意的统计口径差异：

- T0 的 seam pairs 更多（544,086 vs 336,178），因为 UV 展开器不同（T0 用 xatlas，SUPRA 用 o-voxel
  `uv_unwrap`），chart 边界数量与拓扑不同；两者都是"真实 3D 相邻点对"，但**不是同一组点对**，
  只能作量级对比。
- `zero_vis_texels_share`（参考视角下不可见 texel 占 atlas 覆盖的比例）：T0 proxy 0.775、SUPRA proxy 0.864；
  T0 vggt 0.834、SUPRA vggt 0.835（SUPRA 的 chart 更多更碎，单视角条件下 atlas 里不可见 texel 比例更高）。
  这一点对 T0 更有利（可见区覆盖更多），因此上表的 SUPRA 增益不是靠"占更多可见 texel"得来的。

关于 vggt 行（B1）：

- **同样是"修正后口径"**：与 proxy 行共用 `tex_t1_eval.py` 的 axis（`p_canon=[−x,y,−z]`）与
  UV（`V→1−V`）还原，`eval_final/summary.json` 同一份产物；轮廓 IoU 0.9363 与 T0 vggt 的 0.9362
  吻合，可作为对齐正确的旁证。
- **相对自己的 T0**：seen 16.57 vs 15.54（+1.03 dB）、unseen 11.49 vs 6.48（+5.01 dB）、
  unseen LPIPS 0.224 vs 0.372（−40%）、seam 0.0330 vs 0.0533（−38%，p95 −67%）、
  albedo 22.28 vs 19.58。方向与 proxy 行一致，但幅度更小。
- **与 proxy 行的差**（不可直接归因）：seen −2.60 dB、unseen −2.10 dB、unseen LPIPS +0.066。
  vggt 行同时叠加了三种差异：Mesh 本身（VGGT 几何，T0 的 unseen 也更低）、纹理 1024（proxy 是 2048）、
  导出为不重网格化（proxy 是 remesh+decimate）。因此这 2 dB 不能单独归因于 Mesh。

## 3. 资源开销与执行

| 项 | 值 |
|---|---|
| 环境 | 独立 conda 前缀 `/opt/conda/envs/supra-tex`（Python 3.10、torch 2.6.0+cu124、CUDA toolkit 12.4），未污染 `/opt/conda/envs/3dtex` |
| 采样步数 | 25 步 Euler（`sample_material_latents`，`rescale_t=3.0`，`sigma_min=1e-5`） |
| 纹理分辨率 | 2048（proxy，bf16 臂）；1024（vggt 降规格尝试） |
| 种子 | 42（全部 run 相同） |
| proxy 单次耗时 | 161 s（端到端：加载权重 → 编码 → 25 步采样 → 双解码 → remesh/UV/烘焙/导出） |
| proxy 峰值显存 | `peak_all=22,438 MiB`，`peak_delta=6,249 MiB`（同租户进程常驻 16.1 GB） |
| vggt 单次耗时 | 166 s（同上，但跳过 remesh 与 simplify，直接对 348,875 面做 UV 展开） |
| vggt 峰值显存 | `peak_all=22,926 MiB`，`peak_delta=6,737 MiB`（同租户进程常驻 16.2 GB） |
| vggt 产物 | `glb_bf16/supra_vggt_v0_s42_bf16_no_remesh.glb`（18 MB；proxy 为 `supra_proxy_v0_s42_bf16.glb`） |
| GPU 排队 | `flock -n` 轮询 `/root/3dtex/.locks/gpu4090.lock`，空闲 <6.5 GB（B1）/ <10.5 GB（fp32 主臂）立即放锁重试（等待期间不持锁）；**未杀任何他人进程，与几何线严格串行** |
| 锁等待 | B1 在 20:07–20:23 被几何线 `step3_flexicubes.py` 扫参占用（`free` 一度 3134 MiB），按 45 s 重试排队，20:27:59 以 `free=7895 MiB` 拿到锁 |
| 产物 | `outputs/tex_t1_supra_20261105/`（`inputs/`、`glb_bf16/`、`t0_single_view/`、`eval_final/`、`logs/`） |
| 日志 | `logs/supra_proxy_v0_s42_bf16.log`、`logs/b1_*.log`、`logs/vram_b1_*.txt`、`logs/eval_*.log` |

bf16 转换方式（旁路臂，可复现）：

- 从 `completion/config.json` 复制一份 `completion_bf16/config.json`，把
  `models.denoiser.args.dtype` 改成 `bfloat16`；
- `supratex/inference/pipeline.py`：允许用 `SUPRATEX_ABI_IGNORE=dtype` 跳过 ABI 的 dtype 校验；
- `supratex/pipelines/observation_guided_completion.py`：`SUPRATEX_FORCE_BF16=1` 时，在
  `assign=True` 载入前把 checkpoint 张量 cast 到**目标模块自身的 dtype**（attention/FFN blocks 为
  bf16，torso/`adaLN_modulation`/`t_embedder` 仍为 fp32），并放宽归一化 key 之外的 dtype 检查。
- 数值异常检查：输出无 NaN、无像素崩坏（baseColor 2048² 三通道 min 0 / max 1，均值
  0.397 / 0.319 / 0.171）；但**材质分支近乎常数**：roughness `0.498±0.012`（范围 0–0.522）、
  metallic `≈0`（最大 0.031）。原因是输入 `mra.png` 本身是中性先验（R=0 metallic、G=0.5 roughness，
  见 `tex_t1_build_inputs.py`），输出基本是这份先验的延续；因此**材质这一轴在本轮实际不可评**，
  也无法把"常数"归因到 bf16。要给材质轴结论，得先给出非平凡 MRA 输入并做 fp32 对照。

## 4. 权重来源与许可

| 组件 | 来源 | 版本/大小 | 许可 |
|---|---|---|---|
| SUPRA-Tex 代码 | `https://github.com/Supra-3D/supra-tex` | main @ `cee4435652d0f23737bd28d5704e0fae9830b411`（2026-10-29），无 `.git`（tarball 解包） | 仓库 LICENSE：MIT |
| 完成模型（denoiser） | HF `supra3d/supra-tex` → `completion/` | `denoiser_step0030000.safetensors`，7.9 GB（8.38 GB fp32 权重），config `dtype=float32` | 随权重 LICENSE：Apache-2.0 |
| TRELLIS.2 VAE | HF `microsoft/TRELLIS.2-4B`，rev `af44b45f2e35a493886929c6d786e563ec68364d` | 3.1 GB，`ckpts/` 4 个组件（shape enc/dec、tex enc/dec） | **下载目录内没有 LICENSE 文件**；上游仓库标注 MIT，正式发布前需复核模型卡条款 |
| CuMesh / FlexGEMM / nvdiffrast / o-voxel | 随 SUPRA 安装脚本从 GitHub 编译 | — | 各自仓库许可（MIT / Apache-2.0） |

装环境按官方 README 走，编译了 `CuMesh`、`FlexGEMM`、`nvdiffrast`、`o-voxel`；`CuMesh/setup.py`
需要补 `--extended-lambda` 与 `--expt-relaxed-constexpr`（否则 nvcc 报 lambda 错误）。

`third_party/supra_tex/` 是上游源码快照（不随本仓库提交），由 `tex_t1_install.sh` 在缺失时按上述
commit clone；bf16 旁路臂的 2 处改动（ABI dtype 豁免 + checkpoint→目标模块 dtype）与导出旁路的
2 个开关（`SUPRATEX_SKIP_SIMPLIFY=1`、`SUPRATEX_NO_REMESH=1`，见 `mesh_export.py`）只存在于该目录，
若要复现需按 §3 的三条说明重打。

## 5. 失效 run（全部如实登记）

| 时间 | 尝试 | 结果 | 原因 |
|---|---|---|---|
| 16:04 | proxy/vggt × seed 42/1234（fp32，4 跑） | 全部 rc=1，2 s | 环境缺 `safetensors`（随后装 `safetensors`/`huggingface_hub`/`numba`） |
| 16:43 | 同上 4 跑（fp32） | 全部 CUDA OOM | 同租户常驻 23.5 GB，空闲 13 MiB；fp32 权重 8.38 GB + 模型 8.4 GB 装不下 |
| 19:09 | fp32 主臂 | 未启动 | 90 min 排队窗口到期，持锁方始终占用；落 `blocked.json`（`need_free_mib=10500`） |
| 19:14/19:15/19:26 | bf16 proxy/vggt | rc=1，1–2 s | `completion_bf16` 缺 `partial_normalization.json`；随后 ABI dtype 校验拒绝 |
| 19:28–19:31 | bf16 proxy/vggt | rc=1，15–23 s | checkpoint 与模块 dtype 不匹配（`adaLN_modulation`/`t_embedder` 仍 fp32） |
| 19:35 | bf16 proxy | **成功**，161 s | — |
| 19:36 | bf16 vggt（2048 / 500k） | rc=1，119 s，`peak_all=23,444 MiB` | `CuMesh.simplify` CUDA OOM（同租户占 16 GB，只剩 ~7.3 GB） |
| 19:55 | bf16 vggt 重试 | rc=1，秒级 | 误用 `completion_bf16/` 下的 checkpoint 路径（该目录只有 config），随后改指 `completion/` |
| 19:57 | bf16 vggt（1024 / 300k） | rc=1，OOM | 同样 `CuMesh.simplify` CUDA OOM（空闲 7.9 GB 仍不足；降纹理/降简化目标没有降低该步峰值） |
| 19:40–19:58 | 评测口径修正（非 run） | 废弃 3 份 summary | ① 漏了导出器的绕 Y 180° 变换（质心对不上，IoU 0.57）② 漏了 `uvs[:,1]=1−V` 的还原（纹理错位）③ MR 通道 R/B 读反。修正后 IoU 0.9681 |
| 20:07 | B1 base（1024 / 250k，`flock` 已拿到） | rc=1，114 s，`peak_all=22,926 MiB`、`peak_delta=6,737 MiB` | `mesh.simplify` → `cumesh.simplify_step` CUDA OOM；采样与解码已完成，失败在导出阶段 |
| 20:08–20:23 | B1 no_simplify 排队（非 run） | 未启动 | 几何线 `step3_flexicubes.py`（grid-res 160）持锁且 `free` 低至 3134 MiB；按 45 s 短持锁轮询排队，未并发、未杀进程 |
| 20:23 | B1 no_simplify（跳过 simplify，仍 remesh） | rc=1，114 s，`peak_all=22,926 MiB` | remesh 输出面数过大，`cumesh.uv_unwrap → compute_charts` CUDA OOM |
| 20:27:59 | B1 no_remesh（跳过 simplify 且不做 remesh） | **成功**，166 s，`peak_all=22,926 MiB`、`peak_delta=6,737 MiB`，GLB 18 MB | 以 `free=7895 MiB` 起跑；原始 202,660 顶点 / 348,875 面直接 UV 展开 |

> 说明：vggt 臂前两次失败都是"导出阶段显存不足"而非"模型失败"——采样与解码均已完成，
> 失败点在 `CuMesh.simplify` 与 `cumesh.uv_unwrap → compute_charts`。第三条路径（不重网格化 +
> 1024 纹理）跑通，vggt 行因此补齐；但仍需注意它**不是**与 proxy 同一套导出配置。
> 最终口径的 summary 在 `outputs/tex_t1_supra_20261105/eval_final/summary.json`。

## 6. 复现

```bash
# 1) 环境（独立前缀，不污染 3dtex）
bash scripts/experiments/tex_t1_install.sh
bash scripts/experiments/tex_t1_build_ext.sh
bash scripts/experiments/tex_t1_weights.sh

# 2) 输入包：固定 Mesh + 单张参考图（索引 0）-> SUPRA 规范相机
python scripts/experiments/tex_t1_build_inputs.py --ref-index 0 --out outputs/tex_t1_supra_20261105/inputs

# 3) 主臂（fp32，需 >=10.5GB 空闲显存；本轮未完成）
bash scripts/experiments/tex_t1_run.sh

# 4) 受限规格旁路臂（bf16，本轮实际完成的那条）
bash scripts/experiments/tex_t1_fallback.sh
bash scripts/experiments/tex_t1_vggt_bf16_wait.sh

# 4b) B1：vggt 行（1024 / 250k / 不重网格化；三次路径由 MODE 选择）
MODE=base        bash scripts/experiments/tex_t1_vggt_b1.sh   # 1024/250k，simplify（本轮 OOM）
MODE=no_simplify bash scripts/experiments/tex_t1_vggt_b1.sh   # 跳过 simplify，仍 remesh（本轮 OOM）
MODE=no_remesh   bash scripts/experiments/tex_t1_vggt_b1.sh   # 跳过 simplify 与 remesh（本轮跑通）

# 5) 四轴评测（T0 用 t0:<mesh> 规格复用同一渲染口径）
python scripts/experiments/tex_t1_eval.py \
  --glb t0:proxy --glb t0:vggt \
  --glb supra_proxy_v0_s42_bf16=outputs/tex_t1_supra_20261105/glb_bf16/supra_proxy_v0_s42_bf16.glb \
  --glb supra_vggt_v0_s42_bf16_no_remesh=outputs/tex_t1_supra_20261105/glb_bf16/supra_vggt_v0_s42_bf16_no_remesh.glb \
  --out outputs/tex_t1_supra_20261105/eval_final --save-renders
```

## 7. 待办（本轮没做完的部分）

1. **fp32 主臂**：需要一段 ≥10.5 GB 空闲显存的窗口（proxy 峰值 ≈ 8.4 GB + 基线）。
   拿到后重跑 `tex_t1_run.sh`（seed 42/1234 + 两个 Mesh），才能给出可对比的基线。
2. **vggt 同规格臂**：B1 已在 1024 + 不重网格化下补齐（原始 202,660 顶点 / 348,875 面），
   但若要和 proxy 行严格同规格（2048 + remesh + decimate），仍需 ≥8.5 GB 空闲显存的窗口重跑
   `MODE=base`；本轮 `base` 与 `no_simplify` 两次都卡在导出阶段 OOM，未完成分块 simplify 的尝试。
3. **材质轴**：`roughness≈0.498±0.012 / metallic≈0` 是输入 `mra.png` 中性先验的延续，
   本轮不能作为"材质真实性"结论；要评这一轴需给非平凡 MRA 输入 + fp32 对照。
4. **多视图条件**：单列为后续实验，不能把"条件数"收益算进模型结构优势。
