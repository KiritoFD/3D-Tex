# Tex-T2：Hunyuan3D-2.1 纹理生成（固定 Mesh 基线）

**状态：降规格版（256 输入 / 6 视角），本轮未达成官方标准配置。**

- 提交基线：`34780489537d17eb7f4fa8c334ce355b7623db99`
- 运行日期：2026-11-05
- 参考视图：`r_0.png`（所有 run 固定同一张 GT 输入图，与 T1 同口径）
- 产物：`outputs/tex_t2_hunyuan_20261105/`

## 1. 结论

1. **官方标准配置无法在本轮 4090 上跑**：Hunyuan3D-2.1 官方 README 标注纹理生成需 **21 GB 显存**，本线实测 512 输入 6 视角按激活量推算需 **9.5–10.0 GB** 设备级显存；本轮该卡常驻其他任务，空闲仅 **7,043 MiB**（缺口 2.5–3.0 GB）。因此**未跑 512 正跑**，详见 `BLOCKED_VRAM.md`。
2. 实际完成的是 **256 输入 / 6 视角降规格版**，两个 Mesh 都跑通 Paint + 四轴评测链路，进程内 torch 峰值 **6,331 MiB**（proxy）/ **6,326 MiB**（vggt），**远低于 21 GB，但不可与 T0、T1 的 512 结果直接比较**。
3. **四轴指标可出数，但纹理质量明显偏弱**：seen 视图 albedo PSNR 仅 14.09 dB（proxy）/ 14.06 dB（vggt），同 Mesh 同相机的 T0 是 25.29 / 23.15 dB。在 canonical 正面视角（`diag_canonical_front.png`）下也能肉眼看到明显的彩色碎片/跨 chart 串色，说明 256 多视角 Paint 的输出本身不够干净，叠加 atlas 仅 1024（`texture_size=2048` 经 `downsample=True` 减半），进一步放大采样误差。**本轮数字只用于证明评测链路可跑通，不作质量结论。**
4. PBR 支路可跑：albedo + metallic/roughness 三张贴图齐全，固定光照重渲的 PSNR 反而高于同一纹理的 unlit albedo（14.09 → 18.45 dB），因为光照遮蔽压低了部分碎片对比度，这不能当作“材质更真实”的证据；metallic 均值 0.161/0.131、roughness 均值 0.695/0.760 是唯一有参考意义的绝对量。
5. 几何未被修改：xatlas 只重排顶点，`nearest_vertex_max = 1.4e-6`、`bbox_delta_max ≤ 1.1e-6`、`face_count_equal = true`。

## 2. 四轴指标（降规格 256 / 6 视角，参考 `r_0.png`）

### 2.1 Proxy Mesh（140,623 顶点 → UV 重排后 221,343 顶点 / 251,156 面）

| 轴 | 指标 | Hunyuan 256 | T0（同 Mesh 同相机） |
|---|---|---|---|
| seen (r_0) | albedo PSNR full / fg / inter | **14.09 / 7.92 / 7.90** | 25.29 / 20.20 / 21.48 |
| seen (r_0) | albedo SSIM crop / LPIPS crop | 0.470 / **0.431** | 0.734 / 0.213 |
| seen (r_0) | IoU | 0.964 | 0.964 |
| seen (r_0) | PBR 重渲 PSNR full / fg / inter | 18.45 / 12.38 / 12.49 | 20.65 / 14.69 / 14.95 |
| unseen (16 视图) | albedo PSNR full / fg / inter | **12.25±0.68 / 7.27±0.57 / 7.28±0.57** | 22.52±0.82 / 18.67±1.05 / 19.60±0.69 |
| unseen (16 视图) | albedo SSIM crop / LPIPS crop | 0.489±0.063 / **0.419±0.051** | 0.755±0.035 / 0.182±0.029 |
| unseen (16 视图) | PBR 重渲 PSNR full / fg / inter | 13.64±1.99 / 8.66±1.90 / 8.70±1.93 | 16.23±1.81 / 11.35±1.85 / 11.46±1.88 |
| 跨视角一致性 | mean / p95 abs RGB | **0.243 / 0.590** | 0.201 / 0.513 |
| 3D seam-pair | 边界对数 / mean / p95 abs RGB | 505,293 / **0.171** / 0.627 | 未在本轮同口径重算 |
| PBR 材质 | metallic mean / p90 | **0.161 / 0.720** | —（T0 无 MR 通道） |
| PBR 材质 | roughness mean / p10 | **0.695 / 0.262** | —（T0 无 MR 通道） |

### 2.2 VGGT Mesh（202,660 顶点 → UV 重排后 339,752 顶点 / 348,875 面）

| 轴 | 指标 | Hunyuan 256 | T0（同 Mesh 同相机） |
|---|---|---|---|
| seen (r_0) | albedo PSNR full / fg / inter | **14.06 / 8.02 / 8.01** | 23.15 / 17.95 / 19.15 |
| seen (r_0) | albedo SSIM crop / LPIPS crop | 0.442 / **0.489** | 0.648 / 0.312 |
| seen (r_0) | IoU | 0.939 | 0.939 |
| seen (r_0) | PBR 重渲 PSNR full / fg / inter | 18.08 / 12.04 / 12.22 | 20.23 / 14.21 / 14.60 |
| unseen (16 视图) | albedo PSNR full / fg / inter | **12.18±0.68 / 7.29±0.65 / 7.32±0.62** | 19.72±0.79 / 15.65±0.97 / 16.57±0.57 |
| unseen (16 视图) | albedo SSIM crop / LPIPS crop | 0.473±0.067 / **0.460±0.054** | 0.627±0.048 / 0.306±0.042 |
| unseen (16 视图) | PBR 重渲 PSNR full / fg / inter | 13.48±1.93 / 8.52±1.87 / 8.60±1.88 | 15.83±1.72 / 10.93±1.73 / 11.15±1.74 |
| 跨视角一致性 | mean / p95 abs RGB | **0.274 / 0.611** | 0.200 / 0.500 |
| 3D seam-pair | 边界对数 / mean / p95 abs RGB | 266,478 / **0.226** / 0.729 | 未在本轮同口径重算 |
| PBR 材质 | metallic mean / p90 | **0.131 / 0.574** | —（T0 无 MR 通道） |
| PBR 材质 | roughness mean / p10 | **0.760 / 0.514** | —（T0 无 MR 通道） |

> T0 一列取自 `outputs/texture_fixed_mesh_20261105/` 的缓存，在**本轮同一相机、同一 shader**（metal=0、rough=0.9）下重算，供口径参考；不代表官方 T0 报告表。

**读表注意**

- 所有 Hunyuan 数字均来自 **256 输入 / 6 视角 / 15 步 UniPC / cfg 3.0**，与 T0、T1 的 512 口径不同，**禁止横向当作方法优劣结论**。
- albedo 是 unlit 采样（只反映纹理本身），PBR 是在固定世界光 key/fill 下的 Cook-Torrance 重渲；二者差值主要来自光照遮蔽，不是材质准确度。
- T0 无 metallic/roughness，因此 PBR 材质轴只能看 Hunyuan 的绝对分布，没有可比基线；SUPRA（T1）对比留待统一口径后补。

## 3. 资源开销与配置

| run | Mesh | 分辨率 | 视角数 | 步数 | atlas 保存 | 耗时 | 进程内 torch 峰值 | 设备级峰值（估） |
|---|---|---|---|---|---|---|---|---|
| paint proxy `_smoke256` | 140,623 v | 256 | 6 | 15 | 1024 | **311.2 s** | **6,331.3 MiB** | ~6.8–7.0 GB |
| paint vggt `_r256` | 202,660 v | 256 | 6 | 15 | 1024 | **1,696.2 s** | **6,326.2 MiB** | ~6.8–7.0 GB |
| eval proxy+vggt `_r256` | — | — | — | — | — | ~2 min | 未单独记录（远低于 paint） | — |

公共设置：`render_size=1024`、`texture_size=2048`（保存时 `downsample=True` → **1024**）、`use_remesh=False`、`save_glb=False`、`merge_method=fast`；低显存布局 `--cpu-offload --dino-cpu`（UNet 3.7G + VAE 0.32G 常驻 GPU，text_encoder 用完退回 CPU，image_encoder 释放，DINOv2-giant CPU float32）。

**VGGT 耗时 1,696 s 明显长于 proxy 的 311 s**，主要来自 202k 顶点 / 348k 面的 xatlas 展开与 1024 atlas 的 CPU 侧烘焙，不是扩散步数差异。

### 何时补 512 正跑

- 单卡 free ≥ **10 GB**（估算峰值 9.5–10.0 GB），且无其他常驻 GPU 任务；
- 候选命令：`bash scripts/experiments/tex_t2_run.sh full proxy,vggt`（512 → 384 → 256 自动回退，`flock -w 3600` 排队）；
- 建议同时把 `texture_size` 提到 4096（保存 2048），因为 1024 atlas 对 25 万三角面偏稀，会额外压低 PSNR。

## 4. 权重、许可与第三方代码

| 组件 | 来源 | 大小 | 许可 |
|---|---|---|---|
| Paint 权重 `hunyuan3d-paintpbr-v2-1` | `tencent/Hunyuan3D-2.1`（HF） | 6.5 GB | Tencent Hunyuan 3D 2.1 Community License（排除 EU/UK/韩国；非独家；商用与 MAU>1M 有附加条件；禁止用其输出改进其他 AI 模型） |
| `dinov2-giant` | `facebook/dinov2-giant`（HF） | 4.3 GB | Apache-2.0 |
| `RealESRGAN_x4plus.pth` | 官方仓库 `hy3dpaint/ckpt/` | 64 MB | BSD-3-Clause（xinntao/Real-ESRGAN） |
| 上游代码 | `github.com/tencent-hunyuan/hunyuan3d-2.1` | — | 仓库 LICENSE 为 Hunyuan 3D 2.1 Community License；部分源码文件头另注明 third-party/non-commercial 条款，实际使用需按仓库 LICENSE 复核 |

- 权重与代码均**未重新训练**，只做推理。
- 环境装在独立前缀 `/root/conda_prefix/hy3d_paint`（Python 3.10 + torch 2.5.1+cu124），**未污染 `/opt/conda/envs/3dtex`**。
- 第三方代码位于 `third_party/hunyuan3d/`（上游未修改），`third_party/stubs/bpy.py` 为最小 stub。

## 5. 无效 run 与阻塞记录

| 时间 | run | 结果 | 原因 / 处理 |
|---|---|---|---|
| 16:56 | paint proxy 256 | 有效 | 首个成功 run，`peak=6331.3 MiB`、`seconds=311.2` |
| 16:58 | eval proxy 256 | **失败** | `grid_sample` 维度不匹配：input `[1,3,800,800]` vs grid `[1,30000,2]` → grid 改 `[None,None]` |
| 17:18 | eval proxy 256 | **失败** | metallic/roughness 在 CPU、grid 在 CUDA → `torch.cat(...).cuda()` |
| 18:07 | eval proxy,vggt 256 | **失败** | metallic 形状 `[H,W]` 与 albedo `[H,W,3]` 广播失败 → 采样后 `unsqueeze(-1)` |
| 17:33–18:12 | 全部 GPU run | 排队 | 共享锁 `/root/3dtex/.locks/gpu4090.lock` 长期被另一路 10-run battery 与 T1 占用；只 `flock -w` 等待，未抢锁、未杀进程 |
| — | 512 / 6 视角 | **未执行（阻塞）** | 空闲 7,043 MiB < 估算 9.5–10.0 GB，见 `BLOCKED_VRAM.md` |
| 18:11 | eval proxy,vggt 256 | **有效** | 两 Mesh 四轴全部出数，`rc=0` |

## 6. 复现

```bash
# 环境
export CUDA_HOME=/root/conda_prefix/hy3d_paint
export PATH=/root/conda_prefix/hy3d_paint/bin:$PATH
export PYTHONPATH=/root/3dtex/third_party/stubs:/root/3dtex/third_party/hunyuan3d/hy3dpaint

# 降规格 Paint + 四轴评测（自动 flock 排队）
bash /root/tex_t2_step2.sh

# 或单步
python scripts/experiments/tex_t2_hunyuan.py \
  --stage paint --meshes vggt --resolution 256 --max-num-view 6 \
  --render-size 1024 --texture-size 2048 --tag _r256 --cpu-offload --dino-cpu
python scripts/experiments/tex_t2_hunyuan.py \
  --stage eval --meshes proxy,vggt --tag _r256
```

产物清单：

- `paint_proxy_smoke256.json` / `paint_vggt_r256.json`：配置、几何校验、耗时、峰值显存
- `{mesh}_painted.{obj,mtl,jpg}` + `_metallic.jpg` / `_roughness.jpg`（世界系）
- `{mesh}_painted_canonical.*`（canonical 帧原始输出）
- `metrics_{mesh}_r256.json` / `summary_r256.json`：四轴逐视图数字
- `diag_canonical_front.png`：canonical 正面视角渲染（纹理质量问题证据）
- `BLOCKED_VRAM.md`：512 显存阻塞记录
- `logs/`：`paint_vggt_r256.log`、`eval_r256.log`、`step2.log`
