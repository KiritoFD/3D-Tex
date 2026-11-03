# 毫米误差测定工具

2026-11-04。当前交付是可配置工具及端到端短流程验证，**没有完整扫参结果，也不判定容忍范围**。
固定 Lego、24 个名义视角与场景尺寸。入口配置：
[`configs/experiments/measurement_error.json`](../../../configs/experiments/measurement_error.json)。

## 两个分支，三种不同的毫米定义

| 分支/模式 | 被扰动的数据 | 毫米 RMS 的含义 | VGGT 是否重跑 | 下游 |
|---|---|---|---|---|
| RGB / `object_warp` | RGB 和对应的输入 mask | 以局部 GT 深度换算的图像平面二维定位位移；平滑场相关长度可调 | 每档、每 seed 重跑 | 原生预测相机 TSDF |
| RGB / `camera_jitter` | 实际相机中心，GPU 重渲染 RGB | 三维相机中心平移；保持朝向，按全部视图归一到目标 RMS | 每档、每 seed 重跑 | 原生预测相机 TSDF |
| 外部深度 / `iid`、`view_bias` | 独立扫描深度 | 相机 z-depth 的 RMS；逐像素独立误差或整张图共享偏差 | 固定同一份 VGGT 输出 | 几何 M2 + 外部扫描约束 |

三种毫米定义分别出表，不能混成同一种设备精度。RGB 定位场是受控畸变代理，并不等价于
LiDAR 深度噪声、亮度噪声或真实镜头畸变。相机抖动模式评测使用**实际扰动后的相机**；
名义 24 个位置固定，但真实位置发生了已记录的变化。

默认深度分支：8 档 RMS × 2 类噪声 × 3 seeds = 48 组；另有 initial、no_external 对照。
默认 RGB 分支：5 档 RMS × 2 模式 × 3 seeds = 30 组。
零噪声的多 seed 可以完全相同，不能据此声称有三个独立场景。

## 修改什么

| 配置字段 | 用途 |
|---|---|
| `noise_rms_mm`、`noise_modes`、`seeds` | 外部扫描误差档位、类型和重复 |
| `rgb_sweep.rms_mm`、`rgb_sweep.modes` | RGB 分支的档位与可选模式 |
| `rgb_sweep.warp_correlation_pixels` | 物面定位误差场的空间相关长度 |
| `optimizer.*` | 迭代数、形变类型、外部扫描权重、深度先验容差等 |
| `fscore_thresholds_mm` | 任意正毫米阈值，例如 `[0.25, 0.75, 1.5]` |
| `evaluation_samples`、`evaluation_seed` | 固定面积采样数量与种子 |
| `root`、`out`、`rgb_sweep.root/out` | 数据和结果位置；改变已完成实验的输入/优化参数时使用新目录 |

输入观测噪声与评测阈值互相独立。修改阈值后只运行 `evaluate.py --phase table`，
直接用保存的距离数组重新计分，不重新造噪声、不重新推理或优化。
没有硬编码的“5% 合格线”或设备采购结论。

## 本机环境

工作目录为仓库根目录 `G:\GitHub\3D-Tex`。
Windows `.venv/Scripts/python.exe` 用于 VGGT 与原生 TSDF 初始化；
WSL `ext4` 的 `/root/venvs/mesh-splatting/bin/python` 用于 Warp CUDA、nvdiffrast 和优化。
WSL 仓库路径为 `/mnt/g/GitHub/3D-Tex`。新增依赖为 `warp-lang==1.19.0`；其余沿用既有环境。
所有 RGB 重渲染、基准深度和结构深度渲染都使用 GPU，RGB 没有 CPU 回退。

下列 PowerShell 中的 `$blender` 先设置为本机 Blender 可执行文件绝对路径。

### 1. 公共源网格、毫米尺度与精确 GPU 深度

```powershell
& $blender -b --disable-autoexec --python scripts/measurement_error/source.py -- --phase extract
.venv/Scripts/python.exe scripts/measurement_error/source.py --phase calibrate
wsl -d ext4 -- bash -lc 'cd /mnt/g/GitHub/3D-Tex && /root/venvs/mesh-splatting/bin/python scripts/measurement_error/rays.py'
```

所需资产为配置中的 `.blend`、NeRF Lego 训练相机及 RGB，均为现有本地数据，不随 Git 提交。
源网格、视角和源文件 SHA 固定；改变几何或视角要使用新 `master`。
尺度来自用户指定的 8 mm 凸点间距，经过 28 块积木、256 个凸点交叉校验；
当前为 125.83286 mm/源单位，约 160×288×169.48 mm。

### 2A. 外部几何误差分支

```powershell
.venv/Scripts/python.exe scripts/measurement_error/scan.py --phase prepare
.venv/Scripts/python.exe scripts/view_selection.py infer --run-dir data/scenes/measurement_error/rgb24 --seed 42 --aggregator-dtype bfloat16
.venv/Scripts/python.exe scripts/measurement_error/optimize.py --phase init
wsl -d ext4 -- bash -lc 'cd /mnt/g/GitHub/3D-Tex && /root/venvs/mesh-splatting/bin/python scripts/measurement_error/optimize.py --phase run'
wsl -d ext4 -- bash -lc 'cd /mnt/g/GitHub/3D-Tex && /root/venvs/mesh-splatting/bin/python scripts/measurement_error/evaluate.py --phase all'
```

小规模执行可加 `--only no_external,iid_rms0_s42,iid_rms1_s42`。
`scan.py --phase fuse` 是单独的“扫描→TSDF”参考，可用 `--family scan_tsdf` 评测；
它与“VGGT 初始化+扫描约束”分表，不能替代该分支的管线结果。

### 2B. RGB 误差→VGGT 分支

```powershell
wsl -d ext4 -- bash -lc 'cd /mnt/g/GitHub/3D-Tex && /root/venvs/mesh-splatting/bin/python scripts/measurement_error/rgb/prepare.py --phase inputs'
& $blender -b --disable-autoexec --python scripts/measurement_error/rgb/render.py
wsl -d ext4 -- bash -lc 'cd /mnt/g/GitHub/3D-Tex && /root/venvs/mesh-splatting/bin/python scripts/measurement_error/rgb/prepare.py --phase gt'
.venv/Scripts/python.exe scripts/measurement_error/rgb/run.py --phase all
wsl -d ext4 -- bash -lc 'cd /mnt/g/GitHub/3D-Tex && /root/venvs/mesh-splatting/bin/python scripts/measurement_error/evaluate.py --family rgb --phase all'
```

这些命令也支持 `--only object_warp_rms1_s42,camera_jitter_rms1_s42`，先测少量案例。
`rgb/run.py` 的 `infer`、`fuse`、`predict_metrics` 可以单独运行。
RGB 缺帧、图像 SHA 改变、缓存推理数值配置不一致都会报错，不静默混用缓存。
默认推理保留 8 GiB host commit 门槛；`rgb_sweep.min_free_commit_gib` 可覆盖启动门槛，
不改变模型、精度或输入。GPU 作业应串行执行。

### 3. 改阈值与导表

```powershell
.venv/Scripts/python.exe scripts/measurement_error/evaluate.py --phase table
.venv/Scripts/python.exe scripts/measurement_error/evaluate.py --family rgb --phase table
```

| 产物 | 内容 |
|---|---|
| `out/m2_eval/results.csv` | 外部扫描分支逐组指标、每个 F/P/R 阈值、相对同 seed 无噪声模型变化 |
| `rgb_sweep.out/rgb_eval/results.csv` | RGB 分支逐组 Mesh 指标，以及原始 VGGT 深度/位姿指标 |
| `grouped.csv` | 每模式/档位的均值、标准差、已完成数/预期 seed 数 |
| `results.md` | 可直接阅读的完整案例目录；未运行显示 pending，数值留空 |
| `case/distances.npz` | 原始毫米距离、法线角度、薄结构代理像素深度误差，供阈值重算 |
| `case/metadata.json` | 输入定义、配置、尺度、数值底噪、模型/几何产物标识 |

指标定义、限制与已完成验证见 [PROTOCOL.md](PROTOCOL.md) 和 [VALIDATION.md](VALIDATION.md)。
