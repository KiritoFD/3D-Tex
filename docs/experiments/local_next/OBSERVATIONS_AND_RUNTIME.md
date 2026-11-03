# 预测观测加权与完整训练环境推进

2026-11-03，本地 RTX 4070 Laptop 8 GB；承接 [采集/LOO 实测](CAPTURE_RESULTS.md)。
本页新增实验没有达到联合质量门槛，不替换默认模型。

## B2 小规模对照：调整观测权重，不删除整图

固定 Lego16 的原始 Ω 预测、预测相机、输入 RGBA alpha、TSDF ROI/256 对角体素、
4-voxel 截断带和 nearest 投影；不重跑模型、不读取 GT 深度/相机计算权重、不 carve。
alpha 是数据集标注，不声称已有真实照片自动分割方案。基线为 native prediction_init，
不是 GT 辅助的 v35c30+conf。

三组预先固定的假设，不做调参搜索：

- incidence：预测深度反投影、中心差分求表面法线，
  `factor = 0.25 + 0.75 * abs(dot(n, ray))**2`。边界处不使用混合前景/背景的差分。
- boundary：到有效前景边界的像素距离 d，
  `factor = 0.25 + 0.75 * clip(d/3, 0, 1)`，图像边缘也视为边界。
- combined：两因子相乘。各因子保留正下限；单像素细结构仍有非零权重。

将因子乘到原 confidence，沿用 TSDF 的 `[0,8]` clip；没有重新归一化或更换输入深度。
局部噪声也会影响预测法线，因此入射角只是待验证的代理，不是已校准可靠度。
四组预测 ROI 相同：对角 0.7263、voxel 约 0.002837、grid 126×160×198。
各候选融合约 5.7 秒；生成 PLY，不做 M2 形变或重新融合渲染深度。

评测才使用共同训练相机 Sim3 到源坐标、精确 Blender 源网格 BVH；
50k 面积采样、固定随机种子，1 voxel=0.0115014435 源单位；不是官方 benchmark。
Sharp 划分沿用源网格 40° 邻面阈值，不等同于所有薄结构。

| 方案 | F | P | R | Sharp P | Sharp R | Mesh 法线° | Hole fill | 对称均距 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| confidence 基线 | 0.60675 | 0.55510 | 0.66900 | 0.32204 | 0.67617 | 50.81110 | 0.98706 | 0.01306 |
| +incidence | 0.60863 | 0.56732 | 0.65644 | 0.33995 | 0.66354 | 49.75953 | 0.98543 | 0.01299 |
| +boundary | 0.60647 | 0.55526 | 0.66808 | 0.32729 | 0.67509 | 50.65323 | 0.98622 | 0.01310 |
| +combined | 0.60824 | 0.56800 | 0.65462 | 0.33378 | 0.65884 | 49.54973 | 0.98494 | 0.01310 |

结论：入射角降低约 1° 法线误差，但 F 仅 +0.00188，R/Sharp R 下降；边界项近乎无益，
combined 继续牺牲召回。Hole fill 都接近 0.985，不解决开孔。未达 F≥0.70、R≥0.705，
不采用这些权重为默认，也不能把小幅 F 差异宣称统计显著。还未测试多场景/多种子，
本轮不追加 val16 或参数搜索，先停止这一组失败候选的扩展。
free-space/多视图一致性项尚未实现，不能把它们算作已测过的 B2。

代码：`scripts/fusion/observation_weights.py`、`scripts/experiments/prediction_smoke.py`。
每组 `provenance.json` 记录输入/代码 SHA256、运行时 Git 基线、相机和 mask 来源。
输出：`outputs/fusion_observations/{incidence,boundary,combined}/mesh/mesh_init.ply`；
原始指标 `outputs/fusion_observations/eval/{surface_metrics,ablation_report}.json`。
定向回归 11 项通过，新增一项覆盖尺度不变性、正面平面及单像素细结构不被置零。

```powershell
foreach ($m in @('incidence','boundary','combined')) {
  .venv/Scripts/python.exe scripts/experiments/prediction_smoke.py `
    --observation-mode $m --out outputs/fusion_observations/$m
}
```

## 完整 Mesh Splatting 环境：与组件成功分开报告

专用 WSL venv `/root/venvs/mesh-splatting`、Torch 2.14.1+cu126 与基础环境关系见
[上一页](CAPTURE_RESULTS.md)。`torch-scatter 2.1.2` 已源码编译安装，CUDA 上实际运行
`scatter_mean([1,3,2],[0,0,1]) == [2,2]` 通过；不只是 import 成功。
安装日志 `outputs/mesh_splatting/install_torch_scatter.log`。
真实 `train.py --help` 已越过 scatter，前进到缺少 tinycudann。

[tiny-cuda-nn 官方实现](https://github.com/NVlabs/tiny-cuda-nn) 已 recursive clone：
主提交 `3daa6e5b8b8e0fa17971a1a20acfbb3754494b38`，submodule 固定为
cmrc `952ffddba731fc110bd50409e8d2b8a06abbd237`、
cutlass `82f5075946e2569589439d500733b700a3141374`、
fmt `fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa`。

Torch 的 host bindings 需要 C++20；全量 CUDA C++20 先出现 fmt consteval 错误，
禁用 consteval 后又触发 nvcc 12.8 `object.h:666` 内部编译器错误。
最终方案只将 host bindings 改为 C++20，纯 CUDA 源保持上游 C++17，编译成功。
不修改编码、网络或损失数学逻辑。不要保留失败的 fmt 宏绕过。
`scripts/mesh_splatting/prepare_runtime.py --install` 固定源码版本并复现这项构建修改；
不加 `--install` 时只执行实际 CUDA HashGrid 前后向及 scatter 检查并记录结果。
链接还需要 `LIBRARY_PATH=/usr/lib/wsl/lib` 找到 WSL `libcuda.so`，已纳入脚本。
实际 `runtime_probe.json` 检查通过：HashGrid 1024×3→1024×8，输入梯度 norm=0.62081、
参数梯度 norm=68.19383，均有限非零；scatter 输出 `[2,2]`。
`train.py --help` 已真实退出 0，日志 `outputs/mesh_splatting/train_help_complete.log`。
这是依赖、CUDA 运算和训练入口成功，尚不是 DMTet/Remeshing 完整训练成功。

```bash
# 在专用 WSL venv 内运行；源码及其三个 submodule 必须是上述固定版本。
/root/venvs/mesh-splatting/bin/python scripts/mesh_splatting/prepare_runtime.py \
  --repo /root/deps/tiny-cuda-nn --install \
  --out outputs/mesh_splatting/runtime_probe.json
```

## DTU/单目法线的数据来源与限制

[IDR 官方下载脚本](https://github.com/lioryariv/idr/blob/main/data/download_data.sh)
的 Dropbox 本机连接超时。替代来源为
[MonoSDF 作者发布的预处理包](https://github.com/autonomousvision/monosdf/blob/main/scripts/download_dataset.sh)，
S3 `DTU.tar` HEAD 成功：4,583,424,000 bytes，ETag `82a67600a262d7c174ca66b51744cd01-547`，
支持 byte ranges。实测整包及 range 请求仅约 10–30 KB/s，替代 S3 hostname 同样慢；
已停止整包下载，`data/.cache/monosdf_DTU.tar` 只是未完成片段，不能交给训练。
日志：`outputs/mesh_splatting/{download_dtu,range_probe,virtual_probe}.log`。

MonoSDF 的 [Omnidata 提取脚本](https://github.com/autonomousvision/monosdf/blob/main/preprocess/extract_monocular_cues.py)
保存 CHW `[0,1]` 法线；其数据读取再变为 `2*n-1`。
Mesh Splatting IDR reader 直接加载 `normal/*_normal.npy`，不做这个转换；
渲染的监督法线在相机坐标系。直接复制数组会错误，接入时必须核对 signed 范围、
轴方向、384 crop 对应内参/图像，以及数值/可视化一致性，不能用 GT 法线补缺。
DTU 扫描 GT/ObsMask 仍需单独准备；官方 15k（DMTet 5k+remesh 10k）完整训练与评估未运行。

下一步解决标准场景数据获取与法线契约，再执行完整训练；
不继续围绕这三组失败权重调参。416-reproduction 权限、多场景/多种子仍是开放项。
