# 工具验证记录

2026-11-04，本机 RTX 4070 Laptop 8 GB。以下是工具短流程测试，**不是完整噪声曲线或设备容差结论**。

## 已验证链路

| 项目 | 实际证据 |
|---|---|
| 尺度检查 | 28 块积木 / 256 个凸点；125.83285965 mm/源单位 |
| GPU GT 深度 | 每网格 3072 BVH 检查射线，前景支持不一致 0；512 最大误差 0.000337 mm，256 最大误差 0.001119 mm |
| GPU 距离数值底噪 | 50k 源表面自距离 p99.9 约 0.0000060 mm，最大约 0.0000076 mm |
| 深度分支数据 | 默认 48 组观测已生成；两类噪声、8 档、3 seeds；记录实际 RMS |
| 深度分支优化 | `no_external`、`iid_rms0_s42`、`iid_rms1_s42` 各 25 步完成；单组约 6 秒 |
| RGB 定位误差 | `object_warp_rms1_s42` 的 24 图→VGGT→预测评测→原生 TSDF→Mesh 评测完成 |
| 相机抖动误差 | `camera_jitter_rms1_s42` 的 24 张 GPU RGB→VGGT→预测评测→原生 TSDF→Mesh 评测完成 |
| 导表 | 两分支均有逐组 CSV、Markdown 和分组均值/标准差；未运行项标 pending |
| 阈值可修改 | 定向回归验证 0.5/1/2→0.75 mm 重新计分，距离文件 SHA 不变，pending 不填零 |
| 回归 | 既有几何回归 11 项、新增阈值重算回归 1 项通过 |

GPU 近点查询最初在极小三角形上出现明显数值损失；已用距离上界内 AABB 候选枚举与
float64 三角形距离细化解决，再通过自距离检查。GPU 光栅 GT 的遮挡错误也没有被当作
扫描误差继续使用。未添加 CI，未运行庞大的额外测试矩阵。

## 仅用于检查数据流的样例指标

深度分支 smoke 配置位于 `outputs/measurement_quality/error_smoke_config.json`，
单独使用 `outputs/measurement_error_smoke`，迭代数为 **25**；正式配置默认 300。

| 案例 | 全源 Chamfer L1 mm | 可观察源 Chamfer L1 mm |
|---|---:|---:|
| VGGT 原生初始化 | 3.276832 | 1.933405 |
| 25步，不用外部扫描 | 2.992545 | 1.701803 |
| 25步，无噪声外部扫描 | 2.801330 | 1.498552 |
| 25步，1 mm IID 扫描 | 2.785057 | 见原始表 |

1 mm 项略好于零误差项不支持任何“噪声有益”结论：这里只有一个 seed、25 步和
固定采样，工具未完成默认 300 步/三 seed 扫描，也未分析数值波动与优化收敛。

| RGB 案例（各1 seed） | VGGT Depth MAE mm | VGGT Depth RMSE mm | Mesh 全源 Chamfer L1 mm |
|---|---:|---:|---:|
| 1 mm 物面定位代理 | 8.026165 | 14.284829 | 3.262985 |
| 1 mm 实际相机平移 | 6.226783 | 12.327661 | 3.267594 |

这两行的 RGB 来源不同，没有完成各自零误差基线，不能据此比较模式优劣。
其目的仅是证明误差后的 RGB **确实重新经过 VGGT**，并分别生成原始预测与 Mesh 指标。

## 本地产物

- 深度分支：`outputs/measurement_error_smoke/m2_eval/{results.csv,results.md,grouped.csv}`。
- RGB 分支：`outputs/measurement_error_rgb/rgb_eval/{results.csv,results.md,grouped.csv}`。
- 两组 RGB 的 `raw_prediction.json`：在 `data/scenes/measurement_error_rgb/<case>/`。
- GPU 基准精度：`data/scenes/measurement_precision/master/geometry/gt_ray*/ray_audit.json`。
- 短流程日志：`outputs/measurement_quality/` 下 `smoke_*`、`rgb_*smoke*`、`jitter_render_smoke.log`。

完整案例目录已经出表，但数值只对应完成项；正式配置的完整扫参尚未执行。
后续修改档位、优化参数或评测阈值见 [README.md](README.md)，不需要重写实验脚本。
