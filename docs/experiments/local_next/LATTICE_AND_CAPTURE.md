# 本地第二轮：抑制碎皱、固定 Probe 贡献与采集对照

更新：[固定 Probe 完成结果、32 图精确复跑和采集矩阵](CAPTURE_RESULTS.md)。
下文排队/21次的时间快照已被该结果取代；原记录保留用于审计。

2026-11-03。上一轮逐顶点 B1 虽提高 F/R，却明显增加碎皱，本轮保持其损失与
1-voxel 容差，仅改变位移参数化。目标仍要求薄结构保真，不单以 F 达标验收。

## 三线性位移网格：已跑完

在初始 Mesh 包围盒外留一圈控制点，学习以 voxel 为单位的三维位移。
顶点通过固定坐标三线性采样共享位移；连接关系不变。不加入 GT 法线或 RGB loss。
每视图梯度先累积到顶点，再一次回传到位移场，避免保留 16 份采样计算图。
原逐顶点模式保持同一损失定义和归一化。

先试 8/16 voxel 间距；8 的 Recall 未达目标后，再试 4。以下全为 300 步、256²。

| 模式 | F@1vox | P | R | Sharp P | Sharp R | Mesh 法线° | Hole fill | val16 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v35c30+conf | 0.63153 | 0.58076 | 0.69202 | 0.33922 | 0.69350 | 50.44 | 0.98097 | 0.93616 |
| 原逐顶点 B1 | 0.73685 | 0.68764 | 0.79364 | 0.43941 | 0.76895 | 53.19 | 0.95784 | 0.95453 |
| lattice 16 | 0.67297 | 0.67124 | 0.67470 | 0.41204 | 0.63718 | 49.57 | 0.95873 | 未测 |
| lattice 8 | **0.70239** | 0.70746 | 0.69740 | 0.45309 | **0.66643** | 49.56 | 0.93033 | 0.94270 |
| lattice 4 | **0.70413** | 0.71488 | 0.69370 | 0.47138 | **0.66390** | 49.48 | 0.90837 | 0.95056 |

8 voxel：28,188 个参数，32.08 秒，显存 0.160 GiB；16 voxel：5,808 个参数，
30.75 秒，0.160 GiB；4 voxel：176,418 个参数，32.14 秒，0.162 GiB。
参数量增加没有恢复 Sharp R。法线指标与上一轮一致，均为 Mesh→源三角形指标。

**结论：碎皱被抑制，但薄结构召回下降，未通过完整验收。**
整体 F≥0.70 不能掩盖 R<0.705，以及 Sharp R 从 0.69350 降到约 0.666。
可视化 `outputs/m2/visual/lattice_comparison.png`（左 baseline、中逐顶点 B1、右 lattice8）
确认 lattice 不再产生 B1 的密集碎皱，但未恢复 studs、spokes 和大部分孔洞。
不把任何一个配置替换为主线默认；这不是“网格已经修好”。

证据：`outputs/m2/{lattice_eval,lattice_eval_fine}/surface_metrics.json`、对应
`ablation_report.json`、`local_L{4,8,16}_256/m2_log.json`。

### 预测相机模式的独立对照

仍用上一轮原生预测 TSDF，输入 alpha 仅作为已有分割标注；不读取 GT 相机/深度/ROI。
GT 路径继续指向不存在的目录。native lattice8：300 步、28.82 秒、0.146 GiB。

| 原生模式 | F | P | R | Mesh 法线° | Hole fill | val16 IoU |
|---|---:|---:|---:|---:|---:|---:|
| native TSDF | 0.60675 | 0.55510 | 0.66900 | 50.81 | 0.98706 | 0.93299 |
| native 逐顶点 B1 | 0.72687 | 0.68822 | 0.77012 | 52.12 | 0.94642 | 0.95322 |
| native lattice8 | 0.69061 | 0.69430 | 0.68696 | **49.41** | 0.92813 | 0.94986 |

native lattice 的 Sharp R=0.65018，仍不是薄结构修复。不能把 native 与 GT 辅助初始化
混为同一消融。证据 `outputs/m2/prediction_lattice_eval`、`lattice_heldout/metrics.json`。

## 输入 LOO：资源修复，没有改变数值协议

本轮先增加 r_23、r_24、r_30 的正确 bf16 导出。r_32 随后因主机 MemoryError 终止：
推理已完成，错误发生在最后读取 16 MB 缓冲计算 checkpoint SHA256，而非模型质量判断。
已将 hash 移到模型分配之前，采用小缓冲 `hashlib.file_digest`，并在导出前释放 GPU 模型。
未完成的 r_32 缺少 cameras/manifest，重新推理，未将部分深度冒充完整导出。

另一个实测开销是父协调进程：它长期保留数值库和评测内存约 1.53 GiB，导致
8 GiB 主机提交额度门槛反复等待。现在 `loo --phase run` 延迟加载数值库，
将每次评测放到短进程，父进程约 0.01 GiB。保留原 8 GiB / 5 GiB 资源门槛；
checkpoint SHA、bf16、head fp32、512、seed42、输入顺序及固定 Probe 均未改变。
新进程成功恢复；截至 22:16 的检查为 21/24，不能把此快照当作最终完成数。
完成后会自动生成 `loo_report.json` 及评分相关性。

当前日志：`outputs/viewsel/loo_probe8_v3.{stdout,stderr}.log`。
后续导出新增 commit、入口代码 SHA、torch/CUDA/GPU 和逐输入 SHA；
历史导出缺少这些字段的事实保留，不回填臆测的代码版本。

## 已准备并排队的采集小矩阵

入口 `scripts/experiments/local_capture.py`，配置 `configs/experiments/local_capture.json`。
每组固定 16 输入、同样 8 Probe；相机 Sim3 及深度评估只拟合/评价固定 Probe，
Mesh 融合仍用该组所有输入和固定 P0 网格、观测 GT 点。

| 组 | 仅改变什么 | 状态/边界 |
|---|---|---|
| white16 | 原始 anchor16、原白色透明背景处理 | 图像字节不变，复用已有 512 预测作为参考 |
| gray16 | 合成灰色背景 RGB128 | 同视角/前景；使用输入 alpha 注释合成，不是真实重拍 |
| static_texture16 | 同图像坐标的 32px 棋盘背景 64/192 | 背景代理，不代表转台、真实世界背景或照明变化 |
| near_baseline16 | 8 Probe + 离 Probe 相机中心较近的 8 张 | 从 AB32 的预测相机选择，无 GT 选择 |
| diverse_baseline16 | 8 Probe + max-min 中心距离的 8 张 | 互补视角补充；共同 Probe 本身已有大基线，不是纯旋转/全局小视差实验 |
| white16_768 | 同一 16 张、同 checkpoint，将模型输入 512→768 | 显存/外推可行性实验；训练分辨率外推不能当原论文协议 |
| res256/400/512 | 复用已有低采集分辨率、固定模型输入512的预测 | 不重新推理；在共同 Probe 协议重评，区别于上一行模型分辨率 |

准备阶段已验证：所有组包含同样 Probe；实心前景像素变化为 0；白色参考与 768 组
输入 SHA 相同。棋盘被物体遮挡后可见背景均值 127.656–128.251，不声称精确等均值；
边界 alpha 的混合色会改变，必须将此限制纳入解释。
选择规则使用已计算的 AB32 预测作为 warm-up，不能声称这是零额外推理的廉价 Input Selector。

数据与证据：`data/scenes/capture_next/{selection,input_hashes,background_stats}.json`。
GPU 队列 `outputs/capture_next_queue.ps1` 等待指定的 LOO 进程退出后再串行推理，避免
两个 Omega 同时占显存；进程退出可能是失败，**不能据此推断 LOO 成功**。
采集矩阵未出结果前，仅称 prepared/queued，不写收益结论。跨物体、三 seeds 仍未完成。

## 416 对照的外部前置条件

已实际尝试官方 `facebook/VGGT-Omega/vggt_omega_1b_416_reproduce.pt`，返回
Hugging Face `401 GatedRepoError`。当前授权无法下载；已请求用户在本机登录有访问权的账号，
没有索取聊天 token。保留为未完成对照，不用其他 512 权重冒充。
