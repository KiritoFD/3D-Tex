# VGGT-Ω 接入设计：与旧 VGGT 的接口区别

**尚未实现。不要把当前 `scripts/run_vggt.py` 误认为 Ω 推理。** 旧版 `vggt/` vendored 仍用于兼容性 smoke tests；下一步单独增加 `vggt_omega/` 平级源码精简镜像（不使用 submodule），而不是覆盖原目录。

官方：[facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega)；[project](https://vggt-omega.github.io/)，CVPR 2026 Oral / Best Paper Finalist，Oxford VGG / Meta AI。

## 已核实 API

```python
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
images = load_and_preprocess_images(paths, image_resolution=512).to("cuda")
with torch.inference_mode():
    predictions = model(images)
extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"], predictions["images"].shape[-2:])
depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
```

不要假设存在独立 `predictions["point_map"]`：Ω 官方将点坐标监督通过相机/深度反投影实现，减少了独立 dense head；本项目应该用 K/T/depth 生成点云。所有 shape 和坐标符号须在模型实际运行后记录（代码块仅为 API 示例）。

## 必须适配的部分

1. **权重与许可**：HF `facebook/VGGT-Omega` 有访问申请；`1B-512` 是推荐 in-the-wild 应用 checkpoint、`1B-416-Reproduction` 是 2026-09 重训 benchmark checkpoint。记录确切 SHA，不能用不同版本混报论文复现。
2. **预处理**：`load_and_preprocess_images` 的 balanced resize 会随长宽比改变输入尺寸（官方 624×416 示例）；保留原始像素到模型像素的仿射关系，不能只按 512 比例盲缩放 K。
3. **K/T**：确认 `encoding_to_camera` 返回 extrinsics 是 world->camera，前面乘法矩阵格式（行/列），左右手系；实测一组像素 round-trip、cross-view reprojection、物体是否前方。
4. **深度**：检查 z-depth 还是沿 ray 距离，验证 `X_cam = D*K^{-1}[u,v,1]` 与官方 unproject helpers 一致；正确处理 padding mask。
5. **置信度**：`depth_conf` 的单调方向/数值范围待实际 checkpoint 标定。先保存 raw 和重投影误差分桶，再决定阈值或可微权重。
6. **图像切分**：读取现有 `data/scenes/<scene>/images/`（仅 train），不读取 `heldout/`、`ground_truth/`；导出 `geometry/omega/*.npy` 等到新目录，不要覆盖旧 VGGT sparse。
7. **可选 BA**：不要默认加旧 `demo_colmap.py` 的 VGGSfM BA；Ω 原生输出足够先验证 direct depth TSDF。若对照 BA，标出额外 tracker 权重、算力和泛化差异。

## GPU/MFU 与资源

官方在 A100 上 end-to-end inference memory（输入 624x416）：
| 帧数 | 1 | 10 | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| peak GPU GB | 6.02 | 6.67 | **7.80** | 9.66 | 13.37 | 20.82 |

数字是**官方 A100 配置，含权重与前向**，不是 4090 实测。4090 24GB 对 12–25 张视图有理论余量，但应实测、独立记录 pre/post cache、dtype、PyTorch 版本、峰值显存、时间。

## 最小接受测试

- 3 张**合成已知相机**图：检查坐标序列化、intrinsics 反变换（不要求模型预测正确 GT）。
- 合成几何 `unproject -> project` 最大/均值像素误差、depth 正值与有限性、相机前方检查。
- ETH3D pipes 12 training views：可执行 Ω 导出；显存与 shape；**不使用 heldout**。
- 读保存的输出在 CPU 上跑多视图一致性 + TSDF→Mesh；保证 decoder 不依赖 GPU。
- 再逐步增加 25 视图的 relief/kicker 和物体场景。不能用这几个结构检查冒充论文 benchmark。

目前严格状态：只有旧版 VGGT 运行包装器与 CPU 数据测试。真正的 Ω adapter、geometry TSDF、UV pipeline 均是后续实现项。
