# ExMesh 接入：编译、输入适配与资产导出（2026-11-02）

上游：[Fan-Treasure/ExMesh](https://github.com/Fan-Treasure/ExMesh)（CVPR 2026）。
本仓库**不复制**它的源码，只提供适配脚本与补丁说明；ExMesh 本身克隆到仓库外。

## 1. 已验证的事实

在 WSL2（Ubuntu 24.04，CUDA 12.8，torch 2.10.0+cu128，RTX 4070 Laptop 8 GiB）上实测：

| 项 | 结果 |
|---|---|
| `simple-knn` 编译 | ✅ 需补 `#include <cfloat>` |
| `diff-plane-rasterization` 编译 | ✅ 需补 `#include <cstdint>` |
| `nvdiffrast` | ✅ 源码安装 0.4.0，CUDA context 正常 |
| `fetchMesh(mesh.ply)` 读取我们的网格 | ✅ 214k 顶点 / 386k 面，含颜色与法线 |
| 官方 COLMAP reader 读取我们生成的 `sparse/` | ✅ 12 台相机 |
| `train.py` 跑到完成 | ✅ 200 步（冒烟），拓扑 split/merge 生效 5182→9 |
| 导出 OBJ/MTL/albedo/GLB | ✅ 并经 UV 重投影渲染确认贴图正确 |

**尚未验证**：官方建议的 10,000 步收敛质量、DTU/物体级几何指标。

## 2. 上游代码的三处不兼容（我们打的补丁）

这些改动**不**进入本仓库（ExMesh 不在版本控制内），需要在使用时打：

### 2.1 GCC 13 不再隐式包含 `<cfloat>` / `<cstdint>`

```
PGSR/submodules/simple-knn/simple_knn.cu            -> #include <cfloat>
PGSR/submodules/diff-plane-rasterization/cuda_rasterizer/rasterizer_impl.h
                                                    -> #include <cstdint>
```

### 2.2 `train.py` 里 mask 不随图像缩放

`utils/camera_utils.py:loadCam()` 会把图像缩放到 `-r` 分辨率，但 `train.py:158` 读 mask 时用的是**原图尺寸**，于是

```python
depth_mask = (gt_mask > 0) & (rend_depth > 0)   # 624 vs 512 -> RuntimeError
```

补丁（`train.py`，已用 `PATCH(3D-Tex)` 标注）：

```python
gt_mask_pil = gt_mask_pil.resize((image.shape[-1], image.shape[-2]), Image.NEAREST)
```

### 2.3 判读：这是上游局限，不是我们的输入错

`gt_alpha_mask` 分支本来会跟着图像走；只有走 `train_mask/*_gtmask.png` 文件路径时才有这个问题。我们用的是文件路径，所以踩到了。

## 3. 输入目录结构（`scripts/exmesh_adapter.py` 生成）

ExMesh 的 `scene/__init__.py:54` 认 `sparse/`，`dataset_readers.py:251` 找 `mesh.ply`，
`utils/mono_utils.py:25` 读 `mono_priors/da3/<img>.npz["depth"]`，`train.py:158` 读
`train_mask/<img>_gtmask.png`：

```text
<scene>/exmesh_workdir/
├── images/DSC_0487.JPG ...          # 复制并缩放到 --image-long-side（默认 1024）
├── sparse/0/
│   ├── cameras.txt                  # 单个共享 PINHOLE（Ω 主点在中心，与 ExMesh 假设一致）
│   ├── images.txt                   # qvec/tvec，world->camera（COLMAP 约定）
│   └── points3D.txt                 # 空：ExMesh 优化网格，不需要点云
├── mesh.ply                         # ← scripts/tsdf_mesh.py 的 mesh_init.ply
├── mono_priors/da3/<img>.npz        # {"depth": Ω 深度}，不需跑 Depth Anything 3
└── train_mask/<img>_gtmask.png      # 前景 mask（一致性通过的深度 ∩ 深度分位数内）
```

**为什么写 COLMAP 文本格式而不是二进制**：`pycolmap` 的 Python API 在 3.10→4.x 之间把
`Image.id` 改成了 `image_id`，而 3.10 没有 Python 3.12 的 Windows wheel。COLMAP 的**文本**
格式十几年稳定，ExMesh 的 `read_write_model.py` 直接支持。

**尺度一致性**：Ω 的深度、Ω 的相机、由二者融合出的网格都在同一个非米制 gauge 里，
适配器**不做任何缩放**。这一点很重要——任何一处额外缩放都会破坏深度先验与网格的对齐。

## 4. 运行

```bash
# 仓库外
git clone --depth 1 https://github.com/Fan-Treasure/ExMesh.git
cd ExMesh
pip install PGSR/submodules/simple-knn --no-build-isolation        # 打 2.1 的补丁
pip install PGSR/submodules/diff-plane-rasterization --no-build-isolation
pip install git+https://github.com/NVlabs/nvdiffrast.git

# 本仓库
python3 scripts/exmesh_adapter.py --scene eth3d_kicker
python3 train.py -s <abs>/data/scenes/eth3d_kicker/exmesh_workdir \
                 -m <abs>/data/scenes/eth3d_kicker/exmesh_workdir/out -r 2
python3 scripts/export_exmesh.py \
    --checkpoint <...>/out/point_cloud/iteration_10000/model_state_dict.pt
```

**路径必须用绝对路径**：ExMesh 从当前目录解析 `-s/-m`，相对路径在 `/mnt/g` 下容易错位。

## 5. 导出为 Blender 资产（`scripts/export_exmesh.py`）

官方 `auto_mesh_iter_N.ply` **只有顶点颜色，没有 UV 索引**；UV、纹理、每角 UV 索引都在
`point_cloud/iteration_N/model_state_dict.pt` 里：

| tensor | 形状 | 用途 |
|---|---|---|
| `vertices` | N×3 | 顶点 |
| `faces` | M×3 | 面 |
| `uvs` | K×2 | UV 坐标 |
| `uv_indices` | M×3 | 每面每角的 UV 索引 |
| `texture` | 3×H×W | 纹理（默认 2560²） |
| `texture_mask` | H×W | 有效纹理区 |

导出时：

- `vt` 写 `1 - v`（ExMesh 用图像空间 UV，OBJ 用左下原点）；
- MTL 用 `Kd 1 1 1`，否则 trimesh 默认的 `Kd 0.4` 会把贴图压暗；
- 同时写 `mesh.glb`（单文件，含内嵌 PNG 纹理）。

在 Blender：`File → Import → Wavefront (.obj)`，切到 Material Preview 检查贴图，
UV Editing 里看岛与接缝；`File → Export → glTF 2.0` 出 GLB。

**V 轴方向必须目视确认**：脚本提供 `--no-flip-v` 作为退路。判据是贴图上下是否颠倒，
不能只靠数值检查。
