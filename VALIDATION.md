# 发布验证 · 2026-11-25

用当时随包附带的 `004bb8d7` 在 NVIDIA H20 上运行 `scripts/run.py expand`。从单张全景
开始，DepthMaster、选点与 warp、四视图 NVS、WorldMirror 2.0、3DGS＋Difix 五阶段
均通过。DepthMaster 和 WorldMirror 权重的 SHA-256 均与官方记录一致。

| 检查 | 实测结果 |
| --- | --- |
| 发布目录与 Python 语法 | 68 个 Python 文件通过扫描 |
| CPU 单测 | 17 项通过 |
| 新主环境 | Python 3.11 venv 从零安装，重新编译 MaskGaussian gsplat、fused-ssim 和 FlashAttention 2，`pip check` 通过 |
| 新 Difix 环境 | Python 3.11 venv 从零安装，`pip check` 通过。提供的 Difix 环境归档另做了单步推理验证 |
| DepthMaster 与 warp | 深度为 1920×960，有效像素 83.6%。四组 warp/mask 通过检查 |
| 四视图 NVS | 3 步生成四张 512×256 全景，各张均通过解码、尺寸和像素变化检查 |
| WorldMirror 2.0 | 30 张 224×224 切面，30 组相机、30 张深度图和 947,209 个有效点 |
| 3DGS＋Difix | 5 步优化，第 0–4 步的 Difix L1 均为有限正值。导出 19,028 个 Gaussian，逐字段检查 PLY 和重载后的 checkpoint 参数均为有限值 |
| 默认分辨率和切面数 | 2048×1024 的 NVS 生成四张全景。公开 WorldMirror 处理 105 张 504×504 切面，其输出再接入 5 步 3DGS＋Difix，导出 19,027 个 Gaussian |
| 3DGS 关闭 Difix | 使用已有 105 切面数据做了 3 步独立验证，导出 PLY 和 checkpoint |

串行测试使用较小的切面数和短步数验证连通性。默认分辨率的路径也采用 3 步 NVS 和
5 步 3DGS，因此最终渲染仍较模糊。复现步骤见 [INSTALL.md](INSTALL.md) 和
[README.md](README.md)。实测包版本见 [环境记录](environment-tested.json)。

## 既有 NVS 验证 · 2026-11-19

使用当时随附的预渲染 warp/mask 和自训 checkpoint-6400 完成 8 步四视图联合生成。
四张 2048×1024 输出通过解码、尺寸和像素变化检查。此路径对应发布入口 `infer`。

## 目录重命名与依赖合并复测 · 2026-11-25

- 用合并后的 `requirements.txt` 和独立的 `requirements-difix.txt` 创建两个 Python 3.11 venv。两个环境均通过 `pip check`。主环境重装 MaskGaussian gsplat，并使用已验证的同 ABI FlashAttention 2 和 fused-ssim wheel。训练数据依赖 `datasets==4.0.0` 与兼容 NumPy 1.26 的 `pyarrow==21.0.0` 已验证可导入及读取小型数据集。
- 发布检查扫描了 68 个 Python 文件，17 项单测和 10 个四视图数据脚本的 `--help` 均通过。训练入口加载了指定的 MoGe 和训练版 `utils3d`。H20 上验证了 `skip` 和 `depth_v2` 两条在线 warp 渲染路径。
- 使用两套新环境复跑 `004bb8d7` 的五阶段流程。NVS 按 512×256、3 步生成四张全景，WorldMirror 使用 30 张 224×224 切面，3DGS 优化 3 步。3 次 Difix L1 均为有限正值，导出并检查了 19,066 个 Gaussian 的 PLY 及 checkpoint。
