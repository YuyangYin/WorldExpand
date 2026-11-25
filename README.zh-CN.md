# WorldExpand · 发布代码

[English](README.md)

WorldExpand 以 Qwen-Image-Edit-2509、LoRA 和掩码条件为基础，从一张锚点全景和四组几何 warp 与有效掩码联合生成四张目标视角全景。发布内容包括数据准备、训练、推理和单场景 3DGS 重建。

## 从单张全景重建

按照 [安装与下载](INSTALL.md) 准备两个环境和公开权重。自训 Qwen LoRA checkpoint 通过
`WE_CHECKPOINT` 提供。以下命令使用随附的全景，输出每阶段日志、优化后的 3DGS PLY 和 checkpoint。

```bash
source .venv-worldexpand/bin/activate
export WE_CHECKPOINT=/path/to/worldexpand/checkpoint
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"

python scripts/run.py expand \
  --image examples/004bb8d7/start/004bb8d7/origin.png \
  --caption "$(cat examples/004bb8d7/start/004bb8d7/caption.txt)" \
  --output outputs/expand-004bb8d7 --gpu 0
```

流程从 DepthMaster 的 ERP 径向深度开始，经边界约束选点和 gsplat 渲染得到四组 warp/mask。
Qwen 联合生成四张全景。WorldMirror 2.0 接收锚点和四张生成全景的 105 张切面，以及
对应相机和锚点深度先验。HY-World 2.0 的 3DGS 优化器读取同一坐标系下的 RGB、相机、
深度和点云。可用 `--no-difix` 关闭 Difix 辅助损失。

GPU 连通性检查可加 `--height 256 --nvs-steps 3 --mirror-faces 6 --mirror-size 224
--gs-steps 3 --max-points 20000 --difix-start 0 --difix-every 1 --difix-size 256`。
`--difix-gpu 1` 可将 Difix 放在另一张 GPU。用 `--dry-run` 查看命令和检查资源路径，
正式运行使用新的输出目录。短步数和小分辨率仅用于功能测试。实测记录见 [VALIDATION.md](VALIDATION.md)。

每次运行写入 `commands.json` 和 `result.json`。下表列出各阶段产物。

| 输出 | 内容 |
| --- | --- |
| `depth/radial_depth.npy` | 与输入 ERP 同尺寸的径向深度，0 表示无效 |
| `warp/<case>/` | 四组渲染图和有效 mask、选点相机、深度网格 |
| `nvs/<case>/aux_*/output.png` | 四张生成全景 |
| `reconstruction/tangent/` | WorldMirror 输入切面、c2w/K、锚点 Z 深度先验 |
| `reconstruction/worldmirror/` | WorldMirror 的预测相机、深度和点云 |
| `gaussians/ply/`、`gaussians/ckpts/` | 优化后的 3DGS PLY 及 checkpoint |
| `gaussians/difix_loss.jsonl` | 启用 Difix 时记录每次 L1 损失，首组修复图保存在 `difix_samples/` |

Difix 以训练视角为参考，修复朝最近相邻相机位置轻微平移后的 3DGS 渲染。
修复图作为固定监督目标，L1 损失的梯度回传到 3DGS。默认从第 1,000 步开始，
每 5 步执行一次，权重为 0.1。HY-World 的训练器通过运行时副本接入这个损失。

## 使用预渲染 warp

```bash
python scripts/run.py doctor
bash scripts/run_example.sh --gpu 0 --steps 8
```

该入口使用随附的四组 warp/mask，默认正式配置为 40 步及 2048×1024 输出。

## 外部路径配置

资源入口为 [configs/release/local.json](configs/release/local.json)。可以复制到任意位置，通过 `--config` 或 `WE_CONFIG_FILE` 加载。JSON 中相对路径始终相对于 **JSON 文件所在目录**，支持环境变量和 `~`。

| `paths` 字段 | 用途 |
| --- | --- |
| `python`, `accelerate` | 运行环境中的解释器和 launcher |
| `model` | Qwen-Image-Edit-2509 基座，包含 transformer/text_encoder/vae 等子目录 |
| `checkpoint` | 主方法 checkpoint，包含 LoRA 和 mask heads，或完整 transformer |
| `training_config` | 与 checkpoint 对应的训练 YAML；保留原始 attention 层序列与 conditioning |
| `start_root` | 每个 `<case_id>/` 下提供 `origin.png` 和可选 `caption.txt` |
| `warp_root` | 每个 `<case_id>/` 下提供 `erp_aux_0..3.png` 和 `mask_aux_0..3.png` |
| `output`, `cache` | 推理输出、模型缓存 |
| `train_json`, `data_root`, `train_output` | 训练 manifest、原始数据、训练输出 |
| `accelerate_config` | Accelerate 配置；附带单机单卡模板 |

```bash
python scripts/run.py infer --config /path/to/my-paths.json --gpu 0 --steps 8
WE_CONFIG_FILE=/path/to/my-paths.json bash scripts/run_example.sh --gpu 0

# 每个 paths 字段也可通过 WE_<大写字段名> 覆盖
WE_MODEL=/path/to/Qwen-Image-Edit-2509 \
WE_CHECKPOINT=/path/to/checkpoint \
WE_OUTPUT=/path/to/output \
bash scripts/run_example.sh --gpu 0 --steps 8
```

切换示例使用 `--case-id <id>`，可重复指定多个 ID。`--dry-run` 校验输入与 checkpoint/YAML 一致性并保存解析结果，不加载 GPU 模型。

已有数据脚本中的默认绝对路径也经过 `worldexpand.paths.resolve_resource` 解析。通过配置里的 `path_mappings` 可批量重映射已有挂载前缀；最长匹配优先。数据准备脚本还可直接使用原有的 CLI 路径参数。

```json
{
  "path_mappings": {
    "/old/shared/data": "/new/shared/data"
  }
}
```

通过 `source env.sh` 或主入口启动时会传递该映射。直接运行 Python 数据脚本时，可设置 `WE_PATH_CONFIG=/path/to/my-paths.json` 和 `PYTHONPATH` 为本仓库根目录。

## 输出与失败检查

每次运行保存 `resolved.json`、`runtime.yaml`、`case_ids.txt`、`inference.log` 和 `result.json`。预渲染 warp 模式的输出如下。

```text
<output>/<case_id>/
  aux_0/output.png
  aux_1/output.png
  aux_2/output.png
  aux_3/output.png
  four_view_grid.jpg
```

主入口会检查四张图片均存在、能够解码、尺寸正确且不是空白图，然后才把 `result.json` 标记为 `passed`。已有完整 case 可续跑跳过。生成图所在的 aux 顺序与输入 warp 顺序一致。

## 训练与数据准备

按照 [安装指南](INSTALL.md) 下载训练使用的 MoGe 和 `utils3d` 源码。在外部 JSON 中填写 `train_json`、`data_root`，并按硬件修改 `accelerate_config` 后运行以下命令。

```bash
WE_CONFIG_FILE=/path/to/train-paths.json bash scripts/run_train.sh --dry-run
WE_CONFIG_FILE=/path/to/train-paths.json bash scripts/run_train.sh --num-gpus 1
```

训练入口为 `training/train_four_view_online.py`，会在线构建几何 warp。训练运行 YAML 写入输出目录，资源路径由外部 JSON 注入。默认不自动恢复旧训练；如需恢复，在 `paths.resume` 指定 checkpoint。

数据准备脚本可通过 `--help` 查看参数。原始 RGB 为 2:1 ERP，UE 相机位置以厘米为单位，EXR 径向深度以米为单位。

| 目录 | 用途 |
| --- | --- |
| [`four_view_dataset/`](four_view_dataset/) | 统计和采样轨迹，构建四视图 JSON，检查文件、补充文本描述并按场景划分 |
| [`panorama_rendering/`](panorama_rendering/) | ERP 与 cubemap 几何，点云、网格和高斯渲染，供在线 warp 与重建使用 |
| [`dataset_utils/`](dataset_utils/) | 数据路径解析、文本描述查找和基于深度的天空掩码 |

## 依赖与验证

发布环境使用 Python 3.11 和 PyTorch 2.7.1。[`requirements.txt`](requirements.txt) 用于 WorldExpand 的推理、重建和训练，[`requirements-difix.txt`](requirements-difix.txt) 用于独立的 Difix 环境。公开源码、权重及 CUDA 扩展的安装步骤见 [INSTALL.md](INSTALL.md)，实测记录见 [VALIDATION.md](VALIDATION.md)。

```bash
python -m unittest discover -s tests -v
python -m compileall -q reconstruction worldexpand scripts training inference panorama_rendering dataset_utils four_view_dataset
```
