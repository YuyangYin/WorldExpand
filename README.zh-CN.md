# WorldExpand

[English](README.md)

## 环境准备

创建 WorldExpand 和 Difix 环境并下载公开模型。在仓库根目录激活 WorldExpand 环境。运行推理和重建示例前，设置作者提供的 checkpoint 路径及 Difix 解释器路径。[中文安装指南](INSTALL.zh-CN.md)

```bash
source .venv-worldexpand/bin/activate
export WE_CHECKPOINT=/path/to/worldexpand/checkpoint
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

## 1. NVS 推理

默认运行项目主页对比区的四个场景，目录名为 `comparison-01`、`comparison-03`、`comparison-12`、`comparison-13`。每个场景的 2:1 ERP 锚点全景图位于 `examples/start/<case-id>/origin.png`，四组预渲染 warp（`erp_aux_0.png` 至 `erp_aux_3.png`）及对应的有效区域掩码（`mask_aux_0.png` 至 `mask_aux_3.png`）位于 `examples/warps/<case-id>/`。这些 warp 使用发布流程从对应锚点图生成。如需文本提示，可在锚点图旁放置 `caption.txt`。[示例输入](examples/) · [项目主页对比](https://yuyangyin.github.io/WorldExpand_web/#results)

```bash
python scripts/run.py infer
```

| 输出 | 内容 |
| --- | --- |
| `outputs/example/<case-id>/aux_0/output.png` 至 `aux_3/output.png` | 按输入 warp 顺序排列的四张新视角全景 |
| `outputs/example/<case-id>/four_view_grid.jpg` | 锚点图、warp 与生成图的对照预览 |
| `outputs/example/inference.log` 和 `outputs/example/result.json` | 推理日志和运行状态 |

## 2. 三维重建

输入一张 2:1 ERP 全景图，并指定新的或空的输出目录。完整流程会准备 warp、运行 NVS，再完成场景重建。

```bash
python scripts/run.py expand \
  --image examples/start/comparison-01/origin.png \
  --output outputs/expand-comparison-01
```

| `outputs/expand-comparison-01/` 下的输出 | 内容 |
| --- | --- |
| `depth/` 和 `warp/` | 锚点径向深度、渲染得到的 warp 与掩码、相机元数据 |
| `nvs/` | 四张生成的全景图 |
| `reconstruction/` | 中间相机参数、深度图和点云 |
| `gaussians/ply/` 和 `gaussians/ckpts/` | 优化后的 3D 高斯模型和 checkpoint |
| `commands.json`、`result.json` 和阶段日志 | 运行命令、状态及产物路径 |

## 3. 训练

准备训练环境和四视图 JSON 清单。每个样本包含一张锚点全景、四张目标视角全景及其相机位姿。锚点还需提供相机元数据及 EXR 径向深度。训练时在线渲染 warp。[训练依赖](INSTALL.zh-CN.md#4-准备训练) · [数据准备脚本](four_view_dataset/)

在默认发布配置中，将 `paths.train_json` 指向训练清单，将 `paths.data_root` 指向其中引用的数据所在目录。在已激活的 WorldExpand 环境中启动训练。[发布配置](configs/release/local.json)

```bash
bash scripts/run_train.sh
```

`outputs/train/` 包含解析后的训练配置、TensorBoard 日志和 `checkpoint-*` 目录。训练得到的 LoRA 权重与掩码分支保存在 checkpoint 中。
