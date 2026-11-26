# WorldExpand: Sparse joint panorama generation for 3D world expansion

Yuyang Yin<sup>1,2</sup>, Chenjie Cao<sup>2,†</sup>, Xuhui Zuo<sup>2</sup>, Yuning Gong<sup>2</sup>, Pengfei Wang<sup>2</sup>, Junta Wu<sup>2</sup>, Zhenwei Wang<sup>2</sup>, Yao Zhao<sup>1</sup>, Tengfei Wang<sup>2∗</sup>, Yunchao Wei<sup>1∗</sup>, Chunchao Guo<sup>2∗</sup>

<sup>1</sup> Beijing Jiaotong University · <sup>2</sup> Tencent Hunyuan

[English](README.md)

## 环境准备

创建 WorldExpand 和 Difix 环境并下载公开模型。在仓库根目录激活 WorldExpand 环境。示例使用安装指南下载的默认 WorldExpand checkpoint。运行重建前设置 Difix 解释器路径。[中文安装指南](INSTALL.zh-CN.md) · [模型权重](https://huggingface.co/codeyang123/WorldExpand)

```bash
source .venv-worldexpand/bin/activate
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

## 1. NVS 推理

默认运行项目主页上的四个场景，目录名为 `scene-01`、`scene-03`、`scene-12`、`scene-13`。每个场景的 2:1 ERP 锚点全景图位于 `examples/start/<case-id>/origin.png`，四组预渲染 warp（`erp_aux_0.png` 至 `erp_aux_3.png`）及对应的有效区域掩码（`mask_aux_0.png` 至 `mask_aux_3.png`）位于 `examples/warps/<case-id>/`。这些 warp 使用发布流程从对应锚点图生成。每张锚点图旁都提供了 `caption.txt`，可直接编辑其中的文本提示。[示例输入](examples/) · [项目主页](https://yuyangyin.github.io/WorldExpand_web/#results)

```bash
python scripts/run.py infer
```

| 输出 | 内容 |
| --- | --- |
| `outputs/example/<case-id>/aux_0/output.png` 至 `aux_3/output.png` | 按输入 warp 顺序排列的四张新视角全景 |
| `outputs/example/<case-id>/four_view_grid.jpg` | 锚点图、warp 与生成图的对照预览 |
| `outputs/example/inference.log` 和 `outputs/example/result.json` | 推理日志和运行状态 |

## 2. 三维重建

输入一张 2:1 ERP 全景图，并指定新的或空的输出目录。完整流程会准备 warp、运行 NVS，再完成场景重建。如果输入图同目录有 `caption.txt`，流程会读取其中的文本提示。

```bash
python scripts/run.py expand \
  --image examples/start/scene-01/origin.png \
  --output outputs/expand-scene-01
```

| `outputs/expand-scene-01/` 下的输出 | 内容 |
| --- | --- |
| `depth/` 和 `warp/` | 锚点径向深度、渲染得到的 warp 与掩码、相机元数据 |
| `nvs/` | 四张生成的全景图 |
| `reconstruction/` | 中间相机参数、深度图和点云 |
| `gaussians/ply/` 和 `gaussians/ckpts/` | 优化后的 3D 高斯模型和 checkpoint |
| `commands.json`、`result.json` 和阶段日志 | 运行命令、状态及产物路径 |

## 3. 训练

准备训练环境和四视图 JSON 清单。每个样本包含一张锚点全景、四张目标视角全景及其相机位姿。锚点还需提供相机元数据及 EXR 径向深度。训练时在线渲染 warp。数据准备阶段可从已有文本文件补齐训练清单中的 caption。[Caption 补全脚本](four_view_dataset/step7_fill_caption.py) · [训练依赖](INSTALL.zh-CN.md#4-准备训练) · [数据准备脚本](four_view_dataset/)

在默认发布配置中，将 `paths.train_json` 指向训练清单，将 `paths.data_root` 指向其中引用的数据所在目录。在已激活的 WorldExpand 环境中启动训练。[发布配置](configs/release/local.json)

```bash
bash scripts/run_train.sh
```

`outputs/train/` 包含解析后的训练配置、TensorBoard 日志和 `checkpoint-*` 目录。训练得到的 LoRA 权重与掩码分支保存在 checkpoint 中。

## OUR NEXT PROJECT

<img src="assets/tenon-logo.png" alt="Tenon logo" width="160">

### Tenon

Planning, Generating, and Assembling Composable 3D Worlds

Code and paper coming soon.

[项目网站](https://yuyangyin.github.io/Tenon_Web/)
