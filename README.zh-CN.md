# WorldExpand

[English](README.md)

## 环境准备

参考 [安装指南](INSTALL.md) 创建 WorldExpand 和 Difix 环境并下载公开模型。在仓库根目录激活 WorldExpand 环境，设置作者提供的 checkpoint 路径和 Difix 解释器路径。

```bash
source .venv-worldexpand/bin/activate
export WE_CHECKPOINT=/path/to/worldexpand/checkpoint
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

## 1. NVS 推理

使用随仓库提供的全景图和预渲染 warp 运行推理。

```bash
python scripts/run.py infer
```

四张生成全景和预览图保存在 `outputs/example/004bb8d7/`。

## 2. 三维重建

以示例全景为输入。重建命令会先生成新视角全景，再重建三维场景。

```bash
python scripts/run.py expand \
  --image examples/004bb8d7/start/004bb8d7/origin.png \
  --output outputs/expand-004bb8d7
```

请使用新的或空的输出目录。高斯 PLY 和 checkpoint 保存在 `outputs/expand-004bb8d7/gaussians/`。

## 训练与验证

参见 [训练环境准备](INSTALL.md#4-prepare-training--准备训练)、[数据准备脚本](four_view_dataset/) 和 [验证记录](VALIDATION.md)。
