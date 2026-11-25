# WorldExpand

[简体中文](README.zh-CN.md)

## Environment setup

Follow the [installation guide](INSTALL.md) to create the WorldExpand and Difix environments and download the public models. From the repository root, activate WorldExpand and set the paths to the author-provided checkpoint and the Difix interpreter.

```bash
source .venv-worldexpand/bin/activate
export WE_CHECKPOINT=/path/to/worldexpand/checkpoint
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

## 1. NVS inference

Run inference with the bundled panorama and pre-rendered warps.

```bash
python scripts/run.py infer
```

The four generated panoramas and their preview grid are saved in `outputs/example/004bb8d7/`.

## 2. 3D reconstruction

Start from the bundled panorama. The reconstruction command runs NVS before building the 3D scene.

```bash
python scripts/run.py expand \
  --image examples/004bb8d7/start/004bb8d7/origin.png \
  --output outputs/expand-004bb8d7
```

Use a new or empty output directory. The Gaussian PLY and checkpoints are saved in `outputs/expand-004bb8d7/gaussians/`.

## Training and validation

See [training setup](INSTALL.md#4-prepare-training--准备训练), [dataset preparation scripts](four_view_dataset/), and [validation results](VALIDATION.md).
