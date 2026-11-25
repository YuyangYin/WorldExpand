# Installation and model downloads / 安装与模型下载

Run the commands from the release root. Use Linux with an NVIDIA driver, a CUDA 12.x toolkit (including `nvcc`), and a C++ compiler. WorldExpand uses Python 3.11, PyTorch 2.7.1, and torchvision 0.22.1. Difix runs in a separate environment with an older diffusers release. Match the CUDA extensions to the main environment’s PyTorch ABI.

以下命令从发布目录运行。主流程用 Python 3.11、PyTorch 2.7.1 和 torchvision 0.22.1。
在 Linux 上准备 NVIDIA 驱动、CUDA 12.x Toolkit（含 `nvcc`）和 C++ 编译器。
主环境运行 DepthMaster、Qwen、WorldMirror 和 3DGS，独立的 Difix 环境使用旧版 diffusers，
两个进程只交换 PNG。CUDA 扩展需与主环境的 PyTorch ABI 一致。

## 1. Download public source code / 下载公开源码

The commands below download pinned source archives. Their destinations match the default release configuration.

版本固定到提交或标签，可用 Git 或以下源码归档命令。默认目录与
[configs/release/local.json](configs/release/local.json) 一致。

```bash
set -euo pipefail
fetch_source() {
  local repo="$1" revision="$2" dest="$3" archive
  archive="$(mktemp)"
  mkdir -p "$dest"
  if curl -fL --retry 2 "https://codeload.github.com/${repo}/tar.gz/${revision}" -o "$archive" &&
     tar -xzf "$archive" --strip-components=1 -C "$dest"; then
    rm -f "$archive"
  else
    rm -f "$archive"
    return 1
  fi
}

fetch_source PolyU-VCLab/DepthMaster b6a80836343438f63c1b55883a8e7da94e29e6bc external/DepthMaster
fetch_source Tencent-Hunyuan/HY-World-2.0 df9988efb87bfc0f4947eb3889411cf957478b06 external/HY-World-2.0
fetch_source nv-tlabs/Difix3D c76edc595586e16732c91ddee82f3a6d83a8a9cc external/Difix3D
fetch_source huggingface/diffusers 2f4a7177f085f7f603f736e621d553e01191db03 external/diffusers
fetch_source EasternJournalist/utils3d 3fab839f0be9931dac7c8488eb0e1600c236e183 external/utils3d
fetch_source rahul-goel/fused-ssim a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38 external/fused-ssim
gsplat_dir=external/HY-World-2.0/hyworld2/worldgen/third_party/gsplat_maskgaussian
fetch_source g-truc/glm 1.0.1 "$gsplat_dir/gsplat/cuda/csrc/third_party/glm"
```

HY-World source and weights use the Tencent HY-WORLD 2.0 Community License. The upstream repositories and license are linked below.

源码入口对应 [DepthMaster](https://github.com/PolyU-VCLab/DepthMaster)、
[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) 和
[Difix3D](https://github.com/nv-tlabs/Difix3D)。HY-World 的源码和权重适用
[Tencent HY-WORLD 2.0 Community License](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/License.txt)。

## 2. Create the two environments / 创建两个环境

```bash
python3.11 -m venv .venv-worldexpand
.venv-worldexpand/bin/python -m pip install torch==2.7.1 torchvision==0.22.1
.venv-worldexpand/bin/python -m pip install -r requirements.txt \
  ./external/diffusers ./external/utils3d setuptools wheel packaging
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
# Hopper/H20: 9.0. Change this for your GPU, or let nvcc/PyTorch detect it.
export TORCH_CUDA_ARCH_LIST=9.0
gsplat_dir=external/HY-World-2.0/hyworld2/worldgen/third_party/gsplat_maskgaussian
.venv-worldexpand/bin/python -m pip install --no-build-isolation \
  ./external/fused-ssim "$gsplat_dir"
# Build FlashAttention 2 here or use an official wheel with the same PyTorch/CUDA ABI.
.venv-worldexpand/bin/python -m pip install --no-build-isolation flash-attn==2.8.0.post2
.venv-worldexpand/bin/python -m pip check

python3.11 -m venv .venv-difix
.venv-difix/bin/python -m pip install -r requirements-difix.txt
.venv-difix/bin/python -m pip check
```

The 3DGS optimizer uses the HY-World fork of gsplat with MaskGaussian support. Use the CUDA 12.6 defaults of the Linux PyPI PyTorch 2.7.1 wheel, or match the FlashAttention wheel and extension builds to the ABI and CUDA runtime of your PyTorch build.

3DGS 优化器使用 HY-World 附带的 MaskGaussian gsplat。FlashAttention 安装可参考
[官方发行页](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2)。
PyPI 的 PyTorch 2.7.1 Linux CUDA wheel 使用 CUDA 12.6。若使用其他 PyTorch wheel，
按其 CUDA 版本与 C++ ABI 选择匹配的 FlashAttention wheel，并在同一主环境编译 CUDA 扩展。

## 3. Download model weights / 下载模型权重

All weights are loaded locally. Override locations in the release JSON or with the corresponding `WE_*` variables. Set `checkpoint` to the author-provided Qwen checkpoint directory containing `lora/`, `meta.json`, and saved mask heads.

运行时从本地读取。不同路径可在配置 JSON 里指定，或通过相应的 `WE_*` 环境变量覆盖。

| Component / 组件 | Local path / 本地路径 | Config key / 配置键 | Source / 来源 |
| --- | --- | --- | --- |
| Qwen base model | `weights/Qwen-Image-Edit-2509/` | `model` | [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/d3968ef930e841f4c73640fb8afa3b306a78167e) |
| WorldExpand LoRA and mask heads | Author-provided checkpoint directory | `checkpoint` | Provided by the authors / 作者提供 |
| DepthMaster | `weights/DepthMaster/depthmaster.pt` | `depthmaster_weights` | [VCLab-PolyU/DepthMaster](https://huggingface.co/VCLab-PolyU/DepthMaster/tree/add341b64ae9f90372b6483b334ac496a18fe081) |
| WorldMirror 2.0 | `weights/HY-World-2.0/HY-WorldMirror-2.0/` | `worldmirror_model` | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0/tree/d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0/HY-WorldMirror-2.0) |
| Difix | `weights/difix_ref/` | `difix_weights` | [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref/tree/d4830559772a5795c9d136302c5b197d6418d3fb) |
| VGG16 for 3DGS LPIPS | `.cache/torch/hub/checkpoints/vgg16-397923af.pth` | Set `torch_home` to `.cache/torch` | [TorchVision VGG16](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html) |

```bash
hf=.venv-worldexpand/bin/hf
"$hf" download Qwen/Qwen-Image-Edit-2509 --revision d3968ef930e841f4c73640fb8afa3b306a78167e \
  --local-dir weights/Qwen-Image-Edit-2509
"$hf" download VCLab-PolyU/DepthMaster depthmaster.pt --revision add341b64ae9f90372b6483b334ac496a18fe081 \
  --local-dir weights/DepthMaster
"$hf" download tencent/HY-World-2.0 HY-WorldMirror-2.0/config.json HY-WorldMirror-2.0/model.safetensors \
  --revision d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0 --local-dir weights/HY-World-2.0
"$hf" download nvidia/difix_ref --revision d4830559772a5795c9d136302c5b197d6418d3fb \
  --local-dir weights/difix_ref
mkdir -p .cache/torch/hub/checkpoints
curl -fL --retry 2 https://download.pytorch.org/models/vgg16-397923af.pth \
  -o .cache/torch/hub/checkpoints/vgg16-397923af.pth
export WE_CHECKPOINT=/path/to/worldexpand/checkpoint
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

SHA-256 for DepthMaster `depthmaster.pt`
`e61e617acdd0b40bd2c217def5d60b0bd05352e6ae61d940a49c2374c6c75982`。
SHA-256 for WorldMirror `model.safetensors`
`9fff06539d3d9e85338d7de1ffb5afffb7739fa5bf4d62b4b7c319b5ecdde54f`。
[DepthMaster 权重](https://huggingface.co/VCLab-PolyU/DepthMaster/blob/add341b64ae9f90372b6483b334ac496a18fe081/depthmaster.pt)
[WorldMirror 权重](https://huggingface.co/tencent/HY-World-2.0/blob/d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0/HY-WorldMirror-2.0/model.safetensors)

Use `--no-difix` to run with the main environment alone. In this mode you can skip the Difix source, environment, and checkpoint.

使用 `--no-difix` 时只需主环境，可以跳过 Difix 的源码、独立环境及权重。

## 4. Prepare training / 准备训练

Four-view training uses the same main environment. Call the `fetch_source` helper from section 1 in the same shell to add the public MoGe panorama utilities and their matching `utils3d` source. The training launcher adds these two directories to the training subprocess’s `PYTHONPATH`. DepthMaster continues to use the `utils3d` installed in the main environment.

四视图训练共用主环境。使用第 1 节同一终端中的 `fetch_source` 下载 MoGe 全景工具及兼容的 `utils3d` 源码。训练入口将两者加入训练子进程的 `PYTHONPATH`，DepthMaster 仍使用主环境中安装的 `utils3d`。

```bash
fetch_source microsoft/MoGe 0286b495230a074aadf1c76cc5c679e943e5d1c6 external/MoGe
fetch_source EasternJournalist/utils3d c5daf6f6c244d251f252102d09e9b7bcef791a38 external/utils3d-training
```

If you use different locations, set `training_moge_source` and `training_utils3d_source` in `paths`, or override them with `WE_TRAINING_MOGE_SOURCE` and `WE_TRAINING_UTILS3D_SOURCE`. Use the training commands in the README after configuring your data and checkpoint paths. Direct calls to geometry scripts that import MoGe can use `PYTHONPATH="$PWD:$PWD/external/MoGe:$PWD/external/utils3d-training"`.

若源码放在自定义目录，通过 `paths.training_moge_source`、`paths.training_utils3d_source` 或对应的 `WE_TRAINING_MOGE_SOURCE`、`WE_TRAINING_UTILS3D_SOURCE` 指定。配置数据和模型路径后按 README 的训练命令启动。直接调用使用 MoGe 的几何脚本时，可设置上述 `PYTHONPATH`。

The release training profile uses gsplat and skips external sky segmentation. Install the following extras only if you select the corresponding renderer or mask mode. Optional optimizer and logger packages include `bitsandbytes`, `deepspeed`, `prodigyopt`, and `wandb`.

发布训练配置默认使用 gsplat，跳过外部天空分割。仅在切换到相应渲染或掩码模式时安装以下扩展。其他优化器及日志组件按需安装。

```bash
# PyTorch3D point rendering
fetch_source facebookresearch/pytorch3d 50f8efa1cbfbde1a8e70efda53756f7b9cea67a7 external/pytorch3d
.venv-worldexpand/bin/python -m pip install --no-build-isolation ./external/pytorch3d

# nvdiffrast mesh rendering
fetch_source NVlabs/nvdiffrast v0.3.3 external/nvdiffrast
.venv-worldexpand/bin/python -m pip install --no-build-isolation ./external/nvdiffrast

# Optional ZIM sky segmentation
.venv-worldexpand/bin/python -m pip install zim_anything==0.1
```

For external sky segmentation, download `encoder.onnx` and `decoder.onnx` into `weights/zim_vit_l_2092/` and the Grounding DINO Tiny checkpoint into `weights/grounding-dino-tiny/`. Override the paths with `WE_ZIM_CHECKPOINT` and `WE_GD_CHECKPOINT`. The reconstruction example uses the validity masks from DepthMaster and WorldMirror. [ZIM ViT-L](https://huggingface.co/naver-iv/zim-anything-vitl/tree/b7f94faa0ff9d7c07199fab8ef6bf3b3a592116f/zim_vit_l_2092) · [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/main)

外部天空分割使用 `weights/zim_vit_l_2092/` 下的 `encoder.onnx` 和 `decoder.onnx`，以及 `weights/grounding-dino-tiny/` 下的 Grounding DINO Tiny 权重。可用 `WE_ZIM_CHECKPOINT` 和 `WE_GD_CHECKPOINT` 覆盖路径。单场景重建示例使用 DepthMaster 和 WorldMirror 输出的有效深度掩码。
