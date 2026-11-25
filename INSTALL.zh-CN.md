# 安装与模型下载

[English](INSTALL.md) · [WorldExpand 中文首页](README.zh-CN.md)

在 Linux 上从仓库根目录执行以下命令。需要 Python 3.11、NVIDIA 驱动、包含 `nvcc` 的 CUDA 12.x Toolkit 和 C++ 编译器。主环境使用 PyTorch 2.7.1 和 torchvision 0.22.1。Difix 使用独立环境及旧版 diffusers。CUDA 扩展需与主环境的 PyTorch ABI 一致。

## 1. 下载公开源码

按固定的提交或标签下载源码归档，保存到默认配置对应的路径。[发布配置](configs/release/local.json)

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

上游源码包括 [DepthMaster](https://github.com/PolyU-VCLab/DepthMaster)、[HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) 和 [Difix3D](https://github.com/nv-tlabs/Difix3D)。HY-World 的源码和权重适用腾讯 HY-WORLD 2.0 社区许可。[许可条款](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/License.txt)

## 2. 创建运行环境

```bash
python3.11 -m venv .venv-worldexpand
.venv-worldexpand/bin/python -m pip install torch==2.7.1 torchvision==0.22.1
.venv-worldexpand/bin/python -m pip install -r requirements.txt \
  ./external/diffusers ./external/utils3d setuptools wheel packaging
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
# Hopper/H20 为 9.0。其他显卡请填写对应计算能力，或由 nvcc/PyTorch 自动检测。
export TORCH_CUDA_ARCH_LIST=9.0
gsplat_dir=external/HY-World-2.0/hyworld2/worldgen/third_party/gsplat_maskgaussian
.venv-worldexpand/bin/python -m pip install --no-build-isolation \
  ./external/fused-ssim "$gsplat_dir"
# 编译 FlashAttention 2，或采用与 PyTorch/CUDA ABI 匹配的官方 wheel。
.venv-worldexpand/bin/python -m pip install --no-build-isolation flash-attn==2.8.0.post2
.venv-worldexpand/bin/python -m pip check

python3.11 -m venv .venv-difix
.venv-difix/bin/python -m pip install -r requirements-difix.txt
.venv-difix/bin/python -m pip check
```

3DGS 优化器使用 HY-World 附带的 MaskGaussian gsplat。PyPI 上面向 Linux 的 PyTorch 2.7.1 wheel 使用 CUDA 12.6 运行时。请按 PyTorch 的 CUDA 版本和 ABI 选择 FlashAttention wheel，并在主环境中编译 CUDA 扩展。[FlashAttention 发行页](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2)

## 3. 下载模型权重

程序从本地加载权重。可在发布配置中指定路径，也可通过对应的 `WE_*` 环境变量覆盖。WorldExpand 发布了两套 LoRA 和掩码分支权重，默认使用 `checkpoint-6400`。

| 组件 | 本地路径 | 配置键 | 来源 |
| --- | --- | --- | --- |
| Qwen 基座 | `weights/Qwen-Image-Edit-2509/` | `model` | [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/d3968ef930e841f4c73640fb8afa3b306a78167e) |
| WorldExpand LoRA 和掩码分支 | `weights/WorldExpand/default/` | `checkpoint` | [codeyang123/WorldExpand](https://huggingface.co/codeyang123/WorldExpand) |
| DepthMaster | `weights/DepthMaster/depthmaster.pt` | `depthmaster_weights` | [VCLab-PolyU/DepthMaster](https://huggingface.co/VCLab-PolyU/DepthMaster/tree/add341b64ae9f90372b6483b334ac496a18fe081) |
| WorldMirror 2.0 | `weights/HY-World-2.0/HY-WorldMirror-2.0/` | `worldmirror_model` | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0/tree/d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0/HY-WorldMirror-2.0) |
| Difix | `weights/difix_ref/` | `difix_weights` | [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref/tree/d4830559772a5795c9d136302c5b197d6418d3fb) |
| 3DGS LPIPS 所用的 VGG16 | `.cache/torch/hub/checkpoints/vgg16-397923af.pth` | `torch_home` 指向 `.cache/torch` | [TorchVision VGG16](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html) |

```bash
hf=.venv-worldexpand/bin/hf
"$hf" download Qwen/Qwen-Image-Edit-2509 --revision d3968ef930e841f4c73640fb8afa3b306a78167e \
  --local-dir weights/Qwen-Image-Edit-2509
"$hf" download codeyang123/WorldExpand --revision f586ed0ccc39ac7cfa7dd9f7db1b7464cc1a2324 \
  --include 'default/*' --local-dir weights/WorldExpand
"$hf" download VCLab-PolyU/DepthMaster depthmaster.pt --revision add341b64ae9f90372b6483b334ac496a18fe081 \
  --local-dir weights/DepthMaster
"$hf" download tencent/HY-World-2.0 HY-WorldMirror-2.0/config.json HY-WorldMirror-2.0/model.safetensors \
  --revision d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0 --local-dir weights/HY-World-2.0
"$hf" download nvidia/difix_ref --revision d4830559772a5795c9d136302c5b197d6418d3fb \
  --local-dir weights/difix_ref
mkdir -p .cache/torch/hub/checkpoints
curl -fL --retry 2 https://download.pytorch.org/models/vgg16-397923af.pth \
  -o .cache/torch/hub/checkpoints/vgg16-397923af.pth
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

| 文件 | SHA-256 |
| --- | --- |
| DepthMaster `depthmaster.pt` | `e61e617acdd0b40bd2c217def5d60b0bd05352e6ae61d940a49c2374c6c75982` |
| WorldMirror `model.safetensors` | `9fff06539d3d9e85338d7de1ffb5afffb7739fa5bf4d62b4b7c319b5ecdde54f` |

使用 `--no-difix` 时可以跳过 Difix 的源码、独立环境和权重。

### Full attention 权重

默认 checkpoint 使用 30 层局部注意力。Full attention 版为 `checkpoint-10000`，所有层均使用全局注意力。选择该 checkpoint 后，推理入口会从其元数据读取对应的注意力配置。

```bash
.venv-worldexpand/bin/hf download codeyang123/WorldExpand --revision f586ed0ccc39ac7cfa7dd9f7db1b7464cc1a2324 \
  --include 'full-attention/*' --local-dir weights/WorldExpand
export WE_CHECKPOINT="$PWD/weights/WorldExpand/full-attention"
export WE_OUTPUT="$PWD/outputs/example-full-attention"
```

## 4. 准备训练

四视图训练使用 WorldExpand 主环境。在执行第 1 节的同一个 shell 中，用 `fetch_source` 下载 MoGe 全景工具及其兼容的 `utils3d` 源码。训练入口将两者加入训练子进程的 `PYTHONPATH`。DepthMaster 使用主环境安装的版本。

```bash
fetch_source microsoft/MoGe 0286b495230a074aadf1c76cc5c679e943e5d1c6 external/MoGe
fetch_source EasternJournalist/utils3d c5daf6f6c244d251f252102d09e9b7bcef791a38 external/utils3d-training
```

如果源码放在自定义目录，请通过 `paths.training_moge_source`、`paths.training_utils3d_source` 或对应的 `WE_TRAINING_MOGE_SOURCE`、`WE_TRAINING_UTILS3D_SOURCE` 指定。直接调用依赖 MoGe 的几何脚本时可设置 `PYTHONPATH="$PWD:$PWD/external/MoGe:$PWD/external/utils3d-training"`。[训练步骤](README.zh-CN.md#3-训练)

默认训练配置使用 gsplat 并跳过外部天空分割。切换到相应渲染器或掩码模式时，安装对应扩展。可选的优化器和日志组件包括 `bitsandbytes`、`deepspeed`、`prodigyopt` 和 `wandb`。

```bash
# PyTorch3D 点云渲染
fetch_source facebookresearch/pytorch3d 50f8efa1cbfbde1a8e70efda53756f7b9cea67a7 external/pytorch3d
.venv-worldexpand/bin/python -m pip install --no-build-isolation ./external/pytorch3d

# nvdiffrast 网格渲染
fetch_source NVlabs/nvdiffrast v0.3.3 external/nvdiffrast
.venv-worldexpand/bin/python -m pip install --no-build-isolation ./external/nvdiffrast

# 可选的 ZIM 天空分割
.venv-worldexpand/bin/python -m pip install zim_anything==0.1
```

外部天空分割需要将 `encoder.onnx`、`decoder.onnx` 放到 `weights/zim_vit_l_2092/`，将 Grounding DINO Tiny 权重放到 `weights/grounding-dino-tiny/`。可用 `WE_ZIM_CHECKPOINT` 和 `WE_GD_CHECKPOINT` 覆盖路径。[ZIM ViT-L](https://huggingface.co/naver-iv/zim-anything-vitl/tree/b7f94faa0ff9d7c07199fab8ef6bf3b3a592116f/zim_vit_l_2092) · [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/main)
