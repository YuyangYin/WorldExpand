# Installation and model downloads

[简体中文](INSTALL.zh-CN.md) · [WorldExpand](README.md)

Run these commands from the repository root on Linux. You need Python 3.11, an NVIDIA driver, a CUDA 12.x Toolkit with `nvcc`, and a C++ compiler. WorldExpand uses PyTorch 2.7.1 and torchvision 0.22.1. Difix runs in its own environment with an older diffusers release. Build CUDA extensions against the PyTorch ABI in the WorldExpand environment.

## 1. Download public source code

Download the pinned source archives to the locations used by the default release configuration. [Release configuration](configs/release/local.json)

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

The upstream code comes from [DepthMaster](https://github.com/PolyU-VCLab/DepthMaster), [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0), and [Difix3D](https://github.com/nv-tlabs/Difix3D). HY-World code and weights use the Tencent HY-WORLD 2.0 Community License. [License](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/License.txt)

## 2. Create the environments

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

The 3DGS optimizer uses the HY-World fork of gsplat with MaskGaussian support. The Linux PyPI wheel for PyTorch 2.7.1 uses the CUDA 12.6 runtime. Match your FlashAttention wheel and compiled extensions to the CUDA runtime and ABI of your PyTorch build. [FlashAttention release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2)

## 3. Download model weights

The pipeline loads weights from local paths. You can set those paths in the release configuration or override them with the corresponding `WE_*` environment variables. The WorldExpand release provides two LoRA checkpoints with saved mask heads. The default is `checkpoint-6400`.

| Component | Local path | Config key | Source |
| --- | --- | --- | --- |
| Qwen base model | `weights/Qwen-Image-Edit-2509/` | `model` | [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/d3968ef930e841f4c73640fb8afa3b306a78167e) |
| WorldExpand LoRA and mask heads | `weights/WorldExpand/default/` | `checkpoint` | [codeyang123/WorldExpand](https://huggingface.co/codeyang123/WorldExpand) |
| DepthMaster | `weights/DepthMaster/depthmaster.pt` | `depthmaster_weights` | [VCLab-PolyU/DepthMaster](https://huggingface.co/VCLab-PolyU/DepthMaster/tree/add341b64ae9f90372b6483b334ac496a18fe081) |
| WorldMirror 2.0 | `weights/HY-World-2.0/HY-WorldMirror-2.0/` | `worldmirror_model` | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0/tree/d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0/HY-WorldMirror-2.0) |
| Difix | `weights/difix_ref/` | `difix_weights` | [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref/tree/d4830559772a5795c9d136302c5b197d6418d3fb) |
| VGG16 for 3DGS LPIPS | `.cache/torch/hub/checkpoints/vgg16-397923af.pth` | Set `torch_home` to `.cache/torch` | [TorchVision VGG16](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html) |

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

| File | SHA-256 |
| --- | --- |
| DepthMaster `depthmaster.pt` | `e61e617acdd0b40bd2c217def5d60b0bd05352e6ae61d940a49c2374c6c75982` |
| WorldMirror `model.safetensors` | `9fff06539d3d9e85338d7de1ffb5afffb7739fa5bf4d62b4b7c319b5ecdde54f` |

You can skip the Difix source, environment, and weights for runs with `--no-difix`.

### Full attention checkpoint

The default checkpoint uses 30 local attention layers. The full attention variant is `checkpoint-10000` and uses global attention in every layer. Its metadata sets the attention schedule when you select the checkpoint.

```bash
.venv-worldexpand/bin/hf download codeyang123/WorldExpand --revision f586ed0ccc39ac7cfa7dd9f7db1b7464cc1a2324 \
  --include 'full-attention/*' --local-dir weights/WorldExpand
export WE_CHECKPOINT="$PWD/weights/WorldExpand/full-attention"
export WE_OUTPUT="$PWD/outputs/example-full-attention"
```

## 4. Prepare training

The company purchased the training data. Commercial licensing restrictions prevent us from releasing it publicly. The training code is provided for reference only.

Use the WorldExpand environment. Call `fetch_source` from section 1 in the same shell to download the MoGe panorama utilities and their matching `utils3d` source. The training launcher adds both directories to the training subprocess's `PYTHONPATH`. DepthMaster uses the version installed in the main environment.

```bash
fetch_source microsoft/MoGe 0286b495230a074aadf1c76cc5c679e943e5d1c6 external/MoGe
fetch_source EasternJournalist/utils3d c5daf6f6c244d251f252102d09e9b7bcef791a38 external/utils3d-training
```

For different source locations, set `paths.training_moge_source` and `paths.training_utils3d_source` or the corresponding `WE_TRAINING_MOGE_SOURCE` and `WE_TRAINING_UTILS3D_SOURCE` variables. Direct calls to geometry scripts that import MoGe can use `PYTHONPATH="$PWD:$PWD/external/MoGe:$PWD/external/utils3d-training"`. [Training walkthrough](README.md#3-training)

The default training profile uses gsplat and skips external sky segmentation. Install the corresponding package if you select an alternative renderer or mask mode. Optional optimizer and logger packages include `bitsandbytes`, `deepspeed`, `prodigyopt`, and `wandb`.

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

For external sky segmentation, put `encoder.onnx` and `decoder.onnx` in `weights/zim_vit_l_2092/` and the Grounding DINO Tiny checkpoint in `weights/grounding-dino-tiny/`. Use `WE_ZIM_CHECKPOINT` and `WE_GD_CHECKPOINT` to override these paths. [ZIM ViT-L](https://huggingface.co/naver-iv/zim-anything-vitl/tree/b7f94faa0ff9d7c07199fab8ef6bf3b3a592116f/zim_vit_l_2092) · [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/main)
