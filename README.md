# WorldExpand: Sparse joint panorama generation for 3D world expansion

Yuyang Yin<sup>1,2</sup>, Chenjie Cao<sup>2,†</sup>, Xuhui Zuo<sup>2</sup>, Yuning Gong<sup>2</sup>, Pengfei Wang<sup>2</sup>, Junta Wu<sup>2</sup>, Zhenwei Wang<sup>2</sup>, Yao Zhao<sup>1</sup>, Tengfei Wang<sup>2∗</sup>, Yunchao Wei<sup>1∗</sup>, Chunchao Guo<sup>2∗</sup>

<sup>1</sup> Beijing Jiaotong University · <sup>2</sup> Tencent Hunyuan

[简体中文](README.zh-CN.md)

## Environment setup

Create the WorldExpand and Difix environments and download the public models. Activate WorldExpand from the repository root. The examples use the default WorldExpand checkpoint downloaded by the installation guide. Set the Difix interpreter for reconstruction. [Installation guide](INSTALL.md) · [Model weights](https://huggingface.co/codeyang123/WorldExpand)

```bash
source .venv-worldexpand/bin/activate
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

## 1. NVS inference

The default run processes four scenes from the project homepage, with case IDs `scene-01`, `scene-03`, `scene-12`, and `scene-13`. Each case has a 2:1 equirectangular anchor at `examples/start/<case-id>/origin.png` and four target-view warps (`erp_aux_0.png` through `erp_aux_3.png`) with matching validity masks (`mask_aux_0.png` through `mask_aux_3.png`) under `examples/warps/<case-id>/`. The warps were rendered from these anchors using the release pipeline. Each anchor comes with a `caption.txt` text prompt. To use your own prompt, edit that file. [Example inputs](examples/) · [Project homepage](https://yuyangyin.github.io/WorldExpand_web/#results)

```bash
python scripts/run.py infer
```

| Output | Contents |
| --- | --- |
| `outputs/example/<case-id>/aux_0/output.png` through `aux_3/output.png` | Four novel-view panoramas in input warp order |
| `outputs/example/<case-id>/four_view_grid.jpg` | Preview of the anchor, warps, and generated views |
| `outputs/example/inference.log` and `outputs/example/result.json` | Inference log and run status |

## 2. 3D reconstruction

Start with one 2:1 equirectangular panorama and a new or empty output directory. The full pipeline prepares the warps, runs NVS, and reconstructs the scene. It reads `caption.txt` beside the input panorama when present.

```bash
python scripts/run.py expand \
  --image examples/start/scene-01/origin.png \
  --output outputs/expand-scene-01
```

| Output under `outputs/expand-scene-01/` | Contents |
| --- | --- |
| `depth/` and `warp/` | Anchor radial depth, rendered warps and masks, and camera metadata |
| `nvs/` | Four synthesized panoramas |
| `reconstruction/` | Intermediate camera parameters, depth maps, and point cloud |
| `gaussians/ply/` and `gaussians/ckpts/` | Optimized 3D Gaussian Splatting model and checkpoints |
| `commands.json`, `result.json`, and stage logs | Commands, status, and output locations |

## 3. Training

Prepare the training environment and a four-view JSON manifest. Each case groups one anchor panorama with four target panoramas and their camera poses. The anchor provides RGB, camera metadata, and radial depth in EXR format. Training renders the warps online. The dataset preparation script can fill missing captions from existing caption files. [Caption backfill script](four_view_dataset/step7_fill_caption.py) · [Training dependencies](INSTALL.md#4-prepare-training) · [Dataset preparation](four_view_dataset/)

In the default release config, set `paths.train_json` to the manifest and `paths.data_root` to the directory containing the referenced assets. Start training from the activated WorldExpand environment. [Release config](configs/release/local.json)

```bash
bash scripts/run_train.sh
```

`outputs/train/` contains the resolved training configuration, TensorBoard logs, and `checkpoint-*` directories with the trained LoRA weights and mask heads.

## OUR NEXT PROJECT

<img src="assets/tenon-logo.png" alt="Tenon logo" width="160">

### Tenon

Planning, Generating, and Assembling Composable 3D Worlds

Code and paper coming soon.
