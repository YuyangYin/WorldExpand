# WorldExpand

[简体中文](README.zh-CN.md)

## Environment setup

Create the WorldExpand and Difix environments and download the public models. Activate WorldExpand from the repository root. For the inference and reconstruction examples, set the paths to the author-provided checkpoint and the Difix interpreter. [Installation guide](INSTALL.md)

```bash
source .venv-worldexpand/bin/activate
export WE_CHECKPOINT=/path/to/worldexpand/checkpoint
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"
```

## 1. NVS inference

The bundled example supplies one 2:1 equirectangular panorama (`origin.png`), an optional text prompt (`caption.txt`), and four target-view warps (`erp_aux_0.png` through `erp_aux_3.png`). Each warp has a matching validity mask (`mask_aux_0.png` through `mask_aux_3.png`). [Example inputs](examples/004bb8d7/)

```bash
python scripts/run.py infer
```

| Output | Contents |
| --- | --- |
| `outputs/example/004bb8d7/aux_0/output.png` through `aux_3/output.png` | Four novel-view panoramas in input warp order |
| `outputs/example/004bb8d7/four_view_grid.jpg` | Preview of the anchor, warps, and generated views |
| `outputs/example/inference.log` and `outputs/example/result.json` | Inference log and run status |

## 2. 3D reconstruction

Start with one 2:1 equirectangular panorama and a new or empty output directory. The full pipeline prepares the warps, runs NVS, and reconstructs the scene.

```bash
python scripts/run.py expand \
  --image examples/004bb8d7/start/004bb8d7/origin.png \
  --output outputs/expand-004bb8d7
```

| Output under `outputs/expand-004bb8d7/` | Contents |
| --- | --- |
| `depth/` and `warp/` | Anchor radial depth, rendered warps and masks, and camera metadata |
| `nvs/` | Four synthesized panoramas |
| `reconstruction/` | Intermediate camera parameters, depth maps, and point cloud |
| `gaussians/ply/` and `gaussians/ckpts/` | Optimized 3D Gaussian Splatting model and checkpoints |
| `commands.json`, `result.json`, and stage logs | Commands, status, and output locations |

## 3. Training

Prepare the training environment and a four-view JSON manifest. Each case groups one anchor panorama with four target panoramas and their camera poses. The anchor provides RGB, camera metadata, and radial depth in EXR format. Training renders the warps online. [Training dependencies](INSTALL.md#4-prepare-training) · [Dataset preparation](four_view_dataset/)

In the default release config, set `paths.train_json` to the manifest and `paths.data_root` to the directory containing the referenced assets. Start training from the activated WorldExpand environment. [Release config](configs/release/local.json)

```bash
bash scripts/run_train.sh
```

`outputs/train/` contains the resolved training configuration, TensorBoard logs, and `checkpoint-*` directories with the trained LoRA weights and mask heads.
