# WorldExpand · Camera-ready code

[简体中文](README.zh-CN.md)

WorldExpand jointly synthesizes four novel-view panoramas from one anchor panorama and four geometric warps with validity masks. It uses Qwen-Image-Edit-2509 with LoRA and mask conditioning. This release includes dataset preparation, training, inference, and single-scene 3D Gaussian Splatting (3DGS) reconstruction.

## Reconstruct a scene from one panorama

Follow the [installation and download guide](INSTALL.md) to prepare the two environments and the public checkpoints. Set `WE_CHECKPOINT` to the author-provided Qwen checkpoint. This example uses the bundled panorama and saves per-stage logs, a 3DGS PLY, and a checkpoint.

```bash
source .venv-worldexpand/bin/activate
export WE_CHECKPOINT=/path/to/worldexpand/checkpoint
export WE_DIFIX_PYTHON="$PWD/.venv-difix/bin/python"

python scripts/run.py expand \
  --image examples/004bb8d7/start/004bb8d7/origin.png \
  --caption "$(cat examples/004bb8d7/start/004bb8d7/caption.txt)" \
  --output outputs/expand-004bb8d7 --gpu 0
```

DepthMaster estimates radial depth in ERP format. The pipeline selects camera positions using scene boundaries and renders four warp/mask pairs with gsplat. Qwen jointly synthesizes four panoramas. WorldMirror 2.0 takes 105 tangent views from the anchor and the generated panoramas, with camera parameters and an anchor depth prior. The HY-World 2.0 optimizer uses the resulting RGB views, cameras, depths, and points in a shared coordinate system.

For a short GPU smoke test, add `--height 256 --nvs-steps 3 --mirror-faces 6 --mirror-size 224 --gs-steps 3 --max-points 20000 --difix-start 0 --difix-every 1 --difix-size 256`. Use `--difix-gpu 1` to run Difix on a second GPU or `--no-difix` to skip the auxiliary loss. Use `--dry-run` to inspect commands and check resource paths, and choose a fresh output directory for a full run. See [validation results](VALIDATION.md).

Each run writes `commands.json` and `result.json`. Stage outputs are listed below.

| Output | Contents |
| --- | --- |
| `depth/radial_depth.npy` | Radial depth at the input ERP resolution, with zero for invalid pixels |
| `warp/<case>/` | Rendered warp/mask pairs, selected cameras, and the depth mesh |
| `nvs/<case>/aux_*/output.png` | Four synthesized panoramas |
| `reconstruction/tangent/` | WorldMirror tangent views, c2w/K, and anchor Z-depth priors |
| `reconstruction/worldmirror/` | Predicted cameras, depths, and point cloud |
| `gaussians/ply/`, `gaussians/ckpts/` | Optimized 3DGS PLY and checkpoints |
| `gaussians/difix_loss.jsonl` | L1 losses when Difix is enabled. The first restored samples are in `difix_samples/` |

Difix restores a 3DGS rendering after a small translation toward the nearest neighboring camera, using the training view as its reference. The restored image is a fixed supervision target, and the L1 gradient flows into the Gaussian parameters. The default weight is 0.1, applied every five steps starting at step 1,000. The loss is integrated into a runtime copy of the HY-World trainer.

## Run inference with pre-rendered warps

```bash
python scripts/run.py doctor
bash scripts/run_example.sh --gpu 0 --steps 8
```

This entry point uses the bundled four warp/mask pairs. The full inference configuration uses 40 denoising steps and 2048×1024 output.

## Configure external paths

Start from [configs/release/local.json](configs/release/local.json). Pass a copy with `--config` or `WE_CONFIG_FILE`. Relative paths resolve against the directory containing the JSON file. Paths support environment variables and `~`.

| `paths` key | Purpose |
| --- | --- |
| `python`, `accelerate` | Python interpreter and Accelerate launcher |
| `model` | Qwen-Image-Edit-2509 base model directory, with `transformer/`, `text_encoder/`, and `vae/` |
| `checkpoint` | WorldExpand checkpoint with LoRA and mask heads, or a full transformer |
| `training_config` | Matching training YAML with attention and conditioning settings |
| `start_root` | Each `<case_id>/` contains `origin.png` and an optional `caption.txt` |
| `warp_root` | Each `<case_id>/` contains `erp_aux_0..3.png` and `mask_aux_0..3.png` |
| `output`, `cache` | Inference output and model cache |
| `train_json`, `data_root`, `train_output` | Training manifest, original data, and training output |
| `accelerate_config` | Accelerate config. The release includes a single-GPU example |

```bash
python scripts/run.py infer --config /path/to/my-paths.json --gpu 0 --steps 8
WE_CONFIG_FILE=/path/to/my-paths.json bash scripts/run_example.sh --gpu 0

WE_MODEL=/path/to/Qwen-Image-Edit-2509 \
WE_CHECKPOINT=/path/to/checkpoint \
WE_OUTPUT=/path/to/output \
bash scripts/run_example.sh --gpu 0 --steps 8
```

Use `--case-id <id>` once per case. `--dry-run` checks the input files and checkpoint/YAML compatibility, then saves the resolved configuration before model loading.

Set `path_mappings` to remap existing data mount prefixes. The longest matching prefix takes precedence. Dataset preparation scripts also accept explicit CLI paths.

```json
{
  "path_mappings": {
    "/old/shared/data": "/new/shared/data"
  }
}
```

The main launcher and `source env.sh` propagate this mapping. When invoking a dataset script directly, point `WE_PATH_CONFIG` to the JSON and set `PYTHONPATH` to the repository root.

## Inference outputs and validation

The inference launcher saves `resolved.json`, `runtime.yaml`, `case_ids.txt`, `inference.log`, and `result.json`. The pre-rendered warp path produces the following layout.

```text
<output>/<case_id>/
  aux_0/output.png
  aux_1/output.png
  aux_2/output.png
  aux_3/output.png
  four_view_grid.jpg
```

The launcher marks the run as `passed` after all four images are present, decode at the expected resolution, and show nonconstant pixels. Complete cases can be skipped when resuming. The `aux_*` indices follow the warp order.

## Training and dataset preparation

Download the training copies of MoGe and `utils3d` using the [installation guide](INSTALL.md). Set `train_json`, `data_root`, and a suitable `accelerate_config` in the external JSON, then run the following commands.

```bash
WE_CONFIG_FILE=/path/to/train-paths.json bash scripts/run_train.sh --dry-run
WE_CONFIG_FILE=/path/to/train-paths.json bash scripts/run_train.sh --num-gpus 1
```

`training/train_four_view_online.py` renders geometric warps online. The launcher resolves resource paths from the JSON and writes the runtime YAML to the output directory. Set `paths.resume` to resume from a checkpoint.

Dataset scripts expose their arguments through `--help`. UE RGB inputs are 2:1 ERP images, camera positions are in centimeters, and EXR radial depths are in meters.

| Directory | Purpose |
| --- | --- |
| [`four_view_dataset/`](four_view_dataset/) | Trajectory analysis and sampling, four-view manifests, validation, captions, and scene-level splits |
| [`panorama_rendering/`](panorama_rendering/) | ERP and cubemap geometry, point cloud, mesh, and Gaussian rendering for online warps and reconstruction |
| [`dataset_utils/`](dataset_utils/) | Dataset paths, caption lookup, and depth-based sky masks |

## Dependencies and checks

The release uses Python 3.11 and PyTorch 2.7.1. [`requirements.txt`](requirements.txt) covers WorldExpand inference, reconstruction, and training. [`requirements-difix.txt`](requirements-difix.txt) covers the separate Difix environment. See the [installation guide](INSTALL.md) for pinned source and checkpoint downloads and CUDA builds, and [validation results](VALIDATION.md) for the tested configurations.

```bash
python -m unittest discover -s tests -v
python -m compileall -q reconstruction worldexpand scripts training inference panorama_rendering four_view_dataset dataset_utils
```
