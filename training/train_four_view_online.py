import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from omegaconf import OmegaConf

# Silence PyTorch >=2.6 spam: every `accelerator.wait_for_everyone()` call
# (including the per-step sync barrier at line ~1303) emits a UserWarning
# "No device id is provided via `init_process_group` or `barrier`. Using the
# current device set by the user." The comment in the upstream source claims
# "warn only once" but that dedup is broken under multi-process + accelerate's
# fork/spawn lifecycle, so it fires every iteration on every rank. The warning
# is informational only -- NCCL uses the current CUDA device, which is what
# accelerate already set per rank -- so just filter it out here.
warnings.filterwarnings(
    "ignore",
    message=r".*No device id is provided via `init_process_group` or `barrier`.*",
    category=UserWarning,
)

import numpy as np
import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

import diffusers
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from training.models.custom_qwen_transformer_four_view_mask_concat import (
    CustomQwenImageTransformer2DModelFourViewMaskConcat,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    offload_models,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

def _dump_trainer_view(
    *,
    dump_dir,
    step_idx,
    samples,
    start_rgb,
    end_rgb,
    cond_img_prompt,
    warp_rgb,
    warp_mask,
    rgb_circular_pad,
    renderer,
):
    """Dump the EXACT tensors handed to the VAE (post-pad) as PNGs.

    Companion to ``OnlineWarpRenderer._dump_sample`` which writes the
    raw renderer outputs (``06_warp_rgb`` / ``07_warp_mask``) and the
    dataset-padded GT (``08_end_rgb_gt``). Those three are inconsistent
    when ``rgb_circular_pad`` is on (06/07 are raw render size, 08 is
    dataset-padded), so we additionally drop the FINAL post-pad tensors
    as ``09_*`` ~ ``13_*`` into the same per-sample folder. The seam
    region (left ``pad_w``, right ``pad_w``) should look like a clean
    wrap-around when pad is on; the file suffix encodes pad on/off.

    Files written per sample (``B`` samples per batch):
        09_trainer_start_rgb_<tag>.png  - condition into VAE (VAE path, padded if pad on)
        10_trainer_end_rgb_<tag>.png    - img into VAE (model_input, padded if pad on)
        11_trainer_cond_img_prompt_noPad.png - text tower image input (NEVER padded;
                                              fed to Qwen2.5-VL which expects natural images)
        12_trainer_warp_rgb_<tag>.png   - renderer output after pad
        13_trainer_warp_mask_<tag>.png  - renderer mask after pad
    where ``<tag>`` is ``pad`` (rgb_circular_pad=True) or ``raw`` (False).
    """
    import cv2

    step_dir = os.path.join(dump_dir, f"step_{step_idx:04d}")

    def _to_uint8(t):
        if t.dim() == 3 and t.shape[0] == 3:
            arr = t.detach().float().cpu().numpy()
            arr = (arr + 1.0) * 0.5
            arr = np.clip(arr, 0.0, 1.0).transpose(1, 2, 0)
            return (arr * 255.0).astype(np.uint8)
        if t.dim() == 3 and t.shape[0] == 1:
            arr = t.detach().float().cpu().numpy()[0]
            arr = np.clip(arr, 0.0, 1.0)
            return (arr * 255.0).astype(np.uint8)
        raise ValueError(f"_dump_trainer_view: unexpected shape {tuple(t.shape)}")

    pad_tag = "pad" if rgb_circular_pad else "raw"

    for i, s in enumerate(samples):
        case_id_raw = s.get("case_id") or renderer._case_id_fallback(s)
        case_id_safe = renderer._safe_case_id(case_id_raw)
        out_dir = os.path.join(step_dir, f"{i:02d}_{case_id_safe}")
        os.makedirs(out_dir, exist_ok=True)

        start_np = _to_uint8(start_rgb[i])
        end_np = _to_uint8(end_rgb[i])
        cond_np = _to_uint8(cond_img_prompt[i])
        warp_rgb_np = _to_uint8(warp_rgb[i])
        warp_mask_np = _to_uint8(warp_mask[i])

        cv2.imwrite(
            os.path.join(out_dir, f"09_trainer_start_rgb_{pad_tag}.png"),
            cv2.cvtColor(start_np, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(out_dir, f"10_trainer_end_rgb_{pad_tag}.png"),
            cv2.cvtColor(end_np, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(out_dir, "11_trainer_cond_img_prompt_noPad.png"),
            cv2.cvtColor(cond_np, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(out_dir, f"12_trainer_warp_rgb_{pad_tag}.png"),
            cv2.cvtColor(warp_rgb_np, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(out_dir, f"13_trainer_warp_mask_{pad_tag}.png"),
            warp_mask_np,
        )

    print(
        f"[OnlineWarp] trainer-view dump @ step {step_idx}: "
        f"start={tuple(start_rgb.shape)} end={tuple(end_rgb.shape)} "
        f"warp={tuple(warp_rgb.shape)} mask={tuple(warp_mask.shape)} "
        f"tag={pad_tag} -> {step_dir}"
    )


def _print_debug_summary(records):
    """Print an aggregated DEBUG_MODE summary across the recorded steps.

    ``records`` is the list of per-step dicts built in the training loop
    (timings / shapes / loss). Shapes are constant across steps so we print
    them once (from the last record); timings get a per-step table plus a
    mean. Main-process only; no training state is touched.
    """
    keys = ["dl_wait", "render", "prompt_enc", "vae_enc", "sync_barrier", "fwd", "bwd", "opt", "iter_wall"]

    def _row_timings(r):
        t = dict(r["timings"])
        t.setdefault("render", r["render_wallclock"])
        t["dl_wait"] = r["dl_wait"]
        t["iter_wall"] = r["iter_wall"]
        return t

    print(f"\n{'='*60}")
    print(f"[DEBUG_MODE][SUMMARY] {len(records)} step(s) recorded")
    print(f"{'='*60}")
    print("  per-step stage timings (s):")
    print("    step | " + " | ".join(f"{k}" for k in keys))
    for r in records:
        t = _row_timings(r)
        print("    " + f"{r['step']:>4}" + " | " + " | ".join(f"{t.get(k, 0.0):.3f}" for k in keys))
    print("  mean over recorded steps (s):")
    for k in keys:
        vals = [_row_timings(r).get(k, 0.0) for r in records]
        print(f"    {k:<13} = {sum(vals) / len(vals):.3f}")
    print(f"  losses: {[round(r['loss'], 5) for r in records]}")
    print("  key tensor shapes (constant across steps):")
    for kk, vv in records[-1]["shapes"].items():
        print(f"    {kk:<40} {vv}")
    print(f"{'='*60}\n")
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def _log_scheduler_verify(scheduler, flow_shift, use_dynamic_shifting):
    """Emit grep-friendly lines proving the sigma table used by training."""
    s = scheduler.sigmas.detach().float().cpu().numpy()
    n = int(scheduler.config.num_train_timesteps)
    lin = (n - np.arange(len(s))) / float(n)
    mid = len(s) // 2

    if use_dynamic_shifting:
        d_lin = float(np.max(np.abs(s - lin)))
        status = (
            "WARN_LINEAR_despite_dynamic"
            if d_lin < 1e-3
            else f"UNEXPECTED max|sigmas-linear|={d_lin:.3e}"
        )
        ref_mid = float(lin[mid])
        ref_tag = "linear"
    else:
        sh = float(flow_shift)
        ref = sh * lin / (1.0 + (sh - 1.0) * lin)
        d_ref = float(np.max(np.abs(s - ref)))
        status = "OK" if d_ref < 1e-3 else f"MISMATCH max_err={d_ref:.3e}"
        ref_mid = float(ref[mid])
        ref_tag = f"static_shift({sh:g})"

    print("[SCHEDULER_VERIFY] --- four-view training sigma table audit ---")
    print(f"[SCHEDULER_VERIFY] flow_shift={flow_shift}")
    print(f"[SCHEDULER_VERIFY] use_dynamic_shifting={use_dynamic_shifting}")
    print(f"[SCHEDULER_VERIFY] config.use_dynamic_shifting={scheduler.config.use_dynamic_shifting}")
    print(f"[SCHEDULER_VERIFY] config.shift={scheduler.config.shift}")
    print(f"[SCHEDULER_VERIFY] sigmas[mid]={s[mid]:.4f}  ref_{ref_tag}={ref_mid:.4f}  linear={lin[mid]:.4f}")
    print(f"[SCHEDULER_VERIFY] status={status}")
    print("[SCHEDULER_VERIFY] --- end ---", flush=True)


def main(args, config_path=None):
    pcd_render = getattr(args, 'pcd_render', False)
    if pcd_render:
        print("[Dataset] pcd_render flag set, but datasets_pano2pano_pcd was pruned in NVSPano-Project; "
              "falling back to default loader. Restore the pcd dataset module if needed.")

    # Four-view online-warp training ALWAYS uses the online path (PCD build +
    # gsplat render of 4 views inline in the step). The offline 4-view dataset
    # (pre-rendered warps) is intentionally not wired here.
    online_warp = getattr(args, 'online_warp', True)
    if not online_warp:
        raise ValueError(
            "train_four_view_online.py requires online_warp=True; the 4-view "
            "path is built around the online warp renderer."
        )
    from training.data.datasets_pano2pano_four_view_online import loader
    print("[Dataset] FOUR-VIEW online_warp: imported loader from "
          "training.data.datasets_pano2pano_four_view_online "
          "(four-view-v1 JSON from step4/step5; 1 shared start -> 4 warps "
          "via render_batch_multiview, PCD built once per sample).")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Startup self-check: a glance at the log confirms the effective grad-accum
    # config (num_steps + whether sync is tied to the dataloader).
    try:
        _ga_kwargs = dict(accelerator.gradient_state.plugin_kwargs)
    except Exception:
        _ga_kwargs = {}
    accelerator.print(
        f"[grad-accum cfg] gradient_accumulation_steps="
        f"{args.gradient_accumulation_steps} | "
        f"accelerator.gradient_state.plugin_kwargs={_ga_kwargs}"
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()

        os.makedirs(logging_dir, exist_ok=True)
        os.environ["logging_dir"] = logging_dir
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # Save source code for reproducibility
        source_code_dir = os.path.join(args.output_dir, "source_code")
        os.makedirs(source_code_dir, exist_ok=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        files_to_copy = [
            os.path.abspath(__file__),
            os.path.join(script_dir, "models", "custom_qwen_transformer_mask_concat.py"),
            os.path.join(script_dir, "models", "cube_map.py"),
        ]
        if config_path is not None:
            files_to_copy.append(os.path.abspath(config_path))

        for src_file in files_to_copy:
            if os.path.exists(src_file):
                shutil.copy2(src_file, source_code_dir)
                print(f"[Source Code] Saved {os.path.basename(src_file)} -> {source_code_dir}")
            else:
                print(f"[Source Code] Warning: {src_file} not found, skipped")

    # Load the tokenizers
    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=getattr(args, 'revision', None),
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models. Qwen-Image-Edit-2509's scheduler_config.json
    # enables dynamic shifting, where `shift` is not baked into `sigmas` during
    # __init__. Training reads scheduler.sigmas directly and never calls
    # set_timesteps(mu=...), so force static shifting unless explicitly changed.
    flow_shift = float(getattr(args, 'flow_shift', 1.0))
    use_dynamic_shifting = bool(getattr(args, 'use_dynamic_shifting', False))
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=getattr(args, 'revision', None),
        shift=flow_shift,
        use_dynamic_shifting=use_dynamic_shifting,
    )
    if accelerator.is_main_process:
        print(
            f"[Scheduler] FlowMatchEulerDiscrete shift={flow_shift}, "
            f"use_dynamic_shifting={use_dynamic_shifting}"
        )
        _log_scheduler_verify(noise_scheduler, flow_shift, use_dynamic_shifting)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=getattr(args, 'revision', None),
        variant=getattr(args, 'variant', None),
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(accelerator.device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=getattr(args, 'revision', None), torch_dtype=weight_dtype
    )
    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = CustomQwenImageTransformer2DModelFourViewMaskConcat.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=getattr(args, 'revision', None),
        variant=getattr(args, 'variant', None),
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    # 4-view config (from_pretrained reads the base config.json which doesn't
    # carry these keys, so set them explicitly after load).
    num_views = int(getattr(args, 'num_views', 4))
    attention_policy = str(getattr(args, 'attention_policy', 'global'))
    local_attn_variant = str(getattr(args, 'local_attn_variant', 'v1'))
    transformer.num_views = num_views
    transformer.set_attention_policy(attention_policy)
    transformer.set_local_attn_variant(local_attn_variant)
    # Optional explicit per-layer G/L schedule. ``local_layer_indices`` (list of
    # ints, or comma-separated string) sets exactly those block indices to Local
    # and the rest to Global, OVERRIDING attention_policy. Used to convert the
    # least-PSNR-sensitive layers (picked from per_layer_ablation) to local for
    # speed, instead of the fixed "trailing N" pattern the named policies give.
    _lli = getattr(args, 'local_layer_indices', None)
    if _lli is not None and str(_lli).strip() not in ("", "none", "None"):
        if isinstance(_lli, str):
            _lli = [int(x) for x in _lli.replace(" ", "").split(",") if x != ""]
        else:
            _lli = [int(x) for x in _lli]
        _n = len(transformer.transformer_blocks)
        _sched = ["G"] * _n
        for _i in _lli:
            if 0 <= _i < _n:
                _sched[_i] = "L"
        transformer.set_policy_schedule(_sched)
        print(f"[FourView] local_layer_indices override: {sorted(_lli)} "
              f"(L={_sched.count('L')}/{_n}), attention_policy={attention_policy} ignored")
    print(f"[FourView] num_views={num_views}, attention_policy={attention_policy}, "
          f"local_attn_variant={local_attn_variant}")
    # print('transformer.add_infor_in.weight:',transformer.add_infor_in.weight)
    # print('transformer.add_infor_in.bias:',transformer.add_infor_in.bias)
    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    text_encoder.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )
    transformer.to(**transformer_to_kwargs)

    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        #target_modules = ["to_k", "to_q", "to_v", "to_out.0"] #这里可以多训一点
        target_modules = ["to_k", "to_q", "to_v", "to_out.0","add_q_proj","add_k_proj","add_v_proj",'to_add_out',"img_mlp.net.2","img_mod.1"] #这里可以多训一点

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    # Mask-encoder ablation toggle (outermost). When False: the mask injection
    # is skipped in forward() and its two heads are frozen (warp stream stays,
    # mask conditioning removed) -- useful for A/B "does the mask head help?".
    use_mask_encoder = bool(getattr(args, 'use_mask_encoder', True))
    transformer.use_mask_encoder = use_mask_encoder
    transformer.mask_encoder.requires_grad_(use_mask_encoder)
    transformer.mask_proj.requires_grad_(use_mask_encoder)
    if use_mask_encoder:
        print("[Config] use_mask_encoder=True: mask head trainable, "
              "mask tokens injected into the warp slice.")
    else:
        print("[Config] use_mask_encoder=False (ablation): mask injection "
              "skipped, mask_encoder/mask_proj frozen.")

    # Plücker ray conditioning toggle (ERP route). When ``use_plucker`` is on,
    # enable the injection inside the transformer and train its two heads
    # alongside the mask head. When off, the modules stay zero / frozen and the
    # forward pass is byte-for-byte the legacy baseline.
    use_plucker = bool(getattr(args, 'use_plucker', False))
    plucker_trans_scale = float(getattr(args, 'plucker_trans_scale', 1.0))
    transformer.inject_plucker = use_plucker
    if use_plucker:
        transformer.plucker_encoder.requires_grad_(True)
        transformer.plucker_proj.requires_grad_(True)
        print(f"[Config] use_plucker=True (ERP route): plucker heads trainable, "
              f"plucker_trans_scale={plucker_trans_scale}")
    else:
        transformer.plucker_encoder.requires_grad_(False)
        transformer.plucker_proj.requires_grad_(False)
        print("[Config] use_plucker=False: plucker injection disabled (baseline path)")

    # Global camera-position conditioning ("like timestep"), two INDEPENDENT
    # toggles. Option A (use_pose_temb): pose -> temb. Option B (use_pose_film):
    # pose -> global FiLM. ``pose_scale`` normalises the relative translation.
    use_pose_temb = bool(getattr(args, 'use_pose_temb', False))
    use_pose_film = bool(getattr(args, 'use_pose_film', False))
    use_dist_temb = bool(getattr(args, 'use_dist_temb', False))
    pose_scale = float(getattr(args, 'pose_scale', 1.0))
    pose_film_streams = str(getattr(args, 'pose_film_streams', 'end_only')).strip().lower()
    if pose_film_streams not in {"all", "end_only"}:
        raise ValueError(
            f"pose_film_streams must be 'all' or 'end_only', got {pose_film_streams!r}"
        )
    use_pose = use_pose_temb or use_pose_film or use_dist_temb
    # use_warp: master switch for the whole warp pathway. When False: drop the
    # warp image + mask streams entirely (2-stream [noisy, condition] model)
    # and skip the warp render for speed. (Legacy name: use_warp_mask -- still
    # accepted for backward compat with old configs/checkpoints.)
    use_warp = bool(getattr(args, 'use_warp', getattr(args, 'use_warp_mask', True)))
    if not use_warp:
        print("[Config] use_warp=False: dropping warp+mask streams "
              "(2-stream [noisy, condition]; warp render skipped)")
    transformer.inject_pose_temb = use_pose_temb
    transformer.inject_pose_film = use_pose_film
    transformer.inject_dist_temb = use_dist_temb
    transformer.pose_film_streams = pose_film_streams
    if use_pose_temb:
        transformer.pose_temb_embed.requires_grad_(True)
        print(f"[Config] use_pose_temb=True (Option A): pose->temb head trainable")
    else:
        transformer.pose_temb_embed.requires_grad_(False)
    if use_pose_film:
        transformer.pose_film.requires_grad_(True)
        print(f"[Config] use_pose_film=True (Option B): pose FiLM head trainable, "
              f"pose_film_streams={pose_film_streams}")
    else:
        transformer.pose_film.requires_grad_(False)
    if use_dist_temb:
        transformer.dist_temb_embed.requires_grad_(True)
        print(f"[Config] use_dist_temb=True (Option C): dist->temb head trainable")
    else:
        transformer.dist_temb_embed.requires_grad_(False)

    # One-shot weight audit (POSE_DEBUG=1): dump finite/NaN stats of every new
    # head at construction time, BEFORE any training, to confirm init is sane
    # (catches the from_pretrained "_fast_init skips reset_parameters -> NaN
    # uninitialized lin1" class of bug).
    if os.environ.get("POSE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}:
        import torch as _t
        _heads = [
            ("mask_encoder.weight", transformer.mask_encoder.weight),
            ("mask_proj.weight", transformer.mask_proj.weight),
            ("plucker_encoder.weight", transformer.plucker_encoder.weight),
            ("pose_temb_embed.lin1.weight", transformer.pose_temb_embed.lin1.weight),
            ("pose_temb_embed.lin2.weight", transformer.pose_temb_embed.lin2.weight),
            ("pose_film.lin1.weight", transformer.pose_film.lin1.weight),
            ("dist_temb_embed.lin1.weight", transformer.dist_temb_embed.lin1.weight),
        ]
        print("[weight-audit] (at construction, pre-train)")
        for _n, _w in _heads:
            _wf = _w.detach().float()
            print(f"  {_n}: finite={bool(_t.isfinite(_wf).all())} "
                  f"nan={int(_t.isnan(_wf).sum())} numel={_wf.numel()} "
                  f"absmax={_wf.abs().max().item():.3e}", flush=True)
    if use_pose:
        print(f"[Config] pose conditioning ON: pose_scale={pose_scale}")
    else:
        print("[Config] use_pose_temb=use_pose_film=False: pose conditioning disabled")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def _save_lora_aux_only(output_dir, transformer_module):
        """Save LoRA + mask_encoder + mask_proj + meta.json. Cheap (~MB).

        This is the only artefact needed for *inference*. The heavy
        DeepSpeed / FSDP2 model/optimizer shards (needed for *resume*) are
        saved separately by ``accelerator.save_state(...)``; see the
        training loop's checkpoint block for the env-var gate that
        controls whether they are written.

        IMPORTANT: must be called on ALL ranks under FSDP2. Both the
        LoRA adapter params (inside the FSDP-wrapped
        QwenImageTransformerBlocks) AND the mask_encoder / mask_proj
        weights (in the outer CustomQwenImageTransformer2DModelMaskConcat
        scope — also sharded by accelerate's root-level fully_shard())
        are DTensor objects under FSDP2. We need collective
        ``DTensor.full_tensor()`` to gather every saved tensor before
        handing them to ``safetensors.save_file``, which can't read
        DTensor storage. Under DDP / DeepSpeed ZeRO-2 the params are
        already replicated per rank so the gather is a no-op. File I/O
        is rank-0 only.
        """
        from safetensors.torch import save_file

        def _gather(t):
            """Materialize a DTensor to a full local tensor (collective
            under FSDP2; no-op under DDP / DeepSpeed)."""
            if hasattr(t, "full_tensor"):
                t = t.full_tensor()
            return t

        # ---- 1) Collective gather across ALL ranks ---------------------
        # Every tensor that needs to land in a safetensors file goes
        # through full_tensor() here. The duck-check makes this a free
        # pass for non-FSDP backends.
        raw_lora = get_peft_model_state_dict(transformer_module)
        gathered_lora = {k: _gather(v) for k, v in raw_lora.items()}

        mask_encoder_w = _gather(transformer_module.mask_encoder.weight)
        mask_proj_w    = _gather(transformer_module.mask_proj.weight)
        mask_proj_b    = _gather(transformer_module.mask_proj.bias)

        # Plücker heads (only meaningful when the toggle is on; otherwise they
        # are frozen zeros and we skip writing them so old/baseline ckpts stay
        # clean and inference can detect "no plucker" by the absent dirs).
        if use_plucker:
            plucker_encoder_w = _gather(transformer_module.plucker_encoder.weight)
            plucker_proj_w    = _gather(transformer_module.plucker_proj.weight)
            plucker_proj_b    = _gather(transformer_module.plucker_proj.bias)

        # Pose heads (Options A/B). Each gathered+saved only when its toggle is on.
        if use_pose_temb:
            pose_temb_state = {k: _gather(v) for k, v in transformer_module.pose_temb_embed.state_dict().items()}
        if use_pose_film:
            pose_film_state = {k: _gather(v) for k, v in transformer_module.pose_film.state_dict().items()}
        if use_dist_temb:
            dist_temb_state = {k: _gather(v) for k, v in transformer_module.dist_temb_embed.state_dict().items()}

        # ---- 2) Rank-0-only file I/O from here on ----------------------
        if not accelerator.is_main_process:
            return

        # Convert PEFT format to Diffusers format by adding 'transformer.' prefix
        diffusers_lora_state_dict = {
            f"transformer.{k}": v.detach().cpu() for k, v in gathered_lora.items()
        }

        # 3) LoRA weights
        lora_save_path = os.path.join(output_dir, "lora")
        os.makedirs(lora_save_path, exist_ok=True)
        torch.save(diffusers_lora_state_dict, os.path.join(lora_save_path, "pytorch_lora_weights.bin"))
        save_file(diffusers_lora_state_dict, os.path.join(lora_save_path, "pytorch_lora_weights.safetensors"))

        # 4) mask_encoder
        mask_encoder_state = {"weight": mask_encoder_w.detach().cpu()}
        mask_encoder_path = os.path.join(output_dir, "mask_encoder")
        os.makedirs(mask_encoder_path, exist_ok=True)
        torch.save(mask_encoder_state, os.path.join(mask_encoder_path, "pytorch_model.bin"))
        save_file(mask_encoder_state, os.path.join(mask_encoder_path, "model.safetensors"))

        # 5) mask_proj
        mask_proj_state = {
            "weight": mask_proj_w.detach().cpu(),
            "bias":   mask_proj_b.detach().cpu(),
        }
        mask_proj_path = os.path.join(output_dir, "mask_proj")
        os.makedirs(mask_proj_path, exist_ok=True)
        torch.save(mask_proj_state, os.path.join(mask_proj_path, "pytorch_model.bin"))
        save_file(mask_proj_state, os.path.join(mask_proj_path, "model.safetensors"))

        # 5b) plucker_encoder + plucker_proj (only when use_plucker)
        if use_plucker:
            plucker_encoder_state = {"weight": plucker_encoder_w.detach().cpu()}
            plucker_encoder_path = os.path.join(output_dir, "plucker_encoder")
            os.makedirs(plucker_encoder_path, exist_ok=True)
            torch.save(plucker_encoder_state, os.path.join(plucker_encoder_path, "pytorch_model.bin"))
            save_file(plucker_encoder_state, os.path.join(plucker_encoder_path, "model.safetensors"))

            plucker_proj_state = {
                "weight": plucker_proj_w.detach().cpu(),
                "bias":   plucker_proj_b.detach().cpu(),
            }
            plucker_proj_path = os.path.join(output_dir, "plucker_proj")
            os.makedirs(plucker_proj_path, exist_ok=True)
            torch.save(plucker_proj_state, os.path.join(plucker_proj_path, "pytorch_model.bin"))
            save_file(plucker_proj_state, os.path.join(plucker_proj_path, "model.safetensors"))
            logger.info(f"Saved plucker_encoder + plucker_proj layers to {output_dir}")

        # 5c) pose heads (Options A/B), each only when its toggle is on.
        def _save_state_dir(state, name):
            state = {k: v.detach().cpu() for k, v in state.items()}
            d = os.path.join(output_dir, name)
            os.makedirs(d, exist_ok=True)
            torch.save(state, os.path.join(d, "pytorch_model.bin"))
            save_file(state, os.path.join(d, "model.safetensors"))
            logger.info(f"Saved {name} layer to {d}")
        if use_pose_temb:
            _save_state_dir(pose_temb_state, "pose_temb_embed")
        if use_pose_film:
            _save_state_dir(pose_film_state, "pose_film")
        if use_dist_temb:
            _save_state_dir(dist_temb_state, "dist_temb_embed")

        # 4) Reconstruction metadata
        meta_info = {
            "base_model_name_or_path": args.pretrained_model_name_or_path,
            "revision": getattr(args, "revision", None),
            "variant": getattr(args, "variant", None),
            "lora_rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": target_modules,
            "torch_dtype": str(weight_dtype),
            "quantization_config_path": args.bnb_quantization_config_path,
            "flow_shift": float(flow_shift),
            "scheduler_static_shift": not bool(use_dynamic_shifting),
            "use_plucker": bool(use_plucker),
            "plucker_trans_scale": float(plucker_trans_scale),
            "use_pose_temb": bool(use_pose_temb),
            "use_pose_film": bool(use_pose_film),
            "use_dist_temb": bool(use_dist_temb),
            "pose_scale": float(pose_scale),
            "pose_film_streams": str(pose_film_streams),
            "use_warp": bool(use_warp),
            # Legacy alias kept so older inference code still finds the flag.
            "use_warp_mask": bool(use_warp),
            "use_mask_encoder": bool(use_mask_encoder),
            # Four-view specifics (inference must reconstruct the same model).
            "num_views": int(num_views),
            "attention_policy": str(attention_policy),
            "local_attn_variant": str(local_attn_variant),
            # Renderer distribution knobs. Eval MUST re-render warp/mask with the
            # SAME sky_mask_mode (sky-off keeps sky in the warp & unmasked;
            # sky-cached turns sky into holes the model inpaints), else inputs are
            # OOD. yaw_align is handled separately via precompute_eval_pose.
            "sky_mask_mode": str(args.data_config.get("sky_mask_mode", "lazy")),
            "render_engine": str(args.data_config.get("render_engine", "gsplat")),
            "yaw_align_mode": str(args.data_config.get("yaw_align_mode",
                                  args.data_config.get("yaw_align", "off"))),
        }
        import json
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta_info, f, indent=2)

        logger.info(f"Saved LoRA weights to {lora_save_path}")
        logger.info(f"Saved mask_encoder layer to {mask_encoder_path}")
        logger.info(f"Saved mask_proj layer to {mask_proj_path}")
        logger.info(f"Saved metadata to {output_dir}/meta.json")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    _save_lora_aux_only(output_dir, unwrap_model(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            # Critical: clear weights to avoid accelerate's default save (重复&变大)
            while len(weights) > 0:
                weights.pop()


    def load_model_hook(models, input_dir):
        transformer_ = None

        # --- peft 0.19.1 <-> transformers 4.57.6 compat shim --------------------
        # set_peft_model_state_dict() unconditionally calls
        # _maybe_shard_state_dict_for_tp() whenever torch.distributed is
        # initialized (it is, under DeepSpeed), and that function does
        # `from transformers.integrations.tensor_parallel import EmbeddingParallel`
        # — a symbol that does NOT exist in transformers 4.57.6, so the import
        # raises ImportError on EVERY resume (fresh runs never load LoRA state so
        # they don't hit it). We don't use tensor parallelism (ZeRO-2 only), so
        # the TP-shard step is a no-op for us. Neutralize it for the duration of
        # this hook. Pure runtime monkey-patch; touches no frozen dependency.
        import peft.utils.save_and_load as _peft_sl
        _orig_maybe_shard = getattr(_peft_sl, "_maybe_shard_state_dict_for_tp", None)
        if _orig_maybe_shard is not None:
            _peft_sl._maybe_shard_state_dict_for_tp = (
                lambda model, state_dict, adapter_name: None
            )

        # DeepSpeed / FSDP 都需要"unwrap_model + 手动 set_peft_model_state_dict"
        # 路径，因为 accelerate.load_state 之后底层 model 已经被 ZeRO shards 或
        # FSDP2 DTensor 包过了，直接对 raw nn.Module 走 set_peft 会跳过分布式
        # gather。剩余的 single-GPU / 纯 DDP 才走简单加载分支。
        sharded_load = accelerator.distributed_type in (
            DistributedType.DEEPSPEED,
            DistributedType.FSDP,
        )
        if not sharded_load:
            # 非 DeepSpeed / 非 FSDP 环境：正常加载
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)

                    # 1) Load LoRA weights
                    lora_path = os.path.join(input_dir, "lora")
                    if os.path.exists(os.path.join(lora_path, "pytorch_lora_weights.bin")):
                        lora_state_dict = torch.load(
                            os.path.join(lora_path, "pytorch_lora_weights.bin"),
                            map_location="cpu"
                        )
                        # Use PEFT's function to load LoRA weights
                        set_peft_model_state_dict(model, lora_state_dict)
                        logger.info(f"Loaded LoRA weights from {lora_path}")
                    else:
                        logger.warning(f"LoRA weights not found at {lora_path}")

                    # 2) Load mask_encoder layer
                    mask_encoder_path = os.path.join(input_dir, "mask_encoder")
                    if os.path.exists(os.path.join(mask_encoder_path, "pytorch_model.bin")):
                        mask_encoder_state = torch.load(
                            os.path.join(mask_encoder_path, "pytorch_model.bin"),
                            map_location="cpu"
                        )
                        model.mask_encoder.weight.data.copy_(mask_encoder_state["weight"])
                        logger.info(f"Loaded mask_encoder layer from {mask_encoder_path}")
                    else:
                        logger.warning(f"mask_encoder layer not found at {mask_encoder_path}")

                    # 3) Load mask_proj layer
                    mask_proj_path = os.path.join(input_dir, "mask_proj")
                    if os.path.exists(os.path.join(mask_proj_path, "pytorch_model.bin")):
                        mask_proj_state = torch.load(
                            os.path.join(mask_proj_path, "pytorch_model.bin"),
                            map_location="cpu"
                        )
                        model.mask_proj.weight.data.copy_(mask_proj_state["weight"])
                        model.mask_proj.bias.data.copy_(mask_proj_state["bias"])
                        logger.info(f"Loaded mask_proj layer from {mask_proj_path}")
                    else:
                        logger.warning(f"mask_proj layer not found at {mask_proj_path}")

                    # 4) Load plucker heads (only when the toggle is on; absent
                    #    dirs => keep the zero-init weights = baseline).
                    if use_plucker:
                        plucker_encoder_path = os.path.join(input_dir, "plucker_encoder")
                        if os.path.exists(os.path.join(plucker_encoder_path, "pytorch_model.bin")):
                            _st = torch.load(os.path.join(plucker_encoder_path, "pytorch_model.bin"), map_location="cpu")
                            model.plucker_encoder.weight.data.copy_(_st["weight"])
                            logger.info(f"Loaded plucker_encoder layer from {plucker_encoder_path}")
                        else:
                            logger.warning(f"plucker_encoder layer not found at {plucker_encoder_path}")
                        plucker_proj_path = os.path.join(input_dir, "plucker_proj")
                        if os.path.exists(os.path.join(plucker_proj_path, "pytorch_model.bin")):
                            _st = torch.load(os.path.join(plucker_proj_path, "pytorch_model.bin"), map_location="cpu")
                            model.plucker_proj.weight.data.copy_(_st["weight"])
                            model.plucker_proj.bias.data.copy_(_st["bias"])
                            logger.info(f"Loaded plucker_proj layer from {plucker_proj_path}")
                        else:
                            logger.warning(f"plucker_proj layer not found at {plucker_proj_path}")

                    # 5) Load pose heads (Options A/B), each only when its toggle is on.
                    def _load_into(module, name):
                        d = os.path.join(input_dir, name)
                        bp = os.path.join(d, "pytorch_model.bin")
                        if os.path.exists(bp):
                            module.load_state_dict(torch.load(bp, map_location="cpu"), strict=False)
                            logger.info(f"Loaded {name} layer from {d}")
                        else:
                            logger.warning(f"{name} layer not found at {d}")
                    if use_pose_temb:
                        _load_into(model.pose_temb_embed, "pose_temb_embed")
                    if use_pose_film:
                        _load_into(model.pose_film, "pose_film")
                    if use_dist_temb:
                        _load_into(model.dist_temb_embed, "dist_temb_embed")

                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            # DeepSpeed / FSDP 环境：accelerator.load_state 会自动加载
            # 分片好的 base model state（ZeRO shards 或 FSDP2 sharded state
            # dict），这里只补 LoRA + mask_encoder + mask_proj。
            logger.info(
                f"{accelerator.distributed_type} detected, using sharded loading procedure"
            )

            # 获取transformer模型（已经被DeepSpeed加载）
            transformer_ = accelerator.unwrap_model(transformer)

            # 加载LoRA weights使用safetensors格式（更安全）
            lora_path = os.path.join(input_dir, "lora")
            if os.path.exists(os.path.join(lora_path, "pytorch_lora_weights.safetensors")):
                from safetensors.torch import load_file
                lora_state_dict = load_file(os.path.join(lora_path, "pytorch_lora_weights.safetensors"))
                set_peft_model_state_dict(transformer_, lora_state_dict, adapter_name="default")
                logger.info(f"Loaded LoRA weights from {lora_path}")
            elif os.path.exists(os.path.join(lora_path, "pytorch_lora_weights.bin")):
                lora_state_dict = torch.load(
                    os.path.join(lora_path, "pytorch_lora_weights.bin"),
                    map_location="cpu"
                )
                set_peft_model_state_dict(transformer_, lora_state_dict, adapter_name="default")
                logger.info(f"Loaded LoRA weights from {lora_path} (bin format)")
            else:
                logger.warning(f"LoRA weights not found at {lora_path}")

            # Load mask_encoder layer
            mask_encoder_path = os.path.join(input_dir, "mask_encoder")
            if os.path.exists(os.path.join(mask_encoder_path, "model.safetensors")):
                from safetensors.torch import load_file
                mask_encoder_state = load_file(os.path.join(mask_encoder_path, "model.safetensors"))
                transformer_.mask_encoder.weight.data.copy_(mask_encoder_state["weight"])
                logger.info(f"Loaded mask_encoder layer from {mask_encoder_path}")
            elif os.path.exists(os.path.join(mask_encoder_path, "pytorch_model.bin")):
                mask_encoder_state = torch.load(
                    os.path.join(mask_encoder_path, "pytorch_model.bin"),
                    map_location="cpu"
                )
                transformer_.mask_encoder.weight.data.copy_(mask_encoder_state["weight"])
                logger.info(f"Loaded mask_encoder layer from {mask_encoder_path} (bin format)")
            else:
                logger.warning(f"mask_encoder layer not found at {mask_encoder_path}")

            # Load mask_proj layer
            mask_proj_path = os.path.join(input_dir, "mask_proj")
            if os.path.exists(os.path.join(mask_proj_path, "model.safetensors")):
                from safetensors.torch import load_file
                mask_proj_state = load_file(os.path.join(mask_proj_path, "model.safetensors"))
                transformer_.mask_proj.weight.data.copy_(mask_proj_state["weight"])
                transformer_.mask_proj.bias.data.copy_(mask_proj_state["bias"])
                logger.info(f"Loaded mask_proj layer from {mask_proj_path}")
            elif os.path.exists(os.path.join(mask_proj_path, "pytorch_model.bin")):
                mask_proj_state = torch.load(
                    os.path.join(mask_proj_path, "pytorch_model.bin"),
                    map_location="cpu"
                )
                transformer_.mask_proj.weight.data.copy_(mask_proj_state["weight"])
                transformer_.mask_proj.bias.data.copy_(mask_proj_state["bias"])
                logger.info(f"Loaded mask_proj layer from {mask_proj_path} (bin format)")
            else:
                logger.warning(f"mask_proj layer not found at {mask_proj_path}")

            # Load plucker heads (only when the toggle is on).
            if use_plucker:
                from safetensors.torch import load_file as _load_file
                plucker_encoder_path = os.path.join(input_dir, "plucker_encoder")
                if os.path.exists(os.path.join(plucker_encoder_path, "model.safetensors")):
                    _st = _load_file(os.path.join(plucker_encoder_path, "model.safetensors"))
                    transformer_.plucker_encoder.weight.data.copy_(_st["weight"])
                    logger.info(f"Loaded plucker_encoder layer from {plucker_encoder_path}")
                elif os.path.exists(os.path.join(plucker_encoder_path, "pytorch_model.bin")):
                    _st = torch.load(os.path.join(plucker_encoder_path, "pytorch_model.bin"), map_location="cpu")
                    transformer_.plucker_encoder.weight.data.copy_(_st["weight"])
                    logger.info(f"Loaded plucker_encoder layer from {plucker_encoder_path} (bin format)")
                else:
                    logger.warning(f"plucker_encoder layer not found at {plucker_encoder_path}")
                plucker_proj_path = os.path.join(input_dir, "plucker_proj")
                if os.path.exists(os.path.join(plucker_proj_path, "model.safetensors")):
                    _st = _load_file(os.path.join(plucker_proj_path, "model.safetensors"))
                    transformer_.plucker_proj.weight.data.copy_(_st["weight"])
                    transformer_.plucker_proj.bias.data.copy_(_st["bias"])
                    logger.info(f"Loaded plucker_proj layer from {plucker_proj_path}")
                elif os.path.exists(os.path.join(plucker_proj_path, "pytorch_model.bin")):
                    _st = torch.load(os.path.join(plucker_proj_path, "pytorch_model.bin"), map_location="cpu")
                    transformer_.plucker_proj.weight.data.copy_(_st["weight"])
                    transformer_.plucker_proj.bias.data.copy_(_st["bias"])
                    logger.info(f"Loaded plucker_proj layer from {plucker_proj_path} (bin format)")
                else:
                    logger.warning(f"plucker_proj layer not found at {plucker_proj_path}")

            # Pose heads (Options A/B).
            def _load_into_sharded(module, name):
                from safetensors.torch import load_file as _lf
                d = os.path.join(input_dir, name)
                sp = os.path.join(d, "model.safetensors")
                bp = os.path.join(d, "pytorch_model.bin")
                if os.path.exists(sp):
                    module.load_state_dict(_lf(sp), strict=False)
                    logger.info(f"Loaded {name} layer from {d}")
                elif os.path.exists(bp):
                    module.load_state_dict(torch.load(bp, map_location="cpu"), strict=False)
                    logger.info(f"Loaded {name} layer from {d} (bin format)")
                else:
                    logger.warning(f"{name} layer not found at {d}")
            if use_pose_temb:
                _load_into_sharded(transformer_.pose_temb_embed, "pose_temb_embed")
            if use_pose_film:
                _load_into_sharded(transformer_.pose_film, "pose_film")
            if use_dist_temb:
                _load_into_sharded(transformer_.dist_temb_embed, "dist_temb_embed")

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

        # Restore the peft TP-shard function we neutralized at hook entry.
        if _orig_maybe_shard is not None:
            _peft_sl._maybe_shard_state_dict_for_tp = _orig_maybe_shard

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # Only the main rank reports trainable params (avoids 32x interleaved spam
    # under multi-GPU/multi-node). The full per-layer list is gated behind
    # PRINT_TRAINABLE_LAYERS=1; by default we just print the total.
    if accelerator.is_main_process:
        n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        if os.environ.get("PRINT_TRAINABLE_LAYERS", "0").lower() in {"1", "true", "yes", "on"}:
            print("Trainable layers:")
            for name, param in transformer.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.numel()} parameters")
        print(f"Trainable params: {n_trainable / 1e6:.3f} M")
    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Pass rgb_circular_pad to data_config if set
    rgb_circular_pad = getattr(args, 'rgb_circular_pad', False)
    if rgb_circular_pad:
        args.data_config['rgb_circular_pad'] = True
        print(f"[Config] rgb_circular_pad enabled: images will be circular padded in dataset")

    # Dataset and DataLoaders creation:
    print(f"[Dataset] loader function from: {loader.__module__}")
    train_dataloader = loader(**args.data_config)

    # Online warp renderer (per-rank GPU; ZIM + GD are lazy-loaded inside,
    # or eagerly pre-warmed if prewarm_seg_models=True to avoid the
    # "first outdoor case" skew across ranks at training step 0).
    #
    # Profiling: ONLINE_WARP_PROFILE=1 enables per-step CUDA-event timings of
    # render / prompt_enc / vae_enc / sync_barrier / fwd / bwd / opt_step.
    # ONLINE_WARP_PROFILE_WARMUP (default 10): print every step for the first
    # N steps. ONLINE_WARP_PROFILE_EVERY (default 50): cadence afterwards.
    # When the flag is off (default), no events are recorded and no extra
    # syncs happen — zero overhead on the production path.
    # Always import the timer helpers so the offline (non-online_warp) branch
    # of the training loop can reference NOOP_TIMER without paying a per-step
    # ``from ... import`` cost.
    from training.data.online_warp_renderer import (
        CudaTimer,
        NOOP_TIMER,
    )

    # DEBUG_MODE=1: a self-contained single-purpose debug run. It DOES NOT
    # change any training math/behaviour -- it only (a) forces per-step
    # CUDA-event timings of the key stages on, (b) prints the shapes of the
    # key tensors every step, (c) stops after DEBUG_MAX_STEPS optimizer steps
    # (default 2), and (d) prints a final aggregated summary. Everything is
    # gated behind ``debug_mode`` so the production path is untouched.
    debug_mode = os.environ.get("DEBUG_MODE", "0").lower() in {"1", "true", "yes", "on"}
    debug_max_steps = max(1, int(os.environ.get("DEBUG_MAX_STEPS", "2")))
    _debug_records = []  # list[dict]: per-step {"timings": {...}, "shapes": {...}}
    if debug_mode and accelerator.is_main_process:
        print(
            f"\n{'='*60}\n[DEBUG_MODE] ON: per-step stage timings + key tensor shapes "
            f"will be printed; run stops after {debug_max_steps} step(s), then a "
            f"summary is printed. Training math is UNCHANGED.\n{'='*60}\n",
            flush=True,
        )

    online_renderer = None
    online_profile_enabled = False
    online_profile_warmup = 10
    online_profile_every = 50
    if online_warp:
        from training.data.online_warp_renderer import OnlineWarpRenderer
        online_profile_enabled = os.environ.get(
            "ONLINE_WARP_PROFILE", "0"
        ).lower() in {"1", "true", "yes", "on"}
        online_profile_warmup = int(os.environ.get("ONLINE_WARP_PROFILE_WARMUP", "10"))
        online_profile_every = max(1, int(os.environ.get("ONLINE_WARP_PROFILE_EVERY", "50")))

        # Debug dump (off in production). When ``ONLINE_DEBUG_DUMP_DIR`` is
        # set we let the renderer write a per-sample bundle of
        # input / mask / warp / GT to disk for the first
        # ``ONLINE_DEBUG_DUMP_STEPS`` (default 1) ``render_batch`` calls.
        # Only the main rank gets a non-None dir; non-main ranks always
        # pass ``debug_dump_dir=None`` so they never touch disk.
        debug_dump_dir = os.environ.get("ONLINE_DEBUG_DUMP_DIR", "").strip() or None
        debug_dump_first_n = int(os.environ.get("ONLINE_DEBUG_DUMP_STEPS", "1"))
        if debug_dump_dir is not None and not accelerator.is_main_process:
            debug_dump_dir = None

        online_renderer = OnlineWarpRenderer(
            device=accelerator.device,
            sky_mask_mode=args.data_config.get('sky_mask_mode', 'lazy'),
            max_pcd_points=int(args.data_config.get('max_pcd_points', 5_000_000)),
            gs_scale_factor=float(args.data_config.get('gs_scale_factor', 1.0)),
            render_h=int(args.data_config.get('img_size', 512)),
            render_w=int(args.data_config.get('img_size', 512)) * 2,
            depth_contract=float(args.data_config.get('depth_contract', 0.0)),
            prewarm_seg_models=bool(args.data_config.get('prewarm_seg_models', True)),
            profile=online_profile_enabled,
            render_engine=str(args.data_config.get('render_engine', 'gsplat')),
            warp_cache_dir=args.data_config.get('warp_cache_dir', None),
            pcd_render_radius=float(args.data_config.get('pcd_render_radius', 0.008)),
            pcd_points_per_pixel=int(args.data_config.get('pcd_points_per_pixel', 8)),
            pcd_use_fp16_cache=bool(args.data_config.get('pcd_use_fp16_cache', True)),
            debug_dump_dir=debug_dump_dir,
            debug_dump_first_n_steps=debug_dump_first_n,
            cube_layout=args.data_config.get('cube_layout', None),
            cube_face_size=int(args.data_config.get('cube_face_size', 480)),
            cube_grid_pad=int(args.data_config.get('cube_grid_pad', 32)),
            use_plucker=use_plucker,
            plucker_trans_scale=plucker_trans_scale,
            use_pose=use_pose,
            pose_scale=pose_scale,
            use_warp=use_warp,
        )
        print(f"[OnlineWarp] renderer ready on {accelerator.device}: "
              f"sky_mask_mode={online_renderer.sky_mask_mode}, "
              f"render_engine={online_renderer.render_engine}, "
              f"max_pcd_points={online_renderer.max_pcd_points}, "
              f"render={online_renderer.render_h}x{online_renderer.render_w}")
        if online_profile_enabled and accelerator.is_main_process:
            print(
                f"[OnlineWarp][Profile] enabled "
                f"(warmup={online_profile_warmup} steps every-step, "
                f"then every {online_profile_every} steps); "
                f"only main rank prints; ZERO overhead in non-profile steps."
            )
        if online_renderer.sky_mask_mode == 'lazy':
            # Synchronise here so the 8 ranks finish the pre-warm load (if
            # enabled) before any rank starts step 0 -- avoids long-tail
            # waits at the in-step accelerator.wait_for_everyone() barrier.
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print("[OnlineWarp] all ranks past renderer init barrier.")

    def compute_text_embeddings(prompt, condition_images, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                image=condition_images,
                prompt=prompt,
                max_sequence_length=args.max_sequence_length
            )
        return prompt_embeds, prompt_embeds_mask

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # NOTE on FSDP2 param storage dtype: with `mixed_precision: bf16`, accelerate
    # keeps an fp32 MASTER copy of params (standard AMP: fp32 master + bf16
    # autocast compute), so FSDP shards fp32 (~10.4 GB/rank for the 20.8B base,
    # verified via the [mem] line below: shard_factor=8x, dtype=float32). FSDP2's
    # MixedPrecisionPolicy(param_dtype=...) only sets the all-gather/compute cast,
    # NOT the sharded storage dtype, so it can't shrink this. Storing the sharded
    # base in bf16 would require disabling accelerate mixed precision (pure bf16),
    # which also drops the LoRA optimizer state to bf16 -- not worth the
    # convergence risk given the ample headroom (allocated ~29 GB / 96 GB). Left
    # as-is intentionally.

    # Prepare everything with our `accelerator`. The DataLoader is already
    # sharded by a manual DistributedSampler in loader(), so we DISCARD
    # accelerate's wrapped dataloader (the ``_``) to avoid double-sharding and
    # keep iterating the original.
    transformer, optimizer, _, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # One-time memory report so we can tell whether the base transformer is
    # actually sharded (FSDP) vs replicated (DDP), and separate "reserved"
    # (the nvidia-smi number, inflated by the caching allocator + FSDP
    # all-gather/comm buffers) from "allocated" (real live tensors).
    if torch.cuda.is_available():
        try:
            _local_bytes = 0
            _local_numel = 0
            _global_numel = 0
            _dtypes = {}
            for _p in transformer.parameters():
                _t = _p.to_local() if hasattr(_p, "to_local") else _p
                _local_bytes += _t.numel() * _t.element_size()
                _local_numel += _t.numel()
                _global_numel += _p.numel()  # DTensor.numel() = global logical
                _dtypes[str(_t.dtype)] = _dtypes.get(str(_t.dtype), 0) + 1
            _shard = _global_numel / max(_local_numel, 1)
            print(
                f"[mem][rank {accelerator.process_index}] after prepare: "
                f"LOCAL params={_local_bytes / 1e9:.2f} GB "
                f"(local={_local_numel/1e9:.2f}B / global={_global_numel/1e9:.2f}B "
                f"=> shard_factor={_shard:.1f}x) dtypes={_dtypes} | "
                f"allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB | "
                f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )
            sys.stdout.flush()
        except Exception as _e:  # diagnostics must never break training
            print(f"[mem] report skipped: {_e}")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = args.copy()
        # convert <class 'omegaconf.dictconfig.DictConfig'> to dict
        config_dict = OmegaConf.to_container(tracker_config, resolve=True)
        # tensorboard cannot handle list types for config
        if 'data_config' in config_dict:
            config_dict.pop('data_config')

        # Debug info - you can remove after confirming it works
        print("Using config_dict for TensorBoard:", type(config_dict))

        accelerator.init_trackers(args.tracker_project_name, config=config_dict)

        # Ensure tensorboard flushes after initialization
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.flush()

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Optional NaN/Inf localizer: DETECT_ANOMALY=1 makes autograd raise at the
    # exact op that first produces a non-finite value in backward (with a
    # forward-op traceback). Slow — use only for 3-step debugging.
    if os.environ.get("DETECT_ANOMALY", "0").lower() in {"1", "true", "yes", "on"}:
        torch.autograd.set_detect_anomaly(True)
        if accelerator.is_main_process:
            print("[debug] torch.autograd.set_detect_anomaly(True) enabled", flush=True)

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save.
    #
    # Three accepted forms for ``args.resume_from_checkpoint``:
    #   1. "latest"          → auto-scan ``args.output_dir`` for the highest
    #                          ``checkpoint-N`` and resume it. With the new
    #                          per-run timestamp output dir layout this
    #                          almost always yields "no ckpts found" and
    #                          falls through to a fresh start, which is
    #                          the desired safe default.
    #   2. absolute path     → load that exact directory directly. Used to
    #                          continue training from a ckpt that lives in
    #                          a DIFFERENT run dir (e.g. yesterday's run):
    #                            RESUME_FROM_CHECKPOINT=/.../runs/20260522_180000_v2depth/checkpoint-1000
    #   3. relative name     → ``checkpoint-N`` resolved against
    #                          ``args.output_dir`` (legacy semantics).
    resume_arg = args.resume_from_checkpoint
    if isinstance(resume_arg, str):
        resume_arg = resume_arg.strip() or None
    if resume_arg:
        if resume_arg == "latest":
            dirs = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_path = os.path.join(args.output_dir, dirs[-1]) if dirs else None
        elif os.path.isabs(resume_arg):
            # Caller gave a full absolute path to a checkpoint directory.
            resume_path = resume_arg
        else:
            # Caller gave just "checkpoint-N" — resolve under output_dir.
            resume_path = os.path.join(args.output_dir, os.path.basename(resume_arg))

        if resume_path is None or not os.path.isdir(resume_path):
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        elif os.path.exists(os.path.join(resume_path, ".lora_only")):
            # Checkpoint was saved with SAVE_FULL_STATE_EVERY_N gating the
            # heavy DeepSpeed shards out — there's nothing for accelerator
            # to load_state from, only inference artefacts.
            raise RuntimeError(
                f"Cannot resume from {resume_path}: this checkpoint was saved "
                "as lora-only (no DeepSpeed model/optimizer shards). It is "
                "fine for inference but not resumable.\n"
                "Pick a different checkpoint that has DeepSpeed shards (one "
                "saved while SAVE_FULL_STATE_EVERY_N > 0 was active and the "
                "current ckpt index landed on a multiple of N), or relaunch "
                "with SAVE_FULL_STATE_EVERY_N=1 going forward to make every "
                "ckpt resumable."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {resume_path}")
            accelerator.load_state(resume_path)
            ckpt_dirname = os.path.basename(resume_path.rstrip("/"))
            try:
                global_step = int(ckpt_dirname.split("-")[1])
            except (IndexError, ValueError):
                accelerator.print(
                    f"[WARN] could not parse global_step from checkpoint dir "
                    f"name {ckpt_dirname!r}; falling back to global_step=0"
                )
                global_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        accelerator.print("Starting a new training run (resume_from_checkpoint is empty).")
        initial_global_step = 0

    # ---- checkpoint-save policy --------------------------------------------
    # Each ckpt-save event always writes a *lightweight* artefact (lora +
    # mask_* + meta.json, ~MB, inference-only).  The *heavy* DeepSpeed
    # state (model shards + optimizer + scheduler + RNG, ~GB to ~tens of
    # GB) is what the user actually pays disk for, so it is gated by:
    #
    #   SAVE_FULL_STATE_EVERY_N (env, default = 1)
    #     0  → never save DeepSpeed state.  All ckpts are inference-only;
    #          you cannot resume training from any of them.
    #     1  → save DeepSpeed state at every ckpt event (default; resumable).
    #     N  → save DeepSpeed state at every N-th ckpt event.
    #
    # Examples (with checkpointing_steps=500):
    #   N=4  → full state at step 2000, 4000, 6000 ...   (~ every 2k steps)
    #   N=8  → full state at step 4000, 8000, 12000 ... (~ every 4k steps)
    #
    # ckpts that are missing DeepSpeed shards have a ".lora_only" marker
    # file at the root and will refuse RESUME_FROM_CHECKPOINT (see
    # accelerator.load_state failure mode -> AssertionError).
    try:
        save_full_state_every_n = int(os.environ.get("SAVE_FULL_STATE_EVERY_N", "1"))
    except ValueError:
        save_full_state_every_n = 0
    if save_full_state_every_n < 0:
        save_full_state_every_n = 0
    ckpt_save_idx = 0
    if accelerator.is_main_process:
        if save_full_state_every_n == 0:
            accelerator.print(
                "[ckpt policy] SAVE_FULL_STATE_EVERY_N=0 → all checkpoints are "
                "lora-only (inference-ready, NOT resumable). Set "
                "SAVE_FULL_STATE_EVERY_N=1 to save resumable DeepSpeed state "
                "every ckpt, or N>1 to save every N-th ckpt."
            )
        else:
            accelerator.print(
                f"[ckpt policy] SAVE_FULL_STATE_EVERY_N={save_full_state_every_n} → "
                f"DeepSpeed full state every {save_full_state_every_n} ckpt event(s); "
                f"other ckpts are lora-only."
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        mininterval=1.0,  # Update at most once per second
        maxinterval=10.0,  # Force update at least every 10 seconds
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Tracks the wall-clock end-of-iteration timestamp so the next iteration
    # can attribute the gap to "dataloader / barrier / etc." wait. Reset per
    # epoch boundary so we don't bill the epoch transition to dl_wait.
    _step_prev_end_t = None

    _debug_stop = False
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        # Log epoch start for multi-machine visibility
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
            print(f"{'='*60}\n")
            sys.stdout.flush()
        _step_prev_end_t = None

        for step, batch in enumerate(train_dataloader):
            _iter_start_t = time.time()
            _dl_wait_s = (
                _iter_start_t - _step_prev_end_t
                if _step_prev_end_t is not None
                else 0.0
            )

            # Decide whether to record CUDA-event timings this step. Cheap
            # ``int % int`` + ``and`` short-circuit: when profiling is off
            # ``_step_timer`` is the module-level no-op singleton and every
            # ``with _step_timer.section(...)`` is a free function call.
            _profile_this_step = online_profile_enabled and (
                step < online_profile_warmup
                or (step > 0 and step % online_profile_every == 0)
            )
            # DEBUG_MODE forces the CUDA timer on for every (short) debug step so
            # we always get stage timings regardless of the profile cadence.
            if (_profile_this_step or debug_mode) and torch.cuda.is_available() and online_warp:
                _step_timer = CudaTimer(accelerator.device)
            else:
                _step_timer = NOOP_TIMER

            models_to_accumulate = transformer
            #img, prompts, condition, condition_prompt = batch
            # Unpack batch with optional is_latent flag

            # Online warp branch: batch is a dict produced by
            # datasets_pano2pano_four_view_online._four_view_online_collate:
            #   batch["samples"]        : List[sample_dict]  (raw per-sample dicts)
            #   batch["start_rgb_train"]: (B, 3, H, W)       shared start
            #   batch["end_rgb_train"]  : (B, V, 3, H, W)    V GT end views
            #   batch["cond_img_prompt"]: (B, 3, h, w)       Qwen2.5-VL cond image
            #   batch["captions"]       : List[str]           full prompts (sys+cap)
            # render_batch_multiview builds PCD once per start and renders V
            # warps inline. Falls into the is_latent=False VAE-encode path below.
            # Plücker ray maps stay None unless use_plucker=True.
            plucker_start = None
            plucker_end = None
            pose_vec = None
            if online_warp and isinstance(batch, dict):
                _render_t0 = time.time()
                with _step_timer.section("render"):
                    # FOUR-VIEW: PCD built once per shared start, rendered to V
                    # ends. Returns (B,V,3,H,W) warp, (B,V,1,H,W) mask,
                    # (B,6,H,W) plucker_start, (B,V,6,H,W) plucker_end, (B,V,3) pose.
                    wrap_image, wrap_mask, plucker_start, plucker_end, pose_vec = (
                        online_renderer.render_batch_multiview(batch["samples"])
                    )
                _render_wallclock = time.time() - _render_t0

                img = batch["end_rgb_train"].to(accelerator.device)  # (B, V, 3, H, W)
                condition = batch["start_rgb_train"].to(accelerator.device)
                condition_prompt = batch["cond_img_prompt"].to(accelerator.device)
                # wrap_image / wrap_mask are None when use_warp=False
                # (renderer skipped the warp render).
                if wrap_image is not None:
                    wrap_image = wrap_image.to(accelerator.device)
                if wrap_mask is not None:
                    wrap_mask = wrap_mask.to(accelerator.device)
                if plucker_start is not None:
                    plucker_start = plucker_start.to(accelerator.device)
                    plucker_end = plucker_end.to(accelerator.device)
                if pose_vec is not None:
                    pose_vec = pose_vec.to(accelerator.device)
                    # POSE_DEBUG=1: print the FULL pose_vec every step (per
                    # sample) so you can eyeball the relative-translation values
                    # going into the conditioning. Otherwise just a step-0 summary.
                    _pose_dbg = os.environ.get("POSE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
                    if accelerator.is_main_process and (_pose_dbg or step == 0):
                        _pv = pose_vec.detach().float().cpu()
                        _pf = bool(torch.isfinite(_pv).all())
                        _norm = _pv.norm(dim=-1).tolist()
                        print(f"[pose] step{step} pose_vec(shape={tuple(_pv.shape)}, finite={_pf}) "
                              f"values={[[round(v, 4) for v in row] for row in _pv.tolist()]} "
                              f"L2={[round(n, 4) for n in _norm]}", flush=True)
                prompts = list(batch["captions"])
                is_latent = False

                # wrap_image / wrap_mask come from the GPU renderer at the
                # raw ERP size (img_size, img_size*2), so they need the same
                # pixel-space circular pad as the dataset's start/end/cond
                # tensors when rgb_circular_pad is on. ``start/end/cond``
                # were already padded inside OnlineWarpDataset._load_one.
                # Plücker maps live in the same raw ERP pixel space, so they
                # get the identical circular pad (their token grid must match
                # the wrap / condition streams they are added to).
                if rgb_circular_pad and wrap_image is not None:
                    _pad_w_wrap = int(wrap_image.shape[-1]) // 16
                    if _pad_w_wrap > 0:
                        wrap_image = torch.cat(
                            [wrap_image[..., -_pad_w_wrap:], wrap_image, wrap_image[..., :_pad_w_wrap]],
                            dim=-1,
                        )
                        wrap_mask = torch.cat(
                            [wrap_mask[..., -_pad_w_wrap:], wrap_mask, wrap_mask[..., :_pad_w_wrap]],
                            dim=-1,
                        )
                        if plucker_start is not None:
                            plucker_start = torch.cat(
                                [plucker_start[..., -_pad_w_wrap:], plucker_start, plucker_start[..., :_pad_w_wrap]],
                                dim=-1,
                            )
                            plucker_end = torch.cat(
                                [plucker_end[..., -_pad_w_wrap:], plucker_end, plucker_end[..., :_pad_w_wrap]],
                                dim=-1,
                            )

                if step == 0 and accelerator.is_main_process:
                    print(f"[OnlineWarp] step0 render+collect={_render_wallclock:.3f}s "
                          f"warp_image={None if wrap_image is None else tuple(wrap_image.shape)} "
                          f"wrap_mask={None if wrap_mask is None else tuple(wrap_mask.shape)} "
                          f"rgb_circular_pad={rgb_circular_pad}")
                    sys.stdout.flush()

                # NOTE: the single-view ``_dump_trainer_view`` debug dump is
                # disabled in the 4-view path (its tensor layout assumes one
                # warp/end per sample). The renderer's own per-sample dumps
                # still apply if ONLINE_DEBUG_DUMP_DIR is set.
            else:
                raise ValueError(
                    "train_four_view_online.py only supports the online_warp "
                    "dict batch path; got a non-dict batch."
                )

            # When using latents, img shape is [B, C, 1, H_latent, W_latent]
            # We need to convert to pixel dimensions
            if is_latent:
                height = img.shape[-2] * vae_scale_factor
                width = img.shape[-1] * vae_scale_factor
                vae_height = condition.shape[-2] * vae_scale_factor
                vae_width = condition.shape[-1] * vae_scale_factor
            else:
                height, width = img.shape[-2:]
                vae_height, vae_width = condition.shape[-2:]
            with accelerator.accumulate(models_to_accumulate):
                # encode batch prompts when custom prompts are provided for each image -
                with _step_timer.section("prompt_enc"):
                    with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                        prompt_embeds, prompt_embeds_mask = compute_text_embeddings(
                            prompts, condition_prompt, text_encoding_pipeline
                        )
                    # Diffusers >=0.38 QwenImageEditPlusPipeline.encode_prompt returns None mask
                    # when all tokens are valid (all-ones mask). Materialize a ones mask so
                    # downstream code (transformer forward, txt_seq_lens) keeps working.
                    if prompt_embeds_mask is None:
                        prompt_embeds_mask = torch.ones(
                            prompt_embeds.shape[:2],
                            dtype=torch.long,
                            device=prompt_embeds.device,
                        )

                # ===== FOUR-VIEW VAE encode (per-view end/warp; single start) =====
                #   img        : (B, V, 3, H, W)  4 end views (GT targets)
                #   condition  : (B, 3, H, W)     1 shared start
                #   wrap_image : (B, V, 3, H, W)  4 online warps
                #   wrap_mask  : (B, V, 1, H, W)  4 masks
                # Latent layout used throughout the 4-view path is (B, C, V, Hl, Wl)
                # (channels dim1, view dim2) to mirror the legacy four-view recipe.
                if is_latent:
                    raise ValueError("4-view online path does not support precomputed latents.")
                V = img.shape[1]
                with _step_timer.section("vae_enc"):
                    # offload=args.offload: when True, move the VAE onto the GPU
                    # for the duration of the encode and back to CPU after, so
                    # it doesn't sit resident for the whole step. The encode
                    # calls MUST run inside this context (the VAE is on CPU
                    # otherwise -> "weight type CPUBFloat16Type" error).
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        pixel_values = img.to(dtype=vae.dtype)            # (B,V,3,H,W)
                        condition_image = condition.to(dtype=vae.dtype)   # (B,3,H,W)
                        if wrap_image is not None:
                            wrap_image = wrap_image.to(dtype=vae.dtype)
                        if wrap_mask is not None:
                            wrap_mask = wrap_mask.to(dtype=vae.dtype)

                        # end -> model_input (B, 16, V, Hl, Wl)
                        _end_lat = []
                        for v in range(V):
                            lat = vae.encode(pixel_values[:, v].unsqueeze(2)).latent_dist.sample()
                            lat = (lat - latents_mean) * latents_std
                            _end_lat.append(lat.to(dtype=weight_dtype))      # (B,16,1,Hl,Wl)
                        model_input = torch.cat(_end_lat, dim=2)             # (B,16,V,Hl,Wl)

                        # start -> model_condition_input (B,16,1,Hl,Wl)
                        model_condition_input = vae.encode(condition_image.unsqueeze(2)).latent_dist.sample()
                        model_condition_input = (model_condition_input - latents_mean) * latents_std
                        model_condition_input = model_condition_input.to(dtype=weight_dtype)

                        # warp -> wrap_image (B,16,V,Hl,Wl)
                        if wrap_image is not None:
                            _warp_lat = []
                            for v in range(V):
                                lat = vae.encode(wrap_image[:, v].unsqueeze(2)).latent_dist.sample()
                                lat = (lat - latents_mean) * latents_std
                                _warp_lat.append(lat.to(dtype=weight_dtype))
                            wrap_image = torch.cat(_warp_lat, dim=2)         # (B,16,V,Hl,Wl)

                        # mask (B,V,1,H,W) -> stack along H -> (B,1,V*H,W)
                        if wrap_mask is not None:
                            _mB, _mV = wrap_mask.shape[0], wrap_mask.shape[1]
                            _mH, _mW = wrap_mask.shape[-2:]
                            wrap_mask = wrap_mask.permute(0, 2, 1, 3, 4).reshape(_mB, 1, _mV * _mH, _mW)

                # Sample noise (B,16,V,Hl,Wl)
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # ERP latent-space "+4" wrap pad (rgb_circular_pad off). Last-dim
                # (width) pad works for every (..., Hl, Wl) tensor regardless of the
                # view dim. cube_layout is disabled in the 4-view path.
                if not rgb_circular_pad:
                    model_input = torch.cat([model_input, model_input[..., :4]], dim=-1)
                    noise = torch.cat([noise, noise[..., :4]], dim=-1)
                    if wrap_image is not None:
                        wrap_image = torch.cat([wrap_image, wrap_image[..., :4]], dim=-1)
                    model_condition_input = torch.cat([model_condition_input, model_condition_input[..., :4]], dim=-1)
                    if wrap_mask is not None:
                        wrap_mask = torch.cat([wrap_mask, wrap_mask[..., :4*vae_scale_factor]], dim=-1)
                    if plucker_start is not None:
                        plucker_start = torch.cat([plucker_start, plucker_start[..., :4*vae_scale_factor]], dim=-1)
                        plucker_end = torch.cat([plucker_end, plucker_end[..., :4*vae_scale_factor]], dim=-1)

                width_padded = model_input.shape[-1] * vae_scale_factor

                # timestep / sigmas (n_dim=5 for the (B,16,V,Hl,Wl) latent)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # ---- Pack latents per view: (B,C,V,h,w) -> (B, V*seq, 64) ----
                Hl, Wl = model_input.shape[3], model_input.shape[4]
                Hl_w, Wl_w = (
                    (wrap_image.shape[3], wrap_image.shape[4])
                    if wrap_image is not None else (Hl, Wl)
                )

                def _pack_per_view(lat_BCVHW, h, w):
                    perm = lat_BCVHW.permute(0, 2, 1, 3, 4)  # (B,V,C,h,w)
                    parts = []
                    for v in range(lat_BCVHW.shape[2]):
                        parts.append(QwenImageEditPlusPipeline._pack_latents(
                            perm[:, v:v + 1],
                            batch_size=bsz,
                            num_channels_latents=lat_BCVHW.shape[1],
                            height=h, width=w,
                        ))
                    return torch.cat(parts, dim=1)

                packed_noisy_model_input = _pack_per_view(noisy_model_input, Hl, Wl)  # (B, V*L_n, 64)
                packed_model_condition_input = QwenImageEditPlusPipeline._pack_latents(
                    model_condition_input.permute(0, 2, 1, 3, 4),
                    batch_size=bsz,
                    num_channels_latents=model_condition_input.shape[1],
                    height=Hl, width=Wl,
                )                                                                     # (B, L_n, 64)
                if wrap_image is not None:
                    packed_model_wrap_image = _pack_per_view(wrap_image, Hl_w, Wl_w)  # (B, V*L_w, 64)
                else:
                    packed_model_wrap_image = None

                # ---- img_shapes: V noise + 1 start + V warp ----
                _face = (1, Hl // 2, Wl // 2)
                _warp_face = (1, Hl_w // 2, Wl_w // 2)
                img_shapes = [[*([_face] * V), _face, *([_warp_face] * V)]] * bsz

                # Debug: Print before transformer call
                if step == 0:
                    print(f"\n[DEBUG] About to call transformer forward pass...")
                    print(f"[DEBUG] Process rank: {accelerator.process_index}/{accelerator.num_processes}")
                    print(f"[DEBUG] Device: {accelerator.device}")
                    print(f"[DEBUG] packed_noisy_model_input device: {packed_noisy_model_input.device}, dtype: {packed_noisy_model_input.dtype}")
                    print(f"[DEBUG] packed_model_condition_input device: {packed_model_condition_input.device}, dtype: {packed_model_condition_input.dtype}")
                    print(f"[DEBUG] prompt_embeds device: {prompt_embeds.device}, dtype: {prompt_embeds.dtype}")
                    print(f"[DEBUG] packed_model_wrap_image: {None if packed_model_wrap_image is None else (packed_model_wrap_image.device, packed_model_wrap_image.dtype)}")
                    print(f"[DEBUG] wrap_mask: {None if wrap_mask is None else (wrap_mask.device, wrap_mask.dtype, tuple(wrap_mask.shape))}")
                    print(f"[DEBUG] timesteps: {timesteps}")
                    sys.stdout.flush()

                # Force synchronization before transformer call to prevent deadlock.
                # The barrier time below is dominated by per-rank straggler delay
                # (slowest rank's render + vae_enc finishing); profile it so we
                # can tell "online warp work" apart from "wait_for_everyone wait".
                with _step_timer.section("sync_barrier"):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if accelerator.num_processes > 1:
                        accelerator.wait_for_everyone()

                # Print all shapes after processing
                if step == 0:
                    print("\n=== Shape Information ===")
                    print('is letent:',is_latent)
                    print(f'vae_scale_factor: {vae_scale_factor}') #8
                    print(f"img (original): {img.shape}") #([1, 3, 736, 1440])
                    print(f"condition (original): {condition.shape}") #([1, 16, 1, 92, 184])
                    print(f"wrap_image (original): {None if wrap_image is None else wrap_image.shape}") #([1, 16, 1, 64, 132])
                    print(f"model_input (after VAE + padding): {model_input.shape}") #([1, 16, 1, 92, 180])
                    print(f"model_condition_input (after VAE): {model_condition_input.shape}")
                    print(f"noise: {noise.shape}")
                    print(f"noisy_model_input: {noisy_model_input.shape}")
                    print(f"packed_noisy_model_input: {packed_noisy_model_input.shape}")
                    print(f"packed_model_condition_input: {packed_model_condition_input.shape}")
                    print(f"wrap_mask (pixel space): {None if wrap_mask is None else wrap_mask.shape}")
                    print(f"height: {height}, width_padded: {width_padded}")
                    print(f"vae_height: {vae_height}, vae_width: {vae_width}")
                    print("packed_model_wrap_image", None if packed_model_wrap_image is None else packed_model_wrap_image.shape)
                    print("=========================\n")
                    sys.stdout.flush()
                    """
                    Example shapes for the +4 latent-pad path (rgb_circular_pad=False,
                    non-cube). NOTE: model_condition_input IS padded the same +4 as
                    model_input / wrap_image (see line ~1296), so its packed token count
                    matches the other two streams (2112, not 2048). wrap_mask stays in
                    pixel space (B, 1, H, W) and is patchified by the transformer's
                    mask_encoder, so it is NOT packed to (2112, 256) here.

                    vae_scale_factor: 8
                    img (original): torch.Size([1, 3, 512, 1024])
                    condition (original): torch.Size([1, 3, 512, 1024])
                    wrap_image (original): torch.Size([1, 16, 1, 64, 132])
                    model_input (after VAE + padding): torch.Size([1, 16, 1, 64, 132])
                    model_condition_input (after VAE + padding): torch.Size([1, 16, 1, 64, 132])
                    noise: torch.Size([1, 16, 1, 64, 132])
                    noisy_model_input: torch.Size([1, 1, 16, 64, 132])
                    packed_noisy_model_input: torch.Size([1, 2112, 64])
                    packed_model_condition_input: torch.Size([1, 2112, 64])
                    wrap_mask (pixel space, after padding): torch.Size([1, 1, 512, 1056])
                    height: 512, width_padded: 1056
                    vae_height: 512, vae_width: 1024
                    packed_model_wrap_image torch.Size([1, 2112, 64])
                    """

                with _step_timer.section("fwd"):
                    model_pred = transformer(
                        packed_noisy_model_input=packed_noisy_model_input,
                        packed_model_condition_input=packed_model_condition_input,
                        packed_model_wrap_image=packed_model_wrap_image,
                        wrap_mask=wrap_mask,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        timestep=timesteps / 1000,
                        img_shapes=img_shapes,
                        txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                        plucker_start=plucker_start,
                        plucker_end=plucker_end,
                        pose_vec=pose_vec,
                        return_dict=False,
                    )[0]

                # Debug: Print after transformer call
                if step == 0:
                    print(f"[DEBUG] Transformer forward pass completed!")
                    print(f"[DEBUG] model_pred shape: {model_pred.shape}")
                    sys.stdout.flush()

                # Keep only the noise slice (first V*L_n tokens) and unpack per
                # view back to latent maps, then cat along the view dim so the
                # target ``noise - model_input`` (B,16,V,Hl,Wl) aligns.
                s_noise = packed_noisy_model_input.size(1)
                model_pred = model_pred[:, :s_noise]
                seq_per_view = s_noise // V
                _preds = []
                for v in range(V):
                    up = QwenImageEditPlusPipeline._unpack_latents(
                        model_pred[:, seq_per_view * v: seq_per_view * (v + 1)],
                        height, width_padded, vae_scale_factor,
                    )                                          # (B,16,1,Hl,Wl)
                    _preds.append(up)
                model_pred = torch.cat(_preds, dim=2)          # (B,16,V,Hl,Wl)

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                target = noise - model_input                   # (B,16,V,Hl,Wl)

                # Compute regular loss (joint over the 4 views).
                loss_v = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss_v = loss_v.mean()

                # NOTE: the previous "cube_loss" auxiliary that lived here
                # operated on training/models/cube_map.cube_map_from_equirectangular
                # outputs, which use a non-standard ERP convention (v=0 = south
                # pole, plus a horizontal-mirror rot90 on each face). It was
                # disabled even before the cube_layout work and is now
                # removed to avoid suggesting that helper as a building
                # block. If you need a future cube-space loss, sample
                # latents into the 2x4 padding atlas directly via
                # training/data/cube_layout.erp_to_2x4_padding_grid (the
                # convention is documented in
                # CubeMap_Related_Work/IMPLEMENTATION.md).
                loss = loss_v

                with _step_timer.section("bwd"):
                    accelerator.backward(loss)
                with _step_timer.section("opt"):
                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log immediately after gradient sync for multi-machine visibility
                # if accelerator.is_main_process and global_step % 5 == 0:
                #     print(f"[Main Process] Completed step {global_step}/{args.max_train_steps}")
                #     import sys
                #     sys.stdout.flush()

                # Save every 100 steps when global_step < 1000, otherwise use checkpointing_steps
                should_save_ckpt = (global_step < 1000 and global_step % 200 == 0) or (global_step % args.checkpointing_steps == 0)
                if should_save_ckpt:
                    # Bookkeeping for SAVE_FULL_STATE_EVERY_N — count this as
                    # the (ckpt_save_idx)-th ckpt event (1-indexed).
                    ckpt_save_idx += 1
                    save_full_state = (
                        save_full_state_every_n > 0
                        and (ckpt_save_idx % save_full_state_every_n == 0)
                    )

                    # Synchronize all processes before checkpoint
                    if accelerator.is_main_process:
                        kind = "full state (lora + DeepSpeed shards)" if save_full_state else "lora-only (lightweight)"
                        print(f'[Main Process] Starting checkpoint save at step {global_step} → {kind}...')
                        sys.stdout.flush()

                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    save_start_time = time.time()
                    if save_full_state:
                        # Heavy path: full DeepSpeed save_state — model shards
                        # + optimizer state + scheduler + RNG. ~GB to ~tens of
                        # GB per ckpt, but resumable.
                        print(f'[Rank {accelerator.process_index}] Starting accelerator.save_state...')
                        sys.stdout.flush()
                        accelerator.save_state(save_path)
                    else:
                        # Lightweight path: only the inference artefacts (lora
                        # + mask_* + meta.json). ~MB. NOT resumable.
                        # NOTE: _save_lora_aux_only must be called on ALL ranks
                        # because the FSDP2 LoRA gather inside is a collective
                        # op (no-op for DDP / DeepSpeed). Only rank 0 actually
                        # writes files (gated inside the function).
                        if accelerator.is_main_process:
                            os.makedirs(save_path, exist_ok=True)
                        accelerator.wait_for_everyone()
                        _save_lora_aux_only(save_path, unwrap_model(transformer))
                        if accelerator.is_main_process:
                            # Marker so future audits / pruning can tell which
                            # ckpts have full DeepSpeed/FSDP2 shards.
                            with open(os.path.join(save_path, ".lora_only"), "w") as _f:
                                _f.write(
                                    "This checkpoint contains only LoRA + mask_* + meta.json. "
                                    "Full DeepSpeed/FSDP2 model+optimizer shards were skipped "
                                    f"because SAVE_FULL_STATE_EVERY_N={save_full_state_every_n} "
                                    f"did not match this ckpt index ({ckpt_save_idx}).\n"
                                )
                    save_duration = time.time() - save_start_time

                    print(f'[Rank {accelerator.process_index}] checkpoint save completed in {save_duration:.2f}s ({"full" if save_full_state else "lora-only"})')
                    sys.stdout.flush()

                    # Wait for all processes to complete saving
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")
                        kind = "full" if save_full_state else "lora-only"
                        print(f'[Main Process] ✓ Checkpoint saved at step {global_step} (kind={kind}, took {save_duration:.2f}s)')
                        sys.stdout.flush()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # ----- MEM_PROBE: per-step peak VRAM (benchmark instrumentation) -----
            # Gated by env MEM_PROBE=1. Prints this rank's peak allocated/reserved
            # VRAM for the step that just finished, then resets the peak counter so
            # each step's number is a clean single-step fwd+bwd+opt peak. Additive;
            # touches no training tensor. Every rank prints so we can see stragglers.
            if os.environ.get("MEM_PROBE", "0") == "1" and torch.cuda.is_available():
                _mp_peak = torch.cuda.max_memory_allocated() / 1e9
                _mp_res = torch.cuda.max_memory_reserved() / 1e9
                _mp_alloc = torch.cuda.memory_allocated() / 1e9
                print(f"[MEM_PROBE][rank {accelerator.process_index}] step {global_step}: "
                      f"peak_alloc={_mp_peak:.2f}GB peak_reserved={_mp_res:.2f}GB "
                      f"end_alloc={_mp_alloc:.2f}GB", flush=True)
                torch.cuda.reset_peak_memory_stats()

            # Explicit logging for multi-machine training visibility
            if accelerator.is_main_process:
                if global_step % 10 == 0 or global_step == 1:  # Log every 10 steps
                    print(f"[Rank {accelerator.process_index}] Step {global_step}: Loss={logs['loss']:.4f}, LR={logs['lr']:.2e}")
                    sys.stdout.flush()

            # ----- Per-step profile dump -----
            # ``_step_timer.finalize()`` is a single cuda.synchronize on the
            # rank's device when the timer is a CudaTimer; a no-op otherwise.
            # We only print on the main process to keep stdout clean -- if
            # you need to spot stragglers, look at the variance of ``render``
            # across ranks in NCCL_DEBUG logs.
            if _profile_this_step and online_warp and accelerator.is_main_process:
                _step_timings = _step_timer.finalize()
                _iter_wall = time.time() - _iter_start_t
                _render_subs = (
                    online_renderer.last_timings if online_renderer is not None else {}
                )
                # Pretty short keys for the sub-section dump.
                _short = {
                    "io_upload": "io",
                    "sky_mask": "sky",
                    "depth_edge": "dedge",
                    "pcd_filter": "pcd_f",
                    "pcd_downsample": "pcd_d",
                    "gs_build": "gs_b",
                    "gs_render": "gs_r",
                    # pytorch3d engine emits ``pcd_render`` instead of
                    # ``gs_build`` + ``gs_render``; the absence of gs_b in a
                    # profile line is a quick "we are on the pytorch3d path"
                    # tell.
                    "pcd_render": "pcd_r",
                    "out_h2d": "out",
                }
                _render_total = _step_timings.pop("render", _render_wallclock)
                parts = [
                    f"iter={_iter_wall:.2f}s",
                    f"dl_wait={_dl_wait_s:.2f}",
                    f"| render={_render_total:.2f}",
                ]
                if _render_subs:
                    sub_str = " ".join(
                        f"{_short.get(k, k)}={v:.2f}"
                        for k, v in _render_subs.items()
                    )
                    parts.append(f"[{sub_str}]")
                parts.append("|")
                for k in ("prompt_enc", "vae_enc", "sync_barrier", "fwd", "bwd", "opt"):
                    if k in _step_timings:
                        parts.append(f"{k}={_step_timings[k]:.2f}")
                print(f"[OnlineWarp][Profile][step {step}] " + " ".join(parts))
                sys.stdout.flush()

            # ----- DEBUG_MODE per-step dump + early stop -----
            # Collects stage timings (CUDA-event based) and the shapes of the
            # key tensors used this step, prints them, and stops the run after
            # ``debug_max_steps`` steps with an aggregated summary. Purely
            # additive: no training tensor/op is touched here.
            if debug_mode and online_warp:
                _dbg_timings = _step_timer.finalize()
                _dbg_iter_wall = time.time() - _iter_start_t
                _dbg_render_subs = (
                    dict(online_renderer.last_timings)
                    if online_renderer is not None and online_renderer.last_timings
                    else {}
                )

                def _shp(x):
                    return None if x is None else tuple(x.shape)

                _dbg_shapes = {
                    "img (end GT, B,V,3,H,W)": _shp(img),
                    "condition (start, B,3,H,W)": _shp(condition),
                    "cond_img_prompt": _shp(condition_prompt),
                    "wrap_image (latent, B,16,V,Hl,Wl)": _shp(wrap_image),
                    "wrap_mask (pixel, B,1,V*H,W)": _shp(wrap_mask),
                    "model_input (B,16,V,Hl,Wl)": _shp(model_input),
                    "noisy_model_input": _shp(noisy_model_input),
                    "packed_noisy_model_input (B,V*L,64)": _shp(packed_noisy_model_input),
                    "packed_model_condition_input": _shp(packed_model_condition_input),
                    "packed_model_wrap_image": _shp(packed_model_wrap_image),
                    "prompt_embeds": _shp(prompt_embeds),
                    "prompt_embeds_mask": _shp(prompt_embeds_mask),
                    "plucker_start": _shp(plucker_start),
                    "plucker_end": _shp(plucker_end),
                    "pose_vec": _shp(pose_vec),
                    "model_pred (B,16,V,Hl,Wl)": _shp(model_pred),
                    "target (noise-model_input)": _shp(target),
                }
                _debug_records.append({
                    "step": int(step),
                    "iter_wall": float(_dbg_iter_wall),
                    "dl_wait": float(_dl_wait_s),
                    "render_wallclock": float(_render_wallclock),
                    "timings": dict(_dbg_timings),
                    "render_subs": _dbg_render_subs,
                    "shapes": _dbg_shapes,
                    "loss": float(loss.detach().item()),
                })

                if accelerator.is_main_process:
                    _render_total = _dbg_timings.get("render", _render_wallclock)
                    print(f"\n[DEBUG_MODE][step {step}] ---- stage timings (s) ----")
                    print(f"    dl_wait      = {_dl_wait_s:.3f}")
                    print(f"    render       = {_render_total:.3f}")
                    if _dbg_render_subs:
                        _sub = "  ".join(f"{k}={v:.3f}" for k, v in _dbg_render_subs.items())
                        print(f"      render_subs: {_sub}")
                    for k in ("prompt_enc", "vae_enc", "sync_barrier", "fwd", "bwd", "opt"):
                        if k in _dbg_timings:
                            print(f"    {k:<12} = {_dbg_timings[k]:.3f}")
                    print(f"    iter_wall    = {_dbg_iter_wall:.3f}")
                    print(f"    loss         = {_debug_records[-1]['loss']:.5f}")
                    print(f"[DEBUG_MODE][step {step}] ---- key tensor shapes ----")
                    for _k, _v in _dbg_shapes.items():
                        print(f"    {_k:<40} {_v}")
                    sys.stdout.flush()

            _step_prev_end_t = time.time()

            if debug_mode and len(_debug_records) >= debug_max_steps:
                if accelerator.is_main_process:
                    _print_debug_summary(_debug_records)
                _debug_stop = True
                break

            if global_step >= args.max_train_steps:
                break

        if _debug_stop:
            break

    # Save the lora layers
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    config_path = parse_args()
    args = OmegaConf.load(config_path)
    main(args, config_path=config_path)
