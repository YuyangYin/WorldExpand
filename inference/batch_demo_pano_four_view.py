"""Multi-GPU batch inference for the FOUR-VIEW ONLINE-warp panorama model.

Unified with training (``training/train_four_view_online.py``):

* Input JSON  : **four-view-v1** (``{"cases": [{start, views:[{role, end, ...}]}]}``
  produced by ``four_view_dataset/step4_build_four_view_json.py`` and its
  train/test split — e.g. ``data_0519/split/test_subset.json``). Legacy
  **TUPLE-v1** (``center`` + ``front1/back1/front2/back2``) is also accepted;
  ``FourViewOnlineWarpDataset`` auto-detects which and regroups into ONE 4-view
  case per group (1 shared start + 4 ends). Incomplete groups are dropped.
* Warps       : generated **online** on the GPU via
  :class:`OnlineWarpRenderer` (PCD built ONCE per shared start, rendered to the
  4 end cameras) — NOT read from disk ``warp_png``/``mask_png``.
* Dataloader  : :class:`FourViewOnlineWarpDataset` (same preprocessing /
  recenter / ``end_to_start`` yaw as training).
* Pipeline    : :class:`HunyuanPanoPipelineFourViewMaskEncode` (4-view joint
  denoise + mask / Plücker / pose conditioning).

To guarantee train/infer parity, the dataset + renderer are configured from the
training run's YAML (``--config``, e.g. the runtime ``I2P_four_view_runtime.yaml``
written by ``run_train_four_view_online_h20.sh``); only ``train_files`` is
replaced by ``--json_path`` and ``drop_text_prob`` is forced to 0.

Outputs (per case) under ``<output>/<case_id>/``:
    <direction>/output.png    per-view generated panorama (native res)
    <direction>/concat.jpg    START | WARP | MASK  /  OUTPUT | GT  preview
    four_view_grid.jpg        1 header row (START) + 4 rows (WARP|GT|PRED)

Resumable: a case whose all-4 ``output.png`` already exist is skipped.
Multi-node: ``--num_nodes`` / ``--node_rank`` round-robin shard the case list.
"""

# Resolve resource defaults from the external release configuration.
import sys as _we_sys
from pathlib import Path as _WePath
_we_root = next((_p for _p in _WePath(__file__).resolve().parents
                 if (_p / "worldexpand" / "paths.py").is_file()), None)
if _we_root is not None and str(_we_root) not in _we_sys.path:
    _we_sys.path.insert(0, str(_we_root))
from worldexpand.paths import resolve_resource as _release_path


import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler

from inference.pipelines_mask_encode_four_view import (
    HunyuanPanoPipelineFourViewMaskEncode,
)
from training.models.custom_qwen_transformer_four_view_mask_concat import (
    CustomQwenImageTransformer2DModelFourViewMaskConcat,
)
from training.data.datasets_pano2pano_four_view_online import (
    FourViewOnlineWarpDataset,
)
from training.data.datasets_pano2pano_online import _circular_pad_width_torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) float in [-1, 1] -> PIL RGB."""
    arr = ((t.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


def _depad_width_pil(im: Image.Image) -> Image.Image:
    """Strip the ERP circular pad (W//16 per side) off a PIL panorama.

    Mirrors the pipeline's de-pad of the prediction exactly: the padded width
    is W*18/16, so ``crop_w = padded_w // 18`` removes the per-side pad and
    restores the original unpadded width (e.g. 1152 -> 1024).
    """
    w, h = im.size
    crop_w = w // 18
    if crop_w <= 0:
        return im
    return im.crop((crop_w, 0, w - crop_w, h))


def _load_state_any(d: str):
    """Load a {model.safetensors|pytorch_model.bin} state dict from dir ``d``."""
    from safetensors.torch import load_file
    sp = os.path.join(d, "model.safetensors")
    bp = os.path.join(d, "pytorch_model.bin")
    if os.path.exists(sp):
        return load_file(sp)
    if os.path.exists(bp):
        return torch.load(bp, map_location="cpu")
    return None


# ---------------------------------------------------------------------------
# Attention schedule (train/infer parity).
# ---------------------------------------------------------------------------
def _parse_local_layer_indices(local_layer_indices):
    """Normalize a comma-string / list of block indices, or None if unset."""
    _lli = local_layer_indices
    if _lli is None or str(_lli).strip() in ("", "none", "None"):
        return None
    if isinstance(_lli, str):
        return [int(x) for x in _lli.replace(" ", "").split(",") if x != ""]
    return [int(x) for x in _lli]


def _configure_attention(transformer, num_views, attention_policy,
                         local_attn_variant, local_layer_indices):
    """Configure the transformer's attention schedule, mirroring training
    (``training/train_four_view_online.py``). When ``local_layer_indices`` is
    set it builds an explicit per-layer G/L schedule that OVERRIDES
    ``attention_policy`` (so a ``*_v1local30`` ckpt is evaluated with the same
    schedule it was trained with). We set the schedule BEFORE the variant so the
    model logs ``local_attn_variant -> vN``, and skip ``set_attention_policy`` in
    that case to avoid the misleading ``global [G=60, L=0]`` line."""
    transformer.num_views = num_views
    lli = _parse_local_layer_indices(local_layer_indices)
    if lli is not None:
        n = len(transformer.transformer_blocks)
        sched = ["G"] * n
        for i in lli:
            if 0 <= i < n:
                sched[i] = "L"
        transformer.set_policy_schedule(sched)
        transformer.set_local_attn_variant(local_attn_variant)
        print(f"[FourView] attention: {sched.count('L')}/{n} layers LOCAL-{local_attn_variant}, "
              f"rest GLOBAL (local_layers={sorted(lli)}); attention_policy={attention_policy} ignored",
              flush=True)
    else:
        transformer.set_attention_policy(attention_policy)
        transformer.set_local_attn_variant(local_attn_variant)
        print(f"[FourView] attention: policy={attention_policy} variant={local_attn_variant}",
              flush=True)


# ---------------------------------------------------------------------------
# Model wrapper: transformer + heads + renderer, all on one GPU.
# ---------------------------------------------------------------------------
class QwenFourViewMaskEncode:
    # Fallback base-model path (Qwen-Image-Edit-2509). Override via the
    # ``--model_path`` CLI flag or the ``MODEL_PATH`` env var.
    MODEL_PATH = _release_path("@repo/weights/Qwen-Image-Edit-2509")

    def __init__(
        self,
        ckpt_path: str,
        gpu_id: int,
        data_config: Dict[str, Any],
        num_views: int = 4,
        attention_policy: str = "global",
        local_attn_variant: str = "v1",
        local_layer_indices: Optional[Any] = None,
        flow_shift: float = 1.0,
        height: int = 512,
        width: int = 1024,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        skip_lora: bool = False,
        model_path: Optional[str] = None,
        use_mask_encoder_override: str = "auto",
        build_renderer: bool = True,
    ):
        self.gpu_id = gpu_id
        self.ckpt_path = ckpt_path
        if model_path:
            self.MODEL_PATH = model_path
        self.num_views = int(num_views)
        self.height = int(height)
        self.width = int(width)
        self.num_inference_steps = int(num_inference_steps)
        self.true_cfg_scale = float(true_cfg_scale)
        self.guidance_scale = float(guidance_scale)
        self.flow_shift = float(flow_shift)
        # The 4-view online path trains with latent-space "+4" pad (NOT pixel
        # circular pad); mirror that at eval so the token grid matches.
        self.rgb_circular_pad = bool(data_config.get("rgb_circular_pad", False))

        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        self.device = device

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.MODEL_PATH, subfolder="scheduler", shift=self.flow_shift
        )

        transformer_dir = os.path.join(ckpt_path, "transformer")
        lora_dir = os.path.join(ckpt_path, "lora")

        if os.path.exists(transformer_dir):
            print(f"GPU {gpu_id}: loading full transformer from {transformer_dir}", flush=True)
            transformer = CustomQwenImageTransformer2DModelFourViewMaskConcat.from_pretrained(
                transformer_dir, torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True, low_cpu_mem_usage=False,
            )
            _configure_attention(transformer, self.num_views, attention_policy,
                                 local_attn_variant, local_layer_indices)
            self.pipe = HunyuanPanoPipelineFourViewMaskEncode.from_pretrained(
                self.MODEL_PATH, transformer=transformer, scheduler=scheduler,
                local_files_only=True, torch_dtype=torch.bfloat16,
            ).to(device)
        elif os.path.exists(lora_dir):
            print(f"GPU {gpu_id}: loading base transformer + LoRA from {lora_dir}", flush=True)
            transformer = CustomQwenImageTransformer2DModelFourViewMaskConcat.from_pretrained(
                self.MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True, low_cpu_mem_usage=False,
            )
            _configure_attention(transformer, self.num_views, attention_policy,
                                 local_attn_variant, local_layer_indices)
            self.pipe = HunyuanPanoPipelineFourViewMaskEncode.from_pretrained(
                self.MODEL_PATH, transformer=transformer, scheduler=scheduler,
                local_files_only=True, torch_dtype=torch.bfloat16,
            ).to(device)
            if not skip_lora:
                self.pipe.load_lora_weights(
                    lora_dir, weight_name="pytorch_lora_weights.safetensors",
                    torch_dtype=torch.bfloat16,
                )
                print(f"GPU {gpu_id}: loaded LoRA", flush=True)
            self._load_aux_heads()
        else:
            raise FileNotFoundError(
                f"No valid checkpoint at {ckpt_path} (expected transformer/ or lora/)"
            )

        # ---- meta + conditioning toggles -------------------------------
        self._meta: Dict[str, Any] = {}
        _mp = os.path.join(ckpt_path, "meta.json")
        if os.path.exists(_mp):
            try:
                with open(_mp) as f:
                    self._meta = json.load(f)
            except Exception:
                self._meta = {}
        self.pose_scale = float(self._meta.get("pose_scale", 1.0))
        self.plucker_trans_scale = float(self._meta.get("plucker_trans_scale", 1.0))
        # use_warp: warp-pathway master switch (legacy key: use_warp_mask).
        self.use_warp_mask = bool(
            self._meta.get("use_warp", self._meta.get("use_warp_mask", True))
        )
        # Mask-encoder ablation: honor the training-time toggle so a model
        # trained with use_mask_encoder=False doesn't inject mask tokens.
        # meta.json is the default source; a non-"auto" CLI override wins so the
        # user can force a womask run (or A/B a withmask ckpt without injection)
        # even if the checkpoint's meta is missing/stale.
        self.use_mask_encoder = bool(self._meta.get("use_mask_encoder", True))
        _ovr = str(use_mask_encoder_override).lower()
        if _ovr != "auto":
            forced = _ovr in ("true", "1", "yes")
            if forced != self.use_mask_encoder:
                print(
                    f"GPU {gpu_id}: OVERRIDE use_mask_encoder "
                    f"{self.use_mask_encoder} -> {forced} "
                    f"(meta said {self._meta.get('use_mask_encoder', 'absent')})",
                    flush=True,
                )
            self.use_mask_encoder = forced

        tr = self.pipe.transformer
        tr.use_mask_encoder = self.use_mask_encoder
        use_plucker = bool(getattr(tr, "inject_plucker", False))
        use_pose = bool(
            getattr(tr, "inject_pose_temb", False)
            or getattr(tr, "inject_pose_film", False)
            or getattr(tr, "inject_dist_temb", False)
        )
        print(
            f"GPU {gpu_id}: conditioning -> use_warp={self.use_warp_mask}, "
            f"use_mask_encoder={self.use_mask_encoder}, "
            f"use_plucker={use_plucker}, use_pose={use_pose}, "
            f"pose_scale={self.pose_scale}, plucker_trans_scale={self.plucker_trans_scale}",
            flush=True,
        )

        # ---- online renderer (parity with training) --------------------
        # The "wild" disk-fed path supplies warps/masks straight from PNGs, so
        # it skips the renderer entirely (build_renderer=False) — this also
        # avoids the gsplat CUDA-extension dependency for that path.
        if build_renderer:
            from training.data.online_warp_renderer import OnlineWarpRenderer

            self.renderer = OnlineWarpRenderer(
                device=device,
                sky_mask_mode=str(data_config.get("sky_mask_mode", "skip")),
                max_pcd_points=int(data_config.get("max_pcd_points", 5_000_000)),
                gs_scale_factor=float(data_config.get("gs_scale_factor", 1.0)),
                render_h=int(data_config.get("img_size", 512)),
                render_w=int(data_config.get("img_size", 512)) * 2,
                depth_contract=float(data_config.get("depth_contract", 0.0)),
                prewarm_seg_models=bool(data_config.get("prewarm_seg_models", True)),
                render_engine=str(data_config.get("render_engine", "gsplat")),
                pcd_render_radius=float(data_config.get("pcd_render_radius", 0.008)),
                pcd_points_per_pixel=int(data_config.get("pcd_points_per_pixel", 8)),
                pcd_use_fp16_cache=bool(data_config.get("pcd_use_fp16_cache", True)),
                use_plucker=use_plucker,
                plucker_trans_scale=self.plucker_trans_scale,
                use_pose=use_pose,
                pose_scale=self.pose_scale,
                use_warp=self.use_warp_mask,
            )
            print(
                f"GPU {gpu_id}: renderer ready (engine={self.renderer.render_engine}, "
                f"render={self.renderer.render_h}x{self.renderer.render_w})",
                flush=True,
            )
        else:
            self.renderer = None
            print(f"GPU {gpu_id}: wild mode -> online renderer skipped", flush=True)

    # ------------------------------------------------------------------
    def _load_aux_heads(self):
        """Load mask / plucker / pose heads and flip the matching inject flags.
        Mirrors inference/batch_demo_pano_mask_encode.py's loader."""
        tr = self.pipe.transformer
        gpu_id = self.gpu_id

        enc = _load_state_any(os.path.join(self.ckpt_path, "mask_encoder"))
        if enc is not None:
            tr.mask_encoder.weight.data.copy_(enc["weight"].to(torch.bfloat16))
            print(f"GPU {gpu_id}: loaded mask_encoder", flush=True)
        proj = _load_state_any(os.path.join(self.ckpt_path, "mask_proj"))
        if proj is not None:
            tr.mask_proj.weight.data.copy_(proj["weight"].to(torch.bfloat16))
            tr.mask_proj.bias.data.copy_(proj["bias"].to(torch.bfloat16))
            print(f"GPU {gpu_id}: loaded mask_proj", flush=True)

        # Plücker heads
        pe = _load_state_any(os.path.join(self.ckpt_path, "plucker_encoder"))
        pp = _load_state_any(os.path.join(self.ckpt_path, "plucker_proj"))
        if pe is not None and pp is not None:
            tr.plucker_encoder.weight.data.copy_(pe["weight"].to(torch.bfloat16))
            tr.plucker_proj.weight.data.copy_(pp["weight"].to(torch.bfloat16))
            tr.plucker_proj.bias.data.copy_(pp["bias"].to(torch.bfloat16))
            tr.inject_plucker = True
            print(f"GPU {gpu_id}: loaded plucker heads + inject_plucker=True", flush=True)
        else:
            tr.inject_plucker = False

        # Pose heads (Options A/B/C)
        def _load_module(module, name) -> bool:
            st = _load_state_any(os.path.join(self.ckpt_path, name))
            if st is None:
                return False
            module.load_state_dict({k: v.to(torch.bfloat16) for k, v in st.items()}, strict=False)
            return True

        tr.inject_pose_temb = _load_module(tr.pose_temb_embed, "pose_temb_embed")
        if tr.inject_pose_temb:
            print(f"GPU {gpu_id}: loaded pose_temb_embed (Option A)", flush=True)
        tr.inject_pose_film = _load_module(tr.pose_film, "pose_film")
        if tr.inject_pose_film:
            tr.pose_film_streams = self._meta.get("pose_film_streams", "end_only") if getattr(self, "_meta", None) else "end_only"
            print(f"GPU {gpu_id}: loaded pose_film (Option B)", flush=True)
        tr.inject_dist_temb = _load_module(tr.dist_temb_embed, "dist_temb_embed")
        if tr.inject_dist_temb:
            print(f"GPU {gpu_id}: loaded dist_temb_embed (Option C)", flush=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(self, sample: Dict[str, Any], seed: int = 42):
        """Render the 4 warps online + jointly denoise -> 4 panoramas.

        Returns (outputs[V] PIL, start PIL, warp[V] PIL, mask[V] PIL, gt[V] PIL).
        """
        # ---- online warp render (PCD built once, rendered to V ends) ----
        warp_rgb, warp_mask, plucker_start, plucker_end, pose_vec = (
            self.renderer.render_batch_multiview([sample])
        )

        wrap_images: Optional[List[torch.Tensor]] = None
        wrap_masks: Optional[List[torch.Tensor]] = None
        warp_pils: List[Optional[Image.Image]] = [None] * self.num_views
        mask_pils: List[Optional[Image.Image]] = [None] * self.num_views
        if warp_rgb is not None:
            wrap_images = []
            wrap_masks = []
            for v in range(self.num_views):
                wv = warp_rgb[0, v].to(self.device)        # (3,H,W) [-1,1]
                mv = warp_mask[0, v].to(self.device)       # (1,H,W) {0,1}
                wrap_images.append(wv.unsqueeze(0))
                wrap_masks.append(mv.unsqueeze(0))
                warp_pils[v] = _tensor_to_pil(wv)
                mask_pils[v] = Image.fromarray(
                    (mv[0].detach().float().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                ).convert("RGB")

        plucker_start_t = plucker_start.to(self.device) if plucker_start is not None else None
        plucker_end_t = plucker_end.to(self.device) if plucker_end is not None else None
        pose_vec_t = pose_vec.to(self.device) if pose_vec is not None else None

        # ---- start image (already recenter/aligned by the dataset) ------
        start_pil = _tensor_to_pil(sample["start_rgb_train"])

        # ---- GT (end views, yaw-aligned == model output frame) ----------
        # end_rgb_train carries the ERP circular pad when rgb_circular_pad is
        # on; strip it so the saved gt.png matches the de-padded prediction
        # resolution exactly (no resize needed at eval time).
        gt_pils: List[Image.Image] = [
            _tensor_to_pil(sample["end_rgb_train"][v]) for v in range(self.num_views)
        ]
        if self.rgb_circular_pad:
            gt_pils = [_depad_width_pil(g) for g in gt_pils]

        prompt = sample["caption"]   # already SYSTEM_PROMPT + caption

        out = self.pipe(
            image=start_pil,
            prompt=prompt,
            negative_prompt="",
            generator=torch.manual_seed(seed),
            true_cfg_scale=self.true_cfg_scale,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=1,
            height=self.height, width=self.width,
            wrap_images=wrap_images, wrap_masks=wrap_masks,
            plucker_start=plucker_start_t,
            plucker_end=plucker_end_t,
            pose_vec=pose_vec_t,
            rgb_circular_pad=self.rgb_circular_pad,
            num_views=self.num_views,
        ).images

        outputs = [(v[0] if isinstance(v, list) else v) for v in out]
        return outputs, start_pil, warp_pils, mask_pils, gt_pils

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_wild(self, start_pil: Image.Image,
                 wrap_images: List[torch.Tensor],
                 wrap_masks: List[torch.Tensor],
                 prompt: str, seed: int = 42) -> List[Image.Image]:
        """Wild path: warps/masks come from disk (no online render).

        ``wrap_images`` / ``wrap_masks`` are CPU tensors shaped exactly like the
        online path feeds the pipeline ((1,3,H,W) in [-1,1] and (1,1,H,W) in
        {0,1}); plücker/pose are not used. Returns the V output panoramas.
        """
        wi = [w.to(self.device) for w in wrap_images]
        wm = [m.to(self.device) for m in wrap_masks]
        out = self.pipe(
            image=start_pil,
            prompt=prompt,
            negative_prompt="",
            generator=torch.manual_seed(seed),
            true_cfg_scale=self.true_cfg_scale,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=1,
            height=self.height, width=self.width,
            wrap_images=wi, wrap_masks=wm,
            plucker_start=None, plucker_end=None, pose_vec=None,
            rgb_circular_pad=self.rgb_circular_pad,
            num_views=self.num_views,
        ).images
        return [(v[0] if isinstance(v, list) else v) for v in out]


# ---------------------------------------------------------------------------
# wild input: build the per-case tensors straight from disk PNGs
# ---------------------------------------------------------------------------
def _load_wild_case(start_dir, warp_dir, num_views, img_size, rgb_circular_pad,
                    system_prompt):
    """Read origin.png + caption.txt (start_dir) and erp_aux_i/mask_aux_i
    (warp_dir) into the tensors :meth:`QwenFourViewMaskEncode.run_wild` expects.

    Resize mirrors training: start RGB area-downsample to (img_size*2, img_size);
    warps area; masks NEAREST (kept binary). Start gets the same optional ERP
    circular pad as the dataset; warps/masks stay unpadded (the pipeline pads
    them internally), exactly as the online ``run`` does.
    """
    import cv2  # local import: only the wild path needs OpenCV here.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    train_h, train_w = int(img_size), int(img_size) * 2

    origin = os.path.join(start_dir, "origin.png")
    bgr = cv2.imread(origin, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing/unreadable origin.png: {origin}")
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (train_w, train_h), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    start_t = torch.from_numpy(
        np.ascontiguousarray(rgb.astype(np.float32) / 127.5 - 1.0)
    ).permute(2, 0, 1).contiguous()
    if rgb_circular_pad:
        start_t = _circular_pad_width_torch(start_t)
    start_pil = _tensor_to_pil(start_t)

    cap_path = os.path.join(start_dir, "caption.txt")
    caption = ""
    if os.path.exists(cap_path):
        with open(cap_path, "r") as f:
            caption = f.read().strip()
    prompt = system_prompt + caption

    wrap_images: List[torch.Tensor] = []
    wrap_masks: List[torch.Tensor] = []
    warp_pils: List[Image.Image] = []
    mask_pils: List[Image.Image] = []
    for v in range(num_views):
        wp = os.path.join(warp_dir, f"erp_aux_{v}.png")
        mp = os.path.join(warp_dir, f"mask_aux_{v}.png")
        wbgr = cv2.imread(wp, cv2.IMREAD_COLOR)
        if wbgr is None:
            raise FileNotFoundError(f"missing/unreadable warp: {wp}")
        wrgb = cv2.cvtColor(
            cv2.resize(wbgr, (train_w, train_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        wt = torch.from_numpy(
            np.ascontiguousarray(wrgb.astype(np.float32) / 127.5 - 1.0)
        ).permute(2, 0, 1).contiguous()
        wrap_images.append(wt.unsqueeze(0))            # (1,3,H,W)
        warp_pils.append(_tensor_to_pil(wt))

        mgray = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mgray is None:
            raise FileNotFoundError(f"missing/unreadable mask: {mp}")
        mgray = cv2.resize(mgray, (train_w, train_h), interpolation=cv2.INTER_NEAREST)
        mbin = (mgray > 127).astype(np.float32)
        mt = torch.from_numpy(np.ascontiguousarray(mbin))[None]   # (1,H,W)
        wrap_masks.append(mt.unsqueeze(0))             # (1,1,H,W)
        mask_pils.append(
            Image.fromarray((mbin * 255).astype(np.uint8)).convert("RGB")
        )

    return start_pil, wrap_images, wrap_masks, warp_pils, mask_pils, prompt


# ---------------------------------------------------------------------------
# saving / preview
# ---------------------------------------------------------------------------
def _font(size=28):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _paste_label(canvas, draw, x0, y0, im, label, cell=(1024, 512), bar_w=260):
    canvas.paste(im.resize(cell), (x0, y0))
    draw.rectangle([(x0 + 10, y0 + 10), (x0 + 10 + bar_w, y0 + 56)], fill="black")
    draw.text((x0 + 20, y0 + 15), label, fill="white", font=_font())


def _save_case(output_dir, case_id, directions, outputs, start_pil,
               warp_pils, mask_pils, gt_pils):
    case_dir = os.path.join(output_dir, str(case_id))
    os.makedirs(case_dir, exist_ok=True)
    cw, ch = 1024, 512

    for v, dname in enumerate(directions):
        d_dir = os.path.join(case_dir, dname)
        os.makedirs(d_dir, exist_ok=True)
        outputs[v].save(os.path.join(d_dir, "output.png"))
        # Save the YAW-ALIGNED GT (the frame the model actually predicts under
        # end_to_start) so the evaluator compares like-for-like. The raw end.jpg
        # on disk is in the un-rolled frame and would mismatch.
        gt_pils[v].save(os.path.join(d_dir, "gt.png"))

        canvas = Image.new("RGB", (cw * 3, ch * 2), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        row1 = [(start_pil, "START"),
                (warp_pils[v] or Image.new("RGB", (cw, ch)), "WARP"),
                (mask_pils[v] or Image.new("RGB", (cw, ch)), "MASK")]
        row2 = [(outputs[v], "OUTPUT"), (gt_pils[v], "GT")]
        for c, (im, lb) in enumerate(row1):
            _paste_label(canvas, draw, c * cw, 0, im, lb, (cw, ch))
        for c, (im, lb) in enumerate(row2):
            _paste_label(canvas, draw, c * cw, ch, im, lb, (cw, ch))
        canvas.save(os.path.join(d_dir, "concat.jpg"), quality=90)

    # four_view_grid: header (START) + V rows of (WARP | GT | PRED)
    grows = 1 + len(directions)
    grid = Image.new("RGB", (cw * 3, ch * grows), (0, 0, 0))
    gd = ImageDraw.Draw(grid)
    _paste_label(grid, gd, 0, 0, start_pil, "START", (cw, ch))
    for r, dname in enumerate(directions):
        y0 = (1 + r) * ch
        _paste_label(grid, gd, 0, y0, warp_pils[r] or Image.new("RGB", (cw, ch)),
                     f"{dname}|WARP", (cw, ch), bar_w=320)
        _paste_label(grid, gd, cw, y0, gt_pils[r], "GT", (cw, ch), bar_w=140)
        _paste_label(grid, gd, 2 * cw, y0, outputs[r], "PRED", (cw, ch), bar_w=140)
    grid.save(os.path.join(case_dir, "four_view_grid.jpg"), quality=95)


def _is_case_done(output_dir, case_id, directions):
    d = os.path.join(output_dir, str(case_id))
    return all(os.path.exists(os.path.join(d, dr, "output.png")) for dr in directions)


def _save_case_wild(output_dir, case_id, directions, outputs, start_pil,
                    warp_pils, mask_pils):
    """Wild save: per-view output.png + a START|WARP|MASK|OUTPUT preview (no GT)."""
    case_dir = os.path.join(output_dir, str(case_id))
    os.makedirs(case_dir, exist_ok=True)
    cw, ch = 1024, 512
    for v, dname in enumerate(directions):
        d_dir = os.path.join(case_dir, dname)
        os.makedirs(d_dir, exist_ok=True)
        outputs[v].save(os.path.join(d_dir, "output.png"))
        canvas = Image.new("RGB", (cw * 4, ch), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        row = [(start_pil, "START"),
               (warp_pils[v] or Image.new("RGB", (cw, ch)), "WARP"),
               (mask_pils[v] or Image.new("RGB", (cw, ch)), "MASK"),
               (outputs[v], "OUTPUT")]
        for c, (im, lb) in enumerate(row):
            _paste_label(canvas, draw, c * cw, 0, im, lb, (cw, ch))
        canvas.save(os.path.join(d_dir, "concat.jpg"), quality=90)

    # Case-level merged grid (no GT in wild): header START + V rows of WARP|PRED.
    grows = 1 + len(directions)
    grid = Image.new("RGB", (cw * 2, ch * grows), (0, 0, 0))
    gd = ImageDraw.Draw(grid)
    _paste_label(grid, gd, 0, 0, start_pil, "START", (cw, ch))
    for r, dname in enumerate(directions):
        y0 = (1 + r) * ch
        _paste_label(grid, gd, 0, y0, warp_pils[r] or Image.new("RGB", (cw, ch)),
                     f"{dname}|WARP", (cw, ch), bar_w=320)
        _paste_label(grid, gd, cw, y0, outputs[r], "PRED", (cw, ch), bar_w=140)
    grid.save(os.path.join(case_dir, "four_view_grid.jpg"), quality=95)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def worker(gpu_id, indices, output_dir, ckpt_path, data_config,
           num_views, attention_policy, local_attn_variant, local_layer_indices,
           flow_shift, height, width, num_inference_steps,
           true_cfg_scale, guidance_scale, skip_lora, model_path,
           use_mask_encoder_override):
    try:
        print(f"GPU {gpu_id}: building dataset ({len(indices)} cases assigned)...", flush=True)
        # Strip loader-only keys; the rest goes straight to the dataset.
        ds_kwargs = {k: v for k, v in data_config.items()
                     if k not in ("train_batch_size", "num_workers")}
        ds_kwargs["drop_text_prob"] = 0.0   # never drop text at eval
        dataset = FourViewOnlineWarpDataset(**ds_kwargs)

        demo = QwenFourViewMaskEncode(
            ckpt_path, gpu_id, data_config=data_config,
            num_views=num_views, attention_policy=attention_policy,
            local_attn_variant=local_attn_variant,
            local_layer_indices=local_layer_indices, flow_shift=flow_shift,
            height=height, width=width, num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale, guidance_scale=guidance_scale,
            skip_lora=skip_lora, model_path=model_path,
            use_mask_encoder_override=use_mask_encoder_override,
        )
        print(f"GPU {gpu_id}: model + renderer ready.", flush=True)

        for n, idx in enumerate(indices):
            try:
                sample = dataset[idx]
                case_id = sample["case_id"]
                directions = sample["directions"]
                if _is_case_done(output_dir, case_id, directions):
                    print(f"GPU {gpu_id}: case {case_id} already done, skip.", flush=True)
                    continue

                print(f"GPU {gpu_id} [{n+1}/{len(indices)}] case={case_id} "
                      f"dirs={directions}", flush=True)
                outputs, start_pil, warp_pils, mask_pils, gt_pils = demo.run(sample)
                _save_case(output_dir, case_id, directions, outputs,
                           start_pil, warp_pils, mask_pils, gt_pils)
                print(f"GPU {gpu_id}: done case={case_id}", flush=True)
            except Exception as e:
                print(f"GPU {gpu_id}: error on dataset idx {idx}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                continue
    except Exception as e:
        print(f"GPU {gpu_id} FATAL: {e}\n{traceback.format_exc()}", flush=True)


# ---------------------------------------------------------------------------
# worker (wild: disk-fed warps/masks, no dataset, no renderer)
# ---------------------------------------------------------------------------
def worker_wild(gpu_id, case_list, output_dir, ckpt_path, data_config,
                num_views, attention_policy, local_attn_variant, local_layer_indices,
                flow_shift, height, width, num_inference_steps,
                true_cfg_scale, guidance_scale, skip_lora, model_path,
                use_mask_encoder_override):
    try:
        demo = QwenFourViewMaskEncode(
            ckpt_path, gpu_id, data_config=data_config,
            num_views=num_views, attention_policy=attention_policy,
            local_attn_variant=local_attn_variant,
            local_layer_indices=local_layer_indices, flow_shift=flow_shift,
            height=height, width=width, num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale, guidance_scale=guidance_scale,
            skip_lora=skip_lora, model_path=model_path,
            use_mask_encoder_override=use_mask_encoder_override,
            build_renderer=False,
        )
        img_size = int(data_config.get("img_size", 512))
        rgb_circular_pad = bool(data_config.get("rgb_circular_pad", False))
        directions = [f"aux_{v}" for v in range(num_views)]
        print(f"GPU {gpu_id}: wild model ready ({len(case_list)} cases assigned).", flush=True)

        for n, (case_id, start_dir, warp_dir) in enumerate(case_list):
            try:
                if _is_case_done(output_dir, case_id, directions):
                    print(f"GPU {gpu_id}: case {case_id} already done, skip.", flush=True)
                    continue
                print(f"GPU {gpu_id} [{n+1}/{len(case_list)}] wild case={case_id}", flush=True)
                start_pil, wi, wm, warp_pils, mask_pils, prompt = _load_wild_case(
                    start_dir, warp_dir, num_views, img_size, rgb_circular_pad,
                    FourViewOnlineWarpDataset.SYSTEM_PROMPT,
                )
                outputs = demo.run_wild(start_pil, wi, wm, prompt)
                _save_case_wild(output_dir, case_id, directions, outputs,
                                start_pil, warp_pils, mask_pils)
                print(f"GPU {gpu_id}: done case={case_id}", flush=True)
            except Exception as e:
                print(f"GPU {gpu_id}: error on wild case {case_id}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                continue
    except Exception as e:
        print(f"GPU {gpu_id} FATAL: {e}\n{traceback.format_exc()}", flush=True)


# ---------------------------------------------------------------------------
# wild driver: discover <case_id> folders, shard, spawn worker_wild
# ---------------------------------------------------------------------------
def _run_wild(args, data_config, num_views, attention_policy,
              local_attn_variant, local_layer_indices, flow_shift):
    if not args.start_root or not args.warp_root:
        sys.exit("--wild requires both --start_root and --warp_root.")
    if not os.path.isdir(args.start_root):
        sys.exit(f"--start_root not a directory: {args.start_root}")
    if not os.path.isdir(args.warp_root):
        sys.exit(f"--warp_root not a directory: {args.warp_root}")

    keep_ids: Optional[set] = None
    if args.ids_txt:
        with open(args.ids_txt) as f:
            keep_ids = {ln.strip() for ln in f if ln.strip()}

    # A case is valid iff <id>/ exists under BOTH roots (intersection).
    case_list: List[tuple] = []
    for name in sorted(os.listdir(args.start_root)):
        if keep_ids is not None and name not in keep_ids:
            continue
        sd = os.path.join(args.start_root, name)
        wd = os.path.join(args.warp_root, name)
        if os.path.isdir(sd) and os.path.isdir(wd):
            case_list.append((name, sd, wd))

    if not case_list:
        print("[wild] no cases found (need <id>/ under both roots"
              + (" and in --ids_txt" if keep_ids is not None else "") + ").")
        return
    if args.max_samples is not None:
        case_list = case_list[: args.max_samples]

    # Multi-node round-robin shard.
    if args.num_nodes < 1 or not (0 <= args.node_rank < args.num_nodes):
        sys.exit(f"Bad --num_nodes/--node_rank: {args.num_nodes}/{args.node_rank}")
    if args.num_nodes > 1:
        node_cases = case_list[args.node_rank::args.num_nodes]
        print(f"[multi-node] global={len(case_list)} -> node "
              f"{args.node_rank}/{args.num_nodes} gets {len(node_cases)}")
        case_list = node_cases
    if not case_list:
        print("Nothing to do on this node; exit.")
        return

    print(
        f"[WILD INFER CONFIG] cases={len(case_list)} num_views={num_views} "
        f"attention_policy={attention_policy} local_attn_variant={local_attn_variant} "
        f"local_layer_indices={local_layer_indices} "
        f"flow_shift={flow_shift} gen={args.height}x{args.width} "
        f"steps={args.num_inference_steps} true_cfg={args.true_cfg_scale} "
        f"guidance={args.guidance_scale} img_size={data_config.get('img_size', 512)} "
        f"rgb_circular_pad={data_config.get('rgb_circular_pad', False)}",
        flush=True,
    )

    gpu_cases = [case_list[i::args.num_gpus] for i in range(args.num_gpus)]
    for i in range(args.num_gpus):
        print(f"GPU {i} (node {args.node_rank}): {len(gpu_cases[i])} cases")

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id in range(args.num_gpus):
        if not gpu_cases[gpu_id]:
            continue
        p = mp.Process(
            target=worker_wild,
            args=(
                gpu_id, gpu_cases[gpu_id], args.output_path, args.ckpt_path,
                data_config, num_views, attention_policy, local_attn_variant,
                local_layer_indices,
                flow_shift, args.height, args.width, args.num_inference_steps,
                args.true_cfg_scale, args.guidance_scale, args.skip_lora,
                args.model_path, args.use_mask_encoder,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("All processing complete! (wild)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Four-view ONLINE-warp batch inference (TUPLE-v1 input)."
    )
    ap.add_argument("--json_path", type=str, default=None,
                    help="four-view-v1 JSON ({'cases':[{start, views:[...]}]}, "
                         "e.g. data_0519/split/test_subset.json) — same schema "
                         "as training. Legacy TUPLE-v1 is auto-detected too. "
                         "Required unless --wild is set.")
    # ---- wild mode: warps/masks read from disk (no online render) -------
    ap.add_argument("--wild", action="store_true", default=False,
                    help="Wild inference: read precomputed warps/masks from disk "
                         "instead of rendering online. Requires --start_root and "
                         "--warp_root; --json_path is ignored.")
    ap.add_argument("--start_root", type=str, default=None,
                    help="[wild] Parent dir; each <case_id>/ holds origin.png + "
                         "caption.txt (the start panorama + prompt).")
    ap.add_argument("--warp_root", type=str, default=None,
                    help="[wild] Parent dir; each <case_id>/ holds "
                         "erp_aux_0..{V-1}.png + mask_aux_0..{V-1}.png.")
    ap.add_argument("--ids_txt", type=str, default=None,
                    help="[wild] Optional txt (one case_id per line) to restrict "
                         "which subfolders are processed.")
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--model_path", type=str,
                    default=os.environ.get("MODEL_PATH", None),
                    help="Base Qwen-Image-Edit-2509 path. Defaults to the "
                         "MODEL_PATH env var, else the in-script fallback.")
    ap.add_argument("--config", type=str, required=True,
                    help="Training run YAML (e.g. the runtime "
                         "I2P_four_view_runtime.yaml). Its data_config + "
                         "num_views/attention_policy/local_attn_variant are "
                         "reused verbatim for train/infer parity. Only "
                         "train_files is overridden by --json_path.")
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--node_rank", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--skip_lora", action="store_true", default=False)
    ap.add_argument("--use_mask_encoder", type=str, default="auto",
                    choices=["auto", "true", "false"],
                    help="Mask-encoder injection toggle at inference. 'auto' "
                         "(default) uses the checkpoint's meta.json "
                         "use_mask_encoder; 'true'/'false' force it — e.g. "
                         "'false' to run a womask "
                         "(I2P_four_view_online_womaskencoder) checkpoint whose "
                         "meta is missing/stale, or to A/B a withmask ckpt with "
                         "injection disabled.")
    # Sampler knobs (override the YAML / defaults).
    ap.add_argument("--flow_shift", type=float, default=None)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--num_inference_steps", type=int, default=40)
    ap.add_argument("--true_cfg_scale", type=float, default=4.0)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    data_config = OmegaConf.to_container(cfg.data_config, resolve=True)

    num_views = int(cfg.get("num_views", data_config.get("num_views", 4)))
    attention_policy = str(cfg.get("attention_policy", "global"))
    local_attn_variant = str(cfg.get("local_attn_variant", "v1"))
    # Per-layer G/L schedule (e.g. *_v1local30 runs). When set it OVERRIDES
    # attention_policy — read it back here so infer matches train.
    local_layer_indices = cfg.get("local_layer_indices", None)
    if local_layer_indices is not None:
        local_layer_indices = OmegaConf.to_container(local_layer_indices, resolve=True) \
            if OmegaConf.is_config(local_layer_indices) else local_layer_indices
    flow_shift = args.flow_shift if args.flow_shift is not None else float(cfg.get("flow_shift", 1.0))

    if args.wild:
        return _run_wild(args, data_config, num_views, attention_policy,
                         local_attn_variant, local_layer_indices, flow_shift)

    # Unify the input: replace the training JSON with the eval TUPLE-v1 JSON.
    if not args.json_path:
        sys.exit("--json_path is required unless --wild is set.")
    data_config["train_files"] = [args.json_path]

    # Count complete 4-view groups by building the dataset once on CPU.
    ds_kwargs = {k: v for k, v in data_config.items()
                 if k not in ("train_batch_size", "num_workers")}
    ds_kwargs["drop_text_prob"] = 0.0
    probe = FourViewOnlineWarpDataset(**ds_kwargs)
    n_total = len(probe)
    del probe

    all_indices = list(range(n_total))
    if args.max_samples is not None:
        all_indices = all_indices[: args.max_samples]

    # Multi-node round-robin shard.
    if args.num_nodes < 1 or not (0 <= args.node_rank < args.num_nodes):
        sys.exit(f"Bad --num_nodes/--node_rank: {args.num_nodes}/{args.node_rank}")
    if args.num_nodes > 1:
        node_indices = all_indices[args.node_rank::args.num_nodes]
        print(f"[multi-node] global={len(all_indices)} -> "
              f"node {args.node_rank}/{args.num_nodes} gets {len(node_indices)}")
        all_indices = node_indices
    if not all_indices:
        print("Nothing to do on this node; exit.")
        return

    print(
        f"[INFER CONFIG] cases={len(all_indices)} num_views={num_views} "
        f"attention_policy={attention_policy} local_attn_variant={local_attn_variant} "
        f"local_layer_indices={local_layer_indices} "
        f"flow_shift={flow_shift} gen={args.height}x{args.width} "
        f"steps={args.num_inference_steps} true_cfg={args.true_cfg_scale} "
        f"guidance={args.guidance_scale} rgb_circular_pad="
        f"{data_config.get('rgb_circular_pad', False)}",
        flush=True,
    )

    # Round-robin split across GPUs (balances heterogeneous case cost).
    gpu_indices = [all_indices[i::args.num_gpus] for i in range(args.num_gpus)]
    for i in range(args.num_gpus):
        print(f"GPU {i} (node {args.node_rank}): {len(gpu_indices[i])} cases")

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id in range(args.num_gpus):
        if not gpu_indices[gpu_id]:
            continue
        p = mp.Process(
            target=worker,
            args=(
                gpu_id, gpu_indices[gpu_id], args.output_path, args.ckpt_path,
                data_config, num_views, attention_policy, local_attn_variant,
                local_layer_indices,
                flow_shift, args.height, args.width, args.num_inference_steps,
                args.true_cfg_scale, args.guidance_scale, args.skip_lora,
                args.model_path, args.use_mask_encoder,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("All processing complete!")


if __name__ == "__main__":
    main()
