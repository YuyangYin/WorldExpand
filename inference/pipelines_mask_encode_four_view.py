"""Four-view inference pipeline (mask + Plücker + pose), merged.

Merges:
* the 4-view joint denoising of Pano_four_full/pipelines_four_view.py
  (1 shared start + V warps -> V jointly-consistent panoramas), and
* this repo's mask / Plücker / pose conditioning
  (inference/pipelines_mask_encode.py).

Pairs with training/models/custom_qwen_transformer_four_view_mask_concat.py.

Inputs (per __call__):
    image        : shared start panorama (PIL or list with 1 PIL)
    wrap_images  : list of V tensors (1,3,H,W) in [-1,1]  (online-rendered warps)
    wrap_masks   : list of V tensors (1,1,H,W) in [0,1]
    prompt       : text prompt (shared)
    plucker_start: (1,6,H,W)         shared start ERP Plücker, optional
    plucker_end  : (1,V,6,H,W)       per-view end ERP Plücker, optional
    pose_vec     : (1,V,3)           per-view relative translation, optional

Plücker/pose are only forwarded when the loaded transformer has the matching
``inject_*`` flag set, exactly like the single-view pipeline.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
    QwenImagePipelineOutput,
    PipelineImageInput,
    calculate_shift,
    retrieve_timesteps,
    calculate_dimensions,
)
from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 512 * 1024


class HunyuanPanoPipelineFourViewMaskEncode(QwenImageEditPlusPipeline):
    """4-view joint panorama inference with mask + Plücker + pose conditioning."""

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 40,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        wrap_images: Optional[List[torch.Tensor]] = None,
        wrap_masks: Optional[List[torch.Tensor]] = None,
        plucker_start: Optional[torch.Tensor] = None,   # (1,6,H,W)
        plucker_end: Optional[torch.Tensor] = None,     # (1,V,6,H,W)
        pose_vec: Optional[torch.Tensor] = None,        # (1,V,3)
        rgb_circular_pad: bool = True,
        num_views: int = 4,
    ):
        assert wrap_images is not None and len(wrap_images) == num_views, \
            f"Need {num_views} wrap_images, got {0 if wrap_images is None else len(wrap_images)}"
        assert wrap_masks is not None and len(wrap_masks) == num_views, \
            f"Need {num_views} wrap_masks, got {0 if wrap_masks is None else len(wrap_masks)}"

        # ---- resolve sizes ----
        image_size = image[-1].size if isinstance(image, list) else image.size
        calc_w, calc_h = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calc_h
        width = width or calc_w
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self.check_inputs(
            prompt, height, width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # ---- preprocess start image for VLM + VAE ----
        if not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            cond_images = []
            vae_images = []
            for img in image:
                iw, ih = img.size
                cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, iw / ih)
                if rgb_circular_pad:
                    vae_w = iw // multiple_of * multiple_of
                    vae_h = ih // multiple_of * multiple_of
                else:
                    vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, iw / ih)
                cond_images.append(self.image_processor.resize(img, ch, cw))
                vae_images.append(self.image_processor.preprocess(img, vae_h, vae_w))

        has_neg = (negative_prompt is not None) or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=cond_images, prompt=prompt,
            prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
            device=device, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=cond_images, prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds, prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # ---- latent helpers ----
        num_channels_latents = self.transformer.config.in_channels // 4
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(device, prompt_embeds.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(device, prompt_embeds.dtype)

        if rgb_circular_pad:
            padded_w, padded_h = image_size
            actual_h = padded_h // multiple_of * multiple_of
            actual_w = padded_w // multiple_of * multiple_of
        else:
            actual_h, actual_w = vae_h, vae_w

        latent_h = 2 * (actual_h // (self.vae_scale_factor * 2))
        latent_w = 2 * (actual_w // (self.vae_scale_factor * 2))

        # ---- encode start (shared) ----
        start_tensor = vae_images[0].to(device=device, dtype=self.vae.dtype)
        start_lat = self.vae.encode(start_tensor.unsqueeze(2)).latent_dist.sample()
        start_lat = (start_lat - latents_mean) * latents_std
        start_lat = start_lat.to(dtype=prompt_embeds.dtype)
        if not rgb_circular_pad:
            start_lat = torch.cat([start_lat, start_lat[..., :4]], dim=-1)
        packed_start = self._pack_latents(
            start_lat, batch_size=start_lat.shape[0],
            num_channels_latents=start_lat.shape[1],
            height=start_lat.shape[3], width=start_lat.shape[4],
        )

        # ---- encode warp images (V) ----
        packed_warp_list = []
        warp_latent_h, warp_latent_w = None, None
        for w_img in wrap_images:
            w_img = w_img.to(device=device, dtype=self.vae.dtype)
            # Sanitise before VAE encode: some geometry renderers (nvdiffrast
            # mesh, pytorch3d pcd) can emit a few NaN pixels which clamp() does
            # NOT remove; a single NaN pixel poisons the whole VAE latent ->
            # model outputs NaN -> black frame. gsplat rarely NaNs so it was
            # unaffected. Zero-fill any NaN/Inf to make every engine safe.
            w_img = torch.nan_to_num(w_img, nan=0.0, posinf=1.0, neginf=-1.0)
            if w_img.dim() == 3:
                w_img = w_img.unsqueeze(0)
            w_lat = self.vae.encode(w_img.unsqueeze(2)).latent_dist.sample()
            w_lat = (w_lat - latents_mean) * latents_std
            w_lat = w_lat.to(dtype=prompt_embeds.dtype)
            if not rgb_circular_pad:
                w_lat = torch.cat([w_lat, w_lat[..., :4]], dim=-1)
            if warp_latent_h is None:
                warp_latent_h, warp_latent_w = w_lat.shape[3], w_lat.shape[4]
            p = self._pack_latents(
                w_lat, batch_size=w_lat.shape[0],
                num_channels_latents=w_lat.shape[1],
                height=w_lat.shape[3], width=w_lat.shape[4],
            )
            packed_warp_list.append(p)
        packed_warp = torch.cat(packed_warp_list, dim=1)

        # ---- stack V masks along H (pixel space) ----
        mask_stack = []
        for m in wrap_masks:
            if m.dim() == 3:
                m = m.unsqueeze(0)
            m = m.to(device=device, dtype=prompt_embeds.dtype)
            if not rgb_circular_pad:
                m = torch.cat([m, m[..., :4 * self.vae_scale_factor]], dim=-1)
            mask_stack.append(m)
        wrap_mask_tensor = torch.cat(mask_stack, dim=2)  # (B, 1, V*H, W)

        # ---- Plücker / pose tensors (only when the transformer wants them) ----
        plucker_start_tensor = None
        plucker_end_tensor = None
        if getattr(self.transformer, "inject_plucker", False):
            if plucker_start is None or plucker_end is None:
                raise ValueError(
                    "transformer.inject_plucker=True but plucker_start/plucker_end "
                    "were not supplied to the 4-view pipeline."
                )
            plucker_start_tensor = plucker_start.to(device=device, dtype=prompt_embeds.dtype)
            plucker_end_tensor = plucker_end.to(device=device, dtype=prompt_embeds.dtype)
            if not rgb_circular_pad:
                _pp = 4 * self.vae_scale_factor
                # last-dim (W) pad works for (1,6,H,W) and (1,V,6,H,W) alike.
                plucker_start_tensor = torch.cat(
                    [plucker_start_tensor, plucker_start_tensor[..., :_pp]], dim=-1)
                plucker_end_tensor = torch.cat(
                    [plucker_end_tensor, plucker_end_tensor[..., :_pp]], dim=-1)

        pose_vec_tensor = None
        if (getattr(self.transformer, "inject_pose_temb", False)
                or getattr(self.transformer, "inject_pose_film", False)
                or getattr(self.transformer, "inject_dist_temb", False)):
            if pose_vec is None:
                raise ValueError(
                    "transformer pose conditioning is enabled but pose_vec was "
                    "not supplied to the 4-view pipeline."
                )
            pose_vec_tensor = pose_vec.to(device=device, dtype=prompt_embeds.dtype)

        # ---- initial noise for V views ----
        noise_shape = (batch_size * num_images_per_prompt, 1, num_channels_latents, latent_h, latent_w)
        noise_views = []
        for _v in range(num_views):
            n = randn_tensor(noise_shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
            noise_views.append(self._pack_latents(
                n, batch_size=n.shape[0], num_channels_latents=num_channels_latents,
                height=latent_h, width=latent_w,
            ))
        latents = torch.cat(noise_views, dim=1)          # (B, V*seq_per_view, 64)
        seq_per_view = latents.shape[1] // num_views

        # ---- img_shapes per-view (V + 1 + V) ----
        face_shape = (1, latent_h // 2, latent_w // 2)
        if warp_latent_h is not None and (warp_latent_h != latent_h or warp_latent_w != latent_w):
            warp_face_shape = (1, warp_latent_h // 2, warp_latent_w // 2)
        else:
            warp_face_shape = face_shape
        img_shapes = [[
            *[face_shape for _ in range(num_views)],
            face_shape,
            *[warp_face_shape for _ in range(num_views)],
        ]] * batch_size

        # ---- scheduler ----
        sigmas_ = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas_, mu=mu,
        )
        num_warmup = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        neg_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # ---- denoise loop ----
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        packed_noisy_model_input=latents,
                        packed_model_condition_input=packed_start,
                        packed_model_wrap_image=packed_warp,
                        wrap_mask=wrap_mask_tensor,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        plucker_start=plucker_start_tensor,
                        plucker_end=plucker_end_tensor,
                        pose_vec=pose_vec_tensor,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg = self.transformer(
                            packed_noisy_model_input=latents,
                            packed_model_condition_input=packed_start,
                            packed_model_wrap_image=packed_warp,
                            wrap_mask=wrap_mask_tensor,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=neg_txt_seq_lens,
                            plucker_start=plucker_start_tensor,
                            plucker_end=plucker_end_tensor,
                            pose_vec=pose_vec_tensor,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                        neg = neg[:, : latents.size(1)]
                    combined = neg + true_cfg_scale * (noise_pred - neg)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(combined, dim=-1, keepdim=True)
                    noise_pred = combined * (cond_norm / noise_norm)

                lat_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != lat_dtype and torch.backends.mps.is_available():
                    latents = latents.to(lat_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup and (i + 1) % self.scheduler.order == 0):
                    pbar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # ---- decode each view ----
        out_images = []
        for v in range(num_views):
            lat_v = latents[:, seq_per_view * v: seq_per_view * (v + 1)]
            if output_type == "latent":
                out_images.append(lat_v)
                continue
            unpack_w = actual_w if rgb_circular_pad else actual_w + 4 * self.vae_scale_factor
            unpack = self._unpack_latents(lat_v, actual_h, unpack_w, self.vae_scale_factor)
            unpack = unpack.to(self.vae.dtype)
            unpack = unpack / latents_std + latents_mean
            decoded = self.vae.decode(unpack, return_dict=False)[0][:, :, 0]
            if rgb_circular_pad:
                crop_w = decoded.shape[-1] // 18
                decoded = decoded[:, :, :, crop_w:-crop_w]
            img_v = self.image_processor.postprocess(decoded, output_type=output_type)
            out_images.append(img_v)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (out_images,)
        return QwenImagePipelineOutput(images=out_images)
