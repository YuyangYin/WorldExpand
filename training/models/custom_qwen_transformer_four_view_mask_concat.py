"""Four-view mask-concat Qwen-Image transformer (merged).

This MERGES two existing transformers onto one class:

* 4-view joint attention + global/local (G/L) policy schedule + per-view
  packing, from ``Pano_four_full/custom_qwen_transformer_four_view.py``.
* mask_encoder / mask_proj + Plücker ray injection + camera-pose conditioning
  (pose_temb / pose_film / dist_temb), from this repo's
  ``training/models/custom_qwen_transformer_mask_concat.py``.

Token layout along the sequence dim (one batch item), the "G" (global) format::

    [ noise_v0 | noise_v1 | noise_v2 | noise_v3 | start | warp_v0 | warp_v1 | warp_v2 | warp_v3 ]
       V * L_n tokens                               L_n     V * L_w tokens

All conditioning (mask / Plücker / pose-FiLM) is injected ONCE in this G
format, right after ``img_in`` and BEFORE the block loop, so it never has to
reason about the G<->L reshape (the reshape just shuttles the already-injected
tokens). pose_temb / dist_temb ride ``temb`` (global AdaLN); since ``temb`` is a
single per-batch vector, the per-view pose vectors are averaged over V before
the temb embedding (FiLM keeps per-view granularity).

Loss is computed by the trainer on the first ``V * L_n`` (noise) output tokens.

NOTE on diffusers drift: this repo's ``QwenEmbedRope.forward`` (a.k.a.
``self.pos_embed``) uses the signature ``pos_embed(img_shapes, device=,
max_txt_seq_len=)`` — NOT the older ``pos_embed(img_shapes, txt_seq_lens,
device=)`` used by the legacy four-view repo. We use the new signature here.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers


logger = logging.get_logger(__name__)

_POSE_DEBUG = os.environ.get("POSE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


VALID_POLICIES = {"global", "local", "alt_G_L", "alt_L_G", "half_GL", "half_LG",
                  "G_LL", "G_LLLL"}
VALID_LOCAL_VARIANTS = {"v1", "v2", "v3"}


def build_policy_schedule(policy: str, num_layers: int) -> List[str]:
    """Return list[str] of length num_layers with "G"/"L" per block.

    Verbatim from the legacy four-view transformer.
    """
    if policy == "global":
        return ["G"] * num_layers
    if policy == "local":
        return ["L"] * num_layers
    if policy == "alt_G_L":
        return ["G" if (i % 2 == 0) else "L" for i in range(num_layers)]
    if policy == "alt_L_G":
        return ["L" if (i % 2 == 0) else "G" for i in range(num_layers)]
    if policy == "half_GL":
        half = num_layers // 2
        return ["G"] * half + ["L"] * (num_layers - half)
    if policy == "half_LG":
        half = num_layers // 2
        return ["L"] * half + ["G"] * (num_layers - half)
    if policy == "G_LL":
        return ["G" if (i == 0 or i % 3 == 0) else "L" for i in range(num_layers)]
    if policy == "G_LLLL":
        return ["G" if (i == 0 or i % 5 == 0) else "L" for i in range(num_layers)]
    raise ValueError(f"Unknown attention_policy={policy!r}; choose from {sorted(VALID_POLICIES)}")


def _stat(name: str, t: torch.Tensor) -> str:
    tf = t.detach().float()
    finite = bool(torch.isfinite(tf).all().item())
    if tf.numel() == 0:
        return f"{name}: empty"
    return (f"{name}: finite={finite} shape={tuple(t.shape)} "
            f"min={tf.min().item():.3e} max={tf.max().item():.3e} "
            f"absmax={tf.abs().max().item():.3e}")


class PoseEmbed(nn.Module):
    """Fourier-feature camera-pose -> inner_dim embedding (Option A).

    Identical to the single-view recipe: last linear zero-init so the
    embedding is 0 at step 0 (adding to temb is a no-op).
    """

    def __init__(self, inner_dim: int, in_dim: int = 3, n_freq: int = 10):
        super().__init__()
        self.in_dim = int(in_dim)
        freqs = (2.0 ** torch.arange(n_freq, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        feat_dim = in_dim * 2 * n_freq
        self.lin1 = nn.Linear(feat_dim, inner_dim)
        self.lin2 = nn.Linear(inner_dim, inner_dim)
        with torch.no_grad():
            self.lin1.weight.normal_(mean=0.0, std=0.02)
            self.lin1.bias.zero_()
            self.lin2.weight.zero_()
            self.lin2.bias.zero_()

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        pose = torch.nan_to_num(pose.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        ang = pose[..., None] * self.freqs.float()
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
        h = F.silu(F.linear(feat, self.lin1.weight.float(), self.lin1.bias.float()))
        out = F.linear(h, self.lin2.weight.float(), self.lin2.bias.float())
        return torch.nan_to_num(out)


class PoseFiLM(nn.Module):
    """Camera-pose -> global FiLM (scale, shift) modulation (Option B)."""

    def __init__(self, inner_dim: int, in_dim: int = 3, n_freq: int = 10):
        super().__init__()
        self.in_dim = int(in_dim)
        freqs = (2.0 ** torch.arange(n_freq, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        feat_dim = in_dim * 2 * n_freq
        self.lin1 = nn.Linear(feat_dim, inner_dim)
        self.to_scale_shift = nn.Linear(inner_dim, 2 * inner_dim)
        with torch.no_grad():
            self.lin1.weight.normal_(mean=0.0, std=0.02)
            self.lin1.bias.zero_()
            self.to_scale_shift.weight.zero_()
            self.to_scale_shift.bias.zero_()

    def forward(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pose = torch.nan_to_num(pose.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        ang = pose[..., None] * self.freqs.float()
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
        h = F.silu(F.linear(feat, self.lin1.weight.float(), self.lin1.bias.float()))
        ss = torch.nan_to_num(F.linear(h, self.to_scale_shift.weight.float(), self.to_scale_shift.bias.float()))
        scale_raw, shift = ss.chunk(2, dim=-1)
        scale = torch.tanh(scale_raw)
        return scale, shift


class CustomQwenImageTransformer2DModelFourViewMaskConcat(QwenImageTransformer2DModel):
    """Four-view joint Qwen-Image transformer with mask + Plücker + pose heads."""

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        num_views: int = 4,
        attention_policy: str = "global",
        local_attn_variant: str = "v1",
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

        self.num_views = int(num_views)
        if attention_policy not in VALID_POLICIES:
            raise ValueError(f"attention_policy={attention_policy!r} not in {sorted(VALID_POLICIES)}")
        if local_attn_variant not in VALID_LOCAL_VARIANTS:
            raise ValueError(f"local_attn_variant={local_attn_variant!r} not in {sorted(VALID_LOCAL_VARIANTS)}")
        self.attention_policy = attention_policy
        self.local_attn_variant = local_attn_variant
        self._policy_schedule = build_policy_schedule(attention_policy, num_layers)

        # ---- mask head (4 masks stacked along H -> one Conv2d pass) ----
        # Ablation toggle (outermost): when False the mask injection in
        # forward() is skipped entirely (warp stream stays, mask conditioning
        # is removed). Set from the trainer via getattr-config.
        self.use_mask_encoder: bool = True
        self.mask_encoder = nn.Conv2d(
            in_channels=1, out_channels=in_channels, kernel_size=16, stride=16, bias=False,
        )
        # IMPORTANT: only the LAST layer of the 2-layer head is zero-init.
        # Zero-initialising BOTH mask_encoder (conv) and mask_proj (linear)
        # creates a gradient deadlock: mask_proj.weight gets no gradient
        # because its input (zero conv output) is 0, and mask_encoder.weight
        # gets no gradient because it back-props through mask_proj.weight=0.
        # Both stay 0 forever; only mask_proj.bias would train. Random-init
        # the conv so the chain learns, keep mask_proj zero so the injection
        # is still 0 at step 0 (baseline-preserving, ControlNet/LoRA pattern).
        # Use the Tensor method (.normal_), NOT torch.nn.init.* -- diffusers'
        # from_pretrained runs construction under no_init_weights() which patches
        # torch.nn.init.* to NO-OPs, leaving the encoder uninitialized and
        # deadlocking the 2-layer head (see two-view file _fan_in_normal_conv_).
        with torch.no_grad():
            self.mask_encoder.weight.normal_(mean=0.0, std=0.02)
        self.mask_proj = nn.Linear(in_channels, self.inner_dim, bias=True)
        nn.init.zeros_(self.mask_proj.weight)
        nn.init.zeros_(self.mask_proj.bias)

        # ---- Plücker head (ERP route) ----
        # Same two-layer zero-init deadlock fix as the mask head: random-init
        # the encoder conv, keep plucker_proj zero so step-0 injection is 0.
        self.inject_plucker: bool = False
        self.plucker_encoder = nn.Conv2d(
            in_channels=6, out_channels=in_channels, kernel_size=16, stride=16, bias=False,
        )
        # Tensor method (.normal_), not torch.nn.init.* (no-op'd by from_pretrained).
        with torch.no_grad():
            self.plucker_encoder.weight.normal_(mean=0.0, std=0.02)
        self.plucker_proj = nn.Linear(in_channels, self.inner_dim, bias=True)
        nn.init.zeros_(self.plucker_proj.weight)
        nn.init.zeros_(self.plucker_proj.bias)

        # ---- camera-pose heads ----
        self.inject_pose_temb: bool = False
        self.inject_pose_film: bool = False
        self.inject_dist_temb: bool = False
        # which streams pose-FiLM touches: "end_only" (noise + warp) | "all".
        self.pose_film_streams: str = "end_only"
        self.pose_temb_embed = PoseEmbed(self.inner_dim, in_dim=3)
        self.pose_film = PoseFiLM(self.inner_dim, in_dim=3)
        self.dist_temb_embed = PoseEmbed(self.inner_dim, in_dim=1)

        g_cnt = sum(1 for s in self._policy_schedule if s == "G")
        l_cnt = num_layers - g_cnt
        print("[FourViewMaskConcat-Transformer] Initialized:")
        print(f"  - num_layers={num_layers}, inner_dim={self.inner_dim}, num_views={self.num_views}")
        print(f"  - attention_policy={attention_policy} [G={g_cnt}, L={l_cnt}]")
        if l_cnt > 0:
            print(f"  - local_attn_variant={local_attn_variant}")
        print(f"  - mask/plucker/pose heads zero-initialised "
              f"(inject_plucker={self.inject_plucker}, "
              f"inject_pose_temb={self.inject_pose_temb}, "
              f"inject_pose_film={self.inject_pose_film})")

    # ------------------------------------------------------------------
    # Policy setters (verbatim semantics from the legacy four-view model).
    # ------------------------------------------------------------------
    def set_attention_policy(self, policy: str):
        if policy not in VALID_POLICIES:
            raise ValueError(f"attention_policy={policy!r} not in {sorted(VALID_POLICIES)}")
        self.attention_policy = policy
        self._policy_schedule = build_policy_schedule(policy, len(self.transformer_blocks))
        try:
            self.config.attention_policy = policy
        except Exception:
            pass
        g_cnt = sum(1 for s in self._policy_schedule if s == "G")
        l_cnt = len(self._policy_schedule) - g_cnt
        print(f"[FourViewMaskConcat] attention_policy -> {policy} [G={g_cnt}, L={l_cnt}]")

    def set_local_attn_variant(self, variant: str):
        if variant not in VALID_LOCAL_VARIANTS:
            raise ValueError(f"local_attn_variant={variant!r} not in {sorted(VALID_LOCAL_VARIANTS)}")
        self.local_attn_variant = variant
        try:
            self.config.local_attn_variant = variant
        except Exception:
            pass
        l_cnt = sum(1 for s in self._policy_schedule if s == "L")
        if l_cnt > 0:
            print(f"[FourViewMaskConcat] local_attn_variant -> {variant}")

    def set_policy_schedule(self, schedule: List[str]):
        n = len(self.transformer_blocks)
        if len(schedule) != n:
            raise ValueError(f"schedule length {len(schedule)} != num_layers {n}")
        for i, s in enumerate(schedule):
            if s not in ("G", "L"):
                raise ValueError(f"schedule[{i}]={s!r} must be 'G' or 'L'")
        self._policy_schedule = list(schedule)
        self.attention_policy = "custom"
        try:
            self.config.attention_policy = "custom"
        except Exception:
            pass
        g_cnt = sum(1 for s in self._policy_schedule if s == "G")
        print(f"[FourViewMaskConcat] schedule -> custom [G={g_cnt}, L={n - g_cnt}]")

    # ------------------------------------------------------------------
    # G <-> L reshape helpers. Public variant names are ordered by local
    # sequence length: v1=noise_only, v2=noise_warp, v3=full.
    # ------------------------------------------------------------------
    @staticmethod
    def _g_to_l_full(hidden_states, encoder_hidden_states, temb, V, L_n, L_w):
        B = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        noise = hidden_states[:, : V * L_n, :].reshape(B, V, L_n, D)
        start = hidden_states[:, V * L_n : V * L_n + L_n, :]
        warp = hidden_states[:, V * L_n + L_n :, :].reshape(B, V, L_w, D)
        start_rep = start.unsqueeze(1).expand(B, V, L_n, D)
        local = torch.cat([noise, start_rep, warp], dim=2).reshape(B * V, 2 * L_n + L_w, D)
        ehs_l = encoder_hidden_states.repeat_interleave(V, dim=0)
        temb_l = temb.repeat_interleave(V, dim=0)
        return local, ehs_l, temb_l

    @staticmethod
    def _l_to_g_full(hidden_states, encoder_hidden_states, temb, V, L_n, L_w):
        BV = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        B = BV // V
        local = hidden_states.reshape(B, V, 2 * L_n + L_w, D)
        noise = local[:, :, :L_n, :]
        start_views = local[:, :, L_n : 2 * L_n, :]
        warp = local[:, :, 2 * L_n :, :]
        start = start_views.mean(dim=1)
        g = torch.cat([noise.reshape(B, V * L_n, D), start, warp.reshape(B, V * L_w, D)], dim=1)
        ehs = encoder_hidden_states.reshape(B, V, *encoder_hidden_states.shape[1:]).mean(dim=1)
        temb_g = temb.reshape(B, V, *temb.shape[1:]).mean(dim=1) if temb.dim() > 2 else temb.reshape(B, V, -1).mean(dim=1)
        return g, ehs, temb_g

    @staticmethod
    def _g_to_l_noise_only(hidden_states, encoder_hidden_states, temb, V, L_n, L_w):
        B = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        noise = hidden_states[:, : V * L_n, :].reshape(B, V, L_n, D)
        start = hidden_states[:, V * L_n : V * L_n + L_n, :]
        warp_full = hidden_states[:, V * L_n + L_n :, :]
        local = noise.reshape(B * V, L_n, D)
        ehs_l = encoder_hidden_states.repeat_interleave(V, dim=0)
        temb_l = temb.repeat_interleave(V, dim=0)
        return local, ehs_l, temb_l, start, warp_full

    @staticmethod
    def _l_to_g_noise_only(hidden_states, encoder_hidden_states, temb, start_cached, warp_cached, V, L_n, L_w):
        BV = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        B = BV // V
        noise = hidden_states.reshape(B, V, L_n, D)
        g = torch.cat([noise.reshape(B, V * L_n, D), start_cached, warp_cached], dim=1)
        ehs = encoder_hidden_states.reshape(B, V, *encoder_hidden_states.shape[1:]).mean(dim=1)
        temb_g = temb.reshape(B, V, *temb.shape[1:]).mean(dim=1) if temb.dim() > 2 else temb.reshape(B, V, -1).mean(dim=1)
        return g, ehs, temb_g

    @staticmethod
    def _g_to_l_noise_warp(hidden_states, encoder_hidden_states, temb, V, L_n, L_w):
        B = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        noise = hidden_states[:, : V * L_n, :].reshape(B, V, L_n, D)
        start = hidden_states[:, V * L_n : V * L_n + L_n, :]
        warp = hidden_states[:, V * L_n + L_n :, :].reshape(B, V, L_w, D)
        local = torch.cat([noise, warp], dim=2).reshape(B * V, L_n + L_w, D)
        ehs_l = encoder_hidden_states.repeat_interleave(V, dim=0)
        temb_l = temb.repeat_interleave(V, dim=0)
        return local, ehs_l, temb_l, start

    @staticmethod
    def _l_to_g_noise_warp(hidden_states, encoder_hidden_states, temb, start_cached, V, L_n, L_w):
        BV = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        B = BV // V
        local = hidden_states.reshape(B, V, L_n + L_w, D)
        noise = local[:, :, :L_n, :]
        warp = local[:, :, L_n:, :]
        g = torch.cat([noise.reshape(B, V * L_n, D), start_cached, warp.reshape(B, V * L_w, D)], dim=1)
        ehs = encoder_hidden_states.reshape(B, V, *encoder_hidden_states.shape[1:]).mean(dim=1)
        temb_g = temb.reshape(B, V, *temb.shape[1:]).mean(dim=1) if temb.dim() > 2 else temb.reshape(B, V, -1).mean(dim=1)
        return g, ehs, temb_g

    # ------------------------------------------------------------------
    # rope helper using this repo's pos_embed signature.
    # ------------------------------------------------------------------
    def _rope(self, img_shapes, txt_seq_lens, encoder_hidden_states, device):
        if txt_seq_lens is None:
            max_txt = encoder_hidden_states.shape[1]
        elif isinstance(txt_seq_lens, (list, tuple)):
            max_txt = int(max(txt_seq_lens))
        elif torch.is_tensor(txt_seq_lens):
            max_txt = int(txt_seq_lens.max().item()) if txt_seq_lens.numel() > 1 else int(txt_seq_lens.item())
        else:
            max_txt = int(txt_seq_lens)
        return self.pos_embed(img_shapes, device=device, max_txt_seq_len=max_txt)

    def _patchify_cond(self, p: torch.Tensor) -> torch.Tensor:
        """ERP (B, 6, H, W) -> (B, L, inner_dim) via plucker_encoder + proj."""
        p = torch.nan_to_num(
            p.to(dtype=self.plucker_encoder.weight.dtype), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(-1e4, 1e4)
        feat = self.plucker_encoder(p)                  # (B, 64, H/16, W/16)
        tok = feat.flatten(2).transpose(1, 2)           # (B, L, 64)
        return torch.nan_to_num(self.plucker_proj(tok))  # (B, L, inner_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        packed_noisy_model_input: torch.Tensor,     # (B, V*L_n, 64)
        packed_model_condition_input: torch.Tensor,  # (B, L_n, 64) shared start
        packed_model_wrap_image: torch.Tensor,      # (B, V*L_w, 64)
        wrap_mask: torch.Tensor,                     # (B, 1, V*H_m, W_m)
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        plucker_start: Optional[torch.Tensor] = None,  # (B, 6, H, W) shared start
        plucker_end: Optional[torch.Tensor] = None,    # (B, V, 6, H, W)
        pose_vec: Optional[torch.Tensor] = None,       # (B, V, 3)
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `joint_attention_kwargs` is ineffective without PEFT.")

        V = self.num_views
        L_n = packed_noisy_model_input.shape[1] // V
        L_w = packed_model_wrap_image.shape[1] // V
        assert packed_noisy_model_input.shape[1] == V * L_n
        assert packed_model_wrap_image.shape[1] == V * L_w
        s_warp = packed_model_wrap_image.shape[1]
        s_start = packed_model_condition_input.shape[1]
        assert s_start == L_n, f"start must equal one noise view ({L_n}) tokens, got {s_start}"

        # ---------- Build G-format hidden states ----------
        hidden_states = torch.cat(
            [packed_noisy_model_input, packed_model_condition_input, packed_model_wrap_image], dim=1
        )
        hidden_states = self.img_in(hidden_states)

        # ---------- Mask injection (warp slice) ----------
        # Ablation toggle: skip entirely when use_mask_encoder is False (warp
        # stream stays, no mask conditioning) or when no mask is provided.
        if self.use_mask_encoder and (wrap_mask is not None):
            mask_feat = self.mask_encoder(wrap_mask)                 # (B, 64, V*H_m/16, W_m/16)
            mask_token = mask_feat.flatten(2).transpose(1, 2)        # (B, V*L_w, 64)
            mask_token_proj = self.mask_proj(mask_token)
            assert mask_token_proj.shape[1] == s_warp, (
                f"mask tokens ({mask_token_proj.shape[1]}) must match warp tokens ({s_warp}). "
                f"Check wrap_mask H == V * warp_latent_H*16."
            )
            hidden_states[:, -s_warp:, :] = hidden_states[:, -s_warp:, :] + mask_token_proj

        # ---------- Plücker injection (per view; G-format slices) ----------
        # G layout: [noise_v0..v(V-1) | start | warp_v0..v(V-1)].
        #   plucker_end[:, i]  -> noise_vi AND warp_vi
        #   plucker_start      -> start
        if self.inject_plucker and (plucker_start is not None) and (plucker_end is not None):
            warp_base = V * L_n + L_n
            for i in range(V):
                tok_end_i = self._patchify_cond(plucker_end[:, i])  # (B, L, inner)
                # noise view i
                hidden_states[:, i * L_n:(i + 1) * L_n, :] = (
                    hidden_states[:, i * L_n:(i + 1) * L_n, :] + tok_end_i
                )
                # warp view i
                ws = warp_base + i * L_w
                hidden_states[:, ws:ws + L_w, :] = hidden_states[:, ws:ws + L_w, :] + tok_end_i
            tok_start = self._patchify_cond(plucker_start)
            hidden_states[:, V * L_n:V * L_n + L_n, :] = (
                hidden_states[:, V * L_n:V * L_n + L_n, :] + tok_start
            )

        # ---------- pose FiLM (Option B) ----------
        # Per-view multiplicative modulation; built out-of-place (a *multiply*
        # in-place would break autograd, see single-view recipe). "all" uses
        # the view-averaged pose on the whole sequence; "end_only" modulates
        # each noise_vi + warp_vi with its own pose and leaves start untouched.
        if self.inject_pose_film and (pose_vec is not None):
            if self.pose_film_streams == "all":
                ps_mean = pose_vec.mean(dim=1)  # (B, 3)
                fs, fsh = self.pose_film(ps_mean)
                fs = fs.unsqueeze(1).to(hidden_states.dtype)
                fsh = fsh.unsqueeze(1).to(hidden_states.dtype)
                hidden_states = hidden_states * (1 + fs) + fsh
            else:  # end_only, per view
                warp_base = V * L_n + L_n
                noise_parts = []
                warp_parts = []
                for i in range(V):
                    fs, fsh = self.pose_film(pose_vec[:, i])
                    fs = fs.unsqueeze(1).to(hidden_states.dtype)
                    fsh = fsh.unsqueeze(1).to(hidden_states.dtype)
                    n_i = hidden_states[:, i * L_n:(i + 1) * L_n, :] * (1 + fs) + fsh
                    ws = warp_base + i * L_w
                    w_i = hidden_states[:, ws:ws + L_w, :] * (1 + fs) + fsh
                    noise_parts.append(n_i)
                    warp_parts.append(w_i)
                start_part = hidden_states[:, V * L_n:V * L_n + L_n, :]
                hidden_states = torch.cat(noise_parts + [start_part] + warp_parts, dim=1)

        # ---------- text / time / rope ----------
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        # pose temb (Option A) + dist temb (Option C): per-view pose averaged
        # because temb is a single per-batch global AdaLN vector.
        if self.inject_pose_temb and (pose_vec is not None):
            temb = temb + torch.tanh(self.pose_temb_embed(pose_vec.mean(dim=1)).to(temb.dtype))
        if self.inject_dist_temb and (pose_vec is not None):
            dist = pose_vec.mean(dim=1).norm(dim=-1, keepdim=True)  # (B, 1)
            temb = temb + torch.tanh(self.dist_temb_embed(dist).to(temb.dtype))

        image_rotary_emb_global = self._rope(
            img_shapes, txt_seq_lens, encoder_hidden_states, hidden_states.device
        )

        # ---------- local rope shapes ----------
        noise_face_shape = img_shapes[0][0]
        start_face_shape = img_shapes[0][V]
        warp_face_shape = img_shapes[0][V + 1]
        if self.local_attn_variant == "v1":
            local_img_shapes = [[noise_face_shape]] * (packed_noisy_model_input.shape[0] * V)
        elif self.local_attn_variant == "v2":
            local_img_shapes = [[noise_face_shape, warp_face_shape]] * (
                packed_noisy_model_input.shape[0] * V
            )
        else:
            local_img_shapes = [[noise_face_shape, start_face_shape, warp_face_shape]] * (
                packed_noisy_model_input.shape[0] * V
            )
        local_txt_seq_lens = None
        if txt_seq_lens is not None:
            local_txt_seq_lens = [l for l in txt_seq_lens for _ in range(V)]
        image_rotary_emb_local = None

        # ---------- block loop (G<->L reshape) ----------
        fmt = "G"
        ehs = encoder_hidden_states
        ehs_mask = encoder_hidden_states_mask
        temb_cur = temb
        image_rotary_emb = image_rotary_emb_global
        start_cached: Optional[torch.Tensor] = None
        warp_cached: Optional[torch.Tensor] = None

        for index_block, block in enumerate(self.transformer_blocks):
            want = self._policy_schedule[index_block]
            if want == "G" and fmt == "L":
                if self.local_attn_variant == "v1":
                    hidden_states, ehs, temb_cur = self._l_to_g_noise_only(
                        hidden_states, ehs, temb_cur, start_cached, warp_cached, V, L_n, L_w)
                    start_cached = warp_cached = None
                elif self.local_attn_variant == "v2":
                    hidden_states, ehs, temb_cur = self._l_to_g_noise_warp(
                        hidden_states, ehs, temb_cur, start_cached, V, L_n, L_w)
                    start_cached = None
                else:
                    hidden_states, ehs, temb_cur = self._l_to_g_full(hidden_states, ehs, temb_cur, V, L_n, L_w)
                image_rotary_emb = image_rotary_emb_global
                ehs_mask = encoder_hidden_states_mask
                fmt = "G"
            elif want == "L" and fmt == "G":
                if self.local_attn_variant == "v1":
                    hidden_states, ehs_l, temb_l, start_cached, warp_cached = self._g_to_l_noise_only(
                        hidden_states, ehs, temb_cur, V, L_n, L_w)
                elif self.local_attn_variant == "v2":
                    hidden_states, ehs_l, temb_l, start_cached = self._g_to_l_noise_warp(
                        hidden_states, ehs, temb_cur, V, L_n, L_w)
                else:
                    hidden_states, ehs_l, temb_l = self._g_to_l_full(hidden_states, ehs, temb_cur, V, L_n, L_w)
                ehs = ehs_l
                temb_cur = temb_l
                if image_rotary_emb_local is None:
                    image_rotary_emb_local = self._rope(
                        local_img_shapes, local_txt_seq_lens, ehs, hidden_states.device
                    )
                image_rotary_emb = image_rotary_emb_local
                ehs_mask = (
                    encoder_hidden_states_mask.repeat_interleave(V, dim=0)
                    if encoder_hidden_states_mask is not None else None
                )
                fmt = "L"

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ehs, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, ehs, ehs_mask, temb_cur, image_rotary_emb,
                )
            else:
                ehs, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=ehs,
                    encoder_hidden_states_mask=ehs_mask,
                    temb=temb_cur,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

            if controlnet_block_samples is not None:
                import numpy as np
                interval = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval]

        # restore G if ended in L
        if fmt == "L":
            if self.local_attn_variant == "v1":
                hidden_states, ehs, temb_cur = self._l_to_g_noise_only(
                    hidden_states, ehs, temb_cur, start_cached, warp_cached, V, L_n, L_w)
            elif self.local_attn_variant == "v2":
                hidden_states, ehs, temb_cur = self._l_to_g_noise_warp(
                    hidden_states, ehs, temb_cur, start_cached, V, L_n, L_w)
            else:
                hidden_states, ehs, temb_cur = self._l_to_g_full(hidden_states, ehs, temb_cur, V, L_n, L_w)
            fmt = "G"

        # ---------- final norm / projection ----------
        # Use the (G-format) global temb for the final AdaLN, matching the
        # single-view recipe (norm_out consumes the per-batch temb).
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
