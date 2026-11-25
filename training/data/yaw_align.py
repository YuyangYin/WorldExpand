"""ERP yaw-alignment primitives for the online warp pipeline.

Rationale
=========
After ``OnlineWarpDataset._recenter`` we have ``start_yaw=0`` and ``end_yaw``
expressed in start's local UE5 frame. Two cameras at non-zero relative yaw
look at the same world point through *different* ERP columns, which hurts the
diffusion model's ability to learn forward motion as a smooth continuation
of the start view. This module rotates the ERP image(s) around the vertical
axis so the panorama centers face a canonical direction, *before* the
existing warp + circular-pad stages.

The transformation is a pure ``torch.roll`` along the width axis: ERP wraps
360 degrees around Z, so a horizontal pixel shift is mathematically
equivalent to a yaw rotation of the camera. The pixel shift needed to make
an image originally captured at yaw ``theta_orig`` *appear* as if it had
been captured at yaw ``theta_target`` is

    shift = round( (theta_orig - theta_target) / 360 * W )
    new = torch.roll(old, shifts=shift, dims=-1)

Sign convention derivation
--------------------------
Tracing the existing renderer's ERP convention
(:func:`panorama_rendering.render_utils.panorama_utils.spherical_uv_to_directions`
-> ``_PLY_ROT`` -> ``_ROT_MATRIX_4x4`` inside
``training/data/online_warp_renderer.py``):

* ``u=W/2`` maps to UE5 camera-local ``(+1, 0, 0)`` = "forward".
* ``u=W/4`` maps to UE5 camera-local ``(0, -1, 0)`` = "left".
* ``u=3W/4`` maps to UE5 camera-local ``(0, +1, 0)`` = "right".

So at camera yaw ``theta``, column ``u`` corresponds to world azimuth
``theta + (u/W - 0.5) * 360 deg``. Setting the rolled image's world azimuth
equal to a target-yaw image's world azimuth yields the formula above.

Two modes
=========
* :func:`align_end_to_start` -- roll the **end** image only so its ERP
  center looks the same world direction as start's center. Default for
  training: cheap, only one ``torch.roll`` per sample, and keeps the
  start-view PCD untouched.
* :func:`align_to_displacement` -- compute the displacement direction
  ``atan2(end_pos.Y, end_pos.X)`` in start's local UE5 frame and roll
  **both** start and end (plus depth / sky mask / cond_img_prompt) so
  every ERP center points at the motion direction.

Both functions return a new sample dict (they do not mutate inputs in
place beyond shallow alias-safety) and update ``start_yaw`` / ``end_yaw``
in lockstep with the rolled tensors so the downstream
:func:`OnlineWarpRenderer._build_w2c_render_from_ue5` produces a
geometrically consistent w2c matrix.

Tensor handling
===============
* Width axis is always the LAST dim. Tensors handled:
    start_rgb_train      (3, H, W)
    end_rgb_train        (3, H, W)
    cond_img_prompt      (3, h, w)
    start_rgb_for_pcd    (H, W, 3)   # last spatial dim is W (still dim=-2)
    start_depth_for_pcd  (H, W)
    start_sky_mask_for_pcd  (H, W) bool   (may be None)
  For HWC layouts (``start_rgb_for_pcd``) we roll along dim=-2 (= W) so
  the colour-channels last dim is left alone.
* Integer-pixel roll only. At W=1920 one pixel is ~0.19 deg, well below
  every other source of angular jitter in the dataset; interpolation
  would smear depth at object boundaries.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def displacement_yaw_deg(end_position: Dict[str, float]) -> float:
    """Compute the displacement direction (UE5 yaw, degrees) in start's frame.

    ``end_position`` is expected to come out of
    :meth:`OnlineWarpDataset._recenter`, i.e. it is the camera delta
    expressed in start's *local* UE5 frame (where start sits at the origin
    with yaw=0). UE5 yaw is measured CCW around +Z from +X, so the motion
    azimuth is simply ``atan2(Y, X)``.

    When the horizontal displacement is degenerate (|XY| < 1e-6), we fall
    back to 0.0 (i.e. "no rotation"); this matches the recenter convention
    of leaving start_yaw at 0.
    """
    dx = float(end_position["X"])
    dy = float(end_position["Y"])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def yaw_to_pixel_shift(theta_orig_deg: float, theta_target_deg: float, width: int) -> int:
    """Return the integer ``torch.roll`` shift that maps yaw_orig -> yaw_target."""
    delta = (float(theta_orig_deg) - float(theta_target_deg)) / 360.0 * int(width)
    return int(round(delta))


def erp_roll_width(x: torch.Tensor, shift_pixels: int, width_dim: int = -1) -> torch.Tensor:
    """Circularly roll an ERP tensor along its width axis.

    Parameters
    ----------
    x : torch.Tensor
        ERP tensor of any rank. Width is the last spatial dim, indexed by
        ``width_dim`` (default -1 for CHW / (C,H,W) / (H,W) tensors;
        callers using HWC pass ``width_dim=-2``).
    shift_pixels : int
        Number of pixels to roll right (``new[u] = old[(u - shift) % W]``).
        A no-op when ``shift_pixels % W == 0``.
    """
    if x is None:
        return x
    w = int(x.shape[width_dim])
    s = int(shift_pixels) % w
    if s == 0:
        return x
    return torch.roll(x, shifts=s, dims=width_dim)


# ---------------------------------------------------------------------------
# Helpers for shallow-safe mutation
# ---------------------------------------------------------------------------
def _shallow_copy_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level copy; nested ``start_position`` / ``end_position`` get deep
    copied because we may rewrite their values later if we extend the
    contract. Tensors are NOT copied (we replace whole entries)."""
    out = dict(sample)
    if isinstance(sample.get("start_position"), dict):
        out["start_position"] = dict(sample["start_position"])
    if isinstance(sample.get("end_position"), dict):
        out["end_position"] = dict(sample["end_position"])
    return out


def _roll_chw(t: Optional[torch.Tensor], theta_orig: float, theta_target: float) -> Optional[torch.Tensor]:
    if t is None:
        return None
    w = int(t.shape[-1])
    s = yaw_to_pixel_shift(theta_orig, theta_target, w)
    return erp_roll_width(t, s, width_dim=-1)


def _roll_hwc(t: Optional[torch.Tensor], theta_orig: float, theta_target: float) -> Optional[torch.Tensor]:
    """Roll a (H, W, C) tensor along W (= dim -2)."""
    if t is None:
        return None
    w = int(t.shape[-2])
    s = yaw_to_pixel_shift(theta_orig, theta_target, w)
    return erp_roll_width(t, s, width_dim=-2)


def _roll_hw(t: Optional[torch.Tensor], theta_orig: float, theta_target: float) -> Optional[torch.Tensor]:
    """Roll a (H, W) tensor along W (= dim -1)."""
    if t is None:
        return None
    w = int(t.shape[-1])
    s = yaw_to_pixel_shift(theta_orig, theta_target, w)
    return erp_roll_width(t, s, width_dim=-1)


# ---------------------------------------------------------------------------
# Mode A: align end -> start
# ---------------------------------------------------------------------------
def align_end_to_start(sample: Dict[str, Any]) -> Dict[str, Any]:
    """ERP-roll the END image so its center faces the same world direction
    as the start center, then zero out the relative yaw.

    Only ``end_rgb_train`` is touched (the dataset doesn't expose an
    end-side depth / mask). All start tensors and ``cond_img_prompt`` are
    left untouched.

    Assumptions
    -----------
    * ``sample`` already went through ``OnlineWarpDataset._recenter``, so
      ``start_yaw == 0`` and ``end_yaw`` is the start-frame-relative angle
      in degrees, wrapped to ``[-180, 180]``.
    * ``end_rgb_train`` is CHW with ERP width along dim=-1.
    """
    out = _shallow_copy_sample(sample)

    theta_start = float(out.get("start_yaw", 0.0))
    theta_end = float(out.get("end_yaw", 0.0))

    out["end_rgb_train"] = _roll_chw(out.get("end_rgb_train"), theta_orig=theta_end, theta_target=theta_start)
    out["end_yaw"] = theta_start

    out["_yaw_align_meta"] = {
        "mode": "end_to_start",
        "theta_start_before": theta_start,
        "theta_end_before": theta_end,
        "theta_start_after": theta_start,
        "theta_end_after": theta_start,
        "shift_end_train_px": yaw_to_pixel_shift(
            theta_end, theta_start, int(sample["end_rgb_train"].shape[-1])
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Mode B: align both -> displacement direction
# ---------------------------------------------------------------------------
def align_to_displacement(sample: Dict[str, Any]) -> Dict[str, Any]:
    """ERP-roll BOTH start and end so every ERP center faces the
    displacement direction in start's local UE5 frame.

    All width-bearing tensors in ``sample`` are rolled by the same angular
    amount (each at its own width). Positions are unchanged (translation
    is independent of camera rotation). Yaws are rewritten to the
    displacement direction so the renderer's ``_build_w2c_render_from_ue5``
    produces a geometrically consistent w2c for both PCD-build and
    end-view rendering.
    """
    out = _shallow_copy_sample(sample)

    theta_start = float(out.get("start_yaw", 0.0))
    theta_end = float(out.get("end_yaw", 0.0))
    theta_disp = displacement_yaw_deg(out["end_position"])

    # --- START side --------------------------------------------------------
    out["start_rgb_train"] = _roll_chw(out.get("start_rgb_train"), theta_orig=theta_start, theta_target=theta_disp)
    out["start_rgb_for_pcd"] = _roll_hwc(out.get("start_rgb_for_pcd"), theta_orig=theta_start, theta_target=theta_disp)
    out["start_depth_for_pcd"] = _roll_hw(out.get("start_depth_for_pcd"), theta_orig=theta_start, theta_target=theta_disp)
    out["start_sky_mask_for_pcd"] = _roll_hw(
        out.get("start_sky_mask_for_pcd"), theta_orig=theta_start, theta_target=theta_disp
    )
    # cond_img_prompt is sampled from the start image, so it must travel
    # with the start roll. It is CHW (3, h, w).
    out["cond_img_prompt"] = _roll_chw(out.get("cond_img_prompt"), theta_orig=theta_start, theta_target=theta_disp)

    # --- END side ----------------------------------------------------------
    out["end_rgb_train"] = _roll_chw(out.get("end_rgb_train"), theta_orig=theta_end, theta_target=theta_disp)

    # --- yaw bookkeeping ---------------------------------------------------
    out["start_yaw"] = theta_disp
    out["end_yaw"] = theta_disp

    meta: Dict[str, Any] = {
        "mode": "to_displacement",
        "theta_start_before": theta_start,
        "theta_end_before": theta_end,
        "theta_disp": theta_disp,
        "theta_start_after": theta_disp,
        "theta_end_after": theta_disp,
    }
    if "start_rgb_train" in sample:
        meta["shift_start_train_px"] = yaw_to_pixel_shift(
            theta_start, theta_disp, int(sample["start_rgb_train"].shape[-1])
        )
    if "start_rgb_for_pcd" in sample:
        meta["shift_start_pcd_px"] = yaw_to_pixel_shift(
            theta_start, theta_disp, int(sample["start_rgb_for_pcd"].shape[-2])
        )
    if "cond_img_prompt" in sample:
        meta["shift_cond_px"] = yaw_to_pixel_shift(
            theta_start, theta_disp, int(sample["cond_img_prompt"].shape[-1])
        )
    if "end_rgb_train" in sample:
        meta["shift_end_train_px"] = yaw_to_pixel_shift(
            theta_end, theta_disp, int(sample["end_rgb_train"].shape[-1])
        )
    out["_yaw_align_meta"] = meta
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
VALID_MODES = ("off", "end_to_start", "to_displacement")


def apply_yaw_align(sample: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """One-stop dispatcher used by the dataset's __getitem__."""
    mode = str(mode)
    if mode == "off":
        return sample
    if mode == "end_to_start":
        return align_end_to_start(sample)
    if mode == "to_displacement":
        return align_to_displacement(sample)
    raise ValueError(
        f"yaw_align mode must be one of {VALID_MODES}, got {mode!r}"
    )


__all__ = [
    "VALID_MODES",
    "align_end_to_start",
    "align_to_displacement",
    "apply_yaw_align",
    "displacement_yaw_deg",
    "erp_roll_width",
    "yaw_to_pixel_shift",
]
