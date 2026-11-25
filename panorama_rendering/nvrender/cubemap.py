"""Cubemap constants and ERP panorama stitching functions."""

import os

import numpy as np
import torch

from .utils import image_uv_torch, spherical_uv_to_directions_torch, project_cv_torch


# ============================================================================
# Cubemap direction matrices (shared across all renderers)
# ============================================================================

# 6-face cubemap rotation matrices (camera directions)
CUBEMAP_ROTATIONS = np.array([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
], dtype=np.float32)

# Glue matrices for stitching cubemap faces into ERP panorama
_mat_glue_raw = np.array([
    [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
], dtype=np.float32)
CUBEMAP_GLUE = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]]) @ _mat_glue_raw

# Pre-built 4x4 glue extrinsics
CUBEMAP_GLUE_EXTRINSICS = np.eye(4, dtype=np.float32)[None].repeat(6, 0)
CUBEMAP_GLUE_EXTRINSICS[:, :3, :3] = CUBEMAP_GLUE

# Cubemap face resolution & intrinsics (90-degree FOV; cx=cy=size/2, fx=fy=size/2).
#
# Resolution is driven by the CUBEMAP_SIZE env var so a single knob propagates
# to every consumer (gs_renderer / pcd_renderer / mesh_renderer all import the
# two module-level constants below). The launcher
# (scripts/run_train_online_h20.sh) resolves CUBEMAP_SIZE=0 to an auto value
# derived from IMG_SIZE before invoking accelerate, so by the time this file
# imports, the env should always hold a positive integer.
#
# This module enforces a fallback (-> 512) for three failure modes so direct
# imports outside the launcher (notebooks, ad-hoc scripts) Just Work:
#   1. env unset
#   2. env empty / non-integer / "0"
#   3. env < a minimal safe value (256)
#
# Why 512 floor? Cube->ERP grid_sample samples at 1:1 when CUBE >= ERP_W / 4 =
# IMG_SIZE / 2. ERP <= 2048 wide (=> IMG_SIZE <= 1024) is the sweet spot
# bracket where cube=512 still oversamples or matches 1:1. Bumping to 1024
# costs ~4x gsplat VRAM/time -- pair with proportionally larger PCD_BUILD_H/W
# (target ~3-4x cube_pixels in candidate points) so splat coverage scales.
_env_cube_raw = os.environ.get("CUBEMAP_SIZE", "").strip()
try:
    _env_cube = int(_env_cube_raw)
except (ValueError, TypeError):
    _env_cube = 0
if _env_cube < 256:
    if _env_cube_raw not in ("", "512"):
        # Loud so notebook users notice the fallback. Silent for the common
        # "env unset" case and the explicit "512" (which matches the floor).
        print(
            f"[cubemap] CUBEMAP_SIZE env={_env_cube_raw!r} -> falling back "
            f"to 512. Set CUBEMAP_SIZE>=256 to override, or rely on the "
            f"launcher's IMG_SIZE-based auto-decide."
        )
    _env_cube = 512
CUBEMAP_SIZE = _env_cube
_half = float(CUBEMAP_SIZE) / 2.0
CUBEMAP_INTRINSICS = np.array([
    [_half, 0.0, _half],
    [0.0, _half, _half],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def get_cubemap_render_Rts(frame_Rt, mat=None):
    """
    Build 6 cubemap w2c matrices for a given frame's w2c matrix.

    Args:
        frame_Rt: [4, 4] w2c matrix (numpy array)
        mat: [6, 3, 3] cubemap rotation matrices (default: CUBEMAP_ROTATIONS)

    Returns:
        panorama_render_Rts: [6, 4, 4] cubemap w2c matrices
    """
    if mat is None:
        mat = CUBEMAP_ROTATIONS
    cur_Rt = frame_Rt[None]  # [1, 4, 4]
    panorama_render_Rts = np.tile(cur_Rt, (6, 1, 1))  # [6, 4, 4]
    panorama_render_Rts[:, :3, :] = mat @ panorama_render_Rts[:, :3, :]
    return panorama_render_Rts


def merge_panorama_color_torch(width, height, distance_maps, pred_masks,
                                extrinsics, intrinsics_, device,
                                erp_mask_aware=False):
    """
    Stitch cubemap faces into an ERP panorama.

    When erp_mask_aware=True, use mask-aware weighted sampling to avoid
    black-hole artefacts caused by invalid (inf) pixels bleeding into
    bilinear interpolation.

    Args:
        width: panorama output width
        height: panorama output height
        distance_maps: [N, H_cube, W_cube, C] tensor of cubemap face data
        pred_masks: unused (kept for API compatibility)
        extrinsics: [N, 4, 4] glue extrinsics
        intrinsics_: [N, 3, 3] cubemap intrinsics
        device: torch device
        erp_mask_aware: whether to use mask-aware weighted sampling

    Returns:
        panorama: [H, W, C] stitched panorama tensor
    """
    N = len(intrinsics_)
    intrinsics = intrinsics_.clone()
    H, W = distance_maps[0].shape[:2]
    intrinsics[:, 0, :] /= W
    intrinsics[:, 1, :] /= H

    uv = image_uv_torch(width=width, height=height, device=device, dtype=torch.float32)
    spherical_directions = spherical_uv_to_directions_torch(uv)

    panorama_log_distance_map = None

    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv_torch(
            spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i]
        )
        projection_valid_mask = (
            (projected_depth >= 1e-5) &
            (projected_uv >= 0).all(dim=-1) &
            (projected_uv <= 1).all(dim=-1)
        )
        projected_pixels = torch.clip((projected_uv * 2. - 1.), -1., 1.)

        if erp_mask_aware:
            # --- Mask-aware weighted sampling ---
            face_data = distance_maps[i]  # [H_cube, W_cube, C]
            valid_mask_2d = torch.isfinite(face_data).all(dim=-1).float()

            face_clean = face_data.clone()
            face_clean[~torch.isfinite(face_clean)] = 0.0

            grid = projected_pixels[None, :, :, :2]

            sampled_color = torch.nn.functional.grid_sample(
                face_clean[None].permute(0, 3, 1, 2),
                grid, align_corners=True, mode='bilinear', padding_mode='zeros'
            ).squeeze().permute(1, 2, 0)

            sampled_weight = torch.nn.functional.grid_sample(
                valid_mask_2d[None, None],
                grid, align_corners=True, mode='bilinear', padding_mode='zeros'
            ).squeeze()

            weight_positive = sampled_weight > 1e-6
            sampled_weight_expanded = sampled_weight.unsqueeze(-1)
            normalised = torch.where(
                weight_positive.unsqueeze(-1),
                sampled_color / (sampled_weight_expanded + 1e-8),
                torch.zeros_like(sampled_color)
            )
            normalised[~weight_positive] = torch.inf

            panorama_log_distance_map_ = torch.where(
                projection_valid_mask.unsqueeze(-1), normalised, torch.zeros_like(normalised)
            )
        else:
            # --- Original bilinear sampling ---
            panorama_log_distance_map_ = torch.where(
                projection_valid_mask,
                torch.nn.functional.grid_sample(
                    distance_maps[i][None].permute(0, 3, 1, 2),
                    projected_pixels[None, :, :, :2],
                    align_corners=True
                ),
                0
            ).squeeze().permute(1, 2, 0)

        if panorama_log_distance_map is None:
            panorama_log_distance_map = panorama_log_distance_map_
        else:
            panorama_log_distance_map[projection_valid_mask] = panorama_log_distance_map_[projection_valid_mask]

    return panorama_log_distance_map
