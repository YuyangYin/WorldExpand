"""3D Gaussian Splatting panorama rendering (gsplat-backed).

This is a third backend alongside :mod:`pcd_renderer` (pytorch3d /
TorchPointCloudRenderer). It consumes the **same** ``global_pcd.ply`` that the
existing pipeline produces, treats each point as an isotropic spherical
Gaussian, and renders the panorama via the standard cubemap-fan-out + ERP
stitching approach already used by the PCD path.

Spherical, isotropic GS math
----------------------------
For a sphere of radius sigma at z-depth d under a pinhole camera with focal
length f (pixels), the projected radius is ``sigma * f / d``. Setting the
projected radius to 0.5 px gives ``sigma = d / (2 * f) * scale_factor``.

We use the **cubemap face intrinsics** as ``f`` (90 deg FOV, default 512 px,
so ``f = 256``) so the same Gaussian projects to ~1 px on every face. ``d`` is
the distance from the point to the *first-frame* camera origin; spherical
Gaussians are rotation-invariant so this is correct in world space too.

Why first-frame? The pipeline already aligns every point cloud to the
first-frame camera (see :func:`generate_pc_render` in :mod:`camera_motion`),
so distances measured against the first frame match the visual scale that
``--render_engine pytorch3d`` /``point_renderer`` use under the hood.

Public API
----------
* :func:`build_isotropic_gs_from_pcd` — convert (points, colors) -> dict of
  gsplat-friendly tensors (means / quats / scales / opacities / colors).
* :func:`gs_pano_render_color` — render a list of panorama frames; signature
  matches :func:`pcd_pano_render_color` so the camera_motion dispatch only
  needs to swap the function call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .cubemap import (
    CUBEMAP_ROTATIONS, CUBEMAP_GLUE_EXTRINSICS,
    CUBEMAP_INTRINSICS, CUBEMAP_SIZE,
    merge_panorama_color_torch,
)


# ---------------------------------------------------------------------------
# gsplat is an optional dependency. We import lazily so that the rest of the
# package (pytorch3d / point_renderer engines) keeps working in environments
# without gsplat installed.
# ---------------------------------------------------------------------------
def _import_rasterization():
    try:
        from gsplat.rendering import rasterization  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - only triggered without gsplat
        raise ImportError(
            "render_engine='gsplat' requires the `gsplat` package. "
            "Install it (e.g. `pip install gsplat`) or pick a different "
            "--render_engine."
        ) from exc
    return rasterization


# ---------------------------------------------------------------------------
# Build isotropic spherical Gaussians from a (points, colors) point cloud.
# ---------------------------------------------------------------------------
def build_isotropic_gs_from_pcd(
    points: Union[torch.Tensor, np.ndarray],
    colors: Union[torch.Tensor, np.ndarray],
    first_frame_Rt: Union[torch.Tensor, np.ndarray],
    f_iso: float = float(CUBEMAP_INTRINSICS[0, 0]),
    scale_factor: float = 1.0,
    device: Optional[torch.device] = None,
    depth_min: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Convert a point cloud into isotropic spherical Gaussians.

    Parameters
    ----------
    points : (N, 3) tensor / array, float32
        World-space point positions (already rotated/translated to live in the
        same frame the rest of the pipeline expects -- camera_motion does this
        before calling us).
    colors : (N, 3) tensor / array, float32
        Point colors in the **[-1, 1]** convention used by the PCD path. We
        convert to [0, 1] internally because gsplat expects [0, 1] colors.
    first_frame_Rt : (4, 4) tensor / array
        World-to-camera matrix of the first frame. Used to compute per-point
        distance to camera, which sets the Gaussian sigma. Spherical
        Gaussians are rotation-invariant so the choice of frame only affects
        the magnitude of d, not the rendered geometry.
    f_iso : float
        Isotropic focal length (in pixels) used to size each Gaussian. The
        default matches the cubemap face intrinsics (256 px @ 512x512, 90 deg
        FOV) so each splat covers ~1 cube pixel.
    scale_factor : float
        Multiplier on the analytical sigma. 1.0 covers exactly one pixel;
        slightly larger (e.g. 1.2) reduces gaps between neighbouring splats,
        smaller (e.g. 0.65, mirroring the panorama path in World3C) makes
        splats tighter at the cost of more holes.
    device : torch.device, optional
        Output device. Defaults to ``points`` device when it's a tensor,
        else CUDA if available, else CPU.
    depth_min : float
        Drop points with distance < ``depth_min`` to camera (numerically
        unstable / degenerate Gaussians).

    Returns
    -------
    dict with float32 tensors, all on ``device``:
        means     : (M, 3)  positions (world)
        quats     : (M, 4)  unit quaternion (1, 0, 0, 0) -- isotropic
        scales    : (M, 3)  sigma broadcast on all axes
        opacities : (M,)    all ones (gsplat 1-D opacity convention)
        colors    : (M, 3)  in [0, 1]
    """
    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(points, dtype=torch.float32)
    if not isinstance(colors, torch.Tensor):
        colors = torch.as_tensor(colors, dtype=torch.float32)
    if not isinstance(first_frame_Rt, torch.Tensor):
        first_frame_Rt = torch.as_tensor(first_frame_Rt, dtype=torch.float32)

    if device is None:
        device = points.device if points.is_cuda else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    points = points.to(device=device, dtype=torch.float32)
    colors = colors.to(device=device, dtype=torch.float32)
    first_frame_Rt = first_frame_Rt.to(device=device, dtype=torch.float32)

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be (N, 3); got {tuple(points.shape)}")
    if colors.shape != points.shape:
        raise ValueError(f"colors {tuple(colors.shape)} must match points {tuple(points.shape)}")
    if first_frame_Rt.shape != (4, 4):
        raise ValueError(f"first_frame_Rt must be (4, 4); got {tuple(first_frame_Rt.shape)}")

    # Distance from each point to the first-frame camera origin (in the same
    # world frame the rest of camera_motion uses).
    R = first_frame_Rt[:3, :3]
    t = first_frame_Rt[:3, 3]
    pts_cam = points @ R.T + t                   # (N, 3) in cam space
    d = torch.linalg.norm(pts_cam, dim=-1)       # (N,)

    valid = (d > depth_min) & torch.isfinite(d) & torch.isfinite(points).all(dim=-1)
    points = points[valid]
    colors = colors[valid]
    d = d[valid]

    # sigma = d / (2 * f) * scale_factor
    sigma = (d / (2.0 * float(f_iso))) * float(scale_factor)
    scales = sigma.unsqueeze(-1).expand(-1, 3).contiguous()

    quats = torch.zeros((points.shape[0], 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0  # wxyz = (1, 0, 0, 0); unit identity rotation

    opacities = torch.ones((points.shape[0],), device=device, dtype=torch.float32)

    # gsplat expects colors in [0, 1].
    colors_01 = torch.clamp((colors + 1.0) * 0.5, 0.0, 1.0)

    return {
        "means": points,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors_01,
    }


# ---------------------------------------------------------------------------
# Cubemap helpers (mirror the layout used by pcd_renderer.py).
# ---------------------------------------------------------------------------
def _build_cubemap_w2cs(frame_Rt: np.ndarray) -> np.ndarray:
    """For one frame's w2c, return the 6 cubemap-face w2cs (numpy float32)."""
    cur_Rt = frame_Rt[None]                                      # (1, 4, 4)
    out = np.tile(cur_Rt, (6, 1, 1))                             # (6, 4, 4)
    out[:, :3, :] = CUBEMAP_ROTATIONS @ out[:, :3, :]
    return out.astype(np.float32, copy=False)


def _spherical_dist_factor(cube_size: int) -> np.ndarray:
    """Per-pixel z->Euclidean factor for one cubemap face.

    Mirrors the helper used in :func:`pcd_pano_render_color_torch_renderer`:
    ``factor[v, u] = sqrt(((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1)``.
    """
    fx, fy = CUBEMAP_INTRINSICS[0, 0], CUBEMAP_INTRINSICS[1, 1]
    cx, cy = CUBEMAP_INTRINSICS[0, 2], CUBEMAP_INTRINSICS[1, 2]
    v_coords, u_coords = np.meshgrid(
        np.arange(cube_size, dtype=np.float32),
        np.arange(cube_size, dtype=np.float32),
        indexing="ij",
    )
    return np.sqrt(
        ((u_coords - cx) / fx) ** 2 + ((v_coords - cy) / fy) ** 2 + 1.0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Public render entry-point.
# ---------------------------------------------------------------------------
def gs_pano_render_color(
    gs: Dict[str, torch.Tensor],
    Rts: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
    render_depth: bool = False,
    erp_mask_aware: bool = False,
    background: Optional[Tuple[float, float, float]] = None,
    mask_alpha_threshold: float = 1e-2,
    near_plane: float = 1e-3,
    far_plane: float = 1.0e10,
    rasterize_mode: str = "classic",
):
    """Render panorama frames from spherical Gaussians via cubemap + ERP stitch.

    Signature is intentionally aligned with
    :func:`pcd_renderer.pcd_pano_render_color`.

    Parameters
    ----------
    gs : dict
        Output of :func:`build_isotropic_gs_from_pcd` (or anything providing
        the same five keys).
    Rts : (F, 4, 4) numpy array
        Per-frame world-to-camera matrices.
    height, width : int
        Output ERP panorama size (per frame).
    device : torch.device
        CUDA device for rasterization.
    render_depth : bool
        If True, also return per-frame ERP spherical-distance maps (in metres,
        same convention as the PCD path).
    erp_mask_aware : bool
        Forwarded to :func:`merge_panorama_color_torch`.
    background : (r, g, b) tuple in [-1, 1] (post-stitch convention) or None
        Pixels with cumulative alpha < ``mask_alpha_threshold`` keep this
        colour. Internally we map this to gsplat's [0, 1] convention before
        rasterization. ``None`` is equivalent to (0, 0, 0). Negative values
        are valid -- gsplat composes ``output = bg * (1 - alpha) + ...``.
    mask_alpha_threshold : float
        Alpha cutoff for the per-pixel valid mask (and for marking pixels as
        ``inf`` before ERP stitching, so background colour does not bleed).
    near_plane, far_plane, rasterize_mode :
        Passed through to gsplat. Defaults match the World3C reference.

    Returns
    -------
    Without ``render_depth``::
        (rgb_list, mask_list)
    With ``render_depth``::
        (rgb_list, depth_list, mask_list)

    where ``rgb_list[i]`` is an (H, W, 3) numpy array of floats in [0, 1],
    ``depth_list[i]`` is (H, W) spherical Euclidean distance, and
    ``mask_list[i]`` is (H, W) bool (True = valid).
    """
    rasterization = _import_rasterization()

    means = gs["means"].to(device=device, dtype=torch.float32)
    quats = gs["quats"].to(device=device, dtype=torch.float32)
    scales = gs["scales"].to(device=device, dtype=torch.float32)
    opacities = gs["opacities"].to(device=device, dtype=torch.float32)
    colors = gs["colors"].to(device=device, dtype=torch.float32)

    if means.shape[0] == 0:
        raise ValueError("Empty Gaussian set; nothing to render. Check the "
                         "input pcd or filter thresholds.")

    cube_size = CUBEMAP_SIZE
    K_face = torch.from_numpy(CUBEMAP_INTRINSICS).to(device=device, dtype=torch.float32)
    K_batch = K_face.unsqueeze(0).expand(6, 3, 3).contiguous()

    # gsplat expects bg in the same colour space as colors (here [0, 1]).
    if background is None:
        bg_01 = (0.0, 0.0, 0.0)
    else:
        bg_01 = tuple((float(c) + 1.0) * 0.5 for c in background)
    bg_tensor = torch.tensor(bg_01, device=device, dtype=torch.float32)
    bg_face = bg_tensor.unsqueeze(0).expand(6, 3).contiguous()  # (6, 3)

    z_to_dist = _spherical_dist_factor(cube_size)
    z_to_dist_t = torch.from_numpy(z_to_dist).to(device=device)  # (cube, cube)

    glue_ext = torch.from_numpy(CUBEMAP_GLUE_EXTRINSICS).to(device=device, dtype=torch.float32)
    K_merge = K_face.unsqueeze(0).expand(6, 3, 3).contiguous()

    render_mode = "RGB+ED" if render_depth else "RGB"

    out_rgb: List[np.ndarray] = []
    out_mask: List[np.ndarray] = []
    out_depth: List[np.ndarray] = []

    F = Rts.shape[0]
    for i in range(F):
        face_w2cs_np = _build_cubemap_w2cs(Rts[i])                # (6, 4, 4)
        face_w2cs = torch.from_numpy(face_w2cs_np).to(device=device, dtype=torch.float32)

        rendered, alpha, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=face_w2cs,
            Ks=K_batch,
            width=cube_size,
            height=cube_size,
            backgrounds=bg_face,
            packed=False,
            rasterize_mode=rasterize_mode,
            camera_model="pinhole",
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
        )
        # rendered: (6, H, W, 3) for RGB, (6, H, W, 4) for RGB+ED.
        # alpha: (6, H, W, 1)
        face_alpha = alpha[..., 0]                                # (6, H, W)
        face_valid = face_alpha > mask_alpha_threshold

        face_rgb = rendered[..., :3].clamp(0.0, 1.0)              # (6, H, W, 3) in [0, 1]
        # Mark invalid pixels with inf so merge_panorama_color_torch can
        # avoid bleeding background colour through bilinear sampling, exactly
        # like the PCD path.
        rgb_for_merge = face_rgb.clone()
        rgb_for_merge[~face_valid] = float("inf")

        panorama_rgb = merge_panorama_color_torch(
            width, height,
            rgb_for_merge,
            None,
            glue_ext,
            K_merge,
            device,
            erp_mask_aware=erp_mask_aware,
        )
        msk = panorama_rgb < float("inf")
        panorama_rgb[panorama_rgb == float("inf")] = 0.0
        out_rgb.append(panorama_rgb.detach().cpu().numpy().astype(np.float32, copy=False))
        out_mask.append(msk[:, :, 0].detach().cpu().numpy())

        if render_depth:
            face_z = rendered[..., 3]                              # (6, H, W) z-depth
            face_dist = face_z * z_to_dist_t                       # spherical distance
            face_dist_m = face_dist.clone()
            face_dist_m[~face_valid] = float("inf")
            depth_3ch = torch.stack([face_dist_m, face_dist_m, face_dist_m], dim=-1)

            panorama_depth_3ch = merge_panorama_color_torch(
                width, height,
                depth_3ch,
                None,
                glue_ext,
                K_merge,
                device,
                erp_mask_aware=erp_mask_aware,
            )
            panorama_depth = panorama_depth_3ch[:, :, 0]
            panorama_depth[panorama_depth == float("inf")] = 0.0
            out_depth.append(panorama_depth.detach().cpu().numpy().astype(np.float32, copy=False))

    if render_depth:
        return out_rgb, out_depth, out_mask
    return out_rgb, out_mask


def gs_pano_render_cube(
    gs: Dict[str, torch.Tensor],
    Rts: np.ndarray,
    device: torch.device,
    near_plane: float = 1e-3,
    far_plane: float = 1.0e10,
    rasterize_mode: str = "classic",
):
    """Rasterize the 6 cubemap faces per frame WITHOUT ERP stitching.

    This is the fast sibling of :func:`gs_pano_render_color` for metrics that
    can be computed in cube space (coverage via solid-angle-weighted alpha;
    PSNR via comparison against an ERP GT resampled into the same faces). It
    skips both the per-frame ERP merge (6x ``grid_sample``) and the per-frame
    GPU->CPU copy, keeping everything on ``device``.

    Returns
    -------
    face_rgb : (F, 6, cube, cube, 3) float tensor in [0, 1]
    face_alpha : (F, 6, cube, cube) float tensor (cumulative alpha; valid =
        ``alpha > mask_alpha_threshold`` at the call site)

    Face convention matches :func:`gs_pano_render_color` exactly: face i is
    rendered with ``CUBEMAP_ROTATIONS[i] @ frame_w2c`` (via
    :func:`_build_cubemap_w2cs`) and the cube intrinsics ``CUBEMAP_INTRINSICS``.
    """
    rasterization = _import_rasterization()

    means = gs["means"].to(device=device, dtype=torch.float32)
    quats = gs["quats"].to(device=device, dtype=torch.float32)
    scales = gs["scales"].to(device=device, dtype=torch.float32)
    opacities = gs["opacities"].to(device=device, dtype=torch.float32)
    colors = gs["colors"].to(device=device, dtype=torch.float32)
    if means.shape[0] == 0:
        raise ValueError("Empty Gaussian set; nothing to render.")

    cube_size = CUBEMAP_SIZE
    K_face = torch.from_numpy(CUBEMAP_INTRINSICS).to(device=device, dtype=torch.float32)
    K_batch = K_face.unsqueeze(0).expand(6, 3, 3).contiguous()
    bg_face = torch.zeros(6, 3, device=device, dtype=torch.float32)

    rgb_out: List[torch.Tensor] = []
    alpha_out: List[torch.Tensor] = []
    F = Rts.shape[0]
    for i in range(F):
        face_w2cs = torch.from_numpy(_build_cubemap_w2cs(Rts[i])).to(
            device=device, dtype=torch.float32)
        rendered, alpha, _meta = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=face_w2cs, Ks=K_batch,
            width=cube_size, height=cube_size, backgrounds=bg_face,
            packed=False, rasterize_mode=rasterize_mode,
            camera_model="pinhole", near_plane=near_plane, far_plane=far_plane,
            render_mode="RGB",
        )
        rgb_out.append(rendered[..., :3].clamp(0.0, 1.0))  # (6, cs, cs, 3)
        alpha_out.append(alpha[..., 0])                     # (6, cs, cs)
    return torch.stack(rgb_out, dim=0), torch.stack(alpha_out, dim=0)


__all__ = [
    "build_isotropic_gs_from_pcd",
    "gs_pano_render_color",
    "gs_pano_render_cube",
]
