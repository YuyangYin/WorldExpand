"""Mesh geometry extension for the four-view online-warp path.

Standalone helpers to build a **world-space triangle mesh** from a single
panoramic start RGBD frame and render it to arbitrary end-view cameras, so the
online-warp path can use a mesh geometry representation as an alternative to the
point-cloud / isotropic-GS route in ``online_warp_renderer.py``.

Design goals
------------
* Pure functions, no dependency on ``OnlineWarpRenderer`` internals, so the same
  code can be called from the renderer, from a test harness, or from inference.
* **Same world coordinate frame** as the GS/PCD path. ``convert_rgbd2mesh_panorama``
  builds vertices as ``distance * rays`` in the *start-camera sphere frame*; we
  then apply the exact same ``sphere -> render-OpenCV -> world`` transform used in
  ``OnlineWarpRenderer.build_world_gs`` (``verts @ _PLY_ROT.T @ start_c2w``) so a
  mesh warp and a GS warp of the same start frame line up pixel-for-pixel.
* Rendering uses the already-present nvdiffrast cubemap->ERP renderer
  ``panorama_rendering.nvrender.mesh_renderer.mesh_pano_render_color`` (same approach as
  the reference ``PanoWarpingxuhui/demo_exr_xyz_final_multi_gpu.py``).

The heavy imports (open3d via ``convert_rgbd2mesh_panorama``, nvdiffrast via
``mesh_pano_render_color``) are done lazily inside the functions so importing this
module is cheap and never pulls GL libs into a headless process that only needs
the GS path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Sphere-coords -> render-OpenCV, identical to online_warp_renderer._PLY_ROT.
# Kept local so this module does not import renderer internals.
_PLY_ROT = np.array(
    [
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0.0],
    ],
    dtype=np.float32,
)


def build_world_mesh(
    start_rgb: torch.Tensor,          # (H, W, 3) float [0,1], start-cam ERP
    distance: torch.Tensor,           # (H, W) float metres, cleaned distance
    rays: torch.Tensor,               # (H, W, 3) unit ERP ray dirs (sphere frame)
    full_mask: torch.Tensor,          # (H, W) bool, True = exclude (sky|edge)
    start_c2w_np: np.ndarray,         # (4, 4) render-OpenCV cam->world
    device: torch.device,
    mesh_h: Optional[int] = None,
    mesh_w: Optional[int] = None,
    connect_boundary: bool = False,
):
    """Build a world-space ``trimesh.Trimesh`` (vertex-coloured) from start RGBD.

    Returns the mesh, or ``None`` if the geometry is degenerate. Vertices are in
    the SAME world frame as ``build_world_gs`` so the same end w2c poses render
    consistently across mesh / gs / pcd.

    ``connect_boundary=False`` (default) skips ``convert_rgbd2mesh_panorama``'s
    ``_fill_small_boundary_spikes`` step. That step is slow (dominates build time)
    AND is where Open3D can abort with ``malloc(): invalid size`` on large
    (1920x3840) panoramas. The boundary fill only patches small holes at depth
    edges — not essential for a warp condition — so we turn it off by default for
    stability + speed. Set True to restore the original behaviour.
    """
    import trimesh
    from panorama_rendering.render_utils.panorama_utils import convert_rgbd2mesh_panorama

    H, W = distance.shape[-2:]
    # Optional resize to a mesh-build resolution (mirrors step4 mesh path).
    if mesh_h is not None and mesh_w is not None and (mesh_h != H or mesh_w != W):
        import torch.nn.functional as F
        rgb_in = F.interpolate(
            start_rgb.permute(2, 0, 1)[None], size=(mesh_h, mesh_w), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).contiguous()
        dist_in = F.interpolate(
            distance[None, None], size=(mesh_h, mesh_w), mode="nearest"
        )[0, 0]
        rays_in = F.normalize(
            F.interpolate(rays.permute(2, 0, 1)[None], size=(mesh_h, mesh_w), mode="bilinear", align_corners=False)[0],
            dim=0,
        ).permute(1, 2, 0).contiguous()
        mask_in = F.interpolate(
            full_mask.float()[None, None], size=(mesh_h, mesh_w), mode="nearest"
        )[0, 0].bool()
    else:
        rgb_in, dist_in, rays_in, mask_in = start_rgb, distance, rays, full_mask

    # o3d mesh: verts = distance * rays in START-CAM SPHERE frame, colors [0,1].
    o3d_mesh = convert_rgbd2mesh_panorama(
        rgb=rgb_in.to(device),
        distance=dist_in.to(device),
        rays=rays_in.to(device),
        excluded_region_mask=mask_in.to(device),
        device="cuda" if str(device).startswith("cuda") else "cpu",
        connect_boundary_max_dist=(0.5 if connect_boundary else None),
    )

    verts_sphere = np.asarray(o3d_mesh.vertices, dtype=np.float32)   # (N,3) sphere frame
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)          # (F,3)
    colors = np.asarray(o3d_mesh.vertex_colors, dtype=np.float32)   # (N,3) [0,1]
    if verts_sphere.shape[0] == 0 or faces.shape[0] == 0:
        return None

    # sphere -> render-OpenCV -> world (identical math to build_world_gs).
    verts_render = verts_sphere @ _PLY_ROT.T
    c2w = start_c2w_np.astype(np.float32)
    verts_world = verts_render @ c2w[:3, :3].T + c2w[:3, 3]

    # trimesh wants uint8 or [0,255] vertex_colors (RGBA ok); mesh_pano_render_color
    # divides by 255 when max>5, so give it 0..255.
    vcol = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    mesh = trimesh.Trimesh(
        vertices=verts_world.astype(np.float64),
        faces=faces,
        vertex_colors=vcol,
        process=False,
    )
    return mesh


def render_mesh_multiview(
    mesh,                              # trimesh.Trimesh (world frame)
    end_w2c_list: List[np.ndarray],   # V x (4,4) render-OpenCV world->cam
    height: int,
    width: int,
    device: torch.device,
    near: float = 1e-3,
    far: float = 1.0e4,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Render the mesh to each end camera. Returns ``(warps, masks)``:
    warps = V x (H,W,3) float[0,1] on device; masks = V x (H,W) bool on device.
    """
    from panorama_rendering.nvrender.mesh_renderer import mesh_pano_render_color

    Rts = np.stack([w.astype(np.float32) for w in end_w2c_list], axis=0)  # (V,4,4)
    rgb_list, mask_list = mesh_pano_render_color(
        mesh, Rts, height, width, near, far, device
    )
    warps: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for rgb, m in zip(rgb_list, mask_list):
        w = torch.from_numpy(np.ascontiguousarray(rgb.astype(np.float32))).to(device)
        # nvdiffrast can emit a few NaN pixels (degenerate triangles / empty
        # coverage). ``clamp(0,1)`` does NOT remove NaN (nan comparisons are
        # false), so a handful of NaN pixels here would otherwise survive into
        # the VAE encode and poison the ENTIRE latent -> model outputs NaN ->
        # black frame at inference. The offline mesh_cached path never hit this
        # because the uint8 PNG round-trip already maps NaN->0; the online path
        # must sanitise explicitly to match. Zero-fill NaN, then clamp.
        w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
        warps.append(w)
        masks.append(torch.from_numpy(np.ascontiguousarray(m)).to(device).bool())
    return warps, masks
