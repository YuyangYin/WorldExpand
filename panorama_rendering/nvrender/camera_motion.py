"""Camera motion, core rendering orchestration, and public entry-point functions."""

import math
import time

import numpy as np
import torch
import trimesh

from .utils import depth_repair, depth_edge_torch, collect_perf_stats
from .geometry import get_mesh_from_pano_Rt
from .mesh_renderer import mesh_pano_render_color, mesh_pano_render_rgb_and_depth
from .pcd_renderer import pcd_pano_render_color, pcd_pano_render_color_torch_renderer
from .gs_renderer import build_isotropic_gs_from_pcd, gs_pano_render_color


# ============================================================================
# Euler angle / rotation helpers
# ============================================================================

def euler_to_rotation_matrix(pitch, yaw, roll, device='cpu'):
    """
    Convert Euler angles (degrees) to a 3x3 rotation matrix.

    Unreal Engine convention:
    - Pitch: Y-axis rotation
    - Yaw: Z-axis rotation
    - Roll: X-axis rotation
    - Rotation order: Yaw * Pitch * Roll (ZYX)

    Args:
        pitch, yaw, roll: angles in degrees
        device: torch device

    Returns:
        R: [3, 3] rotation matrix
    """
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(roll)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad), math.cos(roll_rad)]
    ], dtype=torch.float32, device=device)

    Ry = torch.tensor([
        [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
        [0, 1, 0],
        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ], dtype=torch.float32, device=device)

    Rz = torch.tensor([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad), math.cos(yaw_rad), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    R = Rz @ Ry @ Rx
    return R


# ============================================================================
# UE5 <-> render coordinate system conversion
# ============================================================================

# UE5: X=forward, Y=right, Z=up
# Render (opencv): X=right, Y=down, Z=forward
_ROT_MATRIX_3x3 = np.array([
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0]
], dtype=np.float32)

_ROT_MATRIX_4x4 = np.eye(4, dtype=np.float32)
_ROT_MATRIX_4x4[:3, :3] = _ROT_MATRIX_3x3


def _build_w2c_render_from_ue5(position, yaw_deg):
    """
    Build a render-space w2c matrix from UE5 position (metres) and yaw (degrees).

    Args:
        position: dict with 'X', 'Y', 'Z' in metres
        yaw_deg: yaw angle in degrees

    Returns:
        w2c_render: [4, 4] numpy array
    """
    yaw_rad = math.radians(yaw_deg)

    rotation_matrix = np.array([
        [math.cos(yaw_rad), math.sin(yaw_rad), 0],
        [-math.sin(yaw_rad), math.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    c2w_ue5 = np.eye(4)
    c2w_ue5[:3, -1] = np.array([position['X'], position['Y'], position['Z']])
    c2w_ue5[:3, :3] = rotation_matrix.T

    w2c_ue5 = np.linalg.inv(c2w_ue5)
    w2c_render = _ROT_MATRIX_4x4 @ w2c_ue5 @ _ROT_MATRIX_4x4.T
    return w2c_render


# ============================================================================
# Core rendering orchestrator
# ============================================================================

def generate_pc_render(Rts, first_frame_rgb, first_frame_depth,
                       first_frame_Rt=None, rtol=0.05, mesh_path=None,
                       angle=0., render_depth=False, render_backend='mesh',
                       render_mode='pytorch3d', max_pcd_points=5_000_000,
                       erp_mask_aware=False, softmin=False, z_soft_alpha=500.0,
                       use_fp16_cache=True,
                       gs_scale_factor=1.0, gs_background=None):
    """
    Core function: render panorama frames from mesh or point cloud.

    Args:
        Rts: [F, 4, 4] w2c matrices
        first_frame_rgb: [H, W, 3] first-frame RGB tensor
        first_frame_depth: [H, W] first-frame depth tensor
        first_frame_Rt: [4, 4] first-frame w2c (default: Rts[0])
        rtol: depth edge tolerance
        mesh_path: path to .ply file
        angle: panorama rotation angle
        render_depth: whether to also render depth
        render_backend: 'mesh' or 'pcd'
        render_mode: 'pytorch3d' / 'point_renderer' / 'gsplat' (only for pcd backend)
        max_pcd_points: max points for PCD downsampling (ignored when render_mode='gsplat')
        erp_mask_aware: use mask-aware ERP stitching
        softmin: use softmin depth blending for point_renderer (default: False)
        z_soft_alpha: softmin sharpness parameter (default: 500.0, lower = smoother)
        gs_scale_factor: per-splat sigma multiplier for the gsplat engine (default 1.0)
        gs_background: optional 3-tuple background colour in [-1, 1] for the gsplat
            engine. ``None`` falls back to (0, 0, 0). Negative values are valid;
            gsplat composes ``out = bg * (1 - alpha) + ...``.

    Returns:
        all_canvas_rgb, [all_canvas_depth,] all_canvas_mask, perf_stats
    """
    device = first_frame_depth.device
    H, W = first_frame_depth.shape[:2]
    frame_size = len(Rts)

    edge_mask_first_frame = ~(depth_edge_torch(first_frame_depth, rtol=rtol))

    all_w2cs = Rts
    all_canvas_mask = torch.zeros((frame_size, H, W), dtype=torch.bool, device=device)
    all_canvas_rgb = torch.zeros((frame_size, H, W, 3), dtype=torch.float32, device=device)

    if first_frame_Rt is None:
        first_frame_Rt = Rts[0]

    # ---- Mesh backend: build or load mesh ----
    mesh_full = None
    mesh_loaded_flag = False
    if render_backend != 'pcd':
        if mesh_path is not None:
            mesh_full = trimesh.load(mesh_path, file_type='ply')

            rot_matrix = np.array([
                [0, 1, 0],
                [0, 0, -1],
                [-1, 0, 0.]
            ], dtype=np.float32)
            mesh_full.vertices = mesh_full.vertices @ rot_matrix.T

            first_frame_Rt_np = first_frame_Rt.cpu().numpy()
            c2w = np.linalg.inv(first_frame_Rt_np)
            mesh_full.vertices = mesh_full.vertices @ c2w[:3, :3].T + c2w[:3, 3]

            mesh_loaded_flag = True
        else:
            mesh_loaded_flag = False
            mesh_full = get_mesh_from_pano_Rt(
                first_frame_depth.cpu().numpy(),
                np.ones((H, W), dtype=np.bool_),
                first_frame_Rt.cpu().numpy(),
                pano_rgb=first_frame_rgb.cpu().numpy(),
                alpha_mask=(edge_mask_first_frame).cpu().numpy(),
                save_pcd=True,
                pcd_path="debug_pointcloud.ply"
            )

    near = 1e-3
    far = first_frame_depth.max() * 1.1

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.memory_summary()

    # ---- PCD backend ----
    if render_backend == 'pcd':
        if mesh_path is None:
            raise ValueError("PCD render backend requires mesh_path to load the .ply file.")

        pcd_raw = trimesh.load(mesh_path, file_type='ply')
        pcd_vertices = np.array(pcd_raw.vertices).astype(np.float32)
        if isinstance(pcd_raw, trimesh.PointCloud):
            pcd_colors_raw = np.array(pcd_raw.colors)[:, :3].astype(np.float32)
        else:
            pcd_colors_raw = np.array(pcd_raw.visual.vertex_colors)[:, :3].astype(np.float32)
        if pcd_colors_raw.max() > 5:
            pcd_colors_raw = pcd_colors_raw / 255.

        rot_matrix = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0.]
        ], dtype=np.float32)
        pcd_vertices = pcd_vertices @ rot_matrix.T

        first_frame_Rt_np = first_frame_Rt.cpu().numpy()
        c2w = np.linalg.inv(first_frame_Rt_np)
        pcd_vertices = pcd_vertices @ c2w[:3, :3].T + c2w[:3, 3]

        pcd_vertices = np.ascontiguousarray(pcd_vertices)
        pcd_colors_raw = np.ascontiguousarray(pcd_colors_raw)

        # Filter NaN / Inf / extreme outliers
        nan_verts = np.isnan(pcd_vertices).any(axis=1)
        inf_verts = np.isinf(pcd_vertices).any(axis=1)
        nan_colors = np.isnan(pcd_colors_raw).any(axis=1)
        inf_colors = np.isinf(pcd_colors_raw).any(axis=1)
        bad_mask = nan_verts | inf_verts | nan_colors | inf_colors
        if bad_mask.sum() > 0:
            good_mask = ~bad_mask
            pcd_vertices = np.ascontiguousarray(pcd_vertices[good_mask])
            pcd_colors_raw = np.ascontiguousarray(pcd_colors_raw[good_mask])

        abs_max = np.abs(pcd_vertices).max(axis=1)
        extreme_mask = abs_max > 1e4
        if extreme_mask.sum() > 0:
            good_mask2 = ~extreme_mask
            pcd_vertices = np.ascontiguousarray(pcd_vertices[good_mask2])
            pcd_colors_raw = np.ascontiguousarray(pcd_colors_raw[good_mask2])

        # Downsample if needed
        if max_pcd_points is not None and pcd_vertices.shape[0] > max_pcd_points:
            indices = np.random.choice(pcd_vertices.shape[0], max_pcd_points, replace=False)
            indices.sort()
            pcd_vertices = np.ascontiguousarray(pcd_vertices[indices])
            pcd_colors_raw = np.ascontiguousarray(pcd_colors_raw[indices])

        pcd_points = torch.from_numpy(pcd_vertices).float()
        pcd_colors_np = np.ascontiguousarray(pcd_colors_raw * 2. - 1.)
        pcd_colors = torch.from_numpy(pcd_colors_np).float()

        # Silenced for tqdm progress bar compatibility
        # print(f"  [generate_pc_render] render_mode={render_mode}, render_depth={render_depth}, "
        #       f"num_points={pcd_points.shape[0]}, num_frames={frame_size}")

        torch.cuda.reset_peak_memory_stats(device)
        per_frame_times = []

        if render_mode == 'pytorch3d':
            if render_depth:
                t0 = time.time()
                torch.cuda.synchronize(device)
                rendered_cmap, rendered_depth_maps, rendered_cmap_mask = pcd_pano_render_color(
                    pcd_points, pcd_colors, all_w2cs.cpu().numpy(), H, W, device,
                    render_radius=0.008, points_per_pixel=8, render_depth=True,
                    erp_mask_aware=erp_mask_aware
                )
                torch.cuda.synchronize(device)
                total_t = time.time() - t0
                per_frame_times = [total_t / frame_size] * frame_size

                all_canvas_depth = torch.zeros((frame_size, H, W), dtype=torch.float32, device=device)
                for i in range(frame_size):
                    cmap_mask = np.ascontiguousarray(rendered_cmap_mask[i])
                    cmap = np.ascontiguousarray(rendered_cmap[i])
                    depth_arr = np.ascontiguousarray(rendered_depth_maps[i])
                    cmap[~cmap_mask] = 0.
                    depth_arr[~cmap_mask] = 0.
                    all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
                    all_canvas_depth[i] = torch.from_numpy(depth_arr).float().to(device)
                    all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)
            else:
                t0 = time.time()
                torch.cuda.synchronize(device)
                rendered_cmap, rendered_cmap_mask = pcd_pano_render_color(
                    pcd_points, pcd_colors, all_w2cs.cpu().numpy(), H, W, device,
                    render_radius=0.008, points_per_pixel=8, render_depth=False,
                    erp_mask_aware=erp_mask_aware
                )
                torch.cuda.synchronize(device)
                total_t = time.time() - t0
                per_frame_times = [total_t / frame_size] * frame_size

                all_canvas_depth = None
                for i in range(frame_size):
                    cmap_mask = np.ascontiguousarray(rendered_cmap_mask[i])
                    cmap = np.ascontiguousarray(rendered_cmap[i])
                    cmap[~cmap_mask] = 0.
                    all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
                    all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)

        elif render_mode == 'point_renderer':
            if render_depth:
                t0 = time.time()
                torch.cuda.synchronize(device)
                rendered_cmap, rendered_depth_maps, rendered_cmap_mask = pcd_pano_render_color_torch_renderer(
                    pcd_points, pcd_colors, all_w2cs.cpu().numpy(), H, W, device,
                    render_depth=True,
                    erp_mask_aware=erp_mask_aware,
                    softmin=softmin, z_soft_alpha=z_soft_alpha,
                    use_fp16_cache=use_fp16_cache,
                )
                torch.cuda.synchronize(device)
                total_t = time.time() - t0
                per_frame_times = [total_t / frame_size] * frame_size

                all_canvas_depth = torch.zeros((frame_size, H, W), dtype=torch.float32, device=device)
                for i in range(frame_size):
                    cmap_mask = np.ascontiguousarray(rendered_cmap_mask[i])
                    cmap = np.ascontiguousarray(rendered_cmap[i])
                    depth_arr = np.ascontiguousarray(rendered_depth_maps[i])
                    cmap[~cmap_mask] = 0.
                    depth_arr[~cmap_mask] = 0.
                    all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
                    all_canvas_depth[i] = torch.from_numpy(depth_arr).float().to(device)
                    all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)
            else:
                t0 = time.time()
                torch.cuda.synchronize(device)
                rendered_cmap, rendered_cmap_mask = pcd_pano_render_color_torch_renderer(
                    pcd_points, pcd_colors, all_w2cs.cpu().numpy(), H, W, device,
                    render_depth=False,
                    erp_mask_aware=erp_mask_aware,
                    softmin=softmin, z_soft_alpha=z_soft_alpha,
                    use_fp16_cache=use_fp16_cache,
                )
                torch.cuda.synchronize(device)
                total_t = time.time() - t0
                per_frame_times = [total_t / frame_size] * frame_size

                all_canvas_depth = None
                for i in range(frame_size):
                    cmap_mask = np.ascontiguousarray(rendered_cmap_mask[i])
                    cmap = np.ascontiguousarray(rendered_cmap[i])
                    cmap[~cmap_mask] = 0.
                    all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
                    all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)
        elif render_mode == 'gsplat':
            # Build isotropic spherical Gaussians from the same point cloud
            # the other engines consume; render via cubemap + ERP stitch.
            gs = build_isotropic_gs_from_pcd(
                points=pcd_points,
                colors=pcd_colors,
                first_frame_Rt=first_frame_Rt,
                scale_factor=gs_scale_factor,
                device=device,
            )
            t0 = time.time()
            torch.cuda.synchronize(device)
            if render_depth:
                rendered_cmap, rendered_depth_maps, rendered_cmap_mask = gs_pano_render_color(
                    gs, all_w2cs.cpu().numpy(), H, W, device,
                    render_depth=True,
                    erp_mask_aware=erp_mask_aware,
                    background=gs_background,
                )
            else:
                rendered_cmap, rendered_cmap_mask = gs_pano_render_color(
                    gs, all_w2cs.cpu().numpy(), H, W, device,
                    render_depth=False,
                    erp_mask_aware=erp_mask_aware,
                    background=gs_background,
                )
            torch.cuda.synchronize(device)
            total_t = time.time() - t0
            per_frame_times = [total_t / frame_size] * frame_size

            if render_depth:
                all_canvas_depth = torch.zeros((frame_size, H, W), dtype=torch.float32, device=device)
                for i in range(frame_size):
                    cmap_mask = np.ascontiguousarray(rendered_cmap_mask[i])
                    cmap = np.ascontiguousarray(rendered_cmap[i])
                    depth_arr = np.ascontiguousarray(rendered_depth_maps[i])
                    cmap[~cmap_mask] = 0.
                    depth_arr[~cmap_mask] = 0.
                    all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
                    all_canvas_depth[i] = torch.from_numpy(depth_arr).float().to(device)
                    all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)
            else:
                all_canvas_depth = None
                for i in range(frame_size):
                    cmap_mask = np.ascontiguousarray(rendered_cmap_mask[i])
                    cmap = np.ascontiguousarray(rendered_cmap[i])
                    cmap[~cmap_mask] = 0.
                    all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
                    all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)
        else:
            raise ValueError(f"Unknown render_mode: {render_mode}. Must be 'pytorch3d', 'point_renderer', or 'gsplat'.")

        perf_stats = collect_perf_stats(per_frame_times, render_mode, frame_size, device,
                                        num_points=pcd_points.shape[0])

    elif render_backend == 'mesh':
        if render_mode == 'gsplat':
            # Catches the user-error combo --render_mode mesh --render_engine gsplat
            # at the deepest call-site, even if the CLI guard is bypassed.
            raise ValueError(
                "render_mode='gsplat' is incompatible with render_backend='mesh'. "
                "gsplat consumes the per-point cloud (global_pcd.ply); pick "
                "--render_mode pcd."
            )
        if render_depth:
            rendered_rgb, rendered_depth_list, rendered_mask = mesh_pano_render_rgb_and_depth(
                mesh_full, all_w2cs.cpu().numpy(), H, W, near, far, device
            )
            all_canvas_depth = torch.zeros((frame_size, H, W), dtype=torch.float32, device=device)
            for i in range(frame_size):
                rgb = rendered_rgb[i]
                depth_arr = rendered_depth_list[i]
                mask = rendered_mask[i]
                rgb[~mask] = 0.
                depth_arr[~mask] = 0.
                all_canvas_rgb[i] = torch.from_numpy(rgb).float().to(device)
                all_canvas_depth[i] = torch.from_numpy(depth_arr).float().to(device)
                all_canvas_mask[i] = torch.from_numpy(mask).float().to(device)
        else:
            rendered_cmap, rendered_cmap_mask = mesh_pano_render_color(
                mesh_full, all_w2cs.cpu().numpy(), H, W, near, far, device
            )
            all_canvas_depth = None
            for i in range(frame_size):
                cmap_mask = rendered_cmap_mask[i]
                cmap = rendered_cmap[i]
                cmap[~cmap_mask] = 0.
                all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
                all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)

        perf_stats = None
    else:
        raise ValueError(f"Unknown render_backend: {render_backend}. Must be 'mesh' or 'pcd'.")

    if render_depth:
        return all_canvas_rgb, all_canvas_depth, all_canvas_mask, perf_stats
    else:
        return all_canvas_rgb, all_canvas_mask, perf_stats


def generate_mesh_render(Rts, first_frame_rgb, first_frame_depth, first_frame_Rt=None):
    """Legacy mesh render function (loads global_mesh1.ply from cwd)."""
    device = first_frame_depth.device
    H, W = first_frame_depth.shape[:2]
    frame_size = len(Rts)
    edge_mask_first_frame = ~(depth_edge_torch(first_frame_depth, rtol=0.05))

    all_canvas_mask = torch.zeros((frame_size, H, W), dtype=torch.bool, device=device)
    all_canvas_rgb = torch.zeros((frame_size, H, W, 3), dtype=torch.float32, device=device)

    all_w2cs = Rts

    if first_frame_Rt is None:
        first_frame_Rt = Rts[0]

    with open('global_mesh1.ply', 'rb') as f:
        mesh_full = trimesh.load(f, file_type='ply')

    near = 1e-3
    far = first_frame_depth.max() * 1.1

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.memory_summary()
    rendered_cmap, rendered_cmap_mask = mesh_pano_render_color(
        mesh_full, all_w2cs.cpu().numpy(), H, W, near, far, device
    )

    for i in range(frame_size):
        cmap_mask = rendered_cmap_mask[i]
        cmap = rendered_cmap[i]
        cmap[~cmap_mask] = 0.
        all_canvas_rgb[i] = torch.from_numpy(cmap).float().to(device)
        all_canvas_mask[i] = torch.from_numpy(cmap_mask).float().to(device)

    return all_canvas_rgb, all_canvas_mask


# ============================================================================
# High-level camera movement functions (JSON-based)
# ============================================================================

def _prepare_firstframe(firstframe_rgb, firstframe_depth, angle):
    """Common preprocessing for perform_camera_movement_* functions."""
    device = firstframe_rgb.device
    depth_np = firstframe_depth.cpu().numpy()
    depth_np = depth_repair(depth_np)
    firstframe_depth = torch.from_numpy(depth_np).to(device)

    H, W = firstframe_rgb.shape[:2]
    pvt = int(angle / 360. * W + W // 2) % W

    firstframe_rgb = torch.cat([firstframe_rgb[:, pvt:], firstframe_rgb[:, :pvt]], dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:, pvt:], firstframe_depth[:, :pvt]], dim=1)

    return firstframe_rgb, firstframe_depth


def perform_camera_movement_with_json_final_only(
        firstframe_rgb, firstframe_depth,
        start_position, start_rotation,
        end_position, end_rotation,
        angle=0., frame_size=81, rtol=0.05, mesh_path=None,
        render_depth=False, render_backend='mesh', render_mode='pytorch3d',
        max_pcd_points=5_000_000, erp_mask_aware=False,
        softmin=False, z_soft_alpha=500.0,
        use_fp16_cache=True,
        gs_scale_factor=1.0, gs_background=None):
    """
    Optimised version: only render the final frame (skip intermediate frames).

    Returns:
        If render_depth: final_rgb, final_depth, final_mask, final_Rt, firstframe_rgb, firstframe_depth, perf_stats
        Else: final_rgb, final_mask, final_Rt, firstframe_rgb, firstframe_depth, perf_stats
    """
    device = firstframe_rgb.device
    firstframe_rgb, firstframe_depth = _prepare_firstframe(firstframe_rgb, firstframe_depth, angle)

    pos_start = {k: start_position[k] / 100.0 for k in ('X', 'Y', 'Z')}
    start_Rt = torch.from_numpy(
        _build_w2c_render_from_ue5(pos_start, start_rotation['Yaw'])
    ).float().to(device)

    pos_end = {k: end_position[k] / 100.0 for k in ('X', 'Y', 'Z')}
    end_Rt = torch.from_numpy(
        _build_w2c_render_from_ue5(pos_end, end_rotation['Yaw'])
    ).float().to(device)

    Rts = end_Rt[None]  # [1, 4, 4]

    torch.cuda.empty_cache()

    result = generate_pc_render(
        Rts,
        firstframe_rgb,
        firstframe_depth,
        first_frame_Rt=start_Rt,
        rtol=rtol,
        mesh_path=mesh_path,
        render_depth=render_depth,
        render_backend=render_backend,
        render_mode=render_mode,
        max_pcd_points=max_pcd_points,
        erp_mask_aware=erp_mask_aware,
        softmin=softmin, z_soft_alpha=z_soft_alpha,
        use_fp16_cache=use_fp16_cache,
        gs_scale_factor=gs_scale_factor,
        gs_background=gs_background,
    )

    if render_depth:
        rendered_rgb, rendered_depth, rendered_mask, perf_stats = result
        final_rgb = rendered_rgb[0]
        final_depth = rendered_depth[0]
        final_mask = rendered_mask[0]
        return final_rgb, final_depth, final_mask, end_Rt, firstframe_rgb, firstframe_depth, perf_stats
    else:
        rendered_rgb, rendered_mask, perf_stats = result
        final_rgb = rendered_rgb[0]
        final_mask = rendered_mask[0]
        return final_rgb, final_mask, end_Rt, firstframe_rgb, firstframe_depth, perf_stats


def perform_camera_movement_with_json(
        firstframe_rgb, firstframe_depth,
        start_position, start_rotation,
        end_position, end_rotation,
        angle=0., frame_size=81, rtol=0.05, mesh_path=None,
        render_depth=False, render_backend='mesh', render_mode='pytorch3d',
        max_pcd_points=5_000_000, erp_mask_aware=False,
        softmin=False, z_soft_alpha=500.0,
        use_fp16_cache=True,
        gs_scale_factor=1.0, gs_background=None):
    """
    Generate camera movement from JSON start/end position/rotation.
    Interpolates all intermediate frames.

    Returns:
        If render_depth: rgbs, depths, masks, Rts, firstframe_rgb, firstframe_depth, perf_stats
        Else: rgbs, masks, Rts, firstframe_rgb, firstframe_depth, perf_stats
    """
    device = firstframe_rgb.device
    firstframe_rgb, firstframe_depth = _prepare_firstframe(firstframe_rgb, firstframe_depth, angle)

    # Build interpolated Rt matrices for each frame
    Rts = []
    for i in range(frame_size):
        t = i / (frame_size - 1)

        pos_t = {
            'X': (start_position['X'] * (1 - t) + end_position['X'] * t) / 100.0,
            'Y': (start_position['Y'] * (1 - t) + end_position['Y'] * t) / 100.0,
            'Z': (start_position['Z'] * (1 - t) + end_position['Z'] * t) / 100.0
        }

        yaw_t = start_rotation['Yaw'] * (1 - t) + end_rotation['Yaw'] * t

        Rt = torch.from_numpy(
            _build_w2c_render_from_ue5(pos_t, yaw_t)
        ).float().to(device)

        Rts.append(Rt)

    Rts = torch.stack(Rts)

    result = generate_pc_render(
        Rts, firstframe_rgb, firstframe_depth,
        rtol=rtol, mesh_path=mesh_path, angle=angle,
        render_depth=render_depth, render_backend=render_backend,
        render_mode=render_mode, max_pcd_points=max_pcd_points,
        erp_mask_aware=erp_mask_aware,
        softmin=softmin, z_soft_alpha=z_soft_alpha,
        use_fp16_cache=use_fp16_cache,
        gs_scale_factor=gs_scale_factor,
        gs_background=gs_background,
    )

    if render_depth:
        rgbs, depths, masks, perf_stats = result
        return rgbs, depths, masks, Rts, firstframe_rgb, firstframe_depth, perf_stats
    else:
        rgbs, masks, perf_stats = result
        return rgbs, masks, Rts, firstframe_rgb, firstframe_depth, perf_stats


def perform_camera_movement(firstframe_rgb, firstframe_depth,
                            angle=0., movement_ratio=0.5, frame_size=81):
    """
    Legacy function: simple forward camera movement (backward compatible).
    """
    device = firstframe_rgb.device
    depth_np = firstframe_depth.cpu().numpy()
    depth_np = depth_repair(depth_np)
    firstframe_depth = torch.from_numpy(depth_np).to(device)

    H, W = firstframe_rgb.shape[:2]
    pvt = int(angle / 360. * W + W // 2) % W
    firstframe_rgb = torch.cat([firstframe_rgb[:, pvt:], firstframe_rgb[:, :pvt]], dim=1)
    firstframe_depth = torch.cat([firstframe_depth[:, pvt:], firstframe_depth[:, :pvt]], dim=1)

    depth_pvt = firstframe_depth[H // 2, int(W // 2 * 1.05)]
    Rt0 = torch.tensor([
        [1., 0, 0, 0],
        [0, 1., 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    Rts = Rt0[None].repeat(frame_size, 1, 1)
    for i in range(frame_size):
        Rts[i, 2, 3] = -(depth_pvt * i * movement_ratio) / (frame_size - 1)

    rgbs, masks = generate_pc_render(Rts, firstframe_rgb, firstframe_depth)

    return rgbs, masks, Rts, firstframe_rgb, firstframe_depth
