"""Point-cloud-based panorama rendering (pytorch3d and TorchPointCloudRenderer)."""

import numpy as np
import torch

from pytorch3d.renderer import (
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
)
from pytorch3d.structures import Pointclouds
from panorama_rendering.render_utils.pointcloud import PointsZbufRenderer, suppress_stdout_stderr, point_rendering
from ..point_renderer import TorchPointCloudRenderer

from .cubemap import (
    CUBEMAP_ROTATIONS, CUBEMAP_GLUE_EXTRINSICS, CUBEMAP_INTRINSICS, CUBEMAP_SIZE,
    merge_panorama_color_torch,
)


def point_rendering_distance(K, w2cs, points, device, h, w,
                              render_radius=0.008, points_per_pixel=8):
    """
    Render spherical distance map from point cloud.
    For each cubemap face, transform points to camera space, compute Euclidean distance,
    and use the renderer to composite the distance values.

    Args:
        K: [F, 3, 3] intrinsics
        w2cs: [F, 4, 4] world-to-camera matrices (opencv convention)
        points: [N, 3] world-space point positions
        device: torch device
        h, w: render resolution
        render_radius: rasterization radius for point splatting
        points_per_pixel: number of points per pixel

    Returns:
        render_distances: [F, 1, H, W] spherical distance map
        render_masks: [F, 1, H, W] masks (1=invalid, 0=valid)
    """
    nframe = w2cs.shape[0]
    K = K.to(device)
    w2cs = w2cs.to(device)
    c2ws = w2cs.inverse()

    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    points = points.to(device)

    all_distances = []
    all_masks = []

    for fi in range(nframe):
        Rt = w2cs[fi]
        pos_cam = points @ Rt[:3, :3].T + Rt[:3, 3]
        distance = torch.norm(pos_cam, dim=1, keepdim=True)

        distance_3ch = distance.expand(-1, 3)

        point_cloud = Pointclouds(points=[points], features=[distance_3ch])

        c2w_fi = c2ws[fi:fi + 1].clone()
        c2w_fi[:, :, 0] = -c2w_fi[:, :, 0]
        c2w_fi[:, :, 1] = -c2w_fi[:, :, 1]
        w2c_fi = c2w_fi.inverse()

        K_fi = K[fi:fi + 1]
        focal_length = torch.stack([K_fi[:, 0, 0], K_fi[:, 1, 1]], dim=1)
        principal_point = torch.stack([K_fi[:, 0, 2], K_fi[:, 1, 2]], dim=1)
        image_shapes = torch.tensor([[h, w]], device=device)

        cameras = PerspectiveCameras(
            focal_length=focal_length, principal_point=principal_point,
            R=c2w_fi[:, :3, :3], T=w2c_fi[:, :3, -1],
            in_ndc=False, image_size=image_shapes, device=device
        )

        raster_settings = PointsRasterizationSettings(
            image_size=(h, w),
            radius=render_radius,
            points_per_pixel=points_per_pixel
        )

        renderer = PointsZbufRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=[0, 0, 0])
        )

        with suppress_stdout_stderr():
            rendered, zbuf = renderer(point_cloud)

        dist_map = rendered[0, :, :, 0]
        mask = (zbuf[0, :, :, 0] == -1).float()

        all_distances.append(dist_map)
        all_masks.append(mask)

    render_distances = torch.stack(all_distances, dim=0).unsqueeze(1)
    render_masks = torch.stack(all_masks, dim=0).unsqueeze(1)

    return render_distances, render_masks


def pcd_pano_render_color(points, colors, Rts, height, width, device,
                          render_radius=0.008, points_per_pixel=8,
                          render_depth=False, erp_mask_aware=False):
    """
    Render panorama from point cloud using pytorch3d point_rendering.
    Uses cubemap -> ERP stitching approach.

    Args:
        points: [N, 3] world-space points (torch.Tensor or np.ndarray)
        colors: [N, 3] point colors in [-1, 1]
        Rts: [F, 4, 4] w2c matrices (np.ndarray)
        height, width: panorama output size
        device: torch device
        render_radius: rasterization radius
        points_per_pixel: number of points per pixel
        render_depth: whether to also render depth
        erp_mask_aware: use mask-aware ERP stitching

    Returns:
        all_pano_rgb_map: list of [H, W, 3] numpy arrays (RGB in [0, 1])
        all_pano_depth_map: (only if render_depth) list of [H, W] numpy arrays
        all_pano_mask_map: list of [H, W] numpy bool arrays
    """
    mat = CUBEMAP_ROTATIONS
    tmp = CUBEMAP_GLUE_EXTRINSICS
    K_cube = CUBEMAP_INTRINSICS
    cube_size = CUBEMAP_SIZE

    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    if not isinstance(colors, torch.Tensor):
        colors = torch.tensor(colors, dtype=torch.float32)

    all_pano_rgb_map = []
    all_pano_depth_map = []
    all_pano_mask_map = []
    N = Rts.shape[0]

    for i in range(N):
        cur_Rt = Rts[i:i + 1]
        panorama_render_Rts = np.tile(cur_Rt, (6, 1, 1))
        panorama_render_Rts[:, :3, :] = mat @ panorama_render_Rts[:, :3, :]

        K_batch = torch.from_numpy(np.tile(K_cube, (6, 1, 1))).float()
        w2c_batch = torch.from_numpy(panorama_render_Rts).float()

        torch.cuda.synchronize(device)
        render_rgbs, render_masks = point_rendering(
            K=K_batch, w2cs=w2c_batch,
            points=points, colors=colors,
            device=device, h=cube_size, w=cube_size,
            background_color=[0, 0, 0],
            render_radius=render_radius,
            points_per_pixel=points_per_pixel,
            return_depth=False
        )
        torch.cuda.synchronize(device)

        if render_depth:
            render_dist_maps, render_dist_masks = point_rendering_distance(
                K=K_batch, w2cs=w2c_batch,
                points=points,
                device=device, h=cube_size, w=cube_size,
                render_radius=render_radius,
                points_per_pixel=points_per_pixel,
            )
            torch.cuda.synchronize(device)

        render_rgbs_np = ((render_rgbs.cpu().permute(0, 2, 3, 1).contiguous().numpy() + 1.) / 2.).clip(0, 1)
        render_rgbs_np = np.ascontiguousarray(render_rgbs_np)
        render_masks_bool = np.ascontiguousarray((render_masks[:, 0].cpu().numpy() < 0.5))

        render_rgbs_np[~render_masks_bool] = np.inf

        panorama_rgb = merge_panorama_color_torch(
            width, height,
            torch.from_numpy(render_rgbs_np).to(device),
            None,
            torch.from_numpy(tmp).to(device),
            torch.from_numpy(np.tile(K_cube, (6, 1, 1))).to(device),
            device,
            erp_mask_aware=erp_mask_aware
        )
        msk = panorama_rgb < torch.inf
        panorama_rgb[panorama_rgb == torch.inf] = 0.
        all_pano_rgb_map.append(panorama_rgb.cpu().numpy())
        all_pano_mask_map.append(msk[:, :, 0].cpu().numpy())

        if render_depth:
            render_dist_raw = render_dist_maps[:, 0]
            render_dist_masks_bool = (render_dist_masks[:, 0].cpu().numpy() < 0.5)

            render_depth_np = np.ascontiguousarray(render_dist_raw.cpu().numpy())
            render_depth_np[~render_dist_masks_bool] = np.inf

            depth_3ch = np.ascontiguousarray(
                np.stack([render_depth_np, render_depth_np, render_depth_np], axis=-1)
            )
            panorama_depth_3ch = merge_panorama_color_torch(
                width, height,
                torch.from_numpy(depth_3ch).to(device),
                None,
                torch.from_numpy(tmp).to(device),
                torch.from_numpy(np.tile(K_cube, (6, 1, 1))).to(device),
                device,
                erp_mask_aware=erp_mask_aware
            )
            panorama_depth = panorama_depth_3ch[:, :, 0]
            panorama_depth[panorama_depth == torch.inf] = 0.
            all_pano_depth_map.append(panorama_depth.cpu().numpy())

    if render_depth:
        return all_pano_rgb_map, all_pano_depth_map, all_pano_mask_map
    else:
        return all_pano_rgb_map, all_pano_mask_map


def pcd_pano_render_color_torch_renderer(points, colors, Rts, height, width, device,
                                          render_depth=False, erp_mask_aware=False,
                                          softmin=False, z_soft_alpha=500.0,
                                          use_fp16_cache=True):
    """
    Render panorama from point cloud using TorchPointCloudRenderer (pure PyTorch).
    Uses cubemap -> ERP stitching approach.

    Args:
        points: [N, 3] world-space points
        colors: [N, 3] point colors in [-1, 1]
        Rts: [F, 4, 4] w2c matrices (np.ndarray)
        height, width: panorama output size
        device: torch device
        render_depth: whether to also render depth
        erp_mask_aware: use mask-aware ERP stitching
        softmin: use softmin depth blending mode (default: False, uses fast mode)
        use_fp16_cache: store the point cloud as fp16 inside
            TorchPointCloudRenderer (default True, ~60 MB saved per 5M-pt
            scene). When the point cloud lives in absolute UE5 world
            coordinates (~10^2 m magnitude after `c2w(first_frame_Rt)`),
            fp16 has ~0.25 m precision and visibly quantises near-ground
            geometry (see docs/WARP_FP16_PRECISION_ROOT_CAUSE.md). Pass
            False to keep fp32 precision for absolute-pose cases.
        z_soft_alpha: softmin sharpness parameter (default: 500.0, lower = smoother)

    Returns:
        all_pano_rgb_map: list of [H, W, 3] numpy arrays (RGB in [0, 1])
        all_pano_depth_map: (only if render_depth) list of [H, W] numpy arrays
        all_pano_mask_map: list of [H, W] numpy bool arrays
    """
    mat = CUBEMAP_ROTATIONS
    tmp = CUBEMAP_GLUE_EXTRINSICS
    K_cube = CUBEMAP_INTRINSICS
    cube_size = CUBEMAP_SIZE

    if isinstance(points, torch.Tensor):
        pts_np = points.cpu().numpy().astype(np.float32)
    else:
        pts_np = np.asarray(points, dtype=np.float32)

    if isinstance(colors, torch.Tensor):
        clr_np = colors.cpu().numpy().astype(np.float32)
    else:
        clr_np = np.asarray(colors, dtype=np.float32)

    clr_np_01 = np.clip((clr_np + 1.0) / 2.0, 0.0, 1.0)

    render_mode = "softmin" if softmin else "fast"
    renderer = TorchPointCloudRenderer(
        points_xyz=pts_np,
        points_rgb=clr_np_01,
        width=cube_size,
        height=cube_size,
        device=str(device),
        mode=render_mode,
        z_soft_alpha=float(z_soft_alpha),
        use_fp16_cache=bool(use_fp16_cache),
    )

    # Pre-compute per-pixel Z-to-spherical-distance factor for cubemap faces.
    # TorchPointCloudRenderer returns Z-buffer depth (camera-space Z component),
    # but ERP panorama needs spherical Euclidean distance, consistent with
    # mesh_renderer and pytorch3d point_rendering_distance.
    # factor[v, u] = sqrt(((u - cx) / fx)^2 + ((v - cy) / fy)^2 + 1)
    fx, fy, cx, cy = K_cube[0, 0], K_cube[1, 1], K_cube[0, 2], K_cube[1, 2]
    v_coords, u_coords = np.meshgrid(
        np.arange(cube_size, dtype=np.float32),
        np.arange(cube_size, dtype=np.float32),
        indexing='ij'
    )
    z_to_dist_factor = np.sqrt(
        ((u_coords - cx) / fx) ** 2 + ((v_coords - cy) / fy) ** 2 + 1.0
    ).astype(np.float32)

    all_pano_rgb_map = []
    all_pano_depth_map = []
    all_pano_mask_map = []
    N = Rts.shape[0]

    for i in range(N):
        cur_Rt = Rts[i:i + 1]
        panorama_render_Rts = np.tile(cur_Rt, (6, 1, 1))
        panorama_render_Rts[:, :3, :] = mat @ panorama_render_Rts[:, :3, :]

        face_rgbs = np.zeros((6, cube_size, cube_size, 3), dtype=np.float32)
        face_masks = np.zeros((6, cube_size, cube_size), dtype=bool)
        face_depths = np.zeros((6, cube_size, cube_size), dtype=np.float32) if render_depth else None

        for face_idx in range(6):
            w2c_face = panorama_render_Rts[face_idx]
            c2w_face = np.linalg.inv(w2c_face).astype(np.float32)

            out = renderer.render(
                K_3x3=K_cube,
                c2w_4x4=c2w_face,
                c2w_is_camera_to_world=True,
            )
            rgb_f = out.rgb_u8.astype(np.float32) / 255.0
            valid = out.depth_f32 > 0

            face_rgbs[face_idx] = rgb_f
            face_masks[face_idx] = valid
            if render_depth:
                # Convert Z-buffer depth to spherical Euclidean distance
                face_depths[face_idx] = out.depth_f32 * z_to_dist_factor

        face_rgbs[~face_masks] = np.inf

        panorama_rgb = merge_panorama_color_torch(
            width, height,
            torch.from_numpy(face_rgbs).to(device),
            None,
            torch.from_numpy(tmp).to(device),
            torch.from_numpy(np.tile(K_cube, (6, 1, 1))).to(device),
            device,
            erp_mask_aware=erp_mask_aware
        )
        msk = panorama_rgb < torch.inf
        panorama_rgb[panorama_rgb == torch.inf] = 0.
        all_pano_rgb_map.append(panorama_rgb.cpu().numpy())
        all_pano_mask_map.append(msk[:, :, 0].cpu().numpy())

        if render_depth:
            face_depths[~face_masks] = np.inf
            depth_3ch = np.stack([face_depths, face_depths, face_depths], axis=-1)
            panorama_depth_3ch = merge_panorama_color_torch(
                width, height,
                torch.from_numpy(depth_3ch).to(device),
                None,
                torch.from_numpy(tmp).to(device),
                torch.from_numpy(np.tile(K_cube, (6, 1, 1))).to(device),
                device,
                erp_mask_aware=erp_mask_aware
            )
            panorama_depth = panorama_depth_3ch[:, :, 0]
            panorama_depth[panorama_depth == torch.inf] = 0.
            all_pano_depth_map.append(panorama_depth.cpu().numpy())

    if render_depth:
        return all_pano_rgb_map, all_pano_depth_map, all_pano_mask_map
    else:
        return all_pano_rgb_map, all_pano_mask_map
