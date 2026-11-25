"""Mesh-based panorama rendering using nvdiffrast."""

import numpy as np
import torch
import nvdiffrast.torch as dr

from .cubemap import (
    CUBEMAP_ROTATIONS, CUBEMAP_GLUE_EXTRINSICS, CUBEMAP_INTRINSICS,
    merge_panorama_color_torch,
)


def get_diffrast_camera_parameter_from_cv(K, H, W, near, far, device):
    """
    Convert OpenCV intrinsics to nvdiffrast projection matrix.

    Args:
        K: [3, 3] intrinsics
        H, W: render resolution
        near, far: clipping planes

    Returns:
        K_: [4, 4] projection matrix
    """
    K_ = torch.zeros((4, 4), dtype=torch.float32, device=device)
    K_[0, 0] = K[0, 0] * 2. / W
    K_[1, 1] = K[1, 1] * 2. / H
    K_[2, 2] = (far + near) / (far - near)
    K_[2, 3] = -2. * (near * far) / (far - near)
    K_[3, 2] = 1.
    return K_


def get_mesh_render_color(mesh, Ks, Rts, H, W, near, far, device):
    """
    Render mesh colour images for given camera parameters using nvdiffrast.

    Args:
        mesh: trimesh.Trimesh object with vertex colours
        Ks: [N, 3, 3] intrinsics (numpy or tensor)
        Rts: [N, 4, 4] w2c matrices (numpy or tensor)
        H, W: render resolution
        near, far: clipping planes
        device: torch device

    Returns:
        all_canvas: [N, H, W, 3] rendered colour images
        all_canvas_mask: [N, H, W] bool masks (True = valid)
    """
    with torch.no_grad():
        if isinstance(Ks, np.ndarray):
            Ks = torch.from_numpy(Ks).float().to(device)
            Rts = torch.from_numpy(Rts).float().to(device)
        glctx = dr.RasterizeCudaContext(device=device)
        pos_tensor = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
        col_tensor = torch.from_numpy(np.array(mesh.visual.vertex_colors)).float().to(device)
        tri_tensor = torch.from_numpy(np.array(mesh.faces)).int().to(device)
        if col_tensor.max() > 5:
            col_tensor /= 255.
        pos_qc = torch.ones((pos_tensor.shape[0], 4), dtype=torch.float32, device=device)

        N = Rts.shape[0]
        all_images = []

        for i in range(N):
            pos_tensor_cam = pos_tensor @ Rts[i, :3, :3].T + Rts[i:i + 1, :3, 3]
            K_ = get_diffrast_camera_parameter_from_cv(Ks[i], H, W, near, far, device)
            pos_qc[:, :3] = pos_tensor_cam
            pos_rast = (pos_qc @ K_.T)[None]
            rast, _ = dr.rasterize(glctx, pos_rast, tri_tensor, resolution=[H, W])
            out, _ = dr.interpolate(col_tensor[None], rast, tri_tensor)
            img = out.cpu().numpy()[0, :, :, :]
            all_images.append(img)
        all_images = np.stack(all_images, axis=0)
        all_canvas = all_images[:, :, :, :3]
        all_canvas_mask = all_images[:, :, :, 3] > 0.999

        return all_canvas, all_canvas_mask


def mesh_pano_render_color(mesh, Rts, height, width, near, far, device):
    """
    Render RGB panorama from mesh using cubemap -> ERP stitching.

    Args:
        mesh: trimesh.Trimesh with vertex colours
        Rts: [N, 4, 4] w2c matrices (numpy)
        height, width: panorama output size
        near, far: clipping planes
        device: torch device

    Returns:
        all_pano_depth_map: list of [H, W, 3] numpy arrays (RGB)
        all_pano_mask_map: list of [H, W] numpy bool arrays
    """
    mat = CUBEMAP_ROTATIONS
    tmp = CUBEMAP_GLUE_EXTRINSICS
    K = CUBEMAP_INTRINSICS

    all_pano_depth_map = []
    all_pano_mask_map = []
    N = Rts.shape[0]

    render_batch_size = 5
    for i in range(0, N, render_batch_size):
        cur_process_size = min(render_batch_size, N - i)
        cur_Rt = Rts[i:i + cur_process_size][:, None, :, :]
        panorama_render_Rts = cur_Rt.repeat(6, 1)
        panorama_render_Rts[:, :, :3, :] = mat[None] @ panorama_render_Rts[:, :, :3, :]
        render_color, render_mask = get_mesh_render_color(
            mesh, K[None].repeat(6 * cur_process_size, 0),
            panorama_render_Rts.reshape(cur_process_size * 6, 4, 4),
            512, 512, near, far, device
        )
        render_color[~render_mask] = torch.inf

        for j in range(cur_process_size):
            panorama_log_distance_map = merge_panorama_color_torch(
                width, height,
                torch.from_numpy(render_color[6 * j:6 * (j + 1)]).to(device),
                None,
                torch.from_numpy(tmp).to(device),
                torch.from_numpy(K[None].repeat(6 * render_batch_size, 0)).to(device),
                device
            )
            msk = panorama_log_distance_map < torch.inf
            panorama_log_distance_map[panorama_log_distance_map == torch.inf] = 0.
            all_pano_depth_map.append(panorama_log_distance_map.cpu().numpy())
            all_pano_mask_map.append(msk[:, :, 0].cpu().numpy())

    return all_pano_depth_map, all_pano_mask_map


def mesh_pano_render_depth(mesh, Rts, height, width, near, far, device):
    """
    Render depth-only panorama from mesh using cubemap -> ERP stitching.
    Uses spherical (Euclidean) distance from camera centre.

    Returns:
        all_pano_depth_map: list of [H, W] numpy arrays
        all_pano_mask_map: list of [H, W] numpy bool arrays
    """
    mat = CUBEMAP_ROTATIONS
    tmp = CUBEMAP_GLUE_EXTRINSICS
    K = CUBEMAP_INTRINSICS

    all_pano_depth_map = []
    all_pano_mask_map = []
    N = Rts.shape[0]
    render_batch_size = 5

    with torch.no_grad():
        glctx = dr.RasterizeCudaContext(device=device)
        pos_tensor = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
        tri_tensor = torch.from_numpy(np.array(mesh.faces)).int().to(device)

        for i in range(0, N, render_batch_size):
            cur_process_size = min(render_batch_size, N - i)
            cur_Rt = Rts[i:i + cur_process_size][:, None, :, :]
            panorama_render_Rts = cur_Rt.repeat(6, 1)
            panorama_render_Rts[:, :, :3, :] = mat[None] @ panorama_render_Rts[:, :, :3, :]

            render_depths = []
            render_masks = []

            for j in range(cur_process_size * 6):
                Rt = torch.from_numpy(
                    panorama_render_Rts.reshape(cur_process_size * 6, 4, 4)[j]
                ).float().to(device)
                K_torch = torch.from_numpy(K).float().to(device)

                pos_tensor_cam = pos_tensor @ Rt[:3, :3].T + Rt[:3, 3]
                distance_tensor = torch.norm(pos_tensor_cam, dim=1, keepdim=True)

                K_ = get_diffrast_camera_parameter_from_cv(K_torch, 512, 512, near, far, device)
                pos_qc = torch.ones((pos_tensor.shape[0], 4), dtype=torch.float32, device=device)
                pos_qc[:, :3] = pos_tensor_cam
                pos_rast = (pos_qc @ K_.T)[None]

                rast, _ = dr.rasterize(glctx, pos_rast, tri_tensor, resolution=[512, 512])
                distance_interpolated, _ = dr.interpolate(distance_tensor[None], rast, tri_tensor)
                distance = distance_interpolated[0, :, :, 0]
                mask = rast[0, :, :, 3] > 0

                render_depths.append(distance.cpu().numpy())
                render_masks.append(mask.cpu().numpy())

            for j in range(cur_process_size):
                depth_cubes = np.stack(render_depths[6 * j:6 * (j + 1)], axis=0)
                mask_cubes = np.stack(render_masks[6 * j:6 * (j + 1)], axis=0)

                depth_cubes_3ch = np.stack([depth_cubes, depth_cubes, depth_cubes], axis=-1)
                panorama_depth = merge_panorama_color_torch(
                    width, height,
                    torch.from_numpy(depth_cubes_3ch).to(device),
                    None,
                    torch.from_numpy(tmp).to(device),
                    torch.from_numpy(K[None].repeat(6, 0)).to(device),
                    device
                )

                mask_cubes_3ch = np.stack([mask_cubes, mask_cubes, mask_cubes], axis=-1).astype(np.float32)
                panorama_mask = merge_panorama_color_torch(
                    width, height,
                    torch.from_numpy(mask_cubes_3ch).to(device),
                    None,
                    torch.from_numpy(tmp).to(device),
                    torch.from_numpy(K[None].repeat(6, 0)).to(device),
                    device
                )

                depth_final = panorama_depth[:, :, 0].cpu().numpy()
                mask_final = (panorama_mask[:, :, 0] > 0.5).cpu().numpy()

                all_pano_depth_map.append(depth_final)
                all_pano_mask_map.append(mask_final)

    return all_pano_depth_map, all_pano_mask_map


def mesh_pano_render_rgb_and_depth(mesh, Rts, height, width, near, far, device):
    """
    Render both RGB and depth panorama simultaneously (avoids double rendering).

    Returns:
        all_pano_rgb_map: list of [H, W, 3] numpy arrays
        all_pano_depth_map: list of [H, W] numpy arrays
        all_pano_mask_map: list of [H, W] numpy bool arrays
    """
    mat = CUBEMAP_ROTATIONS
    tmp = CUBEMAP_GLUE_EXTRINSICS
    K = CUBEMAP_INTRINSICS

    all_pano_rgb_map = []
    all_pano_depth_map = []
    all_pano_mask_map = []
    N = Rts.shape[0]
    render_batch_size = 5

    with torch.no_grad():
        glctx = dr.RasterizeCudaContext(device=device)
        pos_tensor = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
        col_tensor = torch.from_numpy(np.array(mesh.visual.vertex_colors)).float().to(device)
        tri_tensor = torch.from_numpy(np.array(mesh.faces)).int().to(device)

        if col_tensor.max() > 5:
            col_tensor /= 255.

        for i in range(0, N, render_batch_size):
            cur_process_size = min(render_batch_size, N - i)
            cur_Rt = Rts[i:i + cur_process_size][:, None, :, :]
            panorama_render_Rts = cur_Rt.repeat(6, 1)
            panorama_render_Rts[:, :, :3, :] = mat[None] @ panorama_render_Rts[:, :, :3, :]

            render_rgbs = []
            render_depths = []
            render_masks = []

            for j in range(cur_process_size * 6):
                Rt = torch.from_numpy(
                    panorama_render_Rts.reshape(cur_process_size * 6, 4, 4)[j]
                ).float().to(device)
                K_torch = torch.from_numpy(K).float().to(device)

                pos_tensor_cam = pos_tensor @ Rt[:3, :3].T + Rt[:3, 3]
                distance_tensor = torch.norm(pos_tensor_cam, dim=1, keepdim=True)

                K_ = get_diffrast_camera_parameter_from_cv(K_torch, 512, 512, near, far, device)
                pos_qc = torch.ones((pos_tensor.shape[0], 4), dtype=torch.float32, device=device)
                pos_qc[:, :3] = pos_tensor_cam
                pos_rast = (pos_qc @ K_.T)[None]

                rast, _ = dr.rasterize(glctx, pos_rast, tri_tensor, resolution=[512, 512])

                rgb_interpolated, _ = dr.interpolate(col_tensor[None], rast, tri_tensor)
                rgb = rgb_interpolated[0, :, :, :3]

                distance_interpolated, _ = dr.interpolate(distance_tensor[None], rast, tri_tensor)
                distance = distance_interpolated[0, :, :, 0]

                mask = rast[0, :, :, 3] > 0

                render_rgbs.append(rgb.cpu().numpy())
                render_depths.append(distance.cpu().numpy())
                render_masks.append(mask.cpu().numpy())

            for j in range(cur_process_size):
                rgb_cubes = np.stack(render_rgbs[6 * j:6 * (j + 1)], axis=0)
                depth_cubes = np.stack(render_depths[6 * j:6 * (j + 1)], axis=0)
                mask_cubes = np.stack(render_masks[6 * j:6 * (j + 1)], axis=0)

                # Stitch RGB
                panorama_rgb = merge_panorama_color_torch(
                    width, height,
                    torch.from_numpy(rgb_cubes).to(device),
                    None,
                    torch.from_numpy(tmp).to(device),
                    torch.from_numpy(K[None].repeat(6, 0)).to(device),
                    device
                )

                # Stitch depth
                depth_cubes_3ch = np.stack([depth_cubes, depth_cubes, depth_cubes], axis=-1)
                panorama_depth = merge_panorama_color_torch(
                    width, height,
                    torch.from_numpy(depth_cubes_3ch).to(device),
                    None,
                    torch.from_numpy(tmp).to(device),
                    torch.from_numpy(K[None].repeat(6, 0)).to(device),
                    device
                )

                # Stitch mask
                mask_cubes_3ch = np.stack([mask_cubes, mask_cubes, mask_cubes], axis=-1).astype(np.float32)
                panorama_mask = merge_panorama_color_torch(
                    width, height,
                    torch.from_numpy(mask_cubes_3ch).to(device),
                    None,
                    torch.from_numpy(tmp).to(device),
                    torch.from_numpy(K[None].repeat(6, 0)).to(device),
                    device
                )

                rgb_final = panorama_rgb.cpu().numpy()
                depth_final = panorama_depth[:, :, 0].cpu().numpy()
                mask_final = (panorama_mask[:, :, 0] > 0.5).cpu().numpy()

                all_pano_rgb_map.append(rgb_final)
                all_pano_depth_map.append(depth_final)
                all_pano_mask_map.append(mask_final)

    return all_pano_rgb_map, all_pano_depth_map, all_pano_mask_map
