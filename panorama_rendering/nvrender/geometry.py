"""Point cloud spatial transforms and mesh construction from panoramic depth."""

import numpy as np
import torch
import trimesh

from .utils import image_uv, image_uv_torch, spherical_uv_to_directions_torch


def get_pano_pcs_torch(pano_depth):
    """
    Convert panoramic depth map to 3D points in camera space.

    Args:
        pano_depth: [H, W] depth tensor

    Returns:
        vertices: [H*W, 3] point cloud in camera space
    """
    device = pano_depth.device
    H, W = pano_depth.shape
    uv = image_uv_torch(width=W, height=H, device=device)
    directions = spherical_uv_to_directions_torch(uv)
    vertices_cam = pano_depth[:, :, None] * directions
    return vertices_cam.reshape((-1, 3))


def get_world_pcs_pano_torch_Rt(Rt, depth):
    """
    Convert panoramic depth map to world-space 3D points using Rt matrix.

    Args:
        Rt: [4, 4] w2c matrix (numpy array)
        depth: [H, W] depth tensor

    Returns:
        world_pcs: [H*W, 3] world-space point cloud
    """
    device = depth.device

    rot_matrix = torch.tensor([
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0.]
    ], dtype=depth.dtype, device=device)

    cam_pcs = get_pano_pcs_torch(depth) @ rot_matrix.T

    w2c = Rt
    c2w = torch.from_numpy(np.linalg.inv(w2c)).to(device).to(depth.dtype)
    world_pcs = cam_pcs @ c2w[:3, :3].T + c2w[:3, 3][None]
    return world_pcs


def get_mesh_from_pano_Rt(pano_depth, pano_mask, Rt, dtype=np.float32,
                          pano_rgb=None, alpha_mask=None,
                          save_pcd=False, pcd_path="debug_pointcloud.ply"):
    """
    Build a triangle mesh from panoramic depth+RGB.

    Args:
        pano_depth: [H, W] depth map (numpy)
        pano_mask: [H, W] bool mask (numpy)
        Rt: [4, 4] w2c matrix (numpy)
        dtype: output dtype
        pano_rgb: [H, W, 3] optional RGB (numpy, [0, 1])
        alpha_mask: [H, W] optional alpha mask (numpy bool)
        save_pcd: whether to also export a point cloud .ply
        pcd_path: path for the exported point cloud

    Returns:
        mesh_trimesh: trimesh.Trimesh object
    """
    H, W = pano_depth.shape
    uv = image_uv(width=W, height=H)
    pano_depth_ = torch.from_numpy(pano_depth).float()
    wrd_pcs = get_world_pcs_pano_torch_Rt(Rt, pano_depth_)
    vertices = wrd_pcs.reshape(H, W, 3).numpy()

    # Save point cloud if requested
    if save_pcd:
        vertices_flat = vertices.reshape(-1, 3)

        if pano_rgb is not None:
            colors_flat = pano_rgb.reshape(-1, 3)
            if colors_flat.max() <= 1.0:
                colors_flat = (colors_flat * 255).astype(np.uint8)
        else:
            colors_flat = np.ones((vertices_flat.shape[0], 3), dtype=np.uint8) * 128

        if alpha_mask is not None:
            alpha_mask_flat = alpha_mask.reshape(-1)
            vertices_flat = vertices_flat[alpha_mask_flat]
            colors_flat = colors_flat[alpha_mask_flat]

        point_cloud = trimesh.PointCloud(vertices=vertices_flat, colors=colors_flat)
        point_cloud.export(pcd_path)

    # Build mesh topology
    pts_topology = np.zeros((H - 1, W, 2, 3), dtype=np.int32)
    index_ = np.arange(H * W).astype(np.int32).reshape(H, W)
    pts_topology[:, :-1, 0, 0] = index_[:-1, :-1]
    pts_topology[:, :-1, 0, 2] = index_[:-1, 1:]
    pts_topology[:, :-1, 0, 1] = index_[1:, :-1]
    pts_topology[:, -1, 0, 0] = index_[:-1, -1]
    pts_topology[:, -1, 0, 2] = index_[:-1, 0]
    pts_topology[:, -1, 0, 1] = index_[1:, -1]

    pts_topology[:, :-1, 1, 0] = index_[:-1, 1:]
    pts_topology[:, :-1, 1, 1] = index_[1:, :-1]
    pts_topology[:, :-1, 1, 2] = index_[1:, 1:]
    pts_topology[:, -1, 1, 0] = index_[:-1, 0]
    pts_topology[:, -1, 1, 1] = index_[1:, -1]
    pts_topology[:, -1, 1, 2] = index_[1:, 0]

    pano_mask_ext = np.zeros((H, W + 1), dtype=np.bool_)
    pano_mask_ext[:, :-1] = pano_mask[:, :]
    pano_mask_ext[:, -1] = pano_mask[:, 0]

    topo_mask_0 = pano_mask_ext[:-1, :-1] + pano_mask_ext[:-1, 1:] + pano_mask_ext[1:, :-1]
    topo_mask_1 = pano_mask_ext[:-1, 1:] + pano_mask_ext[1:, :-1] + pano_mask_ext[1:, 1:]

    topo0 = pts_topology[:, :, 0][topo_mask_0]
    topo1 = pts_topology[:, :, 1][topo_mask_1]

    topo = np.concatenate([topo0, topo1], axis=0)

    vertices = vertices.reshape(-1, 3)
    if pano_rgb is None and alpha_mask is None:
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo)
    elif alpha_mask is None:
        pano_rgb = pano_rgb.reshape(-1, 3)
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo, vertex_colors=pano_rgb)
    elif alpha_mask is not None:
        if pano_rgb is not None:
            pano_rgb = pano_rgb.reshape(-1, 3)
        else:
            pano_rgb = np.ones((H * W, 3), dtype=np.float32)
        pano_rgb_ = np.ones((H * W, 4), dtype=np.float32)
        pano_rgb_[:, :3] = pano_rgb
        alpha_mask = alpha_mask.reshape(-1)
        pano_rgb_[~alpha_mask] = 0.
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo, vertex_colors=pano_rgb_)
    return mesh_trimesh
