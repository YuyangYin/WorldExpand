"""General-purpose utility functions for the nvrender package."""

import numpy as np
import torch
import skimage as ski


def image_uv(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = np.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, dtype=dtype)
    v = np.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=2)


def image_uv_torch(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def spherical_uv_to_directions_torch(uv):
    theta, phi = (1 - uv[..., 0]) * (2 * torch.pi), uv[..., 1] * torch.pi
    directions = torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], axis=-1)
    return directions


def find_mask_cc(mask):
    canvas = np.zeros_like(mask).astype(np.int32)
    canvas[mask] = 1
    labels, numbers = ski.measure.label(mask, background=None, return_num=True, connectivity=2)
    return labels, numbers


def depth_repair(depth, small_area_threshold=1600, max_depth_threshold=1000.0, use_clip=True):
    """
    Repair depth map.

    Args:
        depth: depth map
        small_area_threshold: small area threshold
        max_depth_threshold: max depth threshold, default 1000.0
        use_clip: whether to use clip mode (True) or 0.99 * max_depth mode (False)

    Returns:
        Repaired depth map
    """
    if use_clip:
        depth = np.clip(depth, 0, max_depth_threshold)
        max_depth = max_depth_threshold
    else:
        max_depth = depth.max()
        mask_valid = depth < 0.99 * max_depth

        labels, numbers = find_mask_cc(mask_valid)
        for i in range(numbers):
            cur_mask = labels == i + 1
            if cur_mask.sum() < small_area_threshold:
                depth[cur_mask] = max_depth

    return depth


def depth_edge_torch(depth: torch.Tensor, atol: float = None, rtol: float = None,
                     kernel_size: int = 3, mask=None) -> torch.Tensor:
    """
    Compute the edge mask from depth map.

    Args:
        depth: shape (..., height, width), linear depth map
        atol: absolute tolerance
        rtol: relative tolerance
        kernel_size: pooling kernel size
        mask: optional mask

    Returns:
        edge: shape (..., height, width) of dtype torch.bool
    """
    func = torch.nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    if mask is None:
        diff = (func(depth[None, None]) + func(-depth[None, None]))[0, 0]
    else:
        diff = (func(torch.where(mask, depth, -torch.inf)[None, None]) +
                func(torch.where(mask, -depth, -torch.inf)[None, None]))[0, 0]

    edge = torch.zeros_like(depth, dtype=torch.bool, device=depth.device)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def project_cv_torch(points, extrinsics, intrinsics):
    """
    Project 3D points to 2D using OpenCV convention.

    Args:
        points: [H, W, 3]
        extrinsics: [4, 4]
        intrinsics: [3, 3]

    Returns:
        points_pix: projected pixel coordinates
        depth: depth values
    """
    eps = 1e-5
    points_cam = points @ (extrinsics[:3, :3].T)[None] + extrinsics[:3, 3][None, None]
    points_pix = points_cam @ (intrinsics.T)[None]
    tmp = points_pix[:, :, 2:].clone()
    tmp_mask = torch.abs(points_pix[:, :, 2:]) < eps
    tmp[tmp_mask] = eps
    points_pix = points_pix / tmp
    return points_pix, tmp[..., 0]


def collect_perf_stats(timings, label, num_frames, gpu_device, num_points=None):
    """
    Collect and print performance statistics for a rendering run.

    Args:
        timings: list of per-frame times in seconds
        label: str, e.g. 'pytorch3d' or 'point_renderer'
        num_frames: int
        gpu_device: torch.device
        num_points: int or None
    """
    total_time = sum(timings)
    avg_time = total_time / max(len(timings), 1)
    fps = 1.0 / avg_time if avg_time > 0 else float('inf')

    mem_allocated = torch.cuda.memory_allocated(gpu_device) / (1024 ** 2)
    mem_reserved = torch.cuda.memory_reserved(gpu_device) / (1024 ** 2)
    mem_peak = torch.cuda.max_memory_allocated(gpu_device) / (1024 ** 2)

    # Silenced for tqdm progress bar compatibility — stats are returned as dict
    # To re-enable, set verbose=True or use the returned stats dict.

    stats = {
        'label': label,
        'num_frames': num_frames,
        'total_time_s': total_time,
        'avg_frame_ms': avg_time * 1000,
        'fps': fps,
        'gpu_mem_allocated_mb': mem_allocated,
        'gpu_mem_reserved_mb': mem_reserved,
        'gpu_mem_peak_mb': mem_peak,
    }
    if num_points is not None:
        stats['num_points'] = num_points
    return stats
