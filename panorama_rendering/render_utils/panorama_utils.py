import numpy as np
import utils3d
import cv2
import torch
from PIL import Image
import trimesh
import open3d as o3d
from typing import Optional, Literal

from moge.utils.panorama import (
    get_panorama_cameras,
    merge_panorama_depth,
    List, convolve, vstack, lsmr, grad_equation, poisson_equation
)
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import eye, csr_array
#import cupy as cp
# from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
# from cupyx.scipy.sparse.linalg import lsmr as cp_lsmr

def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions

def subdivide_icosahedron(subdivisions: int = 1) -> np.ndarray:
    """
    细分正二十面体，生成更密集的球面采样点

    Args:
        subdivisions: 细分次数
            - 0: 12 顶点
            - 1: 42 顶点
            - 2: 162 顶点
            - 3: 642 顶点
            公式: V = 10 * 4^n + 2

    Returns:
        vertices: (N, 3) 细分后的顶点坐标（单位球面上）
    """
    vertices, faces = utils3d.numpy.icosahedron()

    # 转换为列表便于动态添加
    vertices_list = [v for v in vertices]

    for _ in range(subdivisions):
        edge_midpoint_cache = {}  # 缓存边的中点索引，避免重复创建
        new_faces = []

        def get_or_create_midpoint(idx1: int, idx2: int) -> int:
            """
            获取两个顶点之间边的中点索引
            如果中点不存在则创建，存在则直接返回缓存的索引
            """
            # 使用排序后的元组作为key，确保 (a,b) 和 (b,a) 是同一条边
            edge_key = (min(idx1, idx2), max(idx1, idx2))

            if edge_key in edge_midpoint_cache:
                return edge_midpoint_cache[edge_key]

            # 创建新的中点
            v1 = vertices_list[idx1]
            v2 = vertices_list[idx2]
            midpoint = (v1 + v2) / 2.0

            # 投影到单位球面
            midpoint = midpoint / np.linalg.norm(midpoint)

            # 添加到顶点列表
            new_idx = len(vertices_list)
            vertices_list.append(midpoint)
            edge_midpoint_cache[edge_key] = new_idx

            return new_idx

        # 对每个三角形进行细分
        for face in faces:
            v0, v1, v2 = face

            # 获取三条边的中点
            #       v0
            #      /  \
            #     a----c
            #    / \  / \
            #   v1--b----v2
            a = get_or_create_midpoint(v0, v1)
            b = get_or_create_midpoint(v1, v2)
            c = get_or_create_midpoint(v2, v0)

            # 将原三角形细分为4个小三角形
            new_faces.append([v0, a, c])
            new_faces.append([a, v1, b])
            new_faces.append([c, b, v2])
            new_faces.append([a, b, c])

        faces = np.array(new_faces, dtype=np.int32)

    return np.array(vertices_list, dtype=np.float32)


def get_panorama_cameras_v2(subdivisions=0):
    vertices = subdivide_icosahedron(subdivisions=subdivisions)
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def rotate_around_z_axis(points, angle_deg):
    """
    绕Z轴顺时针旋转三维点云

    参数:
        points: 三维点数组，形状为(N, 3)，每行代表一个点(x, y, z)
        angle_deg: 旋转角度（度），正值表示顺时针旋转

    返回:
        rotated_points: 旋转后的三维点数组，形状为(N, 3)
    """
    # 将角度转换为弧度
    angle_rad = np.radians(angle_deg)

    # 构建绕Z轴顺时针旋转的旋转矩阵
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # 顺时针旋转矩阵（右手坐标系中，顺时针旋转相当于负角度的逆时针旋转）
    rotation_matrix = np.array([
        [cos_theta, sin_theta, 0],  # X分量
        [-sin_theta, cos_theta, 0],  # Y分量
        [0, 0, 1]  # Z分量（不变）
    ])

    # 对每个点应用旋转（矩阵乘法）
    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, h: int, w: int, interp):
    height, width = image.shape[:2]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)

        splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=interp, borderMode=cv2.BORDER_WRAP)
        splitted_images.append(splitted_image)
    return splitted_images


def split_panorama_depth(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, h: int, w: int, distance_to_depth=False):
    height, width = depth.shape[:2]
    depth = torch.tensor(depth, dtype=torch.float32)[None, None]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    splitted_depths = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)
        pixels = torch.tensor(pixels, dtype=torch.float32)[None, ...]  # [1,h,w,2]
        pixels[..., 0] /= width
        pixels[..., 1] /= height
        pixels = pixels * 2 - 1.0
        splitted_depth = F.grid_sample(depth, grid=pixels, mode="nearest", align_corners=True)

        if distance_to_depth:
            fx = intrinsics[i][0, 0] * w
            fy = intrinsics[i][1, 1] * h
            cx = intrinsics[i][0, 2] * w
            cy = intrinsics[i][1, 2] * h
            x_cam = (u_grid - cx) / fx
            y_cam = (v_grid - cy) / fy
            z_cam = np.ones_like(x_cam)
            rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)  # (H, W, 3)
            ray_length = np.linalg.norm(rays_cam, axis=-1).astype(np.float32)
            splitted_depth = splitted_depth * (z_cam[None, None] / ray_length[None, None])

        splitted_depths.append(splitted_depth)
    return torch.cat(splitted_depths, dim=0).float()


def smooth_south_pole_depth(depth_map, smooth_height_ratio=0.03):
    """
    平滑全景图南极点（底部区域）的深度，解决左右不一致问题

    Args:
        depth_map: 深度图 (H, W)
        smooth_height_ratio: 平滑区域高度比例，默认0.03表示从底部向上0.03*H的区域

    Returns:
        平滑后的深度图
    """
    height, width = depth_map.shape
    smooth_height = int(height * smooth_height_ratio)

    if smooth_height == 0:
        return depth_map

    # 复制深度图避免修改原图
    smoothed_depth = depth_map.copy()

    # 计算参考深度：当行数大于3时使用最后3行，否则使用最底部一行
    if smooth_height > 3:
        # 使用最后3行计算平均深度
        reference_rows = depth_map[-3:, :]
        reference_data = reference_rows.flatten()
    else:
        # 使用最底部一行
        reference_data = depth_map[-1, :]

    # 过滤异常值：包括无效值、过大或过小的深度
    valid_mask = np.isfinite(reference_data) & (reference_data > 0)

    if np.any(valid_mask):
        valid_depths = reference_data[valid_mask]

        # 使用四分位数方法过滤极端异常值
        lower_bound, upper_bound = np.quantile(valid_depths, [0.1, 0.9])

        # 进一步过滤过大或过小的深度值
        depth_filter_mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)

        if np.any(depth_filter_mask):
            avg_depth = np.mean(valid_depths[depth_filter_mask])
        else:
            # 如果所有值都被过滤掉，使用中位数作为备选
            avg_depth = np.median(valid_depths)
    else:
        avg_depth = np.nanmean(reference_data)

    # 设置最底部一行为平均值
    smoothed_depth[-1, :] = avg_depth

    # 向上平滑到指定高度
    for i in range(1, smooth_height):
        y_idx = height - 1 - i  # 从底部向上的索引
        if y_idx < 0:
            break

        # 计算平滑权重：离底部越近，平滑程度越强
        weight = (smooth_height - i) / smooth_height

        # 对当前行进行平滑
        current_row = depth_map[y_idx, :]
        valid_mask = np.isfinite(current_row) & (current_row > 0)

        if np.any(valid_mask):
            valid_row_depths = current_row[valid_mask]

            # 对当前行也应用异常值过滤
            if len(valid_row_depths) > 1:
                q25, q75 = np.quantile(valid_row_depths, [0.25, 0.75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                depth_filter_mask = (valid_row_depths >= lower_bound) & (valid_row_depths <= upper_bound)

                if np.any(depth_filter_mask):
                    row_avg = np.mean(valid_row_depths[depth_filter_mask])
                else:
                    row_avg = np.median(valid_row_depths)
            else:
                row_avg = valid_row_depths[0] if len(valid_row_depths) > 0 else avg_depth

            # 线性插值：原始深度和平均深度之间
            smoothed_depth[y_idx, :] = (1 - weight) * current_row + weight * row_avg

    return smoothed_depth


# def solve_lsmr_gpu(A, b, x0=None):
#     """GPU 加速的 LSMR"""
#     # 转换到 GPU
#     A_gpu = cp_csr_matrix(A)
#     b_gpu = cp.asarray(b)
#     x0_gpu = cp.asarray(x0) if x0 is not None else None

#     # GPU 上求解
#     x_gpu, *_ = cp_lsmr(A_gpu, b_gpu, atol=1e-5, btol=1e-5, x0=x0_gpu)

#     # 转回 CPU
#     return cp.asnumpy(x_gpu)


# def merge_panorama_depth_gpu(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray]):
#     if max(width, height) > 256:
#         panorama_depth_init, _ = merge_panorama_depth_gpu(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics)
#         panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
#     else:
#         panorama_depth_init = None

#     uv = utils3d.numpy.image_uv(width=width, height=height)
#     spherical_directions = spherical_uv_to_directions(uv)  # [h,w,3]

#     # Warp each view to the panorama
#     panorama_log_distance_grad_maps, panorama_grad_masks = [], []
#     panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
#     panorama_pred_masks = []
#     for i in range(len(distance_maps)):
#         projected_uv, projected_depth = utils3d.numpy.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
#         projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)

#         projected_pixels = utils3d.numpy.uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)

#         log_splitted_distance = np.log(distance_maps[i])
#         panorama_log_distance_map = np.where(projection_valid_mask,
#                                              cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
#         panorama_pred_mask = projection_valid_mask & (
#                 cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

#         # calculate gradient map
#         padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
#         grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

#         padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
#         mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]

#         panorama_log_distance_grad_maps.append((grad_x, grad_y))
#         panorama_grad_masks.append((mask_x, mask_y))

#         # calculate laplacian map
#         padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
#         padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
#         laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

#         padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
#         padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
#         mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

#         panorama_log_distance_laplacian_maps.append(laplacian)
#         panorama_laplacian_masks.append(mask)

#         panorama_pred_masks.append(panorama_pred_mask)

#     panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
#     panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
#     panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
#     panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

#     panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
#     panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

#     panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
#     panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
#     panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

#     grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
#     grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
#     grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
#     laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

#     # Solve overdetermined system
#     A = vstack([
#         grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
#         poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
#     ])
#     b = np.concatenate([
#         panorama_log_distance_grad_x.reshape(-1)[grad_x_mask],
#         panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
#         panorama_laplacian_map.reshape(-1)[laplacian_mask]
#     ])
#     x = solve_lsmr_gpu(A, b, x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None)

#     panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
#     panorama_mask = np.any(panorama_pred_masks, axis=0)

#     return panorama_depth, panorama_mask


def pred_pano_depth(model, image: Image.Image, scale=1.0, resize_to=1920, remove_pano_depth_nan=True):
    """
    last_layer_mask: 前一层物体mask
    last_layer_depth: 前一层深度
    """
    print("\t - Predicting pano depth with moge")
    image_origin = np.array(image)
    height_origin, width_origin = image_origin.shape[:2]

    image, height, width = image_origin, height_origin, width_origin
    if resize_to is not None:
        _height, _width = min(resize_to, int(resize_to * height_origin / width_origin)), min(resize_to, int(resize_to * width_origin / height_origin))
        if _height < height_origin:
            print(f"\t - Resizing image from {width_origin}x{height_origin} to {_width}x{_height} for pano depth prediction")
            image = cv2.resize(image_origin, (_width, _height), cv2.INTER_AREA)
            height, width = _height, _width

    splitted_extrinsics, splitted_intriniscs = get_panorama_cameras_v2(subdivisions=1)
    splitted_resolution = 512
    splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution, splitted_resolution, interp=cv2.INTER_LINEAR)

    # infer moge depth
    num_splitted_images = len(splitted_images)
    splitted_distance_maps = [None] * num_splitted_images
    splitted_masks = [None] * num_splitted_images

    indices_to_process_model = []
    skipped_count = 0

    for i in range(num_splitted_images):
        indices_to_process_model.append(i)

    pred_count = 0
    # Process images that require model inference in batches
    inference_batch_size = 1
    for i in range(0, len(indices_to_process_model), inference_batch_size):
        batch_indices = indices_to_process_model[i: i + inference_batch_size]
        if not batch_indices:
            continue

        current_batch_images = [splitted_images[k] for k in batch_indices]
        current_batch_intrinsics = [splitted_intriniscs[k] for k in batch_indices]

        image_tensor = torch.tensor(
            np.stack(current_batch_images) / 255,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        ).permute(0, 3, 1, 2)

        fov_x, _ = np.rad2deg(  # fov_y is not used by model.infer
            utils3d.numpy.intrinsics_to_fov(np.array(current_batch_intrinsics))
        )
        fov_x_tensor = torch.tensor(
            fov_x, dtype=torch.float32, device=next(model.parameters()).device
        )

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            output = model.infer(image_tensor, fov_x=fov_x_tensor, apply_mask=False)

        batch_distance_maps = output["points"].norm(dim=-1).cpu().numpy()
        batch_masks = output["mask"].cpu().numpy()

        for batch_idx, original_idx in enumerate(batch_indices):
            splitted_distance_maps[original_idx] = batch_distance_maps[batch_idx]
            splitted_masks[original_idx] = batch_masks[batch_idx]
            pred_count += 1

    if (pred_count + skipped_count) == 0:  # Avoid division by zero if num_splitted_images is 0
        skip_ratio_info = "N/A (no images to process)"
    else:
        skip_ratio_info = f"{skipped_count / (pred_count + skipped_count):.2%}"

    print(f"\t 🔍 Predicted {pred_count} splitted images, skipped {skipped_count} splitted images. Skip ratio: {skip_ratio_info}")

    # merge moge depth
    merging_width, merging_height = width, height
    panorama_depth, panorama_mask = merge_panorama_depth_gpu(
        merging_width,
        merging_height,
        splitted_distance_maps,
        splitted_masks,
        splitted_extrinsics,
        splitted_intriniscs,
    )

    panorama_depth = panorama_depth.astype(np.float32)
    # 对齐 dano depth最底部0.03区域左右侧深度
    if remove_pano_depth_nan:
        # for depth inpainting, remove nan
        panorama_depth[~panorama_mask] = 1.0 * np.nanquantile(panorama_depth, 0.999)  # 天空深度
    panorama_depth = cv2.resize(panorama_depth, (width_origin, height_origin), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width_origin, height_origin), cv2.INTER_NEAREST) > 0

    # 平滑南极点（底部区域）深度，解决左右不一致问题
    print("\t - Smoothing south pole depth for consistency")
    panorama_depth = smooth_south_pole_depth(panorama_depth, smooth_height_ratio=0.05)

    rays = torch.from_numpy(spherical_uv_to_directions(utils3d.numpy.image_uv(width=width_origin, height=height_origin))).to(next(model.parameters()).device)

    panorama_depth = (
            torch.from_numpy(panorama_depth).to(next(model.parameters()).device) * scale
    )

    return {
        "rgb": torch.from_numpy(image_origin).to(next(model.parameters()).device),
        "distance": panorama_depth,
        "rays": rays,
        "mask": panorama_mask,
        "splitted_masks": splitted_masks,
        "splitted_distance_maps": splitted_distance_maps,
    }


# 基于normal约束的全景深度拼接
def compute_spherical_ray_derivatives(spherical_directions: np.ndarray):
    """
    计算全景图射线方向对球面坐标的偏导

    Args:
        spherical_directions: (H, W, 3) 全景图每个像素的射线方向（世界坐标系，单位向量）

    Returns:
        dray_dtheta: (H, W, 3) 射线对方位角 θ 的偏导
        dray_dphi: (H, W, 3) 射线对仰角 φ 的偏导
    """
    H, W, _ = spherical_directions.shape

    # 从射线方向反推球面坐标
    # 假设坐标系: x-right, y-up, z-forward (或其他，需要根据实际情况调整)
    ray = spherical_directions

    # 方位角 θ 和仰角 φ
    # ray = (cos(φ)sin(θ), sin(φ), cos(φ)cos(θ))
    # 或者根据你的坐标系定义

    # 方法1: 数值计算（更稳健）
    u = (np.arange(W) + 0.5) / W  # [0, 1]
    v = (np.arange(H) + 0.5) / H  # [0, 1]
    u_grid, v_grid = np.meshgrid(u, v)

    theta = (u_grid - 0.5) * 2 * np.pi  # [-π, π]
    phi = (0.5 - v_grid) * np.pi  # [π/2, -π/2]

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # ∂ray/∂θ（假设标准 equirectangular 映射）
    # 需要根据你的 spherical_directions 的具体计算方式调整
    dray_dtheta = np.stack([
        cos_phi * cos_theta,
        np.zeros_like(theta),
        -cos_phi * sin_theta
    ], axis=-1)

    # ∂ray/∂φ
    dray_dphi = np.stack([
        -sin_phi * sin_theta,
        cos_phi,
        -sin_phi * cos_theta
    ], axis=-1)

    return dray_dtheta, dray_dphi


def normal_to_log_distance_gradient(
        panorama_normal: np.ndarray,
        spherical_directions: np.ndarray,
        width: int,
        height: int
) -> tuple:
    """
    从全景 normal 计算 log distance 的梯度

    数学推导:
        表面点 P = d * ray
        切向量 ∂P/∂θ = (∂d/∂θ) * ray + d * (∂ray/∂θ)
        法线垂直于切向量: n · ∂P/∂θ = 0
        => ∂(log d)/∂θ = -(n · ∂ray/∂θ) / (n · ray)

    Args:
        panorama_normal: (H, W, 3) 全景法线图（世界坐标系）
        spherical_directions: (H, W, 3) 射线方向（世界坐标系）
        width, height: 全景图尺寸

    Returns:
        grad_x: (H, W) x方向梯度 (对应 log_d[j] - log_d[j+1])
        grad_y: (H-1, W) y方向梯度 (对应 log_d[i] - log_d[i+1])
        valid_mask: (H, W) 有效区域
    """
    H, W = height, width

    # 获取射线方向的偏导
    dray_dtheta, dray_dphi = compute_spherical_ray_derivatives(spherical_directions)

    # 计算点积
    n_dot_ray = np.sum(panorama_normal * spherical_directions, axis=-1)  # (H, W)
    n_dot_dray_dtheta = np.sum(panorama_normal * dray_dtheta, axis=-1)  # (H, W)
    n_dot_dray_dphi = np.sum(panorama_normal * dray_dphi, axis=-1)  # (H, W)

    # 有效性检查（法线不能垂直于视线）
    eps = 1e-4
    valid_mask = np.abs(n_dot_ray) > eps

    # 安全除法
    n_dot_ray_safe = np.where(valid_mask, n_dot_ray, 1.0)

    # 连续空间的 log(d) 梯度（相对于角度）
    # ∂(log d)/∂θ = -(n · ∂ray/∂θ) / (n · ray)
    dlogd_dtheta = -n_dot_dray_dtheta / n_dot_ray_safe  # (H, W)
    dlogd_dphi = -n_dot_dray_dphi / n_dot_ray_safe  # (H, W)

    # 转换到离散像素梯度
    # 代码中 grad_x = log_d[j] - log_d[j+1] = -∂(log d)/∂θ * Δθ
    # Δθ = 2π / W（每像素对应的 θ 变化）
    # grad_y = log_d[i] - log_d[i+1] = -∂(log d)/∂φ * Δφ = ∂(log d)/∂φ * (π/H)

    delta_theta = 2 * np.pi / W
    delta_phi = np.pi / H

    # 像素级连续梯度
    pixel_dlogd_dx = -dlogd_dtheta * delta_theta  # 对应 log_d[j] - log_d[j+1]
    pixel_dlogd_dy = dlogd_dphi * delta_phi  # 对应 log_d[i] - log_d[i+1]

    # 离散梯度（边界处取两侧平均）
    # X方向 (wrap)
    padded_grad_x = np.pad(pixel_dlogd_dx, ((0, 0), (0, 1)), mode='wrap')
    grad_x = (padded_grad_x[:, :-1] + padded_grad_x[:, 1:]) / 2  # (H, W)

    # Y方向 (不wrap)
    grad_y = (pixel_dlogd_dy[:-1, :] + pixel_dlogd_dy[1:, :]) / 2  # (H-1, W)

    # Valid masks for gradients
    padded_valid = np.pad(valid_mask, ((0, 0), (0, 1)), mode='wrap')
    mask_x = padded_valid[:, :-1] & padded_valid[:, 1:]  # (H, W)
    mask_y = valid_mask[:-1, :] & valid_mask[1:, :]  # (H-1, W)

    return grad_x, grad_y, mask_x, mask_y, valid_mask


def grad_equation_separate(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False):
    """
    分别返回 x 方向和 y 方向的梯度方程矩阵

    Returns:
        grad_eq_x: x 方向梯度矩阵，行数 = H * W (如果 wrap_x) 或 H * (W-1)
        grad_eq_y: y 方向梯度矩阵，行数 = (H-1) * W (如果不 wrap_y)
    """
    grid_index = np.arange(width * height).reshape(height, width)

    # X 方向
    if wrap_x:
        grid_x = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    else:
        grid_x = grid_index

    n_grad_x = grid_x.shape[0] * (grid_x.shape[1] - 1)
    data_x = np.concatenate([
        np.ones((grid_x.shape[0], grid_x.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
        -np.ones((grid_x.shape[0], grid_x.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_x = np.concatenate([
        grid_x[:, :-1].reshape(-1, 1),
        grid_x[:, 1:].reshape(-1, 1),
    ], axis=1).reshape(-1)
    indptr_x = np.arange(0, n_grad_x * 2 + 1, 2)
    grad_eq_x = csr_array((data_x, indices_x, indptr_x), shape=(n_grad_x, height * width))

    # Y 方向 (通常不 wrap)
    if wrap_y:
        grid_y = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')
    else:
        grid_y = grid_index

    n_grad_y = (grid_y.shape[0] - 1) * grid_y.shape[1]
    data_y = np.concatenate([
        np.ones((grid_y.shape[0] - 1, grid_y.shape[1]), dtype=np.float32).reshape(-1, 1),
        -np.ones((grid_y.shape[0] - 1, grid_y.shape[1]), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_y = np.concatenate([
        grid_y[:-1, :].reshape(-1, 1),
        grid_y[1:, :].reshape(-1, 1),
    ], axis=1).reshape(-1)
    indptr_y = np.arange(0, n_grad_y * 2 + 1, 2)
    grad_eq_y = csr_array((data_y, indices_y, indptr_y), shape=(n_grad_y, height * width))

    return grad_eq_x, grad_eq_y


def merge_panorama_depth_normal_gpu(width: int, height: int, distance_maps: List[np.ndarray], normal_maps: List[np.ndarray],
                                    pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray], panorama_depth_init=None,
                                    normal_weight=0.05):
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth_gpu(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics)
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    uv = utils3d.numpy.image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)  # [h,w,3]

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_normal_grad_maps, panorama_normal_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.numpy.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)

        projected_pixels = utils3d.numpy.uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)

        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(projection_valid_mask,
                                             cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_pred_mask = projection_valid_mask & (
                cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]

        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate normal gradient map
        # 从当前视角采样法线
        normal_map = normal_maps[i].astype(np.float32)
        # 分别 remap 三个通道
        panorama_normal_x = cv2.remap(normal_map[..., 0], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        panorama_normal_y = cv2.remap(normal_map[..., 1], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        panorama_normal_z = cv2.remap(normal_map[..., 2], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        panorama_normal_cam = np.stack([panorama_normal_x, panorama_normal_y, panorama_normal_z], axis=-1)

        # normal从相机坐标系转换到世界坐标系
        # 旋转矩阵（相机到世界）
        R_cam2world = extrinsics[i][:3, :3].T  # extrinsics 是 world2cam
        panorama_normal_world = np.einsum('ij,hwj->hwi', R_cam2world, panorama_normal_cam)

        # 归一化
        normal_norms = np.linalg.norm(panorama_normal_world, axis=-1, keepdims=True)
        normal_norms = np.maximum(normal_norms, 1e-6)
        panorama_normal_world = panorama_normal_world / normal_norms

        # 计算 log distance grad from normal
        normal_grad_x, normal_grad_y, normal_mask_x, normal_mask_y, normal_valid = normal_to_log_distance_gradient(
            panorama_normal_world, spherical_directions, width, height
        )
        # 合并 projection mask
        normal_mask_x = normal_mask_x & np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')[:, :-1] & \
                        np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')[:, 1:]
        normal_mask_y = normal_mask_y & panorama_pred_mask[:-1, :] & panorama_pred_mask[1:, :]
        panorama_log_normal_grad_maps.append((normal_grad_x, normal_grad_y))
        panorama_normal_grad_masks.append((normal_mask_x, normal_mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)

        panorama_pred_masks.append(panorama_pred_mask)

    # distance grad processing
    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    # normal grad processing
    panorama_log_normal_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_normal_grad_maps], axis=0)
    panorama_log_normal_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_normal_grad_maps], axis=0)
    panorama_normal_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_normal_grad_masks], axis=0)
    panorama_normal_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_normal_grad_masks], axis=0)

    panorama_log_normal_grad_x = np.sum(panorama_log_normal_grad_x * panorama_normal_grad_mask_x, axis=0) / np.sum(panorama_normal_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_normal_grad_y = np.sum(panorama_log_normal_grad_y * panorama_normal_grad_mask_y, axis=0) / np.sum(panorama_normal_grad_mask_y, axis=0).clip(1e-3)

    # laplacian processing
    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])

    normal_grad_x_mask = np.any(panorama_normal_grad_mask_x, axis=0).reshape(-1)
    normal_grad_y_mask = np.any(panorama_normal_grad_mask_y, axis=0).reshape(-1)

    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    grad_A = grad_equation(width, height, wrap_x=True, wrap_y=False)
    grad_eq_x, grad_eq_y = grad_equation_separate(width, height, wrap_x=True, wrap_y=False)
    poisson_A = poisson_equation(width, height, wrap_x=True, wrap_y=False)

    # Solve overdetermined system
    A = vstack([
        grad_A[grad_mask],
        grad_eq_x[normal_grad_x_mask] * normal_weight,
        grad_eq_y[normal_grad_y_mask] * normal_weight,
        poisson_A[laplacian_mask],
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask],
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_log_normal_grad_x.reshape(-1)[normal_grad_x_mask] * normal_weight,
        panorama_log_normal_grad_y.reshape(-1)[normal_grad_y_mask] * normal_weight,
        panorama_laplacian_map.reshape(-1)[laplacian_mask]
    ])
    x = solve_lsmr_gpu(A, b, x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None)

    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask


def pred_pano_depth_v2(model, image: Image.Image, scale=1.0, resize_to=1920, remove_pano_depth_nan=True, normal_weight=0.05):
    """
    last_layer_mask: 前一层物体mask
    last_layer_depth: 前一层深度
    """
    print("\t - Predicting pano depth with moge")
    image_origin = np.array(image)
    height_origin, width_origin = image_origin.shape[:2]

    image, height, width = image_origin, height_origin, width_origin
    if resize_to is not None:
        _height, _width = min(resize_to, int(resize_to * height_origin / width_origin)), min(resize_to, int(resize_to * width_origin / height_origin))
        if _height < height_origin:
            print(f"\t - Resizing image from {width_origin}x{height_origin} to {_width}x{_height} for pano depth prediction")
            image = cv2.resize(image_origin, (_width, _height), cv2.INTER_AREA)
            height, width = _height, _width

    splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
    splitted_resolution = 512
    splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution, splitted_resolution, interp=cv2.INTER_LINEAR)

    # infer moge depth
    num_splitted_images = len(splitted_images)
    splitted_distance_maps = [None] * num_splitted_images
    splitted_normal_maps = [None] * num_splitted_images
    splitted_masks = [None] * num_splitted_images

    indices_to_process_model = []
    skipped_count = 0

    for i in range(num_splitted_images):
        indices_to_process_model.append(i)

    pred_count = 0
    # Process images that require model inference in batches
    inference_batch_size = 1
    for i in range(0, len(indices_to_process_model), inference_batch_size):
        batch_indices = indices_to_process_model[i: i + inference_batch_size]
        if not batch_indices:
            continue

        current_batch_images = [splitted_images[k] for k in batch_indices]
        current_batch_intrinsics = [splitted_intriniscs[k] for k in batch_indices]

        image_tensor = torch.tensor(
            np.stack(current_batch_images) / 255,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        ).permute(0, 3, 1, 2)

        fov_x, _ = np.rad2deg(  # fov_y is not used by model.infer
            utils3d.numpy.intrinsics_to_fov(np.array(current_batch_intrinsics))
        )
        fov_x_tensor = torch.tensor(
            fov_x, dtype=torch.float32, device=next(model.parameters()).device
        )

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            output = model.infer(image_tensor, fov_x=fov_x_tensor, apply_mask=False)

        batch_distance_maps = output["points"].norm(dim=-1).cpu().numpy()
        batch_normal_maps = output["normal"].cpu().numpy()
        batch_masks = output["mask"].cpu().numpy()

        for batch_idx, original_idx in enumerate(batch_indices):
            splitted_distance_maps[original_idx] = batch_distance_maps[batch_idx]
            splitted_normal_maps[original_idx] = batch_normal_maps[batch_idx]
            splitted_masks[original_idx] = batch_masks[batch_idx]
            pred_count += 1

    if (pred_count + skipped_count) == 0:  # Avoid division by zero if num_splitted_images is 0
        skip_ratio_info = "N/A (no images to process)"
    else:
        skip_ratio_info = f"{skipped_count / (pred_count + skipped_count):.2%}"

    print(f"\t 🔍 Predicted {pred_count} splitted images, skipped {skipped_count} splitted images. Skip ratio: {skip_ratio_info}")

    # merge moge depth
    merging_width, merging_height = width, height
    panorama_depth, panorama_mask = merge_panorama_depth_normal_gpu(
        width=merging_width,
        height=merging_height,
        distance_maps=splitted_distance_maps,
        normal_maps=splitted_normal_maps,
        pred_masks=splitted_masks,
        extrinsics=splitted_extrinsics,
        intrinsics=splitted_intriniscs,
        normal_weight=normal_weight
    )

    panorama_depth = panorama_depth.astype(np.float32)
    # 对齐 dano depth最底部0.03区域左右侧深度
    if remove_pano_depth_nan:
        # for depth inpainting, remove nan
        panorama_depth[~panorama_mask] = 1.0 * np.nanquantile(panorama_depth, 0.999)  # 天空深度
    panorama_depth = cv2.resize(panorama_depth, (width_origin, height_origin), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width_origin, height_origin), cv2.INTER_NEAREST) > 0

    # 平滑南极点（底部区域）深度，解决左右不一致问题
    print("\t - Smoothing south pole depth for consistency")
    panorama_depth = smooth_south_pole_depth(panorama_depth, smooth_height_ratio=0.05)

    rays = torch.from_numpy(spherical_uv_to_directions(utils3d.numpy.image_uv(width=width_origin, height=height_origin))).to(next(model.parameters()).device)

    panorama_depth = (
            torch.from_numpy(panorama_depth).to(next(model.parameters()).device) * scale
    )

    return {
        "rgb": torch.from_numpy(image_origin).to(next(model.parameters()).device),
        "distance": panorama_depth,
        "rays": rays,
        "mask": panorama_mask,
        "splitted_masks": splitted_masks,
        "splitted_distance_maps": splitted_distance_maps,
    }


def merge_da360_panorama_depth(
        width: int,
        height: int,
        panorama_depth_init,
        distance_maps: List[np.ndarray],
        pred_masks: List[np.ndarray],
        extrinsics: List[np.ndarray],
        intrinsics: List[np.ndarray],
        regularization_weight: float = 0.1,  # 新增：正则化权重
):
    """
    合并多张 distance_maps 为全景 distance

    Args:
        regularization_weight: 正则化权重，越大越接近初始值
        use_adaptive_weight: 是否根据 mask 覆盖度自适应调整权重
    """
    panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    log_depth_init = np.log(panorama_depth_init).reshape(-1)

    uv = utils3d.numpy.image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)  # [h,w,3]

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []

    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.numpy.project_cv(
            spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i]
        )
        projection_valid_mask = (
                (projected_depth > 0) &
                (projected_uv > 0).all(axis=-1) &
                (projected_uv < 1).all(axis=-1)
        )

        projected_pixels = utils3d.numpy.uv_to_pixel(
            np.clip(projected_uv, 0, 1),
            width=distance_maps[i].shape[1],
            height=distance_maps[i].shape[0]
        ).astype(np.float32)

        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(
            projection_valid_mask,
            cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            0
        )
        panorama_pred_mask = projection_valid_mask & (
                cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1],
                          cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
        )

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]

        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)
        panorama_pred_masks.append(panorama_pred_mask)

    # Stack and merge gradients
    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    # ========== 计算正则化权重 ==========
    n_pixels = width * height

    # 均匀权重
    if regularization_weight == 0:
        reg_weight = 0
    else:
        reg_weight = np.sqrt(regularization_weight)
    reg_A = reg_weight * eye(n_pixels, format='csr')
    reg_b = reg_weight * log_depth_init

    # ========== 构建带正则项的优化问题 ==========
    # 原问题: min ||Ax - b||^2
    # 新问题: min ||Ax - b||^2 + λ||x - x0||^2
    #       = min ||[A; sqrt(λ)I] x - [b; sqrt(λ)x0]||^2

    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
        reg_A,  # 正则项：sqrt(λ) * I
    ])

    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask],
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask],
        reg_b,  # 正则项：sqrt(λ) * log(init)
    ])

    x, *_ = lsmr(
        A, b,
        atol=1e-5, btol=1e-5,
        x0=log_depth_init,
        show=False,
    )

    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask


def pred_pano_ensemble_depth(model, image: Image.Image, distance_init, scale=1.0, resize_to=1920, remove_pano_depth_nan=True, regularization_weight=0.1):
    """
    last_layer_mask: 前一层物体mask
    last_layer_depth: 前一层深度
    """
    print("\t - Refine pano depth with moge")
    image_origin = np.array(image)
    height_origin, width_origin = image_origin.shape[:2]

    image, height, width = image_origin, height_origin, width_origin
    if resize_to is not None:
        _height, _width = min(resize_to, int(resize_to * height_origin / width_origin)), min(resize_to, int(resize_to * width_origin / height_origin))
        if _height < height_origin:
            print(f"\t - Resizing image from {width_origin}x{height_origin} to {_width}x{_height} for pano depth prediction")
            image = cv2.resize(image_origin, (_width, _height), cv2.INTER_AREA)
            height, width = _height, _width

    splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
    splitted_resolution = 512
    splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution, splitted_resolution, interp=cv2.INTER_LINEAR)

    # infer moge depth
    num_splitted_images = len(splitted_images)
    splitted_distance_maps = [None] * num_splitted_images
    splitted_masks = [None] * num_splitted_images

    indices_to_process_model = []
    skipped_count = 0

    for i in range(num_splitted_images):
        indices_to_process_model.append(i)

    pred_count = 0
    # Process images that require model inference in batches
    inference_batch_size = 1
    for i in range(0, len(indices_to_process_model), inference_batch_size):
        batch_indices = indices_to_process_model[i: i + inference_batch_size]
        if not batch_indices:
            continue

        current_batch_images = [splitted_images[k] for k in batch_indices]
        current_batch_intrinsics = [splitted_intriniscs[k] for k in batch_indices]

        image_tensor = torch.tensor(
            np.stack(current_batch_images) / 255,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        ).permute(0, 3, 1, 2)

        fov_x, _ = np.rad2deg(  # fov_y is not used by model.infer
            utils3d.numpy.intrinsics_to_fov(np.array(current_batch_intrinsics))
        )
        fov_x_tensor = torch.tensor(
            fov_x, dtype=torch.float32, device=next(model.parameters()).device
        )

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            output = model.infer(image_tensor, fov_x=fov_x_tensor, apply_mask=False)

        batch_distance_maps = output["points"].norm(dim=-1).cpu().numpy()
        batch_masks = output["mask"].cpu().numpy()

        for batch_idx, original_idx in enumerate(batch_indices):
            splitted_distance_maps[original_idx] = batch_distance_maps[batch_idx]
            splitted_masks[original_idx] = batch_masks[batch_idx]
            pred_count += 1

    if (pred_count + skipped_count) == 0:  # Avoid division by zero if num_splitted_images is 0
        skip_ratio_info = "N/A (no images to process)"
    else:
        skip_ratio_info = f"{skipped_count / (pred_count + skipped_count):.2%}"

    print(f"\t 🔍 Predicted {pred_count} splitted images, skipped {skipped_count} splitted images. Skip ratio: {skip_ratio_info}")

    # merge moge depth
    merging_width, merging_height = width, height
    panorama_depth, panorama_mask = merge_da360_panorama_depth(
        merging_width,
        merging_height,
        distance_init,
        splitted_distance_maps,
        splitted_masks,
        splitted_extrinsics,
        splitted_intriniscs,
        regularization_weight=regularization_weight,
    )

    panorama_depth = panorama_depth.astype(np.float32)
    # 对齐 dano depth最底部0.03区域左右侧深度
    if remove_pano_depth_nan:
        # for depth inpainting, remove nan
        panorama_depth[~panorama_mask] = 1.0 * np.nanquantile(panorama_depth, 0.999)  # 天空深度
    panorama_depth = cv2.resize(panorama_depth, (width_origin, height_origin), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width_origin, height_origin), cv2.INTER_NEAREST) > 0

    # 平滑南极点（底部区域）深度，解决左右不一致问题
    print("\t - Smoothing south pole depth for consistency")
    panorama_depth = smooth_south_pole_depth(panorama_depth, smooth_height_ratio=0.05)

    rays = torch.from_numpy(spherical_uv_to_directions(utils3d.numpy.image_uv(width=width_origin, height=height_origin))).to(next(model.parameters()).device)

    panorama_depth = (
            torch.from_numpy(panorama_depth).to(next(model.parameters()).device) * scale
    )

    return {
        "rgb": torch.from_numpy(image_origin).to(next(model.parameters()).device),
        "distance": panorama_depth,
        "rays": rays,
        "mask": panorama_mask,
        "splitted_masks": splitted_masks,
        "splitted_distance_maps": splitted_distance_maps,
    }





def convert_rgbd2pcd_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        max_size: int = 4096,  # Max dimension for resizing
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        dropout_pcd=False
):
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert (
            rgb.shape[:2] == distance.shape[:2] == rays.shape[:2]
    ), "Input shapes must match"

    mask = excluded_region_mask

    if mask is not None:
        assert (
                mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2]
        ), "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    H, W = distance.shape
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)

    rgb_resized = (
        F.interpolate(
            rgb_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )

    distance_resized = (
        F.interpolate(
            distance_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    rays_resized_nchw = F.interpolate(
        rays_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )

    # IMPORTANT: Renormalize ray directions after interpolation
    rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
    rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
    rays_resized = rays_resized / (rays_norm + 1e-8)

    if mask is not None:
        mask_resized = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),  # Needs float for interpolation
                scale_factor=scale,
                mode="nearest",  # Or 'nearest' if sharp boundaries are critical
                # align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask_resized = mask_resized > 0.5  # Convert back to boolean
    else:
        mask_resized = None

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance_resized.reshape(-1, 1)  # (H*W, 1)
    rays_flat = rays_resized.reshape(-1, 3)  # (H*W, 3)
    vertices = distance_flat * rays_flat  # (H*W, 3)
    vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)
    if mask_resized is not None:
        mask_resized = mask_resized.reshape(-1, )
        vertices = vertices[~mask_resized]
        vertex_colors = vertex_colors[~mask_resized]

    # downsample
    if dropout_pcd and vertices.shape[0] > 1_000_000:
        rdx = np.arange(vertices.shape[0])
        np.random.shuffle(rdx)
        rdx = rdx[:1_000_000]
        vertices = vertices[rdx]
        vertex_colors = vertex_colors[rdx]

    pcd = trimesh.PointCloud(vertices=vertices.cpu().numpy(), colors=vertex_colors.cpu().numpy())

    return pcd


def convert_rgbd2pcd_multi_scale_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        depth_intervals=[0, 1, 2, 4, 8]
):
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert (
            rgb.shape[:2] == distance.shape[:2] == rays.shape[:2]
    ), "Input shapes must match"

    mask = excluded_region_mask

    if mask is not None:
        assert (
                mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2]
        ), "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    median_distance = torch.median(distance).item()

    total_points = []
    total_colors = []

    for i in tqdm(range(1, len(depth_intervals)), desc="Processing depth intervals"):
        if i == len(depth_intervals) - 1:
            interval_mask = distance_nchw > (median_distance * depth_intervals[i - 1])
        else:
            interval_mask = ((median_distance * depth_intervals[i - 1]) < distance_nchw) & (distance_nchw <= (median_distance * depth_intervals[i]))

        # pointclouds number ∝ depth^2
        resize_scale = depth_intervals[i]
        if interval_mask.sum() == 0:
            continue

        rgb_resized = (
            F.interpolate(
                rgb_nchw,
                scale_factor=resize_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

        distance_resized = (
            F.interpolate(
                distance_nchw,
                scale_factor=resize_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        rays_resized_nchw = F.interpolate(
            rays_nchw,
            scale_factor=resize_scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )

        interval_mask_resized = F.interpolate(interval_mask.float(),
                                              scale_factor=resize_scale,
                                              mode="nearest",
                                              recompute_scale_factor=False).bool().squeeze(0).squeeze(0)

        # IMPORTANT: Renormalize ray directions after interpolation
        rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
        rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
        rays_resized = rays_resized / (rays_norm + 1e-8)

        if mask is not None:
            mask_resized = (
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  # Needs float for interpolation
                    scale_factor=resize_scale,
                    mode="nearest",  # Or 'nearest' if sharp boundaries are critical
                    # align_corners=False,
                    recompute_scale_factor=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            mask_resized = mask_resized > 0.5  # Convert back to boolean
        else:
            mask_resized = None

        mask_resized = mask_resized & interval_mask_resized

        # --- Calculate 3D Vertices ---
        # Vertex position = origin + distance * ray_direction
        # Assuming origin is (0, 0, 0)
        distance_flat = distance_resized.reshape(-1, 1)  # (H*W, 1)
        rays_flat = rays_resized.reshape(-1, 3)  # (H*W, 3)
        vertices = distance_flat * rays_flat  # (H*W, 3)
        vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)
        if mask_resized is not None:
            mask_resized = mask_resized.reshape(-1, )
            vertices = vertices[~mask_resized]
            vertex_colors = vertex_colors[~mask_resized]

        total_points.append(vertices)
        total_colors.append(vertex_colors)
        print(f"Depth interval: {depth_intervals[i]}, Number of points: {vertices.shape[0]}")

    vertices = torch.cat(total_points, dim=0)
    vertex_colors = torch.cat(total_colors, dim=0)
    pcd = trimesh.PointCloud(vertices=vertices.cpu().numpy(), colors=vertex_colors.cpu().numpy())

    return pcd


def convert_rgbd2pcd_panorama_da360(depth, rgb, mask=None, dropout_pcd=True):
    h, w = depth.shape
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = -np.repeat(Phi, h, axis=0)

    X = depth * np.sin(Theta) * np.sin(Phi)
    Y = depth * np.cos(Theta)
    Z = depth * np.sin(Theta) * np.cos(Phi)

    if mask is None:
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        R = rgb[:, :, 0].flatten()
        G = rgb[:, :, 1].flatten()
        B = rgb[:, :, 2].flatten()
    else:
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        R = rgb[:, :, 0][mask]
        G = rgb[:, :, 1][mask]
        B = rgb[:, :, 2][mask]

    XYZ = np.stack([X, Y, Z], axis=1)
    RGB = np.stack([R, G, B], axis=1)

    # downsample
    if dropout_pcd and XYZ.shape[0] > 1_000_000:
        rdx = np.arange(XYZ.shape[0])
        np.random.shuffle(rdx)
        rdx = rdx[:1_000_000]
        XYZ = XYZ[rdx]
        RGB = RGB[rdx]

    pcd = trimesh.PointCloud(vertices=XYZ, colors=RGB)
    return pcd


def _generate_faces_numpy(H: int, W: int, mask: Optional[torch.Tensor]) -> np.ndarray:
    """
    纯 NumPy 实现，比 PyTorch 版本快 2-3x
    """
    # 预计算所有顶点索引
    idx = np.arange(H * W, dtype=np.int32).reshape(H, W)

    # 四边形的四个角（带环绕）
    tl = idx[:-1, :]  # top-left
    tr = idx[:-1, :].copy()
    tr[:, :-1] = idx[:-1, 1:]
    tr[:, -1] = idx[:-1, 0]  # wrap
    bl = idx[1:, :]  # bottom-left
    br = idx[1:, :].copy()
    br[:, :-1] = idx[1:, 1:]
    br[:, -1] = idx[1:, 0]  # wrap

    # 应用 mask
    if mask is not None:
        mask_np = mask.cpu().numpy()
        # 检查四个角是否有任何一个被 mask
        m_tl = mask_np[:-1, :]
        m_tr = np.roll(mask_np[:-1, :], -1, axis=1)
        m_bl = mask_np[1:, :]
        m_br = np.roll(mask_np[1:, :], -1, axis=1)

        keep = ~(m_tl | m_tr | m_bl | m_br)

        tl, tr, bl, br = tl[keep], tr[keep], bl[keep], br[keep]
    else:
        tl, tr, bl, br = tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()

    # 构建三角形（每个四边形2个三角形）
    n = len(tl)
    faces = np.empty((2 * n, 3), dtype=np.int32)
    faces[0::2] = np.column_stack([tl, tr, bl])
    faces[1::2] = np.column_stack([tr, br, bl])

    return faces


def convert_rgbd2mesh_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        max_size: int = 4096,  # Max dimension for resizing
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        connect_boundary_max_dist: Optional[float] = 0.5,  # Max distance to bridge boundary vertices
        connect_boundary_repeat_times: int = 2
) -> o3d.geometry.TriangleMesh:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    """优化版本：~3-5x 加速"""
    # print('    [Mesh] Step 1: Getting input shapes...')
    H, W = distance.shape
    # print(f'    [Mesh] Input size: {H}x{W}')
    scale = min(1.0, max_size / max(H, W))
    need_resize = scale < 1.0
    # print(f'    [Mesh] Scale: {scale}, need_resize: {need_resize}')

    # ========== 1. 数据准备（异步传输） ==========
    # print('    [Mesh] Step 2: Moving data to device...')
    rgb = rgb.to(device, non_blocking=True)
    # print('    [Mesh]   rgb moved')
    distance = distance.to(device, non_blocking=True)
    # print('    [Mesh]   distance moved')
    rays = rays.to(device, non_blocking=True)
    # print('    [Mesh]   rays moved')
    if excluded_region_mask is not None:
        mask = excluded_region_mask.to(device, non_blocking=True)
        # print('    [Mesh]   mask moved')
    else:
        mask = None
        # print('    [Mesh]   no mask')

    # ========== 2. 缩放处理 ==========
    # print('    [Mesh] Step 3: Resizing if needed...')
    if need_resize:
        H_new, W_new = int(H * scale), int(W * scale)
        # print(f'    [Mesh]   Resizing to {H_new}x{W_new}')

        # 合并 rgb+rays 减少插值调用
        # print('    [Mesh]   Combining rgb and rays...')
        combined = torch.cat([rgb, rays], dim=-1).permute(2, 0, 1).unsqueeze(0)
        # print('    [Mesh]   Interpolating combined...')
        combined_resized = F.interpolate(combined, size=(H_new, W_new), mode='bilinear', align_corners=False)
        # print('    [Mesh]   Reshaping combined...')
        combined_resized = combined_resized.squeeze(0).permute(1, 2, 0)

        # print('    [Mesh]   Extracting rgb and rays...')
        rgb_resized = combined_resized[..., :3]
        rays_resized = F.normalize(combined_resized[..., 3:], dim=-1)

        # print('    [Mesh]   Interpolating distance...')
        distance_resized = F.interpolate(
            distance[None, None], size=(H_new, W_new), mode='bilinear', align_corners=False
        ).squeeze()

        if mask is not None:
            # print('    [Mesh]   Resizing mask...')
            mask_resized = F.max_pool2d(
                mask[None, None].float(),
                kernel_size=int(1 / scale),
                stride=int(1 / scale)
            ).squeeze().bool()
            # 确保尺寸匹配
            if mask_resized.shape != (H_new, W_new):
                mask_resized = F.interpolate(
                    mask[None, None].float(), size=(H_new, W_new), mode='nearest'
                ).squeeze().bool()
        else:
            mask_resized = None
    else:
        # print('    [Mesh]   No resize needed')
        H_new, W_new = H, W
        rgb_resized = rgb
        # print('    [Mesh]   Normalizing rays...')
        rays_resized = F.normalize(rays, dim=-1)
        distance_resized = distance
        mask_resized = mask
    # print('    [Mesh] Step 3 done!')

    # ========== 3. 计算顶点（GPU） ==========
    # print('    [Mesh] Step 4: Computing vertices...')
    vertices = (distance_resized.unsqueeze(-1) * rays_resized).reshape(-1, 3)
    # print(f'    [Mesh]   vertices shape: {vertices.shape}')
    vertex_colors = rgb_resized.reshape(-1, 3)
    # print(f'    [Mesh]   vertex_colors shape: {vertex_colors.shape}')

    # 同步并转换为 numpy
    # print('    [Mesh] Step 5: Synchronizing and converting to numpy...')
    torch.cuda.synchronize() if device == 'cuda' else None
    # print('    [Mesh]   Converting vertices to numpy...')
    vertices_np = vertices.cpu().numpy().astype(np.float64)
    # print('    [Mesh]   Converting colors to numpy...')
    colors_np = vertex_colors.cpu().numpy().astype(np.float64)

    # 检查数据有效性
    # print('    [Mesh]   Checking data validity...')
    # print(f'    [Mesh]     vertices_np: min={vertices_np.min():.3f}, max={vertices_np.max():.3f}')
    # print(f'    [Mesh]     NaN in vertices: {np.isnan(vertices_np).any()}')
    # print(f'    [Mesh]     Inf in vertices: {np.isinf(vertices_np).any()}')
    # print(f'    [Mesh]     colors_np: min={colors_np.min():.3f}, max={colors_np.max():.3f}')

    # 清理无效数据
    if np.isnan(vertices_np).any() or np.isinf(vertices_np).any():
        # print('    [Mesh]   WARNING: Found invalid values in vertices, cleaning...')
        valid_mask = ~(np.isnan(vertices_np).any(axis=1) | np.isinf(vertices_np).any(axis=1))
        vertices_np = vertices_np[valid_mask]
        colors_np = colors_np[valid_mask]
        # print(f'    [Mesh]   Kept {valid_mask.sum()}/{len(valid_mask)} valid vertices')

    # print('    [Mesh] Step 5 done!')

    # ========== 4. 生成面（CPU/NumPy，更快） ==========
    # print('    [Mesh] Step 6: Generating faces...')
    faces_np = _generate_faces_numpy(H_new, W_new, mask_resized)
    # print(f'    [Mesh]   faces shape: {faces_np.shape}')

    # 释放 GPU 内存，避免与 Open3D 冲突
    # print('    [Mesh]   Releasing GPU memory...')
    del vertices, vertex_colors, rgb_resized, rays_resized, distance_resized
    if mask_resized is not None:
        del mask_resized
    if device == 'cuda':
        torch.cuda.empty_cache()
    # print('    [Mesh]   GPU memory released')

    # ========== 5. 创建 Open3D Mesh ==========
    # print('    [Mesh] Step 7: Creating Open3D mesh...')
    mesh = o3d.geometry.TriangleMesh()
    # print('    [Mesh]   Setting vertices...')
    #确保数据是连续的，避免内存问题
    vertices_np = np.ascontiguousarray(vertices_np)
    # print(f'    [Mesh]     vertices_np is C-contiguous: {vertices_np.flags["C_CONTIGUOUS"]}')
    # print(f'    [Mesh]     vertices_np dtype: {vertices_np.dtype}')
    # print(f'    [Mesh]     vertices_np shape: {vertices_np.shape}')

    try:
        mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        # print('    [Mesh]   Vertices set successfully!')
    except Exception as e:
        print(f'    [Mesh]   ERROR setting vertices: {e}')
        raise

    # print('    [Mesh]   Setting triangles...')
    faces_np = np.ascontiguousarray(faces_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    # print('    [Mesh]   Setting vertex colors...')
    colors_np = np.ascontiguousarray(colors_np)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

    # print('    [Mesh] Step 8: Cleaning mesh...')
    mesh.remove_unreferenced_vertices()
    # print('    [Mesh]   Removed unreferenced vertices')
    mesh.remove_degenerate_triangles()
    # print('    [Mesh]   Removed degenerate triangles')

    # ========== 6. 边界处理 ==========
    # print('    [Mesh] Step 9: Boundary processing...')
    if connect_boundary_max_dist is not None and connect_boundary_max_dist > 0:
        # print('    [Mesh]   Filling boundary spikes...')
        mesh = _fill_small_boundary_spikes(mesh, connect_boundary_max_dist, connect_boundary_repeat_times)
        # Recompute normals after potential modification, if mesh still valid
        if mesh.has_triangles() and mesh.has_vertices():
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()  # Also computes triangle normals if vertex normals are computed

    return mesh


def _fill_small_boundary_spikes(
        mesh: o3d.geometry.TriangleMesh,
        max_bridge_dist: float,
        repeat_times: int = 3
) -> o3d.geometry.TriangleMesh:
    print(f"\t - DEBUG: Filling small boundary spikes with max_bridge_dist: {max_bridge_dist} and repeat_times: {repeat_times}")
    for iteration in range(repeat_times):
        if not mesh.has_triangles() or not mesh.has_vertices():
            return mesh

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # 1. Identify boundary edges
        edge_to_triangle_count = defaultdict(int)

        for tri_idx, tri in enumerate(triangles):
            for i in range(3):
                v1_idx, v2_idx = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted((v1_idx, v2_idx)))
                edge_to_triangle_count[edge] += 1

        boundary_edges = [edge for edge, count in edge_to_triangle_count.items() if count == 1]

        if not boundary_edges:
            return mesh

        # 2. Create an adjacency list for boundary vertices using only boundary edges
        boundary_adj = defaultdict(list)
        for v1_idx, v2_idx in boundary_edges:
            boundary_adj[v1_idx].append(v2_idx)
            boundary_adj[v2_idx].append(v1_idx)

        # 3. Process boundary vertices with new smooth filling algorithm
        new_triangles_list = []
        edge_added = defaultdict(bool)

        # print(f"DEBUG: Found {len(boundary_edges)} boundary edges.")
        # print(f"DEBUG: Max bridge distance set to: {max_bridge_dist}")

        new_triangles_added_count = 0

        for v_curr_idx, neighbors in boundary_adj.items():
            if len(neighbors) != 2:  # Only process vertices with exactly 2 boundary neighbors
                continue

            v_a_idx, v_b_idx = neighbors[0], neighbors[1]

            # Skip if these vertices already form a triangle
            potential_edge = tuple(sorted((v_a_idx, v_b_idx)))
            if edge_to_triangle_count[potential_edge] > 0 or edge_added[potential_edge]:
                continue

            # Calculate distances
            v_curr_coord = vertices[v_curr_idx]
            v_a_coord = vertices[v_a_idx]
            v_b_coord = vertices[v_b_idx]

            dist_a_b = np.linalg.norm(v_a_coord - v_b_coord)

            # Skip if distance exceeds threshold
            if dist_a_b > max_bridge_dist:
                continue

            # Create simple triangle (v_a, v_b, v_curr)
            new_triangles_list.append([v_a_idx, v_b_idx, v_curr_idx])
            new_triangles_added_count += 1
            edge_added[potential_edge] = True

            # Mark edges as processed
            edge_added[tuple(sorted((v_curr_idx, v_a_idx)))] = True
            edge_added[tuple(sorted((v_curr_idx, v_b_idx)))] = True

        # 4. Now process multi-step connections for better smoothing
        # First build boundary chains for multi-step connections
        boundary_loops = []
        visited_vertices = set()

        # Find boundary vertices with exactly 2 neighbors (part of continuous chains)
        chain_starts = [v for v in boundary_adj if len(boundary_adj[v]) == 2 and v not in visited_vertices]

        for start_vertex in chain_starts:
            if start_vertex in visited_vertices:
                continue

            chain = []
            curr_vertex = start_vertex

            # Follow the chain in one direction
            while curr_vertex not in visited_vertices:
                visited_vertices.add(curr_vertex)
                chain.append(curr_vertex)

                next_candidates = [n for n in boundary_adj[curr_vertex] if n not in visited_vertices]
                if not next_candidates:
                    break

                curr_vertex = next_candidates[0]

            if len(chain) >= 3:
                boundary_loops.append(chain)

        # print(f"DEBUG: Found {len(boundary_loops)} boundary chains for smoothing.")

        # Process each boundary chain for multi-step smoothing
        for chain in boundary_loops:
            chain_length = len(chain)

            # Skip very small chains
            if chain_length < 3:
                continue

            # Compute multi-step connections
            max_step = min(8, chain_length - 1)

            for i in range(chain_length):
                anchor_idx = chain[i]
                anchor_coord = vertices[anchor_idx]

                for step in range(3, max_step + 1):
                    if i + step >= chain_length:
                        break

                    far_idx = chain[i + step]
                    far_coord = vertices[far_idx]

                    # Check distance criteria
                    dist_anchor_far = np.linalg.norm(anchor_coord - far_coord)
                    if dist_anchor_far > max_bridge_dist * step:
                        continue

                    # Check if anchor and far are already connected
                    edge_anchor_far = tuple(sorted((anchor_idx, far_idx)))
                    if edge_to_triangle_count[edge_anchor_far] > 0 or edge_added[edge_anchor_far]:
                        continue

                    # Create fan triangles
                    fan_valid = True
                    fan_triangles = []

                    prev_mid_idx = anchor_idx

                    for j in range(1, step):
                        mid_idx = chain[i + j]

                        if prev_mid_idx != anchor_idx:
                            tri_edge1 = tuple(sorted((anchor_idx, mid_idx)))
                            tri_edge2 = tuple(sorted((prev_mid_idx, mid_idx)))

                            # Check if edges already exist (not created by our fan)
                            if (edge_to_triangle_count[tri_edge1] > 0 and not edge_added[tri_edge1]) or \
                                    (edge_to_triangle_count[tri_edge2] > 0 and not edge_added[tri_edge2]):
                                fan_valid = False
                                break

                            fan_triangles.append([anchor_idx, prev_mid_idx, mid_idx])

                        prev_mid_idx = mid_idx

                    # Add final triangle to connect to far_idx
                    if fan_valid:
                        fan_triangles.append([anchor_idx, prev_mid_idx, far_idx])

                    # Add all fan triangles if valid
                    if fan_valid and fan_triangles:
                        for triangle in fan_triangles:
                            v_a, v_b, v_c = triangle
                            edge_ab = tuple(sorted((v_a, v_b)))
                            edge_bc = tuple(sorted((v_b, v_c)))
                            edge_ac = tuple(sorted((v_a, v_c)))

                            new_triangles_list.append(triangle)
                            new_triangles_added_count += 1

                            edge_added[edge_ab] = True
                            edge_added[edge_bc] = True
                            edge_added[edge_ac] = True

                        # Once we've added a fan, move to the next anchor
                        break

        # print(f"DEBUG: Total new triangles added in iteration {iteration}: {new_triangles_added_count}")

        if new_triangles_added_count == 0:
            break

        # Update the mesh with new triangles
        if new_triangles_list:
            all_triangles_np = np.vstack((triangles, np.array(new_triangles_list, dtype=np.int32)))

            final_mesh = o3d.geometry.TriangleMesh()
            final_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            final_mesh.triangles = o3d.utility.Vector3iVector(all_triangles_np)

            if mesh.has_vertex_colors():
                final_mesh.vertex_colors = mesh.vertex_colors

            # Clean up the mesh
            final_mesh.remove_degenerate_triangles()
            final_mesh.remove_unreferenced_vertices()
            mesh = final_mesh

    return mesh


def get_view_point_from_panorama_point(global_pcd, w2c, K, image_h, image_w):
    # 获取当前视角对应的valid points
    projected_uv, projected_depth = utils3d.numpy.project_cv(global_pcd.vertices, extrinsics=w2c, intrinsics=K)
    projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
    projected_uv = projected_uv[projection_valid_mask]
    projected_uv[:, 0] = (projected_uv[:, 0] * image_w).round()
    projected_uv[:, 1] = (projected_uv[:, 1] * image_h).round()
    projected_uv = projected_uv.astype(np.int64)
    projected_uv[:, 0] = np.clip(projected_uv[:, 0], 0, image_w - 1)
    projected_uv[:, 1] = np.clip(projected_uv[:, 1], 0, image_h - 1)
    projected_depth = projected_depth[projection_valid_mask]

    projected_uv_1d = projected_uv[:, 1] * image_w + projected_uv[:, 0]  # 坐标转为1d
    original_indices = np.arange(projected_uv_1d.shape[0])
    # 双重排序，1d坐标相同的情况下，depth从小到大排序，最终u,v相同只保留depth最小的point
    sorted_indices = np.lexsort((-projected_depth, projected_uv_1d))
    projected_uv_1d_sorted = projected_uv_1d[sorted_indices]
    sub_uv_1d = projected_uv_1d_sorted[:-1] - projected_uv_1d_sorted[1:]  # 错位相减，非0位为depth最小
    final_valid_indices = original_indices[sorted_indices][:-1][(sub_uv_1d != 0)]

    projected_points = global_pcd.vertices[projection_valid_mask][final_valid_indices]
    projected_uv = projected_uv[final_valid_indices]
    projected_colors = global_pcd.colors[projection_valid_mask][final_valid_indices, :3]

    return projected_points, projected_colors, projected_uv


def smooth_sky_depth_boundary(
        depth: torch.Tensor,
        sky_mask: torch.Tensor,
        transition_width: int = 50,
        depth_max: float = None,
        method: str = 'mean'  # 'mean', 'median', 'gaussian'
) -> torch.Tensor:
    """
    平滑天空与前景交界处的深度过渡。

    Args:
        depth: 深度图（天空区域已设为depth_max）
        sky_mask: 天空 Mask (True=天空)
        transition_width: 过渡区域宽度（像素）
        depth_max: 天空深度值
        method: 边界扩散方法
            - 'mean': 加权均值扩散（推荐）
            - 'median': 边界中位数
            - 'gaussian': 高斯模糊扩散

    Returns:
        边界平滑后的深度图
    """
    # 1. 维度处理
    original_dim = depth.dim()
    if original_dim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        depth = depth.unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0)

    device = depth.device
    dtype = depth.dtype

    if depth_max is None:
        depth_max = depth.max().item()

    sky_mask_np = sky_mask.squeeze().cpu().numpy().astype(np.uint8)
    depth_np = depth.squeeze().cpu().numpy()

    if sky_mask_np.sum() == 0 or sky_mask_np.sum() == sky_mask_np.size:
        return _restore_dim(depth, original_dim)

    # ---------------------------------------------------------
    # 2. 计算天空区域内每个点到边界的距离
    # ---------------------------------------------------------
    dist_to_foreground = cv2.distanceTransform(sky_mask_np, cv2.DIST_L2, 5)

    # ---------------------------------------------------------
    # 3. 获取边界深度值（根据方法选择）
    # ---------------------------------------------------------
    sky_mask_tensor = sky_mask.bool()

    if method == 'mean':
        boundary_depth = _mean_diffusion(depth, sky_mask_tensor, transition_width)
    elif method == 'median':
        boundary_depth = _median_boundary(depth_np, sky_mask_np, depth_max, device, dtype, depth.shape)
    elif method == 'gaussian':
        boundary_depth = _gaussian_diffusion(depth, sky_mask_tensor, depth_max, transition_width)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ---------------------------------------------------------
    # 4. 线性插值：边界处用扩散的depth，深处用depth_max
    # ---------------------------------------------------------
    t = np.clip(dist_to_foreground / max(transition_width, 1), 0, 1)
    t = t * t * (3 - 2 * t)  # smoothstep
    t = torch.from_numpy(t).to(device=device, dtype=dtype).view_as(depth)

    interpolated = boundary_depth * (1 - t) + depth_max * t

    # ---------------------------------------------------------
    # 5. 限制范围并只修改天空区域
    # ---------------------------------------------------------
    interpolated = torch.clamp(interpolated, max=depth_max)
    result = torch.where(sky_mask_tensor, interpolated, depth)

    return _restore_dim(result, original_dim)


def _mean_diffusion(depth: torch.Tensor, sky_mask: torch.Tensor, transition_width: int) -> torch.Tensor:
    """
    加权均值扩散（推荐）
    保持局部深度特性，同时避免极端值
    """
    device = depth.device
    dtype = depth.dtype

    # 有效区域 mask (非天空 = 1)
    valid_mask = (~sky_mask).float()

    # 初始化：天空区域设为0
    flood_depth = depth.clone()
    flood_depth[sky_mask] = 0

    kernel_size = 31
    pad = kernel_size // 2
    max_iterations = (transition_width // pad) + 10

    for _ in range(max_iterations):
        # 计算邻域深度和
        depth_sum = F.avg_pool2d(flood_depth, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)
        # 计算邻域有效像素数
        valid_count = F.avg_pool2d(valid_mask, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)

        # 均值 = 总和 / 有效数量
        mean_depth = depth_sum / (valid_count + 1e-8)

        # 只更新天空区域中、邻域有有效像素的点
        can_update = sky_mask & (valid_count > 0)

        # 检查是否还有需要更新的点
        newly_filled = can_update & (valid_mask < 0.5)
        if not newly_filled.any():
            break

        # 更新
        flood_depth = torch.where(can_update, mean_depth, flood_depth)
        valid_mask = torch.where(can_update, torch.ones_like(valid_mask), valid_mask)

    # 兜底：仍为0的区域用全局均值
    still_zero = (flood_depth == 0) & sky_mask
    if still_zero.any():
        fg_mean = depth[~sky_mask].mean()
        flood_depth = torch.where(still_zero, fg_mean, flood_depth)

    return flood_depth


def _median_boundary(depth_np: np.ndarray, sky_mask_np: np.ndarray,
                     depth_max: float, device, dtype, shape) -> torch.Tensor:
    """
    边界中位数法
    提取边界处的深度，取中位数作为统一过渡值
    """
    # 提取边界：非天空但邻接天空的像素
    kernel = np.ones((3, 3), np.uint8)
    dilated_sky = cv2.dilate(sky_mask_np, kernel)
    boundary_mask = (dilated_sky == 1) & (sky_mask_np == 0)

    boundary_depths = depth_np[boundary_mask]

    if len(boundary_depths) > 0:
        # 过滤掉已经是 depth_max 的值
        valid_depths = boundary_depths[boundary_depths < depth_max * 0.99]
        if len(valid_depths) > 0:
            median_depth = np.median(valid_depths)
        else:
            median_depth = np.median(boundary_depths)
    else:
        median_depth = depth_np[sky_mask_np == 0].mean()

    return torch.full(shape, median_depth, device=device, dtype=dtype)


def _gaussian_diffusion(depth: torch.Tensor, sky_mask: torch.Tensor,
                        depth_max: float, transition_width: int) -> torch.Tensor:
    """
    高斯模糊扩散
    先对前景深度进行模糊，然后扩散到天空区域
    """
    device = depth.device
    dtype = depth.dtype

    # 用前景均值填充天空，避免模糊时被 depth_max 影响
    fg_mean = depth[~sky_mask].mean()
    temp_depth = depth.clone()
    temp_depth[sky_mask] = fg_mean

    # 高斯模糊
    blur_size = max(transition_width // 2, 3)
    if blur_size % 2 == 0:
        blur_size += 1

    from torchvision.transforms.functional import gaussian_blur
    blurred = gaussian_blur(temp_depth, kernel_size=[blur_size, blur_size], sigma=[blur_size / 4])

    # 迭代扩散（用模糊后的值做 min-pooling，此时极端值已被平滑）
    flood_depth = blurred.clone()
    flood_depth[sky_mask] = float('inf')

    kernel_size = 31
    pad = kernel_size // 2

    for _ in range(transition_width // pad + 5):
        if not torch.isinf(flood_depth[sky_mask]).any():
            break
        next_step = -F.max_pool2d(-flood_depth, kernel_size, stride=1, padding=pad)
        is_inf = torch.isinf(flood_depth)
        flood_depth = torch.where(is_inf, next_step, flood_depth)

    flood_depth = torch.where(torch.isinf(flood_depth), fg_mean, flood_depth)

    return flood_depth


def _restore_dim(tensor: torch.Tensor, original_dim: int) -> torch.Tensor:
    if original_dim == 2:
        return tensor.squeeze(0).squeeze(0)
    elif original_dim == 3:
        return tensor.squeeze(0)
    return tensor


def erp_distance_ray_to_normal(distance_map, ray_directions,
                               smooth_sigma=0.0,
                               facing_camera=True):
    """
    将 ERP 距离图和 ray 方向转换为世界坐标系法线图

    Args:
        distance_map: (H, W) 距离图
        ray_directions: (H, W, 3) 每个像素的射线方向（单位向量，从原点射出）
        smooth_sigma: 平滑 sigma，0 表示不平滑
        facing_camera: True 表示法线朝向相机（原点），False 则朝外

    Returns:
        normal_map: (H, W, 3) 世界坐标系法线，范围 [-1, 1]
        normal_rgb: (H, W, 3) RGB 可视化，范围 [0, 255]

    坐标系 (OpenCV):
        X: 右
        Y: 下
        Z: 前
    """
    # 0. 可选：平滑距离图
    if smooth_sigma > 0:
        distance_map = cv2.GaussianBlur(
            distance_map.astype(np.float32), (0, 0), smooth_sigma
        )

    # 1. 计算 3D 点坐标: P = ray * distance
    d = distance_map.astype(np.float64)
    points = ray_directions * d[..., np.newaxis]  # (H, W, 3)

    # 分离 x, y, z
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # 2. 计算切向量
    # 水平方向 Tu：使用循环边界（ERP 左右相连）
    x_right = np.roll(x, -1, axis=1)
    y_right = np.roll(y, -1, axis=1)
    z_right = np.roll(z, -1, axis=1)

    x_left = np.roll(x, 1, axis=1)
    y_left = np.roll(y, 1, axis=1)
    z_left = np.roll(z, 1, axis=1)

    Tu_x = x_right - x_left
    Tu_y = y_right - y_left
    Tu_z = z_right - z_left

    # 垂直方向 Tv：不循环，边界特殊处理
    x_down = np.roll(x, -1, axis=0)
    y_down = np.roll(y, -1, axis=0)
    z_down = np.roll(z, -1, axis=0)

    x_up = np.roll(x, 1, axis=0)
    y_up = np.roll(y, 1, axis=0)
    z_up = np.roll(z, 1, axis=0)

    Tv_x = x_down - x_up
    Tv_y = y_down - y_up
    Tv_z = z_down - z_up

    # 处理上下边界（极点附近）
    # 顶部 (v=0)：单侧差分
    Tv_x[0, :] = x[1, :] - x[0, :]
    Tv_y[0, :] = y[1, :] - y[0, :]
    Tv_z[0, :] = z[1, :] - z[0, :]

    # 底部 (v=H-1)：单侧差分
    Tv_x[-1, :] = x[-1, :] - x[-2, :]
    Tv_y[-1, :] = y[-1, :] - y[-2, :]
    Tv_z[-1, :] = z[-1, :] - z[-2, :]

    # 3. 叉积计算法线: N = Tu × Tv
    normal_x = Tu_y * Tv_z - Tu_z * Tv_y
    normal_y = Tu_z * Tv_x - Tu_x * Tv_z
    normal_z = Tu_x * Tv_y - Tu_y * Tv_x

    # 4. 归一化
    norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    norm = np.where(norm > 1e-10, norm, 1e-10)

    normal_x = normal_x / norm
    normal_y = normal_y / norm
    normal_z = normal_z / norm

    # 5. 确保法线朝向正确
    if facing_camera:
        # 法线应该朝向相机（原点）
        # 检查: dot(normal, -ray) > 0，即 dot(normal, ray) < 0
        dot_product = (normal_x * ray_directions[..., 0] +
                       normal_y * ray_directions[..., 1] +
                       normal_z * ray_directions[..., 2])

        # 如果 dot > 0，翻转法线
        flip_mask = dot_product > 0
        normal_x = np.where(flip_mask, -normal_x, normal_x)
        normal_y = np.where(flip_mask, -normal_y, normal_y)
        normal_z = np.where(flip_mask, -normal_z, normal_z)

    # 6. 堆叠为 (H, W, 3)
    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)

    # 7. 转换为 RGB 可视化
    # [-1, 1] → [0, 255]
    normal_rgb = ((normal_map + 1.0) / 2.0 * 255).astype(np.uint8)

    return normal_map, normal_rgb
