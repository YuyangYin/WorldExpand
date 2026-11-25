"""Plan four auxiliary views from DepthMaster radial depth and render ERP conditioning."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .erp import erp_rays, open_erp


def camera_at(position):
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, 3] = -position
    return w2c


def export_depth_mesh(depth, path):
    """Connect nearby valid depth samples, including the ERP longitude seam."""
    import trimesh

    height, width = depth.shape
    stride = max(1, height // 120)
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    ranges = depth[np.ix_(rows, cols)]
    vertices = (erp_rays(height, width)[np.ix_(rows, cols)] * ranges[..., None]).reshape(-1, 3)
    indices = np.arange(ranges.size).reshape(ranges.shape)
    left = indices[:-1]
    right = np.roll(left, -1, axis=1)
    down = indices[1:]
    down_right = np.roll(down, -1, axis=1)
    faces = np.concatenate((np.stack((left, down, right), axis=-1).reshape(-1, 3),
                            np.stack((right, down, down_right), axis=-1).reshape(-1, 3)))
    tri_depth = ranges.ravel()[faces]
    min_depth, max_depth = tri_depth.min(axis=-1), tri_depth.max(axis=-1)
    faces = faces[(min_depth > 0) & (max_depth <= 1.15 * min_depth)]
    if not len(faces):
        raise ValueError('DepthMaster depth does not contain a connected surface')
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.export(path)


def render(image, depth_path, output, height, max_dist_ratio):
    import cv2
    import torch

    from panorama_rendering.nvrender.gs_renderer import build_isotropic_gs_from_pcd, gs_pano_render_color
    from panorama_rendering.nvrender.utils import depth_edge_torch
    from .boundary import build_horizon_boundary
    from .view_selection import (best_partition_angle, build_candidates_rings, geometric_prefilter,
                                 gs_proxy_tile_masks, partition_select, subsample_points)

    rgb = open_erp(image)
    depth = np.load(depth_path, allow_pickle=False)
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f'Expected matching ERP RGB and depth shapes, got {rgb.shape} and {depth.shape}')
    if depth.shape[0] > 960:
        rgb = cv2.resize(rgb, (1920, 960), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (1920, 960), interpolation=cv2.INTER_NEAREST)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    if np.count_nonzero(depth) < 100:
        raise ValueError('DepthMaster depth contains fewer than 100 valid samples')
    output.mkdir(parents=True)
    mesh = output / 'depth_mesh.ply'
    export_depth_mesh(depth, mesh)

    device = torch.device('cuda:0')
    distance = torch.from_numpy(depth).to(device)
    distance = distance.clamp_max(torch.quantile(distance[distance > 0], 0.99))
    valid = (distance > 0) & ~depth_edge_torch(distance, rtol=0.1)
    rays = torch.from_numpy(erp_rays(*depth.shape)).to(device)
    xyz = rays[valid] * distance[valid].unsqueeze(-1)
    if len(xyz) < 100:
        raise ValueError('Too few points remain after depth edge filtering')
    colors = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)[valid] / 127.5 - 1
    points = xyz.cpu().numpy()
    lo, hi = np.percentile(points, [2, 98], axis=0)
    scale = float((hi - lo).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Depth samples do not define a finite scene extent')
    anchor = np.zeros(3, dtype=np.float32)
    gs = build_isotropic_gs_from_pcd(xyz, colors, camera_at(anchor), device=device)
    boundary = build_horizon_boundary(str(mesh), scale, max_dist_ratio=max_dist_ratio)

    max_dist = scale * max_dist_ratio
    radii = max_dist * np.linspace(0.25, 1, 4)
    candidates, _, _ = build_candidates_rings(anchor, radii, n_dir_base=8)
    inside = boundary.contains(candidates)
    if inside.sum() < 4:
        # Densify the innermost ring while honoring the same boundary.
        inner, _, _ = build_candidates_rings(anchor, radii[0] * np.array([0.25, 0.5]), n_dir=16)
        candidates = np.concatenate((candidates, inner))
        inside = boundary.contains(candidates)
    if inside.sum() < 4:
        raise RuntimeError('Depth boundary contains fewer than four candidate cameras')
    filtered, _ = geometric_prefilter(candidates, subsample_points(points), scale, anchor_xyz=anchor)
    eligible = inside & filtered
    candidates = candidates[eligible if eligible.sum() >= 4 else inside]

    tile_masks, holes = gs_proxy_tile_masks(gs, candidates, camera_at, gs_pano_render_color, device)
    angle = best_partition_angle(boundary.angles, boundary.r_boundary)
    chosen, _ = partition_select(np.zeros_like(tile_masks[0]), tile_masks,
                                 candidates[:, [0, 2]], K=4, rotate_angle=angle)
    if len(chosen) < 4:
        # A saturated tile union can leave fewer than four greedy picks.
        remaining = [int(index) for index in np.argsort(-holes, kind='stable') if index not in chosen]
        chosen.extend(remaining[:4 - len(chosen)])
    positions = candidates[chosen]
    cameras = np.stack([camera_at(position) for position in positions])
    images, masks = gs_pano_render_color(gs, cameras, height, 2 * height, device, erp_mask_aware=True)
    extrinsics = {'center': {'w2c_matrix': camera_at(anchor).tolist(), 'camera_position': anchor.tolist()}}
    for index, (position, camera, warp, mask) in enumerate(zip(positions, cameras, images, masks)):
        if not mask.any():
            raise RuntimeError(f'Auxiliary camera {index} has no valid warp pixels')
        role = f'aux_{index}'
        Image.fromarray((warp.clip(0, 1) * 255).astype(np.uint8)).save(output / f'erp_{role}.png')
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / f'mask_{role}.png')
        extrinsics[role] = {'w2c_matrix': camera.tolist(), 'camera_position': position.tolist()}
    (output / 'render_meta.json').write_text(json.dumps({
        'axis_convention': 'OpenCV: right, down, forward',
        'camera_extrinsics': extrinsics,
        'proxy_hole_ratios': holes[chosen].tolist(),
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--depth-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--max-dist-ratio', type=float, default=0.5)
    render(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
