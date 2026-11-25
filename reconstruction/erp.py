"""ERP rays and tangent views in OpenCV camera coordinates."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .polyhedra import get_face_infos


def erp_rays(height, width):
    longitude = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * (2 * np.pi)
    latitude = (0.5 - (np.arange(height, dtype=np.float32) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    return np.stack((np.sin(lon) * np.cos(lat), -np.sin(lat),
                     np.cos(lon) * np.cos(lat)), axis=-1)


def open_erp(path):
    with Image.open(path) as image:
        if image.width != image.height * 2:
            raise ValueError(f'Expected a 2:1 ERP image at {path}, got {image.size}')
        return np.asarray(image.convert('RGB'))


def cut_tangent_views(views, depth_path, output, *, face_count, size):
    """Write images, c2w/intrinsics priors, and anchor-only camera-Z priors."""
    import cv2

    images = output / 'images'
    depths = output / 'depth'
    images.mkdir(parents=True)
    depths.mkdir()
    depth = np.load(depth_path, allow_pickle=False)
    faces = get_face_infos(face_count, 0.2, prepend_front=(face_count == 20))
    cameras = {'extrinsics': [], 'intrinsics': []}
    y_flip = np.diag([1.0, -1.0, 1.0])
    cy = cx = (size - 1) / 2

    grids = []
    for face in faces:
        focal = cx / np.tan(face['half_angle'])
        rotation = y_flip @ np.column_stack((face['u'], -face['v'], face['center']))
        axis = (np.arange(size) - cx) / focal
        xx, yy = np.meshgrid(axis, axis)
        rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        cos_theta = 1 / np.linalg.norm(rays, axis=-1)
        directions = (rays @ rotation.T) * cos_theta[..., None]
        longitude = np.arctan2(directions[..., 0], directions[..., 2])
        latitude = np.arcsin(np.clip(-directions[..., 1], -1, 1))
        intrinsic = [[focal, 0, cx], [0, focal, cy], [0, 0, 1]]
        grids.append((rotation, intrinsic, longitude, latitude, cos_theta))

    for view_index, (path, pano_c2w) in enumerate(views):
        rgb = open_erp(path)
        height, width = rgb.shape[:2]
        if view_index == 0 and depth.shape != (height, width):
            raise ValueError(f'Anchor RGB shape {rgb.shape[:2]} differs from depth {depth.shape}')
        for face_index, (rotation, intrinsic, lon, lat, cos_theta) in enumerate(grids):
            name = f'view_{view_index * len(faces) + face_index:04d}'
            map_x = np.mod((lon / (2 * np.pi) + 0.5) * width - 0.5, width).astype(np.float32)
            map_y = np.clip((0.5 - lat / np.pi) * height - 0.5, 0, height - 1).astype(np.float32)
            color = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            Image.fromarray(color).save(images / f'{name}.png')
            face_c2w = np.eye(4)
            face_c2w[:3, :3] = rotation
            cameras['extrinsics'].append({'camera_id': name, 'matrix': (pano_c2w @ face_c2w).tolist()})
            cameras['intrinsics'].append({'camera_id': name, 'matrix': intrinsic})
            if view_index == 0:
                radial = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
                np.save(depths / f'{name}.npy', (radial * cos_theta).astype(np.float32))
            else:
                # The public loader requires a file for every view. Zero is unknown depth.
                np.save(depths / f'{name}.npy', np.zeros((size, size), dtype=np.float32))
    prior = output / 'cameras.json'
    prior.write_text(json.dumps(cameras, indent=2) + '\n')
    return prior
