"""Run public WorldMirror 2.0 and prepare its RGB, camera Z and points for 3DGS."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from .erp import cut_tangent_views


def prepare_gs_data(source, tangent_images, output, expected_views):
    metadata = json.loads((source / 'camera_params.json').read_text())
    extrinsics = {e['camera_id']: e['matrix'] for e in metadata['extrinsics']}
    intrinsics = {e['camera_id']: e['matrix'] for e in metadata['intrinsics']}
    if set(extrinsics) != set(range(expected_views)) or extrinsics.keys() != intrinsics.keys():
        raise ValueError('WorldMirror camera IDs do not match the tangent inputs')
    (output / 'images').mkdir(parents=True)
    (output / 'depths').mkdir()
    cameras = {}
    for index in range(expected_views):
        name = f'view_{index:04d}'
        c2w, k = np.asarray(extrinsics[index]), np.asarray(intrinsics[index])
        if (c2w.shape != (4, 4) or k.shape != (3, 3) or not np.isfinite(c2w).all()
                or not np.isfinite(k).all() or not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-5)
                or not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-3)
                or min(k[0, 0], k[1, 1]) <= 0):
            raise ValueError(f'Invalid WorldMirror camera {index}')
        c2w[3] = [0, 0, 0, 1]
        cameras[name] = {'extrinsic': np.linalg.inv(c2w).tolist(), 'intrinsic': k.tolist()}
        depth = np.load(source / 'depth' / f'depth_{index:04d}.npy', allow_pickle=False)
        if depth.ndim != 2 or not np.any(np.isfinite(depth) & (depth > 0)):
            raise ValueError(f'Invalid WorldMirror depth for {name}')
        depth = np.where(np.isfinite(depth) & (depth > 0) & (depth < 65000), depth, 0)
        # The public pipeline saves geometry. Its square input cuts are the
        # RGB frames corresponding to the output camera intrinsics and depth.
        image_path = tangent_images / f'{name}.png'
        with Image.open(image_path) as image:
            if image.size != depth.shape[::-1]:
                raise ValueError(f'{image_path} size {image.size} differs from depth {depth.shape[::-1]}')
        shutil.copyfile(image_path, output / 'images' / image_path.name)
        # The HY-World parser reinterprets these 16 bits as a float16 camera Z.
        Image.fromarray(depth.astype(np.float16).view(np.uint16)).save(
            output / 'depths' / f'{name}.png')
    shutil.copyfile(source / 'points.ply', output / 'points.ply')
    (output / 'cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')


def reconstruct(anchor, warp, nvs, depth, source, model, output, faces, size):
    sys.path.insert(0, str(source))
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    metadata = json.loads((warp / 'render_meta.json').read_text())['camera_extrinsics']
    roles = ['center'] + [f'aux_{index}' for index in range(4)]
    paths = [anchor] + [nvs / role / 'output.png' for role in roles[1:]]
    views = [(path, np.linalg.inv(metadata[role]['w2c_matrix'])) for path, role in zip(paths, roles)]
    tangent = output / 'tangent'
    prior = cut_tangent_views(views, depth, tangent, face_count=faces, size=size)
    mirror = output / 'worldmirror'
    pipeline = WorldMirrorPipeline.from_pretrained(
        str(model), enable_bf16=True, disable_heads=['gs'])
    pipeline(str(tangent / 'images'), strict_output_path=str(mirror), target_size=size,
             max_resolution=size, prior_cam_path=str(prior),
             prior_depth_path=str(tangent / 'depth'), sky_mask_source='model',
             save_normal=False, save_gs=False)
    prepare_gs_data(mirror, tangent / 'images', output / 'gs_data',
                    len(views) * (21 if faces == 20 else 6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('anchor', 'warp', 'nvs', 'depth', 'source', 'model', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--faces', type=int, choices=[6, 20], default=20)
    parser.add_argument('--size', type=int, default=504)
    reconstruct(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
