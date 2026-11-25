"""Run DepthMaster's public panoramic inference and save radial ERP depth."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

from .erp import open_erp


def predict(image, source, checkpoint, output):
    import torch

    sys.path.insert(0, str(source))
    from depthmaster.model import DepthMasterModel

    helper_file = source / 'eval_panorama/eval.py'
    spec = importlib.util.spec_from_file_location('depthmaster_panorama_helpers', helper_file)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    rgb = open_erp(image)
    height, width = rgb.shape[:2]
    device = torch.device('cuda:0')
    model = DepthMasterModel.from_pretrained(str(checkpoint)).to(device).eval()
    pano = torch.tensor(rgb, device=device).permute(2, 0, 1).float()[None] / 255
    face_size, fov_deg = 518, 95.0
    faces = helpers.erp_to_cubemap_gpu(pano, face_w=face_size, fov_deg=fov_deg)
    w2c, intrinsic = helpers._get_camera_params_cached(fov_deg, face_size, device, torch.float32)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        raw = model.forward(faces.to(model.dtype), num_tokens=0, camera_type='Panorama',
                            W2C=w2c[None].to(model.dtype),
                            intrinsics=intrinsic[None].to(model.dtype))
    xyz = raw['pts3d'].float() * raw['metric_scale'].float()[:, None, None, None]
    distance = torch.linalg.vector_norm(xyz[0], dim=-1)
    valid = torch.isfinite(distance) & (distance > 0)
    if raw.get('mask') is not None:
        valid &= raw['mask'].reshape_as(distance) > 0.5
    valid = valid.float()
    weighted = helpers.cubemap_to_erp_gpu(torch.where(valid > 0, distance, 0),
                                          pano_h=height, pano_w=width, fov_deg=fov_deg)
    weights = helpers.cubemap_to_erp_gpu(valid, pano_h=height, pano_w=width, fov_deg=fov_deg)
    depth = torch.where(weights > 0.5, weighted / weights.clamp_min(1e-6), 0).cpu().numpy()
    if np.count_nonzero(depth) < 100:
        raise RuntimeError('DepthMaster returned fewer than 100 valid radial depths')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, depth.astype(np.float32))
    print(f'DepthMaster radial depth {depth.shape} with {(depth > 0).mean():.1%} valid pixels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('image', 'source', 'checkpoint', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    predict(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
