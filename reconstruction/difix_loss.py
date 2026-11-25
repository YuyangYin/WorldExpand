"""Supervise a nearby novel 3DGS view with a detached Difix3D correction."""

import atexit
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def nearby_camera(c2w, centers, step):
    """Move 10% toward the closest distinct panorama center, keeping face orientation."""
    position = c2w[0, :3, 3]
    offsets = centers - position
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    distance, index = torch.where(distances > 1e-5, distances, torch.inf).min(dim=0)
    if not torch.isfinite(distance).item():
        raise ValueError('Difix auxiliary supervision requires two distinct camera centers')
    novel = c2w.detach().clone()
    novel[0, :3, 3] = position + offsets[index] * (0.1 if step % 2 == 0 else 0.15)
    return novel


def save_rgb(value, path):
    pixels = (value[0].detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


class DifixLoss:
    def __init__(self, settings, trainer):
        if trainer.world_size != 1 or trainer.cfg.batch_size != 1:
            raise ValueError('Difix loss requires single-GPU 3DGS training with batch size 1')
        self.settings = settings
        self.centers = torch.as_tensor(trainer.parser.camtoworlds[:, :3, 3].copy(),
                                       dtype=torch.float32, device=trainer.device)
        self.result = Path(trainer.cfg.result_dir)
        self.exchange = None
        self.worker = None
        atexit.register(self.close)

    @classmethod
    def from_env(cls, trainer):
        settings = os.environ.get('WE_DIFIX_SETTINGS')
        return cls(json.loads(settings), trainer) if settings else None

    def active(self, step):
        offset = step - self.settings['start']
        return offset >= 0 and offset % self.settings['every'] == 0

    def start_worker(self):
        settings = self.settings
        self.exchange = tempfile.TemporaryDirectory(prefix='worldexpand-difix-')
        path = Path(self.exchange.name)
        env = os.environ.copy()
        if settings['gpu'] is not None:
            env['CUDA_VISIBLE_DEVICES'] = settings['gpu']
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
        self.worker = subprocess.Popen(
            [settings['python'], '-m', 'reconstruction.difix_worker',
             '--source', settings['source'], '--weights', settings['weights'],
             '--exchange', str(path), '--size', str(settings['size'])],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        return path

    def close(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.stdin.close()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            worker.stdout.close()
        if self.exchange is not None:
            self.exchange.cleanup()
            self.exchange = None

    def __call__(self, trainer, step, c2w, intrinsics, reference, image_ids, mask_gate):
        height, width = reference.shape[1:3]
        novel = nearby_camera(c2w, self.centers, step)
        cfg = trainer.cfg
        rendered, _, _ = trainer.rasterize_splats(
            camtoworlds=novel, Ks=intrinsics, height=height, width=width,
            image_ids=image_ids, sh_degree=cfg.sh_degree, render_mode='RGB',
            near_plane=cfg.near_plane, far_plane=cfg.far_plane, mask_gate=mask_gate)
        first_request = self.worker is None
        exchange = self.start_worker() if first_request else Path(self.exchange.name)
        save_rgb(rendered, exchange / 'render.png')
        save_rgb(reference, exchange / 'reference.png')
        self.worker.stdin.write(json.dumps({'step': step}) + '\n')
        self.worker.stdin.flush()
        response = self.worker.stdout.readline()
        if not response or json.loads(response).get('step') != step:
            raise RuntimeError('Difix3D worker exited before producing the requested target')
        with Image.open(exchange / 'target.png') as image:
            target = torch.tensor(np.array(image.convert('RGB')), device=rendered.device).float() / 255
        target = target.permute(2, 0, 1)[None]
        prediction = F.interpolate(rendered.permute(0, 3, 1, 2), size=target.shape[-2:],
                                   mode='bilinear', align_corners=False)
        loss = F.l1_loss(prediction, target)
        value = loss.detach().item()
        if not np.isfinite(value):
            raise RuntimeError(f'Difix loss is non-finite at step {step}')
        if first_request:
            audit = self.result / 'difix_samples'
            audit.mkdir()
            for name in ('render.png', 'reference.png', 'target.png'):
                (audit / name).write_bytes((exchange / name).read_bytes())
        with (self.result / 'difix_loss.jsonl').open('a') as log:
            log.write(json.dumps({'step': step, 'l1': value, 'weight': self.settings['weight']}) + '\n')
        return self.settings['weight'] * loss
