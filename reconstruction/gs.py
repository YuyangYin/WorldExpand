"""Run the HY-World 2.0 Gaussian optimizer with optional Difix supervision."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def add_difix_hook(source):
    """Attach the Difix worker at the public trainer's loss and lifecycle boundaries."""
    hooks = (
        ('from gs.opencv import Dataset, Parser\n',
         'from gs.opencv import Dataset, Parser\nfrom reconstruction.difix_loss import DifixLoss\n'),
        ('        max_steps = cfg.max_steps\n        init_step = 0\n',
         '        difix = DifixLoss.from_env(self)\n\n'
         '        max_steps = cfg.max_steps\n        init_step = 0\n'),
        ('            loss.backward()\n\n            desc =',
         '            if difix is not None and difix.active(step):\n'
         '                loss = loss + difix(self, step, camtoworlds, Ks, pixels, image_ids, cur_mask_gate)\n\n'
         '            loss.backward()\n\n            desc ='),
        ('    @torch.no_grad()\n    def eval(self,',
         '        if difix is not None:\n            difix.close()\n\n'
         '    @torch.no_grad()\n    def eval(self,'),
    )
    for original, updated in hooks:
        if source.count(original) != 1:
            raise ValueError('HY-World trainer does not match the pinned Difix hook locations')
        source = source.replace(original, updated)
    return source


def fit(source, data, output, steps, max_points, difix_settings=None):
    release = Path(__file__).resolve().parents[1]
    worldgen = source / 'hyworld2/worldgen'
    trainer = worldgen / 'world_gs_trainer.py'
    output.mkdir(parents=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(map(str, (release, worldgen, source)))
    env.pop('WE_DIFIX_SETTINGS', None)
    if difix_settings is not None:
        text = trainer.read_text()
        patched = output / 'world_gs_trainer_difix.py'
        patched.write_text(add_difix_hook(text))
        env['WE_DIFIX_SETTINGS'] = json.dumps(difix_settings)
        trainer = patched
    args = [
        sys.executable, str(trainer), 'default', '--data-dir', str(data),
        '--result-dir', str(output), '--max-steps', str(steps),
        '--save-steps', str(steps), '--ply-steps', str(steps), '--eval-steps', str(steps),
        '--save-ply', '--disable-viewer', '--disable-video', '--no-normalize',
        '--depth-loss', '--depth-lambda1', '0.2', '--depth-lambda2', '0.01',
        '--use-mask-gaussian', '--antialiased', '--downsample-pts-num', str(max_points),
        '--strategy.refine-start-iter', '150', '--strategy.refine-stop-iter', '750',
        '--strategy.refine-every', '100', '--strategy.reset-every', '99990',
        '--use-scale-regularization', '--max-gauss-ratio', '5',
    ]
    (output / 'command.json').write_text(json.dumps({'argv': args, 'difix': difix_settings}, indent=2) + '\n')
    subprocess.run(args, cwd=worldgen, env=env, check=True)
    ply = output / 'ply' / f'point_cloud_{steps - 1}.ply'
    checkpoint = output / 'ckpts' / f'ckpt_{steps - 1}_rank0.pt'
    for artifact in (ply, checkpoint):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f'3DGS trainer did not write {artifact}')
    from plyfile import PlyData

    vertices = PlyData.read(str(ply))['vertex']
    if not len(vertices) or not {'f_dc_0', 'opacity', 'scale_0', 'rot_0'} <= set(vertices.data.dtype.names):
        raise RuntimeError(f'3DGS export has no Gaussian vertices or is missing splat attributes: {ply}')
    expected = [] if difix_settings is None else list(
        range(difix_settings['start'], steps, difix_settings['every']))
    log = output / 'difix_loss.jsonl'
    actual = [json.loads(line)['step'] for line in log.read_text().splitlines()] if log.is_file() else []
    if actual != expected:
        raise RuntimeError(f'Difix ran at steps {actual}, expected {expected}')
    return {'ply': str(ply), 'checkpoint': str(checkpoint), 'gaussians': len(vertices), 'difix_steps': actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'data', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-points', type=int, required=True)
    parser.add_argument('--difix-settings', type=json.loads)
    args = parser.parse_args()
    source = args.source / 'hyworld2/worldgen/world_gs_trainer.py'
    # Persist the public source hash even when the Difix hook is enabled.
    print(f'HY-World trainer SHA256 {hashlib.sha256(source.read_bytes()).hexdigest()}', flush=True)
    result = fit(**vars(args))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
