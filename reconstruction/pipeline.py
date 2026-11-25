"""Run single-panorama WorldExpand through optimized 3D Gaussian splats."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from worldexpand.cli import need, runtime_env
from worldexpand.config import REPO_ROOT, load_config, validate_case_id


def resolve_caption(image, explicit_caption):
    if explicit_caption is not None:
        return explicit_caption
    caption_file = image.with_name('caption.txt')
    if caption_file.is_file():
        return caption_file.read_text(encoding='utf-8').strip()
    return None


def plan(args, config, output):
    paths = config['paths']
    python = paths.get('python', sys.executable)
    if not os.access(python, os.X_OK):
        raise ValueError(f'Python is not executable: {python}')
    for key in ('model', 'checkpoint', 'depthmaster_source', 'hyworld_source', 'worldmirror_model'):
        need(paths, key, True)
    for key in ('depthmaster_weights', 'training_config'):
        need(paths, key)
    for key, rel in (('depthmaster_source', 'eval_panorama/eval.py'),
                     ('hyworld_source', 'hyworld2/worldrecon/pipeline.py'),
                     ('hyworld_source', 'hyworld2/worldgen/world_gs_trainer.py'),
                     ('worldmirror_model', 'config.json'),
                     ('worldmirror_model', 'model.safetensors')):
        if not (Path(paths[key]) / rel).is_file():
            raise FileNotFoundError(f'paths.{key} must contain {rel}')

    if paths.get('torch_home'):
        vgg = Path(paths['torch_home']) / 'hub/checkpoints/vgg16-397923af.pth'
        if not vgg.is_file():
            raise FileNotFoundError(f'LPIPS VGG16 checkpoint is missing: {vgg}')

    image = args.image.resolve()
    case = validate_case_id(args.case_id)
    anchor = output / 'start' / case / 'origin.png'
    depth = output / 'depth' / 'radial_depth.npy'
    warps = output / 'warp'
    nvs = output / 'nvs'
    reconstruction = output / 'reconstruction'
    gs = output / 'gaussians'
    stages = {
        'depth': [python, '-m', 'reconstruction.depth', '--image', str(anchor),
                  '--source', paths['depthmaster_source'], '--checkpoint', paths['depthmaster_weights'],
                  '--output', str(depth)],
        'warp': [python, '-m', 'reconstruction.warp', '--image', str(anchor), '--depth-path', str(depth),
                 '--output', str(warps / case), '--height', str(args.height),
                 '--max-dist-ratio', str(args.max_dist_ratio)],
        'nvs': [python, str(REPO_ROOT / 'scripts/run.py'), 'infer', '--config', config['_config_file'],
                '--case-id', case, '--num-gpus', '1', '--steps', str(args.nvs_steps),
                '--height', str(args.height), '--width', str(2 * args.height), '--output', str(nvs)],
        'worldmirror': [python, '-m', 'reconstruction.mirror', '--anchor', str(anchor),
                        '--warp', str(warps / case), '--nvs', str(nvs / case), '--depth', str(depth),
                        '--source', paths['hyworld_source'], '--model', paths['worldmirror_model'],
                        '--output', str(reconstruction), '--faces', str(args.mirror_faces),
                        '--size', str(args.mirror_size)],
        'gaussians': [python, '-m', 'reconstruction.gs', '--source', paths['hyworld_source'],
                      '--data', str(reconstruction / 'gs_data'), '--output', str(gs),
                      '--steps', str(args.gs_steps), '--max-points', str(args.max_points)],
    }
    if args.difix:
        difix_python = need(paths, 'difix_python')
        difix_source = need(paths, 'difix_source', True)
        difix_weights = need(paths, 'difix_weights', True)
        if not os.access(difix_python, os.X_OK):
            raise ValueError(f'Difix Python is not executable: {difix_python}')
        for file in (difix_source / 'src/pipeline_difix.py', difix_weights / 'model_index.json',
                     difix_weights / 'unet/diffusion_pytorch_model.safetensors',
                     difix_weights / 'vae/diffusion_pytorch_model.safetensors',
                     difix_weights / 'text_encoder/model.safetensors'):
            if not file.is_file():
                raise FileNotFoundError(f'Missing Difix3D resource: {file}')
        settings = {'python': str(difix_python), 'source': str(difix_source),
                    'weights': str(difix_weights), 'start': args.difix_start,
                    'every': args.difix_every, 'weight': args.difix_weight,
                    'size': args.difix_size, 'gpu': args.difix_gpu}
        stages['gaussians'] += ['--difix-settings', json.dumps(settings)]

    return {'image': str(image), 'case_id': case, 'gpu': args.gpu,
            'anchor': str(anchor), 'output': str(output), 'difix': args.difix,
            'stages': stages}


def run(args):
    if (min(args.height, args.mirror_size) < 16 or args.height % 16 or args.mirror_size % 14
            or args.mirror_size < 224 or min(args.nvs_steps, args.gs_steps, args.max_points) < 1
            or not 0 < args.max_dist_ratio <= 1):
        raise ValueError('Use ERP height divisible by 16, Mirror size >= 224 divisible by 14, '
                         'positive step/point counts and max-dist-ratio in (0, 1]')
    if args.difix and (not 0 <= args.difix_start < args.gs_steps or args.difix_every < 1
                       or not 0 < args.difix_weight < float('inf')
                       or args.difix_size < 64 or args.difix_size % 8):
        raise ValueError('Difix needs 0 <= start < gs-steps, positive cadence and weight, '
                         'and size >= 64 divisible by 8')
    config = load_config(args.config)
    output = args.output.resolve()
    record = plan(args, config, output)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    record['caption'] = resolve_caption(args.image, args.caption)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return

    from PIL import Image
    from .erp import open_erp

    rgb = open_erp(args.image)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f'Expansion output must be empty: {output}')
    anchor = Path(record['anchor'])
    anchor.parent.mkdir(parents=True)
    Image.fromarray(rgb).save(anchor)
    if record['caption'] is not None:
        (anchor.parent / 'caption.txt').write_text(record['caption'] + '\n', encoding='utf-8')
    (output / 'commands.json').write_text(json.dumps(record, indent=2) + '\n')
    report = {'status': 'running', 'stages': {}}
    result_path = output / 'result.json'
    env = runtime_env(config['paths'])
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, WE_START_ROOT=str(output / 'start'),
               WE_WARP_ROOT=str(output / 'warp'), WE_PYTHON=record['stages']['depth'][0])
    env.setdefault('OMP_NUM_THREADS', '4')
    if config['paths'].get('torch_home'):
        env['TORCH_HOME'] = config['paths']['torch_home']

    try:
        for stage, command in record['stages'].items():
            start = time.monotonic()
            log = output / f'{stage}.log'
            report['stages'][stage] = {'status': 'running', 'log': log.name}
            result_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'[{stage}] {log}', flush=True)
            try:
                with log.open('w') as handle:
                    subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True)
            except BaseException:
                report['stages'][stage]['status'] = 'failed'
                raise
            finally:
                report['stages'][stage]['elapsed_seconds'] = round(time.monotonic() - start, 2)
            report['stages'][stage]['status'] = 'passed'
        report['artifacts'] = json.loads((output / 'gaussians/result.json').read_text())
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        result_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS {result_path}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scripts/run.py expand', description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=os.environ.get('WE_CONFIG_FILE', REPO_ROOT / 'configs/release/local.json'))
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-id', default='scene')
    parser.add_argument('--caption', help='Text prompt override for caption.txt beside the input panorama')
    parser.add_argument('--gpu', default='0', help='One device for depth, NVS, WorldMirror and GS')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--max-dist-ratio', type=float, default=0.5)
    parser.add_argument('--nvs-steps', type=int, default=40)
    parser.add_argument('--mirror-faces', type=int, choices=[6, 20], default=20)
    parser.add_argument('--mirror-size', type=int, default=504)
    parser.add_argument('--gs-steps', type=int, default=1500)
    parser.add_argument('--max-points', type=int, default=1_500_000)
    parser.add_argument('--difix', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--difix-start', type=int, default=1000, help='Zero-based 3DGS step')
    parser.add_argument('--difix-every', type=int, default=5)
    parser.add_argument('--difix-weight', type=float, default=0.1)
    parser.add_argument('--difix-size', type=int, default=512)
    parser.add_argument('--difix-gpu', help='Optional separate device for the Difix worker')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if ',' in args.gpu or args.difix_gpu is not None and ',' in args.difix_gpu:
        parser.error('Specify one GPU per process')
    run(args)


if __name__ == '__main__':
    main()
