"""WorldExpand release launcher. External resources are supplied by JSON or WE_* env vars."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import REPO_ROOT, expected_outputs, load_config, validate_case_id


def runtime_env(paths):
    env = os.environ.copy()
    # Resolve project modules only from this release, even in a developer shell.
    env['PYTHONPATH'] = str(REPO_ROOT)
    env['REPO_ROOT'] = str(REPO_ROOT)
    env['PYTHONUNBUFFERED'] = '1'
    env['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    env['MODEL_PATH'] = paths['model']
    if paths.get('cache'):
        env['HF_HOME'] = paths['cache']
        env['HUGGINGFACE_HUB_CACHE'] = str(Path(paths['cache']) / 'hub')
    env.setdefault('HF_HUB_OFFLINE', '1')
    env.setdefault('TRANSFORMERS_OFFLINE', '1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    return env


def need(paths, key, directory=False):
    if not paths.get(key):
        raise ValueError(f'Missing paths.{key} in external config')
    p = Path(paths[key])
    if not (p.is_dir() if directory else p.is_file()):
        raise FileNotFoundError(f'paths.{key}: {p}')
    return p


def prepare_inference(cfg, output, case_ids, overrides):
    """Validate files and materialize train-compatible runtime YAML before GPU startup."""
    from omegaconf import OmegaConf
    paths = cfg['paths']
    model = need(paths, 'model', True)
    checkpoint = need(paths, 'checkpoint', True)
    training_config = need(paths, 'training_config')
    starts, warps = need(paths, 'start_root', True), need(paths, 'warp_root', True)
    if not ((checkpoint / 'lora').is_dir() or (checkpoint / 'transformer').is_dir()):
        raise ValueError('Checkpoint must contain lora/ or transformer/')
    train_cfg = OmegaConf.load(training_config)
    meta_file = checkpoint / 'meta.json'
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    for key in ('num_views', 'use_plucker', 'use_pose_temb', 'use_pose_film',
                'use_dist_temb', 'use_mask_encoder', 'use_warp', 'attention_policy',
                'local_attn_variant'):
        if key in meta and key in train_cfg and meta[key] != train_cfg[key]:
            raise ValueError(f'Checkpoint/config mismatch for {key}: {meta[key]} != {train_cfg[key]}')
    views = int(train_cfg.get('num_views', 4))
    for case in case_ids:
        validate_case_id(case)
        required = [starts / case / 'origin.png']
        required += [warps / case / f'{kind}_aux_{v}.png'
                     for v in range(views) for kind in ('erp', 'mask')]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f'Missing example input: {path}')
    train_cfg.pretrained_model_name_or_path = str(model)
    train_cfg.output_dir = str(output)
    # Wild mode reads the provided images; it never opens a training manifest.
    train_cfg.data_config.train_files = []
    if paths.get('data_root'):
        train_cfg.data_config.data_root = paths['data_root']
    train_cfg.resume_from_checkpoint = None
    params = dict(cfg.get('inference', {}))
    params.update({k: v for k, v in overrides.items() if v is not None})
    steps = int(params.get('steps', 40))
    height = int(params.get('height', train_cfg.data_config.get('img_size', 1024)))
    width = int(params.get('width', height * 2))
    if steps < 1 or height < 16 or width != height * 2 or height % 16:
        raise ValueError('Use steps >= 1 and ERP width = 2 * height, height divisible by 16')
    # The circularly padded wild path derives the latent size from its input
    # panorama and warp tensors, so all its streams must use the requested size.
    train_cfg.data_config.img_size = height
    runtime_yaml = output / 'runtime.yaml'
    OmegaConf.save(train_cfg, runtime_yaml)
    ids = output / 'case_ids.txt'
    ids.write_text(''.join(c + '\n' for c in case_ids))
    flow = float(meta.get('flow_shift', train_cfg.get('flow_shift', 1.0)))
    argv = [paths.get('python', sys.executable), '-m', 'inference.batch_demo_pano_four_view',
            '--wild', '--start_root', str(starts), '--warp_root', str(warps),
            '--ckpt_path', str(checkpoint), '--config', str(runtime_yaml),
            '--model_path', str(model), '--output_path', str(output),
            '--ids_txt', str(ids), '--num_gpus', str(int(params.get('num_gpus', 1))),
            '--height', str(height), '--width', str(width),
            '--num_inference_steps', str(steps), '--flow_shift', str(flow),
            '--true_cfg_scale', str(float(params.get('true_cfg', 4.0))),
            '--guidance_scale', str(float(params.get('guidance', 1.0)))]
    record = {'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'config': cfg, 'command': argv, 'repo_root': str(REPO_ROOT),
              'cases': case_ids, 'num_views': views, 'steps': steps,
              'height': height, 'width': width, 'flow_shift': flow,
              'checkpoint_meta': meta}
    return argv, record


def verify_outputs(output, record):
    from PIL import Image, ImageStat
    checked = []
    for path in expected_outputs(output, record['cases'], record['num_views']):
        if not path.is_file():
            raise RuntimeError(f'Inference incomplete: missing {path}. See inference.log for worker errors.')
        with Image.open(path) as image:
            image.load()
            if image.size != (record['width'], record['height']):
                raise RuntimeError(f'Unexpected output resolution: {path}: {image.size}')
            stddev = ImageStat.Stat(image.convert('RGB')).stddev
            if max(stddev) < 1:
                raise RuntimeError(f'Output image is blank/constant: {path}')
            checked.append({'path': str(path), 'size': list(image.size), 'stddev': stddev})
    return checked


def run_inference(args, cfg):
    paths = cfg['paths']
    output = Path(args.output or paths.get('output', REPO_ROOT / 'outputs/example')).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_ids = args.case_id or cfg.get('case_ids', ['scene-01'])
    argv, record = prepare_inference(cfg, output, case_ids,
                                    {'steps': args.steps, 'height': args.height, 'width': args.width,
                                     'num_gpus': args.num_gpus})
    (output / 'resolved.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'command': argv, 'output': str(output)}, indent=2), flush=True)
    if args.dry_run:
        return
    env = runtime_env(paths)
    env['WE_PATH_CONFIG'] = cfg['_config_file']
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    t0 = time.monotonic()
    report = {'status': 'running', 'cases': case_ids}
    report_path = output / 'result.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        with (output / 'inference.log').open('w') as log:
            proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if proc.wait() != 0:
                    raise RuntimeError(f'Inference process exited with code {proc.returncode}')
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
        report.update(status='passed', outputs=verify_outputs(output, record))
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - t0, 3)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(report["outputs"])} panoramas verified. Report: {report_path}', flush=True)


def run_training(args, cfg):
    from omegaconf import OmegaConf
    paths = cfg['paths']
    need(paths, 'model', True)
    profile = need(paths, 'training_config')
    manifest = need(paths, 'train_json')
    data_root = need(paths, 'data_root', True)
    output = Path(args.output or paths.get('train_output', REPO_ROOT / 'outputs/train')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = OmegaConf.load(profile)
    profile.pretrained_model_name_or_path = paths['model']
    profile.output_dir = str(output)
    profile.data_config.train_files = [str(manifest)]
    profile.data_config.data_root = str(data_root)
    profile.resume_from_checkpoint = paths.get('resume') or None
    if args.max_steps is not None:
        profile.max_train_steps = args.max_steps
    runtime = output / 'runtime.yaml'
    OmegaConf.save(profile, runtime)
    accelerate = paths.get('accelerate', 'accelerate')
    argv = [accelerate, 'launch']
    if paths.get('accelerate_config'):
        argv += ['--config_file', paths['accelerate_config']]
    argv += ['--num_processes', str(args.num_gpus or 1),
             str(REPO_ROOT / 'training/train_four_view_online.py'), '--config', str(runtime)]
    (output / 'resolved.json').write_text(json.dumps({'command': argv, 'config': cfg}, indent=2) + '\n')
    print(json.dumps(argv, indent=2), flush=True)
    if not args.dry_run:
        env = runtime_env(paths)
        sources = ('training_moge_source', 'training_utils3d_source')
        if any(paths.get(key) and Path(paths[key]).exists() for key in sources):
            # Keep the training renderer's utils3d API local to this subprocess.
            source_paths = [str(need(paths, key, True)) for key in sources]
            env['PYTHONPATH'] = os.pathsep.join([env['PYTHONPATH'], *source_paths])
        env['WE_PATH_CONFIG'] = cfg['_config_file']
        if args.gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = args.gpu
        subprocess.run(argv, cwd=REPO_ROOT, env=env, check=True)


def doctor(cfg):
    paths = cfg['paths']
    for key in ('model', 'checkpoint', 'training_config', 'start_root', 'warp_root'):
        need(paths, key, key != 'training_config')
        print(f'OK {key}: {paths[key]}')
    required_model = ['scheduler/scheduler_config.json', 'tokenizer/tokenizer_config.json',
                      'model_index.json', 'vae/config.json', 'text_encoder/config.json']
    for rel in required_model:
        if not (Path(paths['model']) / rel).is_file():
            raise FileNotFoundError(f'Missing base model component: {rel}')
    print(f'OK release root: {REPO_ROOT}')
    print('Run infer --dry-run for checkpoint parity and per-case input validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Run scripts/run.py expand --help for the single-panorama 3DGS workflow.')
    parser.add_argument('action', choices=['infer', 'train', 'doctor'])
    parser.add_argument('--config', type=Path, default=os.environ.get('WE_CONFIG_FILE', REPO_ROOT / 'configs/release/local.json'))
    parser.add_argument('--output', help='Output directory (overrides paths.output / train_output)')
    parser.add_argument('--case-id', action='append', help='Repeat for multiple cases')
    parser.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1')
    parser.add_argument('--num-gpus', type=int)
    parser.add_argument('--steps', type=int, help='Diffusion steps')
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--max-steps', type=int, help='Training steps')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == 'infer':
        run_inference(args, cfg)
    elif args.action == 'train':
        run_training(args, cfg)
    else:
        doctor(cfg)


if __name__ == '__main__':
    main()
