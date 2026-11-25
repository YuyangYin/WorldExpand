"""Run the official reference-conditioned Difix3D pipeline in its own environment."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from PIL import Image


def serve(source, weights, exchange, size):
    import torch

    sys.path.insert(0, str(source / 'src'))
    from pipeline_difix import DifixPipeline

    with redirect_stdout(sys.stderr):
        model = DifixPipeline.from_pretrained(str(weights), trust_remote_code=True,
                                              torch_dtype=torch.float16, local_files_only=True)
        model.to('cuda:0')
        model.set_progress_bar_config(disable=True)
    for line in sys.stdin:
        request = json.loads(line)
        with redirect_stdout(sys.stderr):
            with Image.open(exchange / 'render.png') as image:
                rendered = image.convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
            with Image.open(exchange / 'reference.png') as image:
                reference = image.convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
            generator = torch.Generator(device='cuda:0').manual_seed(request['step'])
            result = model('remove degradation', image=rendered, ref_image=reference,
                           width=size, height=size, num_inference_steps=1, timesteps=[199],
                           guidance_scale=0.0, generator=generator).images[0]
            result.save(exchange / 'target.png')
        print(json.dumps({'step': request['step']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'weights', 'exchange'):
        parser.add_argument(f'--{key}', required=True, type=Path)
    parser.add_argument('--size', required=True, type=int)
    serve(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
