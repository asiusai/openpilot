"""Run the pinned, unmodified NVIDIA model on frozen real camera inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.inference import run_data, _json_value
from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
from camera_budgets import pixel_bounds


def load_raw(path):
  data = {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
          else torch.from_numpy(np.load(p, allow_pickle=False)) for p in path.glob('*.npy')}
  metadata_path = path/'metadata.json'
  if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    data.update({key:metadata[key] for key in ('clip_id','t0_us','provenance') if key in metadata})
  return data


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--samples', type=Path, nargs='+', required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--steps', type=int, default=10)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--pixel-scale', type=float, default=1.0)
  parser.add_argument('--camera-pixel-scales', nargs=6, type=float,
                      help='Experimental scales: front-left, front-wide, front-right, rear-left, rear-right, front-tele')
  parser.add_argument('--threads', type=int, default=8)
  parser.add_argument('--dtype', choices=('bfloat16','float32'), default='bfloat16')
  args = parser.parse_args()
  assert 0 < args.pixel_scale <= 1
  assert args.steps > 0
  bounds = pixel_bounds(CAMERA_PIXEL_BOUNDS, args.pixel_scale, args.camera_pixel_scales)
  torch.set_num_threads(args.threads)
  args.output.mkdir(parents=True, exist_ok=True)
  # predict() creates a separate generator with this seed for the expert.
  # Save the identical initial draw for native inference; NumPy uses another RNG.
  noise = torch.randn((1,64,2), generator=torch.Generator(device='cpu').manual_seed(args.seed))
  settings = dict(seed=args.seed, top_p=1.0, top_k=1, temperature=1.0,
                  num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                  diffusion_kwargs={'inference_step':args.steps})
  print('Loading NVIDIA reference on CPU', flush=True)
  started = time.perf_counter()
  model, loading_info = load_model(str(args.checkpoint), device='cpu', dtype=args.dtype,
                                   attn_implementation='sdpa', local_files_only=True)
  report = dict(checkpoint=str(args.checkpoint), device='cpu', dtype=args.dtype,
                torch=torch.__version__, settings=settings, loading_info=loading_info,
                 pixel_scale=args.pixel_scale, camera_pixel_scales=args.camera_pixel_scales,
                 pixel_bounds=bounds, load_seconds=time.perf_counter()-started,
                noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(), samples=[])
  import alpamayo2_edge
  source = Path(alpamayo2_edge.__file__).resolve().parents[2]
  report['upstream_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
  report_path = args.output/'report.json'
  if report_path.exists():
    previous = json.loads(report_path.read_text())
    for key in ('checkpoint','device','dtype','torch','settings','pixel_scale','noise_sha256','upstream_commit'):
      assert previous[key] == report[key], f'Reference configuration differs: {key}'
    assert previous.get('camera_pixel_scales') == report['camera_pixel_scales'], 'Camera budgets differ'
    report['samples'] = previous['samples']
  np.save(args.output/'initial_noise.npy', noise.numpy(), allow_pickle=False)
  (args.output/'report.json').write_text(json.dumps(_json_value(report),indent=2)+'\n')
  print(f'Reference loaded in {report["load_seconds"]:.2f} seconds',flush=True)
  for sample in args.samples:
    data = load_raw(sample)
    data['camera_pixel_bounds'] = bounds
    print(f'Running NVIDIA reference for {sample.name}',flush=True)
    record = run_data(model,data,sample.name,args.output/sample.name,settings,device='cpu')
    metadata_path = sample/'metadata.json'
    if metadata_path.exists():
      record['input_sha256'] = json.loads(metadata_path.read_text())['sha256']
    record['nav_text'] = data['nav_text']
    report['samples'] = [item for item in report['samples'] if item['sample_id'] != sample.name] + [record]
    (args.output/'report.json').write_text(json.dumps(_json_value(report),indent=2)+'\n')


if __name__ == '__main__':
  main()
