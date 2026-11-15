"""Run Harald's model on live snapshots; save results without publishing controls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def sample(path):
  with np.load(path, allow_pickle=False) as source:
    meta = json.loads(str(source['metadata']))
    times = source['motion_times'].astype(np.float64)/1e9
    query = meta['query_ns']/1e9
    wanted = query + np.arange(-19, 1)*.1
    if times[0] > wanted[0] or times[-1] < wanted[-1] or np.any(np.diff(times) <= 0):
      raise ValueError('Missing or unordered measurement timestamps')
    selected = (times >= wanted[0]-.1) & (times <= wanted[-1]+.1)
    if np.diff(times[selected]).max() >= .15:
      raise ValueError('Gap in measured ego history')
    rotations = Rotation.from_euler('xyz', source['motion_rpy']).as_matrix()
    velocities = np.einsum('nij,nj->ni', rotations, source['motion_velocity'])
    positions = np.concatenate((np.zeros((1, 3)), np.cumsum(np.diff(times)[:, None]*(velocities[1:]+velocities[:-1])/2, axis=0)))
    history = np.stack([np.interp(wanted, times, positions[:, axis]) for axis in range(3)], axis=-1)
    world_from_vehicle = Slerp(times, Rotation.from_matrix(rotations))(wanted).as_matrix() @ Rotation.from_euler('xyz', meta['calibration_rpy']).as_matrix()
    axes = np.diag([1., -1., -1.])
    current_from_world = axes @ world_from_vehicle[-1].T
    xyz = ((history-history[-1]) @ current_from_world.T)[None].astype(np.float32)
    rot = (current_from_world @ world_from_vehicle @ axes)[None].astype(np.float32)
    data = {'image_frames': torch.from_numpy(source['image_frames'].copy()),
            'camera_indices': torch.tensor([1]*5+[6]*5),
            'absolute_timestamps': torch.tensor([round(t/1000) for t in meta['camera_timestamps_ns']]),
            'ego_history_xyz': torch.from_numpy(xyz), 'ego_history_rot': torch.from_numpy(rot),
            'nav_text': [meta['navigation']]}
    meta['ego_history_xyz'] = xyz.tolist()
    meta['ego_history_rot'] = rot.tolist()
    return data, meta


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=Path, required=True)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--max-predictions', type=int, default=0)
  args = parser.parse_args()
  import benchmark
  from tinygrad.device import Compiled
  spec = importlib.util.spec_from_file_location('fast_loader', Path(__file__).with_name('benchmark-fast.py'))
  loader = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(loader)
  torch.set_num_threads(2)
  torch.set_num_interop_threads(1)
  Compiled.wait_timeout_ms = 120000
  settings = SimpleNamespace(max_tokens=256, diffusion_steps=10, vision_batch=2, greedy=True)
  output = args.root / 'predictions'
  output.mkdir(exist_ok=True)
  manifest = {'checkpoint': args.checkpoint, 'precision': 'fp8', 'seed': 0, 'settings': vars(settings),
              'trajectory_axes': 'forward/left/up', 'trajectory_times_s': (np.arange(1, 65)*.1).tolist(),
              'small_model_axes': 'forward/right/down', 'publisher': None,
              'source_revisions': json.loads((Path(args.checkpoint).parent / 'revisions.json').read_text()),
              'weights_index': json.loads(Path(os.environ['ALPAMAYO_FP8_CACHE'], 'index.json').read_text())['sha256']}
  (args.root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
  started = time.monotonic()
  model = benchmark.EdgeModel(args.checkpoint, precision='fp8')
  model.synchronize()
  print(f'Model loaded in {time.monotonic()-started:.2f}s', flush=True)
  inputs = None
  count = 0
  while not args.max_predictions or count < args.max_predictions:
    pending = args.root / 'pending.npz'
    if not pending.exists():
      time.sleep(.1)
      continue
    pending.replace(args.root / 'working.npz')
    try:
      data, meta = sample(args.root / 'working.npz')
    except ValueError as exc:
      print(f'Skipped input: {exc}', flush=True)
      continue
    finally:
      (args.root / 'working.npz').unlink(missing_ok=True)
    age = (time.monotonic_ns()-meta['query_ns'])/1e9
    if age > 2:
      print(f'Discarded stale queued input: {age:.2f}s', flush=True)
      continue
    begin = time.monotonic_ns()
    config, tokenizer, prepared = benchmark.prepare(args.checkpoint, data, 'openpilot')
    packed = benchmark.native_inputs(args.checkpoint, config, prepared)
    fresh = benchmark.NativeInputs(settings, config, prepared, packed)
    # Preserve upload JIT buffers across snapshots with identical layouts.
    if (inputs is not None and inputs.batches == fresh.batches and inputs.spans == fresh.spans
        and all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(inputs.cpu, fresh.cpu, strict=True))):
      for old, new in zip(inputs.cpu, fresh.cpu, strict=True):
        old._buffer().as_memoryview(allow_zero_copy=True)[:] = new._buffer().as_memoryview(allow_zero_copy=True)
    else:
      inputs = fresh
    prediction = benchmark.run_native(model, settings, config, tokenizer, prepared, packed, seed=0, inputs=inputs)
    end = time.monotonic_ns()
    result = {'input': meta, 'prediction': prediction, 'started_ns': begin, 'completed_ns': end,
              'input_age_at_start_s': age, 'prediction_age_at_completion_s': (end-meta['query_ns'])/1e9,
              'includes_startup_or_capture': count < 2, 'prompt_tokens': packed[0].shape[1]}
    temporary = output / f'{meta["query_ns"]}.tmp'
    with temporary.open('w') as stream:
      stream.write(json.dumps(result, allow_nan=False)+'\n')
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(temporary.with_suffix('.json'))
    count += 1
    print(f'Saved prediction {count}: {meta["query_ns"]}, age {(end-meta["query_ns"])/1e9:.2f}s', flush=True)


if __name__ == '__main__':
  main()
