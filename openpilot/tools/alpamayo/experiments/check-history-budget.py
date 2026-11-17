"""Check current-image preservation and exact cached preprocessing across history budgets."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--samples', type=Path, nargs='+', required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  fast, load = module('fast-inputs').FastInputs, module('nvidia-reference').load_raw
  full = fast(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True)
  for scale in (.5, .25):
    plain = fast(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True, history_pixel_scale=scale)
    cached = fast(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True, history_pixel_scale=scale, cache_resize=True)
    for sample in args.samples:
      data = load(sample)
      _, _, baseline = full.prepare(data)
      _, _, actual = plain.prepare(data)
      n = len(data['camera_indices'])
      for shuffled in (False, True):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(328)) if shuffled else torch.arange(n)
        candidate = dict(data, **{key:data[key][order] for key in ('image_frames','camera_indices','absolute_timestamps')})
        _, _, reused = cached.prepare(candidate)
        for key in ('ego_history_xyz', 'ego_history_rot'):
          assert torch.equal(actual[key], baseline[key]) and torch.equal(actual[key], reused[key])
        for key, value in actual['tokenized_data'].items():
          assert torch.equal(value, reused['tokenized_data'][key]), (sample, scale, shuffled, key)
        for value, reference in zip(plain.native_inputs(actual), cached.native_inputs(reused), strict=True):
          np.testing.assert_array_equal(value, reference)
      base, result = baseline['tokenized_data'], actual['tokenized_data']
      offsets = [np.concatenate(([0], t['image_grid_thw'].prod(dim=1).cumsum(dim=0).numpy())) for t in (base,result)]
      times = data['absolute_timestamps'].flatten()
      latest = {int(c):int(times[data['camera_indices']==c].max()) for c in data['camera_indices'].unique()}
      for i, (camera, timestamp) in enumerate(zip(data['camera_indices'].tolist(), times.tolist(), strict=True)):
        if timestamp == latest[camera]:
          assert torch.equal(base['image_grid_thw'][i], result['image_grid_thw'][i])
          assert torch.equal(base['pixel_values'][offsets[0][i]:offsets[0][i+1]],
                             result['pixel_values'][offsets[1][i]:offsets[1][i+1]])
        else:
          assert result['image_grid_thw'][i].prod() < base['image_grid_thw'][i].prod()
      print(json.dumps(dict(sample=str(sample), history_pixel_scale=scale, exact_current_images=True,
                            exact_cached_inputs=True, prompt_tokens=result['input_ids'].shape[1],
                            grids=result['image_grid_thw'].tolist())), flush=True)


if __name__ == '__main__':
  main()
