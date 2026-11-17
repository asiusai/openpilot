"""Check full-resolution vision inputs, compressed token layouts, and feature interpolation."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('DEV', 'CPU')
from tinygrad import Tensor
import importlib.util


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--sample', type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  fast = module('fast-inputs').FastInputs
  data = module('nvidia-reference').load_raw(args.sample)
  full = fast(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True, cache_resize=True)
  _, _, baseline = full.prepare(data)
  native = full.native_inputs(baseline)
  base = baseline['tokenized_data']
  for scale in (.5, .25):
    adapter = fast(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True, cache_resize=True,
                   history_feature_scale=scale)
    _, _, prepared = adapter.prepare(data)
    result = prepared['tokenized_data']
    for key in ('pixel_values', 'image_grid_thw'):
      assert torch.equal(result[key], base[key])
    for key in ('ego_history_xyz', 'ego_history_rot'):
      assert torch.equal(prepared[key], baseline[key])
    image_id = adapter.config.vlm_config.image_token_id
    assert torch.equal(result['input_ids'][result['input_ids'] != image_id], base['input_ids'][base['input_ids'] != image_id])
    counts = (result['embedding_grid_thw'].prod(dim=1)//4).sum()
    assert int((result['input_ids']==image_id).sum()) == int(counts)
    np.testing.assert_array_equal(adapter.native_inputs(prepared)[3], native[3])
    times = data['absolute_timestamps'].flatten()
    for camera in data['camera_indices'].unique():
      i = int(torch.nonzero((data['camera_indices']==camera) & (times==times[data['camera_indices']==camera].max()))[0])
      assert torch.equal(result['embedding_grid_thw'][i], base['image_grid_thw'][i])
    order = torch.randperm(len(times), generator=torch.Generator().manual_seed(19))
    shuffled = dict(data, **{key:data[key][order] for key in ('image_frames','camera_indices','absolute_timestamps')})
    _, _, again = adapter.prepare(shuffled)
    for key in result:
      assert torch.equal(result[key], again['tokenized_data'][key])
    print(f'Feature scale {scale}: full pixels, original text, current features, camera ordering, and vision positions preserved')
  cache = module('vision-cache').VisionCache
  rng = np.random.default_rng(76)
  features = rng.normal(size=(180,32)).astype(np.float32)
  for size in ((7,13), (5,9)):
    actual = cache.resample_feature(Tensor(features), (1,20,36), (1,size[0]*2,size[1]*2)).numpy()
    expected = torch.nn.functional.interpolate(torch.from_numpy(features).reshape(1,10,18,32).permute(0,3,1,2),
                  size=size, mode='bilinear', align_corners=False).permute(0,2,3,1).reshape(-1,32).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-6)
  print('Feature interpolation matches independent PyTorch bilinear sampling')


if __name__ == '__main__':
  main()
