"""Compare all CPU upload bytes against the original normalized preprocessing path."""
import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

os.environ.setdefault('DEV', 'CPU')


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
  compact = module('compact-inputs').CompactInputs
  settings = SimpleNamespace(max_tokens=64, vision_batch=1, byte_pixels=True)
  for profile in ({}, {'history_pixel_scale':.5}, {'history_feature_scale':.5}):
    adapters = [fast(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True,
                     cache_resize=True, direct_byte_inputs=direct, **profile) for direct in (False,True)]
    times = [[],[]]
    for sample in args.samples:
      data = load(sample)
      # Repeating a timestamp with changed pixels must invalidate every source cache.
      changed = dict(data, image_frames=data['image_frames'].clone())
      changed['image_frames'][0,...] ^= 255
      for source in (data, data, changed, data):
        outputs = []
        for index, adapter in enumerate(adapters):
          started = time.perf_counter()
          _, _, prepared = adapter.prepare(source)
          packed = adapter.native_inputs(prepared)
          outputs.append(compact.arrays(settings, prepared, packed))
          times[index].append((time.perf_counter()-started)*1000)
        for expected, actual in zip(*outputs, strict=True):
          np.testing.assert_array_equal(actual, expected)
    print(f'{profile or "full resolution"}: all upload bytes exact; cached and changed-source checks passed; '
          f'CPU medians {np.median(times[0]):.2f} -> {np.median(times[1]):.2f} ms', flush=True)


if __name__ == '__main__':
  main()
