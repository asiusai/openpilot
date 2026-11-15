"""Compare real six-camera preprocessing with the official NVIDIA adapter."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import inputs as reference
from alpamayo2_edge.helper import prepare_model_inputs


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--sample', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--scales', nargs='+', type=float, default=(1.0,0.5))
  parser.add_argument('--camera-pixel-scales', nargs=6, type=float)
  parser.add_argument('--mmap-inputs', action='store_true')
  args = parser.parse_args()
  torch.set_num_threads(4)
  data = module('nvidia-reference').load_raw(args.sample)
  candidate_data = dict(data)
  if args.mmap_inputs:
    candidate_data.update({path.stem:torch.from_numpy(np.load(path,allow_pickle=False,mmap_mode='c'))
                           for path in args.sample.glob('*.npy') if path.stem != 'nav_text'})
  results = []
  for scale in args.scales:
    adapter = module('fast-inputs').FastInputs(args.checkpoint, camera_profile='nvidia', pixel_scale=scale, cache_resize=True,
                                             camera_pixel_scales=args.camera_pixel_scales)
    official = prepare_model_inputs(dict(data,camera_pixel_bounds=adapter.pixel_bounds),adapter.config,adapter.tokenizer)
    native = reference.native_inputs(args.checkpoint,adapter.config,official)
    for shuffled in (False,True):
      candidate = dict(candidate_data)
      if shuffled:
        order = torch.randperm(22,generator=torch.Generator().manual_seed(1234))
        candidate.update({key:candidate_data[key][order] for key in ('image_frames','camera_indices','absolute_timestamps')})
      _,_,actual = adapter.prepare(candidate)
      assert actual.keys() == official.keys()
      for key in actual:
        pairs = actual[key].items() if isinstance(actual[key],dict) else [(None,actual[key])]
        for name,value in pairs:
          expected = official[key][name] if name else official[key]
          assert torch.equal(value,expected), (scale,shuffled,key,name)
      for value,expected in zip(adapter.native_inputs(actual),native,strict=True):
        np.testing.assert_array_equal(value,expected)
      result = dict(pixel_scale=scale,camera_pixel_scales=args.camera_pixel_scales,shuffled=shuffled,mmap_inputs=args.mmap_inputs,exact=True,
                    tokens=actual['tokenized_data']['input_ids'].shape[1],
                    grids=actual['tokenized_data']['image_grid_thw'].tolist())
      results.append(result)
      print(json.dumps(result),flush=True)
  args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
  main()
