"""Compare changing replay inputs against the pinned adapter, including disk-read latency."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
import inputs as reference


def check(a, b):
  if isinstance(a, torch.Tensor):
    assert torch.equal(a, b), ('tensor mismatch', a.shape)
  elif isinstance(a, np.ndarray):
    assert np.array_equal(a, b), ('array mismatch', a.shape)
  elif isinstance(a, dict):
    assert a.keys() == b.keys()
    for key in a:
      check(a[key], b[key])
  elif isinstance(a, (list, tuple)):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
      check(x, y)
  else:
    assert a == b


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--samples', nargs='+', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  spec = importlib.util.spec_from_file_location('fast_inputs', Path(__file__).with_name('fast-inputs.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  started = time.perf_counter()
  adapter = module.FastInputs(args.checkpoint)
  report = {'initialization_ms': (time.perf_counter()-started)*1000, 'checks': []}
  for path in args.samples*2:
    started = time.perf_counter()
    data = reference.load_sample(path)
    loaded = time.perf_counter()
    c, t, expected = reference.prepare(args.checkpoint, data, 'openpilot')
    packed = reference.native_inputs(args.checkpoint, c, expected)
    reference_ms = (time.perf_counter()-loaded)*1000
    begin = time.perf_counter()
    _, _, actual = adapter.prepare(data)
    fast_packed = adapter.native_inputs(actual)
    fast_ms = (time.perf_counter()-begin)*1000
    check(dict(expected), dict(actual))
    check(packed, fast_packed)
    item = {'sample': str(path), 'load_ms': (loaded-started)*1000, 'reference_prepare_ms': reference_ms,
            'cached_prepare_ms': fast_ms, 'exact_match': True}
    report['checks'].append(item)
    print(json.dumps(item), flush=True)
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
