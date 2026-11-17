"""Measure paired CPU input preparation using the recorded five-hertz invocation."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

DIRECTORY = Path(__file__).resolve().parent


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--check', type=Path, required=True)
  parser.add_argument('--invocation', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--disable-gc', action='store_true')
  args = parser.parse_args()
  if not args.output.resolve().is_relative_to(DIRECTORY) or not args.output.name.startswith('agent-warp-fast-'):
    raise ValueError('Write only within the owned experiment prefix')
  helper = load('warp_fast_check_helpers', DIRECTORY/'agent-warp-fast-check.py')
  validated = json.loads(args.check.read_text())
  if helper.source_hashes() != validated['source_sha256']:
    raise ValueError('Sampler sources differ from the validated check')
  config = validated['variants'][0]
  manifest, calibrations, raw_path, manifest_path, calibration_path, provenance = helper.inputs(args.invocation)
  invocation = args.invocation
  while True:
    command = json.loads(invocation.read_text())['command']
    if '--samples' in command:
      break
    invocation = Path(command[command.index('--invocation')+1])
  samples = []
  for value in command[command.index('--samples')+1:]:
    if value.startswith('--'):
      break
    samples.append(value)
  baseline_library = helper.fast.build(baseline=True)
  helper.native_warp.library = lambda: baseline_library
  traces = []
  collections = []
  collection_start = {}
  def record_gc(phase, info):
    if phase == 'start':
      collection_start[info['generation']] = time.perf_counter_ns()
    else:
      collections.append(dict(generation=info['generation'], duration_ms=(time.perf_counter_ns()-collection_start[info['generation']])/1e6))
  gc.callbacks.append(record_gc)
  if args.disable_gc:
    gc.disable()
  for trial in range(2):
    helper.native_warp.maps.cache_clear()
    helper.cached_calibrated_coordinates.cache_clear()
    sampler = helper.fast.Sampler(config['kind'], config['workers'], Path(config['provenance']['command'][0]).name,
                                  config['provenance']['native'], config['max_maps'], config['max_bytes'])
    old_module = load('warp_fast_reference_stream', DIRECTORY/'stream-inputs.py')
    new_module = load('warp_fast_candidate_stream', DIRECTORY/'stream-inputs.py')
    new_module.calibrated_image = sampler.calibrated_image
    new_module.library = lambda: sampler.lib
    preparers = {'reference':old_module.NativeFrames(manifest_path, calibration_path, raw_path),
                 'candidate':new_module.NativeFrames(manifest_path, calibration_path, raw_path)}
    try:
      for index, sample in enumerate(samples):
        order = ('reference', 'candidate') if (index+trial) % 2 == 0 else ('candidate', 'reference')
        results = {}
        diagnostics = {}
        for mode in order:
          before_gc = len(collections)
          before_maps = sampler.misses
          results[mode] = preparers[mode].prepare(sample)
          diagnostics[mode] = dict(gc=collections[before_gc:], candidate_map_misses=sampler.misses-before_maps)
        np.testing.assert_array_equal(results['reference'][0], results['candidate'][0])
        old, new = results['reference'][1], results['candidate'][1]
        assert old['calibrated_frame_misses'] == new['calibrated_frame_misses']
        traces.append(dict(trial=trial, sample=sample, reference=old, candidate=new, diagnostics=diagnostics))
    finally:
      sampler.close()
  grouped = {}
  for misses in sorted({r['reference']['calibrated_frame_misses'] for r in traces}):
    rows = [r for r in traces if r['reference']['calibrated_frame_misses'] == misses]
    grouped[misses] = {mode:helper.summary([r[mode]['native_calibration_ms'] for r in rows]) for mode in ('reference','candidate')}
  cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')), platform.machine())
  report = dict(scope='CPU input preparation only; no GPU, Asius, or whole pipeline claim', exact=True,
                boundary='NativeFrames.prepare includes loading, map misses, sampling, frame reuse, and output stacking',
                samples_per_trial=len(samples), trials=2, grouped_by_new_frames=grouped, traces=traces,
                input_sha256=provenance, source_sha256=helper.source_hashes(),
                check_sha256=hashlib.sha256(args.check.read_bytes()).hexdigest(),
                host=dict(cpu=cpu, machine=platform.machine(), scipy=scipy.__version__, numpy=np.__version__),
                gc_enabled=gc.isenabled(), gc_collections=collections)
  gc.callbacks.remove(record_gc)
  if args.disable_gc:
    gc.enable()
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('traces','input_sha256','source_sha256')}, indent=2))


if __name__ == '__main__':
  main()
