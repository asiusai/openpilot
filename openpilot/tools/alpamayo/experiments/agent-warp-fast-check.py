"""Check exact CPU sampling, full recorded windows, and matched four-frame timing."""
import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.ndimage import map_coordinates

from calibration_trials import asof, calibrated_image, cached_calibrated_coordinates
import native_warp

DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INVOCATION = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/dynamic-vision-native-5hz/invocation.json')


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


fast = load('agent_warp_fast_sampler', DIRECTORY/'agent-warp-fast-sampler.py')


def inputs(path):
  paths = []
  while True:
    paths.append(path)
    invocation = json.loads(path.read_text())
    command = invocation['command']
    if '--native-manifest' in command:
      break
    path = Path(command[command.index('--invocation')+1])
  option = lambda flag: Path(command[command.index(flag)+1])
  manifest_path, calibration_path, raw_path = (option(flag) for flag in
      ('--native-manifest', '--native-calibrations', '--native-frame-cache'))
  manifest = json.loads(manifest_path.read_text())
  records = json.loads(calibration_path.read_text())
  calibrations = np.array([r['ns'] for r in records]), records
  provenance = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (*paths, manifest_path, calibration_path)}
  return manifest, calibrations, raw_path, manifest_path, calibration_path, provenance


def source_hashes():
  names = ('native_warp.py', 'native-warp.c', 'calibration_trials.py', 'stream-inputs.py', 'check-native-warp.py', 'camera_profiles.py',
           'agent-warp-fast-kernel.c', 'agent-warp-fast-workers.c', 'agent-warp-fast-sampler.py', 'agent-warp-fast-check.py')
  return {name:hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest() for name in names}


def summary(values):
  return dict(samples=len(values), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
              minimum_ms=float(min(values)))


def boundaries(samplers):
  rng = np.random.default_rng(20261117)
  values, baseline_differences, baseline_examples = 0, 0, []
  for h, w in ((1, 1), (1, 47), (31, 1), (31, 47), (2, 2)):
    source = rng.integers(0, 256, (3, h, w), dtype=np.uint8)
    xy = rng.uniform([-3, -3], [w+3, h+3], (97, 113, 2))
    probes = np.array([-2., -np.finfo(float).eps, 0., np.nextafter(.5, 0), .5,
                       np.nextafter(.5, 1), 1., w-1., w, 2*w])
    xy[:10, :10, 0] = probes[:, None]
    xy[:10, :10, 1] = np.minimum(probes, h+1)[None, :]
    for uniform in (None, 0, 255):
      if uniform is not None:
        source[:] = uniform
      expected = np.stack([map_coordinates(c, (xy[..., 1], xy[..., 0]), order=1, mode='nearest', prefilter=False) for c in source])
      clipped = np.clip(xy, [0, 0], [w-1, h-1])
      offsets = np.floor(clipped).astype(np.int32)
      fractions = clipped-offsets
      old = np.empty_like(expected)
      native_warp.library().warp_u8(source.ctypes.data, old.ctypes.data, offsets.ctypes.data, fractions.ctypes.data,
                                   expected.size//3, h*w, w, h)
      baseline_differences += int(np.count_nonzero(old != expected))
      for c, yy, xx in np.argwhere(old != expected)[:3]:
        baseline_examples.append(dict(shape=[h, w], xy=xy[yy, xx].tolist(), baseline=int(old[c, yy, xx]),
                                      scipy=int(expected[c, yy, xx])))
      for sampler in samplers:
        packed = fast.pack_maps((h, w), xy[..., 1], xy[..., 0], sampler.kind)
        np.testing.assert_array_equal(sampler.sample(source, packed), expected)
      values += expected.size
  # Enumerate exact half ties and adjacent representable coordinates.
  source = np.array([[[0, 1], [254, 255]], [[255, 0], [0, 255]], [[11, 12], [13, 14]]], dtype=np.uint8)
  probes = np.array([0., np.nextafter(.5, 0), .5, np.nextafter(.5, 1), np.nextafter(1., 0), 1.])
  sx, sy = np.meshgrid(probes, probes)
  expected = np.stack([map_coordinates(c, (sy, sx), order=1, mode='nearest', prefilter=False) for c in source])
  for sampler in samplers:
    np.testing.assert_array_equal(sampler.sample(source, fast.pack_maps((2, 2), sy, sx, sampler.kind)), expected)
  return dict(exact=True, compared_values_per_variant=values+expected.size, cases=16,
              baseline_differences=baseline_differences, baseline_examples=baseline_examples)


def microbench(samplers, manifest, calibrations, raw_path, rounds):
  # Four distinct recorded frames: two frames from each front camera.
  chosen = [manifest['schedule'][0]['frames'][i] for i in (3, 4, 8, 9)]
  frames = []
  for frame in chosen:
    path = raw_path/str(frame['view'])/f'{frame["video_index"]}.npy'
    array = np.load(path, allow_pickle=False)
    calibration, _ = asof(calibrations, frame['capture_ns'], 1_000_000_000)
    frames.append((array, frame['view'], manifest['source_camera_geometry'], manifest['target_camera_geometry'], calibration))
  functions = [('baseline', native_warp.calibrated_image)] + [
      (f'{s.kind}-{Path(s.lib.provenance["command"][0]).name}-native{int(s.lib.provenance["native"])}-workers{s.workers}', s.calibrated_image) for s in samplers]
  expected = [native_warp.calibrated_image(*frame) for frame in frames]
  for _, fn in functions:
    for frame, reference in zip(frames, expected, strict=True):
      np.testing.assert_array_equal(fn(*frame), reference)
  timings = {name:[] for name, _ in functions}
  rng = np.random.default_rng(73)
  for _ in range(rounds):
    for index in rng.permutation(len(functions)):
      name, fn = functions[index]
      start = time.perf_counter_ns()
      for frame in frames:
        fn(*frame)
      timings[name].append((time.perf_counter_ns()-start)/1e6)
  return dict(boundary='Four distinct resident frames; allocation and map lookup included; disk loading and map construction excluded',
              timings={name:summary(values) for name, values in timings.items()}, raw_timings_ms=timings,
              frames=[dict(view=f['view'], video_index=f['video_index']) for f in chosen])


def full_route(sampler, manifest, calibrations, raw_path, manifest_path, calibration_path):
  stream = load('agent_warp_fast_stream_check', DIRECTORY/'stream-inputs.py')
  stream.library = lambda: sampler.lib
  independent = dict(frames=0, values=0, raw_sha256={}, calibration_rotations=set())

  def checked(array, camera, source, target, calibration):
    result = sampler.calibrated_image(array, camera, source, target, calibration)
    original = native_warp.calibrated_image(array, camera, source, target, calibration)
    np.testing.assert_array_equal(result, original)
    reference = calibrated_image(array, camera, source, target, calibration)
    np.testing.assert_array_equal(result, reference)
    independent['frames'] += 1
    independent['values'] += result.size
    independent['calibration_rotations'].add((tuple(calibration['rpyCalib']), tuple(calibration['wideFromDeviceEuler'])))
    return result

  stream.calibrated_image = checked
  frames = stream.NativeFrames(manifest_path, calibration_path, raw_path)
  saved_values, snapshots = 0, 0
  for row in manifest['schedule']:
    assert all(f['capture_ns'] <= row['query_ns'] and f['image_calibration_ns'] <= f['capture_ns'] for f in row['frames'])
    result, timing = frames.prepare(row['sample'])
    expected = np.load(Path(row['sample'])/'image_frames.npy', allow_pickle=False, mmap_mode='r')
    np.testing.assert_array_equal(result, expected)
    saved_values += result.size
    snapshots += 1
    if snapshots % 40 == 0:
      print(f'Exact recorded snapshots: {snapshots}/{len(manifest["schedule"])}', flush=True)
    for frame in row['frames']:
      path = raw_path/str(frame['view'])/f'{frame["video_index"]}.npy'
      if str(path) not in independent['raw_sha256']:
        independent['raw_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
  independent['calibration_rotations'] = len(independent['calibration_rotations'])
  return dict(exact=True, snapshots=snapshots, saved_values=saved_values, independent_scipy=independent,
              image_cache_limit=144, sampler=sampler.state())


def invalidation(sampler, manifest, calibrations, raw_path):
  source, target = manifest['source_camera_geometry'], manifest['target_camera_geometry']
  frame = manifest['schedule'][0]['frames'][4]
  array = np.load(raw_path/str(frame['view'])/f'{frame["video_index"]}.npy', allow_pickle=False)
  calibration, _ = asof(calibrations, frame['capture_ns'], 1_000_000_000)
  original = sampler.calibrated_image(array, frame['view'], source, target, calibration)
  array[:] ^= 255
  changed = sampler.calibrated_image(array, frame['view'], source, target, calibration)
  assert np.count_nonzero(changed != original) > 0
  np.testing.assert_array_equal(changed, calibrated_image(array, frame['view'], source, target, calibration))
  cases = []
  for axis in range(3):
    for field in ('rpyCalib', 'wideFromDeviceEuler'):
      modified = copy.deepcopy(calibration)
      modified[field][axis] += 1e-4
      result = sampler.calibrated_image(array, frame['view'], source, target, modified)
      np.testing.assert_array_equal(result, calibrated_image(array, frame['view'], source, target, modified))
      cases.append(field+str(axis))
  for i in range(10):
    modified = copy.deepcopy(calibration)
    modified['rpyCalib'][0] += (i+1)*1e-5
    result = sampler.calibrated_image(array, frame['view'], source, target, modified)
    np.testing.assert_array_equal(result, calibrated_image(array, frame['view'], source, target, modified))
  assert len(sampler.cache) <= 8 and sampler.cache_bytes <= sampler.max_bytes and sampler.evictions > 0
  source_changed = copy.deepcopy(source)
  source_changed['cameras'][str(frame['view'])]['principal_point'][0] += .01
  np.testing.assert_array_equal(sampler.calibrated_image(array, frame['view'], source_changed, target, calibration),
                                calibrated_image(array, frame['view'], source_changed, target, calibration))
  target_changed = copy.deepcopy(target)
  target_changed['agent_warp_fast_cache_probe'] = 1
  misses = sampler.misses
  sampler.calibrated_image(array, frame['view'], source, target_changed, calibration)
  assert sampler.misses == misses+1
  small = np.ascontiguousarray(array[:, ::2, ::2])
  np.testing.assert_array_equal(sampler.calibrated_image(small, frame['view'], source, target, calibration),
                                calibrated_image(small, frame['view'], source, target, calibration))
  for invalid in (array.astype(np.float32), array[:, :, ::2]):
    try:
      sampler.calibrated_image(invalid, frame['view'], source, target, calibration)
    except ValueError:
      pass
    else:
      raise AssertionError('Invalid input accepted')
  return dict(exact=True, changed_calibration=cases, pixel_mutation_values=int(np.count_nonzero(changed != original)),
              changed_source_geometry=True, changed_target_key=True, changed_source_shape=True, sampler=sampler.state())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--invocation', type=Path, default=DEFAULT_INVOCATION)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--screen', action='store_true')
  parser.add_argument('--kind', choices=fast.KINDS, default='channels')
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--compiler', choices=('clang', 'gcc'), default='clang')
  parser.add_argument('--native', action='store_true')
  parser.add_argument('--rounds', type=int, default=15)
  args = parser.parse_args()
  if not args.output.resolve().is_relative_to(DIRECTORY) or not args.output.name.startswith('agent-warp-fast-'):
    raise ValueError('Write reports only within the owned experiment prefix')
  original_hashes = source_hashes()
  baseline = fast.build(baseline=True)
  native_warp.library = lambda: baseline
  manifest, calibrations, raw_path, manifest_path, calibration_path, provenance = inputs(args.invocation)
  if args.screen:
    samplers = [fast.Sampler(kind=kind, compiler=compiler, native=native) for compiler, native in
                (('clang', False), ('clang', True), ('gcc', False)) for kind in fast.KINDS]
    samplers.extend(fast.Sampler(kind=kind, workers=workers, native=native) for kind, native in
                    (('aos', False), ('simd', False), ('channels', True)) for workers in (2, 3))
  else:
    samplers = [fast.Sampler(args.kind, args.workers, args.compiler, args.native)]
  result = dict(scope='CPU only; no Asius, GPU, or complete pipeline performance claim',
                host=dict(machine=platform.machine(), cpu=platform.processor(), affinity=sorted(os.sched_getaffinity(0))),
                source_sha256=original_hashes, input_sha256=provenance, baseline=baseline.provenance)
  try:
    result['boundaries'] = boundaries(samplers)
    result['microbench'] = microbench(samplers, manifest, calibrations, raw_path, args.rounds)
    print(json.dumps(result['microbench']['timings'], indent=2), flush=True)
    if not args.screen:
      result['route'] = full_route(samplers[0], manifest, calibrations, raw_path, manifest_path, calibration_path)
      result['invalidation'] = invalidation(samplers[0], manifest, calibrations, raw_path)
    result['variants'] = [s.state() for s in samplers]
    if source_hashes() != original_hashes:
      raise AssertionError('Sources changed during validation')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Saved {args.output}', flush=True)
  finally:
    for sampler in samplers:
      sampler.close()


if __name__ == '__main__':
  main()
