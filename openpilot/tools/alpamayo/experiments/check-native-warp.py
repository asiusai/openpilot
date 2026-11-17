"""Compare the native sampler with independent SciPy output and saved inputs."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import map_coordinates

from native_warp import library


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--calibrations', type=Path, required=True)
  parser.add_argument('--raw-cache', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(29)
  source = rng.integers(0, 256, (3, 31, 47), dtype=np.uint8)
  xy = rng.uniform([0, 0], [46, 30], (40, 50, 2))
  xy[:2, :2] = [[[0, 0], [46, 30]], [[3.5, 4.5], [46, 0]]]
  offsets = np.floor(xy).astype(np.int32)
  fractions = xy-offsets
  result = np.empty((3, 40, 50), dtype=np.uint8)
  lib = library()
  lib.warp_u8(source.ctypes.data, result.ctypes.data, offsets.ctypes.data, fractions.ctypes.data,
              2000, 31*47, 47, 31)
  expected = np.stack([map_coordinates(c, (xy[..., 1], xy[..., 0]), order=1, mode='nearest', prefilter=False) for c in source])
  np.testing.assert_array_equal(result, expected)
  spec = importlib.util.spec_from_file_location('stream_inputs', Path(__file__).with_name('stream-inputs.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  frames = module.NativeFrames(args.manifest, args.calibrations, args.raw_cache)
  manifest = json.loads(args.manifest.read_text())
  timings = []
  for row in manifest['schedule']:
    pixels, timing = frames.prepare(row['sample'])
    np.testing.assert_array_equal(pixels, np.load(Path(row['sample'])/'image_frames.npy', allow_pickle=False))
    timings.append(timing)
  steady = [t['native_calibration_ms'] for t in timings if t['calibrated_frame_misses']==2]
  result = {'independent_sampler_pixels':6000, 'exact_saved_snapshots':len(timings),
            'two_new_frames_samples':len(steady), 'two_new_frames_median_ms':float(np.median(steady)),
            'two_new_frames_p95_ms':float(np.percentile(steady,95)), 'timings':timings}
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k!='timings'}, indent=2))


if __name__ == '__main__':
  main()
