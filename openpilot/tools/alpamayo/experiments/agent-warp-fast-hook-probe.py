"""Exercise the optional sampler hook through recorded input preparation on CPU."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import native_warp


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--invocation', type=Path, required=True)
  args = parser.parse_args()
  path = args.invocation
  while True:
    command = json.loads(path.read_text())['command']
    if '--native-manifest' in command:
      break
    path = Path(command[command.index('--invocation')+1])
  option = lambda flag: Path(command[command.index(flag)+1])
  manifest = option('--native-manifest')
  source = Path(native_warp.__file__).resolve().parent
  spec = importlib.util.spec_from_file_location('warp_fast_hook_stream', source/'stream-inputs.py')
  stream = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(stream)
  frames = stream.NativeFrames(manifest, option('--native-calibrations'), option('--native-frame-cache'))
  rows = json.loads(manifest.read_text())['schedule']
  for row in rows[:3]:
    actual, _ = frames.prepare(row['sample'])
    np.testing.assert_array_equal(actual, np.load(Path(row['sample'])/'image_frames.npy', allow_pickle=False))
  print(json.dumps(dict(exact_snapshots=3, sampler_hook_used=native_warp.calibrated_image.__self__.state()['misses'] > 0)))


if __name__ == '__main__':
  main()
