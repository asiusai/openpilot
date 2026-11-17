"""Install the validated sampler before recorded or live preparation starts."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--check', type=Path, required=True)
  parser.add_argument('--source-directory', type=Path, required=True)
  parser.add_argument('--audit', type=Path, required=True)
  parser.add_argument('command', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  command = args.command[1:] if args.command[:1] == ['--'] else args.command
  if not command:
    raise ValueError('Supply a Python script after --')
  if args.audit.exists():
    raise FileExistsError('Use a new sampler audit path')
  directory = Path(__file__).resolve().parent
  source = args.source_directory.resolve()
  check = json.loads(args.check.read_text())
  if not check.get('boundaries', {}).get('exact') or not check.get('route', {}).get('exact') or not check.get('invalidation', {}).get('exact'):
    raise ValueError('Require completed boundary, recorded-route, and invalidation checks')
  required = ('native_warp.py', 'native-warp.c', 'calibration_trials.py', 'stream-inputs.py', 'camera_profiles.py')
  for root, names in ((source, required), (directory, ('agent-warp-fast-kernel.c', 'agent-warp-fast-workers.c', 'agent-warp-fast-sampler.py'))):
    for name in names:
      if hashlib.sha256((root/name).read_bytes()).hexdigest() != check['source_sha256'][name]:
        raise ValueError(f'Validated source differs: {root/name}')
  if any(name in sys.modules for name in ('native_warp', 'calibration_trials', 'stream_inputs')):
    raise RuntimeError('Install the sampler before importing input preparation')
  sys.path.insert(0, str(source.parent))
  sys.path.insert(0, str(source))
  spec = importlib.util.spec_from_file_location('agent_warp_fast_sampler_hook', directory/'agent-warp-fast-sampler.py')
  implementation = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(implementation)
  config = check['variants'][0]
  compiler = Path(config['provenance']['command'][0]).name
  sampler = implementation.Sampler(config['kind'], config['workers'], compiler, config['provenance']['native'],
                                   config['max_maps'], config['max_bytes'])
  if sampler.lib.provenance['library_sha256'] != config['provenance']['library_sha256']:
    sampler.close()
    raise ValueError('Compiled sampler differs from the validated binary; repeat CPU checks on this host')
  original = importlib.import_module('native_warp')
  if Path(original.__file__).resolve() != source/'native_warp.py':
    raise ValueError('Unexpected native sampler import path')
  original.calibrated_image = sampler.calibrated_image
  original.library = lambda: sampler.lib
  sys.argv = command
  try:
    runpy.run_path(command[0], run_name='__main__')
    if sampler.hits+sampler.misses == 0:
      raise AssertionError('The replacement sampler was not used')
  finally:
    sampler.close()
    args.audit.write_text(json.dumps(dict(check=str(args.check.resolve()),
      check_sha256=hashlib.sha256(args.check.read_bytes()).hexdigest(), source_directory=str(source),
      command=command, sampler=sampler.state()), indent=2)+'\n')


if __name__ == '__main__':
  main()
