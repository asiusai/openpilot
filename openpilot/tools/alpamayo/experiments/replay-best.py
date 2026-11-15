"""Reproduce the saved six-camera configuration with explicitly supplied assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('checkpoint', 'samples', 'noise', 'tinygrad', 'fp8-cache', 'cache', 'output'):
    parser.add_argument('--'+name, type=Path, required=True)
  parser.add_argument('--verify-vision-warmup', action='store_true',
                      help='Repeat embedding checks; this changes graph allocation from the fastest saved run')
  parser.add_argument('--print-command', action='store_true', help='Print configuration without starting a GPU context')
  args = parser.parse_args()
  here = Path(__file__).resolve().parent
  config = json.loads((here/'best-config.json').read_text())
  vocab = here/'coc_vocab.json'
  assert hashlib.sha256(vocab.read_bytes()).hexdigest() == config['vocabulary_sha256']
  env = dict(os.environ, PROFILE='0', PYTHONPATH=f'{args.tinygrad.resolve()}:{here.parent}',
             ALPAMAYO_FP8_CACHE=str(args.fp8_cache.resolve()), CACHEDB=str(args.cache.resolve()))
  for key, value in config['environment'].items():
    if value is None:
      env.pop(key, None)
    else:
      env[key] = value
  samples = [args.samples/f'sample-{time}' for time in config['timestamps_us']]
  command = [sys.executable, str(here/'benchmark-pipeline.py'),
             '--checkpoint', str(args.checkpoint), '--samples', *map(str, samples),
             '--initial-noise', str(args.noise), '--reduced-vocab', str(vocab),
             '--output', str(args.output), *config['arguments']]
  if args.verify_vision_warmup:
    command.append('--verify-vision-warmup')
  if args.print_command:
    print(json.dumps({'command': command, 'environment': {k: env.get(k) for k in
                     (*config['environment'], 'PROFILE', 'PYTHONPATH', 'ALPAMAYO_FP8_CACHE', 'CACHEDB')}}, indent=2))
    return
  revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.tinygrad, text=True).strip()
  assert revision == config['tinygrad_revision'], 'Use the recorded tinygrad revision'
  assert args.checkpoint.is_dir() and args.noise.is_file() and all(path.is_dir() for path in samples)
  subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
  main()
