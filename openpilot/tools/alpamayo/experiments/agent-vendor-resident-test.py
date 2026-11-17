"""Run the existing vendor benchmark or route test with scoped image residency.

This changes kernel loading only. The original numerical and timing checks
remain in their owning runners. A parent process must enforce a hard timeout.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load(path):
  spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__, add_help=False)
  parser.add_argument('--stage', choices=('bench', 'route'), required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--upload-seconds', type=float, default=30.)
  args, remaining = parser.parse_known_args()
  if not 1 <= args.upload_seconds <= 60:
    parser.error('upload-seconds must be within 1..60')
  root = Path(__file__).parent
  owner = load(root/f'agent-vendor-{args.stage}.py')
  resident = load(root/'agent-vendor-resident.py')
  original_runtime, original_argv = owner.common.runtime, sys.argv
  pools = []

  @contextmanager
  def resident_runtime(*positional, **keywords):
    with original_runtime(*positional, **keywords) as (device, vendor):
      with resident.install(device, vendor, owner.common.SOLUTIONS, seconds=args.upload_seconds) as pool:
        pools.append(pool)
        yield device, vendor

  owner.common.runtime = resident_runtime
  sys.argv = [str(root/f'agent-vendor-{args.stage}.py'), '--output', str(args.output), *remaining]
  try:
    owner.main()
    assert pools and all(pool.closed and pool.stats['builder_and_compile_cache_restored'] for pool in pools)
  finally:
    owner.common.runtime, sys.argv = original_runtime, original_argv
    if args.output.exists():
      report = json.loads(args.output.read_text())
      report['resident_image_loading'] = {
        'stage': args.stage, 'scopes': [pool.stats for pool in pools],
        'runtime_wrapper_restored': owner.common.runtime is original_runtime,
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          (Path(__file__), root/'agent-vendor-resident.py')},
        'limit': 'Residency changes loading only; model quality and complete-pipeline speed require separate validation.',
      }
      args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
