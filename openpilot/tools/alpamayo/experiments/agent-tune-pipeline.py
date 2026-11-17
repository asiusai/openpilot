"""Run the existing pipeline with only exact, validated tuning-cache AST matches.

Set AGENT_TUNE_CACHE and optionally AGENT_TUNE_WGP=1. All arguments pass through.
This wrapper does not claim trajectory validation; compare the resulting report.
"""
import importlib.util
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
  if '--invocation' in sys.argv:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--invocation', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--runs', type=int)
    args = parser.parse_args()
    source = json.loads(args.invocation.read_text())
    assert Path(source['command'][1]).name == 'benchmark-pipeline.py'
    for key, value in source['environment'].items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    command = source['command'][1:]
    command[command.index('--output')+1] = str(args.output)
    for key in ('warmup', 'runs'):
      if getattr(args, key) is not None: command[command.index('--'+key)+1] = str(getattr(args, key))
    sys.argv = command
    wgp = '--llvm-wgp' in command
    cache = args.cache
    baseline = args.baseline
    assert baseline or cache is not None
    args.output.with_name('agent-tune-replay-invocation.json').write_text(json.dumps({
      'source_invocation': str(args.invocation), 'command': command, 'baseline': baseline,
      'cache': str(cache), 'source_environment': source['environment']}, indent=2)+'\n')
  else:
    baseline, cache = False, Path(os.environ['AGENT_TUNE_CACHE'])
    wgp = os.environ.get('AGENT_TUNE_WGP', '0') == '1'
  path = Path(__file__).with_name('agent-tune.py')
  spec = importlib.util.spec_from_file_location('agent_tune', path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  stats = {'baseline': True} if baseline else module.install_cache(cache, wgp)
  # A separate file survives even if the existing benchmark exits exceptionally.
  output = Path(sys.argv[sys.argv.index('--output')+1])
  try:
    runpy.run_path(str(Path(__file__).with_name('benchmark-pipeline.py')), run_name='__main__')
  finally:
    output.with_name('agent-tune-cache-hits.json').write_text(json.dumps(stats, indent=2)+'\n')


if __name__ == '__main__': main()
