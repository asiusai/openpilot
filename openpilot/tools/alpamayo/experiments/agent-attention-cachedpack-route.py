"""Run cached expert packing through a saved complete route invocation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--invocation', type=Path,
    default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-native-5hz-fixed-byte/invocation.json'))
  parser.add_argument('--variant', choices=('baseline', 'cachedpack'), default='cachedpack')
  parser.add_argument('--warmup', type=int)
  parser.add_argument('--runs', type=int)
  parser.add_argument('--reference', type=Path)
  args = parser.parse_args()
  invocation = json.loads(args.invocation.read_text())
  command = invocation['command'][1:]
  assert Path(command[0]).name == 'benchmark-pipeline.py'
  assert '--expert-kernels' in command and '--online-expert' not in command
  for key, value in invocation['environment'].items():
    if key.startswith('ALPAMAYO_'):
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value
  for key, value in (('output', str(args.output)), ('warmup', args.warmup), ('runs', args.runs)):
    if value is not None:
      flag = '--'+key
      if flag in command:
        command[command.index(flag)+1] = str(value)
      else:
        command.extend([flag, str(value)])
  benchmark = load('benchmark-pipeline')
  candidate = load('agent-attention-cachedpack')
  original_loader = benchmark.local_module
  hooks = []
  def local_module(name):
    module = original_loader(name)
    if name == 'expert-kernels':
      original_install = module.install
      def install(model, *a, **kw):
        result = original_install(model, *a, **kw)
        if args.variant == 'cachedpack':
          candidate.install(model)
        hooks.append(args.variant)
        return result
      module.install = install
    return module
  benchmark.local_module = local_module
  provenance = {'variant': args.variant, 'source_invocation': str(args.invocation),
    'source_invocation_sha256': hashlib.sha256(args.invocation.read_bytes()).hexdigest(),
    'effective_command': command, 'route_complete': False}
  metadata = args.output.with_suffix('.agent-attention-cachedpack.json')
  metadata.write_text(json.dumps(provenance, indent=2)+'\n')
  sys.argv = command
  benchmark.main()
  assert hooks == [args.variant], hooks
  provenance['route_complete'] = True
  if args.reference:
    # Reuse the already reviewed strict comparator without changing its source.
    provenance['comparison'] = load('agent-attention-route').compare(args.reference, args.output)
  metadata.write_text(json.dumps(provenance, indent=2)+'\n')
  report = json.loads(args.output.read_text())
  report['agent_attention_cachedpack'] = provenance
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
