"""Parent-only route replay with isolated attention hooks and exact comparisons.

Run through run-alpamayo-speed-trial.py --program. Reuse a saved invocation's
complete input schedule, model settings, noise, and preprocessing boundary.
"""
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


def entries(path):
  data = json.loads(path.read_text())
  return data['warmup']+data['runs']


def compare(reference, output):
  import numpy as np
  saved = {Path(row['sample']).name: row for row in entries(reference)}
  compared = []
  for row in entries(output):
    name = Path(row['sample']).name
    assert name in saved, f'Missing reference sample: {name}'
    a, b = row['prediction'], saved[name]['prediction']
    fields = [key for key in ('xyz', 'rot', 'rotation', 'yaw', 'generated_tokens', 'generated_token_ids', 'reasoning') if key in a or key in b]
    assert 'xyz' in fields and 'generated_token_ids' in fields
    result = {'sample': name, 'fields': fields, 'exact': True}
    for key in fields:
      assert key in a and key in b, (name, key)
      if key == 'reasoning':
        assert a[key] == b[key], (name, key)
      else:
        aa, bb = np.asarray(a[key]), np.asarray(b[key])
        assert np.isfinite(aa).all() and np.isfinite(bb).all(), (name, key)
        np.testing.assert_array_equal(aa, bb, err_msg=f'{name} {key}')
    compared.append(result)
  return {'reference': str(reference), 'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
    'samples': len(compared), 'all_exact': True, 'rows': compared}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--invocation', type=Path,
    default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-native-5hz-fixed-byte/invocation.json'))
  parser.add_argument('--variant', choices=('baseline', 'shuffle', 'lds', 'stage', 'shuffle-stage', 'shuffle-fma'), default='shuffle-stage')
  parser.add_argument('--target', choices=('vision', 'prefill', 'all'), default='all')
  parser.add_argument('--waves', type=int, choices=(4, 8), default=8)
  parser.add_argument('--warmup', type=int)
  parser.add_argument('--runs', type=int)
  parser.add_argument('--reference', type=Path)
  args = parser.parse_args()
  invocation = json.loads(args.invocation.read_text())
  command = invocation['command'][1:]
  assert Path(command[0]).name == 'benchmark-pipeline.py'
  assert '--online-attention' in command and '--online-expert' not in command
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
  candidate = load('agent-attention-rowsoftmax')
  original_loader = benchmark.local_module
  hook_calls = []
  def local_module(name):
    module = original_loader(name)
    if name == 'online-attention':
      original_install = module.install
      def install(model, *a, **kw):
        original_install(model, *a, **kw)
        if args.variant != 'baseline':
          candidate.install(model, args.target, args.variant, args.waves)
        hook_calls.append(args.variant)
      module.install = install
    return module
  benchmark.local_module = local_module
  provenance = {'variant': args.variant, 'target': args.target, 'waves': args.waves,
    'source_invocation': str(args.invocation), 'source_invocation_sha256': hashlib.sha256(args.invocation.read_bytes()).hexdigest(),
    'effective_command': command, 'gpu_validation_pending': True}
  args.output.with_suffix('.agent-attention.json').write_text(json.dumps(provenance, indent=2)+'\n')
  sys.argv = command
  benchmark.main()
  assert hook_calls == [args.variant], hook_calls
  provenance['gpu_validation_pending'] = False
  provenance['route_completed'] = True
  if args.reference:
    provenance['comparison'] = compare(args.reference, args.output)
  args.output.with_suffix('.agent-attention.json').write_text(json.dumps(provenance, indent=2)+'\n')
  report = json.loads(args.output.read_text())
  report['agent_attention'] = provenance
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
