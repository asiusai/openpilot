"""Optional vision-only hook around an unchanged recorded pipeline invocation.

Supports benchmark-pipeline and its existing nominal-prefill wrapper. Parent
GPU serialization is required. --plan validates configuration without GPU use.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent


def inspect_invocation(path):
  chain = []
  for _ in range(4):
    invocation = json.loads(path.read_text())
    command = invocation['command'][1:]
    chain.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'program': command[0]})
    if Path(command[0]).name == 'benchmark-pipeline.py': break
    assert Path(command[0]).name == 'agent-speculation-run.py' and '--invocation' in command, 'Unsupported wrapper; preserve the recorded settings explicitly'
    path = Path(command[command.index('--invocation')+1])
  else: raise AssertionError('Invocation nesting exceeds four levels')
  assert all(flag in command for flag in ('--batched-vision', '--grouped-quantize-reduction', '--bf16-vision-layers', '--llvm-wgp'))
  assert command[command.index('--bf16-vision-layers')+1] == '2'
  assert command[command.index('--pixel-scale')+1] == '1'
  assert command[command.index('--vision-group-limit')+1] == '2'
  return chain, command


def compare(reference, output):
  import numpy as np
  before, after = (json.loads(path.read_text()) for path in (reference, output))
  old, new = before['warmup']+before['runs'], after['warmup']+after['runs']
  assert len(old) == len(new) and old, 'Compare the same complete input schedule'
  checks = []
  for expected, actual in zip(old, new, strict=True):
    assert expected['sample'] == actual['sample'], 'Input order or source changed'
    a, b = expected['prediction'], actual['prediction']
    assert a['generated_token_ids'] == b['generated_token_ids']
    assert a['reasoning'] == b['reasoning']
    for field in ('xyz', 'rot'):
      x, y = np.asarray(a[field], np.float64), np.asarray(b[field], np.float64)
      assert np.isfinite(x).all() and np.isfinite(y).all()
      np.testing.assert_array_equal(x.view(np.uint64), y.view(np.uint64), err_msg=f'{actual["sample"]} {field}')
    checks.append({'sample': actual['sample'], 'tokens_exact': True, 'reasoning_exact': True, 'xyz_bits_exact': True, 'rotation_bits_exact': True})
  return {'reference': str(reference), 'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(), 'all_exact': True, 'samples': checks}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--invocation', type=Path, required=True)
  parser.add_argument('--variant', choices=('current', 'candidate'), default='candidate')
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--warmup', type=int)
  parser.add_argument('--runs', type=int)
  parser.add_argument('--plan', action='store_true')
  args = parser.parse_args()
  chain, terminal = inspect_invocation(args.invocation)
  invocation = json.loads(args.invocation.read_text())
  command = invocation['command'][1:]
  for key, value in (('output', str(args.output)), ('warmup', args.warmup), ('runs', args.runs)):
    if value is None: continue
    flag = '--'+key
    if flag in command: command[command.index(flag)+1] = str(value)
    else: command.extend((flag, str(value)))
  report = {'variant': args.variant, 'source_chain': chain, 'effective_command': command,
    'terminal_program': terminal[0], 'plan_only': args.plan, 'route_completed': False, 'exactness_validated': False,
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE/'agent-vision-tiles.py')},
    'changes': 'Only eligible FP8 vision MLP matrix tiles. Recorded quantization, language, speculation, inputs, noise, and trajectory settings persist.'}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  metadata_path = args.output.with_suffix('.agent-vision-tiles.json')
  def save(): metadata_path.write_text(json.dumps(report, indent=2)+'\n')
  save()
  if args.plan:
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'plan_only': True, 'gpu_executed': False, 'source_chain': chain}), flush=True)
    return
  for key, value in invocation['environment'].items():
    if value is None: os.environ.pop(key, None)
    else: os.environ[key] = value
  os.environ['PROFILE'] = '0'
  original_spec = importlib.util.spec_from_file_location
  spec = original_spec('agent_vision_tiles_route_hook', HERE/'agent-vision-tiles.py')
  candidate = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(candidate)
  instances = []
  def hooked_spec(name, location, *values, **options):
    spec = original_spec(name, location, *values, **options)
    path = Path(location)
    if path.name != 'batched-vision.py' or path.parent.name != 'experiments': return spec
    original_loader = spec.loader
    class Loader:
      def create_module(self, value): return original_loader.create_module(value)
      def exec_module(self, module):
        original_loader.exec_module(module)
        original_install = module.install
        def install(model, *a, **kw):
          encoder = original_install(model, *a, **kw)
          info = candidate.install(model) if args.variant == 'candidate' else {'candidate': 'current', 'trace_calls': {}}
          instances.append(info)
          return encoder
        module.install = install
    spec.loader = Loader()
    return spec
  importlib.util.spec_from_file_location = hooked_spec
  sys.argv = command
  try:
    runpy.run_path(command[0], run_name='__main__')
    assert len(instances) == 1, 'Expected exactly one batched vision model'
    if args.variant == 'candidate':
      assert any(name.endswith('/batch2') and count for name, count in instances[0]['trace_calls'].items()), 'Candidate never traced an actual two-image MLP'
    report['route_completed'] = True
    if args.reference:
      report['comparison'] = compare(args.reference, args.output)
      report['exactness_validated'] = True
  except BaseException as exc:
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
  finally:
    importlib.util.spec_from_file_location = original_spec
    report['hooks'] = instances
    save()
    if args.output.exists():
      result = json.loads(args.output.read_text())
      result['agent_vision_tiles'] = report
      args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
