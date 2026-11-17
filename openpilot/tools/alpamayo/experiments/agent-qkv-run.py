"""Replay the existing pipeline with optional QKV fusion and strict paired outputs.

Use through the parent's serialized GPU wrapper. This helper does not alter
quantization, route inputs, reference gates, attention, or the vendor installer.
"""
import argparse
from contextlib import ExitStack
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def local(name):
  spec = importlib.util.spec_from_file_location('agent_qkv_run_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def compare(actual, reference):
  import numpy as np
  count = 0
  for section in ('warmup', 'runs'):
    assert len(actual[section]) == len(reference[section]), 'Paired warmup/run counts differ'
    for a, b in zip(actual[section], reference[section], strict=True):
      assert a['sample'] == b['sample'] and a['input_latest_timestamp_us'] == b['input_latest_timestamp_us'], 'Paired route inputs differ'
      for key in ('generated_tokens', 'generated_token_ids', 'reasoning'):
        assert a['prediction'][key] == b['prediction'][key], f'Exact {key} comparison failed'
      for key in ('xyz', 'rot'):
        x, y = (np.asarray(r['prediction'][key], dtype=np.float64) for r in (a, b))
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.testing.assert_array_equal(x.view(np.uint64), y.view(np.uint64), err_msg=f'Exact {key} comparison failed')
      count += 1
  return {'paired_predictions': count, 'tokens_reasoning_xyz_rot_exact': True, 'full_route_adoption': False}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--invocation', type=Path, required=True)
  p.add_argument('--micro-report', type=Path)
  p.add_argument('--reference', type=Path)
  p.add_argument('--scope', choices=('expert', 'prefill', 'both'), default='expert')
  p.add_argument('--baseline', action='store_true')
  p.add_argument('--warmup', type=int)
  p.add_argument('--runs', type=int)
  args = p.parse_args()
  assert args.baseline or (args.micro_report is not None and args.reference is not None), 'Candidate needs GPU kernel checks and a paired baseline'
  saved = json.loads(args.invocation.read_text())
  command = saved['command'][1:]
  assert Path(command[0]).resolve() == HERE/'benchmark-pipeline.py'
  assert '--expert-kernels' in command and '--packed-quantize' in command
  assert '--fused-qkv' not in command and '--row-quantize' not in command, 'Keep other QKV and row-scale installers separate'
  assert not any('agent-vendor' in part for part in command), 'Test independently of the vendor candidate'
  parent_cache = os.environ.get('CACHEDB')
  for key, value in saved['environment'].items():
    if value is None: os.environ.pop(key, None)
    else: os.environ[key] = value
  if parent_cache is not None: os.environ['CACHEDB'] = parent_cache
  for path in reversed(os.environ.get('PYTHONPATH', '').split(os.pathsep)):
    if path: sys.path.insert(0, path)
  os.environ.update(BEAM='0', JITBEAM='0')
  check, fusion = local('agent-qkv-check'), local('agent-qkv')
  provenance = check.provenance()
  if not args.baseline:
    micro = json.loads(args.micro_report.read_text())
    assert micro.get('complete') and micro.get('gpu_executed') and micro['mode'] == 'gpu'
    assert micro['provenance'] == provenance, 'QKV source or tinygrad changed after GPU checks'
    assert micro['wgp'] == ('--llvm-wgp' in command)
    assert micro['environment']['TC_OPT'] == os.environ.get('TC_OPT')
    required = {'integer'} | ({'expert'} if args.scope == 'expert' else
                             {'prefill-tail', 'prefill-padded'} if args.scope == 'prefill' else set(check.CASES))
    assert required <= {r['name'] for r in micro['cases']}
    for name in required:
      rows = [r for r in micro['cases'] if r['name'] == name]
      assert any(not r['bias'] for r in rows)
      if name == 'integer': assert {r['bias'] for r in rows} == {False, True}
      for r in rows:
        assert r['setup_exact'] and len(r['checks']) == 7 and all(c['exact_bf16'] for c in r['checks'])
        assert len(r['installer_projection_checks']) == 3 and all(c['exact_bf16'] for c in r['installer_projection_checks'])
        if name == 'integer': assert all(c['integer_oracle'] for c in r['checks'])
        else: assert r['timings'] and all(c['qkv_groups'] == check.CALLS for c in r['captures'])
    # Load the paired baseline before initializing a GPU or loading model weights.
    reference = json.loads(args.reference.read_text())
    assert reference.get('runs') and reference.get('warmup') is not None
  command[command.index('--output')+1] = str(args.output)
  for flag in ('warmup', 'runs'):
    value = getattr(args, flag)
    if value is not None:
      assert value >= (0 if flag == 'warmup' else 1)
      command[command.index('--'+flag)+1] = str(value)
  assert not args.output.exists(), 'Use a fresh report path'
  args.output.parent.mkdir(parents=True, exist_ok=True)
  dispatch = {'baseline': args.baseline, 'scope': args.scope, 'provenance': provenance, 'resolved_command': command,
              'micro_report': str(args.micro_report) if args.micro_report else None, 'installations': [], 'paired_exact': None}
  dispatch['environment'] = {k: os.environ.get(k) for k in ('DEV', 'CACHEDB', 'TC_OPT', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL')}
  status = args.output.with_name('agent-qkv-status.json')
  pipeline = local('benchmark-pipeline')
  loader = pipeline.local_module
  packed_types = []
  try:
    with ExitStack() as stack:
      def load(name):
        result = loader(name)
        if name == 'quantize-packed': packed_types.append(result.PackedActivation)
        if name in ('draft-decode', 'block-decode') and not args.baseline:
          cls = result.DraftDecoder if name == 'draft-decode' else result.BlockDecoder
          original_init = cls.__init__
          @functools.wraps(original_init)
          def initialize(decoder, model, *values, **kwargs):
            original_init(decoder, model, *values, **kwargs)
            if not dispatch['installations']:
              assert len(packed_types) == 1
              # Decoder construction follows all precision, quantizer, expert,
              # attention, projector, and vision hooks, before any JIT replay.
              dispatch['installations'].append(stack.enter_context(fusion.install(model, packed_types[0], args.scope)))
          stack.enter_context(patch.object(cls, '__init__', initialize))
        return result
      stack.enter_context(patch.object(pipeline, 'local_module', load))
      stack.enter_context(patch.object(sys, 'argv', command))
      check.save(status, dispatch)
      pipeline.main()
      if not args.baseline:
        assert len(dispatch['installations']) == 1 and dispatch['installations'][0]['python_trace_build_calls'], 'Fusion was never traced'
        report = json.loads(args.output.read_text())
        dispatch['paired_exact'] = compare(report, reference)
    dispatch['methods_restored'] = True
    report = json.loads(args.output.read_text())
    report['agent_qkv'] = dispatch
    check.save(args.output, report)
  except BaseException as exc:
    dispatch['failure'] = f'{type(exc).__name__}: {exc}'
    raise
  finally:
    check.save(status, dispatch)


if __name__ == '__main__': main()
