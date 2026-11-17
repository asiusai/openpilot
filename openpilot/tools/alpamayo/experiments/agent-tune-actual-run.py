"""Parent-only pipeline capture or strict replay of actual-AST tuning entries.

Uses the explicitly saved pipeline source. It never edits that source. Capture
timings include instrumentation and compilation, so they are not speed evidence.
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


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def compare(actual, reference):
  import numpy as np
  checked = 0
  for section in ('warmup', 'runs'):
    assert len(actual[section]) == len(reference[section]), 'Paired sample counts differ'
    for a, b in zip(actual[section], reference[section], strict=True):
      for key in ('sample', 'input_latest_timestamp_us'): assert a[key] == b[key], 'Paired inputs differ'
      for key in ('generated_tokens', 'generated_token_ids', 'reasoning'):
        assert a['prediction'][key] == b['prediction'][key], f'Exact {key} comparison failed'
      for key in ('xyz', 'rot'):
        x, y = (np.asarray(r['prediction'][key], np.float64) for r in (a, b))
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.testing.assert_array_equal(x.view(np.uint64), y.view(np.uint64), err_msg=f'Exact {key} comparison failed')
      checked += 1
  return {'predictions': checked, 'tokens_reasoning_xyz_rot_exact': True, 'full_route_validated': False}


def arm_at_decoder(pipeline, stack, stats, status, base, sidecar):
  loader = pipeline.local_module
  def local(name):
    result = loader(name)
    if name in ('draft-decode', 'block-decode'):
      cls = result.DraftDecoder if name == 'draft-decode' else result.BlockDecoder
      original = cls.__init__
      @functools.wraps(original)
      def initialize(decoder, model, *values, **kwargs):
        original(decoder, model, *values, **kwargs)
        if not status['inference_boundary_armed']:
          import tinygrad.codegen as cg
          # All model installers precede decoder construction. Clear only
          # in-process compiled-program lookup, then observe inference ASTs.
          cg.to_program_cache.clear()
          stats['enabled'] = True
          status['inference_boundary_armed'] = True
          base.atomic_json(sidecar, status)
      stack.enter_context(patch.object(cls, '__init__', initialize))
    return result
  stack.enter_context(patch.object(pipeline, 'local_module', local))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode', choices=('capture', 'replay'), required=True)
  p.add_argument('--invocation', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--cache', type=Path, required=True, help='Legacy cache to diagnose, or accepted actual-AST cache to replay')
  p.add_argument('--capture-dir', type=Path)
  p.add_argument('--reference', type=Path)
  p.add_argument('--limit', type=int, default=16)
  p.add_argument('--warmup', type=int, default=0)
  p.add_argument('--runs', type=int, default=2)
  args = p.parse_args()
  assert 1 <= args.limit <= 32 and 0 <= args.warmup <= 24 and 1 <= args.runs <= 48
  if args.mode == 'capture': assert args.warmup == 0 and args.runs <= 2, 'Capture is a bounded diagnostic'
  else: assert args.reference is not None, 'Replay requires an independent paired pipeline report'
  saved = json.loads(args.invocation.read_text())
  command = saved['command'][1:]
  program = Path(command[0]).resolve()
  assert program.name == 'benchmark-pipeline.py' and program.exists()
  root = program.parent.parent
  assert '--draft-previous' in command or '--decode-block' in command, 'A known inference boundary is required'
  assert not any('agent-vendor' in part for part in command), 'Keep the vendor experiment separate'
  parent_cache = os.environ.get('CACHEDB')
  for key, value in saved['environment'].items():
    if value is None: os.environ.pop(key, None)
    else: os.environ[key] = value
  if parent_cache is not None: os.environ['CACHEDB'] = parent_cache
  os.environ.update(BEAM='0', JITBEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  for path in reversed(os.environ.get('PYTHONPATH', '').split(os.pathsep)):
    if path: sys.path.insert(0, path)
  sys.path.insert(0, str(root))
  for flag, value in (('output', args.output), ('warmup', args.warmup), ('runs', args.runs)):
    command[command.index('--'+flag)+1] = str(value)
  assert not args.output.exists(), 'Use a fresh report path'
  args.output.parent.mkdir(parents=True, exist_ok=True)
  helper = load(HERE/'agent-tune-actual.py', 'agent_tune_actual_run_helper')
  base = helper.load_base(root)
  reference = json.loads(args.reference.read_text()) if args.reference is not None else None
  cache = json.loads(args.cache.read_text())
  if args.mode == 'replay':
    assert cache['entries'] and all(row['oracle'] == 'numpy-preopt-exact' and row['validated'] and row['accepted']
                                    for row in cache['entries'].values()), 'No accepted exact actual-AST entries'
  wgp = '--llvm-wgp' in command
  directory = args.capture_dir or args.output.with_name('agent-tune-actual-capture')
  status = {'mode': args.mode, 'resolved_command': command, 'pipeline_sha256': base.file_sha(program),
            'helper_sha256': base.file_sha(helper.__file__), 'source_root': str(root), 'parallel_forced_zero': True,
            'capture_directory': str(directory) if args.mode == 'capture' else None, 'inference_boundary_armed': False,
            'timing_is_diagnostic': args.mode == 'capture', 'paired_exact': None, 'full_route_validated': False}
  sidecar = args.output.with_name('agent-tune-actual-status.json')
  pipeline = load(program, 'agent_tune_actual_pipeline')
  try:
    with ExitStack() as stack:
      stats = stack.enter_context(helper.capture(base, directory, wgp, limit=args.limit, legacy_cache=args.cache)
                                  if args.mode == 'capture' else helper.replay(base, args.cache, wgp))
      status['optimizer'] = stats
      arm_at_decoder(pipeline, stack, stats, status, base, sidecar)
      stack.enter_context(patch.object(sys, 'argv', command))
      base.atomic_json(sidecar, status)
      pipeline.main()
      assert status['inference_boundary_armed'], 'No inference boundary was reached'
      if args.mode == 'capture':
        assert stats['gpu_seen'] and stats['records'], 'No eligible actual inference AST was captured'
      else:
        assert stats['hits'], 'No actual-AST tuning entry was applied'
        status['paired_exact'] = compare(json.loads(args.output.read_text()), reference)
    status['hooks_restored'] = True
    report = json.loads(args.output.read_text())
    report['agent_tune_actual'] = status
    base.atomic_json(args.output, report)
  except BaseException as exc:
    status['failure'] = f'{type(exc).__name__}: {exc}'
    raise
  finally: base.atomic_json(sidecar, status)


if __name__ == '__main__': main()
