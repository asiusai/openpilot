"""Run one parent-owned paired composition trial from a frozen protocol."""
import argparse
from contextlib import ExitStack
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('combined_common_run', HERE/'agent-exact-combined-common.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def save(path, value):
  Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def hook_chain(composition):
  current = importlib.util.spec_from_file_location
  if getattr(current, '__self__', None) is composition: return
  if Path(current.__code__.co_filename).resolve() != c.MLP/'experiments/agent-speculation-run.py':
    raise ValueError('Unexpected outer import hook')
  previous = inspect.getclosurevars(current).nonlocals.get('original_spec')
  if getattr(previous, '__self__', None) is not composition:
    raise ValueError('Canonical speculation hook bypassed the composition')


def run(args):
  manifest = c.read(args.manifest)
  c.verify(manifest)
  if args.output.exists() or args.output.with_suffix('.agent-exact-combined.json').exists():
    raise FileExistsError('Use a fresh paired output path')
  stage = manifest['stages'][args.stage]
  invocation = c.read(stage['invocation'])
  if Path(sys.executable).resolve() != Path(invocation['command'][0]).resolve():
    raise ValueError('Use the recorded Python executable')
  reference = None
  if args.mode == 'candidate':
    if args.reference is None: raise ValueError('Candidate requires its completed paired reference')
    reference = c.read(args.reference)
    prior = reference['agent_exact_combined']
    if not prior['passed'] or prior['mode'] != 'reference' or prior['stage'] != args.stage:
      raise ValueError('Reference trial is not qualified')
    if prior['manifest_sha256'] != c.sha(args.manifest): raise ValueError('Paired protocol identity differs')
  elif args.reference is not None: raise ValueError('Reference mode cannot consume another reference')
  command = invocation['command'][1:]
  command[command.index('--output')+1] = str(args.output)
  status_path = args.output.with_suffix('.agent-exact-combined.json')
  status = dict(stage=args.stage, mode=args.mode, manifest_sha256=c.sha(args.manifest),
    command=command, passed=False, hooks_restored=False, paired_exact=None,
    original_kernel_evidence=manifest['evidence'], original_harness_audit=manifest['audit'],
    source_sha256={path:digest for path,digest in manifest['files'].items()
                   if Path(path).name in ('agent-packed-epilogue.py','agent-fused-mlp.py','agent-vision-tiles.py',
                     'agent-warp-fast-sampler.py','agent-warp-fast-kernel.c','agent-warp-fast-workers.c')},
    combined_gpu_qualified=False, speed_claim=False)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  original_spec = importlib.util.spec_from_file_location
  composition = None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ, dict(os.environ), clear=True))
      for key, value in invocation['environment'].items():
        if value is None: os.environ.pop(key, None)
        else: os.environ[key] = str(value)
      original_path = list(sys.path)
      stack.callback(lambda:sys.path.__setitem__(slice(None), original_path))
      sys.path[:0] = [str(c.TINYGRAD), str(c.MLP/'experiments'), str(c.MLP)]
      installer = c.load(HERE/'agent-exact-combined-install.py', 'combined_installer_run')
      composition = installer.Composition(c, manifest, args.stage, args.mode == 'candidate')
      with composition:
        stack.enter_context(patch.object(sys, 'argv', command))
        save(status_path, status)
        runpy.run_path(command[0], run_name='__main__')
        hook_chain(composition)
        status['coverage'] = composition.coverage()
        actual = c.read(args.output)
        c.validate_report(actual, args.stage)
        if reference is not None:
          status['paired_exact'] = c.compare(reference, actual, args.stage)
          old_spec = c.read(args.reference.with_suffix('.speculation.json'))
          new_spec = c.read(args.output.with_suffix('.speculation.json'))
          c.equal(old_spec, new_spec, 'complete_speculation_records')
      status['restorations'] = composition.restorations
      status['hooks_restored'] = (importlib.util.spec_from_file_location is original_spec and
        all(row['restored'] for row in composition.restorations) and
        (composition.session is None or composition.session.stats['methods_restored']))
      if not status['hooks_restored']: raise ValueError('Composed hooks did not restore cleanly')
    c.verify(manifest)
    status['passed'] = True
    status['combined_gpu_qualified'] = args.mode == 'candidate'
    actual['agent_exact_combined'] = status
    save(args.output, actual)
  except BaseException as exc:
    status['failure'] = dict(type=type(exc).__name__, message=str(exc))
    status['hooks_restored'] = importlib.util.spec_from_file_location is original_spec
    if composition is not None: status['restorations'] = composition.restorations
    raise
  finally:
    save(status_path, status)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--stage', choices=('quality','native'), required=True)
  parser.add_argument('--mode', choices=('reference','candidate'), required=True)
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  run(parser.parse_args())
