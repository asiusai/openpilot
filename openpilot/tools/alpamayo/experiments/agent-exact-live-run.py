"""Parent-only fresh parity or separately reviewed paced execution."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exact_live_run_common',HERE/'agent-exact-live-common.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(args):
  manifest = m.read(args.manifest)
  m.verify(manifest)
  if Path(sys.executable).resolve() != Path(m.c.PYTHON).resolve(): raise ValueError('Use the recorded Python executable')
  if any(args.output.parent.glob(args.output.stem+'*')): raise FileExistsError('Use fresh exact live outputs')
  if args.mode != 'reference' and args.reference is None: raise ValueError('Require the fresh paired reference')
  prior = None
  if args.mode == 'consumer':
    prior = m.read(args.reference)
    s = prior['agent_exact_live']
    if not s['passed'] or s['mode'] != 'reference' or s['manifest_sha256'] != m.sha(args.manifest):
      raise ValueError('Reference did not complete this frozen protocol')
  if args.mode == 'paced':
    if args.parity is None or args.reviewed_parity_sha256 != m.sha(args.parity):
      raise ValueError('Parent must inspect parity and pass its exact report hash separately')
    m.qualify(m.read(args.reference),m.read(args.parity),manifest,m.sha(args.manifest))
  status = dict(mode=args.mode,manifest_sha256=m.sha(args.manifest),passed=False,gpu_executed=False,
    hooks_restored=False,actual_parity_reused=False,original_evidence=manifest['evidence'],speed_claim=False)
  sidecar = args.output.with_suffix('.agent-exact-live.json')
  args.output.parent.mkdir(parents=True,exist_ok=True)
  original_spec = importlib.util.spec_from_file_location
  hooks = None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ))
      for key,value in m.read(m.INVOCATION)['environment'].items():
        if value is None: os.environ.pop(key,None)
        else: os.environ[key] = str(value)
      os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
      stack.enter_context(patch.object(sys,'path',list(sys.path)))
      sys.path[:0] = [str(m.c.TINYGRAD),str(m.ROOT),str(m.ROOT/'experiments'),
        '/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
      installer = m.load(HERE/'agent-exact-live-install.py','exact_live_runner_installer')
      hooks = installer.Hooks(m,manifest,args.mode)
      with hooks:
        if args.mode == 'paced':
          module = m.load(m.ROOT/'experiments/agent-paced-replay.py','exact_live_paced_runner')
          options = ['--mode','run','--invocation',str(m.INVOCATION),'--consumer-parity-report',str(args.parity),
            '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180',
            '--raw-frames',str(m.RGB),'--warp','native','--output',str(args.output)]
        else:
          module = m.load(m.ROOT/'experiments/agent-live-replay.py','exact_live_parity_runner')
          options = ['--invocation',str(m.INVOCATION),'--mode',args.mode,'--decoder','speculative',
            '--prefill-attention','shuffle-fma','--execute-gpu','--output',str(args.output)]
          if args.mode == 'consumer': options += ['--compare-reference',str(args.reference)]
        module.main(options)
        expected = manifest['expected_transforms']
        expected = expected if args.mode == 'paced' else {k:v for k,v in expected.items() if k != 'paced'}
        m.equal(hooks.transforms,expected,'source_bound_startup_and_query_transforms')
        status.update(coverage=hooks.coverage(),transforms=hooks.transforms,gpu_executed=True,
          source_sha256={path:sha for path,sha in manifest['files'].items() if Path(path).name in
            ('agent-packed-epilogue.py','agent-fused-mlp.py','agent-vision-tiles.py','agent-warp-fast-sampler.py',
             'agent-warp-fast-kernel.c','agent-warp-fast-workers.c','agent-startup-spec-warm.py')})
        actual = m.read(args.output)
        actual['startup_spec'] = hooks.startup_report
        if args.mode != 'reference':
          warm = hooks.startup_report
          for key in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained'):
            if warm.get(key) is not True: raise ValueError('Original full startup restoration failed')
        if prior is not None: status['comparison'] = m.compare(prior,actual)
        if args.mode == 'paced':
          if (actual['requested_hz'] != 10 or actual['source_window_s'] != 8 or actual['maximum_active_inference'] != 1 or
              actual['queued_prediction_snapshots'] or actual['inference_budget_seconds'] != 180):
            raise ValueError('Original paced scheduling bounds changed')
          if not actual['rows'] or actual['rows'][0]['vision_hits'] != 0 or actual['rows'][0]['vision_misses'] != 10:
            raise ValueError('First cold logical image-cache cost was not retained')
      status['hooks_restored'] = (importlib.util.spec_from_file_location is original_spec and
        (hooks.composition.session is None or hooks.composition.session.stats['methods_restored']))
      status['restorations'] = hooks.restorations+hooks.composition.restorations
      if not status['hooks_restored']: raise ValueError('Live composition restoration failed')
    m.verify(manifest)
    status['passed'] = True
    actual['agent_exact_live'] = status
    if args.mode == 'consumer': m.qualify(prior,actual,manifest,m.sha(args.manifest))
    save(args.output,actual)
  except BaseException as error:
    status['passed'] = False
    status['failure'] = dict(type=type(error).__name__,message=str(error))
    status['hooks_restored'] = importlib.util.spec_from_file_location is original_spec
    raise
  finally: save(sidecar,status)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest',type=Path,required=True)
  parser.add_argument('--mode',choices=('reference','consumer','paced'),required=True)
  parser.add_argument('--reference',type=Path)
  parser.add_argument('--parity',type=Path)
  parser.add_argument('--reviewed-parity-sha256')
  parser.add_argument('--output',type=Path,required=True)
  run(parser.parse_args())
