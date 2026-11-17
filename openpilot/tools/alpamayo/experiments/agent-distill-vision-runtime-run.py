"""One opt-in vision-only diagnostic under the parent's normal GPU wrapper. Default is CPU preflight."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
ROOT = WORK/'agent-distill-vision-runtime-artifacts'
PLAN = WORK/'agent-distill-command-artifacts/capture-v1/agent-distill-command-capture-plan.json'
PLAN_SHA = '5cb4a9baa5c448f507b64b250686d8cae16e43405a7d990b2c48c5212bf9871c'
TEACHERS = {
  'train-command-0': (WORK/'agent-distill-joint-teacher-v1/report.json', '5b796b1490bf234cf874a007be575612cb03599d61ba4cde041302e606b4052c'),
  'validation-command-0': (WORK/'agent-distill-joint-validation-v1/report.json', 'fdfc6a47355e9f164bc63df1ec240062931be43a0e05a56794593e6ade234d7c')}
SETTINGS = ('prefill_calibration_sha256', 'vision_calibration_sha256', 'initial_noise_sha256', 'settings', 'runtime_environment')


def module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
  return value


def hook(): return module('vision_runtime_hook_run', HERE/'agent-distill-vision-runtime-hook.py')


def checked_plan():
  H = hook()
  if H.sha(PLAN) != PLAN_SHA: raise ValueError('Changed selected capture recipe')
  plan = H.read(PLAN)
  # Reuse the existing source inventory; create no copied source tree or new pin chain.
  for name, digest in plan['runtime_files_sha256'].items():
    if H.sha(Path(plan['runtime_root'])/name) != digest: raise ValueError('Changed selected runtime source '+name)
  if H.sha(plan['inputs_manifest']['path']) != plan['inputs_manifest']['sha256']: raise ValueError('Changed admitted inputs')
  return plan


def teacher_reference(row, prepared, settings, plan, vision_cache):
  import numpy as np
  H = hook(); path, digest = TEACHERS[row['id']]
  if H.sha(path) != digest: raise ValueError('Changed teacher report')
  report = H.read(path)
  if not report['complete'] or report['mode'] != 'capture' or not report['gpu_executed']: raise ValueError('Incomplete teacher')
  if report['settings'] != settings: raise ValueError('Teacher arithmetic settings or noise changed')
  for source in (HERE/'agent-distill-joint-parent.py', HERE/'agent-distill-joint-runtime.py', Path(plan['runtime_root'])/'model.py'):
    if report['source_sha256'].get(str(source)) != H.sha(source): raise ValueError('Teacher source changed')
  saved = next(x for x in report['rows'] if x['id'] == row['id'])
  for key in ('split', 'route_id', 'query_ns', 'input_file_sha256', 'actual_prompt_tokens'):
    if saved[key] != row[key]: raise ValueError('Teacher input differs: '+key)
  keys = [dict(grid=[int(n) for n in key[0]], pixel_sha256=key[1].hex()) for key in vision_cache.input_keys(prepared)]
  if keys != saved['vision_input_keys'] or len(keys) != 10 or any(k['grid'] != [1, 20, 36] for k in keys):
    raise ValueError('Ten original full720 history images changed')
  if len(saved['targets']['vision_batches']) != 5 or any(x['layer_order'] != list(range(27)) for x in saved['targets']['vision_batches']):
    raise ValueError('Teacher vision execution incomplete')
  for name in ('xyz', 'rot'):
    if not np.isfinite(saved['teacher'][name]).all(): raise ValueError('Nonfinite teacher prediction')
  return dict(prediction=saved['teacher'], reused=True, report_sha256=digest,
              exact_inputs_sources_settings_noise=True, independent_anchor=True, performance_evidence=False)


def execute_prediction(options, plan, ready, weights, selection, *, cpu_mock=False, initialize=None):
  """Initialize injection is used only by the CPU test, after the actual selected launcher/parser."""
  import numpy as np
  from tinygrad.helpers import Context
  H = hook(); P = module('vision_runtime_actual_parent', HERE/'agent-distill-joint-parent.py')
  row, prepared, packed = ready
  if row['id'] not in TEACHERS: raise ValueError('Only train-command-0 or validation-command-0 is admitted')
  outcome = dict(complete=False, gpu_executed=not cpu_mock, cpu_integration_mock=cpu_mock,
                 sample_id=row['id'], quality_qualified=False, timings_valid_for_throughput=False,
                 sources={str(p): H.sha(p) for p in HERE.glob('agent-distill-vision-runtime-*.py')})
  def pipeline(value, args):
    components = (initialize or value.initialize)(args, sources=outcome['sources'], source_dir=Path(plan['runtime_root'])/'experiments')
    model = components.model
    if not cpu_mock and components.Device[components.Device.DEFAULT].arch != 'gfx1200': raise ValueError('Expected normal Chestnut inference target')
    if model.config['vlm_config']['text_config']['num_hidden_layers'] != 28 or model.config['expert_config']['llm_config']['num_hidden_layers'] != 28:
      raise ValueError('This diagnostic isolates vision: language and expert must retain28 blocks')
    if components.vision_cache.verify: raise ValueError('Disable teacher-cache verification during changed-architecture diagnostics')
    padded, pack = components.pad_inputs(prepared, packed, 2048)
    if padded['actual_prompt_length'] != row['actual_prompt_tokens']: raise ValueError('Changed prompt length')
    settings = {k: components.report[k] for k in SETTINGS}; outcome['settings'] = settings
    noise = np.asarray(components.settings.initial_noise, np.float32)
    if H.sha(PLAN) != PLAN_SHA or __import__('hashlib').sha256(noise.tobytes()).hexdigest() != settings['initial_noise_sha256']:
      raise ValueError('Initial diffusion noise changed')
    with Context(JIT=0):
      H.reset_state(model, components.decoder, components.vision_cache)
      outcome['teacher'] = teacher_reference(row, padded, settings, plan, components.vision_cache)
      old_config = model.config
      with H.vision_pilot(model, weights, selection=selection, decoder=components.decoder, vision_cache=components.vision_cache) as receipt:
        if model.cache is not None or components.decoder.history or components.vision_cache.cache: raise ValueError('Candidate did not start fresh')
        components.vision_cache.prepare(padded)
        inputs = components.input_class(components.settings, components.config, padded, pack)
        if args.selective_pixels: inputs.selective_cache = components.vision_cache
        prediction = components.runner(model, components.settings, components.config, components.tokenizer, padded, pack, seed=0, inputs=inputs)
        if len(model.cache) != 28: raise ValueError('Language cache depth changed')
        if components.vision_cache.misses != 10: raise ValueError('Candidate did not encode all ten historical images')
        for key, shape in (('xyz', (1,64,3)), ('rot', (1,64,3,3))):
          if np.asarray(prediction[key]).shape != shape or not np.isfinite(prediction[key]).all(): raise ValueError('Invalid candidate trajectory')
        outcome.update(adapter=receipt, candidate=prediction, candidate_vision_misses=10, current_language_cache_layers=28)
      if model.config is not old_config or model.cache is not None or components.vision_cache.cache:
        raise ValueError('Vision adapter failed to restore the original model and clear changed-feature state')
    teacher = outcome['teacher']['prediction']; delta = np.asarray(prediction['xyz'])-np.asarray(teacher['xyz'])
    outcome['comparison'] = dict(tokens_equal=prediction['generated_token_ids'] == teacher['generated_token_ids'],
                                 reasoning_equal=prediction['reasoning'] == teacher['reasoning'],
                                 xyz_max_absolute_m=float(np.abs(delta).max()),
                                 xyz_mean_distance_m=float(np.linalg.norm(delta, axis=-1).mean()))
    outcome.update(complete=True, adapter_restored=True, actual_launcher_parser_and_complete_runner=True)
  try: P.launch(plan, options.output, [row['sample']], pipeline)
  except BaseException as error:
    outcome['failure'] = dict(type=type(error).__name__, message=str(error)); raise
  finally: options.output.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
  return outcome


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--sample-id', choices=tuple(TEACHERS), required=True)
  parser.add_argument('--execute-gpu', action='store_true')
  args = parser.parse_args(); args.output = args.output.resolve(); root = args.output.parent
  allowed_parent_files = {'usb.acl', 'invocation.json', 'run.log'}
  if (args.output.name != 'report.json' or not root.is_relative_to(ROOT) or not root.is_dir()
      or any(p.name not in allowed_parent_files for p in root.iterdir())):
    raise ValueError('Use a fresh normal-wrapper directory under agent-distill-vision-runtime-artifacts')
  if subprocess.check_output(['findmnt', '-n', '-o', 'TARGET', '-T', str(root)], text=True, timeout=5).strip() != '/mnt/nvme-500':
    raise ValueError('500GB workspace is not mounted')
  if shutil.disk_usage(root).free < 5*1024**3: raise ValueError('Need5GiB free workspace')
  if args.execute_gpu:
    parent_args = Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    wrapper = b'/tmp/opencode/run-alpamayo-speed-trial.py'
    if wrapper not in parent_args or b'--program' not in parent_args or os.fsencode(__file__) not in parent_args:
      raise ValueError('Only the normal speed-trial wrapper may launch this diagnostic')
    # The normal ACL wrapper owns this lock. Direct unwrapped execution is rejected.
    with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
      try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError: pass
      else: raise ValueError('Parent normal GPU wrapper must hold the global lock')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('300-second vision diagnostic limit')))
    signal.alarm(300)
  os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                    CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
  if not args.execute_gpu: os.environ['DEV'] = 'CPU'
  plan = checked_plan(); H = hook()
  if args.execute_gpu:
    # Set the exact target/compiler environment before the first tinygrad import.
    seed = plan['compiler_cache_seed']
    if H.sha(seed['path']) != seed['sha256']: raise ValueError('Changed compiler cache seed')
    cache = root/'vision-kernels.db'; shutil.copyfile(seed['path'], cache)
    temporary = root/'tmp'; temporary.mkdir()
    invocation = H.read(plan['invocation_path']); invocation['environment'].update(CACHEDB=str(cache), TMPDIR=str(temporary))
    invocation_path = root/'vision-invocation.json'; invocation_path.write_text(json.dumps(invocation)+'\n')
    plan['invocation_path'] = str(invocation_path)
    for key, value in invocation['environment'].items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
  sys.path.insert(0, str(BASE/'tinygrad'))
  weights, selection = H.load_selected()
  P = module('vision_runtime_prepare', HERE/'agent-distill-joint-parent.py')
  row = P.selected_rows(P.INPUTS, [args.sample_id]); ready = P.prepare(plan, row)[0]
  if not args.execute_gpu:
    P.no_devices(); args.output.write_text(json.dumps(dict(complete=True, gpu_executed=False, preflight_only=True,
      selection=selection, sample_id=args.sample_id), indent=2)+'\n'); return
  execute_prediction(args, plan, ready, weights, selection)


if __name__ == '__main__': main()
