"""Standalone NUMERICAL whole-prompt FastV hook around the frozen pipeline.

Parent owns GPU execution and paired 64-window/full-route quality gates.
Retains 50% per image after two full language layers. Never claims exact parity.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
from types import MethodType
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = result
  spec.loader.exec_module(result)
  return result


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(command, flag, default=None):
  return command[command.index(flag)+1] if flag in command else default


def qualify_command(saved, output, runs=None, warmup=None):
  command = saved['command'][1:]
  if Path(command[0]).resolve() != (HERE/'benchmark-pipeline.py').resolve():
    raise ValueError('Only the adjacent frozen benchmark-pipeline.py is supported')
  if not all(flag in command for flag in ('--packed-quantize', '--prefill-calibration', '--pipeline-inputs',
                                        '--compact-inputs', '--decode-block')):
    raise ValueError('Requires fixed packed prefill, current-query input workers, compact inputs, and the real decoder')
  for flag in ('--validation-reference', '--fused-qkv', '--row-quantize', '--front-history-indices',
               '--capture-prefill-scales', '--capture-vision-scales'):
    if flag in command: raise ValueError(f'Incompatible numerical-policy option: {flag}')
  for flag in ('--pixel-scale', '--history-pixel-scale', '--history-feature-scale', '--narrow-crop'):
    if float(value(command, flag, '1')) != 1: raise ValueError('Preserve the full current visual inputs and histories')
  if '--camera-pixel-scales' in command:
    start = command.index('--camera-pixel-scales')+1
    if list(map(float, command[start:start+6])) != [1.0]*6: raise ValueError('Camera pixel reduction is outside this policy')
  for name, setting in (('output', output), ('runs', runs), ('warmup', warmup)):
    if setting is None: continue
    if name != 'output' and int(setting) < (1 if name == 'runs' else 0): raise ValueError(name)
    flag = '--'+name
    if flag in command: command[command.index(flag)+1] = str(setting)
    else: command += [flag, str(setting)]
  return command


def prepared_observations(data, observations):
  query, frames = observations
  cameras = tuple(map(int, data['camera_indices'].reshape(-1).tolist()))
  times = tuple(map(int, data['absolute_timestamps'].reshape(-1).tolist()))
  if cameras != tuple(camera for camera, _ in frames) or times != tuple(stamp//1000 for _, stamp in frames):
    raise ValueError('Prepared camera records do not match the current native query')
  if any(stamp > query for _, stamp in frames): raise ValueError('Unavailable future camera record')
  return query, frames


def snapshot_observations(data, sample, runtime):
  match = re.fullmatch(r'sample-(\d+)', Path(sample).stem)
  if match is None: raise ValueError('Snapshot query timestamp must be explicit in sample-<microseconds>')
  query = int(match[1])*1000
  cameras = data['camera_indices'].reshape(-1).tolist()
  stamps = data['absolute_timestamps'].reshape(-1).tolist()
  row = {'query_ns': query, 'frames': [{'view': int(camera), 'capture_ns': int(stamp)*1000}
                                     for camera, stamp in zip(cameras, stamps, strict=True)]}
  return runtime.current_observations({'current': row}, 'current')


def attach_decoder(decoder, runtime):
  owner = decoder.model._agent_token_pruner
  owner.register_decoder(decoder)
  original = decoder.run
  def run(self, model, args, config, tokenizer, prepared, packed, seed, inputs):
    prompt = int(prepared.get('actual_prompt_length', packed[0].shape[1]))
    observations = prepared.get('_agent_token_prune_observations')
    if observations is None: raise ValueError('Missing current native-query observations')
    grids = prepared['tokenized_data']['image_grid_thw'].tolist()
    spans = runtime.frame_spans(packed[0], config.vlm_config.image_token_id, prompt,
                                observations, inputs.spans, grids, strict=True)
    owner.prepare(prompt, spans)
    before = len(owner.events)
    prediction = original(model, args, config, tokenizer, prepared, packed, seed, inputs)
    if len(owner.events) != before+1: raise AssertionError('Expected exactly one current-input prefill policy decision')
    prediction['token_prune_policy'] = dict(owner.events[-1], query_ns=observations[0],
      availability_source=prepared['_agent_token_prune_availability_source'])
    return prediction
  decoder.run = MethodType(run, decoder)


def install_pipeline_hooks(pipeline, runtime, stack, installed):
  """Patch imported module objects only; preprocessing metadata stays thread-local."""
  availability = threading.local()
  original_loader = pipeline.local_module
  executor = pipeline.ThreadPoolExecutor
  class CurrentInputExecutor(executor):
    def submit(self, fn, *args, **kwargs):
      if fn.__name__ != 'prepare_input' or len(args) != 1:
        raise ValueError('Only the frozen prepare_input worker contract is supported')
      def current():
        availability.sample = args[0]
        try: return fn(*args, **kwargs)
        finally:
          for field in ('sample', 'current'):
            if hasattr(availability, field): delattr(availability, field)
      return super().submit(current)
  stack.enter_context(patch.object(pipeline, 'ThreadPoolExecutor', CurrentInputExecutor))
  def load(name):
    result = original_loader(name)
    if name == 'benchmark-fast':
      import benchmark
      constructor = benchmark.EdgeModel
      def create(*args, **kwargs):
        model = constructor(*args, **kwargs)
        owner = runtime.NumericalPruner(model, require_packed=True)
        installed.append(owner)
        stack.callback(owner.restore)
        return model
      stack.enter_context(patch.object(benchmark, 'EdgeModel', create))
    elif name == 'stream-inputs':
      original = result.NativeFrames.prepare
      def prepare(self, sample):
        observations = runtime.current_observations(self.rows, Path(sample).name)
        values = original(self, sample)
        availability.current = observations
        return values
      stack.enter_context(patch.object(result.NativeFrames, 'prepare', prepare))
    elif name == 'fast-inputs':
      original = result.FastInputs.prepare
      def prepare(self, data):
        observations = getattr(availability, 'current', None)
        if observations is None:
          sample = getattr(availability, 'sample', None)
          if sample is None: raise ValueError('Current snapshot query must precede tokenization')
          observations = snapshot_observations(data, sample, runtime)
          source = 'explicit current snapshot sample timestamp'
        else:
          del availability.current
          source = 'native current-row capture_ns <= query_ns'
        observations = prepared_observations(data, observations)
        config, tokenizer, prepared = original(self, data)
        return config, tokenizer, dict(prepared, _agent_token_prune_observations=observations,
                                      _agent_token_prune_availability_source=source)
      stack.enter_context(patch.object(result.FastInputs, 'prepare', prepare))
    elif name == 'padded-prefill':
      original = result.install
      def install(model):
        original(model)
        model._agent_token_pruner.install_padded()
      stack.enter_context(patch.object(result, 'install', install))
    elif name in ('block-decode', 'draft-decode', 'lookup-decode'):
      cls = getattr(result, {'block-decode': 'BlockDecoder', 'draft-decode': 'DraftDecoder', 'lookup-decode': 'LookupDecoder'}[name])
      original = cls.__init__
      def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        attach_decoder(self, runtime)
      stack.enter_context(patch.object(cls, '__init__', initialize))
    return result
  stack.enter_context(patch.object(pipeline, 'local_module', load))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--invocation', required=True, type=Path)
  parser.add_argument('--policy-report', type=Path, default=HERE/'agent-token-prune-numerical-policy-report.json')
  parser.add_argument('--compiler-report', type=Path, default=HERE/'agent-token-prune-compiler-report.json')
  parser.add_argument('--runs', type=int)
  parser.add_argument('--warmup', type=int)
  parser.add_argument('--dry-run', action='store_true', help='CPU preflight only; no imports of the model/device runtime')
  args = parser.parse_args()
  saved = json.loads(args.invocation.read_text())
  command = qualify_command(saved, args.output, args.runs, args.warmup)
  qualification = json.loads(args.policy_report.read_text())
  if not qualification.get('generation_causality_policy_eligible') or qualification.get('prompt_prefix_invariance_pass') is not False:
    raise ValueError('Missing separate numerical-policy CPU qualification')
  for name in ('agent-token-prune-runtime.py', 'agent-token-prune-primitives.py', 'agent-token-prune-reference.py'):
    if qualification['source_sha256'].get(name) != sha(HERE/name): raise ValueError(f'CPU qualification is stale: {name}')
  for name, expected in qualification['frozen_contract_sources'].items():
    if sha(HERE.parent/name) != expected: raise ValueError(f'Frozen pipeline contract changed: {name}')
  compiler = json.loads(args.compiler_report.read_text())
  if compiler['runtime_sha256'] != sha(HERE/'agent-token-prune-runtime.py') or len(compiler.get('records', [])) != 3:
    raise ValueError('Missing current static indexed-copy compilation checks')
  provenance = {'policy_report_sha256': sha(args.policy_report), 'invocation_sha256': sha(args.invocation),
                'compiler_report_sha256': sha(args.compiler_report),
                'calibration_sha256': sha(value(command, '--prefill-calibration')),
                'checkpoint_config_sha256': sha(Path(value(command, '--checkpoint'))/'config.json'),
                'checkpoint_index_sha256': sha(Path(value(command, '--checkpoint'))/'model.safetensors.index.json'),
                'classification': 'NUMERICAL NEW MODEL; whole-prompt-conditioned, not exact',
                'prompt_prefix_invariance_pass': False, 'retention': 0.5, 'original_command': command,
                'quality_passed': None, 'required_parent_gates': ['paired 64-window ADE/FDE', 'full-route ADE/FDE',
                                                               'trajectory shape', 'speed', 'slowing', 'reasoning']}
  if args.dry_run:
    print(json.dumps({'dry_run': True, 'no_gpu_opened': True, 'provenance': provenance}, indent=2))
    return
  if args.output.exists(): raise FileExistsError('Use a new parent trial output')
  for key, setting in saved['environment'].items():
    if setting is None: os.environ.pop(key, None)
    else: os.environ[key] = setting
  for path in reversed(os.environ.get('PYTHONPATH', '').split(os.pathsep)):
    if path: sys.path.insert(0, path)
  sys.path.insert(0, str(HERE))
  runtime = module('agent-token-prune-runtime')
  pipeline = module('benchmark-pipeline')
  installed = []
  with ExitStack() as stack:
    install_pipeline_hooks(pipeline, runtime, stack, installed)
    stack.enter_context(patch.object(sys, 'argv', command))
    pipeline.main()
    if len(installed) != 1 or not installed[0].events: raise AssertionError('Numerical prefill was not executed')
    report = json.loads(args.output.read_text())
    report['token_prune_numerical_policy'] = dict(provenance, events=installed[0].events,
       source_sha256={p.name: sha(p) for p in HERE.glob('agent-token-prune*.py')},
       measurement='Full existing pipeline boundary, including score readback, CPU ranking, gather/scatter, and fallback.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
