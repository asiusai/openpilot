#!/usr/bin/env python3
"""Standalone original-pipeline sparsity diagnostic. --prepare/--preflight are CPU only.

Use reference and observe in separate fresh processes. Observe requires a passing
parent GPU control report and bit-exact paired model outputs. No throughput claim.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
ALLOWED = WORK/'calibration-samples.json'
ALLOWED_SHA = '41179a62dc25bd164b8cda4c11ff83df6a42086a9ea7837acf7821687d0ddb4d'
INDICES = (0, 4, 8, 12, 16, 20, 24, 28)
FILES = ('absolute_timestamps.npy', 'camera_indices.npy', 'ego_history_rot.npy',
         'ego_history_xyz.npy', 'image_frames.npy', 'nav_text.npy')
SCHEMA = 'agent-activation-sparsity-v1'


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()


def sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as f:
    while b := f.read(8*1024*1024): h.update(b)
  return h.hexdigest()


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def blocks(command):
  result, key = {}, None
  for item in command[2:]:
    if item.startswith('--'):
      if '=' in item or item in result: raise ValueError('Ambiguous invocation option')
      key = item;result[key] = []
    else:
      if key is None: raise ValueError('Unexpected positional option')
      result[key].append(item)
  return result


def one(opts, name, default=None):
  v = opts.get('--'+name)
  if v is None: return default
  if len(v) != 1: raise ValueError('Require one value for '+name)
  return v[0]


def input_inventory(sample):
  files = {f: dict(bytes=(Path(sample)/f).stat().st_size, sha256=sha(Path(sample)/f)) for f in FILES}
  return dict(path=sample, files=files, sha256=digest(files),
              nav_text=str(np.load(Path(sample)/'nav_text.npy', allow_pickle=False).item()))


def selected():
  if sha(ALLOWED) != ALLOWED_SHA: raise ValueError('Frozen allowed calibration list changed')
  allowed = json.loads(ALLOWED.read_text())
  if len(allowed) != 32 or len(set(allowed)) != 32: raise ValueError('Require exactly original allowed32')
  paths = [allowed[i] for i in INDICES]
  # Selection is fixed by original allowed-list indices, never by measured sparsity.
  return [input_inventory(p) for p in paths]


def source_inventory():
  files = [p for p in HERE.glob('*.py') if not p.name.startswith('agent-')]
  files += list(HERE.glob('agent-activation-sparsity*.py'))
  files += list(HERE.glob('*.c'))
  files += list(HERE.parent.glob('*.py'))
  files += list((BASE/'tinygrad/tinygrad').rglob('*.py'))
  return {str(p.resolve()): sha(p) for p in sorted(set(files))}


def prepare(args):
  saved = json.loads(args.invocation.read_text())
  command = saved['command']
  if Path(command[1]).resolve() != HERE/'benchmark-pipeline.py':
    raise ValueError('Use the adjacent original benchmark-pipeline.py invocation')
  opts = blocks(command)
  required = ('packed-quantize', 'prefill-calibration', 'compact-inputs', 'byte-pixels',
              'direct-byte-inputs', 'decode-block', 'initial-noise', 'allow-missing-cameras', 'pad-prompt-to')
  if any('--'+key not in opts for key in required): raise ValueError('Missing selected original pipeline option')
  for key in ('native-manifest', 'native-calibrations', 'validation-reference', 'warmup-graphs', 'warm-current-shapes',
              'row-quantize', 'fused-qkv', 'prefill-kernels', 'front-history-indices', 'capture-prefill-scales',
              'capture-vision-scales', 'profile-kernels', 'profile-host'):
    if '--'+key in opts: raise ValueError('Unsupported standalone diagnostic option: '+key)
  for key in ('pixel-scale', 'history-pixel-scale', 'history-feature-scale', 'narrow-crop'):
    if float(one(opts, key, 1)) != 1: raise ValueError('Preserve complete full-resolution histories')
  if '--camera-pixel-scales' in opts and any(float(v) != 1 for v in opts['--camera-pixel-scales']):
    raise ValueError('Preserve all camera pixels')
  if (one(opts, 'camera-profile') != 'nvidia' or one(opts, 'steps') != '3'
      or one(opts, 'trajectory-solver', 'euler') != 'euler' or one(opts, 'pad-prompt-to') != '2112'
      or int(one(opts, 'bf16-vision-layers', '0')) != 2 or '--bf16-expert-projections' not in opts
      or float(one(opts, 'wide-equidistant-focal', 0)) != 0):
    raise ValueError('Require selected original FP8/full-history/2112-row/three-Euler policy')
  checkpoint = Path(one(opts, 'checkpoint')).resolve()
  reference = json.loads((HERE/'agent-int4model-artifacts/reference-provenance.json').read_text())['reference']
  if str(checkpoint) != reference['checkpoint']['path']: raise ValueError('Wrong original checkpoint')
  disk = {}
  for name, record in reference['checkpoint']['files'].items():
    path = checkpoint/name
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
      raise ValueError('Original BF16 checkpoint changed: '+name)
    disk[str(path)] = record
  cache = Path(saved['environment']['ALPAMAYO_FP8_CACHE']).resolve()
  expected = reference['fp8_cache']
  if str(cache) != expected['path']: raise ValueError('Wrong original FP8 cache')
  for name, digest_key in (('index.json', 'index_sha256'), ('weights.bin', 'weights_sha256')):
    path = cache/name
    if sha(path) != expected[digest_key]: raise ValueError('Original FP8 cache bytes changed')
    disk[str(path)] = dict(bytes=path.stat().st_size, sha256=expected[digest_key])
  for name in ('prefill-calibration', 'vision-calibration', 'initial-noise', 'reduced-vocab'):
    path = one(opts, name)
    if path:
      path = Path(path).resolve();disk[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
  noise = np.load(one(opts, 'initial-noise'), allow_pickle=False)
  if noise.shape != (1, 64, 2) or noise.dtype != np.float32 or not np.isfinite(noise).all(): raise ValueError('Invalid initial noise')
  environment = saved['environment']
  if environment['PYTHONPATH'].split(':') != [str(BASE/'tinygrad'), str(HERE.parent)]:
    raise ValueError('Require explicitly pinned adjacent model and tinygrad paths')
  if environment.get('DEV') != 'USB+AMD:LLVM': raise ValueError('Require the selected gfx1200 LLVM GPU backend')
  selected_inputs = selected()
  if not (all(r['nav_text'] == 'Follow the road.' for r in selected_inputs[:4])
          and all(r['nav_text'].startswith('Turn right') for r in selected_inputs[4:])):
    raise ValueError('The fixed straight/right calibration subsets changed')
  # Numerical options remain byte-for-byte. Only explicit diagnostic scheduling changes.
  for key in ('output', 'samples', 'warmup', 'runs'): opts.pop('--'+key, None)
  plan = dict(schema=SCHEMA, invocation=str(args.invocation.resolve()), invocation_sha256=sha(args.invocation),
    pipeline=str(HERE/'benchmark-pipeline.py'), numerical_options=opts, environment=environment,
    selected_inputs=selected_inputs, allowed_list_sha256=ALLOWED_SHA, allowed_indices=list(INDICES),
    checkpoint_identity=reference['checkpoint']['identity_sha256'], disk_files=disk,
    sources=source_inventory(), initial_noise_array_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
    tinygrad_revision=subprocess.check_output(['git', '-C', str(BASE/'tinygrad'), 'rev-parse', 'HEAD'], text=True).strip(),
    warmup=2, runs=8, activation_readback=False, route_is_blind_test=False,
    observation='original language prefill fc2 BF16 ReLU-square and actual selected packed FP8 bytes',
    measurement='DIAGNOSTIC; extra GPU reduction and small statistics readback invalidate throughput timings')
  plan['identity_sha256'] = digest(plan)
  return plan


def verify(plan):
  if plan['schema'] != SCHEMA or digest({k: v for k, v in plan.items() if k != 'identity_sha256'}) != plan['identity_sha256']:
    raise ValueError('Invalid frozen diagnostic plan')
  if sha(ALLOWED) != plan['allowed_list_sha256'] or selected() != plan['selected_inputs']:
    raise ValueError('Allowed/current sample input content changed')
  for path, expected in plan['sources'].items():
    if sha(path) != expected: raise ValueError('Frozen source changed: '+path)
  for path, record in plan['disk_files'].items():
    if Path(path).stat().st_size != record['bytes'] or sha(path) != record['sha256']:
      raise ValueError('Frozen checkpoint/cache/calibration/noise bytes changed: '+path)
  if plan['tinygrad_revision'] != '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e':
    raise ValueError('Unsupported pinned tinygrad revision')


def array_identity(value):
  if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
  a = np.ascontiguousarray(np.asarray(value))
  if a.dtype.hasobject: raise ValueError('Object arrays do not establish provenance')
  return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=hashlib.sha256(a.tobytes()).hexdigest())


def prepared_identity(prepared, packed):
  tokens = prepared['tokenized_data']
  values = dict(input_ids=packed[0], position_ids=packed[1], rotary_delta=np.asarray(packed[2]),
    vision_position=packed[3], ego_history_xyz=prepared['ego_history_xyz'], ego_history_rot=prepared['ego_history_rot'],
    **prepared['_activation_sparsity_source'])
  # Keep actual native IDs separate from tokenizer IDs containing history_pad.
  values.update({'tokenized_'+name: value for name, value in tokens.items()})
  records = {name: array_identity(value) for name, value in values.items()}
  return dict(arrays=records, sha256=digest(records))


def prediction_identity(prediction):
  # All numerical outputs must be bit-exact; diagnostic timing fields differ.
  record = {key: prediction[key] for key in ('prompt_tokens', 'prefill_tokens', 'generated_tokens',
    'generated_token_ids', 'reasoning', 'xyz', 'rot')}
  return dict(values=record, sha256=digest(record))


def initialize_checked(initialize, device_guard, result, *args, **kwargs):
  """Use the shared actual identity gate before and after the original factory."""
  before = device_guard.require_gpu_device()
  components = initialize(*args, **kwargs)
  after = device_guard.require_gpu_device()
  if after != before: raise ValueError('GPU identity changed during original pipeline initialization')
  result['gpu_device_identity'] = after
  return components


def execute(args, plan):
  paired = None
  if args.mode == 'observe':
    if args.kernel_report is None or args.compare_reference is None: raise ValueError('Observe requires passing GPU controls and paired reference')
    gate = json.loads(args.kernel_report.read_text())
    if gate.get('cpu_only') is not False or not gate.get('complete') or not gate.get('all_pass'):
      raise ValueError('Require complete actual-GPU controls first')
    for name, expected in gate['sources'].items():
      if sha(HERE/name) != expected: raise ValueError('GPU gate source changed: '+name)
    for path, expected in gate['quantizer_dependencies'].items():
      if plan['sources'].get(path) != expected or sha(path) != expected:
        raise ValueError('Actual selected quantizer differs from parent GPU controls: '+path)
    paired = json.loads(args.compare_reference.read_text())
    if paired.get('mode') != 'reference' or not paired.get('complete') or paired.get('plan_id') != plan['identity_sha256']:
      raise ValueError('Require completed paired reference from the identical frozen plan')
  if args.output.exists(): raise FileExistsError('Use a new trial output')
  result = dict(schema=SCHEMA, mode=args.mode, complete=False, plan_id=plan['identity_sha256'], plan=plan,
                timings_valid_for_throughput=False, queries=[], gpu_gate_sha256=sha(args.kernel_report) if args.kernel_report else None)
  for key, value in plan['environment'].items():
    if value is None: os.environ.pop(key, None)
    else: os.environ[key] = value
  for path in reversed(os.environ['PYTHONPATH'].split(':')): sys.path.insert(0, path)
  sys.path.insert(0, str(HERE))
  pipeline = local('benchmark-pipeline')
  device_guard = local('agent-activation-sparsity-device')
  paths = [r['path'] for r in plan['selected_inputs']]
  opts = dict(plan['numerical_options'], **{'--samples': paths, '--warmup': ['2'], '--runs': ['8'], '--output': [str(args.output.with_suffix('.pipeline.json'))]})
  command = [plan['pipeline'], *[item for key, values in opts.items() for item in (key, *values)]]
  observer = None
  try:
    with ExitStack() as stack:
      original_loader = pipeline.local_module
      def load(name):
        module = original_loader(name)
        path = HERE/(name+'.py')
        if str(path) not in plan['sources'] or sha(path) != plan['sources'][str(path)]:
          raise ValueError('Unfrozen runtime module: '+str(path))
        if name == 'fast-inputs':
          original_prepare = module.FastInputs.prepare
          def prepare_input(adapter, data):
            config, tokenizer, prepared = original_prepare(adapter, data)
            source = {key: data[key] for key in ('camera_indices', 'absolute_timestamps', 'nav_text')}
            return config, tokenizer, dict(prepared, _activation_sparsity_source=source)
          stack.enter_context(patch.object(module.FastInputs, 'prepare', prepare_input))
        return module
      stack.enter_context(patch.object(pipeline, 'local_module', load))
      original_initialize = pipeline.initialize
      def initialize(*values, **kwargs):
        nonlocal observer
        components = initialize_checked(original_initialize, device_guard, result, *values, **kwargs)
        if args.mode == 'observe':
          observer = local('agent-activation-sparsity').Observer(components.model)
          stack.callback(observer.restore)
        original_runner = components.runner
        def run(model, settings, config, tokenizer, prepared, packed, seed, inputs):
          index = len(result['queries'])
          if index >= 10: raise ValueError('Unexpected extra prediction')
          valid = int(prepared['actual_prompt_length'])
          identity = prepared_identity(prepared, packed)
          # Both complete histories and every timestamp enter the unchanged tokenizer.
          cams = np.asarray(prepared['_activation_sparsity_source']['camera_indices']).reshape(-1)
          stamps = np.asarray(prepared['_activation_sparsity_source']['absolute_timestamps']).reshape(-1)
          if sorted(cams.tolist()) != [1]*5+[6]*5 or len(stamps) != 10: raise ValueError('Missing real camera history')
          query_us = int(Path(paths[index % 8]).name.split('-')[-1])
          if (stamps > query_us).any(): raise ValueError('Future camera timestamp')
          before = sum(observer.trace_counts.values()) if observer else None
          if observer: observer.prepare(valid)
          prediction = original_runner(model, settings, config, tokenizer, prepared, packed, seed, inputs)
          entry = dict(index=index, sample=paths[index % 8], phase='warmup' if index < 2 else 'diagnostic',
                       valid_rows=valid, padded_rows=packed[0].shape[1]-valid, prepared=identity,
                       prediction=prediction_identity(prediction))
          if observer:
            entry['layers'] = observer.read()
            after = sum(observer.trace_counts.values())
            entry['python_fc2_traces_before'], entry['python_fc2_traces_after'] = before, after
            entry['captured_replay'] = before == after
            expected = paired['queries'][index]
            if expected['prepared'] != identity or expected['prediction'] != entry['prediction'] or expected['sample'] != entry['sample']:
              raise ValueError('Paired original numerical outputs or exact CURRENT inputs changed')
            entry['paired_output_bits_exact'] = True
            for name, stats in entry['layers'].items():
              for bank in ('bf16_before', 'fp8_after_actual'):
                if stats[bank]['nonfinite'] or stats[bank]['negative_nonzero']:
                  raise ValueError('Actual post-ReLU-square nonfinite/negative violation: '+name)
          result['queries'].append(entry)
          # Incremental small report survives a later model/device failure.
          args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
          return prediction
        components.runner = run
        return components
      stack.enter_context(patch.object(pipeline, 'initialize', initialize))
      stack.enter_context(patch.object(sys, 'argv', command))
      pipeline.main()
      if len(result['queries']) != 10: raise ValueError('Incomplete bounded sample schedule')
      if observer:
        if not all(q['captured_replay'] for q in result['queries'][2:]):
          raise ValueError('Diagnostic did not prove current statistics during all eight JIT replays')
        result['python_fc2_trace_counts'] = observer.trace_counts
        result['stats_readback_bytes_per_query'] = 28*(72+4)*4
        result['persistent_stats_bytes'] = 28*(32*72+76)*4+8
        summary = {}
        for bank in ('bf16_before', 'fp8_after_actual'):
          records = [stats[bank] for query in result['queries'][2:] for stats in query['layers'].values()]
          summary[bank] = {metric: dict(min=float(min(v[metric] for v in records)),
            median=float(np.median([v[metric] for v in records])), max=float(max(v[metric] for v in records)))
            for metric in ('zero_fraction', 'exact_2of4_eligible_fraction', 'row_k32_eligible_fraction', 'tile16_k32_eligible_fraction')}
        result['summary'] = dict(diagnostic_only=True, timings_valid_for_throughput=False,
          samples=8, layers=28, all_paired_outputs_bits_exact=True, sample_layer_ratios=summary)
      result['complete'] = True
  except BaseException as e:
    result['error'] = dict(type=type(e).__name__, message=str(e));raise
  finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--plan', type=Path, default=HERE/'agent-activation-sparsity-plan.json')
  p.add_argument('--invocation', type=Path, default=WORK/'full-padded-quality/invocation.json')
  p.add_argument('--mode', choices=('reference', 'observe'), default='reference')
  p.add_argument('--kernel-report', type=Path)
  p.add_argument('--compare-reference', type=Path)
  p.add_argument('--prepare', action='store_true', help='CPU-only exact source/input/checkpoint manifest; output becomes the plan')
  p.add_argument('--preflight', action='store_true', help='CPU-only verification of a frozen plan')
  args = p.parse_args()
  if args.prepare:
    if args.output.exists(): raise FileExistsError('Use a new plan filename')
    plan = prepare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cpu_only=True, plan_id=plan['identity_sha256'], output=str(args.output), selected=8)))
    return
  plan = json.loads(args.plan.read_text());verify(plan)
  if args.preflight:
    print(json.dumps(dict(cpu_only=True, all_provenance_verified=True, plan_id=plan['identity_sha256'], selected=8)));return
  execute(args, plan)


if __name__ == '__main__': main()
