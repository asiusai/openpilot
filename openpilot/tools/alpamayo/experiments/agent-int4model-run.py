#!/usr/bin/env python3
"""Explicit recorded-pipeline reference/collect/candidate runner. --preflight is CPU only.

The reviewed bundle/exporter/collector remain unchanged. No default runtime edits.
Actual execution initializes the parent's recorded GPU and must be run by the parent.
"""
import argparse
from contextlib import ExitStack
from functools import partial, wraps
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import subprocess
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SCHEMA = 'agent-int4model-run-v1'


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), EXP / (name + '.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


ptq = local('agent-int4model')  # NumPy/stdlib only; no tinygrad import.


def option_blocks(command):
  """Retain every numerical flag; reject ambiguous or duplicated spellings."""
  if len(command) < 2:
    raise ValueError('Recorded command requires Python and benchmark-pipeline.py')
  blocks, order, active = {}, [], None
  for token in command[2:]:
    if token.startswith('--'):
      if '=' in token or token in blocks:
        raise ValueError(f'Ambiguous recorded argument: {token}')
      active = token
      blocks[token] = []
      order.append(token)
    elif active is None:
      raise ValueError('Unexpected positional pipeline argument')
    else:
      blocks[active].append(token)
  return blocks, order


def value(blocks, name, default=None):
  values = blocks.get('--' + name)
  if values is None:
    return default
  if len(values) != 1:
    raise ValueError(f'--{name} requires one value')
  return values[0]


def validate_reference_settings(blocks, reference):
  settings = reference['settings']
  expected = {
    'steps': ('diffusion_steps', 10, int),
    'bf16-vision-layers': ('bf16_vision_layers', 0, int),
    'pixel-scale': ('pixel_scale', 1., float),
    'history-pixel-scale': ('history_pixel_scale', 1., float),
    'history-feature-scale': ('history_feature_scale', 1., float),
    'trajectory-solver': ('trajectory_solver', 'euler', str),
  }
  for flag, (key, default, convert) in expected.items():
    if convert(value(blocks, flag, default)) != settings[key]:
      raise ValueError(f'Baseline --{flag} differs from verified reference setting {key}')
  if settings['precision'] != 'fp8' or settings['attention_and_kv'] != 'bfloat16':
    raise ValueError('Only the verified FP8/BF16-attention reference is supported')
  if ('--bf16-expert-projections' in blocks) != settings['bf16_expert_projections']:
    raise ValueError('Baseline expert projection precision differs from reference')
  if value(blocks, 'camera-profile', 'openpilot') != 'nvidia' or '--allow-missing-cameras' not in blocks:
    raise ValueError('Require complete two-front-camera NVIDIA-profile inputs')
  indices = value(blocks, 'front-history-indices')
  if indices is not None and [int(i) for i in indices.split(',')] != settings['front_history_indices']:
    raise ValueError('Baseline history-frame selection differs from reference')
  if '--camera-pixel-scales' in blocks and any(float(x) != 1 for x in blocks['--camera-pixel-scales']):
    raise ValueError('Reduced or altered per-camera resolution is incompatible')
  if float(value(blocks, 'narrow-crop', 1)) != 1 or float(value(blocks, 'wide-equidistant-focal', 0)) != 0:
    raise ValueError('Additional camera cropping/warping is incompatible with the frozen inputs')
  for flag in ('--fused-qkv', '--row-quantize', '--validation-reference', '--warmup-graphs', '--warm-current-shapes'):
    if flag in blocks:
      raise ValueError(f'Baseline {flag} is incompatible with this bounded runner')
  if '--packed-quantize' not in blocks:
    raise ValueError('This runner requires the recorded packed FP8 quantization wrapper')
  calibration = reference['calibration_files']
  for flag, key in (('prefill-calibration', 'language_fp8_activation_scales'),
                    ('vision-calibration', 'vision_fp8_activation_scales')):
    path = value(blocks, flag)
    expected_path = settings.get(key)
    if not expected_path:
      if path is not None:
        raise ValueError(f'Baseline --{flag} silently changes verified dynamic FP8 scales')
      continue
    if path is None:
      raise ValueError(f'Baseline lacks verified fixed --{flag}; refusing to overwrite its scale policy')
    path = str(Path(path).resolve())
    matches = [name for name in calibration if name == expected_path or name.endswith('/' + expected_path)]
    if len(matches) != 1 or path != matches[0] or ptq.sha_file(path) != calibration[path]:
      raise ValueError(f'Baseline --{flag} bytes differ from verified calibration')
    scales = json.loads(Path(path).read_text())['scales']
    if not scales or not all(np.isfinite(s) and s > 0 for s in scales.values()):
      raise ValueError(f'Baseline --{flag} has invalid fixed scales')
    if '--capture-' + ('prefill' if flag == 'prefill-calibration' else 'vision') + '-scales' in blocks:
      raise ValueError(f'Baseline cannot capture scales while using fixed --{flag}')
  if not value(blocks, 'initial-noise'):
    raise ValueError('Paired controls require the recorded initial-noise file')


def check_kernel_gate(path, mode, sources):
  if path is None:
    raise ValueError('Candidate requires --kernel-report from the parent scaled-kernel GPU gate')
  report = json.loads(Path(path).read_text())
  if report.get('cpu_only') is not False or not report.get('complete') or not report.get('all_pass'):
    raise ValueError('Scaled-kernel report is not a completed passing GPU gate')
  selected = report.get('modes', {}).get(mode)
  if not selected or not selected.get('complete') or not selected.get('all_pass') or not selected.get('activation_tail_exact'):
    raise ValueError(f'Scaled-kernel GPU gate does not qualify mode {mode}')
  controls = selected.get('small_controls', [])
  for name in ('original', 'activation-outlier', 'weight-tail'):
    rows = [r for r in controls if r.get('name') == name and r.get('shape') == [2112, 2048, 9216]]
    if len(rows) != 1 or not rows[0].get('raw_all_exact') or not rows[0].get('scaled_bias_and_no_bias_all_exact'):
      raise ValueError(f'Scaled-kernel actual-shape gate lacks passing {name}')
  replay = selected.get('captured_mutations', [])
  if {r.get('change') for r in replay if r.get('all_bits_exact')} != {'original', 'activation', 'weight-scale-bias'}:
    raise ValueError('Scaled-kernel captured-mutation gate is incomplete')
  recorded = report.get('source_hashes', {})
  roots = {Path(name).resolve().parent for name in recorded if Path(name).name == 'agent-int4-scaled-kernel.py'}
  if len(roots) != 1:
    raise ValueError('Kernel gate must identify one source tree')
  recorded_root = roots.pop()
  active_hashes = {}
  for filename, expected in recorded.items():
    file = Path(filename).resolve()
    if file.parent not in (recorded_root, recorded_root.parent) or ptq.sha_file(file) != expected:
      raise ValueError(f'Kernel gate dependency changed: {filename}')
    # Frozen benchmark trees can live at different paths. Both the tested bytes
    # and the bytes this runner will import must match the recorded checksum.
    active = (EXP if file.parent == recorded_root else EXP.parent) / file.name
    if str(active) in active_hashes or ptq.sha_file(active) != expected:
      raise ValueError(f'Active kernel dependency differs from tested bytes: {active}')
    active_hashes[str(active)] = expected
    sources[str(active)] = expected
  for source in ('agent-int4-scaled-kernel.py', 'agent-int4k32-kernel.py', 'wgp_compiler.py'):
    file = EXP / source
    if active_hashes.get(str(file)) != sources[str(file)]:
      raise ValueError(f'Kernel gate source changed: {source}; rerun the parent GPU gate')
  return dict(path=str(Path(path).resolve()), sha256=ptq.sha_file(path), mode=mode,
              tested_sources=recorded, active_sources=active_hashes)


def controls_path(output):
  return Path(output).with_suffix('.int4model-controls.json')


def numerical_protocol(blocks, environment, checkpoint, source_hashes):
  skip = {'--output', '--samples', '--warmup', '--runs'}
  selected = {key: values for key, values in blocks.items() if key not in skip}
  return dict(arguments=selected, environment={k: v for k, v in environment.items() if k != 'CACHEDB'},
              checkpoint=checkpoint['identity_sha256'], reference_sources=source_hashes)


def runtime_sources(environment):
  roots = [Path(entry).resolve() for entry in environment.get('PYTHONPATH', '').split(os.pathsep)
           if entry and (Path(entry) / 'tinygrad' / '__init__.py').is_file()]
  if len(set(roots)) != 1:
    raise ValueError('Recorded PYTHONPATH must identify one explicit tinygrad source checkout')
  root = roots[0]
  files = {str(file.relative_to(root)): ptq.sha_file(file) for file in sorted((root / 'tinygrad').rglob('*.py'))}
  revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                             capture_output=True, text=True).stdout.strip()
  return dict(tinygrad_path=str(root), tinygrad_revision=revision,
              tinygrad_python_files=len(files), tinygrad_python_tree_sha256=hashlib.sha256(ptq.canonical(files)).hexdigest(),
              python=sys.version, executable=sys.executable,
              packages={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'transformers')})


def preflight(args):
  """No pipeline, torch, tinygrad, kernel compiler, or device imports here."""
  saved = json.loads(args.invocation.read_text())
  command = saved['command']
  if Path(command[1]).resolve() != EXP / 'benchmark-pipeline.py':
    raise ValueError('Only the adjacent benchmark-pipeline.py invocation is supported')
  blocks, order = option_blocks(command)
  source_reference = json.loads(args.reference_provenance.read_text())
  reference = source_reference['reference']
  body = {k: v for k, v in reference.items() if k != 'identity_sha256'}
  if hashlib.sha256(ptq.canonical(body)).hexdigest() != reference['identity_sha256']:
    raise ValueError('Reference provenance identity mismatch')
  validate_reference_settings(blocks, reference)
  if args.mode == 'collect' and '--native-manifest' in blocks:
    raise ValueError('Collect cannot bind native image replacement to the frozen sample-image inventory; use the recorded prepared-file quality invocation')
  reference_roots = {Path(name).resolve().parent for name in reference['sources'] if Path(name).name == 'agent-int4model.py'}
  if len(reference_roots) != 1:
    raise ValueError('Reference provenance must identify one source tree')
  reference_root = reference_roots.pop()
  active_reference_sources = {}
  for filename, expected in reference['sources'].items():
    if ptq.sha_file(filename) != expected:
      raise ValueError(f'Verified reference source changed: {filename}')
    original = Path(filename).resolve()
    if original.parent not in (reference_root, reference_root.parent):
      raise ValueError(f'Reference source outside its recorded tree: {filename}')
    active = (EXP if original.parent == reference_root else EXP.parent) / original.name
    if str(active) in active_reference_sources or ptq.sha_file(active) != expected:
      raise ValueError(f'Active reference source differs from verified bytes: {active}')
    active_reference_sources[str(active)] = expected
  checkpoint = Path(value(blocks, 'checkpoint')).resolve()
  if str(checkpoint) != reference['checkpoint']['path']:
    raise ValueError('Invocation checkpoint differs from the frozen reference')
  for filename, expected in reference['checkpoint']['files'].items():
    path = checkpoint / filename
    if path.stat().st_size != expected['bytes'] or ptq.sha_file(path) != expected['sha256']:
      raise ValueError(f'Checkpoint bytes differ from frozen reference: {filename}')
  cache = Path(saved['environment'].get('ALPAMAYO_FP8_CACHE', '')).resolve()
  expected_cache = reference['fp8_cache']
  if str(cache) != expected_cache['path'] or ptq.sha_file(cache / 'index.json') != expected_cache['index_sha256']:
    raise ValueError('Invocation FP8 cache index/path differs from reference')
  index = json.loads((cache / 'index.json').read_text())
  if ptq.sha_file(cache / 'weights.bin') != expected_cache['weights_sha256'] or index['sha256'] != expected_cache['weights_sha256']:
    raise ValueError('Known on-disk FP8 cache weights differ from reference')
  manifest, cpu_weights = ptq.load_bundle(args.bundle, reference['checkpoint']['identity_sha256'])
  if manifest['exporter_sha256'] != ptq.sha_file(EXP / 'agent-int4model.py'):
    raise ValueError('Reviewed bundle/exporter source differs; frozen semantics must be retained')
  if len(manifest['records']) != 28 or {int(ptq.FC2.fullmatch(r['name'])[1]) for r in manifest['records']} != set(range(28)):
    raise ValueError('Initial candidate requires exactly the reviewed 28 fc2 layers')
  if args.mode == 'candidate' and value(blocks, 'pad-prompt-to') != '2112':
    raise ValueError('Initial candidate requires the GPU-gated 2112-row padded prefill')
  noise = np.load(value(blocks, 'initial-noise'), allow_pickle=False)
  if noise.dtype != np.float32 or noise.shape != (1, 64, 2) or not np.isfinite(noise).all():
    raise ValueError('Invalid paired initial noise')
  selection = ptq.CalibrationSelection(args.calibration_samples, args.reserved_samples)
  if (len(selection.allowed) != 32
      or selection.provenance()['calibration_list_sha256'] != source_reference['selection']['calibration_list_sha256']
      or selection.provenance()['reserved_list_sha256'] != source_reference['selection']['reserved_list_sha256']):
    raise ValueError('Use exactly the frozen calibration-samples.json 32 and reserved path manifest')
  before = {k: list(v) for k, v in blocks.items()}
  if args.mode == 'collect':
    if args.warmup not in (None, 0) or args.runs not in (None, 32) or args.samples_json is not None:
      raise ValueError('Collect fixes exactly the 32 allowed samples, zero warmups, and 32 runs')
    if not {'--direct-byte-inputs', '--compact-inputs', '--byte-pixels'}.issubset(blocks):
      raise ValueError('Bounded prepared-array collection requires the recorded direct uint8 pixel inputs')
    blocks['--samples'], blocks['--warmup'], blocks['--runs'] = list(selection.allowed), ['0'], ['32']
    expected_inputs = {row['path']: row for row in source_reference['calibration_inputs']}
    if len(expected_inputs) != 32:
      raise ValueError('Frozen reference has an incomplete calibration input inventory')
    for sample in selection.allowed:
      if selection.identity(sample) != expected_inputs[sample]:
        raise ValueError(f'Calibration input changed since provenance freeze: {sample}')
    prepared_bound = 32 * (10 * 196608 * 3 + 2112 * 28 + 65536)
    collector_bound = 32 * 28 * (32 * 9216 * 2 + 4 * 9216 * 4 + 32 * 4 + 4096)
    if prepared_bound + collector_bound >= reference['settings']['collector_budget_bytes']:
      raise ValueError('Prepared-array plus activation bounds exceed collection budget')
    if args.capture_dir.exists():
      raise FileExistsError('Use a new --capture-dir')
  else:
    if args.samples_json is not None:
      samples = json.loads(args.samples_json.read_text())
      if not samples or not all(isinstance(s, str) for s in samples):
        raise ValueError('Invalid explicit sample list')
      blocks['--samples'] = samples
    for name, setting in (('warmup', args.warmup), ('runs', args.runs)):
      if setting is not None:
        if setting < (0 if name == 'warmup' else 1):
          raise ValueError('Invalid bounded run count')
        blocks['--' + name] = [str(setting)]
  blocks['--output'] = [str(args.output.resolve())]
  for name in blocks:
    if name not in order:
      order.append(name)
  resolved = [command[1], *[item for name in order for item in (name, *blocks[name])]]
  samples = blocks['--samples']
  warmup, runs = int(value(blocks, 'warmup', 2)), int(value(blocks, 'runs', 6))
  if not samples or warmup < 0 or runs <= 0:
    raise ValueError('Invalid prediction schedule')
  schedule = [str(Path(samples[i % len(samples)]).resolve()) for i in range(warmup + runs)]
  native = value(blocks, 'native-manifest')
  if native:
    native_rows = json.loads(Path(native).read_text())['schedule']
    names = {Path(row['sample']).name for row in native_rows}
    missing = [s for s in schedule if Path(s).name not in names]
    if missing:
      raise ValueError(f'Native baseline manifest does not cover {len(missing)} selected queries; use a compatible recorded quality invocation')
  files = set(active_reference_sources)
  files.update(str(EXP / name) for name in ('benchmark-pipeline.py', 'block-decode.py', 'draft-decode.py', 'lookup-decode.py',
               'agent-int4-scaled-kernel.py', 'agent-int4k32-kernel.py', 'wgp_compiler.py', 'agent-int4model-run.py'))
  files.add(str(EXP.parent / 'inputs.py'))
  if native:
    files.update(str(EXP / name) for name in ('stream-inputs.py', 'native_warp.py', 'native-warp.c', 'calibration_trials.py'))
  sources = {name: ptq.sha_file(name) for name in sorted(files)}
  gate = check_kernel_gate(args.kernel_report, args.kernel_mode, sources) if args.mode == 'candidate' else None
  baseline_sources = {name: digest for name, digest in sources.items() if Path(name).name not in
                      ('agent-int4-scaled-kernel.py', 'agent-int4-scaled-check.py', 'agent-int4k32-kernel.py')}
  protocol = numerical_protocol(blocks, saved['environment'], reference['checkpoint'], baseline_sources)
  protocol['runtime_sources'] = runtime_sources(saved['environment'])
  protocol['auxiliary_input_files'] = {flag: ptq.sha_file(value(blocks, flag)) for flag in
    ('initial-noise', 'reduced-vocab', 'native-manifest', 'native-calibrations') if value(blocks, flag) is not None}
  paired = None
  if args.compare_reference:
    paired = json.loads(controls_path(args.compare_reference).read_text())
    if paired.get('schema') != SCHEMA or not paired.get('complete') or paired.get('mode') != 'reference':
      raise ValueError('Paired controls must come from a completed reference runner')
    if paired['protocol'] != protocol or [r['sample'] for r in paired['queries']] != schedule:
      raise ValueError('Paired reference numerical settings or exact query schedule differ')
  changes = {k: dict(recorded=before.get(k), resolved=blocks[k]) for k in blocks if before.get(k) != blocks[k]}
  plan = dict(schema=SCHEMA, mode=args.mode, cpu_preflight_complete=True, gpu_started=False,
    recorded_invocation_sha256=ptq.sha_file(args.invocation), resolved_command=resolved,
    explicit_schedule_and_output_changes=changes, environment=saved['environment'], sources=sources,
    checkpoint=reference['checkpoint']['identity_sha256'], bundle=manifest['identity_sha256'],
    bundle_manifest_sha256=ptq.sha_file(args.bundle / 'manifest.json'), reference=reference['identity_sha256'],
    reference_file_sha256=ptq.sha_file(args.reference_provenance), kernel_gate=gate,
    initial_noise_sha256=ptq.sha_file(value(blocks, 'initial-noise')), protocol=protocol, schedule=schedule,
    warmup=warmup, runs=runs, collection_profiling=args.mode == 'collect',
    profiling=bool(args.mode == 'collect' or {'--profile-kernels', '--profile-host', '--capture-prefill-scales', '--capture-vision-scales'} & blocks.keys()),
    timing_boundary='Original pipeline loop includes runner CPU provenance/finite-output checks; model timing excludes those host checks.',
    route_is_blind_test=False, quality_passed=None)
  plan['timings_valid_for_throughput'] = not plan['profiling']
  return SimpleNamespace(plan=plan, reference=reference, selection=selection, manifest=manifest, weights=cpu_weights,
                         cache=cache, cache_index=index, paired=paired, blocks=blocks)


def verify_loaded_records(index, cache_file, read_record, chunk_bytes=4 * 1024**2):
  """Diagnostic only. read_record(record,start,count) returns actual used tensor bytes."""
  if not 1 <= chunk_bytes <= 16 * 1024**2:
    raise ValueError('Readback chunks must be bounded to at most 16 MiB')
  rows, total = [], 0
  with Path(cache_file).open('rb') as stream:
    for record in index['records']:
      observed, expected = hashlib.sha256(), hashlib.sha256()
      for start in range(0, record['nbytes'], chunk_bytes):
        count = min(chunk_bytes, record['nbytes'] - start)
        stream.seek(record['offset'] + start)
        wanted = stream.read(count)
        actual = bytes(read_record(record, start, count))
        if len(actual) != count or actual != wanted:
          raise ValueError(f'Loaded original tensor mismatch: {record["name"]}/{record["kind"]} at byte {start}')
        observed.update(actual)
        expected.update(wanted)
        total += count
      rows.append(dict(name=record['name'], kind=record['kind'], packed=record['packed'],
                       bytes=record['nbytes'], actual_sha256=observed.hexdigest(), expected_sha256=expected.hexdigest()))
  return dict(complete=True, all_bytes_equal=True, tensor_bytes=total, chunk_bytes=chunk_bytes, records=rows,
              scope='Every actually bound original FP8/BF16 tensor and float32 scale before precision restoration/vocabulary reduction')


def gpu_record_reader(model):
  """Created only by parent collect execution, immediately after bulk model loading."""
  from tinygrad import dtypes
  from amd_gemm import pack
  current = dict(key=None, raw=None)

  def read(record, start, count):
    key = record['name'], record['kind']
    if current['key'] != key:
      tensor = (model.weight_scales if record['kind'] == 'scale' else model.weights)[record['name']]
      dtype = {'scale': dtypes.float32, 'fp8': dtypes.fp8e4m3, 'bf16': dtypes.bfloat16}[record['kind']]
      if tensor.dtype != dtype or list(tensor.shape) != record['shape']:
        raise ValueError(f'Loaded tensor shape/dtype differs: {key}')
      # Invert the cache loader's logical unpacked FP8 view before byte comparison.
      if record['packed']:
        tensor = pack(tensor).realize()
      current.update(key=key, raw=tensor.bitcast(dtypes.uint8).reshape(-1))
    return current['raw'][start:start + count].contiguous().realize().numpy().tobytes()

  return read


def cpu_array(value):
  if isinstance(value, np.ndarray):
    return value
  if hasattr(value, 'detach'):
    value = value.detach().cpu()
  try:
    return np.asarray(value.numpy())
  except (AttributeError, TypeError):
    if hasattr(value, 'float'):
      return value.float().numpy()
    return np.asarray(value)


def source_metadata(data):
  cameras, times = cpu_array(data['camera_indices']), cpu_array(data['absolute_timestamps']).reshape(-1)
  order = np.lexsort((times, cameras))
  return dict(camera_indices=cameras[order].copy(), absolute_timestamps=times[order].copy(),
              nav_text=np.asarray(data['nav_text']))


def prepared_arrays(prepared, packed):
  if '_agent_int4model_source' not in prepared:
    raise ValueError('Actual prepared CPU input metadata hook did not execute')
  meta = prepared['_agent_int4model_source']
  return dict(input_ids=np.asarray(packed[0], dtype=np.int32), position_ids=np.asarray(packed[1]),
              pixel_values=cpu_array(prepared['tokenized_data']['pixel_values']),
              ego_history_xyz=cpu_array(prepared['ego_history_xyz']), ego_history_rot=cpu_array(prepared['ego_history_rot']),
              **meta)


def finite_prediction(prediction):
  xyz, rot = np.asarray(prediction['xyz']), np.asarray(prediction['rot'])
  tokens = prediction['generated_token_ids']
  if xyz.shape != (1, 64, 3) or rot.shape != (1, 64, 3, 3) or not np.isfinite(xyz).all() or not np.isfinite(rot).all():
    raise ValueError('Invalid or nonfinite prediction arrays')
  if not tokens or any(type(t) is not int for t in tokens) or prediction['generated_tokens'] != len(tokens):
    raise ValueError('Invalid verified reasoning token serialization')
  ptq.canonical(prediction)  # Also rejects NaN/Inf in any serialized scalar/timing.


def compare_control(before, after):
  if before['sample'] != after['sample'] or before['prepared']['sha256'] != after['prepared']['sha256']:
    raise ValueError('Paired control used different actual prepared input arrays')
  a, b = before['prediction'], after['prediction']
  finite_prediction(a)
  finite_prediction(b)
  delta = np.abs(np.asarray(a['xyz']) - np.asarray(b['xyz']))
  return dict(sample=after['sample'], tokens_equal=a['generated_token_ids'] == b['generated_token_ids'],
              xyz_mean_delta_m=float(delta.mean()), xyz_max_delta_m=float(delta.max()),
              rotations_max_delta=float(np.abs(np.asarray(a['rot']) - np.asarray(b['rot'])).max()), quality_passed=None)


def install_candidate_guards(model, manifest, weights, quantizer, kernel, stack, counts):
  """Install outside packed FP8 and JIT installers. Prove no duplicate fc2 FP8 quantization."""
  state = dict(prefill=False, name=None)
  original_quantize = model.quantize

  def guarded_quantize(this, x):
    if state['prefill'] and state['name'] in weights:
      raise ValueError('Candidate fc2 entered the original FP8 quantizer: duplicate quantization')
    return original_quantize(x)

  def unsigned(x):
    name = state['name']
    if not state['prefill'] or name not in weights:
      raise ValueError('A4 quantizer escaped language prefill fc2')
    counts[name] = counts.get(name, 0) + 1
    return quantizer(x)

  uninstall = ptq.install_fc2(model, manifest, weights, unsigned, kernel)
  stack.callback(uninstall)
  original_linear, original_layer = model.linear, model.text_layer

  def linear(this, x, name, quantized_input=None, heads=None):
    previous, state['name'] = state['name'], name
    try:
      return original_linear(x, name, quantized_input, heads)
    finally:
      state['name'] = previous

  def layer(this, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    previous = state['prefill']
    state['prefill'] = ptq.prefill_context(prefix, cache_pos, math.prod(x.shape[:-1]))
    try:
      return original_layer(x, cos, sin, prefix, cache, mask, cache_pos)
    finally:
      state['prefill'] = previous

  stack.enter_context(patch.object(model, 'linear', MethodType(linear, model)))
  stack.enter_context(patch.object(model, 'text_layer', MethodType(layer, model)))
  stack.enter_context(patch.object(model, 'quantize', MethodType(guarded_quantize, model)))


class RunnerHooks:
  def __init__(self, args, verified, stack):
    self.args, self.v, self.stack = args, verified, stack
    self.model, self.collector = None, None
    self.loaded_verification, self.index = None, 0
    self.packed_models, self.padded_models = [], []
    self.queries, self.comparisons, self.counts, self.loaded_sources = [], [], {}, {}
    self.prepared_bytes = 0
    self.restore_expectations = {}

  def attach(self, model):
    if self.model is not None:
      if self.model is not model:
        raise ValueError('Runner supports one model allocation')
      return
    if model not in self.packed_models:
      raise ValueError('Decoder initialized before packed FP8 installation')
    if '--pad-prompt-to' in self.v.blocks and model not in self.padded_models:
      raise ValueError('Decoder initialized before padded-prefill installation')
    self.model = model
    self.restore_expectations = {name: getattr(model, name) for name in ('linear', 'text_layer', 'quantize', 'prefill_jit')}
    if self.args.mode == 'collect':
      if not self.loaded_verification or not self.loaded_verification['complete']:
        raise ValueError('Original loaded weight verification did not complete before collection')
      self.collector = ptq.Collector(self.v.selection, self.args.capture_dir, self.v.reference)
      ptq.json_write(self.args.capture_dir / 'runner-loaded-weights.json', dict(
        reference_identity=self.v.reference['identity_sha256'], cache=self.v.reference['fp8_cache'],
        verification=self.loaded_verification))
    elif self.args.mode == 'candidate':
      source = EXP / 'agent-int4-scaled-kernel.py'
      if ptq.sha_file(source) != self.v.plan['sources'][str(source)]:
        raise ValueError('Scaled kernel changed after its preflight gate')
      scaled = local('agent-int4-scaled-kernel')
      kernel_mode = self.args.kernel_mode
      module = SimpleNamespace(quantize_pack=partial(scaled.quantize_pack, wgp=kernel_mode == 'wgp'),
                               gemm=partial(scaled.gemm, wgp=kernel_mode == 'wgp'))
      from tinygrad import Tensor, Device
      upload = lambda a: Tensor(np.asarray(a).copy(), device='CPU').to(Device.DEFAULT).realize()
      runtime, quantizer, kernel = ptq.scaled_module_adapter(module, self.v.weights, upload)
      install_candidate_guards(model, self.v.manifest, runtime, quantizer, kernel, self.stack, self.counts)
      model.synchronize()  # All one-time weights/scales are ready before timing.

  def prediction(self, original, decoder, model, settings, config, tokenizer, prepared, packed, seed, inputs):
    if self.index >= len(self.v.plan['schedule']):
      raise ValueError('Pipeline executed an unexpected additional query')
    sample = self.v.plan['schedule'][self.index]
    arrays = prepared_arrays(prepared, packed)
    valid = int(prepared.get('actual_prompt_length', arrays['input_ids'].shape[1]))
    if not 128 <= valid <= arrays['input_ids'].shape[1] or settings.diffusion_steps != 3:
      raise ValueError('Actual prompt length or Euler count differs from the verified pipeline')
    record = dict(sample=sample, phase='warmup' if self.index < self.v.plan['warmup'] else 'run',
                  index=self.index, valid_rows=valid, prepared=ptq.prepared_identity(arrays))
    if self.args.mode == 'collect':
      if arrays['pixel_values'].dtype != np.uint8:
        raise ValueError('Prepared pixel representation differs from bounded direct-byte collection')
      estimate = sum(a.nbytes for a in arrays.values()) + 8192
      if self.collector.manifest['worst_payload_bytes'] + self.prepared_bytes + estimate >= self.collector.budget_bytes:
        raise ValueError('Combined activation/prepared-array collection budget exhausted')
      filename = f'prepared-{self.index:02d}.npz'
      file = self.args.capture_dir / filename
      with file.open('xb') as stream:
        np.savez(stream, **arrays, valid_rows=np.array(valid, np.int32))
      self.prepared_bytes += file.stat().st_size
      record['prepared_archive'] = dict(file=filename, sha256=ptq.sha_file(file), bytes=file.stat().st_size)
      with self.collector.sample(model, sample, valid, arrays):
        result = original(decoder, model, settings, config, tokenizer, prepared, packed, seed, inputs)
    else:
      result = original(decoder, model, settings, config, tokenizer, prepared, packed, seed, inputs)
    finite_prediction(result)
    record['prediction'] = result
    if self.v.paired is not None:
      self.comparisons.append(compare_control(self.v.paired['queries'][self.index], record))
    self.queries.append(record)
    self.index += 1
    return result

  def patch_module(self, module, path):
    self.loaded_sources[str(path)] = ptq.sha_file(path)
    if path.name == 'benchmark-fast.py':
      # This module mutates EdgeModel.__init__ while executing. Restore that original too.
      self.stack.callback(setattr, module.EdgeModel, '__init__', module.original_init)
      if self.args.mode == 'collect':
        original = module.EdgeModel.__init__
        @wraps(original)
        def init(model, *args, **kwargs):
          original(model, *args, **kwargs)
          if self.loaded_verification is not None:
            raise ValueError('Unexpected repeated original model load')
          self.loaded_verification = verify_loaded_records(self.v.cache_index, self.v.cache / 'weights.bin', gpu_record_reader(model))
        self.stack.enter_context(patch.object(module.EdgeModel, '__init__', init))
    elif path.name == 'fast-inputs.py':
      original = module.FastInputs.prepare
      @wraps(original)
      def prepare(adapter, data):
        config, tokenizer, prepared = original(adapter, data)
        prepared = dict(prepared, _agent_int4model_source=source_metadata(data))
        return config, tokenizer, prepared
      self.stack.enter_context(patch.object(module.FastInputs, 'prepare', prepare))
    elif path.name in ('quantize-packed.py', 'padded-prefill.py'):
      original = module.install
      @wraps(original)
      def install(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        (self.packed_models if path.name == 'quantize-packed.py' else self.padded_models).append(model)
        return result
      self.stack.enter_context(patch.object(module, 'install', install))
    elif path.name == 'block-decode.py':
      original_init, original_run = module.BlockDecoder.__init__, module.BlockDecoder.run
      @wraps(original_init)
      def init(decoder, model, *args, **kwargs):
        original_init(decoder, model, *args, **kwargs)
        self.attach(model)
      @wraps(original_run)
      def run(decoder, model, settings, config, tokenizer, prepared, packed, seed, inputs):
        return self.prediction(original_run, decoder, model, settings, config, tokenizer, prepared, packed, seed, inputs)
      self.stack.enter_context(patch.object(module.BlockDecoder, '__init__', init))
      self.stack.enter_context(patch.object(module.BlockDecoder, 'run', run))

  def finish(self):
    if self.index != len(self.v.plan['schedule']):
      raise ValueError('Pipeline did not execute the complete selected query schedule')
    if self.args.mode == 'collect':
      completed = self.collector.finish()
      if len(completed['samples']) != 32 or sum(len(s['layers']) for s in completed['samples']) != 32 * 28:
        raise ValueError('Collector did not complete exactly 32x28 captures')
      total = sum(p.stat().st_size for p in self.args.capture_dir.iterdir() if p.is_file())
      if total >= self.collector.budget_bytes:
        raise ValueError('Combined prepared arrays, activations, and metadata exceed collection budget')
    if self.args.mode == 'candidate' and set(self.counts) != {r['name'] for r in self.v.manifest['records']}:
      raise ValueError('Candidate did not trace every intended language prefill fc2')


def execute(args, verified):
  args.output.parent.mkdir(parents=True, exist_ok=True)
  result = dict(verified.plan, complete=False, hooks_restored=False)
  hooks = None
  prior_spec, prior_argv, prior_environment, prior_path = importlib.util.spec_from_file_location, sys.argv, dict(os.environ), list(sys.path)
  try:
    with ExitStack() as stack:
      old_environment, old_path = dict(os.environ), list(sys.path)
      def restore_environment():
        os.environ.clear()
        os.environ.update(old_environment)
        sys.path[:] = old_path
      stack.callback(restore_environment)
      for key, setting in verified.plan['environment'].items():
        if setting is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = setting
      for path in reversed(os.environ.get('PYTHONPATH', '').split(os.pathsep)):
        if path:
          sys.path.insert(0, path)
      sys.path.insert(0, str(EXP))
      hooks = RunnerHooks(args, verified, stack)
      original_spec = importlib.util.spec_from_file_location
      targeted = {'benchmark-fast.py', 'fast-inputs.py', 'block-decode.py', 'quantize-packed.py', 'padded-prefill.py'}
      def specification(name, location, *values, **options):
        spec = original_spec(name, location, *values, **options)
        if location is None:
          return spec
        path = Path(location).resolve()
        if path.parent != EXP or path.name not in targeted:
          return spec
        expected = verified.plan['sources'].get(str(path))
        if expected is not None and ptq.sha_file(path) != expected:
          raise ValueError(f'Pipeline source changed after preflight: {path.name}')
        loader = spec.loader
        class Loader:
          def create_module(self, value):
            return loader.create_module(value)
          def exec_module(self, value):
            loader.exec_module(value)
            hooks.patch_module(value, path)
        spec.loader = Loader()
        return spec
      stack.enter_context(patch.object(importlib.util, 'spec_from_file_location', specification))
      stack.enter_context(patch.object(sys, 'argv', verified.plan['resolved_command']))
      pipeline = local('benchmark-pipeline')
      result['gpu_started'] = True
      pipeline.main()
      hooks.finish()
      result['complete'] = True
  except BaseException as error:
    result['complete'] = False
    result['error'] = dict(type=type(error).__name__, message=str(error))
    raise
  finally:
    result['hooks_restored'] = (importlib.util.spec_from_file_location is prior_spec and sys.argv is prior_argv
      and dict(os.environ) == prior_environment and sys.path == prior_path
      and (hooks is None or hooks.model is None or (
        not getattr(hooks.model, 'agent_int4model_hook', False)
        and all(getattr(hooks.model, name) == expected for name, expected in hooks.restore_expectations.items()))))
    if hooks is not None:
      result.update(actual_loaded_weight_verification=hooks.loaded_verification,
                    candidate_python_trace_calls=hooks.counts, loaded_sources=hooks.loaded_sources,
                    prepared_archive_bytes=hooks.prepared_bytes, comparisons=hooks.comparisons,
                    collector_complete=bool(hooks.collector and hooks.collector.manifest['complete']))
      controls = dict(schema=SCHEMA, mode=args.mode, complete=result['complete'], protocol=verified.plan['protocol'], queries=hooks.queries)
      ptq.json_write(controls_path(args.output), controls)
    if args.output.exists():
      report = json.loads(args.output.read_text())
      report.setdefault('summary', {})['profiling_enabled'] = bool(report.get('summary', {}).get('profiling_enabled') or args.mode == 'collect')
      report['int4model_runner'] = {k: v for k, v in result.items() if k not in ('comparisons', 'actual_loaded_weight_verification')}
      ptq.json_write(args.output, report)
    ptq.json_write(args.output.with_suffix('.int4model.json'), result)


def parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode', choices=('reference', 'collect', 'candidate'), required=True)
  p.add_argument('--invocation', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--reference-provenance', type=Path, default=EXP / 'agent-int4model-artifacts/reference-provenance.json')
  p.add_argument('--bundle', type=Path, default=EXP / 'agent-int4model-artifacts/peroutput-bundle')
  p.add_argument('--calibration-samples', type=Path, default=WORK / 'calibration-samples.json')
  p.add_argument('--reserved-samples', type=Path, default=WORK / 'holdout-samples.json')
  p.add_argument('--capture-dir', type=Path)
  p.add_argument('--samples-json', type=Path)
  p.add_argument('--warmup', type=int)
  p.add_argument('--runs', type=int)
  p.add_argument('--kernel-report', type=Path)
  p.add_argument('--kernel-mode', choices=('cu', 'wgp'), default='cu')
  p.add_argument('--compare-reference', type=Path)
  p.add_argument('--preflight', action='store_true', help='Validate and serialize the resolved plan on CPU; never import the pipeline')
  return p


def main():
  args = parser().parse_args()
  args.output = args.output.resolve()
  args.capture_dir = (args.capture_dir or args.output.with_suffix('.calibration')).resolve()
  if any(path.exists() for path in (args.output, controls_path(args.output), args.output.with_suffix('.int4model.json'))):
    raise FileExistsError('Use a new output path for this runner')
  verified = preflight(args)
  if args.preflight:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ptq.json_write(args.output, verified.plan)
    print(json.dumps(dict(cpu_only=True, mode=args.mode, preflight_passed=True, queries=len(verified.plan['schedule']), output=str(args.output)), indent=2))
    return
  execute(args, verified)


if __name__ == '__main__':
  main()
