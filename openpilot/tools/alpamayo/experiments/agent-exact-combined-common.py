"""Frozen-source gates and strict paired comparisons for the composed experiment."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
HERE = Path(__file__).resolve().parent
MLP = WORK/'agent-fused-mlp-2048-protocol-v2/alpamayo'
VISION = WORK/'vision-tiles-complete-source/alpamayo'
SAMPLER = WORK/'warp-fast-pathfix-source'
PYTHON = '/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
TINYGRAD = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def sha(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def read(path): return json.loads(Path(path).read_text())


def write_new(path, value):
  with Path(path).open('x') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')


def option(command, name):
  flag = '--'+name
  if command.count(flag) != 1: raise ValueError(f'Expected one {flag}')
  return command[command.index(flag)+1]


def equal(a, b, path='value'):
  if type(a) is not type(b): raise ValueError(f'Type differs: {path}')
  if isinstance(a, dict):
    if a.keys() != b.keys(): raise ValueError(f'Fields differ: {path}')
    for key in a: equal(a[key], b[key], path+'.'+key)
  elif isinstance(a, list):
    if len(a) != len(b): raise ValueError(f'Length differs: {path}')
    for i, (x, y) in enumerate(zip(a, b, strict=True)): equal(x, y, f'{path}[{i}]')
  elif isinstance(a, float):
    if not math.isfinite(a) or not math.isfinite(b) or struct.pack('<d', a) != struct.pack('<d', b):
      raise ValueError(f'Float bits differ: {path}')
  elif a != b: raise ValueError(f'Value differs: {path}')


TIMING_FIELDS = {'load_ms', 'preprocess_ms', 'prepared_queue_ms', 'cpu_buffers_ms', 'pipeline_ms', 'service_ms',
                 'gc_ms', 'native_calibration_ms'}
ROW_FIELDS = {'sample', 'prediction', 'allocated_bytes_by_device', 'vision_reference_checks', 'input_latest_timestamp_us',
               'vision_hits', 'vision_misses', 'uploaded_pixel_bytes', 'draft', 'calibrated_frame_hits', 'calibrated_frame_misses'}
PAIRED_SETTINGS = (
  'prefill_calibration_sha256', 'pad_prompt_to', 'trajectory_solver', 'capture_prefill_scales',
  'capture_vision_scales', 'vision_calibration_sha256', 'packed_bf16_vision', 'camera_profile',
  'vision_cache_capacity', 'mmap_inputs', 'compact_inputs', 'selective_pixels', 'pipeline_inputs',
  'late_inputs', 'first_measured_input_after_startup', 'prefill_kernels', 'cache_resize',
  'draft_previous', 'draft_attention', 'draft_prefix', 'draft_history', 'history_full_only',
  'draft_tile', 'lookup_draft', 'phrase_drafts', 'compact_draft_io', 'vision_cache', 'group_vision',
  'vision_group_limit', 'batched_vision', 'vision_views', 'verify_vision_cache', 'verify_vision_warmup',
  'pixel_scale', 'camera_pixel_scales', 'pixel_bounds', 'history_pixel_scale', 'history_feature_scale',
  'direct_byte_inputs', 'front_history_indices', 'allow_missing_cameras', 'minimum_cache_capacity',
  'initial_noise_sha256', 'prefill_speculation')


def compare(reference, candidate, stage):
  expected = (2, 62) if stage == 'quality' else (12, 24)
  compared = 0
  for group, count in zip(('warmup', 'runs'), expected, strict=True):
    a, b = reference[group], candidate[group]
    if len(a) != count or len(b) != count: raise ValueError(f'Incorrect {stage} {group} count')
    for i, (old, new) in enumerate(zip(a, b, strict=True)):
      if old.keys() != new.keys() or set(old)-TIMING_FIELDS-ROW_FIELDS:
        raise ValueError('Unknown or changed per-query fields require explicit review')
      for key in old:
        if key in TIMING_FIELDS: continue
        # Different captured kernels can retain different allocation totals.
        if key == 'allocated_bytes_by_device': continue
        left, right = old[key], new[key]
        if key == 'prediction':
          left, right = ({k:v for k,v in item.items() if k != 'timings_ms'} for item in (left, right))
        equal(left, right, f'{group}.{i}.{key}')
      compared += 1
  for key in PAIRED_SETTINGS:
    if key not in reference or key not in candidate: raise ValueError(f'Missing paired setting: {key}')
    equal(reference[key], candidate[key], key)
  if stage == 'native':
    timestamps = [row['input_latest_timestamp_us'] for row in candidate['warmup']+candidate['runs']]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
      raise ValueError('Native fixtures must advance chronologically')
  return dict(exact=True, predictions=compared, all_prediction_float_bits=True,
              history_inputs_cache_and_speculation_exact=True,
              allocation_totals_excluded='Captured kernel graphs can retain different allocation totals')


def evidence():
  mlp = load(MLP/'experiments/agent-fused-mlp-run.py', 'combined_original_mlp_gate')
  micro = mlp.micro_gate(WORK/'agent-fused-mlp-2048-micro/report.json', (2048,))
  vision = load(VISION/'experiments/agent-vision-tiles-transfer-check-coverage.py', 'combined_original_vision_gate')
  paths = [WORK/name/'report.json' for name in
           ('agent-vision-tiles-kernels', 'agent-vision-tiles-b2fc2-dynamic', 'agent-vision-tiles-b2fc2-fixed')]
  coverage = vision.audit(paths)
  if not coverage['eligible_for_exact_route_pairs'] or len(coverage['completed_cases']) != 16:
    raise ValueError('All original vision micro gates must pass')
  current = vision.source_manifest(VISION/'experiments/agent-vision-tiles-check.py')
  equal(current, coverage['numeric_kernel_sources'], 'vision.current_source')
  for name, digest in current['files'].items():
    path = MLP/('amd_gemm.py' if name == 'amd_gemm.py' else 'experiments/'+name)
    if sha(path) != digest: raise ValueError(f'Canonical vision foundation differs: {name}')
  check = read(SAMPLER/'agent-warp-fast-validated-portable-v6.json')
  if not all(check[k]['exact'] for k in ('boundaries', 'route', 'invalidation')):
    raise ValueError('Sampler CPU qualification is incomplete')
  for name in ('agent-warp-fast-sampler.py', 'agent-warp-fast-kernel.c', 'agent-warp-fast-workers.c'):
    if sha(SAMPLER/name) != check['source_sha256'][name]: raise ValueError(f'Sampler source changed: {name}')
  for name in ('native_warp.py', 'native-warp.c', 'calibration_trials.py', 'stream-inputs.py', 'camera_profiles.py'):
    if sha(MLP/'experiments'/name) != check['source_sha256'][name]: raise ValueError(f'Sampler foundation changed: {name}')
  return dict(mlp=micro, vision=coverage, sampler_check_sha256=sha(SAMPLER/'agent-warp-fast-validated-portable-v6.json'))


def verify(manifest):
  if manifest['schema'] != 1: raise ValueError('Unsupported composition manifest')
  for path, expected in manifest['files'].items():
    if sha(path) != expected: raise ValueError(f'Frozen bytes changed: {path}')
  for item in manifest['relocations']:
    if sha(item['from']) != item['sha256'] or sha(item['to']) != item['sha256']:
      raise ValueError('Source relocation lost byte identity')
  equal(evidence(), manifest['evidence'], 'original_evidence')


def validate_report(report, stage):
  required = {'pad_prompt_to':2048, 'vision_group_limit':2, 'vision_calibration_sha256':None,
              'capture_vision_scales':False, 'history_pixel_scale':1.0, 'history_feature_scale':1.0,
              'trajectory_solver':'euler'}
  for name, value in required.items(): equal(report[name], value, name)
  spec = report['prefill_speculation']
  for name, value in dict(mode='candidate', prefill_attention='shuffle-fma', padded_prompt_bound=2048,
                           audit=False, draft=[124549,11135,3174,4286,1561,135084]).items(): equal(spec[name], value, name)
  expected = (2, 62) if stage == 'quality' else (12, 24)
  if tuple(len(report[k]) for k in ('warmup', 'runs')) != expected: raise ValueError('Unexpected prediction schedule')
