"""Private decoder GEMMs admitted by complete, source-bound per-variant evidence."""
from contextlib import contextmanager
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
LABELS = ('1x2', '1x1-accum')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def require(value, message):
  if not value: raise ValueError(message)


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def digest(value):
  require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Invalid SHA256')
  return value


def serial(proof, shape, bias):
  require(proof['calls'] == 40 and proof['ordered_execution_barriers'] == 39 and proof['actual_queue_builder'] is True,
          'Missing actual 40-call WAW proof')
  require(proof['output_input_slot'] == 5 and proof['output_byte_offset'] == 0
          and proof['output_span_bytes'] == shape[0]*shape[1]*2, 'Changed output storage span')
  require(proof['kernel_global_input_slots'] == [5, 0, 1, 2, 3]+([4] if bias else [])
          and proof['output_argument_indices'] == [0]
          and proof['input_argument_indices'] == list(range(1, 6 if bias else 5)), 'Changed queue argument mapping')


def variant(row, label, control):
  require('baseline_failure' not in row and row['baseline_label'] == '1x4', 'Baseline failed or changed')
  checks = row['checks']
  require(len(checks) == 6 and [r['state'] for r in checks] == list(range(6)), 'Missing mutation states')
  for state, check in enumerate(checks):
    require(check['independent_oracle'] is control and check['isolated_row_scale'] is (state == 2), 'Mutation/oracle contract changed')
    digest(check['baseline_sha256'])
  require(all(checks[i]['baseline_sha256'] != checks[i-1]['baseline_sha256'] for i in range(1, 6)), 'Mutation had no effect')
  v = row['variants'][label]
  require(v['status'] == ('passed_control' if control else 'passed') and v['failures'] == [], 'Variant did not fully pass')
  require(v['eligible_for_timing'] is (not control), 'Timing admission differs')
  stages = [('single_mutation', state) for state in range(6)]
  if not control: stages += [('batch_captured_state', 5), ('batch_mutated_replay', 0), ('post_timing', 0)]
  require([(r['stage'], r['state']) for r in v['checks']] == stages, 'Missing variant mutation/replay/postcheck')
  for check in v['checks']:
    require(check['exact_bf16'] is True and check['output_sha256'] == checks[check['state']]['baseline_sha256'], 'Variant output differs')
  compiled = row['compiled'][label]
  digest(compiled['source_sha256']); digest(compiled['binary_sha256'])
  require(compiled['k16_contract'] == row['compiled']['1x4']['k16_contract']
          and compiled['k16_contract']['sequential_k16_steps'] == row['shape_mnk'][2]//16, 'K16 arithmetic changed')
  # Small controls may be fully unrolled. Actual model shapes retain a running C.
  if not control:
    contract = compiled['lowered_accumulator_contract']
    require(contract['running_wmma_accumulator'] is True and contract['loop_scalar_fadd_count'] == 0
            and contract['llvm_wmma_calls'] > 0 and not any(contract['third_argument_is_zero_vector']), 'Accumulator arithmetic changed')
  serial(compiled['offline_serial_proof'], row['shape_mnk'], row['bias'])
  if not control:
    require(row['complete'] is True, 'Actual case incomplete')
    serial(row['serial_proofs'][label], row['shape_mnk'], row['bias'])
    require(row['serial_proofs'][label] == compiled['offline_serial_proof'], 'Offline and captured queues differ')
    t = row['timings']
    require(t['calls_per_batch'] == 40 and t['host_inclusive'] is True, 'Changed timing boundary')
    require(label in t['eligible_labels_before_timing'] and label in t['accepted_labels_after_timing'], 'Variant timing rejected')
    samples = t['pairs_batch_ms']
    require(len(samples) in (3, 6), 'Incomplete timing rounds')
    for sample in samples:
      require(sample['order'].count(label) == 1 and math.isfinite(sample[label]) and sample[label] > 0, 'Invalid measured sample')
    require(t['median_ms_per_call'][label] == statistics.median(r[label] for r in samples)/40, 'Timing median mismatch')


def validate(report, choices, provenance, environment, wgp=True):
  require(report['complete'] is True and report['gpu_executed'] is True and report['mode'] == 'gpu'
          and report['strict_equality_required'] is True, 'Completed strict GPU report required')
  require(report['provenance'] == provenance and report['wgp'] is wgp, 'Source/compiler provenance differs')
  for key in ('TC_OPT', 'LLVMOPT', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL'):
    require(report['environment'][key] == environment.get(key), f'Environment differs: {key}')
  controls = report['controls']
  expected = {(name, bias) for name in ('integer', 'integer-tail', 'sparse-outliers') for bias in (False, True)}
  require(len(controls) == 6 and {(r['case'], r['bias']) for r in controls} == expected, 'All six controls required')
  require(choices, 'Empty shape selection')
  selected = {}
  for key, label in choices.items():
    shape = tuple(map(int, key.split(',')))
    require(label in LABELS and len(shape) == 3 and shape[0] in (16, 32, 48, 64), 'Unsupported shape or variant')
    for row in controls:
      require(row['shape_mnk'] == ([16, 64, 64] if row['case'] == 'integer' else [17, 48, 32]), 'Changed control shape')
      for tested in ('1x4', label): variant(row, tested, True)
    rows = [r for r in report['cases'] if tuple(r['shape_mnk']) == shape and r['bias'] is False]
    require(len(rows) == 1, 'Exactly one actual case required for each selected shape')
    row = rows[0]
    identity = row['model_identity']
    require(identity['all_selected_fp8_finite'] is True and identity['shape_nk'] == list(shape[1:]), 'Actual finite model weights required')
    for field in ('selected_weight_bytes_sha256', 'scale_bytes_sha256', 'cache_index_sha256', 'declared_full_cache_sha256'):
      digest(identity[field])
    for tested in ('1x4', label): variant(row, tested, False)
    require(row['timings']['median_ms_per_call'][label] < row['timings']['median_ms_per_call']['1x4'], 'Selected tile is not faster')
    selected[shape] = label
  # Aggregate failure flags include sibling variants. Their failures stay in the
  # immutable report; only the separately complete selected proof admits a tile.
  return selected


def evidence(path, choices, environment=None):
  check = load('agent-decode-gemm')
  raw = Path(path).read_bytes()
  report = json.loads(raw)
  permitted = validate(report, choices, check.provenance(), os.environ if environment is None else environment)
  return permitted, dict(path=str(Path(path).resolve()), sha256=sha(raw), choices=choices,
    scope='Actual layer-0 weights and synthetic activations; full-route verification remains required')


def dispatch(original, permitted, repaired, stats):
  def gemm(x, weight, scales, tile=(4, 4), row_scale=False, bias=None, **kwargs):
    key = (x.shape[0], weight.shape[0], weight.shape[1])
    require(tuple(tile) == (1, 4) and row_scale is True and scales[0].shape == (x.shape[0], 1), 'Decoder row-scale contract changed')
    require(not kwargs, 'Unexpected decoder GEMM options')
    label = permitted.get(key) if bias is None else None
    if label is None: return original.gemm(x, weight, scales, tile=tile, row_scale=row_scale, bias=bias)
    name = ','.join(map(str, key))
    stats['python_trace_calls'][name] = stats['python_trace_calls'].get(name, 0)+1
    if label == '1x2': return original.gemm(x, weight, scales, tile=(1, 2), row_scale=True, bias=None)
    require(label == '1x1-accum', 'Unadmitted variant')
    return repaired(x, weight, scales, tile=(1, 1), row_scale=True, bias=None)
  return SimpleNamespace(gemm=gemm)


@contextmanager
def scoped(decoder, permitted, *, renderer=None):
  require(not decoder.verifiers and tuple(decoder.tile) == (1, 4), 'Install before verifier capture, with original 1x4 tile')
  original = decoder.gemm
  require(sha(Path(original.__file__).read_bytes()) == sha((HERE/'tiled-gemm.py').read_bytes()), 'Decoder kernel source differs')
  repaired = None
  if '1x1-accum' in permitted.values():
    require(renderer is not None, 'Explicit live renderer required for the private lowering')
    repair = load('agent-decode-gemm-rounding')
    def kernel(*args, **kwargs):
      return repair.kernel(*args, raw_kernel=partial(original.kernel, **kwargs), renderer=renderer)
    # Clone the original packing/epilogue wrapper with one private kernel global.
    # Quantizer calls, pack(), output allocation, and custom-kernel arguments stay intact.
    repaired = FunctionType(original.gemm.__code__, dict(original.gemm.__globals__, kernel=kernel),
                            original.gemm.__name__, original.gemm.__defaults__)
    repaired.__kwdefaults__ = original.gemm.__kwdefaults__
  stats = {'scope': 'decoder.verify GEMMs only', 'python_trace_calls': {},
           'selected': {','.join(map(str, key)): label for key, label in permitted.items()}}
  decoder.gemm = dispatch(original, permitted, repaired, stats)
  try: yield stats
  finally:
    decoder.gemm = original
    try:
      for jit in decoder.verifiers.values(): jit.reset()
    finally:
      decoder.verifiers.clear()
      stats['restored'] = True
