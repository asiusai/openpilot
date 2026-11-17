"""Bounded six-head exactness and GPU-timestamp checks. Default: CPU only."""
import argparse
from collections import Counter
from functools import lru_cache, partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time
from types import MethodType

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
CALLS = 40


@lru_cache(None)
def load(name):
  spec = importlib.util.spec_from_file_location('agent_spec_head_check_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def bf16_bits(values):
  bits = np.asarray(values, np.float32).view(np.uint32)
  return ((bits+np.uint32(32767)+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(bits): return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def fp8_values(bits):
  bits = np.asarray(bits, np.uint8)
  exp, mantissa = (bits >> 3) & 15, bits & 7
  magnitude = np.where(exp == 0, mantissa.astype(np.float32)*2**-9,
    (1+mantissa.astype(np.float32)/8)*np.exp2(exp.astype(np.int32)-7))
  return (magnitude*np.where(bits & 128, -1, 1)).astype(np.float32)


def fp8_bits(values):
  """Independent nearest-neighbor finite E4M3 encoder, including subnormals."""
  values = np.asarray(values, np.float32)
  assert np.isfinite(values).all()
  table, magnitude = fp8_values(np.arange(127, dtype=np.uint8)).astype(np.float64), np.abs(values).astype(np.float64)
  hi = np.minimum(np.searchsorted(table, magnitude), 126)
  lo = np.maximum(hi-1, 0)
  lower, upper = magnitude-table[lo], table[hi]-magnitude
  code = np.where((upper < lower) | ((upper == lower) & ((hi & 1) == 0)), hi, lo).astype(np.uint8)
  return code | (np.signbit(values).astype(np.uint8) << 7)


def lane_reference(queries, weight, scales, weight_scale):
  """Independent FP32 four-product/K/shuffle reference, used for integer controls.

  E4M3 products fit exactly in FP32. Contracting product+sum therefore does not
  change these individual four-term sums. K and shuffle associations still matter.
  """
  q, w = fp8_values(queries), fp8_values(weight)
  result = []
  for row, scale in zip(q, scales, strict=True):
    lanes = np.zeros((len(w), 32), np.float32)
    for k in range(w.shape[1]//128):
      cols = k*128+np.arange(32)[:, None]*4+np.arange(4)[None, :]
      products = w[:, cols]*row[cols]
      block = products[..., 0]
      for j in (1, 2, 3): block = np.float32(block+products[..., j])
      lanes = np.float32(lanes+block)
    for shift in (16, 8, 4, 2, 1): lanes = np.float32(lanes+lanes[:, np.arange(32)^shift])
    result.append(bf16_bits(lanes[:, 0]*np.float32(np.float32(scale)*np.float32(weight_scale))))
  return np.stack(result)


def cpu_checks():
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.dtype import float_to_fp8
  from tinygrad.uop.ops import Ops, UOp
  from amd_decode import _gemv
  from model import EdgeModel
  H, policy = load('agent-spec-head'), load('agent-speculation-policy')
  checks = []
  for vocab, width in ((64, 256), (2112, 2048)):
    specs = [((6, vocab), dtypes.bfloat16), ((vocab, width), dtypes.fp8e4m3), ((1,), dtypes.float32)]
    specs += [spec for _ in range(6) for spec in (((1, width), dtypes.fp8e4m3), ((1,), dtypes.float32))]
    args = [UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(specs)]
    combined = H.kernel(*args)
    for i in range(6):
      original = _gemv(UOp.placeholder((vocab,), dtypes.bfloat16, 30), args[3+2*i], args[1], args[4+2*i], args[2], packed=False)
      left = next(u.src[1] for u in original.toposort() if u.op is Ops.STORE)
      right = next(u.src[1] for u in combined.src[i].toposort() if u.op is Ops.STORE)
      assert left is right, f'Query {i} does not retain the original expression DAG'
    checks.append({'case': 'pre-codegen-six-independent-original-expression-DAGs', 'shape': [6, vocab, width], 'exact': True})
  table = fp8_values(np.arange(127, dtype=np.uint8))
  midpoints = (table[:-1]+table[1:])*np.float32(.5)
  rounding = np.concatenate([table, midpoints, np.nextafter(midpoints, np.float32(0)), np.nextafter(midpoints, np.float32(np.inf))])
  rounding = np.concatenate([rounding, -rounding])
  np.testing.assert_array_equal(fp8_bits(rounding), np.array([float_to_fp8(float(v), dtypes.fp8e4m3) for v in rounding], np.uint8))
  rng = np.random.default_rng(906061)
  for state in range(4):
    q, w = rng.integers(-3, 4, (6, 256)), rng.integers(-3, 4, (64, 256))
    scales = np.exp2(np.arange(6, dtype=np.float32)-4-state)
    ws = np.float32(2**(-3+state))
    expected = bf16_bits((q.astype(np.int64) @ w.astype(np.int64).T).astype(np.float32)*(scales[:, None]*ws))
    np.testing.assert_array_equal(lane_reference(fp8_bits(q), fp8_bits(w), scales, ws), expected)
  checks.append({'case': 'independent-integer-dot-lane-tree-and-scale-mutations', 'states': 4, 'exact': True})
  checks.append({'case': 'independent-fp8-neighbor-rounding-including-subnormals', 'values': len(rounding), 'exact': True})
  # Every row retains its own dynamic scale; a neighboring outlier cannot change it.
  model = EdgeModel.__new__(EdgeModel)
  model.cuda_kernels = False
  model.weights = {'norm.weight': Tensor(np.ones(2048, np.float32), device='CPU').cast(dtypes.bfloat16).realize()}
  data = rng.normal(size=(6, 2048)).astype(np.float32)
  data[0] = 0
  data[1] *= 2**-30
  data[2, 0] = 1000
  def prep(values):
    outputs = []
    x = Tensor(values, device='CPU').cast(dtypes.bfloat16).realize()
    for i in range(6):
      q, s = model.quantize(model.norm(x[i:i+1], 'norm', rms=True))
      q.realize(s)
      outputs.append((q.bitcast(dtypes.uint8).numpy(), s.numpy()))
    return outputs
  before = prep(data)
  data[4] = 0
  after = prep(data)
  assert not np.array_equal(before[4][0], after[4][0])
  for i in (0, 1, 2, 3, 5):
    for a, b in zip(before[i], after[i], strict=True): np.testing.assert_array_equal(a, b)
  assert float(before[0][1]) == float(np.float32(1e-12))
  assert len(set(float(row[1]) for row in before)) == 6
  checks.append({'case': 'zero-near-zero-outlier-single-row-scale-isolation', 'exact': True,
                 'cpu_fp8_subnormal_flush_is_not_gpu_validation': True})
  # This exercises the actual verification expression and captured mutation.
  draft = np.asarray(policy.CHECKPOINT_DRAFT_IDS, np.int32)
  mapping_np = np.concatenate([draft, np.arange(6, 64, dtype=np.int32)+200000])
  mapping_np[-1] = mapping_np[-2]
  mapping = Tensor(mapping_np, device='CPU').realize()
  values = np.full((6, 64), -1, np.float32)
  values[np.arange(6), np.arange(6)] = 1
  logits = Tensor(values, device='CPU').cast(dtypes.bfloat16).realize()
  verifier = TinyJit(H.verification_payload)
  for _ in range(3): payload = verifier(logits, mapping).numpy()
  np.testing.assert_array_equal(payload, np.concatenate([draft, np.ones(6, np.int32)]))
  assert policy.accept_targets(payload[:6], draft, draft[-1], bool(np.all(payload[6:] == 1)))
  for row in range(6):
    changed = values.copy()
    changed[row, row] = -1
    changed[row, (row+1) % 6] = 1
    logits.uop.buffer.host[:] = bf16_bits(changed).tobytes()
    payload = verifier(logits, mapping).numpy()
    assert not policy.accept_targets(payload[:6], draft, draft[-1], bool(np.all(payload[6:] == 1)))
  for nonfinite in (np.inf, np.nan):
    for row in range(6):
      changed = values.copy()
      changed[row, 63] = nonfinite
      logits.uop.buffer.host[:] = bf16_bits(changed).tobytes()
      payload = verifier(logits, mapping).numpy()
      assert payload[6+row] == 0 and np.count_nonzero(payload[6:] == 0) == 1
      assert not policy.accept_targets(draft, draft, draft[-1], bool(np.all(payload[6:] == 1)))
  near_tie = np.zeros((6, 64), np.float32)
  near_tie[:, 2] = 1
  near_tie[:, 3] = np.nextafter(np.float32(1), np.float32(np.inf))
  logits.uop.buffer.host[:] = bf16_bits(near_tie).tobytes()
  payload = verifier(logits, mapping).numpy()
  np.testing.assert_array_equal(payload[:6], np.full(6, mapping_np[2], np.int32))
  checks.append({'case': 'captured-six-argmax-finite-flags-near-tie-and-full-rejection', 'mutations': 19, 'exact': True})
  assert not any(device.startswith('AMD') for device in Device._opened_devices)
  return {'gpu_executed': False, 'checks': checks}


def upload(values, dtypes_):
  from tinygrad import Tensor
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  tensors = []
  for value, dtype in zip(values, dtypes_, strict=True):
    value = np.ascontiguousarray(value)
    buffer = Buffer('AMD', value.nbytes, options=BufferSpec(cpu_access=True)).ensure_allocated()
    buffer.host.view(size=buffer.nbytes, fmt='B')[:] = value.tobytes()
    tensors.append(Tensor(UOp.from_buffer(buffer, dtype)).reshape(value.shape))
  return tuple(tensors)


def replace(tensors, values, device):
  device.synchronize()
  for tensor, value in zip(tensors, values, strict=True):
    buffer = tensor.uop.buffer
    assert value.nbytes == buffer.nbytes
    buffer.host.view(size=buffer.nbytes, fmt='B')[:] = value.tobytes()


def read(tensor, device, dtype):
  device.synchronize()
  buffer, chunks = tensor.uop.buffer, []
  offset, remaining = buffer.offset, tensor.nbytes()
  for address, size in buffer.base.get_storage().meta.mapping.paddrs:
    if offset >= size: offset -= size; continue
    count = min(remaining, size-offset)
    for start in range(0, count, 65536):
      view = device.iface.pci_dev.map_bar(bar=device.iface.vram_bar, off=address+offset+start, size=min(count-start, 65536))
      chunks.append(bytes(view[:]))
    remaining, offset = remaining-count, 0
    if not remaining: break
  assert remaining == 0
  return np.frombuffer(b''.join(chunks), dtype).reshape(tensor.shape)


class Timestamps:
  """One synchronized graph call per sample; timestamp reads stay outside its span."""
  def __init__(self, device):
    from tinygrad.helpers import PROFILE
    assert PROFILE
    self.events = []
    def collect(dev):
      buffers = {}
      for (buffer, _), entry in dev.prof_ents.items():
        if buffer not in buffers:
          raw = buffer.host.view(fmt='Q')[:]
          if isinstance(raw, (bytes, bytearray)) or (isinstance(raw, memoryview) and raw.itemsize == 1):
            raw = struct.unpack('<'+'Q'*(len(raw)//8), raw)
          buffers[buffer] = raw
        values = buffers[buffer]
        start, end = int(values[entry.st_id]), int(values[entry.en_id])
        assert 0 <= end-start < dev.timestamp_divider*1000*300000, (entry.name, start, end)
        self.events.append({'name': str(entry.name), 'start': start, 'end': end})
      dev.prof_ents.clear()
    device.collect_prof = MethodType(collect, device)
    device._at_profile_finalize = lambda: None

  def sample(self, fn, args, device):
    device.synchronize()
    self.events.clear()
    begin = time.perf_counter()
    fn(*args)
    device.synchronize()
    host_ms = (time.perf_counter()-begin)*1000
    assert self.events, 'No device timestamp events; refuse host-only kernel timing'
    start, end = min(e['start'] for e in self.events), max(e['end'] for e in self.events)
    return {'gpu_span_ms': (end-start)/device.timestamp_divider/1000,
      'sum_kernel_ms': sum(e['end']-e['start'] for e in self.events)/device.timestamp_divider/1000,
      'host_with_sync_and_profile_reads_ms': host_ms, 'events': self.events.copy()}


def baseline_prepared(weight, ws, *pairs):
  from amd_decode import gemv
  return tuple(gemv(pairs[2*i], weight, (pairs[2*i+1], ws), packed=False).realize() for i in range(6))


def candidate_prepared(weight, ws, *pairs):
  H = load('agent-spec-head')
  result = H.project_prepared(weight, ws, pairs[::2], pairs[1::2]).realize()
  return tuple(result[i:i+1] for i in range(6))


def prepared_values(vocab, width, state, integer):
  rng = np.random.default_rng(906062+state)
  if integer:
    q, w = fp8_bits(rng.integers(-3, 4, (6, width))), fp8_bits(rng.integers(-3, 4, (vocab, width)))
  else:
    q = rng.integers(0, 255, (6, width), dtype=np.uint8)
    w = rng.integers(0, 255, (vocab, width), dtype=np.uint8)
    q[(q & 127) == 127] = 0
    w[(w & 127) == 127] = 0
  # Distinct dynamic scales and mutations catch stale or incorrectly shared scalars.
  scales = np.exp2(np.arange(6, dtype=np.float32)-10-state) if integer else np.array([.00031, .00417, .00001, .00731, .00218, .00171], np.float32)*(1+state*.13)
  ws = np.array([2**(-3+state) if integer else .000713*(1+state*.19)], np.float32)
  return [w, ws, *(item for i in range(6) for item in (q[i:i+1].copy(), np.array([scales[i]], np.float32)))], q, scales


def prepared_check(device, vocab, width, integer):
  from tinygrad import TinyJit, dtypes
  functions = {'baseline': TinyJit(baseline_prepared), 'candidate': TinyJit(candidate_prepared)}
  data, _, _ = prepared_values(vocab, width, 0, integer)
  inputs = upload(data, [dtypes.fp8e4m3, dtypes.float32]+[dtype for _ in range(6) for dtype in (dtypes.fp8e4m3, dtypes.float32)])
  for fn in functions.values():
    for _ in range(3): fn(*inputs)
  states = []
  previous = None
  for state in range(4):
    data, q, scales = prepared_values(vocab, width, state, integer)
    replace(inputs, data, device)
    expected = np.stack([read(t, device, np.uint16).reshape(-1) for t in functions['baseline'](*inputs)])
    actual = np.stack([read(t, device, np.uint16).reshape(-1) for t in functions['candidate'](*inputs)])
    np.testing.assert_array_equal(actual, expected, err_msg=f'Prepared six-head {vocab}x{width} state {state}')
    if integer: np.testing.assert_array_equal(actual, lane_reference(q, data[0], scales, data[1][0]))
    if previous is not None: assert not np.array_equal(previous, actual), 'Mutation did not change logits'
    previous = actual.copy()
    states.append({'state': state, 'all_logits_bf16_exact': True, 'independent_integer': integer})
  return {'shape': [6, vocab, width], 'checks': states}


def cache_weights(cache, config_path, vocab_path):
  index = json.loads((cache/'index.json').read_text())
  config, raw_vocab = json.loads(config_path.read_text()), vocab_path.read_bytes()
  mapping = sorted(set(json.loads(raw_vocab)))
  mapping = [i for i in mapping if
    not config['traj_ids']['history_id0'] <= i < config['traj_ids']['history_id0']+config['history_vocab_size'] and
    not config['traj_ids']['future_id0'] <= i < config['traj_ids']['future_id0']+config['future_vocab_size'] and
    i != config['vlm_config']['text_config']['eos_token_id']]
  assert len(mapping) == 2093
  mapping += [mapping[-1]]*((-len(mapping)) % 64)
  mapping = np.asarray(mapping, np.int32)
  assert len(mapping) == 2112
  def item(name, kind, dtype):
    record = next(r for r in index['records'] if r['name'] == name and r['kind'] == kind and not r['packed'])
    return np.memmap(cache/'weights.bin', mode='r', dtype=dtype, offset=record['offset'], shape=tuple(record['shape']))
  weight = np.array(item('vlm.lm_head.weight', 'fp8', np.uint8)[mapping], copy=True)
  ws = np.array(item('vlm.lm_head.weight', 'scale', np.float32), copy=True)
  norm = np.array(item('vlm.model.language_model.norm.weight', 'bf16', np.uint16), copy=True)
  assert weight.shape == (2112, 2048) and norm.shape == (2048,) and ws.shape == (1,)
  provenance = {'cache_index_sha256': hashlib.sha256((cache/'index.json').read_bytes()).hexdigest(),
    'vocab_sha256': hashlib.sha256(raw_vocab).hexdigest(), 'selected_weight_sha256': hashlib.sha256(weight.tobytes()).hexdigest(),
    'scale_bits': int(ws.view(np.uint32)[0]), 'reduced_tokens': 2093, 'padded_tokens': 2112}
  return weight, ws, norm, mapping, provenance


def full_stage(variant, x, weight, ws, norm, mapping, length):
  from tinygrad import Tensor, dtypes
  from model import EdgeModel
  model = EdgeModel.__new__(EdgeModel)
  model.cuda_kernels, model.fp8_wmma, model.dtype = False, True, dtypes.bfloat16
  model.weights = {'vlm.lm_head.weight': weight, 'vlm.model.language_model.norm.weight': norm}
  model.weight_scales = {'vlm.lm_head.weight': ws}
  model.output_token_mapping = mapping
  if variant == 'candidate': return load('agent-spec-head').heads(model, x, length, return_all=True)
  targets, checks, logits = [], [], []
  for index in range(6):
    hidden = model.norm(x[:, length-1+index:length+index], 'vlm.model.language_model.norm', rms=True)
    row = model.linear(hidden, 'vlm.lm_head').realize()
    logits.append(row)
    targets.append(mapping[row.reshape(-1).argmax()].reshape(1))
    checks.append(row.isfinite().all().cast(dtypes.int32).reshape(1))
  return logits[0], Tensor.cat(*targets, *checks).contiguous().realize(), *logits


def full_values(weights, state):
  weight, ws, norm, mapping = [item.copy() for item in weights]
  rng = np.random.default_rng(906063+state)
  hidden = rng.normal(size=(1, 2112, 2048)).astype(np.float32)
  length = 2027 if state % 2 == 0 else 2100
  if state == 1:
    hidden[:, length-1] = 0
    hidden[:, length] *= np.float32(2**-30)
    hidden[:, length+1, 0] = 1000
  if state == 2:
    weight = np.roll(weight, 1, axis=0).copy()
    norm = bf16_bits(bf16_values(norm)*np.float32(1.0625))
    ws *= np.float32(1.17)
  if state == 3:
    # Duplicate the entire vocabulary to force exact ties after BF16 rounding.
    weight[:] = weight[:1]
    hidden[:, length-1:length+5] = hidden[:, length-1:length]
  return [bf16_bits(hidden), weight, ws, norm, mapping], length


def full_check(device, timestamps, cache, config, vocab):
  from tinygrad import TinyJit, dtypes
  from tinygrad.uop.ops import UOp
  weight, ws, norm, mapping, provenance = cache_weights(cache, config, vocab)
  weights = (weight, ws, norm, mapping)
  values, length = full_values(weights, 0)
  inputs = upload(values, (dtypes.bfloat16, dtypes.fp8e4m3, dtypes.float32, dtypes.bfloat16, dtypes.int32))
  functions = {name: TinyJit(partial(full_stage, name)) for name in ('baseline', 'candidate')}
  def arguments(value): return (*inputs, UOp.variable('valid_prefill_draft_length', 1, 2106).bind(value))
  for fn in functions.values():
    for _ in range(3): fn(*arguments(length))
  report = {'shape': [6, 2112, 2048], 'cache': provenance, 'checks': [], 'samples': {name: [] for name in functions}}
  for state in range(4):
    values, length = full_values(weights, state)
    replace(inputs, values, device)
    results = {}
    for name, fn in functions.items():
      result = fn(*arguments(length))
      results[name] = (read(result[0], device, np.uint16).reshape(-1), read(result[1], device, np.int32),
        np.stack([read(row, device, np.uint16).reshape(-1) for row in result[2:]]))
    for actual, expected in zip(results['candidate'], results['baseline'], strict=True):
      np.testing.assert_array_equal(actual, expected, err_msg=f'Complete six-head stage mutation {state}')
    if state == 3:
      np.testing.assert_array_equal(results['candidate'][1][:6], np.full(6, mapping[0], np.int32))
    report['checks'].append({'state': state, 'valid_length': length, 'all_logits_bf16_exact': True,
      'first_logits_bf16_exact': True, 'all_six_tokens_and_finite_flags_exact': True})
  # Restore real head weights and normal random inputs before timing.
  values, length = full_values(weights, 0)
  replace(inputs, values, device)
  for fn in functions.values():
    for _ in range(5): fn(*arguments(length))
  device.synchronize()
  for iteration in range(CALLS):
    for name in (('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')):
      sample = timestamps.sample(functions[name], arguments(length), device)
      counts = Counter(e['name'] for e in sample['events'])
      if name == 'candidate': assert counts['agent_spec_head_shared_gemv6'] == 1, counts
      else: assert counts['alpamayo_fp8_gemv'] == 6, counts
      sample['iteration'] = iteration
      report['samples'][name].append(sample)
  report['timing'] = {name: {'median_gpu_span_ms': statistics.median(s['gpu_span_ms'] for s in samples),
    'median_sum_kernel_ms': statistics.median(s['sum_kernel_ms'] for s in samples), 'calls': len(samples)}
    for name, samples in report['samples'].items()}
  report['timing_limit'] = '40 individually synchronized calls per variant. Device spans include all preparation and verification. Profiling changes clocks/host overhead; full-pipeline speed still requires a separate unprofiled route run.'
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--gpu', action='store_true')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--cache', type=Path, default=Path('/dev/shm/alpamayo-9060-fp8'))
  parser.add_argument('--config', type=Path, default=BASE/'model/config.json')
  parser.add_argument('--vocab', type=Path, default=HERE/'coc_vocab.json')
  args = parser.parse_args()
  if not args.gpu: os.environ.update(DEV='CPU', PROFILE='0')
  else: os.environ['PROFILE'] = '1'
  report = {'complete': False, 'gpu_executed': args.gpu, 'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
    (Path(__file__), HERE/'agent-spec-head.py', HERE/'agent-speculation-prefill.py', HERE/'agent-speculation-policy.py', HERE.parent/'model.py', HERE.parent/'amd_decode.py')},
    'environment': {key: os.environ.get(key) for key in ('DEV', 'AMD', 'AMD_IFACE', 'AMD_LLVM', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL', 'GMMU', 'PROFILE', 'BEAM', 'TC_OPT', 'PYTHONPATH')}}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  def save(): args.output.write_text(json.dumps(report, indent=2)+'\n')
  save()
  try:
    if not args.gpu:
      report['cpu'] = cpu_checks()
      weight, ws, norm, mapping, provenance = cache_weights(args.cache, args.config, args.vocab)
      assert np.isfinite(bf16_values(norm)).all() and np.isfinite(ws).all()
      assert not np.any((weight & 127) == 127)
      np.testing.assert_array_equal(weight[2092:], np.repeat(weight[2092:2093], 20, axis=0))
      report['read_only_cache_fixture'] = provenance
    else:
      from tinygrad import Device
      from tinygrad.renderer.llvmir import AMDLLVMRenderer
      device = Device['AMD']
      assert device.arch == 'gfx1200' and isinstance(device.renderer, AMDLLVMRenderer)
      report['device'] = {'arch': device.arch, 'renderer': type(device.renderer).__name__, 'timestamp_divider': device.timestamp_divider,
        'clock_policy': 'Inherited from the serialized parent wrapper; this program changes no power or clock policy.'}
      timestamps = Timestamps(device)
      report['prepared'] = []
      for vocab, width, integer in ((64, 256, True), (2112, 2048, True), (2112, 2048, False)):
        report['prepared'].append(prepared_check(device, vocab, width, integer))
        save()
      report['full_stage'] = full_check(device, timestamps, args.cache, args.config, args.vocab)
    report['complete'] = True
  except Exception as exc:
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    save()
    raise
  save()
  print(json.dumps({'complete': True, 'output': str(args.output), 'gpu_executed': args.gpu,
    'timing': report.get('full_stage', {}).get('timing')}), flush=True)


if __name__ == '__main__': main()
