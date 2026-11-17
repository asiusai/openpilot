"""Strict packed BF16 PV checks. GPU mode is for the parent's serialized wrapper.

Only original 4x4, candidate 1x4, and candidate 2x2 are executed. All gates
precede timing. The timed scope is packed A/B through the complete BF16 output;
packing, QK, softmax, uploads, and result downloads are outside that scope.
"""
import argparse
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CALLS = 40
LABELS = ('original', '1x4', '2x2')
STATES = ('base', 'a-only', 'b-only-after-a', 'edges', 'restore')


@lru_cache(None)
def load(name):
  spec = importlib.util.spec_from_file_location('agent_expert_pv_check_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def pack_bits(values):
  """Independent integer scatter into the original 16x16 packed format."""
  batch, heads, rows, width = values.shape
  pr, pk = ((x+15)//16*16 for x in (rows, width))
  result = np.zeros((batch, heads, pr, pk), dtype=np.uint16)
  k = np.arange(width, dtype=np.int64)[None, :]
  for b in range(batch):
    for h in range(heads):
      destination = result[b, h].reshape(-1)
      for start in range(0, rows, 32):
        r = np.arange(start, min(start+32, rows), dtype=np.int64)[:, None]
        index = ((r//16)*(pk//16)+k//16)*256+(k%16//4)*64+(r%16)*4+k%4
        destination[index] = values[b, h, start:start+32]
  return result


def fixture(shape, state):
  E, C = load('agent-expert-gemm'), load('agent-expert-pv-compile')
  batch, heads, kv_heads, rows, columns, width = shape
  integer = width <= 64 and state != 'edges'
  seed = 9161200 + (100 if state == 'replay' else 200 if state == 'post' else 0)
  rng = np.random.default_rng(seed)
  if integer:
    a = E.bf16(rng.integers(-3, 4, (batch, heads, rows, width)).astype(np.float32))
    b = E.bf16(rng.integers(-3, 4, (batch, kv_heads, columns, width)).astype(np.float32))
  else:
    a = E.bf16(rng.uniform(0, 2/width, (batch, heads, rows, width)).astype(np.float32))
    b = E.bf16(rng.normal(0, 2, (batch, kv_heads, columns, width)).astype(np.float32))
  if state in ('a-only', 'b-only-after-a'):
    # Change one query head in one batch only. No other A or B byte changes.
    a[-1, -1, 0, 0] = E.bf16(np.float32(5 if integer else .125))
  if state == 'b-only-after-a':
    b[-1, -1, :, 0] = E.bf16(np.full(columns, -7 if integer else -8, np.float32))
  if width == 2368:
    # Representative real prefix gap: capacity 2304 plus all 64 action keys.
    # Masking has already happened upstream; these probabilities remain zero.
    a[..., 2017:2304] = 0
  if state == 'edges':
    edge = np.array([0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
                     0x3a80, 0xba80, 0x3b80, 0xbb80, 0x0002, 0x8002, 0, 0x8000], np.uint16)
    a[..., 0, :16] = edge
    b[..., 0, :16] = edge[::-1]
    b[..., min(1, columns-1), 8] = 0x7f7f  # Largest finite BF16; probabilities keep products finite.
    b[..., min(2, columns-1), 9] = 0xff7f
    a[..., -1, :] = 0
    a[..., -1, ::2] = 0x8000
    if width == 2368: a[..., 2017:2304] = 0
  values = (pack_bits(a), pack_bits(b).reshape(C.specs(shape)[2]))
  return {'values': values, 'a': a, 'b': b, 'integer': integer, 'state': state}


def integer_reference(data):
  E = load('agent-expert-gemm')
  assert data['integer']
  a, b = E.unpack_bf16(data['a']), E.unpack_bf16(data['b'])
  np.testing.assert_array_equal(a, a.astype(np.int64))
  np.testing.assert_array_equal(b, b.astype(np.int64))
  batch, heads, rows, _ = a.shape
  kv_heads, columns = b.shape[1:3]
  out = np.empty((batch, heads, rows, columns), dtype=np.int64)
  for bi in range(batch):
    for h in range(heads): out[bi, h] = a[bi, h].astype(np.int64) @ b[bi, h//(heads//kv_heads)].astype(np.int64).T
  # Every partial integer sum also fits exactly, independent of tree association.
  assert np.max(np.abs(a))*np.max(np.abs(b))*a.shape[-1] < 2**24
  return E.bf16(out.astype(np.float32))


class SerialPV:
  def __init__(self, shape, fn, count):
    from tinygrad import TinyJit
    self.shape, self.count, self.output, self.proofs = shape, count, None, []
    def execute(a, b, out):
      from tinygrad import Tensor
      for _ in range(count): out = Tensor.custom_kernel(out, a, b, fxn=fn)[0].realize()
      return (out,)
    self.inner = TinyJit(execute)

  @property
  def captured(self): return self.inner.captured

  def __call__(self, *inputs):
    from tinygrad import Tensor, dtypes
    E = load('agent-expert-gemm')
    if self.output is None:
      b, h, _, m, n, _ = self.shape
      self.output = Tensor.empty(b, h, m, n, device=inputs[0].device, dtype=dtypes.bfloat16).realize()
      self.output.uop.buffer.ensure_allocated()
    if self.count == CALLS and self.inner.cnt == 1:
      with E.observe_serial_queue(self.proofs): result = self.inner(*inputs, self.output)
    else: result = self.inner(*inputs, self.output)
    assert result[0].uop.buffer.base is self.output.uop.buffer.base
    assert result[0].uop.buffer.offset == self.output.uop.buffer.offset
    return result


def check_queue(jit):
  E = load('agent-expert-gemm')
  names = E.captured_kernels(jit)
  assert len(names) == CALLS and all('alpamayo_attention_values' in n or 'alpamayo_expert_pv_' in n for n in names)
  assert len(jit.proofs) == 1
  proof = jit.proofs[0]
  assert proof['output_input_slot'] == 2 and proof['ordered_execution_barriers'] == CALLS-1
  records = []
  for call in jit.captured._linear.src:
    for kernel in call.without_after.arg.aux.kernels:
      records.append(kernel)
      assert list(kernel[5]) == proof['kernel_global_input_slots']
      assert tuple(kernel[6][0]) == tuple(proof['output_argument_indices'])
      assert tuple(kernel[6][1]) == tuple(proof['input_argument_indices'])
  assert len(records) == CALLS
  return proof


def cpu_checks():
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.uop.ops import KernelInfo, UOp
  from amd_attention import _pack
  E, K, T = (load(n) for n in ('agent-expert-gemm', 'agent-expert-pv-kernel', 'agent-vision-tiles-transfer-check'))
  checks = []
  for name, shape in K.CASES.items():
    base = fixture(shape, 'base')
    for state in (*STATES, 'replay', 'post'):
      data = fixture(shape, state)
      for key, packed in zip(('a', 'b'), data['values'], strict=True):
        # uint16 checks the exact packed bytes, including BF16 subnormal bits.
        actual = _pack(Tensor(data[key], device='CPU')).numpy()
        np.testing.assert_array_equal(actual.reshape(-1), packed.reshape(-1))
      if shape[-1] == 2368:
        assert not data['a'][..., 2017:2304].any()
      if data['integer']:
        expected = integer_reference(data)
        a, b = E.unpack_bf16(data['a']), E.unpack_bf16(data['b'])
        expanded = np.repeat(b, shape[1]//shape[2], axis=1)
        result = Tensor(a, device='CPU').matmul(Tensor(expanded, device='CPU').transpose(-1, -2))
        np.testing.assert_array_equal(result.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy(), expected)
      if state == 'a-only':
        np.testing.assert_array_equal(data['b'], base['b'])
        changed = np.argwhere(data['a'] != base['a'])
        np.testing.assert_array_equal(changed, [[shape[0]-1, shape[1]-1, 0, 0]])
      if state == 'b-only-after-a':
        np.testing.assert_array_equal(data['a'], fixture(shape, 'a-only')['a'])
        assert (np.argwhere(data['b'] != base['b'])[:, :2] == (shape[0]-1, shape[2]-1)).all()
      if state == 'restore':
        for x, y in zip(data['values'], base['values'], strict=True): np.testing.assert_array_equal(x, y)
      checks.append({'case': name, 'state': state, 'packing_bit_exact': True, 'independent_integer_reference': data['integer']})
  # Focused GQA/batch isolation checks use exact integer values independently.
  shape = K.CASES['integer-tail-gqa']
  base, aa, bb = [integer_reference(fixture(shape, s)) for s in STATES[:3]]
  np.testing.assert_array_equal(base[0], aa[0])
  np.testing.assert_array_equal(base[1, :-1], aa[1, :-1])
  assert not np.array_equal(base[1, -1], aa[1, -1])
  np.testing.assert_array_equal(aa[0], bb[0])
  np.testing.assert_array_equal(aa[1, :2], bb[1, :2])
  assert not np.array_equal(aa[1, 2:], bb[1, 2:])
  # Preserve signed-zero and subnormal bits through independent BF16 conversion.
  edge = np.array([0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080, 0x7f7f, 0xff7f], np.uint16)
  np.testing.assert_array_equal(E.bf16(E.unpack_bf16(edge)), edge)
  def fill(out, a, b):
    r = UOp.range(out.numel(), 0)
    return out.flatten().index(r).store((a.flatten().index(0).float()+b.flatten().index(0).float()).cast(out.dtype)).end(r).sink(arg=KernelInfo())
  transfers = T.InputTransfers(Device['CPU'])
  values = [np.array([2], np.float32), np.array([3], np.float32)]
  inputs = transfers.upload(values, (dtypes.float32, dtypes.float32))
  jit = SerialPV((1, 1, 1, 2, 3, 1), fill, CALLS)
  for _ in range(3): jit(*inputs)
  assert len(jit.captured._linear.src) == CALLS
  proof = E.prove_cpu_queue(jit.captured._linear.src)
  assert proof['output_input_slot'] == 2
  for av, bv in ((2, 3), (7, 3), (7, -4), (2, 3)):
    values[0][0], values[1][0] = av, bv
    transfers.replace(inputs, values)
    actual = jit(*inputs)[0].bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(actual, E.bf16(np.full((1, 1, 2, 3), av+bv, np.float32)))
  # Reject a timing graph with independent outputs: it lacks serial WAW order.
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  independent = []
  for i, call in enumerate(jit.captured._linear.src):
    src = list(call.src)
    base, lane, offset = unwrap_lane(src[1])
    assert lane is None and offset == 0
    src[1] = src[1].substitute({base: UOp.param(10+i, base.dtype, base.max_numel(), base.device)}, walk=True)
    independent.append(call.replace(src=tuple(src)))
  try: E.prove_cpu_queue(independent)
  except AssertionError as exc: assert 'shared output storage' in str(exc)
  else: raise AssertionError('Independent outputs passed the serial proof')
  # Reuse frozen fake-buffer tests for stale versions, changed/restored bytes,
  # failed copies, one-image mutations, identity, and exact shadow decisions.
  transfer_checks = T.cpu_checks()
  assert not any(d.startswith('AMD') for d in Device._opened_devices)
  return {'checks': checks, 'integer_gqa_batch_isolation': True, 'bf16_edge_bits_preserved': True,
          'serial_queue_proof': proof, 'independent_output_rejected': True, 'cpu_transfer_checks': transfer_checks}


def gpu_checks(args, report, save):
  from tinygrad import Device, dtypes
  from tinygrad.device import Compiled
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  E, K, C, T = (load(n) for n in ('agent-expert-gemm', 'agent-expert-pv-kernel', 'agent-expert-pv-compile',
                                  'agent-vision-tiles-transfer-check'))
  Compiled.wait_timeout_ms = 10000
  load('prefill-kernels').fix_dispatch_packet()
  device = Device[Device.DEFAULT]
  assert device.arch == 'gfx1200' and device.is_usb and isinstance(device.renderer, AMDLLVMRenderer)
  load('wgp_compiler').install(device)
  report['gpu_executed'] = True
  started = time.perf_counter()
  def budget():
    assert time.perf_counter()-started < 240, 'Internal 240s budget reached; coverage remains incomplete'
  def reference(jit, inputs, data):
    E.poison((jit.output,), device)
    expected = E.read_output(jit(*inputs)[0], device)
    assert np.isfinite(E.unpack_bf16(expected)).all(), 'Reference produced a nonfinite value'
    if data['integer']: np.testing.assert_array_equal(expected, integer_reference(data))
    return expected
  actual_runtime = None
  for name, shape in K.CASES.items():
    budget()
    record = {'case': name, 'shape_b_h_kvh_m_n_k': shape, 'complete': False, 'checks': [], 'compiler': {}}
    report['cases'].append(record)
    save()
    functions = C.functions(device.renderer, shape)
    source_case = next(c for c in report['cpu_compilation']['cases'] if c['case'] == name)
    for label in LABELS:
      compiled = C.compile_one(device.renderer, shape, label)
      for key in ('llvm_sha256', 'binary_sha256'):
        assert compiled[key] == source_case['variants'][label][key], f'GPU/CPU compiled program differs: {name}/{label}/{key}'
      record['compiler'][label] = compiled
    transfers = T.InputTransfers(device)
    data = fixture(shape, 'base')
    inputs = transfers.upload(data['values'], (dtypes.bfloat16, dtypes.bfloat16))
    singles = {label: SerialPV(shape, functions[label], 1) for label in LABELS}
    for jit in singles.values():
      for _ in range(3): jit(*inputs)
      assert len(E.captured_kernels(jit)) == 1
    previous = base = None
    for state in STATES:
      budget()
      data = fixture(shape, state)
      transfer = transfers.replace(inputs, data['values'])
      if state == 'a-only': assert transfer['writes'] == [True, False]
      if state == 'b-only-after-a': assert transfer['writes'] == [False, True]
      expected = reference(singles['original'], inputs, data)
      if state == 'base': base = expected.copy()
      elif state == 'a-only':
        np.testing.assert_array_equal(expected[:-1], previous[:-1])
        np.testing.assert_array_equal(expected[-1, :-1], previous[-1, :-1])
        assert not np.array_equal(expected[-1, -1], previous[-1, -1])
      elif state == 'b-only-after-a':
        np.testing.assert_array_equal(expected[:-1], previous[:-1])
        unaffected = shape[1]-(shape[1]//shape[2])
        np.testing.assert_array_equal(expected[-1, :unaffected], previous[-1, :unaffected])
        assert not np.array_equal(expected[-1, unaffected:], previous[-1, unaffected:])
      elif state == 'restore': np.testing.assert_array_equal(expected, base)
      for label in LABELS:
        E.poison((singles[label].output,), device)
        E.exact_output(singles[label](*inputs)[0], expected, device)
        record['checks'].append({'state': state, 'variant': label, 'exact_bf16_bits': True,
                                 'independent_integer_reference': data['integer'], 'reference_sha256': E.sha(expected.tobytes())})
      previous = expected.copy()
      save()
    if name == 'expert':
      # Only masked history values change. Probabilities and the full history
      # capacity stay intact; output must remain exactly equal for all variants.
      changed = dict(data)
      changed['b'] = data['b'].copy()
      changed['b'][..., 2017:2304] = E.bf16(np.float32(17))
      changed['values'] = (data['values'][0], pack_bits(changed['b']).reshape(data['values'][1].shape))
      event = transfers.replace(inputs, changed['values'])
      assert event['writes'] == [False, True]
      for label in LABELS:
        E.poison((singles[label].output,), device)
        E.exact_output(singles[label](*inputs)[0], base, device)
      record['masked_history_value_mutation_exact'] = True
      batches = {label: SerialPV(shape, functions[label], CALLS) for label in LABELS}
      for label, jit in batches.items():
        budget()
        for _ in range(3): jit(*inputs)
        record.setdefault('queue_proofs', {})[label] = check_queue(jit)
      data = fixture(shape, 'replay')
      transfers.replace(inputs, data['values'])
      expected = reference(singles['original'], inputs, data)
      assert not np.array_equal(expected, base)
      for label, jit in batches.items():
        E.poison((jit.output,), device)
        E.exact_output(jit(*inputs)[0], expected, device)
      record['changed_a_and_b_after_40_call_capture_exact'] = True
      actual_runtime = (shape, singles, batches, transfers, inputs, expected, record)
    record['transfers'] = transfers.report()
    record['complete'] = True
    save()
  assert all(c['complete'] for c in report['cases']) and len(report['cases']) == len(K.CASES)
  report['all_pre_timing_exact_gates_passed'] = True
  save()
  shape, singles, batches, transfers, inputs, expected, record = actual_runtime
  timings = []
  def measure(jit):
    budget()
    device.synchronize()  # Outside the measured span.
    start = time.perf_counter()
    jit(*inputs)
    device.synchronize()  # The only completion fence inside the span.
    return (time.perf_counter()-start)*1000
  for candidate in LABELS[1:]:
    pairs = []
    for iteration in range(3):
      order = ['original', candidate] if iteration % 2 == 0 else [candidate, 'original']
      pair = {label: measure(batches[label]) for label in order}
      pairs.append({'order': order, 'batch_ms': pair})
    original = statistics.median(p['batch_ms']['original'] for p in pairs)/CALLS
    variant = statistics.median(p['batch_ms'][candidate] for p in pairs)/CALLS
    timings.append({'variant': candidate, 'pairs': pairs, 'calls_per_batch': CALLS,
                    'original_amortized_ms': original, 'candidate_amortized_ms': variant,
                    'candidate_over_original': variant/original, 'completion_fences_per_batch': 1,
                    'host_inclusive': True, 'device_only': False, 'warm_repeated_inputs': True})
  # Publish no timings until final reads and a fresh post-timing replay pass.
  for jit in batches.values(): E.exact_output(jit.output, expected, device)
  data = fixture(shape, 'post')
  event = transfers.replace(inputs, data['values'])
  assert event['writes'] == [True, True]
  final_expected = reference(singles['original'], inputs, data)
  assert not np.array_equal(final_expected, expected)
  for jit in batches.values():
    E.poison((jit.output,), device)
    E.exact_output(jit(*inputs)[0], final_expected, device)
  record['final_post_timing_fresh_a_b_replay_exact'] = True
  record['final_reference_sha256'] = E.sha(final_expected.tobytes())
  record['transfers'] = transfers.report()
  assert C.provenance()['source_sha256'] == report['source_sha256'], 'Sources changed during the GPU suite'
  report.update(timing=timings, final_post_timing_exact=True, complete=True, elapsed_gpu_suite_seconds=time.perf_counter()-started)
  save()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=('cpu', 'gpu'), default='cpu')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--compile-report', type=Path)
  parser.add_argument('--cpu-report', type=Path)
  args = parser.parse_args()
  os.environ.update(BEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  if args.mode == 'cpu': os.environ['DEV'] = 'CPU'
  E, C = load('agent-expert-gemm'), load('agent-expert-pv-compile')
  report = dict(C.provenance(), mode=args.mode, complete=False, gpu_executed=False, cases=[], timing=[],
    numerical_candidate=False, exactness_qualified=False,
    timing_scope='Complete packed BF16 PV operation; original K16 WMMA through BF16 output. No QK, softmax, packing or transfers.',
    runtime_environment={k: os.environ.get(k) for k in ('DEV', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL', 'CACHEDB', 'TC_OPT')})
  def save(): E.write_json(args.output, report)
  save()
  try:
    if args.mode == 'cpu':
      report['cpu_checks'] = cpu_checks()
      report['complete'] = True
    else:
      assert args.compile_report and args.compile_report.is_file()
      assert args.cpu_report and args.cpu_report.is_file()
      compiled = json.loads(args.compile_report.read_text())
      checked = json.loads(args.cpu_report.read_text())
      assert compiled['complete'] and not compiled['gpu_executed'] and compiled['wgp']
      assert checked['complete'] and checked['mode'] == 'cpu' and not checked['gpu_executed']
      assert compiled['source_sha256'] == report['source_sha256'], 'Pinned source bytes changed; repeat CPU qualification'
      assert checked['source_sha256'] == report['source_sha256'], 'Pinned CPU checks are stale'
      assert compiled['tinygrad_revision'] == report['tinygrad_revision']
      report['compile_report_sha256'] = E.sha(args.compile_report.read_bytes())
      report['cpu_report_sha256'] = E.sha(args.cpu_report.read_bytes())
      report['cpu_compilation'] = {'cases': compiled['cases'], 'wgp': True}
      gpu_checks(args, report, save)
      report['exactness_qualified'] = True
  except BaseException as exc:
    report.update(complete=False, exactness_qualified=False, timing=[], failure=f'{type(exc).__name__}: {str(exc)[:1600]}')
    save()
    raise
  save()
  print(json.dumps({'complete': report['complete'], 'gpu_executed': report['gpu_executed'],
                    'exactness_qualified': report['exactness_qualified']}), flush=True)


if __name__ == '__main__': main()
