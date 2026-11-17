"""Independent rounding controls and complete FC1+activation+packing timing.

GPU mode must run through the parent's serialized wrapper. Default: CPU checks.
"""
import argparse
from contextlib import contextmanager
from functools import lru_cache, partial
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
CALLS = 40


def load(name):
  spec = importlib.util.spec_from_file_location('agent_packed_check_'+name.replace('-', '_'), HERE/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = result
  spec.loader.exec_module(result)
  return result


E, P, F = (load(name) for name in ('agent-expert-gemm', 'agent-packed-epilogue', 'gemm-epilogue'))


def rounded_bf16(values):
  """Independent nearest-neighbor BF16 rounding, with even tie selection."""
  values = np.asarray(values, np.float32)
  bits = values.view(np.uint32)
  magnitude = bits & np.uint32(0x7fffffff)
  lower_bits = magnitude & np.uint32(0xffff0000)
  upper_bits = lower_bits+np.uint32(0x10000)
  with np.errstate(invalid='ignore'):
    lower, upper = lower_bits.view(np.float32).astype(np.float64), upper_bits.view(np.float32).astype(np.float64)
  value = np.abs(values).astype(np.float64)
  with np.errstate(invalid='ignore'):
    dl, du = value-lower, upper-value
    up = (du < dl) | ((du == dl) & ((lower_bits >> 16) & 1).astype(bool))
  # Infinite input is retained; large finite values use the usual overflow midpoint.
  up |= np.isfinite(values) & (magnitude >= np.uint32(0x7f7f8000))
  output = np.where(up, upper_bits, lower_bits) | (bits & np.uint32(0x80000000))
  return output.astype(np.uint32).view(np.float32)


def encode_fp8(values):
  values = np.clip(np.asarray(values, np.float32), -448, 448)
  assert not np.isnan(values).any()
  table = E.decode_fp8(np.arange(127, dtype=np.uint8)).astype(np.float64)
  magnitude = np.abs(values).astype(np.float64)
  high = np.minimum(np.searchsorted(table, magnitude), 126)
  low = np.maximum(high-1, 0)
  dl, du = magnitude-table[low], table[high]-magnitude
  result = np.where((du < dl) | ((du == dl) & ((high & 1) == 0)), high, low).astype(np.uint8)
  return result | ((np.signbit(values)).astype(np.uint8) << 7)


def reference_epilogue(sums, xs, ws, output_scale, inverse=None):
  scale = np.float32(np.float32(xs)*np.float32(ws))
  with np.errstate(over='ignore', under='ignore', invalid='ignore'):
    h = rounded_bf16(np.asarray(sums, np.float32)*scale)
    positive = np.maximum(h, np.float32(0))
    squared = rounded_bf16(positive*positive)
    quant = encode_fp8(squared/np.float32(output_scale) if inverse is None else squared*np.float32(inverse))
  return E.pack_cpu(quant)


def rounding_values():
  rng = np.random.default_rng(906020)
  # Finite arbitrary FP32 values include BF16 subnormals and overflow boundaries.
  bits = rng.integers(0, 2**32, 8192, dtype=np.uint32)
  bits[(bits & 0x7f800000) == 0x7f800000] &= np.uint32(0xff7fffff)
  table = E.decode_fp8(np.arange(127, dtype=np.uint8))
  mid = (table[:-1]+table[1:])*np.float32(.5)
  # Before square, these target FP8 midpoints and adjacent FP32 values.
  roots = np.sqrt(mid).astype(np.float32)
  boundaries = np.concatenate([roots, np.nextafter(roots, np.float32(0)), np.nextafter(roots, np.float32(np.inf))])
  bfmid_bits = ((np.arange(0x3b00, 0x4200, dtype=np.uint32) << 16)+0x8000)
  values = np.concatenate([bits.view(np.float32), boundaries, bfmid_bits.view(np.float32),
    np.array([0, -0., 2**-149, -2**-149, 2**-133, -2**-133, 2**-126, .03125, -.03125, 448, -448], np.float32)])
  return np.pad(values, (0, (-len(values)) % 64)).reshape(-1, 64)


def postprocess(h, output_scale):
  from tinygrad import dtypes
  from amd_gemm import pack
  return pack((h.relu().square().float()/output_scale).clamp(-448, 448).cast(dtypes.fp8e4m3))


def cpu_encoded_reference(h, output_scale):
  # Pinned CPU codegen flushes FP8 subnormals (codegen/decomp/dtype.py:114).
  # Its scalar dtype converter preserves them and independently checks our nearest-neighbor encoder.
  from tinygrad import dtypes
  from tinygrad.dtype import float_to_fp8
  quant = (h.relu().square().float()/output_scale).clamp(-448, 448).numpy()
  bits = np.array([float_to_fp8(float(value), dtypes.fp8e4m3) for value in quant.reshape(-1)], np.uint8).reshape(quant.shape)
  return E.pack_cpu(bits)


@lru_cache(None)
def post_template(rows, columns):
  """Compile the existing expression while permitting explicit output reuse."""
  from tinygrad import Tensor, dtypes
  from tinygrad.uop.ops import Ops
  h = Tensor.empty(rows, columns, device='CPU', dtype=dtypes.bfloat16)
  output_scale = Tensor.empty(1, device='CPU', dtype=dtypes.float32)
  calls = postprocess(h, output_scale).schedule_linear().src
  assert len(calls) == 1
  body = calls[0].body
  params = sorted((u for u in body.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
  assert [u.arg.slot for u in params] == [0, 1, 2]
  assert [u.dtype for u in params] == [dtypes.fp8e4m3, dtypes.bfloat16, dtypes.float32]
  return body, tuple(params)


def post_program(out, h, output_scale):
  body, params = post_template(*h.shape)
  return body.substitute({p: value.flatten() for p, value in zip(params, (out, h, output_scale), strict=True)}, walk=True)


def queue_proof(batch, queues, count, observed=False):
  """Require the complete operation sequence and explicit storage dependencies."""
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  from tinygrad.uop.ops import Ops
  assert count == CALLS and len(batch) in (CALLS, 2*CALLS), len(batch)
  stages = len(batch)//CALLS
  assert len(queues) == 1 and len({(d, q) for _, d, q in batch}) == 1
  signatures, output_spans = [], []
  for call, _, _ in batch:
    assert call.body.op is Ops.PROGRAM
    outs, ins = get_call_outs_ins(call)
    assert len(outs) == 1
    output = call.src[1+outs[0]]
    base, lane, offset = unwrap_lane(output)
    assert base.op is Ops.PARAM and lane is None, 'Every stage must use explicit reusable output storage'
    output_spans.append((base.arg.slot, offset, output.max_numel()*output.dtype.itemsize))
    args = [unwrap_lane(call.src[1+g]) for g in call.body.arg.globals]
    assert all(b.op is Ops.PARAM and lane is None for b, lane, _ in args), 'No hidden matrix or quantization buffers'
    signatures.append((call.body, tuple((b.arg.slot, offset) for b, _, offset in args), tuple(outs), tuple(ins)))
  assert all(signatures[i] == signatures[i % stages] for i in range(len(batch)))
  assert all(output_spans[i] == output_spans[i % stages] for i in range(len(batch)))
  assert output_spans[stages-1][0] == 5
  if stages == 2: assert output_spans[0][0] == 6 and output_spans[0] != output_spans[1]
  ordered = []
  for cmd in next(iter(queues.values())):
    if cmd.op is Ops.CALL and cmd.body.op is Ops.PROGRAM: ordered.append(cmd)
    elif cmd.op is Ops.CALL and cmd.body.op is Ops.CUSTOM_FUNCTION and cmd.body.arg.name == 'exec_barrier': ordered.append('barrier')
  expected = [item for i, (call, _, _) in enumerate(batch) for item in ((call,) if i == 0 else ('barrier', call))]
  assert ordered == expected, 'Missing execution barrier between complete operation stages'
  return {'operations': count, 'kernels': len(batch), 'kernels_per_operation': stages,
    'ordered_execution_barriers': len(batch)-1, 'stage_output_spans': output_spans[:stages],
    'stage_argument_mappings': [s[1] for s in signatures[:stages]], 'actual_hcq_queue_observed': observed,
    'pinned_queue_builder_executed': True}


@contextmanager
def observe(proofs, count):
  from tinygrad.runtime.support import hcq2
  old = hcq2._build_queues
  def build(ctx):
    queues = old(ctx)
    if len(ctx.batch) >= count: proofs.append(queue_proof(ctx.batch, queues, count, observed=True))
    return queues
  hcq2._build_queues = build
  try: yield
  finally: hcq2._build_queues = old


class Batch:
  def __init__(self, shape, variant, count, matrix=None):
    from tinygrad import Tensor, TinyJit, dtypes
    from amd_gemm import _gemm
    self.shape, self.variant, self.count, self.proofs = shape, variant, count, []
    self.output = Tensor.empty((shape[0]+15)//16*16, shape[1], dtype=dtypes.fp8e4m3).realize()
    self.scratch = Tensor.empty(shape[0], shape[1], dtype=dtypes.bfloat16).realize()
    self.output.uop.buffer.ensure_allocated()
    self.scratch.uop.buffer.ensure_allocated()
    def run(a, w, xs, ws, output_scale, out, temp):
      for _ in range(count):
        if variant == 'baseline':
          temp = Tensor.custom_kernel(temp, a, w, xs, ws, fxn=_gemm if matrix is None else matrix)[0].realize()
          # Redirect the existing postprocess expression's output PARAM. assign(pack(...))
          # would insert an extra copy and unfairly inflate the baseline.
          out = Tensor.custom_kernel(out, temp, output_scale, fxn=post_program)[0].realize()
        elif variant == 'old-fused':
          out = Tensor.custom_kernel(out, a, w, xs, ws, output_scale, fxn=partial(F.program, rows=shape[0]))[0].realize()
        else: out = P.packed(a, w, xs, ws, output_scale, shape[0], variant, out=out).realize()
      return out
    self.inner = TinyJit(run)

  def __call__(self, *inputs):
    if self.count == CALLS and self.inner.cnt == 1:
      with observe(self.proofs, self.count): result = self.inner(*inputs, self.output, self.scratch)
    else: result = self.inner(*inputs, self.output, self.scratch)
    assert result.uop.buffer.base is self.output.uop.buffer.base and result.uop.buffer.offset == self.output.uop.buffer.offset
    return result


def cpu_checks():
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.dtype import float_to_fp8
  from tinygrad.uop.ops import UOp, KernelInfo
  from amd_gemm import pack
  checks = []
  values = rounding_values()
  table = E.decode_fp8(np.arange(127, dtype=np.uint8))
  midpoints = (table[:-1]+table[1:])*np.float32(.5)
  boundaries = np.concatenate([table, midpoints, np.nextafter(midpoints, np.float32(0)),
                               np.nextafter(midpoints, np.float32(np.inf))])
  boundaries = np.concatenate([boundaries, -boundaries])
  scalar_bits = np.array([float_to_fp8(float(v), dtypes.fp8e4m3) for v in boundaries], np.uint8)
  np.testing.assert_array_equal(scalar_bits, encode_fp8(boundaries))
  checks.append({'case': 'all-fp8-midpoints-and-neighbors', 'elements': len(boundaries), 'signed_subnormals_exact': True})
  scales = ((1., 1., 1.), (.3, .2, .35), (.071234569, .01933221, .0937129), (2**-20, 2**-20, 2**-80))
  for xs, ws, output_scale in scales:
    x, s, t, q = (Tensor(v, device='CPU') for v in (values, np.array([xs], np.float32), np.array([ws], np.float32), np.array([output_scale], np.float32)))
    h = (x*(s*t)).cast(dtypes.bfloat16)
    actual = cpu_encoded_reference(h, q)
    expected = reference_epilogue(values, xs, ws, output_scale)
    np.testing.assert_array_equal(actual, expected)
    emitted = postprocess(h, q).bitcast(dtypes.uint8).numpy()
    difference = emitted != expected
    assert np.all((emitted[difference] == 0) & (expected[difference] > 0) & (expected[difference] < 8)), 'Unexpected CPU conversion difference'
    checks.append({'case': 'rounding', 'elements': values.size, 'scales': [xs, ws, output_scale], 'independent_scalar_exact_bytes': True,
                   'known_cpu_codegen_flushed_fp8_subnormals': int(np.count_nonzero(difference))})
  for shape in ((17, 48, 32), (65, 80, 64)):
    for state in range(5):
      data = E.fixture(shape, state, True)
      sums = E.decode_fp8(data['a_unpacked']).astype(np.int64) @ E.decode_fp8(data['w_unpacked']).astype(np.int64).T
      expected = reference_epilogue(sums.astype(np.float32), data['sx'][0], data['sw'][0], .35)
      a, w = (Tensor(E.decode_fp8(data[key+'_unpacked']), device='CPU') for key in ('a', 'w'))
      h = (a.dot(w.T)*(Tensor(data['sx'], device='CPU')*Tensor(data['sw'], device='CPU'))).cast(dtypes.bfloat16)
      actual = cpu_encoded_reference(h, Tensor(np.array([.35], np.float32), device='CPU'))
      np.testing.assert_array_equal(actual, expected)
      for key in ('a', 'w'): np.testing.assert_array_equal(pack(Tensor(data[key+'_unpacked'], device='CPU')).numpy(), data[key])
      checks.append({'case': 'integer-matrix', 'shape': shape, 'state': state, 'exact_bytes': True})
  # The fixture must detect removal of either BF16 boundary.
  positive = np.maximum(rounded_bf16(values), np.float32(0))
  with np.errstate(over='ignore', under='ignore'):
    exact = reference_epilogue(values, 1., 1., 1.)
    omit_first = E.pack_cpu(encode_fp8(rounded_bf16(np.maximum(values, np.float32(0))**2)))
    omit_second = E.pack_cpu(encode_fp8(positive*positive))
  first, second = np.count_nonzero(exact != omit_first), np.count_nonzero(exact != omit_second)
  assert first > 0 and second > 0
  checks.append({'case': 'rounding-boundary-counterexamples', 'omit_first_changes_bytes': int(first), 'omit_second_changes_bytes': int(second)})
  mapping_checks = 0
  for rows, columns in ((17, 48), (65, 80)):
    values = np.arange(rows*columns, dtype=np.uint32).astype(np.uint8).reshape(rows, columns)
    expected = E.pack_cpu(values)
    for tm, tn in P.TILES.values():
      result, writes = np.full(expected.shape, 0xa5, np.uint8), np.zeros(expected.size//4, np.uint8)
      for mt in range(math.ceil(rows/(16*tm))):
        for nt in range(math.ceil(columns/(16*tn))):
          for lane in range(32):
            for i in range(tm):
              for j in range(tn):
                row, col = mt*16*tm+i*16+lane%16, nt*16*tn+j*16+lane//16*8
                if row >= expected.shape[0] or col >= columns: continue
                loc = (row//16*(columns//16)+col//16)*64+(col%16//4)*16+row%16
                for q in range(2):
                  payload = values[row, col+q*4:col+q*4+4] if row < rows else np.zeros(4, np.uint8)
                  result.reshape(-1)[4*(loc+q*16):4*(loc+q*16)+4] = payload
                  writes[loc+q*16] += 1
                  mapping_checks += 1
      np.testing.assert_array_equal(result, expected)
      assert (writes == 1).all()
  # The same baseline expression must form exactly two serial kernels per operation.
  def fill(out, a, w, xs, ws):
    r = UOp.range(out.numel(), 0)
    return out.flatten().index(r).store((a.flatten().index(0).float()*xs.flatten().index(0)).cast(out.dtype)).end(r).sink(arg=KernelInfo())
  inputs = tuple(Tensor(np.array([v], np.float32), device='CPU').realize() for v in (2, 1, .5, 1, 1))
  batch = Batch((2, 16, 16), 'baseline', CALLS, matrix=fill)
  for _ in range(3): batch(*inputs)
  linear = batch.inner.captured._linear.src
  from tinygrad.runtime.support.hcq2 import _build_queues
  calls = [(call, 'CPU', 'COMPUTE:0') for call in linear]
  ctx = SimpleNamespace(batch=calls, calls={('CPU', 'COMPUTE:0'): list(range(len(calls)))},
    peers={('CPU', 'COMPUTE:0'): set()}, queues={'CPU': ['COMPUTE:0']}, stamps=lambda *_: ())
  proof = queue_proof(calls, _build_queues(ctx), CALLS)
  # Separate packed outputs must fail the reuse proof, even with a shared BF16 scratch buffer.
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  changed = []
  for index, (call, device, queue) in enumerate(calls):
    if index % 2:
      out_index = get_call_outs_ins(call)[0][0]
      sources = list(call.src)
      base, lane, offset = unwrap_lane(sources[1+out_index])
      assert lane is None and offset == 0
      sources[1+out_index] = sources[1+out_index].substitute({base: UOp.param(100+index, base.dtype, base.max_numel(), base.device)}, walk=True)
      call = call.replace(src=tuple(sources))
    changed.append((call, device, queue))
  bad = SimpleNamespace(batch=changed, calls=ctx.calls, peers=ctx.peers, queues=ctx.queues, stamps=ctx.stamps)
  try: queue_proof(changed, _build_queues(bad), CALLS)
  except AssertionError: pass
  else: raise AssertionError('Independent packed outputs passed the shared-storage proof')
  inputs[0].uop.buffer.host[:] = np.array([4], np.float32).tobytes()
  actual = batch(*inputs).bitcast(dtypes.uint8).numpy()
  np.testing.assert_array_equal(actual, E.pack_cpu(np.full((2, 16), encode_fp8(np.float32(4)), np.uint8)))
  assert not any(device.startswith('AMD') for device in Device._opened_devices)
  return {'checks': checks, 'packed_word_mapping_checks': mapping_checks, 'serial_baseline_proof': proof,
          'changed_input_after_capture_exact': True, 'independent_outputs_rejected': True, 'gpu_executed': False}


def upload(values, dtypes_):
  from tinygrad import Tensor
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  result = []
  for value, dtype in zip(values, dtypes_, strict=True):
    buffer = Buffer('AMD', value.nbytes, options=BufferSpec(cpu_access=True)).ensure_allocated()
    buffer.host.view(size=buffer.nbytes, fmt='B')[:] = value.tobytes()
    result.append(Tensor(UOp.from_buffer(buffer, dtype)).reshape(value.shape))
  return tuple(result)


def replace(inputs, values, device):
  device.synchronize()
  for tensor, value in zip(inputs, values, strict=True):
    buffer = tensor.uop.buffer
    assert buffer.nbytes == value.nbytes
    buffer.host.view(size=buffer.nbytes, fmt='B')[:] = value.tobytes()


def read_output(tensor, device, storage_dtype=np.uint8):
  device.synchronize()
  buffer, chunks = tensor.uop.buffer, []
  offset, remaining = buffer.offset, tensor.nbytes()
  for address, size in buffer.base.get_storage().meta.mapping.paddrs:
    if offset >= size: offset -= size; continue
    count = min(remaining, size-offset)
    for start in range(0, count, 65536):
      view = device.iface.pci_dev.map_bar(bar=device.iface.vram_bar, off=address+offset+start, size=min(count-start, 65536))
      chunks.append(bytes(view[:]))
    remaining -= count
    offset = 0
    if not remaining: break
  assert remaining == 0
  return np.frombuffer(b''.join(chunks), storage_dtype).reshape(tensor.shape)


def reciprocal_operand(scale, device):
  # Isolate the baseline's hardware reciprocal. The independent CPU oracle uses
  # this operand and implements all later arithmetic/rounding itself. No candidate output enters it.
  # Full IEEE division would be a different oracle from the baseline's fdiv-afn lowering.
  return read_output(scale.reciprocal().realize(), device, np.float32).reshape(-1)[0]


def values_for(shape, state, integer):
  data = E.fixture(shape, state, integer)
  # Avoid saturation dominating the random matrix test; still retain complete FP8 input support.
  if not integer:
    data['sx'] = np.array([.0003125 if state < 2 else .000417137], np.float32)
    data['sw'] = np.array([.000271831 if state < 3 else .000351193], np.float32)
  scale = np.array([.35 if state < 2 else .071234569 if state < 4 else .0937129], np.float32)
  return [data['a'], data['w'], data['sx'], data['sw'], scale], data


def run_case(shape, variants, device, rounds, integer, include_old_fused=True):
  from tinygrad import dtypes
  values, data = values_for(shape, 0, integer)
  inputs = upload(values, (dtypes.fp8e4m3, dtypes.fp8e4m3, dtypes.float32, dtypes.float32, dtypes.float32))
  record = {'shape_mnk': shape, 'integer_control': integer, 'variants': {}, 'timing': []}
  singles = {label: Batch(shape, label, 1) for label in ('baseline', *(('old-fused',) if include_old_fused else ()), *variants)}
  for label, batch in singles.items():
    for _ in range(3): batch(*inputs)
    record['variants'][label] = {'status': 'pending', 'checks': []}
  previous = None
  for state in range(4):
    values, data = values_for(shape, state, integer)
    replace(inputs, values, device)
    E.poison([singles['baseline'].output], device)
    expected = read_output(singles['baseline'](*inputs), device)
    if integer:
      sums = E.decode_fp8(data['a_unpacked']).astype(np.int64) @ E.decode_fp8(data['w_unpacked']).astype(np.int64).T
      inverse = reciprocal_operand(inputs[4], device)
      np.testing.assert_array_equal(expected, reference_epilogue(sums.astype(np.float32), values[2][0], values[3][0], values[4][0], inverse=inverse))
      record.setdefault('independent_reference_operands', []).append({'state': state,
        'output_scale_fp32_bits': int(values[4].view(np.uint32)[0]), 'reciprocal_fp32_bits': int(np.asarray(inverse).view(np.uint32))})
    if previous is not None: assert not np.array_equal(expected, previous), f'Mutation state {state} did not change output'
    previous = expected.copy()
    if state == 0:
      from tinygrad import Tensor
      from amd_gemm import _gemm
      Q = load('quantize-packed')
      h = Tensor.custom_kernel(Tensor.empty(shape[0], shape[1], dtype=dtypes.bfloat16), *inputs[:4], fxn=_gemm)[0]
      public = Q.quantize(h.relu().square(), inputs[4])[0]
      np.testing.assert_array_equal(read_output(public, device), expected)
      record['public_gemm_relu_square_quantize_packed_exact'] = True
    for label in list(singles):
      E.poison([singles[label].output], device)
      actual = read_output(singles[label](*inputs), device)
      try: np.testing.assert_array_equal(actual, expected)
      except AssertionError as exc:
        if label == 'baseline': raise
        record['variants'][label].update(status='incorrect', failed_state=state, mismatch_bytes=int(np.count_nonzero(actual != expected)), error=str(exc)[:800])
        del singles[label]
        continue
      record['variants'][label]['checks'].append({'state': state, 'exact_bytes': True, 'independent_integer': integer})
      record['variants'][label]['status'] = 'exact-single'
  if integer: return record
  batches = {label: Batch(shape, label, CALLS) for label in singles}
  for label, batch in batches.items():
    for _ in range(3): batch(*inputs)
    assert len(batch.proofs) == 1
    expected_kernels = 2*CALLS if label == 'baseline' else CALLS
    assert batch.proofs[0]['kernels'] == expected_kernels
    record['variants'][label]['serial_queue_proof'] = batch.proofs[0]
  values, _ = values_for(shape, 4, False)
  replace(inputs, values, device)
  expected = read_output(singles['baseline'](*inputs), device)
  assert not np.array_equal(expected, previous)
  for label in list(batches):
    E.poison([batches[label].output, batches[label].scratch], device)
    actual = read_output(batches[label](*inputs), device)
    try: np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
      if label == 'baseline': raise
      record['variants'][label].update(status='incorrect-batch-replay', error=str(exc)[:800])
      del batches[label]
      continue
    record['variants'][label].update(status='exact', changed_inputs_and_scales_after_capture_exact=True)
  def measure(batch):
    device.synchronize()
    start = time.perf_counter()
    batch(*inputs)
    device.synchronize()
    return (time.perf_counter()-start)*1000
  for label in batches:
    if label == 'baseline': continue
    pairs = []
    for iteration in range(rounds):
      order = ('baseline', label) if iteration % 2 == 0 else (label, 'baseline')
      pairs.append({key: measure(batches[key]) for key in order})
    base, value = (statistics.median(pair[key] for pair in pairs)/CALLS for key in ('baseline', label))
    record['timing'].append({'variant': label, 'baseline_complete_operation_ms': base, 'candidate_complete_operation_ms': value,
      'candidate_over_baseline': value/base, 'paired_40_operation_graph_ms': pairs, 'host_inclusive': True,
      'includes_gemm_relu_bf16_square_fixed_quantization_packing': True, 'input_packing_and_uploads_excluded_identically': True})
  return record


def probe_gpu(variants, device):
  from tinygrad import Tensor, dtypes
  values = rounding_values()
  rows, columns = values.shape
  arrays = [values, np.zeros((16, 16), np.uint8), np.array([1.], np.float32), np.array([1.], np.float32), np.array([1.], np.float32)]
  inputs = upload(arrays, (dtypes.float32, dtypes.fp8e4m3, dtypes.float32, dtypes.float32, dtypes.float32))
  records = []
  for xs, ws, scale in ((1., 1., 1.), (.3, .2, .35), (.071234569, .01933221, .0937129), (2**-20, 2**-20, 2**-80)):
    arrays[2:] = [np.array([v], np.float32) for v in (xs, ws, scale)]
    replace(inputs, arrays, device)
    reference = postprocess((inputs[0]*(inputs[2]*inputs[3])).cast(dtypes.bfloat16), inputs[4]).realize()
    expected = read_output(reference, device)
    inverse = reciprocal_operand(inputs[4], device)
    np.testing.assert_array_equal(expected, reference_epilogue(values, xs, ws, scale, inverse=inverse))
    ideal_division = reference_epilogue(values, xs, ws, scale)
    for variant in variants:
      out = Tensor.empty((rows+15)//16*16, columns, dtype=dtypes.fp8e4m3)
      result = Tensor.custom_kernel(out, *inputs, fxn=partial(P.program, rows=rows, variant=variant, epilogue_only=True))[0].realize()
      np.testing.assert_array_equal(read_output(result, device), expected, err_msg=f'Independent epilogue probe {variant}')
      records.append({'variant': variant, 'elements': values.size, 'scales': [xs, ws, scale], 'exact_bytes': True,
        'independent_reference_uses_isolated_baseline_reciprocal': True,
        'reciprocal_operand_fp32_bits': int(np.asarray(inverse).view(np.uint32)),
        'baseline_vs_ideal_full_division_changed_bytes': int(np.count_nonzero(expected != ideal_division))})
  return records


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=('cpu-check', 'gpu'), default='cpu-check')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--rounds', type=int, default=3)
  parser.add_argument('--variants', default=','.join(P.TILES))
  parser.add_argument('--rows', default='2112,2027', help='Bounded actual FC1 rows; small controls always run')
  parser.add_argument('--skip-old-fused', action='store_true', help='Exclude the previously rejected alternative from this qualification')
  args = parser.parse_args()
  variants = args.variants.split(',')
  assert 1 <= len(variants) <= 3 and len(variants) == len(set(variants)) and all(v in P.TILES for v in variants)
  assert 3 <= args.rounds <= 5
  actual_rows = tuple(map(int,args.rows.split(',')))
  assert actual_rows and len(set(actual_rows)) == len(actual_rows) and set(actual_rows) <= {2027,2048,2112}
  os.environ.update(BEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  if args.mode == 'cpu-check': os.environ['DEV'] = 'CPU'
  from tinygrad import Device
  import tinygrad
  sources = (Path(__file__), HERE/'agent-packed-epilogue.py', HERE/'gemm-epilogue.py', HERE/'quantize-packed.py', HERE/'agent-expert-gemm.py', HERE.parent/'amd_gemm.py')
  report = {'mode': args.mode, 'gpu_executed': False, 'complete': False, 'cases': [],
    'source_sha256': {str(path): E.sha(path.read_bytes()) for path in sources}, 'variants': variants, 'calls_per_timed_graph': CALLS,
    'tinygrad_revision': subprocess.check_output(['git', '-C', str(Path(tinygrad.__file__).parent.parent), 'rev-parse', 'HEAD'], text=True).strip(),
    'runtime_environment': {key: os.environ.get(key) for key in ('DEV', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL', 'CACHEDB')},
    'timing_scope': 'Complete no-bias FC1 + BF16 ReLU square + fixed FP32-scale quantization + packed output; warm weights.',
    'pipeline_validated': False}
  report['requested_rows'],report['old_fused_enabled'] = actual_rows,not args.skip_old_fused
  report['compiled_selected_kernels'] = []
  try:
    for rows in actual_rows:
      for variant in variants:
        text,binary = P.compiled(rows,9216,2048,variant)
        report['compiled_selected_kernels'].append({'rows':rows,'variant':variant,'source_sha256':E.sha(text.encode()),
          'binary_sha256':E.sha(binary),'binary_bytes':len(binary),'gpu_execution_proven':False})
    if args.mode == 'cpu-check': report['checks'] = cpu_checks()
    else:
      assert os.environ.get('AM_POWER_LIMIT') == '144' and os.environ.get('AM_PERF_LEVEL') == 'auto'
      from tinygrad.device import Compiled
      from tinygrad.renderer.llvmir import AMDLLVMRenderer
      Compiled.wait_timeout_ms = 10000
      load('prefill-kernels').fix_dispatch_packet()
      device = Device[Device.DEFAULT]
      assert device.arch == 'gfx1200' and device.is_usb and isinstance(device.renderer, AMDLLVMRenderer)
      load('wgp_compiler').install(device)
      report['gpu_executed'] = True
      report['epilogue_probes'] = probe_gpu(variants, device)
      E.write_json(args.output, report)
      rejected = set()
      for shape in ((17, 48, 32), (65, 80, 64), *((rows,9216,2048) for rows in actual_rows)):
        integer = shape[0] < 128
        report['active_shape'] = shape
        E.write_json(args.output, report)
        row = run_case(shape, [v for v in variants if v not in rejected], device, args.rounds, integer,not args.skip_old_fused)
        report['cases'].append(row)
        if integer: rejected.update(label for label, info in row['variants'].items() if info['status'] == 'incorrect')
        E.write_json(args.output, report)
        print(json.dumps({'shape': shape, 'statuses': {k: v['status'] for k, v in row['variants'].items()}, 'timing': row['timing']}), flush=True)
    report.pop('active_shape', None)
    report.update(complete=True, summary={'gpu_executed': report['gpu_executed'], 'cases': len(report['cases']), 'pipeline_validated': False})
    E.write_json(args.output, report)
    print(json.dumps(report['summary']), flush=True)
  except BaseException as exc:
    report.update(complete=False, failure=f'{type(exc).__name__}: {exc}')
    E.write_json(args.output, report)
    raise


if __name__ == '__main__': main()
