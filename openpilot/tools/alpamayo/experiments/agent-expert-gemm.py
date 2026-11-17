"""Bounded expert GEMM tile checks. Default mode compiles without GPU access.

GPU mode belongs to the parent's serialized trial wrapper. Each timed graph
contains exactly 40 serial matrix calls and uses one completion fence.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
TILES = ((4, 4), (1, 4), (2, 4), (1, 8), (2, 8), (2, 2))
CASES = {
  'q': (64, 2048, 512), 'kv': (64, 1024, 512), 'qkv-combined': (64, 4096, 512),
  'o': (64, 512, 2048), 'fc1': (64, 3840, 512), 'fc2': (64, 512, 3840),
}
CONTROLS = {'integer': (64, 128, 64), 'integer-tail': (17, 48, 32)}
CALLS = 40


def sha(data): return hashlib.sha256(data).hexdigest()


def write_json(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  temp = path.with_name(path.name+'.new')
  temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
  temp.replace(path)


def local(name):
  spec = importlib.util.spec_from_file_location('agent_expert_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def decode_fp8(bits):
  exponent, mantissa = ((bits >> 3) & 15).astype(np.int32), (bits & 7).astype(np.float32)
  value = np.where(exponent == 0, mantissa*2.0**-9, (1+mantissa/8)*np.exp2(exponent-7))
  value = np.where((bits & 127) == 127, np.nan, value)
  return np.where(bits & 128, -value, value).astype(np.float32)


def bf16(value):
  bits = np.asarray(value, dtype=np.float32).view(np.uint32)
  assert np.isfinite(np.asarray(value, dtype=np.float32)).all()
  return ((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def unpack_bf16(bits): return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def pack_cpu(values):
  """Independent scatter using the documented packed address, including tails."""
  rows, width = values.shape
  result = np.zeros((((rows+15)//16)*16, width), dtype=values.dtype)
  k = np.arange(width, dtype=np.int64)[None, :]
  for begin in range(0, rows, 64):
    r = np.arange(begin, min(rows, begin+64), dtype=np.int64)[:, None]
    address = (r//16*(width//16)+k//16)*256+(k%16//4)*64+(r%16)*4+k%4
    result.reshape(-1)[address] = values[begin:begin+64]
  return result


def int_fp8(values):
  table = decode_fp8(np.arange(127, dtype=np.uint8))
  magnitude = np.abs(values).astype(np.float32)
  code = np.searchsorted(table, magnitude).astype(np.uint8)
  np.testing.assert_array_equal(table[code], magnitude)
  return code | ((values < 0).astype(np.uint8) << 7)


def fixture(shape, state, integer=False):
  """State changes separately cover inputs, scales, weights/bias, then batch replay."""
  m, n, k = shape
  a_rng = np.random.default_rng(906016 + (0 if state == 0 else 1 if state < 4 else 2))
  w_rng = np.random.default_rng(906117 + (0 if state < 3 else 1 if state == 3 else 2))
  if integer:
    a, w = int_fp8(a_rng.integers(-7, 8, (m, k))), int_fp8(w_rng.integers(-7, 8, (n, k)))
  else:
    def bits(rng, dimensions):
      return rng.integers(0, 127, dimensions, dtype=np.uint8) | (rng.integers(0, 2, dimensions, dtype=np.uint8) << 7)
    a, w = bits(a_rng, (m, k)), bits(w_rng, (n, k))
    a.reshape(-1)[:8] = np.array([0, 128, 1, 129, 126, 254, 56, 184], np.uint8)
  bias = bf16(w_rng.integers(-8, 9, n).astype(np.float32)/8)
  if state < 2: sx, sw = .125, .25
  elif integer: sx, sw = (.25, .5) if state < 4 else (.5, .125)
  else: sx, sw = (.071234569, .01933221) if state < 4 else (.0937129, .0291127)
  return {'a': pack_cpu(a), 'w': pack_cpu(w), 'sx': np.array([sx], np.float32), 'sw': np.array([sw], np.float32),
          'bias': bias, 'a_unpacked': a, 'w_unpacked': w}


def integer_reference(data, with_bias):
  a, w = decode_fp8(data['a_unpacked']), decode_fp8(data['w_unpacked'])
  np.testing.assert_array_equal(a, a.astype(np.int64))
  np.testing.assert_array_equal(w, w.astype(np.int64))
  sums = a.astype(np.int64) @ w.astype(np.int64).T
  # The control sums fit exactly in FP32, and scales are powers of two.
  assert np.max(np.abs(sums)) < 2**24
  result = sums.astype(np.float32)*(data['sx'][0]*data['sw'][0])
  if with_bias: result += unpack_bf16(data['bias'])
  return bf16(result)


def bind_renderer(function, renderer):
  """Clone a private function namespace; do not patch production modules."""
  class Devices:
    DEFAULT = 'agent-expert-offline'
    def __getitem__(self, key): return SimpleNamespace(renderer=renderer)
  clone = FunctionType(function.__code__, dict(function.__globals__, Device=Devices()), function.__name__, function.__defaults__)
  clone.__kwdefaults__ = function.__kwdefaults__
  return clone


def functions(renderer, tiles):
  from amd_gemm import _gemm
  tiled = local('tiled-gemm')
  result = {'original': bind_renderer(_gemm, renderer)}
  for tm, tn in tiles:
    result[f'{tm}x{tn}'] = partial(bind_renderer(tiled.kernel, renderer), tile_m=tm, tile_n=tn,
                                  row_scale=False, column_scale=False, local_m=1, local_n=1,
                                  swap=False, unroll_k=1, group_m=1)
  return result


def compiler_record(shape, with_bias, fn, renderer):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt import KernelOptError
  from tinygrad.uop.ops import UOp
  m, n, k = shape
  args = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate([
    ((m, n), dtypes.bfloat16), ((((m+15)//16)*16, k), dtypes.fp8e4m3),
    ((((n+15)//16)*16, k), dtypes.fp8e4m3), ((1,), dtypes.float32), ((1,), dtypes.float32)])]
  if with_bias: args.append(UOp.placeholder((n,), dtypes.bfloat16, 5))
  try: program = to_program(fn(*args), renderer)
  except (KernelOptError, StopIteration, ValueError, AssertionError) as exc:
    return {'status': 'invalid-axis-or-schedule', 'error': f'{type(exc).__name__}: {exc}'}
  except RuntimeError as exc:
    return {'status': 'compile-rejected', 'error': str(exc)}
  binary = program.src[3].arg
  metadata = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
  names = ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
           'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode')
  resources = {key: int(value) for key, value in re.findall(r'\.('+'|'.join(names)+r'):\s*(\d+)', metadata)}
  assert {'vgpr_count', 'wavefront_size', 'private_segment_fixed_size', 'group_segment_fixed_size'} <= resources.keys()
  groups, threads = math.prod(program.arg.global_size), math.prod(program.arg.local_size)
  # Analyze the real AMD program with symbolic PARAMs and pinned HCQ's queue builder.
  # This never allocates a GPU buffer or opens a device.
  params = [UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip((5, 0, 1, 2, 3, 4), args)]
  call = program.call(*params)
  serial = prove_cpu_queue([call]*CALLS, device='AMD')
  assert serial['output_input_slot'] == 5
  return {'status': 'compiled', 'global_size': program.arg.global_size, 'local_size': program.arg.local_size,
          'workgroups': groups, 'threads_per_workgroup': threads,
          'waves_per_workgroup': threads//resources['wavefront_size'], 'resources': resources,
          'binary_bytes': len(binary), 'binary_sha256': sha(binary), 'offline_serial_queue_proof': serial}


def cpu_checks():
  from tinygrad import Tensor, Device, dtypes
  from amd_gemm import pack
  checks = []
  for name, shape in CONTROLS.items():
    for state in range(5):
      data = fixture(shape, state, integer=True)
      for key in ('a', 'w'):
        source = data[key+'_unpacked']
        np.testing.assert_array_equal(pack(Tensor(source, device='CPU')).numpy(), data[key])
      for with_bias in (False, True):
        a, w = (Tensor(decode_fp8(data[key+'_unpacked']), device='CPU') for key in ('a', 'w'))
        result = a.dot(w.T)*(float(data['sx'][0])*float(data['sw'][0]))
        if with_bias: result += Tensor(unpack_bf16(data['bias']), device='CPU')
        np.testing.assert_array_equal(result.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy(), integer_reference(data, with_bias))
        checks.append({'case': name, 'state': state, 'bias': with_bias, 'independent_integer_exact': True, 'packing_exact': True})
  # Exercise the exact 40-call capture structure using a CPU custom kernel.
  from tinygrad.uop.ops import UOp, KernelInfo
  def fill(out, a, w, sx, sw, *bias):
    r = UOp.range(out.numel(), 0)
    return out.flatten().index(r).store((a.flatten().index(0).float()+sx.flatten().index(0)).cast(out.dtype)).end(r).sink(arg=KernelInfo())
  inputs = tuple(Tensor(np.array([v], np.float32), device='CPU').realize() for v in (2, 1, 3, 1, 0))
  jit = make_jit((2, 3, 1), False, fill, CALLS)
  for _ in range(3): jit(*inputs)
  assert len(jit.captured._linear.src) == CALLS
  proof = prove_cpu_queue(jit.captured._linear.src)
  assert proof['output_input_slot'] == 5 and proof['ordered_execution_barriers'] == CALLS-1
  inputs[0].uop.buffer.host[:] = np.array([7], np.float32).tobytes()
  actual = jit(*inputs)[0].bitcast(dtypes.uint16).numpy()
  np.testing.assert_array_equal(actual, bf16(np.full((2, 3), 10, np.float32)))
  checks.append({'case': 'serial-40-call-capture', 'mutated_input_exact': True, **proof})
  # Compile a second calling convention whose output is argument 1, not 0.
  from tinygrad import TinyJit
  def output_second(a, out): return fill(out, a, a, a, a)
  def unusual(a, out):
    for _ in range(CALLS): out = Tensor.custom_kernel(a, out, fxn=output_second)[1].realize()
    return out
  out = Tensor.empty(2, 3, dtype=dtypes.bfloat16, device='CPU').realize()
  other = TinyJit(unusual)
  for _ in range(3): other(inputs[0], out)
  proof = prove_cpu_queue(other.captured._linear.src)
  assert proof['output_argument_indices'] == [1] and proof['output_input_slot'] == 1
  # Distinct storage removes WAW ordering; the proof must reject that graph.
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  independent = []
  for i, call in enumerate(other.captured._linear.src):
    src = list(call.src)
    b, lane, offset = unwrap_lane(src[2])
    assert lane is None and offset == 0
    src[2] = src[2].substitute({b: UOp.param(10+i, b.dtype, b.max_numel(), b.device)}, walk=True)
    independent.append(call.replace(src=tuple(src)))
  try: prove_cpu_queue(independent)
  except AssertionError as exc: assert 'shared output storage' in str(exc)
  else: raise AssertionError('Independent outputs incorrectly passed the serial proof')
  checks.append({'case': 'nonzero-output-argument-and-overlap-rejection', 'independent_outputs_rejected': True, **proof})
  assert not any(d.startswith('AMD') for d in Device._opened_devices)
  return checks


def upload(data):
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  result = []
  for key in ('a', 'w', 'sx', 'sw', 'bias'):
    value = data[key]
    dtype = dtypes.fp8e4m3 if key in ('a', 'w') else dtypes.bfloat16 if key == 'bias' else dtypes.float32
    buffer = Buffer('AMD', value.nbytes, options=BufferSpec(cpu_access=True)).ensure_allocated()
    buffer.host.view(size=buffer.nbytes, fmt='B')[:] = value.tobytes()
    result.append(Tensor(UOp.from_buffer(buffer, dtype)).reshape(value.shape))
  return tuple(result)


def change_inputs(tensors, data, device):
  device.synchronize()
  for tensor, key in zip(tensors, ('a', 'w', 'sx', 'sw', 'bias'), strict=True):
    value, buffer = data[key].tobytes(), tensor.uop.buffer
    assert len(value) == buffer.nbytes
    buffer.host.view(size=len(value), fmt='B')[:] = value


def read_output(tensor, device):
  device.synchronize()
  buffer, chunks = tensor.uop.buffer, []
  offset, remaining = buffer.offset, tensor.nbytes()
  for address, size in buffer.base.get_storage().meta.mapping.paddrs:
    if offset >= size:
      offset -= size
      continue
    count = min(remaining, size-offset)
    for start in range(0, count, 65536):
      view = device.iface.pci_dev.map_bar(bar=device.iface.vram_bar, off=address+offset+start, size=min(count-start, 65536))
      chunks.append(bytes(view[:]))
    remaining -= count
    offset = 0
    if not remaining: break
  assert remaining == 0
  return np.frombuffer(b''.join(chunks), dtype=np.uint16).reshape(tensor.shape)


def poison(outputs, device):
  from tinygrad.device import Buffer
  device.synchronize()
  # Keep each staging buffer alive until the normal copy completes.
  for output in outputs:
    buffer = output.uop.buffer.ensure_allocated()
    source = Buffer('CPU', buffer.nbytes, initial_value=bytes([0xa5])*buffer.nbytes)
    buffer.copy_from(source)
    device.synchronize()


def operation(shape, with_bias, fn, tensors, out=None):
  from tinygrad import Tensor, dtypes
  a, w, sx, sw, bias = tensors
  if out is None: out = Tensor.empty(shape[0], shape[1], device=a.device, dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out, a, w, sx, sw, *((bias,) if with_bias else ()), fxn=fn)[0].realize()


def queue_proof(batch, queues):
  """Prove actual storage identity and ordered barriers before HCQ encoding.

  HCQ's optional buffer metadata is empty for internal ALLOC arguments. Observe
  the actual queue builder instead, and keep the shared output a JIT PARAM.
  Output argument indices come from ProgramInfo, never an assumed slot zero.
  """
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  from tinygrad.uop.ops import Ops
  assert len(batch) == CALLS and len(queues) == 1
  spans, mappings, io = [], [], []
  calls = [c for c, _, _ in batch]
  assert len({(d, q) for _, d, q in batch}) == 1
  for call in calls:
    assert call.body.op is Ops.PROGRAM
    outs, ins = get_call_outs_ins(call)
    assert len(outs) == 1, 'Expected exactly one matrix output'
    value = call.src[1+outs[0]]
    base, lane, offset = unwrap_lane(value)
    assert base.op is Ops.PARAM and lane is None, 'Output must be an explicit unsharded JIT input'
    spans.append((base, offset, value.max_numel()*value.dtype.itemsize))
    args = [unwrap_lane(call.src[1+g]) for g in call.body.arg.globals]
    assert all(b.op is Ops.PARAM and lane is None for b, lane, _ in args), 'Every matrix buffer must be a JIT PARAM'
    mappings.append(tuple(b.arg.slot for b, _, _ in args))
    io.append((outs, ins))
  assert len(set(spans)) == 1, 'All 40 calls must write the same shared output storage span'
  assert len(set(mappings)) == len(set(io)) == 1
  ordered = []
  for cmd in next(iter(queues.values())):
    if cmd.op is Ops.CALL and cmd.body.op is Ops.PROGRAM: ordered.append(cmd)
    elif cmd.op is Ops.CALL and cmd.body.op is Ops.CUSTOM_FUNCTION and cmd.body.arg.name == 'exec_barrier': ordered.append('barrier')
  expected = [item for i, c in enumerate(calls) for item in ((c,) if i == 0 else ('barrier', c))]
  assert ordered == expected, 'Every adjacent matrix call must have an ordered execution barrier'
  return {'captured_calls': len(calls), 'output_input_slot': spans[0][0].arg.slot, 'output_byte_offset': spans[0][1],
          'output_span_bytes': spans[0][2], 'output_argument_indices': list(io[0][0]),
          'input_argument_indices': list(io[0][1]), 'kernel_global_input_slots': list(mappings[0]),
          'ordered_execution_barriers': CALLS-1, 'actual_queue_builder_observed': True}


def prove_cpu_queue(calls, device='CPU'):
  """Run pinned HCQ dependency construction on compiled CPU calls; no GPU opens."""
  from tinygrad.runtime.support.hcq2 import _build_queues
  batch = [(c, device, 'COMPUTE:0') for c in calls]
  ctx = SimpleNamespace(batch=batch, calls={(device, 'COMPUTE:0'): list(range(len(calls)))},
                        peers={(device, 'COMPUTE:0'): set()}, queues={device: ['COMPUTE:0']}, stamps=lambda *_: ())
  return queue_proof(batch, _build_queues(ctx))


@contextmanager
def observe_serial_queue(proofs):
  from tinygrad.runtime.support import hcq2
  original = hcq2._build_queues
  def build(ctx):
    queues = original(ctx)
    if len(ctx.batch) == CALLS: proofs.append(queue_proof(ctx.batch, queues))
    return queues
  hcq2._build_queues = build
  try: yield
  finally: hcq2._build_queues = original


class SerialJit:
  def __init__(self, shape, with_bias, fn, count):
    from tinygrad import TinyJit
    self.shape, self.count, self.output, self.proofs = shape, count, None, []
    def execute(*arguments):
      tensors, out = arguments[:-1], arguments[-1]
      for _ in range(count): out = operation(shape, with_bias, fn, tensors, out)
      return (out,)
    self.inner = TinyJit(execute)

  @property
  def captured(self): return self.inner.captured

  def __call__(self, *tensors):
    from tinygrad import Tensor, dtypes
    if self.output is None:
      self.output = Tensor.empty(self.shape[0], self.shape[1], device=tensors[0].device, dtype=dtypes.bfloat16).realize()
      self.output.uop.buffer.ensure_allocated()
    if self.count == CALLS and self.inner.cnt == 1:
      with observe_serial_queue(self.proofs): result = self.inner(*tensors, self.output)
    else: result = self.inner(*tensors, self.output)
    assert result[0].uop.buffer.base is self.output.uop.buffer.base
    assert result[0].uop.buffer.offset == self.output.uop.buffer.offset
    return result


def make_jit(shape, with_bias, fn, count): return SerialJit(shape, with_bias, fn, count)


def captured_kernels(jit):
  names = []
  for call in jit.captured._linear.src:
    aux = call.without_after.arg.aux
    if not hasattr(aux, 'kernels'): raise AssertionError('Expected an HCQ graph with explicit matrix kernel records')
    names.extend(str(k[1]) for k in aux.kernels)
  return names


def shared_output_slot(jit):
  assert len(jit.proofs) == 1, 'Expected one observed 40-call HCQ queue'
  proof = jit.proofs[0]
  assert proof['output_input_slot'] == 5 and proof['ordered_execution_barriers'] == CALLS-1
  records = []
  for call in jit.captured._linear.src:
    for kernel in call.without_after.arg.aux.kernels:
      records.append(kernel)
      assert list(kernel[5]) == proof['kernel_global_input_slots'], 'HCQ metadata does not match the observed matrix arguments'
      assert tuple(kernel[6][0]) == tuple(proof['output_argument_indices'])
      assert tuple(kernel[6][1]) == tuple(proof['input_argument_indices'])
  assert len(records) == CALLS
  return proof['output_input_slot']


def exact_output(output, expected, device):
  actual = read_output(output, device)
  assert np.isfinite(unpack_bf16(actual)).all()
  np.testing.assert_array_equal(actual, expected)


def run_case(name, shape, with_bias, fns, resources, rounds, device):
  from amd_gemm import gemm, unpack
  record = {'case': name, 'shape_mnk': [shape[0], shape[1], shape[2]], 'shape_order': 'M,N,K', 'bias': with_bias,
            'combined_qkv_diagnostic': name == 'qkv-combined', 'variants': {}, 'timing': []}
  integer = name in CONTROLS
  data = fixture(shape, 0, integer)
  tensors = upload(data)
  singles = {}
  for label, fn in fns.items():
    record['variants'][label] = dict(resources[label], checks=[])
    if resources[label]['status'] != 'compiled': continue
    jit = make_jit(shape, with_bias, fn, 1)
    for _ in range(3): jit(*tensors)
    assert len(captured_kernels(jit)) == 1
    singles[label] = jit
  assert 'original' in singles
  previous = None
  for state in range(4):
    data = fixture(shape, state, integer)
    change_inputs(tensors, data, device)
    poison(singles['original'].captured.ret, device)
    expected = read_output(singles['original'](*tensors)[0], device)
    assert np.isfinite(unpack_bf16(expected)).all()
    if integer: np.testing.assert_array_equal(expected, integer_reference(data, with_bias))
    if previous is not None: assert not np.array_equal(previous, expected), f'Mutation {state} did not change reference output'
    previous = expected.copy()
    if state == 0:
      # Confirm the packed matrix-only baseline agrees with the public production wrapper.
      a, w, sx, sw, bias = tensors
      public = gemm(unpack(a)[:shape[0]], unpack(w)[:shape[1]], (sx, sw), bias if with_bias else None).realize()
      exact_output(public, expected, device)
      record['production_gemm_exact'] = True
    for label in list(singles):
      poison(singles[label].captured.ret, device)
      try: exact_output(singles[label](*tensors)[0], expected, device)
      except AssertionError as exc:
        if label == 'original': raise
        record['variants'][label].update(status='incorrect', failed_state=state, error=str(exc)[:800])
        del singles[label]
        continue
      record['variants'][label]['checks'].append({'state': state, 'exact_bf16': True, 'independent_numpy': integer,
                                                'reference_sha256': sha(expected.tobytes())})
  if integer:
    record['controls_only'] = True
    return record
  batches = {}
  for label in singles:
    jit = make_jit(shape, with_bias, fns[label], CALLS)
    for _ in range(3): jit(*tensors)
    names = captured_kernels(jit)
    assert len(names) == CALLS and all('alpamayo_fp8_gemm' in n for n in names), names
    output_slot = shared_output_slot(jit)
    batches[label] = jit
    record['variants'][label]['captured_matrix_calls'] = len(names)
    record['variants'][label]['shared_output_slot'] = output_slot
    record['variants'][label]['serial_write_dependencies'] = CALLS-1
    record['variants'][label]['serial_queue_proof'] = jit.proofs[0]
  # Change inputs and scales after capture. Check the last output of the serial batch.
  data = fixture(shape, 4)
  change_inputs(tensors, data, device)
  poison(singles['original'].captured.ret, device)
  expected = read_output(singles['original'](*tensors)[0], device)
  assert not np.array_equal(previous, expected)
  for label in list(batches):
    poison(batches[label].captured.ret, device)
    try:
      for out in batches[label](*tensors): exact_output(out, expected, device)
    except AssertionError as exc:
      if label == 'original': raise
      record['variants'][label].update(status='incorrect-batch-replay', error=str(exc)[:800])
      del batches[label]
      continue
    record['variants'][label]['batch_mutated_input_scales_exact'] = True
  def measure(jit):
    device.synchronize()
    start = time.perf_counter()
    jit(*tensors)
    device.synchronize()
    return (time.perf_counter()-start)*1000
  # Each pair uses 40 kernels per graph, then one explicit completion fence.
  for label in batches:
    if label == 'original': continue
    pairs = []
    for iteration in range(rounds):
      order = [('original', batches['original']), ('candidate', batches[label])]
      if iteration % 2: order.reverse()
      pair = {key: measure(jit) for key, jit in order}
      pairs.append(pair)
    base = statistics.median(p['original'] for p in pairs)
    candidate = statistics.median(p['candidate'] for p in pairs)
    record['timing'].append({'variant': label, 'paired_batch_ms': pairs, 'matrix_calls_per_batch': CALLS,
                            'explicit_completion_fences_per_batch': 1, 'original_amortized_ms': base/CALLS,
                            'candidate_amortized_ms': candidate/CALLS, 'candidate_over_original': candidate/base,
                            'host_inclusive': True, 'device_only': False, 'warm_repeated_weights': True})
  return record


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=('compile', 'cpu-check', 'gpu'), default='compile')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--cases', default=','.join(CASES))
  parser.add_argument('--tiles', default=','.join(f'{m}x{n}' for m, n in TILES))
  parser.add_argument('--bias', choices=('none', 'both'), default='none')
  parser.add_argument('--rounds', type=int, default=5)
  parser.add_argument('--wgp', action='store_true')
  args = parser.parse_args()
  selected = args.cases.split(',')
  tiles = [tuple(map(int, t.split('x'))) for t in args.tiles.split(',')]
  assert selected and len(selected) <= len(CASES) and len(set(selected)) == len(selected) and all(n in CASES for n in selected)
  assert 1 <= len(tiles) <= 6 and len(set(tiles)) == len(tiles) and all(t in TILES for t in tiles)
  assert 3 <= args.rounds <= 7
  os.environ.update(BEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  if args.mode != 'gpu': os.environ['DEV'] = 'CPU'
  if args.mode == 'compile': os.environ['ALLOW_DEVICE_USAGE'] = '0'
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  import tinygrad
  report = {'mode': args.mode, 'source_sha256': {p.name: sha(p.read_bytes()) for p in
            (Path(__file__), HERE/'tiled-gemm.py', HERE/'expert-kernels.py', HERE.parent/'amd_gemm.py')},
            'tinygrad_revision': subprocess.check_output(['git', '-C', str(Path(tinygrad.__file__).parent.parent), 'rev-parse', 'HEAD'], text=True).strip(),
            'wgp': args.wgp, 'matrix_calls_per_timing_batch': CALLS, 'cases': [], 'gpu_executed': False,
             'timing_scope': 'One captured 40-matrix graph with shared-output WAW dependencies; host-inclusive wall time divided by 40. Packing and uploads excluded.',
            'settings': {'tiles': tiles, 'cases': selected, 'rounds': args.rounds, 'bias': args.bias},
            'runtime_environment': {k: os.environ.get(k) for k in ('DEV', 'CACHEDB', 'TC_OPT', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL')}}
  if args.mode == 'cpu-check':
    report['checks'] = cpu_checks()
    report['complete'] = True
    write_json(args.output, report)
    print(json.dumps({'cpu_checks': len(report['checks']), 'gpu_executed': False}))
    return
  if args.mode == 'compile':
    renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
    if args.wgp: renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  else:
    from tinygrad.device import Compiled
    Compiled.wait_timeout_ms = 10000
    local('prefill-kernels').fix_dispatch_packet()
    device = Device[Device.DEFAULT]
    assert device.arch == 'gfx1200' and device.is_usb and isinstance(device.renderer, AMDLLVMRenderer)
    if args.wgp: local('wgp_compiler').install(device)
    renderer = device.renderer
  fns = functions(renderer, tiles)
  control_rejections = {}
  try:
    for name, shape in (*CONTROLS.items(), *((n, CASES[n]) for n in selected)):
      bias_cases = (False, True) if name in CONTROLS or args.bias == 'both' or args.mode == 'compile' else (False,)
      for with_bias in bias_cases:
        report['active_case'] = {'name': name, 'bias': with_bias}
        write_json(args.output, report)
        print(json.dumps({'starting_case': name, 'bias': with_bias}), flush=True)
        resources = {label: compiler_record(shape, with_bias, fn, renderer) for label, fn in fns.items()}
        assert resources['original']['status'] == 'compiled', resources['original']
        for label, rejection in control_rejections.items():
          resources[label].update(status='independent-control-rejected', control_failure=rejection)
        if args.mode == 'compile':
          row = {'case': name, 'shape_mnk': shape, 'shape_order': 'M,N,K', 'bias': with_bias, 'variants': resources}
        else:
          report['gpu_executed'] = True
          row = run_case(name, shape, with_bias, fns, resources, args.rounds, device)
          if name in CONTROLS:
            for label, variant in row['variants'].items():
              if variant['status'] != 'compiled' and label not in control_rejections:
                control_rejections[label] = {'case': name, 'bias': with_bias, 'status': variant['status'],
                                             'error': variant.get('error', '')}
            report['independent_control_rejections'] = control_rejections
        report['cases'].append(row)
        write_json(args.output, report)
        print(json.dumps({'case': name, 'shape': shape, 'bias': with_bias,
                          'statuses': {k: v['status'] for k, v in row['variants'].items()}, 'timing': row.get('timing', [])}), flush=True)
    report['complete'] = True
    report.pop('active_case', None)
    report['summary'] = {'cases': len(report['cases']), 'gpu_executed': report['gpu_executed'],
                         'compiled_variants': sum(v['status'] == 'compiled' for c in report['cases'] for v in c['variants'].values()),
                         'rejected_variants': sum(v['status'] != 'compiled' for c in report['cases'] for v in c['variants'].values()),
                         'pipeline_validated': False}
    if args.mode == 'compile': assert not Device._opened_devices
    write_json(args.output, report)
  except BaseException as exc:
    report.update(complete=False, failure=f'{type(exc).__name__}: {exc}')
    write_json(args.output, report)
    raise


if __name__ == '__main__': main()
