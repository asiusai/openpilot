"""Bounded QKV fusion checks. Default mode compiles on CPU without device access."""
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
CASES = {'expert': (64, 512, (2048, 1024, 1024)), 'prefill-tail': (2027, 2048, (2048, 1024, 1024)),
         'prefill-padded': (2112, 2048, (2048, 1024, 1024))}
CONTROL = (17, 32, (64, 64, 64))
CALLS = 40
KEYS = ('raw', 'packed', 'wq', 'wk', 'wv', 'wc', 'sx', 'sq', 'sk', 'sv', 'sc', 'bq', 'bk', 'bv', 'bc')


def local(name):
  spec = importlib.util.spec_from_file_location('agent_qkv_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def digest(data): return hashlib.sha256(data).hexdigest()


def save(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name+'.new')
  tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
  tmp.replace(path)


def provenance():
  import tinygrad
  files = [HERE/'agent-qkv.py', HERE/'agent-qkv-host.py', HERE/'agent-qkv-run.py', Path(__file__), HERE/'tiled-gemm.py', HERE.parent/'amd_gemm.py', HERE.parent/'model.py',
           HERE/'quantize-packed.py', HERE/'expert-kernels.py', HERE/'batched-vision.py', HERE/'quantize-reduction.py']
  return {'source_sha256': {p.name: digest(p.read_bytes()) for p in files},
          'tinygrad_revision': subprocess.check_output(['git', '-C', str(Path(tinygrad.__file__).parent.parent), 'rev-parse', 'HEAD'], text=True).strip()}


def decode(bits):
  e, f = ((bits >> 3) & 15).astype(np.int32), (bits & 7).astype(np.float32)
  values = np.where(e == 0, f*2.0**-9, (1+f/8)*np.exp2(e-7))
  assert np.all((bits & 127) != 127)
  return np.where(bits & 128, -values, values).astype(np.float32)


def bf16(values):
  values = np.asarray(values, np.float32)
  assert np.isfinite(values).all()
  bits = values.view(np.uint32)
  return ((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf32(values): return (values.astype(np.uint32) << 16).view(np.float32)


def pack_numpy(values):
  m, k = values.shape
  out = np.zeros((((m+15)//16)*16, k), values.dtype)
  c = np.arange(k, dtype=np.int64)[None, :]
  for begin in range(0, m, 64):
    r = np.arange(begin, min(m, begin+64), dtype=np.int64)[:, None]
    addr = (r//16*(k//16)+c//16)*256+(c%16//4)*64+(r%16)*4+c%4
    out.reshape(-1)[addr] = values[begin:begin+64]
  return out


def fixture(shape, state, integer=False):
  m, k, sizes = shape
  def raw(seed, dimensions):
    rng = np.random.default_rng(seed)
    if integer:
      v = rng.integers(-7, 8, dimensions)
      codes = np.searchsorted(decode(np.arange(127, dtype=np.uint8)), np.abs(v)).astype(np.uint8)
      return codes | ((v < 0).astype(np.uint8) << 7)
    v = rng.integers(0, 127, dimensions, dtype=np.uint8) | (rng.integers(0, 2, dimensions, dtype=np.uint8) << 7)
    v.reshape(-1)[:8] = np.array([0, 128, 1, 129, 56, 184, 126, 254], np.uint8)
    return v
  a = raw(391 + (state > 0) + (state == 6), (m, k))
  weights = [raw(711+i+(11 if state == 6 else 0), (n, k)) for i, n in enumerate(sizes)]
  sx = .125 if state < 2 else .25 if integer else .071234569
  scales = [.25, .5, .125]
  for i in range(3):
    if state >= 3+i: scales[i] = [1., .25, .5][i] if integer else [.01933221, .03192311, .02378129][i]
  bias = [bf16(np.random.default_rng(1031+i+(10 if state == 6 else 0)).integers(-8, 9, n).astype(np.float32)/8) for i, n in enumerate(sizes)]
  data = dict(raw=a, packed=pack_numpy(a), wc=pack_numpy(np.concatenate(weights)), sx=np.array([sx], np.float32),
              sc=np.concatenate([np.full(n, s, np.float32) for n, s in zip(sizes, scales)]), bc=np.concatenate(bias))
  for i, c in enumerate('qkv'):
    data['w'+c], data['s'+c], data['b'+c] = pack_numpy(weights[i]), np.array([scales[i]], np.float32), bias[i]
  return data, weights


def integer_reference(data, weights, bias):
  a = decode(data['raw']).astype(np.int64)
  result = []
  for w, c in zip(weights, 'qkv'):
    sums = a @ decode(w).astype(np.int64).T
    assert np.max(np.abs(sums)) < 2**24
    v = sums.astype(np.float32)*(data['sx'][0]*data['s'+c][0])
    if bias: v += bf32(data['b'+c])
    result.append(bf16(v))
  return tuple(result)


def upload_array(value, dtype, device):
  from tinygrad import Tensor
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  b = Buffer(device, value.nbytes, options=BufferSpec(cpu_access=True)).ensure_allocated()
  b.host.view(size=b.nbytes, fmt='B')[:] = value.tobytes()
  return Tensor(UOp.from_buffer(b, dtype)).reshape(value.shape)


def upload(data, device):
  from tinygrad import dtypes
  return tuple(upload_array(data[key], dtypes.bfloat16 if key.startswith('b') else dtypes.float32 if key.startswith('s') else dtypes.fp8e4m3,
                            device) for key in KEYS)


def change(inputs, data, device):
  device.synchronize()
  for tensor, key in zip(inputs, KEYS):
    b = tensor.uop.buffer
    assert b.nbytes == data[key].nbytes
    b.host.view(size=b.nbytes, fmt='B')[:] = data[key].tobytes()


def bytes_from(tensor, device):
  device.synchronize()
  b, chunks = tensor.uop.buffer, []
  offset, remaining = b.offset, tensor.nbytes()
  for address, size in b.base.get_storage().meta.mapping.paddrs:
    if offset >= size:
      offset -= size
      continue
    count = min(remaining, size-offset)
    for start in range(0, count, 65536):
      chunks.append(bytes(device.iface.pci_dev.map_bar(bar=device.iface.vram_bar, off=address+offset+start,
                                                      size=min(count-start, 65536))[:]))
    remaining -= count
    offset = 0
    if not remaining: break
  assert remaining == 0
  return b''.join(chunks)


def outputs(jit, inputs, sizes, device, fused):
  from tinygrad.device import Buffer
  device.synchronize()
  for t in jit.captured.ret:
    b = t.uop.buffer.ensure_allocated()
    staging = Buffer('CPU', b.nbytes, initial_value=bytes([0xa5])*b.nbytes)
    b.copy_from(staging)
    device.synchronize()
  tensors = jit(*inputs)
  values = tuple(np.frombuffer(bytes_from(t, device), np.uint16).reshape(t.shape) for t in tensors)
  assert all(np.isfinite(bf32(v)).all() for v in values)
  return tuple(np.split(values[0], np.cumsum(sizes)[:-1], axis=1)) if fused else values


def exact(actual, expected):
  for a, b in zip(actual, expected, strict=True): np.testing.assert_array_equal(a, b)


def make_jit(shape, bias, fused, boundary, count, original_fn, fused_fn):
  from tinygrad import Tensor
  rows, width, sizes = shape
  outputs_count = 1 if fused else 3
  def run(*arguments):
    d = dict(zip(KEYS, arguments[:len(KEYS)]))
    outs = arguments[len(KEYS):len(KEYS)+outputs_count]
    a = arguments[-1] if boundary == 'rowmajor-with-pack' else d['packed']
    for _ in range(count):
      if boundary == 'rowmajor-with-pack':
        # The production pack permutation, assigned directly to the reusable output.
        # Keeping pack()'s internal contiguous() here would add a second copy per group.
        view = d['raw'].pad(((0, (-rows) % 16), (0, 0))).reshape(-1, 16, width//16, 4, 4).permute(0, 2, 3, 1, 4).reshape(-1, width)
        a = a.assign(view).realize()
      current = []
      for out, c in zip(outs, ('c',) if fused else 'qkv'):
        current.append(Tensor.custom_kernel(out, a, d['w'+c], d['sx'], d['s'+c], *((d['b'+c],) if bias else ()),
                                            fxn=fused_fn if fused else original_fn)[0].realize())
      outs = tuple(current)
    return outs
  return QKVJit(run, shape, fused, boundary, count)


def queue_proof(batch, queues, fused, count, boundary):
  """Prove serial groups from actual storage spans and HCQ execution barriers."""
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  matrix_slots = tuple(range(len(KEYS), len(KEYS)+(1 if fused else 3)))
  pack_slot = len(KEYS)+len(matrix_slots)
  slots_per_group = ((pack_slot,) if boundary == 'rowmajor-with-pack' else ())+matrix_slots
  assert len(batch) == count*len(slots_per_group) and len(queues) == 1
  assert len({(d, q) for _, d, q in batch}) == 1
  spans, descriptions, expected = {}, [], []
  for i, (call, _, _) in enumerate(batch):
    assert call.body.op is Ops.PROGRAM
    outs, ins = get_call_outs_ins(call)
    assert len(outs) == 1, 'Expected one output argument per packing or matrix call'
    output = call.src[1+outs[0]]
    base, lane, offset = unwrap_lane(output)
    assert base.op is Ops.PARAM and lane is None, 'Output must be an explicit unsharded JIT PARAM'
    wanted = slots_per_group[i % len(slots_per_group)]
    assert base.arg.slot == wanted, 'QKV group must reuse the expected shared output storage'
    span = (base, offset, output.max_numel()*output.dtype.itemsize)
    assert wanted not in spans or spans[wanted] == span, 'QKV output storage span changed'
    spans[wanted] = span
    args = [unwrap_lane(call.src[1+g]) for g in call.body.arg.globals]
    assert all(b.op is Ops.PARAM and lane is None for b, lane, _ in args), 'Every kernel buffer must be a JIT PARAM'
    descriptions.append({'output_argument_indices': list(outs), 'input_argument_indices': list(ins),
                         'global_input_slots': [b.arg.slot for b, _, _ in args]})
    # Packed input: one barrier before each subsequent QKV group.
    # Live packing: a barrier before each subsequent pack, then before its first GEMM.
    first = i % len(slots_per_group) == 0
    first_matrix = boundary == 'rowmajor-with-pack' and i % len(slots_per_group) == 1
    if (first and i > 0) or first_matrix: expected.append('barrier')
    expected.append(call)
  ordered = []
  for cmd in next(iter(queues.values())):
    if cmd.op is Ops.CALL and cmd.body.op is Ops.PROGRAM: ordered.append(cmd)
    elif cmd.op is Ops.CALL and cmd.body.op is Ops.CUSTOM_FUNCTION and cmd.body.arg.name == 'exec_barrier': ordered.append('barrier')
  assert ordered == expected, 'QKV groups or their activation packing lack ordered execution barriers'
  return {'qkv_groups': count, 'matrix_calls': count*len(matrix_slots),
          'packing_calls': count if boundary == 'rowmajor-with-pack' else 0,
          'reused_output_slots': list(matrix_slots), 'packing_output_slot': pack_slot if boundary == 'rowmajor-with-pack' else None,
          'ordered_execution_barriers': count*2-1 if boundary == 'rowmajor-with-pack' else count-1,
          'output_span_bytes': {str(k): s[2] for k, s in spans.items()}, 'kernel_arguments': descriptions,
          'actual_queue_builder_observed': True}


def cpu_queue_proof(calls, fused, count, boundary, device='CPU'):
  from tinygrad.runtime.support.hcq2 import _build_queues
  batch = [(c, device, 'COMPUTE:0') for c in calls]
  ctx = SimpleNamespace(batch=batch, calls={(device, 'COMPUTE:0'): list(range(len(calls)))},
                        peers={(device, 'COMPUTE:0'): set()}, queues={device: ['COMPUTE:0']}, stamps=lambda *_: ())
  return queue_proof(batch, _build_queues(ctx), fused, count, boundary)


@contextmanager
def observe_queue(jit):
  from tinygrad.runtime.support import hcq2
  original = hcq2._build_queues
  expected_calls = jit.count*((1 if jit.fused else 3)+(jit.boundary == 'rowmajor-with-pack'))
  def build(ctx):
    result = original(ctx)
    if len(ctx.batch) == expected_calls: jit.proofs.append(queue_proof(ctx.batch, result, jit.fused, jit.count, jit.boundary))
    return result
  hcq2._build_queues = build
  try: yield
  finally: hcq2._build_queues = original


class QKVJit:
  def __init__(self, run, shape, fused, boundary, count):
    from tinygrad import TinyJit
    self.inner, self.shape, self.fused, self.boundary, self.count = TinyJit(run), shape, fused, boundary, count
    self.outputs, self.packed, self.proofs = None, None, []

  @property
  def captured(self): return self.inner.captured

  def __call__(self, *inputs):
    from tinygrad import Tensor, dtypes
    if self.outputs is None:
      rows, width, sizes = self.shape
      self.outputs = tuple(Tensor.empty(rows, n, device=inputs[0].device, dtype=dtypes.bfloat16).realize()
                           for n in ((sum(sizes),) if self.fused else sizes))
      for out in self.outputs: out.uop.buffer.ensure_allocated()
      if self.boundary == 'rowmajor-with-pack':
        self.packed = Tensor.empty((rows+15)//16*16, width, device=inputs[0].device, dtype=dtypes.fp8e4m3).realize()
        self.packed.uop.buffer.ensure_allocated()
    arguments = (*inputs, *self.outputs, *((self.packed,) if self.packed is not None else ()))
    if self.inner.cnt == 1:
      with observe_queue(self): result = self.inner(*arguments)
    else: result = self.inner(*arguments)
    for out, stable in zip(result, self.outputs, strict=True):
      assert out.uop.buffer.base is stable.uop.buffer.base and out.uop.buffer.offset == stable.uop.buffer.offset
    return result


def metadata(jit, fused, count, boundary):
  kernels = [k for c in jit.captured._linear.src for k in c.without_after.arg.aux.kernels]
  assert len(jit.proofs) == 1, 'Expected one observed serial QKV queue'
  proof = jit.proofs[0]
  assert proof['qkv_groups'] == count
  assert len(kernels) == len(proof['kernel_arguments'])
  for kernel, expected in zip(kernels, proof['kernel_arguments'], strict=True):
    assert list(kernel[5]) == expected['global_input_slots'], 'HCQ buffer metadata differs from actual call arguments'
    assert list(kernel[6][0]) == expected['output_argument_indices']
    assert list(kernel[6][1]) == expected['input_argument_indices']
  assert sum('alpamayo_fp8_gemm' in str(k[1]) for k in kernels) == proof['matrix_calls']
  return proof


def clone(fn, renderer):
  class Devices:
    DEFAULT = 'agent-qkv-offline'
    def __getitem__(self, key): return SimpleNamespace(renderer=renderer)
  result = FunctionType(fn.__code__, dict(fn.__globals__, Device=Devices()), fn.__name__, fn.__defaults__)
  result.__kwdefaults__ = fn.__kwdefaults__
  return result


def compile_one(shape, sizes, bias, column, fn, renderer, label='q', calls=None):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp
  m, k = shape
  n = sum(sizes)
  args = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate([
    ((m, n), dtypes.bfloat16), (((m+15)//16*16, k), dtypes.fp8e4m3), ((n, k), dtypes.fp8e4m3),
    ((1,), dtypes.float32), ((n if column else 1,), dtypes.float32)])]
  if bias: args.append(UOp.placeholder((n,), dtypes.bfloat16, 5))
  p = to_program(fn(*args), renderer)
  if calls is not None:
    c = 'c' if column else label
    out_slot = len(KEYS)+(0 if column else 'qkv'.index(label))
    slots = (out_slot, KEYS.index('packed'), KEYS.index('w'+c), KEYS.index('sx'), KEYS.index('s'+c), KEYS.index('b'+c))
    symbolic = [UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip(slots, args)]
    calls['fused' if column else label] = p.call(*symbolic)
  binary = p.src[3].arg
  meta = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
  keys = ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
          'group_segment_fixed_size', 'wavefront_size')
  resources = {k: int(v) for k, v in re.findall(r'\.('+'|'.join(keys)+r'):\s*(\d+)', meta)}
  assert set(keys) <= resources.keys()
  return {'global_size': p.arg.global_size, 'local_size': p.arg.local_size, 'workgroups': math.prod(p.arg.global_size),
          'resources': resources, 'binary_sha256': digest(binary), 'binary_bytes': len(binary)}


def gpu_case(name, shape, bias, rounds, device, fusion, original_fn, fused_fn):
  from tinygrad import Tensor, TinyJit, dtypes
  m, width, sizes = shape
  integer = name == 'integer'
  data, weights = fixture(shape, 0, integer)
  inputs = upload(data, device.device)
  # Exercise the installer's real one-time concatenation, including already-packed logical views.
  from amd_gemm import pack, unpack
  logical = [unpack(pack(upload_array(w, dtypes.fp8e4m3, device.device)).realize()) for w in weights]
  d = dict(zip(KEYS, inputs))
  start = time.perf_counter()
  prepared = fusion.prepare(logical, [d['s'+c] for c in 'qkv'], [d['b'+c] if bias else None for c in 'qkv'])
  device.synchronize()
  setup_ms = (time.perf_counter()-start)*1000
  assert bytes_from(prepared[0], device) == data['wc'].tobytes()
  assert bytes_from(prepared[1], device) == data['sc'].tobytes()
  if bias: assert bytes_from(prepared[2], device) == data['bc'].tobytes()
  record = {'name': name, 'shape_m_k_sizes': shape, 'bias': bias, 'setup_exact': True, 'setup_ms': setup_ms, 'checks': [], 'timings': []}
  # Also capture the actual installer projection and its split views. Materializing
  # these views is for correctness reads only; these copies are never timed.
  packed_type = local('quantize-packed').PackedActivation
  h = Tensor.empty(1, m, width, device=device.device, dtype=dtypes.bfloat16)
  use_packed = name not in ('integer', 'expert')
  def projected(raw, packed, scale):
    value = packed_type(packed, m, width) if use_packed else raw
    parts = fusion.project(h, (value, scale), prepared, packed_type)
    return tuple(p.reshape(m, n).contiguous().realize() for p, n in zip(parts, sizes))
  installer_jit = TinyJit(projected)
  for _ in range(3): installer_jit(d['raw'], d['packed'], d['sx'])
  record['installer_projection_checks'] = []
  boundaries = ('packed', 'rowmajor-with-pack') if name in ('integer', 'expert') else ('packed',)
  jits = {}
  for boundary in boundaries:
    for fused in (False, True):
      jit = make_jit(shape, bias, fused, boundary, 1, original_fn, fused_fn)
      for _ in range(3): jit(*inputs)
      metadata(jit, fused, 1, boundary)
      jits[boundary, fused] = jit
  previous = None
  for state in range(7):
    data, weights = fixture(shape, state, integer)
    change(inputs, data, device)
    expected = outputs(jits['packed', False], inputs, sizes, device, False)
    if integer: exact(expected, integer_reference(data, weights, bias))
    if previous is not None:
      changed = [not np.array_equal(a, b) for a, b in zip(previous, expected)]
      assert any(changed), 'Mutation did not affect output'
      if state in (3, 4, 5): assert changed == [i == state-3 for i in range(3)], 'Distinct projection-scale control failed'
    previous = tuple(v.copy() for v in expected)
    for (boundary, fused), jit in jits.items(): exact(outputs(jit, inputs, sizes, device, fused), expected)
    if state < 3:
      exact(outputs(installer_jit, (d['raw'], d['packed'], d['sx']), sizes, device, False), expected)
      record['installer_projection_checks'].append({'state': state, 'exact_bf16': True, 'packed_activation': use_packed})
    record['checks'].append({'state': state, 'exact_bf16': True, 'integer_oracle': integer,
                             'reference_sha256': [digest(v.tobytes()) for v in expected]})
  if integer: return record
  for boundary in boundaries:
    # Each boundary must capture a different state from its later replay check.
    change(inputs, fixture(shape, 6)[0], device)
    batches = {}
    for fused in (False, True):
      jit = make_jit(shape, bias, fused, boundary, CALLS, original_fn, fused_fn)
      for _ in range(3): jit(*inputs)
      capture = metadata(jit, fused, CALLS, boundary)
      batches[fused] = jit
      record.setdefault('captures', []).append(dict(capture, fused=fused, boundary=boundary))
    # Return to an earlier input after capture to reject stale arguments and packing.
    data, weights = fixture(shape, 1)
    change(inputs, data, device)
    expected = outputs(jits['packed', False], inputs, sizes, device, False)
    assert any(not np.array_equal(a, b) for a, b in zip(previous, expected))
    for fused, jit in batches.items(): exact(outputs(jit, inputs, sizes, device, fused), expected)
    pairs = []
    for iteration in range(rounds):
      pair = {}
      for fused in ((False, True) if iteration % 2 == 0 else (True, False)):
        device.synchronize()
        start = time.perf_counter()
        batches[fused](*inputs)
        device.synchronize()
        pair['fused' if fused else 'separate'] = (time.perf_counter()-start)*1000
      pairs.append(pair)
    separate, fused = (statistics.median(p[k] for p in pairs)/CALLS for k in ('separate', 'fused'))
    record['timings'].append({'boundary': boundary, 'paired_batch_ms': pairs, 'qkv_groups_per_batch': CALLS,
                              'separate_ms_per_qkv': separate, 'fused_ms_per_qkv': fused, 'fused_over_separate': fused/separate,
                              'host_inclusive': True, 'device_only': False, 'warm_repeated_weights': True})
  return record


def cpu_checks(fusion):
  from tinygrad import Tensor, Device, dtypes
  from amd_gemm import pack
  checks = []
  codes = np.concatenate((np.arange(127, dtype=np.uint8), np.arange(128, 255, dtype=np.uint8)))
  converted = Tensor(codes, device='CPU').bitcast(dtypes.fp8e4m3).float().numpy()
  subnormal = ((codes & 120) == 0) & ((codes & 7) != 0)
  # Pinned dtype.py:f2f explicitly flushes CPU conversion denormals to signed zero.
  # Preserve IEEE decoding for the independent oracle; its integer inputs are normal.
  np.testing.assert_array_equal(converted[~subnormal].view(np.uint32), decode(codes[~subnormal]).view(np.uint32))
  np.testing.assert_array_equal(converted[subnormal].view(np.uint32), (codes[subnormal].astype(np.uint32) & 128) << 24)
  checks.append({'normal_and_zero_fp8_decode_exact': int((~subnormal).sum()),
                 'pinned_cpu_conversion_subnormals_flush_to_signed_zero': int(subnormal.sum())})
  for state in range(7):
    data, weights = fixture(CONTROL, state, True)
    for bias in (False, True):
      expected = integer_reference(data, weights, bias)
      a = Tensor(decode(data['raw']), device='CPU')
      wc = Tensor(np.concatenate([decode(w) for w in weights]), device='CPU')
      value = a.dot(wc.T)*(Tensor(data['sx'], device='CPU')*Tensor(data['sc'], device='CPU'))
      if bias: value += Tensor(bf32(data['bc']), device='CPU')
      actual = value.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy()
      exact(tuple(np.split(actual, np.cumsum(CONTROL[2])[:-1], axis=1)), expected)
      checks.append({'state': state, 'bias': bias, 'numpy_integer_exact': True})
  for name, shape in [('integer', CONTROL), *CASES.items()]:
    data, weights = fixture(shape, 0, name == 'integer')
    np.testing.assert_array_equal(np.concatenate([pack_numpy(w) for w in weights]), data['wc'])
    np.testing.assert_array_equal(pack(Tensor(np.concatenate(weights), device='CPU')).numpy(), data['wc'])
    checks.append({'case': name, 'packed_concat_exact': True})
  # Use the exact capture loop with a small CPU custom kernel. Packing must repeat 40 times.
  from tinygrad.uop.ops import UOp, KernelInfo
  def fill(out, a, w, sx, sw, *bias):
    r = UOp.range(out.numel(), 0)
    return out.flatten().index(r).store((a.flatten().index(0).float()+sx.flatten().index(0)).cast(out.dtype)).end(r).sink(arg=KernelInfo())
  data, _ = fixture(CONTROL, 0, True)
  inputs = upload(data, 'CPU')
  for boundary in ('packed', 'rowmajor-with-pack'):
    for fused in (False, True):
      change(inputs, fixture(CONTROL, 0, True)[0], Device['CPU'])
      jit = make_jit(CONTROL, False, fused, boundary, CALLS, fill, fill)
      for _ in range(3): jit(*inputs)
      calls = jit.captured._linear.src
      assert len(calls) == CALLS*((1 if fused else 3)+(boundary == 'rowmajor-with-pack')), (boundary, fused, len(calls))
      proof = cpu_queue_proof(calls, fused, CALLS, boundary)
      change(inputs, fixture(CONTROL, 1, True)[0], Device['CPU'])
      value = jit(*inputs)[0].bitcast(dtypes.uint16).numpy()
      wanted = decode(fixture(CONTROL, 1, True)[0]['raw'])[0, 0]+data['sx'][0]
      np.testing.assert_array_equal(value, bf16(np.full(value.shape, wanted, np.float32)))
      checks.append({'boundary': boundary, 'fused': fused, 'captured_calls': len(calls), 'mutated_capture_exact': True,
                     'serial_queue_proof': proof})
  # Compile and inspect a different argument order: output is argument 1.
  from tinygrad.codegen import to_program
  a, out = UOp.placeholder((1,), dtypes.float32, 0), UOp.placeholder((17, 64), dtypes.bfloat16, 1)
  program = to_program(fill(out, a, a, a, a), Device['CPU'].renderer)
  unusual = [program.call(UOp.param(0, dtypes.float32, 1, 'CPU'), UOp.param(15+i, dtypes.bfloat16, 17*64, 'CPU')) for i in range(3)]
  proof = cpu_queue_proof(unusual*CALLS, False, CALLS, 'packed')
  assert all(k['output_argument_indices'] == [1] for k in proof['kernel_arguments'])
  independent = list(unusual*CALLS)
  first = independent[3]
  independent[3] = first.replace(src=(first.src[0], first.src[1], UOp.param(900, dtypes.bfloat16, 17*64, 'CPU')))
  try: cpu_queue_proof(independent, False, CALLS, 'packed')
  except AssertionError as exc: assert 'shared output storage' in str(exc)
  else: raise AssertionError('Independent group output passed the serial proof')
  checks.append({'nonzero_output_argument_proven': True, 'independent_group_output_rejected': True})
  checks.extend(local('agent-qkv-host').check(fusion))
  assert not any(d.startswith('AMD') for d in Device._opened_devices)
  return checks


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode', choices=('compile', 'cpu-check', 'gpu'), default='compile')
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--cases', default=','.join(CASES))
  p.add_argument('--bias', choices=('none', 'both'), default='none')
  p.add_argument('--rounds', type=int, default=5)
  p.add_argument('--wgp', action='store_true')
  args = p.parse_args()
  names = args.cases.split(',')
  assert 1 <= len(names) <= len(CASES) and len(set(names)) == len(names) and all(n in CASES for n in names)
  assert 3 <= args.rounds <= 7
  os.environ.update(BEAM='0', JITBEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  if args.mode != 'gpu': os.environ['DEV'] = 'CPU'
  if args.mode == 'compile': os.environ['ALLOW_DEVICE_USAGE'] = '0'
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from amd_gemm import _gemm
  fusion = local('agent-qkv')
  report = {'mode': args.mode, 'provenance': provenance(), 'wgp': args.wgp, 'cases': [], 'gpu_executed': False,
            'settings': {'cases': names, 'bias': args.bias, 'rounds': args.rounds, 'calls': CALLS},
            'environment': {k: os.environ.get(k) for k in ('DEV', 'CACHEDB', 'TC_OPT', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL')}}
  if args.mode == 'cpu-check':
    report['checks'] = cpu_checks(fusion)
    report['complete'] = True
    save(args.output, report)
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
    assert device.is_usb and device.arch == 'gfx1200' and isinstance(device.renderer, AMDLLVMRenderer)
    if args.wgp: local('wgp_compiler').install(device)
    renderer = device.renderer
  original_fn = clone(_gemm, renderer)
  fused_fn = partial(clone(fusion.TILED.kernel, renderer), tile_m=4, tile_n=4, column_scale=True)
  try:
    for name, shape in [('integer', CONTROL), *((n, CASES[n]) for n in names)]:
      for bias in ((False, True) if name == 'integer' or args.mode == 'compile' or args.bias == 'both' else (False,)):
        report['active_case'] = {'name': name, 'bias': bias}
        save(args.output, report)
        print(json.dumps({'starting': name, 'bias': bias}), flush=True)
        m, k, sizes = shape
        symbolic_calls = {}
        resources = {c: compile_one((m, k), (n,), bias, False, original_fn, renderer, label=c, calls=symbolic_calls) for c, n in zip('qkv', sizes)}
        resources['fused'] = compile_one((m, k), sizes, bias, True, fused_fn, renderer, calls=symbolic_calls)
        offline = {'separate': cpu_queue_proof([symbolic_calls[c] for _ in range(CALLS) for c in 'qkv'], False, CALLS, 'packed', 'AMD'),
                   'fused': cpu_queue_proof([symbolic_calls['fused']]*CALLS, True, CALLS, 'packed', 'AMD')}
        if args.mode == 'compile': row = {'name': name, 'shape_m_k_sizes': shape, 'bias': bias}
        else:
          report['gpu_executed'] = True
          row = gpu_case(name, shape, bias, args.rounds, device, fusion, original_fn, fused_fn)
        report['cases'].append(dict(row, resources=resources, offline_serial_queue_proofs=offline))
        save(args.output, report)
    report.update(complete=True, pipeline_validated=False)
    report.pop('active_case', None)
    if args.mode == 'compile': assert not Device._opened_devices
    save(args.output, report)
    print(json.dumps({'cases': len(report['cases']), 'gpu_executed': report['gpu_executed']}))
  except BaseException as exc:
    report.update(complete=False, failure=f'{type(exc).__name__}: {exc}')
    save(args.output, report)
    raise


if __name__ == '__main__': main()
