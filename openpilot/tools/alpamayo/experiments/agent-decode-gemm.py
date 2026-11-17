"""Bounded draft-verification GEMM checks; GPU execution belongs to the parent.

Compare the original 1x4 baseline, 1x2, original 1x1, and repaired 1x1-accum.
The repair preserves the baseline's running WMMA accumulator. Strict failures
remain rejected per variant. No rejected variant enters a timing graph.
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
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TILES = ((1, 4), (1, 2), (1, 1))
ROWS = (16, 32, 48, 64)
CALLS = 40
CASES = {'q': (2048, 2048, 'self_attn.q_proj'), 'kv': (1024, 2048, 'self_attn.k_proj'),
         'fc1': (9216, 2048, 'mlp.fc1'), 'fc2': (2048, 9216, 'mlp.fc2'), 'head': (2112, 2048, None)}
CONTROLS = {'integer': (16, 64, 64), 'integer-tail': (17, 48, 32), 'sparse-outliers': (17, 48, 32)}
KEYS = ('a', 'w', 'sx', 'sw', 'bias')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def save(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name+'.new')
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
  temporary.replace(path)


def local(name):
  spec = importlib.util.spec_from_file_location('agent_decode_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def provenance():
  import tinygrad
  root = Path(tinygrad.__file__).resolve().parent.parent
  files = [Path(__file__), HERE/'agent-decode-gemm-hook.py', HERE/'tiled-gemm.py', HERE/'draft-decode.py', HERE/'lookup-decode.py',
           HERE/'reduced-vocab.py', HERE.parent/'amd_gemm.py', HERE.parent/'model.py', HERE/'agent-decode-gemm-rounding.py']
  return {'sources': {str(p.relative_to(HERE.parent)): sha(p.read_bytes()) for p in files},
          'tinygrad_revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
          'tinygrad_diff_sha256': sha(subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', 'HEAD', '--', 'tinygrad'])),
          'hcq_sha256': sha((root/'tinygrad/runtime/support/hcq2.py').read_bytes()),
          'wgp_sha256': sha((HERE/'wgp_compiler.py').read_bytes()),
          'compiler_sources': {str(p.relative_to(root)): sha(p.read_bytes()) for p in
            [root/'tinygrad/codegen/__init__.py', root/'tinygrad/renderer/llvmir.py', root/'tinygrad/runtime/support/compiler_llvm.py']}}


def decode(bits):
  b = np.asarray(bits, np.uint8)
  assert np.all((b & 127) != 127), 'Nonfinite FP8 fixture'
  e, f = ((b >> 3) & 15).astype(np.int32), (b & 7).astype(np.float32)
  value = np.where(e == 0, f*2.0**-9, (1+f/8)*np.exp2(e-7))
  return np.where(b & 128, -value, value).astype(np.float32)


def bf16(values):
  values = np.asarray(values, np.float32)
  assert np.isfinite(values).all()
  b = values.view(np.uint32)
  return ((b+np.uint32(0x7fff)+((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf32(bits): return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def encode_integer(value):
  table = decode(np.arange(127, dtype=np.uint8))
  magnitude = np.abs(value).astype(np.float32)
  index = np.searchsorted(table, magnitude).astype(np.uint8)
  np.testing.assert_array_equal(table[index], magnitude)
  return index | ((value < 0).astype(np.uint8) << 7)


def pack_numpy(values):
  rows, width = values.shape
  assert width % 16 == 0
  packed = np.zeros((((rows+15)//16)*16, width), values.dtype)
  k = np.arange(width, dtype=np.int64)[None, :]
  for start in range(0, rows, 64):
    r = np.arange(start, min(start+64, rows), dtype=np.int64)[:, None]
    address = (r//16*(width//16)+k//16)*256+(k%16//4)*64+(r%16)*4+k%4
    packed.reshape(-1)[address] = values[start:start+64]
  return packed


def unpack_numpy(packed):
  rows, width = packed.shape
  return packed.reshape(rows//16, width//16, 4, 16, 4).transpose(0, 3, 1, 2, 4).reshape(rows, width).copy()


def model_fixture(name, directory, checkpoint, vocabulary):
  """Read only the exact required cache records. Never dequantize/requantize weights."""
  index_raw = (directory/'index.json').read_bytes()
  index = json.loads(index_raw)
  assert index['format'] == 1 and (directory/'weights.bin').stat().st_size == index['nbytes']
  columns, width, suffix = CASES[name]
  key = 'vlm.lm_head.weight' if name == 'head' else 'vlm.model.language_model.layers.0.'+suffix+'.weight'
  def record(kind):
    row = next(r for r in index['records'] if r['name'] == key and r['kind'] == kind)
    assert 0 <= row['offset'] <= index['nbytes']-row['nbytes']
    with (directory/'weights.bin').open('rb') as source:
      source.seek(row['offset'])
      raw = source.read(row['nbytes'])
    assert len(raw) == row['nbytes']
    return row, raw
  w_record = next(r for r in index['records'] if r['name'] == key and r['kind'] == 'fp8')
  if name == 'head':
    assert not w_record['packed'] and w_record['shape'][1] == width
    config = json.loads((checkpoint/'config.json').read_text())
    ids = sorted(set(json.loads(vocabulary.read_text())))
    ids = [i for i in ids if not config['traj_ids']['history_id0'] <= i < config['traj_ids']['history_id0']+config['history_vocab_size']
           and not config['traj_ids']['future_id0'] <= i < config['traj_ids']['future_id0']+config['future_vocab_size']
           and i != config['vlm_config']['text_config']['eos_token_id']]
    assert config['traj_ids']['future_start'] in ids and 0 <= min(ids) <= max(ids) < w_record['shape'][0]
    ids += [ids[-1]]*((-len(ids)) % 64)
    assert len(ids) == columns, 'Reduced head shape changed; update the bounded shape set explicitly'
    mapped = np.memmap(directory/'weights.bin', mode='r', dtype=np.uint8, offset=w_record['offset'], shape=tuple(w_record['shape']))
    logical = np.asarray(mapped[ids]).copy()
    del mapped
    packed = pack_numpy(logical)
    raw_sha = sha(logical.tobytes())
  else:
    w_record, raw = record('fp8')
    assert w_record['packed'] and w_record['shape'] == [columns, width]
    packed = np.frombuffer(raw, np.uint8).reshape(columns, width).copy()
    raw_sha = sha(raw)
  assert np.all((packed & 127) != 127), 'Actual model contains nonfinite FP8 weights'
  s_record, scale_raw = record('scale')
  scale = np.frombuffer(scale_raw, np.float32).copy()
  assert scale.shape == (1,) and np.isfinite(scale).all() and scale[0] > 0
  assert not any(r['name'] == key.removesuffix('.weight')+'.bias' for r in index['records']), 'Unexpected real projection bias'
  return {'w': packed, 'sw': scale, 'identity': {'name': key, 'shape_nk': [columns, width], 'record': w_record,
          'selected_weight_bytes_sha256': raw_sha, 'scale_bytes_sha256': sha(scale_raw), 'cache_index_sha256': sha(index_raw),
          'declared_full_cache_sha256': index['sha256'], 'full_cache_rehashed': False,
          'vocabulary_sha256': sha(vocabulary.read_bytes()) if name == 'head' else None, 'all_selected_fp8_finite': True}}


def fixture(shape, state, model=None, control=None):
  m, n, k = shape
  assert 0 <= state <= 5
  seed = 16 if state == 0 else 17 if state < 5 else 18
  rng = np.random.default_rng(20261116+seed)
  if control == 'sparse-outliers':
    a = np.zeros((m, k), np.uint8)
    codes = np.array([126, 254, 1, 129, 56, 184, 0, 128, 120, 248], np.uint8)
    a[np.arange(m), (np.arange(m)+seed) % k] = codes[(np.arange(m)+seed) % len(codes)]
    w = np.arange(n*k, dtype=np.uint64).reshape(n, k)
    w = ((w % 127).astype(np.uint8) | ((w % 2).astype(np.uint8) << 7))
  elif control:
    a = encode_integer(rng.integers(-7, 8, (m, k)))
    w = encode_integer(np.random.default_rng(123).integers(-7, 8, (n, k)))
    a[0, 0], w[0, 0] = 126, 254  # Finite 448 outliers, still exact integer sums.
  else:
    assert model is not None
    a = rng.integers(0, 127, (m, k), dtype=np.uint8) | (rng.integers(0, 2, (m, k), dtype=np.uint8) << 7)
    a[:, :8] = np.array([0, 128, 1, 129, 126, 254, 56, 184], np.uint8)
    w = None
  packed = model['w'].copy() if model is not None else pack_numpy(w)
  if state >= 3: packed ^= np.uint8(128)
  scales = np.exp2(-(np.arange(m) % 7)-4).astype(np.float32).reshape(m, 1)
  if not control: scales *= np.float32(1.071234569)
  if state >= 2: scales[m//2] *= np.float32(2)
  if state == 5: scales = scales[::-1].copy()*np.float32(.75)
  sw = model['sw'].copy() if model is not None else np.array([.25], np.float32)
  if state >= 4: sw *= np.float32(1.125)
  bias = bf16(np.random.default_rng(39+(state >= 4)).integers(-8, 9, n).astype(np.float32)/8)
  return {'a': pack_numpy(a), 'w': packed, 'sx': scales, 'sw': sw, 'bias': bias, 'logical_a': a}


def reference(data, bias, sparse=False):
  a, w = decode(data['logical_a']), decode(unpack_numpy(data['w']))
  if sparse:
    assert np.all(np.count_nonzero(a, axis=1) <= 1)
    indices = np.argmax(np.abs(a), axis=1)
    # WMMA starts from +0. Adding its sole nonzero product also fixes the
    # signed-zero result for rows containing only +0/-0 products.
    sums = np.float32(0)+a[np.arange(a.shape[0]), indices, None]*w[:, indices].T
  else:
    np.testing.assert_array_equal(a, a.astype(np.int64))
    np.testing.assert_array_equal(w, w.astype(np.int64))
    sums = a.astype(np.int64) @ w.astype(np.int64).T
    assert np.max(np.abs(sums)) < 2**24
    sums = sums.astype(np.float32)
  value = sums*(data['sx']*data['sw'][0])
  if bias: value = value+bf32(data['bias'])
  return bf16(value)


def bind_renderer(fn, renderer):
  class Devices:
    DEFAULT = 'agent-decode-offline'
    def __getitem__(self, _): return SimpleNamespace(renderer=renderer)
  result = FunctionType(fn.__code__, dict(fn.__globals__, Device=Devices()), fn.__name__, fn.__defaults__)
  result.__kwdefaults__ = fn.__kwdefaults__
  return result


def kernels(renderer):
  fn = bind_renderer(local('tiled-gemm').kernel, renderer)
  result = {f'{m}x{n}': partial(fn, tile_m=m, tile_n=n, row_scale=True, column_scale=False,
                              local_m=1, local_n=1, swap=False, unroll_k=1, group_m=1) for m, n in TILES}
  repaired = partial(local('agent-decode-gemm-rounding').kernel, raw_kernel=result['1x1'], renderer=renderer)
  repaired.raw_kernel = result['1x1']
  result['1x1-accum'] = repaired
  return result


def serial_proof(batch, queues, count=CALLS):
  """Copied from the reviewed expert proof; no metadata slot-zero assumption."""
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  assert len(batch) == count and len(queues) == 1
  assert len({(d, q) for _, d, q in batch}) == 1
  calls, spans, mappings, ios = [], [], [], []
  for call, _, _ in batch:
    assert call.body.op is Ops.PROGRAM
    outs, ins = get_call_outs_ins(call)
    assert len(outs) == 1
    output = call.src[1+outs[0]]
    b, lane, offset = unwrap_lane(output)
    assert b.op is Ops.PARAM and lane is None
    spans.append((b, offset, output.max_numel()*output.dtype.itemsize))
    args = [unwrap_lane(call.src[1+g]) for g in call.body.arg.globals]
    assert all(base.op is Ops.PARAM and lane is None for base, lane, _ in args)
    mappings.append(tuple(base.arg.slot for base, _, _ in args))
    ios.append((outs, ins))
    calls.append(call)
  assert len(set(spans)) == 1, 'All calls must share the same output storage span'
  assert len(set(mappings)) == len(set(ios)) == 1
  ordered = []
  for cmd in next(iter(queues.values())):
    if cmd.op is Ops.CALL and cmd.body.op is Ops.PROGRAM: ordered.append(cmd)
    elif cmd.op is Ops.CALL and cmd.body.op is Ops.CUSTOM_FUNCTION and cmd.body.arg.name == 'exec_barrier': ordered.append('barrier')
  assert ordered == [x for i, call in enumerate(calls) for x in ((call,) if i == 0 else ('barrier', call))], 'Missing ordered WAW barriers'
  return {'calls': count, 'ordered_execution_barriers': count-1, 'output_input_slot': spans[0][0].arg.slot,
          'output_byte_offset': spans[0][1], 'output_span_bytes': spans[0][2], 'kernel_global_input_slots': list(mappings[0]),
          'output_argument_indices': list(ios[0][0]), 'input_argument_indices': list(ios[0][1]), 'actual_queue_builder': True}


def offline_proof(calls, device='CPU'):
  from tinygrad.runtime.support.hcq2 import _build_queues
  batch = [(c, device, 'COMPUTE:0') for c in calls]
  ctx = SimpleNamespace(batch=batch, calls={(device, 'COMPUTE:0'): list(range(len(calls)))},
                        peers={(device, 'COMPUTE:0'): set()}, queues={device: ['COMPUTE:0']}, stamps=lambda *_: ())
  return serial_proof(batch, _build_queues(ctx), len(calls))


@contextmanager
def observe_queue(jit):
  from tinygrad.runtime.support import hcq2
  original = hcq2._build_queues
  def build(ctx):
    queues = original(ctx)
    if len(ctx.batch) == jit.count: jit.proofs.append(serial_proof(ctx.batch, queues, jit.count))
    return queues
  hcq2._build_queues = build
  try: yield
  finally: hcq2._build_queues = original


class SerialJit:
  def __init__(self, shape, bias, fn, count):
    from tinygrad import TinyJit, Tensor
    self.shape, self.count, self.output, self.proofs = shape, count, None, []
    def run(a, w, sx, sw, b, out):
      for _ in range(count): out = Tensor.custom_kernel(out, a, w, sx, sw, *((b,) if bias else ()), fxn=fn)[0].realize()
      return (out,)
    self.inner = TinyJit(run)

  @property
  def captured(self): return self.inner.captured

  def __call__(self, *inputs):
    from tinygrad import Tensor, dtypes
    if self.output is None:
      self.output = Tensor.empty(self.shape[0], self.shape[1], dtype=dtypes.bfloat16, device=inputs[0].device).realize()
      self.output.uop.buffer.ensure_allocated()
    if self.inner.cnt == 1:
      with observe_queue(self): result = self.inner(*inputs, self.output)
    else: result = self.inner(*inputs, self.output)
    assert result[0].uop.buffer.base is self.output.uop.buffer.base and result[0].uop.buffer.offset == self.output.uop.buffer.offset
    return result


def metadata(jit):
  assert len(jit.proofs) == 1, 'Expected exactly one observed serial graph'
  proof = jit.proofs[0]
  assert proof['output_input_slot'] == 5
  records = [k for call in jit.captured._linear.src for k in call.without_after.arg.aux.kernels]
  assert len(records) == jit.count
  for kernel in records:
    assert list(kernel[5]) == proof['kernel_global_input_slots']
    assert list(kernel[6][0]) == proof['output_argument_indices'] and list(kernel[6][1]) == proof['input_argument_indices']
  return proof


def compile_one(shape, bias, fn, renderer):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp, Ops, AxisType
  m, n, k = shape
  args = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate([
    ((m, n), dtypes.bfloat16), (((m+15)//16*16, k), dtypes.fp8e4m3), (((n+15)//16*16, k), dtypes.fp8e4m3),
    ((m, 1), dtypes.float32), ((1,), dtypes.float32)])]
  if bias: args.append(UOp.placeholder((n,), dtypes.bfloat16, 5))
  ast = getattr(fn, 'raw_kernel', fn)(*args)
  wmma = [u for u in ast.toposort() if u.op is Ops.WMMA]
  reduce = [u for u in ast.toposort() if u.op is Ops.REDUCE]
  assert len(wmma) == len(reduce) == 1 and wmma[0].arg[0] == (16, 16, 16)
  assert reduce[0].arg == (Ops.ADD, 0) and len(reduce[0].src) == 2
  r = reduce[0].src[1]
  assert r.op is Ops.RANGE and r.axis_type is AxisType.WEAK and r.vmax+1 == k//16
  assert wmma[0] in reduce[0].src[0].toposort(), 'Sequential reduction must accumulate WMMA results'
  p = fn(*args) if hasattr(fn, 'raw_kernel') else to_program(ast, renderer)
  binary = p.src[3].arg
  note = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
  names = ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
           'group_segment_fixed_size', 'wavefront_size')
  resources = {key: int(value) for key, value in re.findall(r'\.('+'|'.join(names)+r'):\s*(\d+)', note)}
  assert set(names) <= resources.keys()
  call = p.call(*(UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip((5, 0, 1, 2, 3, 4), args)))
  proof = offline_proof([call]*CALLS, 'AMD')
  source = next(u.arg for u in p.src if u.op is Ops.SOURCE)
  loop_source = source[source.index('define amdgpu_kernel'):source.index('loop_exit_2:')]
  wmma_lines = [line.strip() for line in loop_source.splitlines() if 'call <8 x float> @llvm.amdgcn.wmma.' in line]
  fadd_lines = [line.strip() for line in loop_source.splitlines() if ' = fadd ' in line]
  # This catches the original 1x1 lowering error that the high-level AST gate
  # missed. The repaired tile and selected baseline must carry C into WMMA.
  zero_vectors = set(re.findall(r'(%v\d+) = .*<8 x float> .*float 0x0000000000000000', loop_source))
  accumulators = [re.search(r'<8 x float> (%v\d+)\)', line).group(1) for line in wmma_lines]
  accumulator_contract = dict(llvm_wmma_calls=len(wmma_lines), loop_scalar_fadd_count=len(fadd_lines),
    third_argument_is_zero_vector=[arg in zero_vectors for arg in accumulators],
    running_wmma_accumulator=bool(wmma_lines) and not fadd_lines and not any(arg in zero_vectors for arg in accumulators))
  if hasattr(fn, 'raw_kernel'):
    assert accumulator_contract['running_wmma_accumulator'], accumulator_contract
  return {'global_size': p.arg.global_size, 'local_size': p.arg.local_size, 'workgroups': math.prod(p.arg.global_size),
          'waves_per_workgroup': math.prod(p.arg.local_size)//resources['wavefront_size'], 'resources': resources,
          'binary_sha256': sha(binary), 'k16_contract': {'wmma': repr(wmma[0].arg), 'sequential_k16_steps': k//16},
          'offline_serial_proof': proof, 'source_sha256': sha(source.encode()), 'lowered_accumulator_contract': accumulator_contract}


def upload(data, device):
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  result = []
  for key in KEYS:
    value = data[key]
    dt = dtypes.bfloat16 if key == 'bias' else dtypes.float32 if key in ('sx', 'sw') else dtypes.fp8e4m3
    b = Buffer(device, value.nbytes, options=BufferSpec(cpu_access=True)).ensure_allocated()
    b.host.view(size=b.nbytes, fmt='B')[:] = value.tobytes()
    result.append(Tensor(UOp.from_buffer(b, dt)).reshape(value.shape))
  return tuple(result)


def mutate(inputs, data, device):
  device.synchronize()
  for tensor, key in zip(inputs, KEYS, strict=True):
    b = tensor.uop.buffer
    assert b.nbytes == data[key].nbytes
    b.host.view(size=b.nbytes, fmt='B')[:] = data[key].tobytes()


def poison(output, device):
  from tinygrad.device import Buffer
  device.synchronize()
  b = output.uop.buffer.ensure_allocated()
  source = Buffer('CPU', b.nbytes, initial_value=bytes([0xa5])*b.nbytes)
  b.copy_from(source)
  device.synchronize()


def read_output(output, device):
  device.synchronize()
  b, chunks = output.uop.buffer, []
  offset, left = b.offset, output.nbytes()
  for address, size in b.base.get_storage().meta.mapping.paddrs:
    if offset >= size:
      offset -= size
      continue
    count = min(left, size-offset)
    for start in range(0, count, 65536):
      chunks.append(bytes(device.iface.pci_dev.map_bar(bar=device.iface.vram_bar, off=address+offset+start, size=min(count-start, 65536))[:]))
    left -= count
    offset = 0
    if not left: break
  assert left == 0
  values = np.frombuffer(b''.join(chunks), np.uint16).reshape(output.shape)
  assert np.isfinite(bf32(values)).all()
  return values


def checked_run(jit, inputs, device):
  poison(jit.captured.ret[0], device)
  return read_output(jit(*inputs)[0], device)


def gpu_case(name, shape, bias, model, fns, compiled, rounds, device, control=None, *, blocked_variants=(), on_progress=None):
  data = fixture(shape, 0, model, control)
  inputs = upload(data, device.device)
  record = {'case': name, 'shape_mnk': shape, 'bias': bias, 'model_identity': model['identity'] if model else None,
            'checks': [], 'serial_proofs': {}, 'compiled': compiled, 'timings': None,
            'baseline_label': '1x4', 'baseline_source': 'Original selected decoder.gemm tile=(1,4), row_scale=True',
            'variants': {label: {'status': 'blocked_control' if label in blocked_variants else 'pending',
              'checks': [], 'failures': [], 'eligible_for_timing': False} for label in fns}}
  assert '1x4' in fns and '1x4' not in blocked_variants
  def checkpoint():
    if on_progress is not None: on_progress(record)
  def reject(label, stage, state, error, actual=None, expected=None):
    row = {'stage': stage, 'state': state, 'error': f'{type(error).__name__}: {error}'}
    if actual is not None:
      row['actual_sha256'] = sha(actual.tobytes())
    if actual is not None and expected is not None and actual.shape == expected.shape:
      different = np.argwhere(actual != expected)
      row.update(mismatched_bf16=len(different), elements=expected.size,
        first_mismatches=[{'index': index.tolist(), 'actual': int(actual[tuple(index)]),
          'expected': int(expected[tuple(index)])} for index in different[:16]],
        expected_sha256=sha(expected.tobytes()))
    if label == '1x4':
      record['baseline_failure'] = row
      checkpoint()
      raise AssertionError(f'Original 1x4 baseline gate failed: {stage}, state={state}: {error}') from error
    variant = record['variants'][label]
    variant.update(status='rejected', eligible_for_timing=False)
    variant['failures'].append(row)
    print(json.dumps({'case': name, 'shape_mnk': shape, 'bias': bias, 'variant': label, 'correctness_failure': row}), flush=True)
    checkpoint()
  def compare(label, jit, expected, stage, state):
    actual = None
    try:
      actual = checked_run(jit, inputs, device)
      np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
      reject(label, stage, state, exc, actual, expected)
      return False
    record['variants'][label]['checks'].append({'stage': stage, 'state': state, 'exact_bf16': True,
      'output_sha256': sha(actual.tobytes())})
    return True
  checkpoint()
  singles = {}
  for label, fn in fns.items():
    if label in blocked_variants: continue
    jit = singles[label] = SerialJit(shape, bias, fn, 1)
    try:
      for _ in range(3): jit(*inputs)
      metadata(jit)
    except AssertionError as exc:
      reject(label, 'single_graph_capture', None, exc)
  previous, first = None, None
  for state in range(6):
    data = fixture(shape, state, model, control)
    mutate(inputs, data, device)
    try:
      baseline = checked_run(singles['1x4'], inputs, device)
      if control: np.testing.assert_array_equal(baseline, reference(data, bias, control == 'sparse-outliers'))
      if previous is not None:
        changed = np.any(previous != baseline, axis=1)
        assert changed.any(), 'Mutation did not affect output'
        if state == 2: np.testing.assert_array_equal(changed, np.arange(shape[0]) == shape[0]//2)
    except AssertionError as exc:
      reject('1x4', 'independent_reference_or_mutation', state, exc)
    if first is None: first = baseline.copy()
    for label, jit in singles.items():
      if record['variants'][label]['status'] != 'rejected':
        compare(label, jit, baseline, 'single_mutation', state)
    previous = baseline.copy()
    record['checks'].append({'state': state,
                             'all_tiles_exact_bf16': all(v['status'] not in ('rejected', 'blocked_control') for v in record['variants'].values()),
                             'independent_oracle': bool(control),
                             'isolated_row_scale': state == 2, 'baseline_sha256': sha(baseline.tobytes())})
    checkpoint()
  accepted = [label for label, row in record['variants'].items() if row['status'] == 'pending']
  for label in accepted: record['variants'][label]['status'] = 'verified_singles'
  if control:
    for label in accepted: record['variants'][label]['status'] = 'passed_control'
    checkpoint()
    return record
  batches = {}
  for label in accepted:
    jit = batches[label] = SerialJit(shape, bias, fns[label], CALLS)
    try:
      for _ in range(3): jit(*inputs)
      record['serial_proofs'][label] = metadata(jit)
    except AssertionError as exc:
      reject(label, 'batch_graph_capture', 5, exc)
      continue
    compare(label, jit, previous, 'batch_captured_state', 5)
  # Captured state 5 must consume changed state 0, including original cache weights.
  mutate(inputs, fixture(shape, 0, model), device)
  try:
    expected = checked_run(singles['1x4'], inputs, device)
    np.testing.assert_array_equal(expected, first)
    assert not np.array_equal(previous, expected)
  except AssertionError as exc:
    reject('1x4', 'original_input_replay', 0, exc)
  for label, jit in batches.items():
    if record['variants'][label]['status'] != 'rejected' and compare(label, jit, expected, 'batch_mutated_replay', 0):
      record['variants'][label].update(status='verified_batch', eligible_for_timing=True)
  eligible = [label for label in fns if record['variants'][label]['eligible_for_timing']]
  assert '1x4' in eligible
  checkpoint()
  if len(eligible) == 1:
    record['timing_skipped'] = 'No independently verified alternative remains'
    checkpoint()
    return record
  orders = []
  for r in range(rounds):
    order = eligible if r < len(eligible) else eligible[::-1]
    shift = r % len(eligible)
    orders.append(tuple(order[shift:]+order[:shift]))
  pairs = []
  for r in range(rounds):
    pair = {'order': orders[r]}
    for label in orders[r]:
      device.synchronize()
      start = time.perf_counter()
      batches[label](*inputs)
      device.synchronize()
      pair[label] = (time.perf_counter()-start)*1000
    pairs.append(pair)
  # Validate again before publishing any timing. A later failure invalidates
  # that variant's observed samples; it cannot become an accepted result.
  final = []
  for label in eligible:
    if compare(label, batches[label], expected, 'post_timing', 0):
      record['variants'][label]['status'] = 'passed'
      final.append(label)
    else:
      record.setdefault('invalidated_timing_samples', {})[label] = [pair[label] for pair in pairs]
  values = {label: statistics.median(p[label] for p in pairs)/CALLS for label in final}
  published_pairs = [{key: value for key, value in pair.items() if key == 'order' or key in final} for pair in pairs]
  record['timings'] = {'pairs_batch_ms': pairs, 'calls_per_batch': CALLS, 'median_ms_per_call': values,
                       'ratios_to_1x4': {key: value/values['1x4'] for key, value in values.items()},
                        'host_inclusive': True, 'warm_repeated_weights': True, 'packing_uploads_excluded': True,
                        'real_model_weight_streaming_reproduced': False}
  record['timings']['pairs_batch_ms'] = published_pairs
  record['timings']['eligible_labels_before_timing'] = eligible
  record['timings']['accepted_labels_after_timing'] = final
  record['mutated_batch_replay_exact'] = record['post_timing_exact'] = all(
    row['status'] == 'passed' for row in record['variants'].values())
  record['complete'] = True
  checkpoint()
  return record


def cpu_checks():
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.uop.ops import UOp, KernelInfo
  from tinygrad.codegen import to_program
  from amd_gemm import pack
  checks = []
  for name, shape in CONTROLS.items():
    previous = {}
    for state in range(6):
      data = fixture(shape, state, control=name)
      np.testing.assert_array_equal(pack(Tensor(data['logical_a'], device='CPU')).numpy(), data['a'])
      np.testing.assert_array_equal(pack(Tensor(unpack_numpy(data['w']), device='CPU')).numpy(), data['w'])
      for bias in (False, True):
        expected = reference(data, bias, name == 'sparse-outliers')
        a, w = Tensor(decode(data['logical_a']), device='CPU'), Tensor(decode(unpack_numpy(data['w'])), device='CPU')
        result = a.dot(w.T)*(Tensor(data['sx'], device='CPU')*Tensor(data['sw'], device='CPU'))
        if bias: result = result+Tensor(bf32(data['bias']), device='CPU')
        np.testing.assert_array_equal(result.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy(), expected)
        if state == 2: np.testing.assert_array_equal(np.any(previous[bias] != expected, axis=1), np.arange(shape[0]) == shape[0]//2)
        previous[bias] = expected
        checks.append({'control': name, 'state': state, 'bias': bias, 'independent_exact': True, 'packing_exact': True})
  def fill(out, a, w, sx, sw, *bias):
    r = UOp.range(out.numel(), 0)
    return out.flatten().index(r).store((a.flatten().index(0).float()+sx.flatten().index(r//out.shape[1])).cast(out.dtype)).end(r).sink(arg=KernelInfo())
  shape = (16, 64, 64)
  data = fixture(shape, 0, control='integer')
  inputs = upload(data, 'CPU')
  jit = SerialJit(shape, False, fill, CALLS)
  for _ in range(3): jit(*inputs)
  proof = offline_proof(jit.captured._linear.src)
  assert proof['output_input_slot'] == 5 and proof['ordered_execution_barriers'] == 39
  changed = fixture(shape, 2, control='integer')
  mutate(inputs, changed, Device['CPU'])
  poison(jit.captured.ret[0], Device['CPU'])
  actual = jit(*inputs)[0].bitcast(dtypes.uint16).numpy()
  expected = bf16(np.broadcast_to(decode(changed['a'].reshape(-1)[:1])[0]+changed['sx'], actual.shape))
  np.testing.assert_array_equal(actual, expected)
  checks.append({'serial_capture_mutation_poison_exact': True, 'proof': proof})
  a, out = UOp.placeholder((1,), dtypes.float32, 0), UOp.placeholder((1, 1), dtypes.bfloat16, 1)
  program = to_program(fill(out, a, a, a, a), Device['CPU'].renderer)
  call = program.call(UOp.param(0, dtypes.float32, 1, 'CPU'), UOp.param(1, dtypes.bfloat16, 1, 'CPU'))
  proof = offline_proof([call]*CALLS)
  assert proof['output_argument_indices'] == [1]
  broken = [call]*CALLS
  broken[1] = call.replace(src=(call.src[0], call.src[1], UOp.param(99, dtypes.bfloat16, 1, 'CPU')))
  try: offline_proof(broken)
  except AssertionError as exc: assert 'output storage' in str(exc)
  else: raise AssertionError('Independent output passed the WAW proof')
  checks.append({'output_argument_one_proven': True, 'independent_output_rejected': True})
  assert not any('AMD' in d for d in Device._opened_devices)
  return checks


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode', choices=('cpu-check', 'compile', 'gpu'), default='compile')
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--cases', default='q,kv,fc2')
  p.add_argument('--rows', default='16,48')
  p.add_argument('--bias', choices=('none', 'both'), default='none')
  p.add_argument('--rounds', type=int, choices=(3, 6), default=6)
  p.add_argument('--wgp', action='store_true')
  p.add_argument('--fp8-cache', type=Path, default=Path('/dev/shm/alpamayo-9060-fp8'))
  p.add_argument('--checkpoint', type=Path, default=Path('/tmp/opencode/alpamayo-asius-20261114/model'))
  p.add_argument('--vocabulary', type=Path, default=HERE/'coc_vocab.json')
  args = p.parse_args()
  cases, rows = args.cases.split(','), [int(r) for r in args.rows.split(',')]
  assert len(set(cases)) == len(cases) and set(cases) <= CASES.keys() and cases
  assert len(set(rows)) == len(rows) and set(rows) <= set(ROWS) and rows
  if args.mode == 'gpu': assert len(cases)*len(rows) <= 8, 'At most eight actual shapes per GPU invocation'
  os.environ.update(BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
  if args.mode != 'gpu': os.environ['DEV'] = 'CPU'
  if args.mode == 'compile': os.environ['ALLOW_DEVICE_USAGE'] = '0'
  from tinygrad import Device
  report = {'mode': args.mode, 'provenance': provenance(), 'wgp': args.wgp, 'gpu_executed': False, 'complete': False,
            'environment': {k: os.environ.get(k) for k in ('TC_OPT', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL', 'LLVMOPT')},
             'cases': [], 'controls': [], 'scope': 'Draft verification only; packing and uploads excluded from 40-call timings',
             'variant_contract': {'1x4': 'Unmodified selected baseline', '1x2': 'Unmodified original alternative',
               '1x1': 'Original diagnostic control; rejected if it differs from baseline',
               '1x1-accum': 'Local reshape-WMMA lowering repair; existing model hook cannot select this label'},
             'strict_equality_required': True}
  try:
    if args.mode == 'cpu-check':
      report['cpu_checks'] = cpu_checks()
      report['cpu_checks'].extend(local('agent-decode-gemm-hook').cpu_check())
    else:
      if args.mode == 'gpu':
        from tinygrad.device import Compiled
        Compiled.wait_timeout_ms = 10000
        local('prefill-kernels').fix_dispatch_packet()
        device = Device['AMD']
        assert device.is_usb and device.arch == 'gfx1200'
        if args.wgp: local('wgp_compiler').install(device)
        renderer = device.renderer
      else:
        from tinygrad.helpers import Target
        from tinygrad.renderer.llvmir import AMDLLVMRenderer
        renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
        if args.wgp: renderer.compiler = local('wgp_compiler').WGPCompiler('gfx1200')
      fns = kernels(renderer)
      specs = [(name, shape, bias, None, name) for name, shape in CONTROLS.items() for bias in (False, True)]
      for name in cases:
        model = model_fixture(name, args.fp8_cache, args.checkpoint, args.vocabulary)
        specs += [(name, (m, *CASES[name][:2]), bias, model, None) for m in rows for bias in ((False, True) if args.bias == 'both' else (False,))]
      blocked = set()
      for name, shape, bias, model, control in specs:
        print(json.dumps({'case': name, 'shape': shape, 'bias': bias}), flush=True)
        compiled = {label: compile_one(shape, bias, fn, renderer) for label, fn in fns.items()}
        assert all(r['k16_contract'] == compiled['1x4']['k16_contract'] for r in compiled.values())
        for label, row in compiled.items():
          tm, tn = (int(v) for v in label.split('-')[0].split('x'))
          assert row['workgroups'] == math.ceil(shape[0]/(16*tm))*math.ceil(shape[1]/(16*tn))
        if args.mode == 'gpu':
          report['gpu_executed'] = True
          target = report['controls' if control else 'cases']
          def persist(value):
            if not any(value is previous for previous in target): target.append(value)
            save(args.output, report)
          result = gpu_case(name, shape, bias, model, fns, compiled, args.rounds, device, control,
            blocked_variants=() if control else blocked, on_progress=persist)
          if control:
            blocked.update(label for label, row in result['variants'].items() if row['status'] == 'rejected')
        else: result = {'case': name, 'shape_mnk': shape, 'bias': bias, 'compiled': compiled, 'model_identity': model['identity'] if model else None}
        target = report['controls' if control else 'cases']
        if not any(result is previous for previous in target): target.append(result)
        save(args.output, report)
      if args.mode != 'gpu': assert not any('AMD' in d for d in Device._opened_devices)
    report['complete'] = True
    report['summary'] = {'mode': args.mode, 'cpu_checks': len(report.get('cpu_checks', [])), 'actual_cases': len(report['cases']),
                          'control_cases': len(report['controls']), 'gpu_executed': report['gpu_executed']}
    rejected = [{'case': row['case'], 'shape_mnk': row['shape_mnk'], 'bias': row['bias'], 'variant': label,
                 'status': variant['status']} for row in report['controls']+report['cases']
                for label, variant in row.get('variants', {}).items() if variant['status'] in ('rejected', 'blocked_control')]
    report['summary'].update(rejected_variants=rejected,
      all_requested_variants_verified=(not rejected) if args.mode == 'gpu' else None)
    save(args.output, report)
    print(json.dumps(report['summary']))
    if rejected:
      raise AssertionError(f'{len(rejected)} strict variant checks rejected; independently verified results remain in the report')
  except BaseException as exc:
    report['failure'] = f'{type(exc).__name__}: {exc}'
    save(args.output, report)
    raise


if __name__ == '__main__': main()
