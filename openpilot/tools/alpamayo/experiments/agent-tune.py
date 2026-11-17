"""Bounded tinygrad schedule tuning for Alpamayo's remaining non-matrix kernels.

Default mode compiles on the CPU without opening a GPU. See agent-tune.md.
Only --mode tune/replay opens AMD. This file never changes tinygrad source.
"""
import argparse
import ctypes
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np

SCHEMA = 1
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = 'relu-pack-prefill,quant-pack-prefill,quant-vision-pair,decode-rms,residual-prefill'


@dataclass(frozen=True)
class Case:
  kind: str
  rows: int
  width: int
  origin: str
  images: int = 1


CASES = {
  'relu-pack-prefill': Case('relu-pack', 2027, 9216, 'model.py:text_layer + quantize-packed.py:quantize'),
  'relu-pack-padded': Case('relu-pack', 2112, 9216, 'model.py:text_layer + quantize-packed.py:quantize + padded-prefill.py'),
  'quant-pack-prefill': Case('quant-pack', 2027, 2048, 'quantize-packed.py:quantize'),
  'quant-pack-padded': Case('quant-pack', 2112, 2048, 'quantize-packed.py:quantize + padded-prefill.py'),
  'quant-vision': Case('quant', 720, 4304, 'quantize-reduction.py:quantize conversion, supplied tensor-wide scale'),
  'quant-vision-pair': Case('grouped-quant', 1440, 4304, 'quantize-reduction.py:grouped_quantize conversion, two image scales', 2),
  'quant-vision-hidden-pair': Case('grouped-quant', 1440, 1152, 'quantize-reduction.py:grouped_quantize conversion, two image scales', 2),
  'quant-expert': Case('quant', 64, 3840, 'quantize-reduction.py:quantize conversion, supplied tensor-wide scale'),
  'decode-rms': Case('rms', 16, 2048, 'model.py:EdgeModel.norm, fused-norm.py fallback below 64 rows'),
  'decode-rms-32': Case('rms', 32, 2048, 'model.py:EdgeModel.norm, fused-norm.py fallback below 64 rows'),
  'decode-rms-48': Case('rms', 48, 2048, 'model.py:EdgeModel.norm, fused-norm.py fallback below 64 rows'),
  'residual-prefill': Case('residual', 2027, 2048, 'model.py:text_layer residual add'),
  'residual-padded': Case('residual', 2112, 2048, 'model.py:text_layer residual add + padded-prefill.py'),
  'relu-prefill': Case('relu', 2027, 9216, 'model.py:text_layer ReLU-square'),
  'gelu-vision': Case('gelu', 720, 4304, 'model.py:vision_layer approximate GELU'),
  'pack-tail-check': Case('quant-pack', 17, 48, 'independent padded-layout boundary check'),
}


def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(Path(path).read_bytes())


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def atomic_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name+'.new')
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
  temporary.replace(path)


def bf16(x):
  """Independent round-to-nearest-even BF16; all test inputs are finite."""
  bits = np.asarray(x, dtype=np.float32).view(np.uint32)
  return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def unbf16(x): return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
def rounded(x): return unbf16(bf16(x))


def fp8(x):
  """Independent finite E4M3FN lookup with nearest-even ties and signed zero."""
  codes = np.arange(127, dtype=np.uint8)
  exp, mant = (codes >> 3).astype(np.int32), (codes & 7).astype(np.float64)
  values = np.where(exp == 0, mant * 2.0**-9, (1 + mant / 8) * np.exp2(exp - 7))
  source = np.asarray(x, dtype=np.float32)
  result = np.empty(source.shape, dtype=np.uint8)
  for start in range(0, source.size, 262144):
    chunk = source.reshape(-1)[start:start+262144]
    a = np.minimum(np.abs(chunk), 448).astype(np.float64)
    hi = np.clip(np.searchsorted(values, a, side='left'), 0, 126)
    lo = np.maximum(hi - 1, 0)
    dl, dh = a-values[lo], values[hi]-a
    chosen = np.where((dl < dh) | ((dl == dh) & ((lo & 1) == 0)), lo, hi).astype(np.uint8)
    result.reshape(-1)[start:start+262144] = chosen | (np.signbit(chunk).astype(np.uint8) << 7)
  return result


def pack_reference(x):
  """Scatter by the WMMA address formula, independent of pack's reshape/permute."""
  rows, width = x.shape
  result = np.zeros((((rows + 15)//16)*16, width), dtype=x.dtype)
  k = np.arange(width, dtype=np.int64)[None, :]
  for start in range(0, rows, 128):
    r = np.arange(start, min(rows, start+128), dtype=np.int64)[:, None]
    index = (r//16*(width//16)+k//16)*256 + (k%16//4)*64 + r%16*4 + k%4
    result.reshape(-1)[index] = x[start:start+128]
  return result


def inputs(case, seed):
  rng = np.random.default_rng(seed)
  x = rng.standard_normal((case.rows, case.width), dtype=np.float32)*3
  # Boundary values, zero, and outliers occur in every independent seed.
  edges = np.array([0., -0., 2**-10, -2**-10, 1., -1., 7., -7., 28., -28., 1e-13, -1e-13], np.float32)
  x.reshape(-1)[:len(edges)] = edges
  if seed == 2: x[:max(1, case.rows//8)] = 0
  if seed == 3: x[:max(1, case.rows//8)] = np.float32(1e-13)
  return {'x': bf16(x), 'weight': bf16(1 + rng.standard_normal(case.width, dtype=np.float32)*.1),
          'other': bf16(rng.standard_normal(x.shape, dtype=np.float32)),
          'scale': np.array([(.0625 if seed == 1 else .03125 if seed == 2 else .071234569)*(i+1)
                             for i in range(case.images)], np.float32)}


def reference(case, data, cpu_fp8_ftz=False):
  x = unbf16(data['x'])
  if case.kind in ('quant', 'quant-pack', 'relu-pack', 'grouped-quant'):
    if case.kind == 'relu-pack': x = rounded(np.maximum(x, 0)**2)
    scale = np.repeat(data['scale'], case.rows//case.images)[:, None] if case.kind == 'grouped-quant' else data['scale'][0]
    out = fp8(np.clip(x/scale, -448, 448))
    # Pinned CPU software conversion flushes FP8 subnormals. AMD checks retain them.
    if cpu_fp8_ftz: out = np.where((out & 127) < 8, out & 128, out).astype(np.uint8)
    return pack_reference(out) if 'pack' in case.kind else out
  if case.kind == 'rms':
    # Float64 accumulation is independent from tinygrad's optimized reduction.
    y = (x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2, axis=-1, keepdims=True)+1e-5)).astype(np.float32)
    return bf16(rounded(y)*unbf16(data['weight']))
  if case.kind == 'residual': return bf16(x+unbf16(data['other']))
  if case.kind == 'relu': return bf16(np.maximum(x, 0)**2)
  if case.kind == 'gelu':
    y = x.astype(np.float64)
    return bf16(.5*y*(1+np.tanh(np.sqrt(2/np.pi)*(y+.044715*y**3))))
  raise ValueError(case.kind)


def expression(case, tensors):
  from tinygrad import dtypes
  from amd_gemm import pack
  x = tensors['x']
  if case.kind in ('quant', 'quant-pack', 'relu-pack', 'grouped-quant'):
    if case.kind == 'relu-pack': x = x.relu().square()
    if case.kind == 'grouped-quant':
      q = (x.reshape(case.images, -1, case.width).float()/tensors['scale'].reshape(case.images, 1, 1))
      q = q.clamp(-448, 448).cast(dtypes.fp8e4m3).reshape(case.rows, case.width)
    else: q = (x.float()/tensors['scale'].reshape(())).clamp(-448, 448).cast(dtypes.fp8e4m3)
    return pack(q) if 'pack' in case.kind else q.contiguous()
  if case.kind == 'rms':
    from model import EdgeModel
    dummy = SimpleNamespace(cuda_kernels=False, weights={'n.weight': tensors['weight']})
    return EdgeModel.norm(dummy, x, 'n', rms=True)
  if case.kind == 'residual': return x+tensors['other']
  if case.kind == 'relu': return x.relu().square()
  if case.kind == 'gelu': return x.float().gelu().cast(dtypes.bfloat16)
  raise ValueError(case.kind)


def input_names(case):
  return ['x'] + (['scale'] if case.kind in ('quant', 'quant-pack', 'relu-pack', 'grouped-quant') else
                 ['weight'] if case.kind == 'rms' else ['other'] if case.kind == 'residual' else [])


def make_tensors(case, device, data=None):
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  result = {}
  for name in input_names(case):
    shape = (case.width,) if name == 'weight' else (case.images,) if name == 'scale' else (case.rows, case.width)
    dtype = dtypes.float32 if name == 'scale' else dtypes.bfloat16
    if data is None:
      result[name] = Tensor.empty(shape, dtype=dtype, device=device)
    else:
      raw = data[name].tobytes()
      b = Buffer(device, (len(raw)+3)//4*4, options=BufferSpec(cpu_access=True)).ensure_allocated()
      b.host.view(size=b.nbytes, fmt='B')[:] = raw.ljust(b.nbytes, b'\0')
      result[name] = Tensor(UOp.from_buffer(b, dtype))[:int(np.prod(shape))].reshape(shape)
  return result


def write_inputs(tensors, data, device):
  from tinygrad import Device
  if device == 'AMD': Device[device].synchronize()
  for name, tensor in tensors.items():
    b, raw = tensor.uop.buffer, data[name].tobytes()
    b.host.view(size=len(raw), fmt='B')[:] = raw


def raw_read(tensor, device):
  from tinygrad import Device
  b = tensor.uop.buffer
  if device != 'AMD': return bytes(b.as_memoryview())[:tensor.nbytes()]
  dev = Device[device]
  dev.synchronize()
  offset, remaining, chunks = b.offset, tensor.nbytes(), []
  for address, size in b.base.get_storage().meta.mapping.paddrs:
    if offset >= size:
      offset -= size
      continue
    take = min(remaining, size-offset)
    for start in range(0, take, 65536):
      view = dev.iface.pci_dev.map_bar(bar=dev.iface.vram_bar, off=address+offset+start, size=min(take-start, 65536))
      chunks.append(bytes(view[:]))
    remaining -= take
    offset = 0
    if not remaining: break
  assert remaining == 0
  return b''.join(chunks)


def poison_buffer(buffer):
  """Fill every byte, using the regular transfer path for unmapped storage."""
  from tinygrad import Device
  from tinygrad.device import Buffer
  b = buffer.ensure_allocated()
  chunk = bytes([0xa5])*min(65536, b.nbytes)
  if (host := b.get_storage().host) is not None:
    for start in range(0, b.nbytes, len(chunk)):
      host.view(offset=start, size=min(len(chunk), b.nbytes-start), fmt='B')[:] = chunk[:b.nbytes-start]
    return
  # This revision exposes Buffer.copy_from, not Buffer.copyin. AMDAllocator has
  # no _copyin override. Its normal copy schedule stages host bytes for USB.
  staging = Buffer('CPU', len(chunk), initial_value=chunk)
  for start in range(0, b.nbytes, len(chunk)):
    count = min(len(chunk), b.nbytes-start)
    target = b if start == 0 and count == b.nbytes else b.view(count, start).ensure_allocated()
    source = staging if count == staging.nbytes else staging.view(count, 0).ensure_allocated()
    target.copy_from(source)
  # Keep staging alive until transfers finish. Poisoning never enters timings.
  Device[b.device].synchronize()


def poison_outputs(linear, device):
  """Clear all writes to prevent stale outputs from hiding incomplete stores."""
  from tinygrad import Device
  if device == 'AMD': Device[device].synchronize()
  for call in linear.src:
    for slot in call.body.arg.outs: poison_buffer(call.src[1+slot].buffer)


def check_output(case, output, expected, device):
  actual = np.frombuffer(raw_read(output, device), dtype=expected.dtype).reshape(expected.shape)
  exact = bool(np.array_equal(actual, expected))
  if case.kind in ('rms', 'gelu'):
    a, e = unbf16(actual), unbf16(expected)
    assert np.isfinite(a).all()
    # Reduction order and native transcendental instructions may shift two BF16 ULPs.
    # This is a numerical gate; route quality remains a separate mandatory check.
    np.testing.assert_allclose(a, e, rtol=.016, atol=2e-5)
    return {'exact': exact, 'max_abs': float(np.max(np.abs(a-e))), 'rtol': .016, 'atol': 2e-5}
  np.testing.assert_array_equal(actual, expected)
  return {'exact': exact, 'sha256': sha(actual.tobytes())}


def encode(value):
  if isinstance(value, Enum): return {'enum': type(value).__name__, 'name': value.name}
  if isinstance(value, tuple): return [encode(v) for v in value]
  return value


def opts_json(opts): return [{'op': o.op.name, 'axis': o.axis, 'arg': encode(o.arg)} for o in opts]


def opts_from_json(rows):
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.uop.ops import AxisType
  def decode(v):
    if isinstance(v, list): return tuple(decode(x) for x in v)
    if isinstance(v, dict):
      assert v['enum'] == 'AxisType'
      return AxisType[v['name']]
    return v
  return [Opt(OptOps[r['op']], r['axis'], decode(r['arg'])) for r in rows]


def provenance(wgp):
  import tinygrad
  import tinygrad.codegen as cg
  from tinygrad.runtime.autogen import llvm
  tree = Path(tinygrad.__file__).resolve().parent.parent
  tracked = ['tinygrad/codegen/__init__.py', 'tinygrad/codegen/opt/search.py', 'tinygrad/codegen/opt/postrange.py',
             'tinygrad/codegen/opt/heuristic.py', 'tinygrad/renderer/llvmir.py', 'tinygrad/runtime/support/compiler_llvm.py',
             'tinygrad/runtime/support/hcq2.py', 'tinygrad/runtime/support/usb.py', 'tinygrad/engine/realize.py',
             'tinygrad/device.py', 'tinygrad/runtime/ops_amd.py']
  version = [ctypes.c_uint32() for _ in range(3)]
  llvm.LLVMGetVersion(*(ctypes.byref(v) for v in version))
  return {'schema': SCHEMA, 'tinygrad_commit': subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip(),
          'tinygrad_diff_sha256': sha(subprocess.check_output(['git', '-C', str(tree), 'diff', '--binary', 'HEAD', '--', 'tinygrad'])),
          'llvm_version': [v.value for v in version], 'llvmopt': os.environ.get('LLVMOPT', '1'),
          'power_policy': {k: os.environ.get(k) for k in ('AM_POWER_LIMIT', 'AM_PERF_LEVEL')},
          'tinygrad_files': {p: file_sha(tree/p) for p in tracked}, 'harness_sha256': file_sha(__file__),
          'production_files': {str(p.relative_to(ROOT)): file_sha(p) for p in
                               [ROOT/'model.py', ROOT/'amd_gemm.py', ROOT/'experiments/quantize-packed.py', ROOT/'experiments/quantize-reduction.py']},
          'target': 'AMD:LLVM:gfx1200', 'wgp': wgp,
          'wgp_compiler_sha256': file_sha(Path(__file__).with_name('wgp_compiler.py')) if wgp else None,
          'codegen_context': {v.key: v.value for v in cg.to_program_context if v.key not in ('DEV', 'BEAM')},
          'search_limits': {k: os.environ.get(k) for k in ('BEAM_UOPS_MAX', 'BEAM_UPCAST_MAX', 'BEAM_LOCAL_MAX', 'BEAM_TIMEOUT_SEC')}}


def cache_key(sink, prov):
  # Exclude harness and model edits from the hardware key; AST and compiler remain exact.
  binding = {k: prov[k] for k in ('schema', 'tinygrad_commit', 'tinygrad_files', 'tinygrad_diff_sha256',
                                'llvm_version', 'llvmopt', 'power_policy', 'target', 'wgp', 'wgp_compiler_sha256', 'codegen_context')}
  return sha(sink.key + json.dumps(binding, sort_keys=True).encode())


@contextmanager
def capture_scheduler():
  import tinygrad.codegen as cg
  from tinygrad.codegen.opt.postrange import Scheduler
  original, captured = cg.apply_opts, []
  def capture(ast, ren, beam=0):
    k = Scheduler(ast, ren)
    k.convert_loop_to_global()
    captured.append((ast, k))
    return original(ast, ren, beam=0)
  cg.apply_opts = capture
  try: yield captured
  finally: cg.apply_opts = original


def prepare(case, ren, device, data=None):
  from tinygrad.uop.ops import Ops, UOp
  from tinygrad.codegen import to_program, to_program_cache
  tensors = make_tensors(case, device, data)
  output = expression(case, tensors).contiguous()
  linear = output.schedule_linear()
  calls, schedulers = [], []
  for call in linear.src:
    call = call.without_after
    assert call.body.op is Ops.SINK, 'Only ordinary scheduled kernels belong in this harness'
    assert not any(u.op is Ops.WMMA for u in call.body.toposort()), 'Matrix kernels are excluded'
    # Capture always, including repeated compile probes in the same Python process.
    to_program_cache.clear()
    with capture_scheduler() as captured: prg = to_program(call.body, ren)
    assert len(captured) == 1
    calls.append(call.replace(src=(prg, *call.src[1:])))
    schedulers.append(captured[0])
    if device == 'AMD' and data is not None:
      for slot in prg.arg.outs:
        buf = call.src[1+slot].buffer
        assert not buf.is_allocated()
        buf.options = replace(buf.options, cpu_access=True)
  return tensors, output, UOp(Ops.LINEAR, src=tuple(calls)), schedulers


class Timer:
  """GPU timestamp durations only. USB readback and host submit time are excluded."""
  def __init__(self, device):
    self.device, self.samples = device, []
    self.old_collect = device.collect_prof
    self.old_finalize = device._at_profile_finalize
    device.collect_prof = MethodType(self.collect, device)
    device._at_profile_finalize = lambda: None

  def collect(self, dev):
    buffers, rows = {}, []
    for (buf, _), entry in dev.prof_ents.items():
      if buf not in buffers:
        # Read bytes explicitly. USB's fmt='Q' slice returns bytes, not uint64s.
        raw = bytes(buf.host.view(size=buf.nbytes, fmt='B')[:])
        assert len(raw) % 8 == 0
        buffers[buf] = struct.unpack('<'+'Q'*(len(raw)//8), raw)
      values = buffers[buf]
      start, end = values[entry.st_id], values[entry.en_id]
      duration = (end-start)/dev.timestamp_divider
      if not (start > 0 and end > start and 0 < duration < 1e6):
        raise RuntimeError(f'Invalid GPU timestamps: {start}, {end}, {duration}')
      rows.append({'name': str(entry.name), 'start': start, 'end': end, 'us': duration})
    dev.prof_ents.clear()
    self.samples.extend(rows)

  def run(self, linked):
    from tinygrad.engine.realize import run_linear
    from tinygrad.helpers import Context
    self.samples.clear()
    with Context(PROFILE=1, DEBUG=0):
      run_linear(linked, jit=True, update_stats=False, wait=False)
      self.device.synchronize()
    assert self.samples, 'No hardware timestamp records; refuse host-clock fallback'
    return sum(x['us'] for x in self.samples), list(self.samples)

  def close(self):
    self.device.synchronize()
    self.device.collect_prof, self.device._at_profile_finalize = self.old_collect, self.old_finalize


def linked(linear):
  from tinygrad.engine.realize import compile_linear, link_linear
  return link_linear(compile_linear(linear, beam=0, profile=True, cache=False))


def replace_program(linear, index, program):
  calls = list(linear.src)
  calls[index] = calls[index].replace(src=(program, *calls[index].src[1:]))
  return linear.replace(src=tuple(calls))


def choices(schedulers, limit):
  """Use the pinned BEAM action generator; exclude TC and padding exploration."""
  from tinygrad.codegen.opt import OptOps
  from tinygrad.codegen.opt.search import get_kernel_actions
  pool = {}
  for s in schedulers:
    for candidate in get_kernel_actions(s, include_0=False).values():
      if candidate.applied_opts[-1].op not in (OptOps.SPLIT, OptOps.SWAP): continue
      key = json.dumps(opts_json(candidate.applied_opts), sort_keys=True)
      pool[key] = candidate
  # Interleave action families/axes instead of exhausting the first upcast axis.
  ordered = sorted(pool.values(), key=lambda s: (len(s.applied_opts), str(s.applied_opts[-1].arg), s.applied_opts[-1].axis))
  if len(ordered) <= limit: return ordered
  return [ordered[i] for i in np.linspace(0, len(ordered)-1, limit, dtype=int)]


def compile_candidate(scheduler):
  from tinygrad.codegen.opt.search import _try_compile
  _, value = _try_compile((0, scheduler))
  return None if value is None else value[0]


def install_cache(path, wgp=False):
  """Opt-in process-local cache replay. Returns counters for the model report."""
  import tinygrad.codegen as cg
  from tinygrad.codegen.opt.postrange import Scheduler
  current, cache = provenance(wgp), json.loads(Path(path).read_text())
  assert cache['schema'] == SCHEMA
  stats = {'hits': 0, 'misses': 0, 'keys': {}, 'cache_sha256': file_sha(path)}
  original = cg.apply_opts
  def apply(ast, ren, beam=0):
    key = cache_key(ast, current)
    row = cache['entries'].get(key)
    if (row is None or repr(ren.target).split('+')[-1] != current['target'] or
        (type(ren.compiler).__name__ == 'WGPCompiler') != wgp):
      stats['misses'] += 1
      return original(ast, ren, beam=0)
    assert row['validated'] and row['accepted'], 'Unvalidated tuning entry'
    k = Scheduler(ast, ren)
    k.convert_loop_to_global()
    for opt in opts_from_json(row['opts']): k.apply_opt(opt)
    stats['hits'] += 1
    stats['keys'][key] = stats['keys'].get(key, 0)+1
    return k.get_optimized_ast(name_override=ast.arg.name)
  cg.apply_opts = apply
  return stats


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode', choices=['compile', 'cpu-check', 'tune', 'replay'], default='compile')
  p.add_argument('--cases', default=DEFAULT_CASES)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--cache', type=Path)
  p.add_argument('--budget', type=int, default=24, help='Maximum attempted candidates per scheduled kernel')
  p.add_argument('--rounds', type=int, default=2)
  p.add_argument('--beam-width', type=int, default=2)
  p.add_argument('--seconds', type=float, default=180, help='Soft search deadline per case; compilation has a separate timeout')
  p.add_argument('--samples', type=int, default=5)
  p.add_argument('--wgp', action='store_true')
  args = p.parse_args()
  assert 1 <= args.budget <= 96 and 1 <= args.rounds <= 4 and 1 <= args.beam_width <= 4
  assert 1 <= args.seconds <= 900 and 3 <= args.samples <= 15
  gpu = args.mode in ('tune', 'replay')
  # No implicit GPU initialization in CPU modes, regardless of the caller's DEV.
  os.environ.update(BEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  os.environ.setdefault('BEAM_UOPS_MAX', '3000')
  os.environ.setdefault('BEAM_UPCAST_MAX', '64')
  os.environ.setdefault('BEAM_LOCAL_MAX', '256')
  os.environ.setdefault('BEAM_TIMEOUT_SEC', '10')
  os.environ.setdefault('MAX_BUFFER_SIZE', '268435456')
  if not gpu: os.environ['DEV'] = 'CPU'
  if args.mode == 'compile': os.environ['ALLOW_DEVICE_USAGE'] = '0'
  from tinygrad import Device
  from tinygrad.helpers import Context, Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
  from tinygrad.engine.realize import run_linear
  from tinygrad.device import Compiled
  import tinygrad.codegen as cg
  selected = list(CASES) if args.cases == 'all' else args.cases.split(',')
  assert selected and all(x in CASES for x in selected), f'Choose from {list(CASES)}'
  cache_path = args.cache or args.output.with_name('agent-tune-cache.json')
  cache = json.loads(cache_path.read_text()) if cache_path.exists() else {'schema': SCHEMA, 'entries': {}}
  assert cache['schema'] == SCHEMA
  report = {'mode': args.mode, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'cases': [], 'gpu_validated': False, 'pipeline_validated': False,
            'timing': 'sum of full-size GPU kernel timestamps, excluding USB readback; not pipeline throughput'}
  device = 'AMD' if gpu or args.mode == 'compile' else 'CPU'
  timer = None
  if gpu:
    Compiled.wait_timeout_ms = 10000
    local('prefill-kernels').fix_dispatch_packet()
    dev = Device['AMD']
    assert dev.arch == 'gfx1200' and dev.is_usb, 'This run requires the parent-owned Chestnut gfx1200 device'
    assert isinstance(dev.renderer, AMDLLVMRenderer)
    if args.wgp: local('wgp_compiler').install(dev)
    ren, timer = dev.renderer, Timer(dev)
  elif args.mode == 'compile':
    ren = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
    if args.wgp: ren.compiler = local('wgp_compiler').WGPCompiler('gfx1200')
  else: ren = Device['CPU'].renderer
  prov = provenance(args.wgp)
  report['provenance'] = prov
  if args.mode == 'replay':
    report['cache_replay'] = install_cache(cache_path, args.wgp)
  try:
    for name in selected:
      case, started = CASES[name], time.monotonic()
      data = None if args.mode == 'compile' else inputs(case, 1)
      tensors, output, linear, schedulers = prepare(case, ren, device, data)
      row = {'name': name, **case.__dict__, 'kernels': [], 'checks': []}
      report['cases'].append(row)
      if args.mode == 'cpu-check':
        for seed in (1, 2, 3):
          data = inputs(case, seed)
          write_inputs(tensors, data, device)
          poison_outputs(linear, device)
          run_linear(linear, update_stats=False)
          row['checks'].append({'seed': seed, 'cpu_fp8_flush_subnormals': True,
                                **check_output(case, output, reference(case, data, cpu_fp8_ftz=True), device)})
      elif args.mode == 'compile':
        for index, (sink, base) in enumerate(schedulers):
          seed = hand_coded_optimizations(base.copy())
          candidates = choices([base, seed], min(args.budget, 8))
          built = [compile_candidate(s) for s in candidates]
          for candidate, program in zip(candidates, built):
            if program is None: continue
            restored = base.copy()
            for opt in opts_from_json(opts_json(candidate.applied_opts)): restored.apply_opt(opt)
            assert compile_candidate(restored).src[3].arg == program.src[3].arg
          row['kernels'].append({'index': index, 'ast_key': sink.key.hex(), 'cache_key': cache_key(sink, prov),
                                 'baseline_binary_bytes': len(linear.src[index].body.src[3].arg),
                                 'candidate_compiles': sum(x is not None for x in built), 'candidate_attempts': len(built),
                                 'opts_json_recompile_exact': True})
      else:
        expected = reference(case, data)
        baseline = linked(linear)
        timer.run(baseline)
        row['checks'].append({'seed': 1, 'baseline': True, **check_output(case, output, expected, device)})
        if args.mode == 'tune':
          for index, (sink, base) in enumerate(schedulers):
            best, incumbent = None, linear
            seed = hand_coded_optimizations(base.copy())
            beam, seen, attempts = [base, seed], set(), 0
            history = []
            kernel_row = {'index': index, 'cache_key': cache_key(sink, prov), 'history': history, 'accepted': False}
            row['kernels'].append(kernel_row)
            incumbent_linked = linked(incumbent)
            baseline_times = [timer.run(incumbent_linked)[0] for _ in range(3)]
            best_us = statistics.median(baseline_times)
            for depth in range(args.rounds):
              if time.monotonic()-started >= args.seconds: break
              limit = max(1, (args.budget-attempts)//(args.rounds-depth))
              scores = []
              for candidate in choices(beam, limit):
                if attempts >= args.budget or time.monotonic()-started >= args.seconds: break
                attempts += 1
                record = {'opts': opts_json(candidate.applied_opts), 'depth': depth}
                history.append(record)
                prg = compile_candidate(candidate)
                if prg is None:
                  record['status'] = 'compile-rejected'
                  continue
                binary = sha(prg.src[3].arg)
                record['binary_sha256'] = binary
                if binary in seen:
                  record['status'] = 'duplicate-binary'
                  continue
                seen.add(binary)
                candidate_linear = replace_program(incumbent, index, prg)
                candidate_linked = linked(candidate_linear)
                # Fail closed on runtime/timestamp errors. Never continue after a device fault.
                poison_outputs(candidate_linear, device)
                timer.run(candidate_linked)
                try: record['check'] = check_output(case, output, expected, device)
                except AssertionError as exc:
                  record.update(status='incorrect', error=str(exc)[:600])
                  continue
                tms = [timer.run(candidate_linked)[0] for _ in range(3)]
                score = statistics.median(tms)
                record.update(status='valid', gpu_us=tms, median_us=score)
                scores.append((score, candidate))
                if score < best_us: best_us, best = score, (candidate, candidate_linear, binary)
                atomic_json(args.output, report)
              beam = [s for _, s in sorted(scores, key=lambda x: x[0])[:args.beam_width]]
              if not beam: break
            if best is not None:
              candidate, candidate_linear, binary = best
              candidate_linked, base_linked = linked(candidate_linear), linked(incumbent)
              checks = []
              for check_seed in (1, 2, 3):
                d = inputs(case, check_seed)
                write_inputs(tensors, d, device)
                poison_outputs(candidate_linear, device)
                timer.run(candidate_linked)
                checks.append({'seed': check_seed, **check_output(case, output, reference(case, d), device)})
              write_inputs(tensors, data, device)
              pairs, raw_stamps = [], []
              for sample in range(args.samples):
                pair = {}
                for label, graph in ([('base', base_linked), ('candidate', candidate_linked)] if sample%2 == 0 else
                                     [('candidate', candidate_linked), ('base', base_linked)]):
                  pair[label], stamps = timer.run(graph)
                  raw_stamps.append({'sample': sample, 'label': label, 'stamps': stamps})
                pairs.append(pair)
              baseline_us = statistics.median(x['base'] for x in pairs)
              winner_us = statistics.median(x['candidate'] for x in pairs)
              # Keep a noise margin. The final pipeline replay decides integration.
              accepted = winner_us < baseline_us*.95 and sum(x['candidate'] < x['base'] for x in pairs) >= (args.samples+1)//2
              kernel_row.update(accepted=accepted, checks=checks, paired_gpu_us=pairs, raw_timestamps=raw_stamps,
                                baseline_us=baseline_us, candidate_us=winner_us, ratio=winner_us/baseline_us,
                                winner_opts=opts_json(candidate.applied_opts))
              if accepted:
                linear = candidate_linear
                cache['entries'][cache_key(sink, prov)] = {'opts': opts_json(candidate.applied_opts), 'validated': True, 'accepted': True,
                    'provenance': prov, 'case': name, 'binary_sha256': binary, 'checks': checks, 'paired_gpu_us': pairs,
                    'ast_key': sink.key.hex(), 'source_report': str(args.output.resolve()), 'created_unix': time.time()}
                atomic_json(cache_path, cache)
        final = linked(linear)
        for seed in (1, 2, 3):
          d = inputs(case, seed)
          write_inputs(tensors, d, device)
          poison_outputs(linear, device)
          timer.run(final)
          row['checks'].append({'seed': seed, **check_output(case, output, reference(case, d), device)})
        row['final_gpu_us'] = [timer.run(final)[0] for _ in range(args.samples)]
      row['elapsed_s'] = time.monotonic()-started
      atomic_json(args.output, report)
      print(json.dumps({'case': name, 'mode': args.mode, 'kernels': len(schedulers), 'seconds': row['elapsed_s']}), flush=True)
    report['gpu_validated'] = gpu
    report['complete'] = True
    report['summary'] = {'mode': args.mode, 'cases': len(report['cases']), 'gpu_validated': gpu,
                         'pipeline_validated': False,
                         'accepted_schedules': sum(k.get('accepted', False) for c in report['cases'] for k in c['kernels']),
                         'cache_entries': len(cache['entries'])}
    if args.mode == 'tune': atomic_json(cache_path, cache)
    atomic_json(args.output, report)
  except BaseException as exc:
    report['complete'] = False
    report['failure'] = f'{type(exc).__name__}: {exc}'
    atomic_json(args.output, report)
    raise
  finally:
    if timer is not None: timer.close()


if __name__ == '__main__': main()
