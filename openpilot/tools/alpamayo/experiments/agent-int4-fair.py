"""Complete selected-policy FP8 versus passed flat A4/W4 fc2, CPU by default.

Parent-only GPU execution: two variants, 40 dependent operations, three blocks.
No numerical kernel is changed. The original fixed-scale quantizer's captured
AST is rebound to explicit scratch, avoiding an extra assign/copy kernel.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
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
from types import FunctionType, SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
FROZEN = WORK/'int4-serial-fixed-source/alpamayo/experiments'
TG = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
CALLS = 40
SHAPE = (2112, 2048, 9216)
NAME = 'vlm.model.language_model.layers.14.mlp.fc2'
HASHES = {'agent-int4-scaled-kernel.py': '0f603efeaa221cb95c93c23027100f448feb65b9477e1e0a59ba17cf6158f2d3',
          'agent-int4k32-kernel.py': '378f72c50a7618f8dac2ce7a17b1a1b54c41cb1b8ea7e604dd7136a0d455706c',
          'agent-int4-scaled-check.py': 'ec4c5e36685a4baa3f07f0857ae61d3f097da264a54b9a68e749fcd593b0d745'}
FPHASH = 'c57368db69b04d2c05c2140712b3a4f337c04717a165004d94693aedccbc141d'
GEMMHASH = 'a52cc1cd5331be686ef8fcd1516bba2cfbc1092ca1d64251176491add99041b2'


def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def save(path, obj):
  path.parent.mkdir(parents=True, exist_ok=True)
  temp = path.with_name(path.name+'.new');temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n');temp.replace(path)


def load(path):
  spec = importlib.util.spec_from_file_location('agent_int4_fair_'+sha(str(path).encode())[:12], path)
  mod = importlib.util.module_from_spec(spec);sys.modules[spec.name] = mod;spec.loader.exec_module(mod);return mod


@lru_cache(None)
def modules():
  import amd_gemm
  assert sha(Path(amd_gemm.__file__).read_bytes()) == GEMMHASH
  assert sha((HERE/'quantize-packed.py').read_bytes()) == FPHASH
  for name, digest in HASHES.items(): assert sha((FROZEN/name).read_bytes()) == digest, name
  gate = json.loads((WORK/'agent-int4-scaled-actual-serial-fixed/report.json').read_text())
  assert gate['complete'] and gate['all_pass'] and not gate['cpu_only']
  kernel = load(FROZEN/'agent-int4-scaled-kernel.py')
  return load(HERE/'quantize-packed.py'), amd_gemm, kernel, load(FROZEN/'agent-int4-scaled-check.py')


def bf(values):
  u = np.asarray(values, np.float32).view(np.uint32)
  return ((u+np.uint32(0x7fff)+((u >> 16) & 1)) >> 16).astype(np.uint16)
def unbf(bits): return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def decode_fp8(bits):
  bits = np.asarray(bits, np.uint8);e = ((bits >> 3) & 15).astype(np.int32);m = (bits & 7).astype(np.float32)
  value = np.where(e == 0, m*2.**-9, (1+m/8)*np.exp2(e-7))
  return np.where((bits & 127) == 127, np.nan, np.where(bits & 128, -value, value)).astype(np.float32)


def encode_fp8(value):
  value = np.clip(np.asarray(value, np.float32), -448, 448);assert np.isfinite(value).all()
  table = decode_fp8(np.arange(127, dtype=np.uint8)).astype(np.float64);v = np.abs(value).astype(np.float64)
  hi = np.minimum(np.searchsorted(table, v), 126);lo = np.maximum(hi-1, 0)
  a, b = v-table[lo], table[hi]-v
  out = np.where((b < a) | ((b == a) & ((hi & 1) == 0)), hi, lo).astype(np.uint8)
  return out | (np.signbit(value).astype(np.uint8) << 7)


def fp8_layout(values, unpack=False, rows=None):
  count, width = values.shape;rows = count if rows is None else rows
  out = np.zeros((rows if unpack else (rows+15)//16*16, width), values.dtype);k = np.arange(width)[None, :]
  for start in range(0, rows, 32):
    r = np.arange(start, min(rows, start+32))[:, None]
    index = (r//16*(width//16)+k//16)*256+(k%16//4)*64+(r%16)*4+k%4
    if unpack: out[start:start+len(r)] = values.reshape(-1)[index]
    else: out.reshape(-1)[index] = values[start:start+len(r)]
  return out


def unpack4(packed, rows, signed=False):
  assert packed.dtype == np.uint8 and packed.shape[-2:] == (32, 8)
  width = packed.shape[1]*32;out = np.empty((packed.shape[0]*16, width), np.int8);k = np.arange(width)[None, :]
  for start in range(0, len(out), 32):
    r = np.arange(start, min(len(out), start+32))[:, None]
    byte = packed[r//16, k//32, ((k >> 4) & 1)*16+r%16, (k%16)//2]
    v = ((byte >> (4*(k%2))) & 15).astype(np.int8)
    out[start:start+len(r)] = np.where(v >= 8, v-16, v) if signed else v
  np.testing.assert_array_equal(out[rows:], 0)
  return out[:rows]


def quant4(bits):
  x = unbf(bits);assert np.isfinite(x).all() and (x >= 0).all()
  top = x.max(axis=1);scale = np.divide(top, np.float32(15), dtype=np.float32);scale[top == 0] = 1
  q = np.rint(np.clip(np.divide(x, scale[:, None], dtype=np.float32), 0, 15)).astype(np.uint8)
  return q, scale


@lru_cache(None)
def quant_template(m, k):
  """Capture the unchanged public quantizer, including its real realize boundary."""
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import capturing
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  from tinygrad.uop.ops import Ops
  qp = modules()[0]
  x = Tensor.empty(m, k, device='CPU', dtype=dtypes.bfloat16).realize()
  s = Tensor.empty((), device='CPU', dtype=dtypes.float32).realize()
  recorder = SimpleNamespace(calls=[])
  recorder.add_linear = lambda linear: recorder.calls.extend(linear.src)
  assert not capturing, 'Prepare original AST templates before JIT capture'
  capturing.append(recorder)
  try: q, actual_s = qp.quantize(x, s)
  finally: assert capturing.pop() is recorder
  assert actual_s is s and len(recorder.calls) == 1, 'Original quantizer changed: investigate complete operation count'
  call = recorder.calls[0].without_after
  roles = {v.uop.storage_base: role for role, v in (('out', q), ('x', x), ('scale', s))}
  args = tuple(roles[unwrap_lane(v)[0]] for v in call.src[1:])
  assert len(args) == 3 and set(args) == {'out', 'x', 'scale'}
  params = sorted((p for p in call.body.toposort() if p.op is Ops.PARAM), key=lambda p: p.arg.slot)
  assert [p.arg.slot for p in params] == list(range(3))
  return call.body, tuple(params), args


def quant_program(out, x, scale):
  body, params, roles = quant_template(*x.shape)
  values = {'out': out, 'x': x, 'scale': scale}
  return body.substitute({p: values[role].flatten() for p, role in zip(params, roles, strict=True)}, walk=True)


def bind_renderer(fn, renderer):
  class Devices:
    DEFAULT = 'agent-int4-fair-offline'
    def __getitem__(self, key): return SimpleNamespace(renderer=renderer)
  clone = FunctionType(fn.__code__, dict(fn.__globals__, Device=Devices()), fn.__name__, fn.__defaults__)
  clone.__kwdefaults__ = fn.__kwdefaults__;return clone


def expected_roles(variant):
  return ({'outs': {4}, 'ins': {0, 2}}, {'outs': {5}, 'ins': {4, 1, 2, 3}}) if variant == 'fp8' else (
          {'outs': {3, 4}, 'ins': {0}}, {'outs': {5}, 'ins': {3, 1, 4, 2}})


def queue_proof(batch, queues, variant):
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import DepsTracker, unwrap_lane
  calls = [c.without_after for c, _, _ in batch]
  assert len(calls) == 2*CALLS and len(queues) == 1
  queue_of = [(d, q) for _, d, q in batch];assert len(set(queue_of)) == 1
  tracker = DepsTracker(queue_of);records = [];deps = [];signatures = []
  for i, call in enumerate(calls):
    assert call.op is Ops.CALL and call.body.op is Ops.PROGRAM
    outs, ins = get_call_outs_ins(call);spans = []
    for arg in call.src[1:]:
      base, lane, offset = unwrap_lane(arg)
      assert base.op is Ops.PARAM and lane is None, 'Every argument must be an explicit reusable unsharded JIT input'
      spans.append((base.arg.slot, offset, arg.max_numel()*arg.dtype.itemsize))
    assert {spans[j][0] for j in outs} == expected_roles(variant)[i%2]['outs']
    assert {spans[j][0] for j in ins} == expected_roles(variant)[i%2]['ins']
    assert all(offset == 0 for _, offset, _ in spans), 'Unexpected view offset'
    signature = (call.body.key, tuple(spans), outs, ins)
    if i >= 2: assert signature == signatures[i%2], 'Storage/program changed between operations'
    signatures.append(signature)
    waits = tracker.access_resources([(b, j in outs) for j, b in enumerate(call.src[1:])], i)
    if i: assert i-1 in waits, ('Missing adjacent RAW/WAR edge', i, waits)
    if i >= 2: assert i-2 in waits, ('Missing same-phase WAW edge', i, waits)
    deps.append(waits)
    records.append(dict(device=batch[i][1], name=call.body.src[0].arg.name, program_key=call.body.key.hex(),
      binary_sha256=sha(call.body.src[3].arg), source_sha256=sha(call.body.src[2].arg.encode()),
      global_input_slots=[spans[g][0] for g in call.body.arg.globals], output_argument_indices=list(outs),
      input_argument_indices=list(ins), argument_spans=spans))
  ordered = []
  for item in next(iter(queues.values())):
    item = item.without_after
    if item.op is Ops.CALL and item.body.op is Ops.PROGRAM: ordered.append(item)
    elif item.op is Ops.CALL and item.body.op is Ops.CUSTOM_FUNCTION and item.body.arg.name == 'exec_barrier': ordered.append('barrier')
  assert ordered == [v for i, c in enumerate(calls) for v in ((c,) if i == 0 else ('barrier', c))], 'Missing real HCQ execution barrier'
  return dict(operations=CALLS, kernel_calls=2*CALLS, kernels_per_operation=2, ordered_execution_barriers=2*CALLS-1,
    variant=variant, dependencies=deps, kernel_arguments=records, fixed_output_storage=True,
    raw_war_waw_verified=True, actual_queue_builder_observed=True)


@contextmanager
def observe(owner):
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  old = hcq2._build_queues
  def build(ctx):
    queues = old(ctx)
    # Eager BUFFER calls are not the reusable graph. Any PARAM graph must pass.
    if any(hcq2.unwrap_lane(b)[0].op is Ops.PARAM for c, _, _ in ctx.batch for b in c.src[1:]):
      owner.proofs.append(queue_proof(ctx.batch, queues, owner.variant))
    return queues
  hcq2._build_queues = build
  try: yield
  finally: hcq2._build_queues = old


def inspect(owner):
  from tinygrad.runtime.support import hcq2
  entries = [c.without_after for c in owner.jit.captured._linear.src]
  assert entries and all(isinstance(getattr(c.arg, 'aux', None), hcq2.HCQInfo) for c in entries)
  assert len(owner.proofs) == 1, 'Exactly one actual captured HCQ queue proof required'
  proof = owner.proofs[0];kernels = [k for c in entries for k in c.arg.aux.kernels]
  assert len(kernels) == 2*CALLS
  for actual, expected in zip(kernels, proof['kernel_arguments'], strict=True):
    assert actual[0] == expected['device'] and actual[1] == expected['name']
    assert actual[4] == bytes.fromhex(expected['program_key'])
    assert list(actual[5]) == expected['global_input_slots'], 'HCQ input binding mismatch or absent metadata'
    assert list(actual[6][0]) == expected['output_argument_indices']
    assert list(actual[6][1]) == expected['input_argument_indices']
  return dict(proof, hcq_metadata_verified=True)


class Batch:
  def __init__(self, variant, shape, count=CALLS, matrix=None):
    from tinygrad import Tensor, TinyJit, dtypes
    self.variant, self.proofs, self.count = variant, [], count
    self.m, self.n, self.k = shape;m, n, k = shape
    qp, amd, kernel, _ = modules();quant_template(m, k)
    self.out = Tensor.empty(m, n, dtype=dtypes.bfloat16).realize()
    if variant == 'fp8':
      self.q = Tensor.empty((m+15)//16*16, k, dtype=dtypes.fp8e4m3).realize();self.scales = None
      def run(x, w, xs, ws, q, out):
        for _ in range(count):
          q = Tensor.custom_kernel(q, Tensor(x.uop.after(out.uop)), xs, fxn=quant_program)[0]
          out = Tensor.custom_kernel(out, q, w, xs, ws, fxn=amd._gemm if matrix is None else matrix)[0]
        return out.realize()
    else:
      assert variant == 'a4w4'
      self.q = Tensor.empty((m+15)//16, k//32, 32, 8, dtype=dtypes.uint8).realize()
      self.scales = Tensor.empty(m, dtype=dtypes.float32).realize()
      def run(x, w, ws, q, scales, out):
        for _ in range(count):
          q, scales = kernel.quantize_pack(Tensor(x.uop.after(out.uop)), packed_out=q, scales_out=scales, wgp=True)
          out = kernel.gemm(q, w, scales, ws, rows=m, columns=n, out=out, wgp=True)
        return out.realize()
    self.jit = TinyJit(run)

  def __call__(self, *inputs):
    args = (*inputs, self.q, self.out) if self.variant == 'fp8' else (*inputs, self.q, self.scales, self.out)
    if self.jit.cnt == 1 and self.count == CALLS:
      with observe(self): out = self.jit(*args)
    else: out = self.jit(*args)
    assert out.uop.buffer.base is self.out.uop.buffer.base
    return out


def copyin(tensor, data):
  from tinygrad.device import Buffer
  data = np.ascontiguousarray(data);assert data.nbytes == tensor.nbytes()
  tensor.uop.buffer.ensure_allocated().copy_from(Buffer('PYTHON', data.nbytes, opaque=memoryview(data).cast('B')))


def poison(batch):
  from tinygrad import Device
  Device[batch.out.device].synchronize()
  for t in (batch.q, batch.scales, batch.out):
    if t is not None: copyin(t, np.full(t.nbytes(), 0xA5, np.uint8))


def read(tensor, dtype):
  """Bounded USB-safe reads outside timing. No assumption of host-mapped VRAM."""
  from tinygrad import Device
  dev = Device[tensor.device];dev.synchronize();b = tensor.uop.buffer.ensure_allocated()
  if not tensor.device.startswith('AMD'): return np.frombuffer(bytes(b.as_memoryview()), dtype).reshape(tensor.shape).copy()
  chunks = [];offset = b.offset;remaining = tensor.nbytes()
  for address, size in b.base.get_storage().meta.mapping.paddrs:
    if offset >= size: offset -= size;continue
    count = min(remaining, size-offset)
    for start in range(0, count, 65536):
      view = dev.iface.pci_dev.map_bar(bar=dev.iface.vram_bar, off=address+offset+start, size=min(count-start, 65536))
      chunks.append(bytes(view[:]))
    remaining -= count;offset = 0
    if not remaining: break
  assert remaining == 0
  return np.frombuffer(b''.join(chunks), dtype).reshape(tensor.shape)


def assets(args):
  """Read and verify only the selected layer, never modify cache/master/bundle."""
  idx_path = args.fp8_cache/'index.json';index = json.loads(idx_path.read_text());cache = args.fp8_cache/'weights.bin'
  assert cache.stat().st_size == index['nbytes'] and index['format'] == 1
  records = [r for r in index['records'] if r['name'] == NAME+'.weight'];assert len(records) == 2
  arrays = {};proof = {'fp8_index_sha256': sha(idx_path.read_bytes()), 'fp8_cache_declared_sha256': index['sha256'],
    'whole_cache_rehashed': False, 'selected_extents_rehashed': True, 'fp8_records': records, 'fp8_extent_hashes': {}}
  with cache.open('rb') as stream:
    for r in records:
      stream.seek(r['offset']);raw = stream.read(r['nbytes']);assert len(raw) == r['nbytes']
      proof['fp8_extent_hashes'][r['kind']] = sha(raw)
      arrays[r['kind']] = np.frombuffer(raw, np.float32 if r['kind'] == 'scale' else np.uint8).reshape(r['shape'])
  assert next(r for r in records if r['kind'] == 'fp8')['packed']
  assert arrays['fp8'].shape == (SHAPE[1], SHAPE[2]) and arrays['scale'].shape == (1,)
  assert proof['fp8_extent_hashes'] == {
    'scale': '9f83e9b7c9bc943c6921f34fdecd6ab18a97ef284f65bacb9c673b2835af30ff',
    'fp8': '179084dcb07f09eb139422587f4a95e1366eb852118dcde9f1126da2cf3ddbd3'}, 'Selected original FP8 cache extent changed'
  assert np.isfinite(arrays['scale']).all() and (arrays['scale'] > 0).all()
  assert not np.any((arrays['fp8'] & 127) == 127), 'Nonfinite cached FP8 weight'
  manifest_path = args.bundle/'manifest.json';manifest = json.loads(manifest_path.read_text())
  assert manifest['complete'] and manifest['mode'] == 'peroutput'
  assert manifest['layout'] == 'rdna4-k32-row16-ktile-lane32-reg2'
  assert sha(canonical({k: v for k, v in manifest.items() if k != 'identity_sha256'})) == manifest['identity_sha256']
  record, = [r for r in manifest['records'] if r['name'] == NAME]
  loaded = {}
  for field in ('packed', 'scale'):
    info = record[field];path = args.bundle/info['file'];assert path.resolve().parent == args.bundle.resolve()
    assert sha(path.read_bytes()) == info['file_sha256']
    value = np.load(path, allow_pickle=False);assert list(value.shape) == info['shape'] and value.dtype.str == info['dtype']
    assert sha(value.tobytes()) == info['sha256'];loaded[field] = value
  w4 = loaded['packed'].view(np.uint8).reshape(SHAPE[1]//16, SHAPE[2]//32, 32, 8)
  codes = unpack4(w4, SHAPE[1], True);assert codes.min() >= -7 and codes.max() <= 7
  # Verify the existing W4 bundle against selected original BF16 master bytes.
  shard = args.checkpoint/record['master_shard'];assert shard.resolve().parent == args.checkpoint.resolve()
  with shard.open('rb') as stream:
    size, = struct.unpack('<Q', stream.read(8));header = json.loads(stream.read(size))
    info = header[NAME+'.weight'];assert info['dtype'] == 'BF16' and info['shape'] == record['shape']
    assert info['data_offsets'] == record['master_data_offsets']
    start, end = info['data_offsets'];stream.seek(8+size+start);master = stream.read(end-start)
  assert sha(master) == record['master_bytes_sha256']
  master_w = unbf(np.frombuffer(master, np.uint16).reshape(record['shape']))
  expected_scale = np.maximum(np.max(np.abs(master_w), axis=1)/np.float32(7), np.float32(1e-12))
  np.testing.assert_array_equal(loaded['scale'].view(np.uint32), expected_scale.view(np.uint32))
  np.testing.assert_array_equal(codes, np.clip(np.rint(master_w/expected_scale[:, None]), -7, 7).astype(np.int8))
  calibration = json.loads(args.calibration.read_text());scalar = np.float32(calibration['scales']['vlm.model.language_model.layers.14/3'])
  assert scalar > 0 and np.isfinite(scalar)
  assert int(np.asarray(scalar).view(np.uint32)) == 1024908142, 'Original layer-14/3 calibration changed'
  proof.update(bundle_manifest_sha256=sha(manifest_path.read_bytes()), bundle_identity=manifest['identity_sha256'],
    selected_w4_record=record, original_master_extent_verified=True, w4_codes_scales_from_original_bf16_verified=True,
    calibration_sha256=sha(args.calibration.read_bytes()), calibration_key='vlm.model.language_model.layers.14/3',
    activation_scale=float(scalar), activation_scale_fp32_bits=int(np.asarray(scalar).view(np.uint32)),
    fp8_weight_scale=float(arrays['scale'][0]), fp8_weight_scale_fp32_bits=int(arrays['scale'].view(np.uint32)[0]))
  for v in (*arrays.values(), w4, loaded['scale'], codes): v.flags.writeable = False
  return dict(fp8=arrays['fp8'], fp8_scale=arrays['scale'], w4=w4, w4_scale=loaded['scale'], w4_codes=codes,
              activation_scale=scalar), proof


def activation(args):
  if args.activation is not None:
    bits = np.load(args.activation, allow_pickle=False)
    assert bits.dtype == np.uint16 and bits.shape == (SHAPE[0], SHAPE[2])
    info = dict(scope='caller-supplied BF16 post-ReLU-square activation', file=str(args.activation), sha256=sha(args.activation.read_bytes()))
  else:
    rng = np.random.default_rng(906041)
    h = unbf(bf(rng.standard_normal((SHAPE[0], SHAPE[2]), dtype=np.float32)))
    bits = bf(np.maximum(h, np.float32(0))**2);bits[0] = 0
    info = dict(scope='deterministic synthetic BF16 post-ReLU-square; actual layer-14 resident weights and calibration', seed=906041,
                route_activation=False, distribution='N(0,1) -> BF16 -> ReLU -> square -> BF16; first row zero')
  x = unbf(bits);assert np.isfinite(x).all() and (x >= 0).all()
  return bits, dict(info, payload_sha256=sha(bits.tobytes()), shape=list(bits.shape))


def source_provenance():
  paths = [Path(__file__), HERE/'agent-int4-fair-check.py', HERE/'quantize-packed.py', HERE.parent/'amd_gemm.py',
           *(FROZEN/n for n in HASHES), FROZEN/'wgp_compiler.py', WORK/'agent-int4-scaled-actual-serial-fixed/report.json',
           TG/'tinygrad/runtime/support/hcq2.py', TG/'tinygrad/engine/realize.py', TG/'tinygrad/engine/jit.py']
  return {str(p): sha(p.read_bytes()) for p in paths}


@contextmanager
def cpu_only_devices():
  """Permit CPU execution while rejecting every accelerator device lookup."""
  from tinygrad import Device
  from tinygrad.helpers import Context
  cls = type(Device);old = cls.__getitem__
  def guarded(self, key):
    assert key.split(':')[0] in ('CPU', 'PYTHON', 'NPY', 'DISK'), ('CPU-only mode rejected device', key)
    return old(self, key)
  cls.__getitem__ = guarded
  try:
    with Context(ALLOW_DEVICE_USAGE=1, DEV='CPU'): yield
  finally: cls.__getitem__ = old


def parse():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True);p.add_argument('--gpu', action='store_true')
  p.add_argument('--fp8-cache', type=Path, default=Path('/dev/shm/alpamayo-9060-fp8'))
  p.add_argument('--bundle', type=Path, default=HERE/'agent-int4model-artifacts/peroutput-bundle')
  p.add_argument('--checkpoint', type=Path, default=Path('/tmp/opencode/alpamayo-asius-20261114/model'))
  p.add_argument('--calibration', type=Path, default=WORK/'prefill-calibration/report.prefill-scales.json')
  p.add_argument('--activation', type=Path)
  return p.parse_args()


def main():
  args = parse();report = dict(cpu_only=not args.gpu, complete=False, all_pass=False, shape_mnk=SHAPE,
    variants=['fp8-complete-fixed-scalar', 'flat-a4w4-complete-wgp'], calls_per_graph=CALLS, timing_blocks=3)
  try:
    if not args.gpu: assert os.environ.get('DEV') == 'CPU' and os.environ.get('ALLOW_DEVICE_USAGE') == '1'
    modules();weights, report['assets'] = assets(args);bits, report['activation'] = activation(args)
    check = load(HERE/'agent-int4-fair-check.py')
    with cpu_only_devices(): report['cpu'] = check.cpu_checks(sys.modules[__name__])
    if args.gpu: report['gpu'] = check.gpu_checks(sys.modules[__name__], weights, bits, report, args.output)
    from tinygrad import Device
    report['opened_devices'] = list(Device._opened_devices)
    if not args.gpu: assert not any(d.startswith(('AMD', 'NV', 'CUDA')) for d in Device._opened_devices)
    report.update(complete=True, all_pass=True)
  except BaseException as exc:
    report['failure'] = dict(type=type(exc).__name__, message=str(exc));raise
  finally:
    report['source_hashes'] = source_provenance();save(args.output, report)
  print(json.dumps(dict(complete=True, all_pass=True, cpu_only=not args.gpu, output=str(args.output))), flush=True)


if __name__ == '__main__': main()
