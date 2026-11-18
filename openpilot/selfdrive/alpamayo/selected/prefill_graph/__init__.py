"""Bind the six-head prefill program sequence to the current inputs, weights and KV buffers."""
import ast
import hashlib
import json
from pathlib import Path

from tinygrad import Device, Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.engine.jit import CapturedJit, _TinyJit, _prepare_jit_inputs
from tinygrad.engine.realize import compile_linear, link_linear
from tinygrad.helpers import prod
from tinygrad.uop.ops import Ops, UOp

from ..frozen import ASSETS
from .programs import DTYPES, TARGET as TARGET, load_programs, require

ASSET = Path(__file__).with_name('gfx1200.json.gz')
SHA256 = 'f6f935fa1a1c9e78a8732e31a700f07209245e6dafebd8573ccc5dad8279d5c9'
ORDER = 'fe67296128f53f6ab463f2f5750216b865da3da1c6ccbad0c53dae14084c9e7c'
INPUT_SHAPES = ((1, 2048, 2048), (1, 1, 2048, 128), (1, 1, 2048, 128))
LENGTH = 'valid_prefill_draft_length'


def check_sources(sources):
  root = ASSETS.parents[1]
  for filename, methods in sources.items():
    data = (root / filename).read_bytes()
    lines, tree = data.splitlines(keepends=True), ast.parse(data)
    for name, digest in methods.items():
      node = tree
      for part in name.split('.'):
        node, = (n for n in node.body if getattr(n, 'name', None) == part)
      start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
      require(hashlib.sha256(b''.join(lines[start:node.end_lineno])).hexdigest() == digest, 'Prefill source changed: ' + name)


def graph_inputs(inputs, device):
  for t, shape in zip(inputs, INPUT_SHAPES, strict=True):
    b = t.uop.base
    if (t.device != device or t.dtype != dtypes.bfloat16 or t.shape != shape or b.op is not Ops.BUFFER or b.dtype != t.dtype or
        b.max_numel() != prod(shape) or t.uop is not b.reshape(shape) or b.realized is None or b.buffer.base.options.external_ptr is not None):
      return None
  x, cos, sin = (t.uop.base.buffer for t in inputs)
  if x is not x.base or x in (cos.base, sin.base) or cos is sin:
    return None
  if cos is cos.base and sin is sin.base:
    return inputs
  # The angle cache packs cos and sin at the start of one allocation. Give each parameter its original independent storage.
  if cos is cos.base or cos.base is not sin.base or cos.offset != 0 or sin.offset != cos.nbytes:
    return None
  separate = []
  for t in inputs[1:]:
    u = UOp.new_buffer(device, t.numel(), t.dtype)
    u.buffer.ensure_allocated().copy_from(t.uop.base.buffer)
    separate.append(Tensor(u).reshape(t.shape))
  return inputs[0], *separate


class PrefillWeights:
  prefixes = ('vlm.model.language_model.', 'vlm.lm_head.')

  def __init__(self, model):
    # Capture immutable owners immediately after the selected loader verified the checkpoint and built the head and scales.
    self.model = model
    self.bulk, self.mapping = model._bulk_storage, model.output_token_mapping.uop
    self.roles = {field: {k: t.uop for k, t in getattr(model, field).items() if k.startswith(self.prefixes)}
                  for field in ('weights', 'weight_scales', 'packed_weights', 'fixed')}

  def check(self):
    m = self.model
    require(m._bulk_storage is self.bulk and m.output_token_mapping.uop is self.mapping, 'Prefill bulk storage or token mapping changed')
    for field, roles in self.roles.items():
      table = getattr(m, field)
      require(roles.keys() == {k for k in table if k.startswith(self.prefixes)}, 'Prefill weight inventory changed: ' + field)
      for key, view in roles.items():
        require(table[key].uop is view, 'Prefill weight binding changed: ' + key)


def bind_owners(plan, weights, inputs, outputs):
  m = weights.model
  weights.check()
  buffers, writable, separate = [], set(), {weights.bulk}
  for i, (field, key, shape, dtype, offset) in enumerate(plan['owners']):
    if field == 'input':
      t = inputs[key]
    elif field == 'output':
      t = outputs[key]
    elif field == 'mapping':
      t = m.output_token_mapping
    else:
      t = getattr(m, field)[key]
    b = t.uop.base
    # pack(unpack(weight)).realize() can retain intermediate reshapes. The command reads the complete physical packed buffer.
    view = t.uop
    if field == 'packed_weights':
      while view.op is Ops.RESHAPE:
        view = view.src[0]
      canonical = view is b
    else:
      canonical = view is b.reshape(tuple(shape))
    require(b.op is Ops.BUFFER and b.device == weights.bulk.device and b.dtype.name == dtype and t.shape == tuple(shape) and canonical,
            f'Unsupported prefill owner layout: {field}/{key}')
    buf = b.buffer
    require(buf.nbytes == prod(shape) * b.dtype.itemsize and buf.options.external_ptr is None, 'Prefill owner extent differs')
    if offset is not None:
      require(buf.base is weights.bulk and buf.offset == offset, 'Prefill checkpoint range differs: ' + key)
      if field == 'packed_weights':
        require(m.weights[key + '.weight'].uop.base.buffer is buf, 'Prefill packed weight alias differs: ' + key)
    else:
      require(buf is buf.base and buf.offset == 0 and buf not in separate, 'Unexpected prefill owner alias')
      separate.add(buf)
    buffers.append(UOp.param(key, b.dtype, b.max_numel(), b.device) if field == 'input' else b)
    if field in ('cache', 'output'):
      writable.add(i)
  return buffers, writable


def numerical_linear(plan, programs, buffers, writable, device):
  size = plan['scratch_bytes']
  arena = UOp.alloc((size,), dtypes.int8, device=device)
  arguments = buffers.copy()
  for offset, nbytes, dtype in plan['views']:
    dt = DTYPES[dtype]
    require(nbytes > 0 and offset >= 0 and offset + nbytes <= size and offset % dt.itemsize == nbytes % dt.itemsize == 0,
            'Invalid prefill scratch slice')
    arguments.append(UOp(Ops.SHRINK, src=(arena, UOp.const(offset), UOp.const(nbytes))).bitcast(dt))
  calls, variables = [], set()
  for key, slots in plan['calls']:
    p = programs[key]
    require(all(type(i) is int and 0 <= i < len(arguments) for i in slots), 'Invalid prefill operand slot')
    args = [arguments[i] for i in slots]
    require(all(slots[i] >= len(buffers) or slots[i] in writable for i in p.arg.outs), 'Prefill writes a read-only binding')
    for param in p.arg.params:
      if param.addrspace is AddrSpace.ALU:
        require(param.is_variable and not param.is_bound_var, 'Unexpected constant prefill scalar')
        variables.add((param.expr, param.vmin, param.vmax))
      else:
        a = args[param.arg.slot]
        require(a.dtype == param.dtype and a.nbytes() == param.nbytes(), 'Prefill operand ABI differs')
    calls.append(p.call(*args))
  require(variables == {(LENGTH, 1, 2042)} and hashlib.sha256(b''.join(c.body.key for c in calls)).hexdigest() == ORDER,
          'Prefill program sequence or scalar ABI differs')
  return UOp(Ops.LINEAR, src=tuple(calls))


class PrefillGraph:
  def __init__(self, weights, inputs, length):
    m, device = weights.model, weights.bulk.device
    plan, programs = load_programs(ASSET, SHA256, Device[device].renderer)
    check_sources(plan['sources'])
    expected = plan['checkpoint']
    artifacts = json.loads((ASSETS / 'artifacts.json').read_bytes())
    require(all(artifacts[k] == v for k, v in expected.items()) and m.weight_digest == expected['fp8_weights_sha256'] and
            weights.bulk.nbytes == artifacts['fp8_bytes'], 'Prefill checkpoint or calibration differs')
    require(len(programs) == 30 and len(plan['calls']) == 563 and len(plan['owners']) == 541 and plan['scratch_bytes'] == 48234496,
            'Incomplete prefill graph')
    self.weights = weights
    self.outputs = (Tensor(UOp.new_buffer(device, 2112, dtypes.bfloat16)).reshape(1, 1, 2112), Tensor(UOp.new_buffer(device, 12, dtypes.int32)))
    buffers, writable = bind_owners(plan, weights, inputs, self.outputs)
    linear = numerical_linear(plan, programs, buffers, writable, device)
    iu, values, names, info = _prepare_jit_inputs((*inputs, length), {})
    require(iu == [t.uop.base for t in inputs] and names == [0, 1, 2] and set(values) == {LENGTH}, 'Invalid prefill inputs')
    compiled = compile_linear(linear, beam=0, input_uops=iu, cache=False)
    keys = [k[4] for c in compiled.src for k in c.without_after.arg.aux.kernels]
    require(keys == [c.body.key for c in linear.src], 'HCQ changed prefill program identities')
    captured = CapturedJit(self.outputs, compiled, names, info)
    # With three independent input allocations, CapturedJit's link cache key is the empty alias tuple.
    captured._linears[()] = (link_linear(compiled), {})
    self.jit = _TinyJit(None, captured=captured)
    self.kv_views = tuple(t.uop for t in m.cache)
    self.output_views = tuple(t.uop for t in self.outputs)
    # A Buffer can survive deallocation. Keep its storage identity as well as its view for the lifetime of the linked commands.
    closed = {b.buffer for b in buffers if b.op is Ops.BUFFER}
    self.roots = {b.base for b in closed}
    self.storages = [(b, b.get_storage()) for b in closed | self.roots]

  def __call__(self, inputs, length):
    self.weights.check()
    require(tuple(t.uop for t in self.weights.model.cache) == self.kv_views and tuple(t.uop for t in self.outputs) == self.output_views,
            'Prefill KV or output binding changed')
    require(all(b.is_allocated() and b.get_storage() is storage for b, storage in self.storages), 'Prefill storage changed after binding')
    require(self.roots.isdisjoint(t.uop.base.buffer.base for t in inputs), 'Prefill input aliases a closed binding')
    require(length.is_bound_var and length.expr == LENGTH and 1 <= length.arg.val <= 2042, 'Invalid prefill prompt length')
    return self.jit(*inputs, length)
