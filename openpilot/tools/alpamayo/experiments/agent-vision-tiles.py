"""Optional vision MLP tiles with explicit independent image address ranges.

Hardware validation is pending. Quantization, bias, K16 WMMA order, and BF16
output conversion retain the current expressions. No language kernel changes.
"""
from collections import Counter
from functools import partial
import re
from types import MethodType

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from amd_gemm import pack

TILES = {'fc1': (3, 3), 'fc2': (3, 6)}
SHAPES = {'fc1': (720, 4304, 1152), 'fc2': (720, 1152, 4304)}
NAME = re.compile(r'^vlm\.model\.visual\.encoder\.layers\.(\d+)\.mlp\.(fc[12])$')


def kernel(out, x, weight, x_scale, w_scale, *bias, images, rows, tile_m, tile_n):
  columns, width = weight.shape
  assert images in (1, 2) and out.shape == (images, rows, columns)
  assert x.shape == (images*round_up(rows, 16), width) and x_scale.numel() == images*rows
  assert width % 16 == columns % 16 == 0 and (tile_m, tile_n) in ((4, 4), (3, 3), (3, 6))
  m = UOp.range(round_up(rows, 16*tile_m), 0)
  n, k = UOp.range(round_up(columns, 16*tile_n), 1), UOp.range(width, 2)
  def index(row): return (row//16*(width//16)+k//16)*256+(k%16//4)*64+(row%16)*4+k%4
  a = x.flatten().index(index(m).valid(m < rows))
  b = weight.flatten().index(index(n).valid(n < columns))
  value = (a*b).cast(dtypes.float32).reduce(k, arg=Ops.ADD)
  value = value*(x_scale.flatten().index(m.valid(m < rows))*w_scale.flatten().index(0))
  if bias: value = value+bias[0].flatten().index(n.valid(n < columns)).cast(dtypes.float32)
  store = out.flatten().index((m*columns+n).valid((m < rows)&(n < columns))).store(value.cast(out.dtype))
  schedule = Scheduler(store.end(m, n).sink(arg=KernelInfo()), Device[Device.DEFAULT].renderer)
  schedule.convert_loop_to_global()
  ranges = schedule.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
  row_axis = ranges[1].axis_id
  for axis, count in ((1, tile_m), (0, tile_n)):
    schedule.apply_opt(Opt(OptOps.SPLIT, schedule.rngs.index(ranges[axis]), (count, AxisType.UPCAST)))
  ast = schedule.get_optimized_ast(name_override=f'agent_vision_tiles_{tile_m}x{tile_n}_b{images}')
  # Extend only the row workgroup dispatch. Each image retains the complete
  # single-image reduction and padding predicates, as in image-gemm.py.
  original = next(r for r in schedule.rngs if r.axis_type is AxisType.GLOBAL and r.axis_id == row_axis)
  blocks = int(original.vmax+1)
  expanded = original.replace(src=(original.src[0].const_like(images*blocks),))
  image, mapping = expanded//blocks, {original: expanded % blocks}
  offsets = {out.base: rows*columns, x.base: round_up(rows, 16)*width, x_scale.base: rows}
  indexes, seen = {}, set()
  for node in ast.toposort():
    if node.op is not Ops.INDEX: continue
    updated = node.substitute(mapping, walk=True)
    if node.src[0].base in offsets:
      seen.add(node.src[0].base)
      assert len(updated.src) == 2
      updated = updated.replace(src=(updated.src[0], updated.src[1]+image*offsets[node.src[0].base]))
    indexes[node] = updated
  assert seen == set(offsets)
  return ast.substitute(indexes, walk=True).substitute({original: expanded}, walk=True)


def pack_images(x):
  # uint8 permits a byte-exact CPU layout check. Pinned CPU FP8 padding flushes
  # subnormals; the GPU gate checks the real FP8 packing expression separately.
  assert x.ndim == 3 and x.dtype in (dtypes.fp8e4m3, dtypes.uint8)
  images, rows, width = x.shape
  assert width % 16 == 0
  padded = x.pad(((0, 0), (0, round_up(rows, 16)-rows), (0, 0)))
  return padded.reshape(images, -1, 16, width//16, 4, 4).permute(0, 1, 3, 4, 2, 5).contiguous().reshape(-1, width)


def row_scales(scale, images, rows):
  assert scale.dtype == dtypes.float32
  if scale.numel() == images*rows: return scale.reshape(-1)
  assert scale.numel() == images, 'Require separate image scales or existing row scales'
  return scale.reshape(images, 1).expand(images, rows).reshape(-1).contiguous()


def packed(packed_x, packed_weight, scales, bias, images, rows, tile, out=None):
  if out is None: out = Tensor.empty(images, rows, packed_weight.shape[0], device=packed_x.device, dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out, packed_x, packed_weight, *scales, bias,
    fxn=partial(kernel, images=images, rows=rows, tile_m=tile[0], tile_n=tile[1]))[0]


def gemm(x, weight, scales, bias, tile):
  images, rows, width = x.shape
  assert x.dtype == weight.dtype == dtypes.fp8e4m3 and bias.dtype == dtypes.bfloat16
  assert weight.shape[1] == width
  scales = (row_scales(scales[0], images, rows), scales[1])
  return packed(pack_images(x), pack(weight), scales, bias, images, rows, tile)


def install(model):
  """Install after batched-vision.install, before any encoder graph capture."""
  assert not hasattr(model, 'agent_vision_tiles'), 'Already installed'
  assert model.dtype == dtypes.bfloat16 and hasattr(model, 'grouped_quantize')
  # Fail closed on precision restoration. No BF16 operation enters this hook.
  protected = ['vlm.model.visual.embeddings.patch_embedding.weight']
  protected += [f'vlm.model.visual.encoder.layers.{i}.mlp.{fc}.weight' for i in (0, 1) for fc in TILES]
  assert all(model.weights[name].dtype == dtypes.bfloat16 and name not in model.weight_scales for name in protected)
  old_linear = model.linear
  info = {'candidate': 'vision-mlp-image-tiles', 'hardware_exactness_validated': False,
    'tiles': TILES.copy(), 'trace_calls': Counter(), 'protected_bf16_weights': protected}
  def linear(self, x, name, quantized_input=None, heads=None):
    match = NAME.fullmatch(name)
    if (match is None or int(match[1]) < 2 or x.ndim != 3 or x.shape[0] not in (1, 2) or x.shape[1] != 720
        or x.dtype != dtypes.bfloat16 or heads is not None or name+'.weight' not in self.weight_scales
        or self.weights[name+'.weight'].dtype != dtypes.fp8e4m3):
      return old_linear(x, name, quantized_input, heads)
    fc, images = match[2], x.shape[0]
    weight, bias = self.weights[name+'.weight'], self.weights.get(name+'.bias')
    assert (*x.shape[1:2], weight.shape[0], x.shape[-1]) == SHAPES[fc]
    assert bias is not None and bias.dtype == dtypes.bfloat16
    # Preserve exactly one call at the same calibrated/grouped quantizer index.
    quantized, scale = self.quantize(x) if quantized_input is None else quantized_input
    assert isinstance(quantized, Tensor) and quantized.numel() == x.numel()
    info['trace_calls'][f'{name}/batch{images}'] += 1
    return gemm(quantized.reshape(images, 720, x.shape[-1]), weight,
      (scale, self.weight_scales[name+'.weight']), bias, TILES[fc]).reshape(*x.shape[:-1], weight.shape[0])
  model.linear = MethodType(linear, model)
  model.agent_vision_tiles = info
  return info
