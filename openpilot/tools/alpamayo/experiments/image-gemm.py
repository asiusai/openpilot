"""Experimental multi-image dispatch retaining each image's matrix tile bounds."""
from functools import partial

from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from amd_gemm import pack


def kernel(out, x, weight, x_scale, w_scale, *bias, images, rows):
  columns, width = weight.shape
  m, n, k = UOp.range(round_up(rows, 64), 0), UOp.range(round_up(columns, 64), 1), UOp.range(width, 2)
  def index(row): return (row//16*(width//16)+k//16)*256+(k%16//4)*64+(row%16)*4+k%4
  a = x.flatten().index(index(m).valid(m < rows))
  b = weight.flatten().index(index(n).valid(n < columns))
  value = (a*b).cast(dtypes.float32).reduce(k, arg=Ops.ADD)
  value = value*(x_scale.flatten().index(0)*w_scale.flatten().index(0))
  if bias:
    value = value+bias[0].flatten().index(n.valid(n < columns)).cast(dtypes.float32)
  store = out.flatten().index((m*columns+n).valid((m < rows)&(n < columns))).store(value.cast(out.dtype))
  schedule = Scheduler(store.end(m,n).sink(arg=KernelInfo()), Device[Device.DEFAULT].renderer)
  schedule.convert_loop_to_global()
  ranges = schedule.apply_opt(Opt(OptOps.TC, 0, (-1,0,1)))
  row_axis = ranges[1].axis_id
  for i in (1,0):
    schedule.apply_opt(Opt(OptOps.SPLIT, schedule.rngs.index(ranges[i]), (4,AxisType.UPCAST)))
  ast = schedule.get_optimized_ast(name_override='alpamayo_image_gemm')
  original = next(r for r in schedule.rngs if r.axis_type is AxisType.GLOBAL and r.axis_id == row_axis)
  blocks = int(original.vmax+1)
  expanded = original.replace(src=(original.src[0].const_like(images*blocks),))
  image = expanded//blocks
  mapping = {original:expanded%blocks}
  offsets = {out.base:rows*columns, x.base:round_up(rows,16)*width, x_scale.base:1}
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
  return ast.substitute(indexes,walk=True).substitute({original:expanded},walk=True)


def gemm(x, weight, scales, bias=None):
  assert x.ndim == 3 and weight.ndim == 2 and x.dtype == weight.dtype == dtypes.fp8e4m3
  images, rows, width = x.shape
  assert rows > 64, 'This experimental dispatch needs at least two row tiles'
  assert scales[0].numel() == images and weight.shape[1] == width
  # Each image retains its own padded rows and its original FP8 scale.
  packed = x.pad(((0,0),(0,round_up(rows,16)-rows),(0,0)))
  packed = packed.reshape(images,-1,16,width//16,4,4).permute(0,1,3,4,2,5).contiguous().reshape(-1,width)
  out = Tensor.empty(images,rows,weight.shape[0],device=x.device,dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out,packed,pack(weight),*scales,*(() if bias is None else (bias,)),
                             fxn=partial(kernel,images=images,rows=rows))[0]
