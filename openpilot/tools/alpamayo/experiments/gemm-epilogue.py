"""Fuse the calibrated language MLP activation into the first FP8 matrix kernel."""
from functools import partial
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


def program(out, x, weight, x_scale, w_scale, output_scale, *bias, rows):
  columns, width = weight.shape
  m, n, k = UOp.range(round_up(rows,64),0), UOp.range(round_up(columns,64),1), UOp.range(width,2)
  def index(row): return (row//16*(width//16)+k//16)*256 + (k%16//4)*64 + (row%16)*4 + k%4
  a = x.flatten().index(index(m).valid(m < rows))
  b = weight.flatten().index(index(n).valid(n < columns))
  value = (a*b).cast(dtypes.float32).reduce(k,arg=Ops.ADD)
  value = value*(x_scale.flatten().index(0)*w_scale.flatten().index(0))
  if bias: value = value+bias[0].flatten().index(n.valid(n<columns)).cast(dtypes.float32)
  # Preserve the original BF16 rounding boundaries before the FP8 conversion.
  value = value.cast(dtypes.bfloat16).maximum(0)
  value = (value*value).cast(dtypes.bfloat16).cast(dtypes.float32)
  value = (value/output_scale.flatten().index(0)).maximum(-448).minimum(448).cast(dtypes.fp8e4m3)
  location = (m//16*(columns//16)+n//16)*256+(n%16//4)*64+(m%16)*4+n%4
  store = out.flatten().index(location.valid((m < out.shape[0]) & (n < columns))).store((m<rows).where(value,0))
  schedule = Scheduler(store.end(m,n).sink(arg=KernelInfo()),Device[Device.DEFAULT].renderer)
  schedule.convert_loop_to_global()
  ranges = schedule.apply_opt(Opt(OptOps.TC,0,(-1,0,1)))
  for i in (1,0): schedule.apply_opt(Opt(OptOps.SPLIT,schedule.rngs.index(ranges[i]),(4,AxisType.UPCAST)))
  return schedule.get_optimized_ast(name_override='alpamayo_fp8_mlp_packed')


def mlp_input(x, weight, x_scale, w_scale, output_scale, rows, bias=None):
  out = Tensor.empty(round_up(rows,16),weight.shape[0],device=x.device,dtype=dtypes.fp8e4m3)
  return Tensor.custom_kernel(out,x,weight,x_scale,w_scale,output_scale,*(() if bias is None else (bias,)),
                             fxn=partial(program,rows=rows))[0].realize()
