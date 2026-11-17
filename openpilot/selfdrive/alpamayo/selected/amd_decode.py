from functools import partial

from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from openpilot.selfdrive.alpamayo.amd_attention import _warp_reduce
from .amd_gemm import pack


def _gemv(out, x, weight, x_scale, w_scale, *bias, packed=True):
  rows, width = weight.shape
  group = UOp.range(rows//(16 if packed else 4), 0, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  if packed:
    row, k = group*16+lane%16, UOp.range(width//64, 2, AxisType.LOOP)
    cols = [(k*4+wave)*16+(lane//16)*4+j%4+(j//4)*8 for j in range(8)]
    indices = [(group*(width//16)+c//16)*256+(c%16//4)*64+(row%16)*4+c%4 for c in cols]
  else:
    row, k = group*4+wave, UOp.range(width//128, 2, AxisType.LOOP)
    cols = [k*128+lane*4+j for j in range(4)]
    indices = [row*width+c for c in cols]
  value = sum(x.flatten().index(c).cast(dtypes.float32) * weight.flatten().index(i).cast(dtypes.float32)
              for c, i in zip(cols, indices, strict=True)).reduce(k, arg=Ops.ADD)
  if packed:
    shuffle = UOp.custom_function('llvm.amdgcn.ds.swizzle', dtype=dtypes.int32)
    value = value + shuffle.call(value.bitcast(dtypes.int32), UOp.const(0x1f | (16 << 10), dtypes.int32)).bitcast(dtypes.float32)
    partials = UOp.alloc((64,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    stored = partials.index((wave*16+lane%16).valid(lane < 16)).store(value)
    value = sum(partials.after(stored).index(i*16+lane%16) for i in range(4))
    valid = wave.eq(0) & (lane < 16)
  else:
    value, valid = _warp_reduce(value), lane.eq(0)
  value = value * (x_scale.flatten().index(0) * w_scale.flatten().index(0))
  if bias:
    value = value + bias[0].flatten().index(row).cast(dtypes.float32)
  store = out.flatten().index(row.valid(valid)).store(value.cast(out.dtype))
  return store.end(group, wave, lane).sink(arg=KernelInfo(name='alpamayo_fp8_gemv', opts_to_apply=()))


def gemv(x, weight, scales, bias=None, packed=True, dtype=dtypes.bfloat16):
  out = Tensor.empty(weight.shape[0], dtype=dtype, device=x.device)
  return Tensor.custom_kernel(out, x, pack(weight) if packed else weight, *scales, *(() if bias is None else (bias,)),
                              fxn=partial(_gemv, packed=packed))[0]
