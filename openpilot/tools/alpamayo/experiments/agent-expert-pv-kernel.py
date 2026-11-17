"""Experimental packed BF16 PV schedules. No model installation or GPU access."""
import math
from tinygrad import Device, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

ORIGINAL_SHA256 = 'fa9d2f01c6557c97541f8a62e4e02ee7cf2608584787cad7d42b9922a00866f4'
TINYGRAD_REVISION = '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
TILES = ((1, 4), (2, 2))
CASES = {'integer': (1, 2, 2, 16, 16, 16), 'integer-tail-gqa': (2, 4, 2, 17, 19, 33),
         'expert': (1, 16, 8, 64, 128, 2368)}


def kernel(out, a, b, shape, causal=False, scaled=False, *, tile_m=4, tile_n=4):
  """Original _matmul arithmetic; only output tile ranges and splits differ.

  The 4x4 control must compile to the original LLVM and binary, byte for byte.
  This is only a PV experiment: score scaling and causal masking stay upstream.
  """
  assert (tile_m, tile_n) in ((4, 4), *TILES)
  assert not causal and not scaled
  assert out.dtype == a.dtype == b.dtype == dtypes.bfloat16
  batch, heads, kv_heads, rows, columns, width = shape
  assert heads % kv_heads == 0 and min(shape) > 0
  padded_rows, padded_columns, padded_width = (round_up(n, 16) for n in (rows, columns, width))
  head = UOp.range(batch*heads, 0)
  m, n, k = UOp.range(round_up(rows, 16*tile_m), 1), UOp.range(round_up(columns, 16*tile_n), 2), UOp.range(padded_width, 3)

  def index(h, row, length):
    return ((h*(length//16)+row//16)*(padded_width//16)+k//16)*256 + (k%16//4)*64+(row%16)*4+k%4

  aa = a.flatten().index(index(head, m, padded_rows).valid(m < rows))
  kv_head = (head//heads)*kv_heads+(head % heads)//(heads//kv_heads)
  bb = b.flatten().index(index(kv_head, n, padded_columns).valid(n < columns))
  value = (aa*bb).cast(dtypes.float32).reduce(k, arg=Ops.ADD).cast(out.dtype)
  if scaled: value = value*UOp.const(width**-0.5, out.dtype)
  if causal: value = (n <= m).where(value, UOp.const(-math.inf, out.dtype))
  store = out.flatten().index(((head*rows+m)*columns+n).valid((m < rows)&(n < columns))).store(value)
  schedule = Scheduler(store.end(head, m, n).sink(arg=KernelInfo()), Device[Device.DEFAULT].renderer)
  schedule.convert_loop_to_global()
  ranges = schedule.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
  # Tensor-core ranges are returned in N, M, K order.
  for i, size in ((1, tile_m), (0, tile_n)):
    if size > 1: schedule.apply_opt(Opt(OptOps.SPLIT, schedule.rngs.index(ranges[i]), (size, AxisType.UPCAST)))
  name = 'alpamayo_attention_values' if (tile_m, tile_n) == (4, 4) else f'alpamayo_expert_pv_{tile_m}x{tile_n}'
  return schedule.get_optimized_ast(name_override=name)
