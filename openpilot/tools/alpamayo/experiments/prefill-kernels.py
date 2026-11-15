"""Use pinned tinygrad's RDNA4 online-softmax attention for causal prefill."""
from functools import partial
from inspect import unwrap
from types import MethodType, FunctionType
import os
from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.amd import _amd_flash_attention, BLOCK_M, BLOCK_N
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import AddrSpace

BLOCK_N = int(os.environ.get('ALPAMAYO_FLASH_BLOCK_N', BLOCK_N))
assert BLOCK_N in (32, 64)


def fix_dispatch_packet():
  # Pinned runtime puts Python grid integers into UOp.LINEAR.src for kernels
  # requiring the dispatch pointer. Keep its packet bytes, but type those words.
  from tinygrad.runtime import ops_amd
  if getattr(ops_amd.dispatch_packet, '_alpamayo_typed_grid', False):
    return
  original = ops_amd.dispatch_packet
  def typed(*args, **kwargs):
    return [UOp.const(word, dtypes.uint32) if isinstance(word, int) else word for word in original(*args, **kwargs)]
  typed._alpamayo_typed_grid = True
  ops_amd.dispatch_packet = typed


def warp_reduce(value, maximum=False, full_wave=False):
  shuffle = UOp.custom_function('llvm.amdgcn.ds.swizzle', dtype=dtypes.int32)
  for shift in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if value.op is Ops.INDEX and value.addrspace == AddrSpace.REG:
      value = value.load()
    other = shuffle.call(value.bitcast(dtypes.int32), UOp.const(0x1f | (shift << 10), dtypes.int32)).bitcast(dtypes.float32)
    value = value.maximum(other) if maximum else value + other
  return value


# The upstream kernel emits HIP source for lane shuffles. Substitute LLVM intrinsics
# in a private function namespace without changing tinygrad's global implementation.
source_flash = unwrap(_amd_flash_attention)
llvm_flash = FunctionType(source_flash.__code__, dict(source_flash.__globals__, warp_reduce=warp_reduce, BLOCK_N=BLOCK_N),
                          source_flash.__name__, source_flash.__defaults__)


def attention(q, k, v):
  fix_dispatch_packet()
  batch, heads, queries, width = q.shape
  assert width == 128 and k.shape[2] == queries
  rows, keys = round_up(queries, BLOCK_M), round_up(queries, BLOCK_N)
  query = q.pad(((0, 0), (0, 0), (0, rows-queries), (0, 0))).half().contiguous()
  cache = Tensor.stack(k, v).pad(((0, 0), (0, 0), (0, 0), (0, keys-queries), (0, 0))).half().contiguous()
  out = Tensor.empty(batch*heads, rows, width, dtype=dtypes.float32, device=q.device)
  out = Tensor.custom_kernel(out, query.reshape(batch*heads, rows, width), cache,
    fxn=partial(llvm_flash, valid_kv_len=queries, q_start=0, rdna4=True))[0]
  return out.reshape(batch, heads, rows, width)[:, :, :queries].cast(q.dtype)


def install(model):
  assert model.fp8_wmma
  original = model.attention
  def attended(self, q, k, v, mask=None, causal=False):
    if causal and mask is None and q.shape[-1] == 128 and q.shape[2] > 1:
      return attention(q, k, v)
    return original(q, k, v, mask, causal)
  model.attention = MethodType(attended, model)
