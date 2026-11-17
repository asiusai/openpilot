"""One fixed-shape CK entry. Importing this module never opens a device.

The parent owns the device, serialization, scope lifetime, and acceptance.
The ordinary tinygrad ELF loader, code allocation, and scratch allocator remain in use.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path
import struct

ROOT = Path(__file__).with_name('agent-ck-attention-v2')
NAME = 'alpamayo_ck_d128'
ELF_SHA = 'cf087e9fbd2b094ead1069c4b0f8183ceb10f6da5253bdc904502c9552b0248b'
SOURCE_SHA = '55582878f7ccde8fc70d130c4ea3d354f8b93c627ee65bb734f9750111aeef6e'
QSHAPE, KVSHAPE, LSHAPE = (1, 16, 2048, 128), (1, 8, 2048, 128), (1, 16, 2048)
ACTIVE = False


@lru_cache(None)
def artifacts():
  blob = (ROOT/'native-overlay/kernel.hsaco').read_bytes()
  source = (ROOT/'native-overlay/source.cpp').read_text()
  assert hashlib.sha256(blob).hexdigest() == ELF_SHA
  assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
  return source, blob


def implicit_args():
  # AMDHSA code object V5: implicit arguments follow five eight-byte pointers.
  data = bytearray(256)
  struct.pack_into('<III', data, 0, 16, 32, 1)  # block counts, absolute offsets 40/44/48
  struct.pack_into('<HHH', data, 12, 128, 1, 1) # workgroup sizes, offsets 52/54/56
  struct.pack_into('<H', data, 64, 3)          # grid dimensions, offset 104
  # Remainders, global offsets, and unused implicit fields remain zero.
  return bytes(data)


def argument_words(args):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.runtime.support.hcq2 import layout_args, pack_args
  layout = layout_args(args)
  assert len(layout) == 5 and [offset for offset, _ in layout] == [0, 8, 16, 24, 32]
  assert all(value.dtype == dtypes.uint64 for _, value in layout)
  return pack_args(layout, 40) + [UOp(Ops.BINARY, arg=implicit_args())]


@contextmanager
def abi_scope():
  """Patch only exact CK kernargs in this parent process, then restore the method."""
  global ACTIVE
  from tinygrad.runtime import ops_amd
  from tinygrad.engine.realize import get_call_kernel_args
  from tinygrad.uop.ops import UOp, Ops
  assert not ACTIVE
  original = ops_amd.AMDComputeQueue.kernargs
  def kernargs(queue, call, prg, data):
    if prg.src[0].arg.name != NAME: return original(queue, call, prg, data)
    assert ACTIVE and prg.src[3].arg == artifacts()[1]
    assert queue.dev.arch == 'gfx1200' and tuple(queue.dev.target) == (12, 0, 0)
    assert tuple(prg.arg.global_size) == (16, 32, 1) and tuple(prg.arg.local_size) == (128, 1, 1)
    assert tuple(prg.arg.outs) == (0, 1) and tuple(prg.arg.ins) == (2, 3, 4)
    assert data.kernargs_segment_size == 296 and data.group_segment_size == 9216
    assert data.private_segment_size == 144 and data.wave32 and not data.enable_dispatch_ptr
    assert data.rsrc1 == 1074528668 and data.rsrc2 == 590725 and data.rsrc3 == 0
    words = argument_words(get_call_kernel_args(call, prg, queue.devs))
    return UOp(Ops.LINEAR, src=tuple(words), arg='kernargs').getaddr(queue.devs)
  ops_amd.AMDComputeQueue.kernargs, ACTIVE = kernargs, True
  try: yield
  finally:
    assert ops_amd.AMDComputeQueue.kernargs is kernargs, 'Concurrent ABI patch detected'
    ops_amd.AMDComputeQueue.kernargs, ACTIVE = original, False


def program(out, lse, q, k, v):
  from tinygrad import dtypes
  from tinygrad.uop.ops import KernelInfo, UOp, Ops
  assert out.shape == q.shape == QSHAPE and k.shape == v.shape == KVSHAPE and lse.shape == LSHAPE
  assert out.dtype == q.dtype == k.dtype == v.dtype == dtypes.bfloat16 and lse.dtype == dtypes.float32
  # Both stores are dependency declarations for the supplied ELF, not executed arithmetic.
  value = q.flatten().index(0).load()+k.flatten().index(0).load()+v.flatten().index(0).load()
  sink = UOp.sink(out.flatten().index(0).store(value), lse.flatten().index(0).store(value.cast(dtypes.float32)),
    UOp.special(16, 'gidx0'), UOp.special(32, 'gidx1'), UOp.special(128, 'lidx0'), arg=KernelInfo(name=NAME))
  source, blob = artifacts()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
    UOp(Ops.SOURCE, arg=source), UOp(Ops.BINARY, arg=blob)))


def attention(q, k, v, *, out=None, lse=None):
  from tinygrad import Tensor, dtypes
  assert ACTIVE, 'The parent must retain abi_scope through graph execution'
  assert q.shape == QSHAPE and k.shape == v.shape == KVSHAPE
  assert q.dtype == k.dtype == v.dtype == dtypes.bfloat16 and q.device == k.device == v.device
  out = Tensor.empty(QSHAPE, device=q.device, dtype=dtypes.bfloat16) if out is None else out
  lse = Tensor.empty(LSHAPE, device=q.device, dtype=dtypes.float32) if lse is None else lse
  return tuple(Tensor.custom_kernel(out, lse, q.contiguous(), k.contiguous(), v.contiguous(), fxn=program)[:2])
