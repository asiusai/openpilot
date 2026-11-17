"""Integer-only whole-buffer validation helpers. No model arithmetic or hashes."""
from functools import partial
import math

BLOCK = 4096
FORMATS = {'activation':'bf16', 'partial':'f32', 'maximum':'f32', 'scale':'f32',
           'fp8':'fp8', 'packed':'fp8', 'row_scale':'f32'}
MASK = {'bf16':0x7f80, 'f32':0x7f800000, 'fp8':0x7f}
POISON = {'bf16':0x7fc1, 'f32':0x7fc00001, 'fp8':0x7f}
ONE = {'bf16':0x3f80, 'f32':0x3f800000, 'fp8':0x38}


def uint_dtype(fmt):
  from tinygrad import dtypes
  return {'bf16':dtypes.uint16, 'f32':dtypes.uint32, 'fp8':dtypes.uint8}[fmt]


def raw_view(tensor, fmt):
  from tinygrad import Tensor
  from tinygrad.uop.ops import UOp, Ops
  assert tensor.dtype.itemsize == uint_dtype(fmt).itemsize
  result = Tensor(UOp.from_buffer(tensor.uop.buffer,uint_dtype(fmt)))
  assert result.uop.base.op is Ops.BUFFER and result.uop.is_realized and not result.uop.is_virtual
  assert result.uop.buffer.base is tensor.uop.buffer.base and result.uop.buffer.offset == tensor.uop.buffer.offset
  assert result.nbytes() == tensor.nbytes()
  return result


def reduce_flags(a, b, valid, item, fmt):
  from tinygrad import dtypes
  from tinygrad.uop.ops import Ops
  mismatch = ((a != b) & valid).cast(dtypes.uint32).reduce(item, arg=Ops.MAX)
  left_bad = ((a & MASK[fmt]).eq(MASK[fmt]) & valid).cast(dtypes.uint32).reduce(item, arg=Ops.MAX)
  right_bad = ((b & MASK[fmt]).eq(MASK[fmt]) & valid).cast(dtypes.uint32).reduce(item, arg=Ops.MAX)
  return mismatch | (left_bad << 1) | (right_bad << 2)


def compare(flags, left, right, *, fmt):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp, KernelInfo
  size = left.numel()
  assert left.dtype == right.dtype == uint_dtype(fmt) and right.numel() == size
  assert flags.dtype == dtypes.uint32 and flags.numel() == (size+BLOCK-1)//BLOCK
  group, item = UOp.range(flags.numel(), 0), UOp.range(BLOCK, 1)
  index = group*BLOCK+item
  valid = index < size
  a = left.flatten().index(index.valid(valid)).load()
  b = right.flatten().index(index.valid(valid)).load()
  # Three independent Boolean maxima implement an exact three-bit OR reduction.
  value = reduce_flags(a,b,valid,item,fmt)
  return flags.flatten().index(group).store(value).end(group).sink(arg=KernelInfo(name='agent_vision_gelu_all_bits_'+fmt))


def pattern(flags, source, control, *, fmt):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp, KernelInfo
  size = source.numel()
  assert source.dtype == uint_dtype(fmt) and flags.dtype == dtypes.uint32 and flags.numel() == (size+BLOCK-1)//BLOCK
  group, item = UOp.range(flags.numel(),0), UOp.range(BLOCK,1)
  index = group*BLOCK+item
  a = source.flatten().index(index.valid(index<size)).load()
  b = control.flatten().index(0).load().cast(source.dtype)
  value = reduce_flags(a,b,index<size,item,fmt)
  return flags.flatten().index(group).store(value).end(group).sink(arg=KernelInfo(name='agent_vision_gelu_constant_bits_'+fmt))


def fill(out, control):
  from tinygrad.uop.ops import UOp, KernelInfo
  item = UOp.range(out.numel(), 0)
  value = control.flatten().index(0).load().cast(out.dtype)
  return out.flatten().index(item).store(value).end(item).sink(arg=KernelInfo(name='agent_vision_gelu_fill_bits'))


def poke(out, control):
  from tinygrad.uop.ops import KernelInfo
  index = control.flatten().index(1).load()
  value = control.flatten().index(0).load().cast(out.dtype)
  return out.flatten().index(index).store(value).sink(arg=KernelInfo(name='agent_vision_gelu_poke_bits'))


def copy(out, source):
  from tinygrad.uop.ops import UOp, KernelInfo
  assert out.dtype == source.dtype and out.numel() == source.numel()
  item = UOp.range(out.numel(), 0)
  return out.flatten().index(item).store(source.flatten().index(item).load()).end(item).sink(arg=KernelInfo(name='agent_vision_gelu_copy_bits'))


def key(kind, size, fmt): return f'{kind}-{fmt}-{size}'


def specifications(layout):
  from tinygrad import dtypes
  result = {}
  for row in layout[1:]:
    fmt = FORMATS[row['name']]
    size, dt = math.prod(row['shape']), uint_dtype(fmt)
    blocks = (size+BLOCK-1)//BLOCK
    result[key('compare', size, fmt)] = (partial(compare, fmt=fmt), [((blocks,),dtypes.uint32), ((size,),dt), ((size,),dt)])
    result[key('pattern', size, fmt)] = (partial(pattern, fmt=fmt), [((blocks,),dtypes.uint32), ((size,),dt), ((2,),dtypes.uint32)])
    for kind, fn in (('fill', fill), ('poke', poke)):
      result[key(kind, size, fmt)] = (fn, [((size,),dt), ((2,),dtypes.uint32)])
    result[key('copy', size, fmt)] = (copy, [((size,),dt), ((size,),dt)])
    result[key('fill', blocks, 'f32')] = (fill, [((blocks,),dtypes.uint32), ((2,),dtypes.uint32)])
  return result


def compile_all(layout, renderer, micro):
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp, Ops
  programs, records = {}, {}
  for name, (fn, specs) in specifications(layout).items():
    ast = fn(*(UOp.placeholder(shape, dt, slot) for slot, (shape, dt) in enumerate(specs)))
    assert not any(u.op in (Ops.EXP2, Ops.RECIP, Ops.WMMA) for u in ast.toposort())
    program = to_program(ast, renderer)
    programs[name] = program
    records[name] = dict(micro.program_record(program), specs=[dict(shape=list(s), dtype=str(dt)) for s, dt in specs])
    assert all(records[name]['resources'][field] == 0 for field in
               ('private_segment_fixed_size', 'vgpr_spill_count', 'sgpr_spill_count'))
  return programs, records


def positions(size):
  return sorted({0, size//2, size-1, min(BLOCK-1, size-1), min(BLOCK, size-1), (size-1)//BLOCK*BLOCK})


def expected_flags(a, b, fmt):
  """Independent NumPy oracle; full arrays, with no output fingerprint comparison."""
  import numpy as np
  assert a.shape == b.shape and a.dtype == b.dtype
  a, b = a.reshape(-1), b.reshape(-1)
  bad = (a != b).astype(np.uint32) | (((a & MASK[fmt]) == MASK[fmt]).astype(np.uint32) << 1) | \
        (((b & MASK[fmt]) == MASK[fmt]).astype(np.uint32) << 2)
  padded = np.pad(bad, (0, (-len(bad)) % BLOCK))
  return np.bitwise_or.reduce(padded.reshape(-1, BLOCK), axis=1)


class Dispatch:
  """Capture helper programs separately from every measured numerical graph."""
  def __init__(self, programs, micro):
    self.programs, self.micro, self.jits, self.identities = programs, micro, {}, {}

  def __call__(self, name, *tensors):
    from tinygrad import Tensor, TinyJit
    from tinygrad.runtime.support import hcq2
    from tinygrad.uop.ops import Ops
    program = self.programs[name]
    if name not in self.jits:
      def run(*values): return Tensor.custom_kernel(*values, fxn=lambda *unused:program)[0].realize()
      self.jits[name] = TinyJit(run)
      self.jits[name].cnt = 1
    jit = self.jits[name]
    original = hcq2._build_queues
    def build(ctx):
      assert len(ctx.batch) == 1
      call = ctx.batch[0][0]
      actual = self.micro.program_record(call.body)
      assert actual == self.micro.program_record(program)
      for slot,arg in enumerate(call.src[1:]):
        base,lane,offset = hcq2.unwrap_lane(arg)
        assert base.op is Ops.PARAM and base.arg.slot == slot and lane is None and offset == 0
      self.identities[name] = actual
      return original(ctx)
    if jit.cnt == 1: hcq2._build_queues = build
    try: return jit(*tensors)
    finally: hcq2._build_queues = original

  def audit(self):
    from tinygrad.engine.realize import HCQInfo
    for name, jit in self.jits.items():
      assert self.identities[name] == self.micro.program_record(self.programs[name])
      calls = jit.captured._linear.src
      assert len(calls) == 1 and isinstance(info:=calls[0].without_after.arg.aux, HCQInfo)
      assert len(info.kernels) == 1 and info.kernels[0][4].hex() == self.identities[name]['program_key']
    return self.identities


class Verifier:
  def __init__(self, io, layout, programs, micro):
    from tinygrad import dtypes
    import numpy as np
    self.io, self.micro, self.device, self.layout = io, micro, io.device, layout
    self.dispatch = Dispatch(programs, micro)
    words = {0xffffffff, *POISON.values(), *ONE.values()}
    self.controls = {word:io.allocate((2,), dtypes.uint32, np.array([word, 0], np.uint32).tobytes()) for word in words}
    self.control = io.allocate((2,), dtypes.uint32, np.zeros(2, np.uint32).tobytes())
    self.flags = {row['name']:io.allocate(((math.prod(row['shape'])+BLOCK-1)//BLOCK,), dtypes.uint32) for row in layout[1:]}
    self.records = []

  def raw(self, tensor, fmt): return raw_view(tensor,fmt)

  def fill(self, tensor, fmt, word):
    self.dispatch(key('fill', tensor.numel(), fmt), self.raw(tensor, fmt), self.controls[word])
    self.io.counters['gpu_fill_bytes'] += tensor.nbytes()

  def poke(self, tensor, fmt, index, word):
    import numpy as np
    self.device.synchronize()
    self.io.write(self.control, np.array([word, index], np.uint32).tobytes())
    self.dispatch(key('poke', tensor.numel(), fmt), self.raw(tensor, fmt), self.control)

  def copy(self, dest, source, fmt):
    assert dest.shape == source.shape and dest.dtype == source.dtype
    assert dest.uop.buffer.base is not source.uop.buffer.base
    self.dispatch(key('copy', source.numel(), fmt), self.raw(dest, fmt), self.raw(source, fmt))
    self.io.counters['gpu_copy_bytes'] += source.nbytes()

  def poison(self, outputs):
    for label, tensor in zip(self.micro.LABELS[1:], outputs, strict=True): self.fill(tensor, FORMATS[label], POISON[FORMATS[label]])
    for label, tensor in zip(self.micro.LABELS[1:], outputs, strict=True): self.check_pattern(label,tensor,POISON[FORMATS[label]],'complete-poison')

  def check_pattern(self,label,tensor,word,stage,expected=None):
    import numpy as np
    fmt,flags = FORMATS[label],self.flags[label]
    self.fill(flags,'f32',0xffffffff)
    self.dispatch(key('pattern',tensor.numel(),fmt),flags,self.raw(tensor,fmt),self.controls[word])
    self.device.synchronize()
    values = np.frombuffer(self.io.read(flags),np.uint32).copy()
    if expected is None: expected = np.full(flags.numel(),6 if word & MASK[fmt] == MASK[fmt] else 0,np.uint32)
    np.testing.assert_array_equal(values,expected)
    artifact = self.io.artifacts/f'{len(self.records):04d}-{stage}-{label}.flags-u32'
    artifact.write_bytes(values.tobytes())
    self.records.append(dict(stage=stage,output=label,elements=tensor.numel(),blocks=len(values),all_flags_read=True,
      full_constant_pattern_checked=True,expected_word=word,comparison_uses_hash=False,full_flags_artifact=str(artifact),download_bytes=flags.nbytes()))

  def compare(self, label, left, right, stage, expected=None, changed=False):
    import numpy as np
    fmt, flags = FORMATS[label], self.flags[label]
    self.fill(flags, 'f32', 0xffffffff)
    self.dispatch(key('compare', left.numel(), fmt), flags, self.raw(left, fmt), self.raw(right, fmt))
    self.device.synchronize()
    values = np.frombuffer(self.io.read(flags), np.uint32).copy()
    assert len(values) == (left.numel()+BLOCK-1)//BLOCK
    assert not (values & np.uint32(0xfffffff8)).any(), 'Unwritten/invalid comparison flags'
    if expected is not None: np.testing.assert_array_equal(values, expected)
    elif changed:
      assert not (values & 6).any() and (values & 1).any(), f'Finite changed-output control failed: {stage}'
    else: assert not values.any(), f'Exact finite comparison failed: {stage}/{label}'
    artifact = self.io.artifacts/f'{len(self.records):04d}-{stage}-{label}.flags-u32'
    artifact.write_bytes(values.tobytes())
    record = dict(stage=stage, output=label, elements=left.numel(), blocks=len(values), all_flags_read=True,
      flag_semantics='bit0: any bit mismatch; bit1: any nonfinite left; bit2: any nonfinite right',
      comparison_uses_hash=False, bad_blocks=int(np.count_nonzero(values)), changed_expected=changed,
      independent_expected_flags=expected is not None, full_flags_artifact=str(artifact), download_bytes=flags.nbytes())
    self.records.append(record)
    return values

  def selftest(self, outputs, references, progress):
    """Actual-shape native checker controls, required before numerical comparisons."""
    import numpy as np
    for label, a, b in zip(self.micro.LABELS[1:], outputs, references, strict=True):
      with progress(f'validator-native-{label}'):
        fmt, n = FORMATS[label], a.numel()
        count, one = (n+BLOCK-1)//BLOCK, ONE[fmt]
        def check(tag, expected): self.compare(label, a, b, f'validator-{tag}', expected=expected)
        self.fill(a, fmt, one); self.fill(b, fmt, one)
        check('equal', np.zeros(count, np.uint32))
        self.check_pattern(label,a,one,'validator-complete-one-left')
        self.check_pattern(label,b,one,'validator-complete-one-right')
        for index in positions(n):
          self.poke(b, fmt, index, one ^ 1)
          expected = np.zeros(count, np.uint32); expected[index//BLOCK] = 1
          check(f'mutation-{index}', expected)
          self.check_pattern(label,b,one,f'validator-constant-mutation-{index}',expected)
          self.poke(b, fmt, index, one)
        # Both signs of zero remain finite, but differing sign bits must compare unequal.
        sign = 1 << (uint_dtype(fmt).itemsize*8-1)
        self.poke(a, fmt, n-1, 0); self.poke(b, fmt, n-1, sign)
        expected = np.zeros(count, np.uint32); expected[(n-1)//BLOCK] = 1
        check('signed-zero', expected)
        for word in (MASK[fmt], POISON[fmt]):
          self.poke(a, fmt, n-1, word); self.poke(b, fmt, n-1, word)
          expected[-1] = 6; check(f'equal-nonfinite-{word}', expected)
          self.poke(b, fmt, n-1, one); expected[-1] = 3; check(f'left-nonfinite-{word}', expected)
          self.poke(a, fmt, n-1, one); self.poke(b, fmt, n-1, word)
          expected[-1] = 5; check(f'right-nonfinite-{word}', expected)
        self.fill(a, fmt, POISON[fmt]); self.fill(b, fmt, POISON[fmt])
        check('equal-whole-poison', np.full(count, 6, np.uint32))
        self.check_pattern(label,a,POISON[fmt],'validator-complete-poison-left')
        self.check_pattern(label,b,POISON[fmt],'validator-complete-poison-right')
        self.fill(a, fmt, one)
        self.copy(b, a, fmt)
        check('copy-restoration', np.zeros(count, np.uint32))
