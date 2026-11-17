"""CPU proof and parent-only GPU correctness/timing for dynamic A4U/W4S fc2."""
import argparse
import ctypes
from contextlib import contextmanager
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


class ModeKernel:
  def __init__(self,module,wgp):self.module,self.wgp=module,wgp
  def __getattr__(self,name):
    value=getattr(self.module,name)
    return partial(value,wgp=self.wgp) if name in ('gemm','integer_gemm','quantize_pack','linear') else value


def bf16_bits(x):
  bits = np.asarray(x, dtype=np.float32).view(np.uint32)
  return ((bits+0x7fff+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(bits): return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


_libm = ctypes.CDLL('libm.so.6')
_fmaf = _libm.fmaf
_fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_fmaf.restype = ctypes.c_float


def epilogue_cpu(acc, xs, ws, bias_bits=None):
  scale = np.multiply(xs[:, None], ws[None, :], dtype=np.float32)
  acc = np.asarray(acc, dtype=np.float32)
  if bias_bits is None: return bf16_bits(np.multiply(acc, scale, dtype=np.float32))
  bias = bf16_values(bias_bits)
  values = np.empty(acc.shape, np.float32)
  for r in range(len(acc)):
    for c in range(acc.shape[1]): values[r, c] = _fmaf(float(acc[r, c]), float(scale[r, c]), float(bias[c]))
  return bf16_bits(values)


def quantize_cpu(bits):
  x = bf16_values(bits)
  if x.ndim != 2 or not np.all(np.isfinite(x)) or np.any(x < 0): raise ValueError('Expected finite nonnegative BF16 matrix')
  maximum = x.max(axis=1)
  scale = np.divide(maximum, np.float32(15), dtype=np.float32)
  scale[maximum == 0] = 1
  q = np.rint(np.clip(np.divide(x, scale[:, None], dtype=np.float32), 0, 15)).astype(np.uint8)
  return q, scale


def unpack_by_isa(packed, rows, signed=False):
  width = packed.shape[1]*32
  out = np.zeros((packed.shape[0]*16, width), np.int32)
  for row in range(len(out)):
    for k in range(width):
      lane = ((k >> 4) & 1)*16+row % 16
      byte = int(packed[row//16, k//32, lane, (k % 16)//2])
      value = (byte >> (4*(k % 2))) & 15
      out[row, k] = value-16 if signed and value >= 8 else value
  np.testing.assert_array_equal(out[rows:], 0)
  return out[:rows]


def integer_cases():
  rng = np.random.default_rng(415832)
  a = rng.integers(0, 16, (17, 32), dtype=np.int32)
  w = rng.integers(-8, 8, (48, 32), dtype=np.int32)
  for label, aa, ww in [('asymmetric', a, w), ('all15-neg8', np.full_like(a, 15), np.full_like(w, -8)),
                        ('all15-pos7', np.full_like(a, 15), np.full_like(w, 7)), ('allzero', np.zeros_like(a), w)]:
    yield label, aa.copy(), ww.copy()
  for pos in (0, 7, 8, 15, 16, 23, 24, 31):
    aa, ww = a.copy(), w.copy()
    aa[-1, pos] = 15 if aa[-1, pos] != 15 else 0
    ww[-1, pos] = -8 if ww[-1, pos] != -8 else 7
    yield f'mutate-k{pos}', aa, ww


def activation_bits(m, k, seed=415832):
  rng = np.random.default_rng(seed)
  x = np.square(np.maximum(rng.standard_normal((m, k)).astype(np.float32), 0), dtype=np.float32)
  x *= np.exp2((np.arange(m) % 15-7).astype(np.float32))[:, None]
  bits = bf16_bits(x)
  bits[0] = 0
  if m > 1: bits[1] = bf16_bits((np.arange(k) % 30).astype(np.float32)/2); bits[1, -1] = bf16_bits(15.)
  if m > 2: bits[2] = bf16_bits(15.)
  if m > 3: bits[3] = bf16_bits(1.); bits[3, -1] = bf16_bits(65536.)
  if m > 4: bits[4] = 0; bits[4, -1] = np.uint16(0x7f7f)  # Largest finite BF16.
  if m > 5: bits[5] = np.uint16(1)  # BF16 subnormal, requires IEEE FP32 division.
  return bits


def scales(m, n):
  rng = np.random.default_rng(8405+m+n)
  xs = rng.uniform(.01, 2., m).astype(np.float32)*np.exp2((np.arange(m) % 9-4).astype(np.float32))
  ws = rng.uniform(.01, 2., n).astype(np.float32)*np.exp2((np.arange(n) % 11-5).astype(np.float32))
  xs[0] = 0; ws[0] = 0
  bias = bf16_bits(rng.standard_normal(n).astype(np.float32))
  return xs, ws, bias


def fma_fixture():
  """Find a BF16-sensitive cancellation, with an integer dot realizable at K32."""
  rng = np.random.default_rng(8541)
  for _ in range(10000):
    acc = int(rng.integers(1, 3000))
    xs, ws = (np.float32(rng.uniform(.01, 3.)) for _ in range(2))
    scale = np.float32(xs*ws)
    bias = bf16_bits(-np.float32(acc)*scale)
    fused = epilogue_cpu(np.array([[acc]]), np.array([xs]), np.array([ws]), np.array([bias]))
    separate = bf16_bits(np.float32(np.float32(acc)*scale)+bf16_values(bias))
    if fused[0, 0] != separate:
      a, w = np.zeros((17, 32), np.int32), np.zeros((48, 32), np.int32)
      units, remainder = divmod(acc, 105)
      a[0, :units] = 15; w[0, :units] = 7
      q, r = divmod(remainder, 7)
      a[0, units] = q; w[0, units] = 7
      a[0, units+1] = r; w[0, units+1] = 1
      xscale, wscale, biases = np.ones(17, np.float32), np.ones(48, np.float32), np.zeros(48, np.uint16)
      xscale[0], wscale[0], biases[0] = xs, ws, bias
      assert (a.astype(np.int64) @ w.astype(np.int64).T)[0, 0] == acc
      return a, w, xscale, wscale, biases, dict(acc=acc, xs=float(xs), ws=float(ws), bias_bits=int(bias),
                                              fused_bits=int(fused[0, 0]), separate_bits=int(separate))
  raise AssertionError('No sensitive FMA fixture')


def metadata_proof(kernel):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import DepsTracker
  shapes = [((17,48),dtypes.bfloat16),((2,1,32,8),dtypes.uint8),((3,1,32,8),dtypes.uint8),
            ((17,),dtypes.float32),((48,),dtypes.float32),((48,),dtypes.bfloat16)]
  params = [UOp.placeholder(shape, dt, i) for i, (shape, dt) in enumerate(shapes)]
  gp = kernel.gemm_program(*params, rows=17, columns=48, width=32)
  qp_args = [UOp.placeholder(shape,dt,i) for i,(shape,dt) in enumerate([shapes[1],shapes[3],((17,32),dtypes.bfloat16)])]
  qp = kernel.quantize_program(*qp_args, rows=17, width=32)
  assert gp.arg.outs == (0,) and gp.arg.ins == (1,2,3,4,5)
  assert qp.arg.outs == (0,1) and qp.arg.ins == (2,)
  buffers = [UOp.alloc(shape, dt, device='AMD') for shape,dt in shapes]
  x = UOp.alloc((17,32),dtypes.bfloat16,device='AMD')
  calls = []
  for _ in range(40):
    calls += [qp.call(buffers[1], buffers[3], x), gp.call(*buffers)]
  tracker = DepsTracker(['compute']*len(calls)); dependencies = []
  for tag, call in enumerate(calls):
    outs, _ = get_call_outs_ins(call)
    accesses = [(buf, i in outs) for i,buf in enumerate(call.src[1:])]
    deps = tracker.access_resources(accesses, tag)
    if tag: assert tag-1 in deps, (tag,deps)
    dependencies.append(deps)
  # A different input allocation cannot create the GEMM dependency: only the
  # same actual output buffer overlaps between these two calls.
  tracker = DepsTracker(['compute']*2)
  gs = []
  for tag in range(2):
    bufs = [buffers[0], *[UOp.alloc(shape,dt,device='AMD') for shape,dt in shapes[1:]]]
    call = gp.call(*bufs); outs,_ = get_call_outs_ins(call)
    gs.append(tracker.access_resources([(b,i in outs) for i,b in enumerate(call.src[1:])],tag))
  assert gs == [[],[0]], gs
  return dict(gemm_outs=list(gp.arg.outs),quantize_outs=list(qp.arg.outs),serial_pipeline_calls=80,
              every_adjacent_edge_proven=True,isolated_gemm_WAW_dependencies=gs)


def cpu_checks(kernel):
  records = []
  for name,a,w in integer_cases():
    pa,pw=kernel.pack_unsigned_cpu(a),kernel.pack_weights_cpu(w)
    np.testing.assert_array_equal(unpack_by_isa(pa,len(a)),a)
    np.testing.assert_array_equal(unpack_by_isa(pw,len(w),signed=True),w)
    assert not pw.flags.writeable and not np.shares_memory(pw,w)
    raw=a.astype(np.int64)@w.astype(np.int64).T
    if name=='all15-neg8': np.testing.assert_array_equal(raw,-3840)
    if name=='all15-pos7': np.testing.assert_array_equal(raw,3360)
    xs,ws,bias=scales(len(a),len(w))
    for bb in (None,bias): epilogue_cpu(raw,xs,ws,bb)
    records.append(dict(name=name,shape=[len(a),len(w),a.shape[1]]))
  bits=activation_bits(17,32);q,xs=quantize_cpu(bits)
  np.testing.assert_array_equal(unpack_by_isa(kernel.pack_unsigned_cpu(q),17),q)
  np.testing.assert_array_equal(q[0],0);assert xs[0]==1
  np.testing.assert_array_equal(q[1,:29],np.rint(np.arange(29,dtype=np.float32)/2))
  assert q[5].min()==15 and q[5].max()==15 and xs[5]>0
  source=np.ones((48,32),np.int32);packed=kernel.pack_weights_cpu(source);digest=hashlib.sha256(packed).hexdigest();source[:]=-8
  assert hashlib.sha256(packed).hexdigest()==digest
  try:packed.flags.writeable=True
  except ValueError:pass
  else:raise AssertionError('Weight bytes are not immutable')
  invalid=[np.full((17,32),8,np.int32),np.full((17,32),-9,np.int32)]
  for value in invalid:
    try:kernel.pack_weights_cpu(value)
    except ValueError:pass
    else:raise AssertionError('Invalid signed weight accepted')
  fixture=fma_fixture()[-1]
  return dict(cpu_only=True,integer_pack_cases=records,activation_controls='zero/ties/15/outlier/maxfinite/subnormal',
              frozen_weight_copy=True,fma_fixture=fixture,metadata=metadata_proof(kernel),
              scheduled_tensor_graph={mode:scheduled_proof(ModeKernel(kernel,mode=='wgp')) for mode in ('cu','wgp')},
              k9216_int32_bounds=[-120*9216,105*9216])


def reference(out,a,b,xs,ws,*bias,raw=False,renderer=None):
  """Independent FP8 WMMA reduction with actual row/column scales."""
  from tinygrad import Device
  from tinygrad.codegen.opt import Opt,OptOps
  from tinygrad.codegen.opt.postrange import Scheduler
  from tinygrad.helpers import round_up
  from tinygrad.uop.ops import UOp,Ops,KernelInfo,AxisType
  from tinygrad import dtypes
  m,n=out.shape;k=a.shape[1]
  mr,nr,kr=UOp.range(round_up(m,64),0),UOp.range(round_up(n,64),1),UOp.range(k,2)
  def index(r):return (r//16*(k//16)+kr//16)*256+(kr%16//4)*64+(r%16)*4+kr%4
  av=a.flatten().index(index(mr).valid(mr<m));bv=b.flatten().index(index(nr).valid(nr<n))
  value=(av*bv).cast(dtypes.float32).reduce(kr,arg=Ops.ADD)
  if not raw:
    value=value*(xs.flatten().index(mr.valid(mr<m))*ws.flatten().index(nr.valid(nr<n)))
    if bias:value=value+bias[0].flatten().index(nr.valid(nr<n)).cast(dtypes.float32)
  store=out.flatten().index((mr*n+nr).valid((mr<m)&(nr<n))).store(value.cast(out.dtype))
  sched=Scheduler(store.end(mr,nr).sink(arg=KernelInfo()),Device[Device.DEFAULT].renderer if renderer is None else renderer)
  sched.convert_loop_to_global();ranges=sched.apply_opt(Opt(OptOps.TC,0,(-1,0,1)))
  for axis in (1,0):sched.apply_opt(Opt(OptOps.SPLIT,sched.rngs.index(ranges[axis]),(4,AxisType.UPCAST)))
  return sched.get_optimized_ast(name_override='agent_int4_scaled_fp8_reference')


def gpu_reference(a,b,xs,ws,bias=None,raw=False,out=None):
  from tinygrad import Tensor,dtypes
  if out is None:out=Tensor.empty(xs.shape[0],ws.shape[0],dtype=dtypes.float32 if raw else dtypes.bfloat16,device=a.device)
  return Tensor.custom_kernel(out,a,b,xs,ws,*(() if bias is None or raw else (bias,)),fxn=partial(reference,raw=raw))[0]


def tensor_bits(bits):
  from tinygrad import Tensor,dtypes
  return Tensor(np.asarray(bits,dtype=np.uint16)).bitcast(dtypes.bfloat16).realize()


def integer_check_gpu(kernel,name,a,w,xs,ws,bias_bits):
  from tinygrad import Tensor,dtypes
  from amd_gemm import pack
  m,n,k=len(a),len(w),a.shape[1]
  pa,pw=Tensor(kernel.pack_unsigned_cpu(a)).realize(),Tensor(kernel.pack_weights_cpu(w).copy()).realize()
  tx,tw=Tensor(xs).realize(),Tensor(ws).realize()
  af,wf=(pack(Tensor(v.astype(np.float32)).cast(dtypes.fp8e4m3)).realize() for v in (a,w))
  integer=kernel.integer_gemm(pa,pw,rows=m,columns=n).numpy()
  ref=gpu_reference(af,wf,tx,tw,raw=True).numpy()
  np.testing.assert_array_equal(integer,ref)
  ri=np.arange(m) if m<128 else np.array([0,1,2,3,4,5,7,8,15,16,31,32,m//2,m-2,m-1])
  ci=np.arange(n) if m<128 else np.array([0,3,7,15,16,31,32,n//2,n-2,n-1])
  oracle=a[ri].astype(np.int64)@w[ci].astype(np.int64).T
  np.testing.assert_array_equal(integer[np.ix_(ri,ci)],oracle)
  for bb in (None,bias_bits):
    bias=None if bb is None else tensor_bits(bb)
    out=kernel.gemm(pa,pw,tx,tw,rows=m,columns=n,bias=bias).bitcast(dtypes.uint16).numpy()
    expected=gpu_reference(af,wf,tx,tw,bias).bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(out,expected)
    cpu=epilogue_cpu(oracle,xs[ri],ws[ci],None if bb is None else bb[ci])
    np.testing.assert_array_equal(out[np.ix_(ri,ci)],cpu)
  return dict(name=name,shape=[m,n,k],raw_all_exact=True,scaled_bias_and_no_bias_all_exact=True,cpu_elements=int(oracle.size))


def quantize_check_gpu(kernel,bits):
  from tinygrad import dtypes
  x=tensor_bits(bits);packed,scales=kernel.quantize_pack(x)
  q,expected_scale=quantize_cpu(bits)
  np.testing.assert_array_equal(packed.numpy(),kernel.pack_unsigned_cpu(q))
  np.testing.assert_array_equal(scales.bitcast(dtypes.uint32).numpy(),expected_scale.view(np.uint32))
  return q,expected_scale


def replay_check_gpu(kernel):
  """Replay the same captured two-kernel graph with changed pixels and weights."""
  from tinygrad import Tensor,TinyJit,dtypes
  from amd_gemm import pack
  bits=activation_bits(17,32);rng=np.random.default_rng(415832)
  w=rng.integers(-8,8,(48,32),dtype=np.int32);_,ws,bias_bits=scales(17,48)
  ws=np.multiply(ws,np.float32(1e-6),dtype=np.float32)
  jit=TinyJit(lambda xx,ww,ss,bb:kernel.linear(xx,ww,ss,bias=bb).realize())
  results=[]
  for change in ('original','activation','weight-scale-bias'):
    if change=='activation':bits=bits.copy();bits[-1,-1]=bf16_bits(1024.)
    if change=='weight-scale-bias':
      w=w.copy();w[-1,-1]=-8 if w[-1,-1]!=-8 else 7
      ws=ws.copy();ws[-1]=np.float32(ws[-1]*1.25)
      bias_bits=bias_bits.copy();bias_bits[-1]=bf16_bits(1.75)
    x=tensor_bits(bits);pw=Tensor(kernel.pack_weights_cpu(w).copy()).realize();sw=Tensor(ws).realize();bb=tensor_bits(bias_bits)
    if change=='original':
      for _ in range(3):jit(x,pw,sw,bb)
    result=jit(x,pw,sw,bb).bitcast(dtypes.uint16).numpy()
    q,xs=quantize_cpu(bits)
    af,wf=(pack(Tensor(v.astype(np.float32)).cast(dtypes.fp8e4m3)).realize() for v in (q,w))
    expected=gpu_reference(af,wf,Tensor(xs).realize(),sw,bb).bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(result,expected)
    cpu=epilogue_cpu(q.astype(np.int64)@w.astype(np.int64).T,xs,ws,bias_bits)
    np.testing.assert_array_equal(result,cpu)
    results.append(dict(change=change,all_bits_exact=True))
  return results


def serial_queue_proof(batch,queues,expected_name,calls,full=False):
  """Verify actual PARAM spans, dependencies, and pinned HCQ execution barriers."""
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import DepsTracker,unwrap_lane
  selected=[c.without_after for c,_,_ in batch]
  assert len(selected)==calls and len(queues)==1,(len(selected),calls,len(queues))
  queue_of=[(d,q) for _,d,q in batch]
  assert len(set(queue_of))==1,'Expected one serial compute queue'
  tracker=DepsTracker(queue_of);deps=[];outputs=[];descriptions=[]
  def span(value):
    base,lane,offset=unwrap_lane(value)
    assert base.op is Ops.PARAM and lane is None,'Output and input storage must be explicit unsharded JIT PARAMs'
    return base,offset,value.max_numel()*value.dtype.itemsize
  def storage_key(spans):return tuple((base.key,offset,size) for base,offset,size in spans)
  for tag,c in enumerate(selected):
    assert c.op is Ops.CALL and c.body.op is Ops.PROGRAM
    outs,ins=get_call_outs_ins(c)
    name=c.body.src[0].arg.name
    assert name in expected_name,(tag,name,expected_name)
    if name in ('agent_int4_quantize','agent_int4_grouped_quantize'):
      assert outs==(0,1) and ins==(2,),(name,outs,ins)
    elif name in ('agent_int4_scaled','agent_int4_scaled_fp8_reference','agent_int4_grouped'):
      assert outs==(0,) and ins in ((1,2,3,4),(1,2,3,4,5)),(name,outs,ins)
    else:assert len(outs)==1,'Expected one declared output for a matrix control'
    current=tuple(span(c.src[1+i]) for i in outs)
    assert len(set(storage_key(current)))==len(current),'Written outputs must have distinct storage spans'
    accesses=[(b,i in outs) for i,b in enumerate(c.src[1:]) if not b.is_variable]
    waits=tracker.access_resources(accesses,tag)
    if tag:assert tag-1 in waits,(tag,name,waits)
    if full and tag>=2:assert tag-2 in waits,('Missing same-phase WAW dependency',tag,name,waits)
    deps.append(waits)
    outputs.append(current)
    args=[span(c.src[1+g]) for g in c.body.arg.globals]
    descriptions.append(dict(name=name,device=batch[tag][1],global_input_slots=[s[0].arg.slot for s in args],
      output_argument_indices=list(outs),input_argument_indices=list(ins),program_key=c.body.key.hex(),
      output_spans=[dict(input_slot=s[0].arg.slot,byte_offset=s[1],bytes=s[2]) for s in current]))
  if full:
    assert tuple(expected_name) in (('agent_int4_quantize','agent_int4_scaled'),('agent_int4_grouped_quantize','agent_int4_grouped'))
    assert all(selected[i].body.src[0].arg.name==expected_name[i%2] for i in range(calls))
    assert calls%2==0 and len({storage_key(s) for s in outputs[::2]})==len({storage_key(s) for s in outputs[1::2]})==1,'Each phase must reuse shared output storage'
    for tag in range(1,calls,2):
      assert storage_key((span(selected[tag].src[2]),))==storage_key((outputs[tag-1][0],)),'GEMM packed input lacks the quantizer RAW dependency'
      assert storage_key((span(selected[tag].src[4]),))==storage_key((outputs[tag-1][1],)),'GEMM row scale lacks the quantizer RAW dependency'
    written=(*outputs[0],*outputs[1])
    assert all(a[0] is not b[0] or a[1]+a[2]<=b[1] or b[1]+b[2]<=a[1]
               for i,a in enumerate(written) for b in written[i+1:]),'Pipeline outputs overlap'
  else:assert len({storage_key(s) for s in outputs})==1,'All calls must reuse shared output storage'
  ordered=[]
  for command in next(iter(queues.values())):
    command=command.without_after
    if command.op is Ops.CALL and command.body.op is Ops.PROGRAM:ordered.append(command)
    elif command.op is Ops.CALL and command.body.op is Ops.CUSTOM_FUNCTION and command.body.arg.name=='exec_barrier':ordered.append('barrier')
  expected=[item for i,c in enumerate(selected) for item in ((c,) if i==0 else ('barrier',c))]
  assert ordered==expected,'Every adjacent kernel must have an ordered HCQ execution barrier'
  return dict(captured_kernel_calls=calls,every_adjacent_edge=True,output_storage_fixed=True,
              metadata_outs=[list(get_call_outs_ins(c)[0]) for c in selected[:2]],dependencies=deps,
              ordered_execution_barriers=calls-1,kernel_arguments=descriptions,full_pipeline=full,
              phase_waw_edges_verified=True,actual_queue_builder_observed=True)


@contextmanager
def observe_serial_queue(jit,expected_name,calls,full=False):
  """Observe capture-time HCQ lowering only; restore the runtime on every exit."""
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  original=hcq2._build_queues
  proofs=jit._int4_serial_proofs=[]
  def build(ctx):
    queues=original(ctx)
    # The same capture invocation can execute eager BUFFER calls before its
    # PARAM graph is lowered. Only the latter represents reusable JIT storage.
    named=[c for c,_,_ in ctx.batch if c.body.op is Ops.PROGRAM and c.body.src[0].arg.name in expected_name]
    if named and all(hcq2.unwrap_lane(c.src[1+i])[0].op is Ops.PARAM for c in named for i in c.body.arg.globals):
      proofs.append(serial_queue_proof(ctx.batch,queues,expected_name,calls,full))
    return queues
  hcq2._build_queues=build
  try:yield
  finally:hcq2._build_queues=original


def inspect_serial(jit,expected_name,calls,full=False):
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support import hcq2
  entries=[c.without_after for c in jit.captured._linear.src]
  graphs=[c for c in entries if isinstance(getattr(c.arg,'aux',None),hcq2.HCQInfo)]
  if graphs:
    assert len(graphs)==len(entries),'Unexpected mixed raw and HCQ captured calls'
    proofs=getattr(jit,'_int4_serial_proofs',())
    assert len(proofs)==1,'Expected one observed capture-time serial HCQ queue'
    proof=proofs[0]
    assert proof['captured_kernel_calls']==calls and proof['full_pipeline']==full
    kernels=[k for c in graphs for k in c.arg.aux.kernels]
    assert len(kernels)==calls,(len(kernels),calls)
    for actual,expected in zip(kernels,proof['kernel_arguments'],strict=True):
      assert actual[0]==expected['device'],'HCQ device differs from observed queue'
      assert actual[1]==expected['name'] and actual[1] in expected_name,'HCQ kernel name differs from observed queue'
      assert list(actual[5])==expected['global_input_slots'],'HCQ buffer slots differ from actual kernel arguments'
      assert list(actual[6][0])==expected['output_argument_indices'],'HCQ writes differ from observed ProgramInfo'
      assert list(actual[6][1])==expected['input_argument_indices'],'HCQ reads differ from observed ProgramInfo'
      assert actual[4]==bytes.fromhex(expected['program_key']),'HCQ program differs from the observed kernel'
    return dict(proof,captured_representation='HCQInfo',hcq_metadata_verified=True)
  assert all(c.op is Ops.CALL and c.body.op is Ops.PROGRAM for c in entries),'Expected PROGRAM calls or HCQ submit calls'
  # CPU scheduled/captured controls still use raw PROGRAM calls. Exercise the
  # same real HCQ dependency/barrier builder with their actual PARAM arguments.
  device='CPU';batch=[(c,device,'COMPUTE:0') for c in entries]
  ctx=SimpleNamespace(batch=batch,calls={(device,'COMPUTE:0'):list(range(len(batch)))},
                      peers={(device,'COMPUTE:0'):set()},queues={device:['COMPUTE:0']},stamps=lambda *_:())
  proof=serial_queue_proof(batch,hcq2._build_queues(ctx),expected_name,calls,full)
  return dict(proof,captured_representation='PROGRAM',hcq_metadata_verified=False)


def scheduled_proof(kernel):
  """Schedule the actual Tensor API on CPU buffers, without executing binaries."""
  from tinygrad import Tensor,dtypes
  x=Tensor.empty(17,32,dtype=dtypes.bfloat16,device='CPU')
  w=Tensor.empty(3,1,32,8,dtype=dtypes.uint8,device='CPU')
  ws=Tensor.empty(48,dtype=dtypes.float32,device='CPU');bias=Tensor.empty(48,dtype=dtypes.bfloat16,device='CPU')
  q=Tensor.empty(2,1,32,8,dtype=dtypes.uint8,device='CPU');xs=Tensor.empty(17,dtype=dtypes.float32,device='CPU')
  out=Tensor.empty(17,48,dtype=dtypes.bfloat16,device='CPU')
  for _ in range(40):
    q,xs=kernel.quantize_pack(Tensor(x.uop.after(out.uop)),packed_out=q,scales_out=xs)
    out=kernel.gemm(q,w,xs,ws,rows=17,columns=48,bias=bias,out=out)
  linear,_=out.linear_with_vars()
  from tinygrad.uop.ops import Ops,UOp
  buffers=[u for u in linear.toposort() if u.op is Ops.BUFFER]
  linear=linear.substitute({u:UOp.param(i,u.dtype,u.max_numel(),u.device) for i,u in enumerate(buffers)},walk=True)
  return inspect_serial(SimpleNamespace(captured=SimpleNamespace(_linear=linear)),
                        ('agent_int4_quantize','agent_int4_scaled'),80,full=True)


def benchmark_gpu(kernel,bits,w,ws,bias_bits,args):
  from tinygrad import Tensor,Device,TinyJit,dtypes
  from amd_gemm import pack
  m,k=bits.shape;n=len(w)
  x=tensor_bits(bits);qw=Tensor(kernel.pack_weights_cpu(w).copy()).realize();wscale=Tensor(ws).realize();bias=tensor_bits(bias_bits)
  qcpu,xcpu=quantize_cpu(bits)
  qa=Tensor(kernel.pack_unsigned_cpu(qcpu)).realize();xscale=Tensor(xcpu).realize()
  af,wf=(pack(Tensor(v.astype(np.float32)).cast(dtypes.fp8e4m3)).realize() for v in (qcpu,w))
  out=Tensor.zeros(m,n,dtype=dtypes.bfloat16).realize()
  def gemms(aa,ww,xs,wws,bb,oo):
    for _ in range(args.runs):oo=kernel.gemm(aa,ww,xs,wws,rows=m,columns=n,bias=bb,out=oo)
    return oo.realize()
  def quantizers(xx,qq,ss):
    for _ in range(args.runs):qq,ss=kernel.quantize_pack(xx,packed_out=qq,scales_out=ss)
    Tensor.realize(qq,ss);return qq,ss
  def full(xx,ww,wws,bb,qq,ss,oo):
    for _ in range(args.runs):
      # Structural dependency prevents schedule reordering before HCQ sees aliases.
      dependent=Tensor(xx.uop.after(oo.uop))
      qq,ss=kernel.quantize_pack(dependent,packed_out=qq,scales_out=ss)
      oo=kernel.gemm(qq,ww,ss,wws,rows=m,columns=n,bias=bb,out=oo)
    return oo.realize()
  def fp8s(aa,ww,xs,wws,bb,oo):
    for _ in range(args.runs):oo=gpu_reference(aa,ww,xs,wws,bb,out=oo)
    return oo.realize()
  variants=[('a4_quantize_pack',quantizers,(x,qa,xscale),('agent_int4_quantize',),args.runs,False),
            ('a4w4_scaled_gemm',gemms,(qa,qw,xscale,wscale,bias,out),('agent_int4_scaled',),args.runs,False),
            ('a4w4_full_serial',full,(x,qw,wscale,bias,qa,xscale,out),('agent_int4_quantize','agent_int4_scaled'),args.runs*2,True),
            ('fp8_scaled_gemm_control',fp8s,(af,wf,xscale,wscale,bias,out),('agent_int4_scaled_fp8_reference',),args.runs,False)]
  records={}
  expected=gpu_reference(af,wf,xscale,wscale,bias).bitcast(dtypes.uint16).numpy()
  for name,fn,inputs,names,count,full_flag in variants:
    jit=TinyJit(fn)
    for _ in range(3):
      if jit.cnt==1:
        with observe_serial_queue(jit,names,count,full_flag):jit(*inputs)
      else:jit(*inputs)
    proof=inspect_serial(jit,names,count,full=full_flag)
    if name=='a4_quantize_pack':
      np.testing.assert_array_equal(jit.captured.ret[0].numpy(),kernel.pack_unsigned_cpu(qcpu))
      np.testing.assert_array_equal(jit.captured.ret[1].bitcast(dtypes.uint32).numpy(),xcpu.view(np.uint32))
    else:np.testing.assert_array_equal(jit.captured.ret.bitcast(dtypes.uint16).numpy(),expected)
    values=[]
    for _ in range(args.rounds):
      Device[Device.DEFAULT].synchronize();start=time.perf_counter();jit(*inputs);Device[Device.DEFAULT].synchronize()
      values.append((time.perf_counter()-start)*1000/args.runs)
    records[name]=dict(amortized_ms=values,median_ms=float(np.median(values)),serial_proof=proof)
    print(json.dumps(dict(timing=name,median_ms=records[name]['median_ms'])),flush=True)
  return records


def gpu_checks(kernel,args):
  from tinygrad import Device
  local('prefill-kernels').fix_dispatch_packet();local('wgp_compiler').install(Device[Device.DEFAULT])
  records=[]
  for name,a,w in integer_cases():
    xs,ws,bias=scales(len(a),len(w));record=integer_check_gpu(kernel,name,a,w,xs,ws,bias)
    records.append(record);print(json.dumps(record),flush=True)
  a,w,xs,ws,bias,fixture=fma_fixture()
  records.append(integer_check_gpu(kernel,'fma-cancellation',a,w,xs,ws,bias))
  bits=activation_bits(17,32);quantize_check_gpu(kernel,bits)
  changed=bits.copy();changed[-1,-1]=bf16_bits(128.);quantize_check_gpu(kernel,changed)
  report=dict(cpu_only=False,complete=False,small_controls=records,fma_fixture=fixture,activation_tail_exact=True,
              captured_mutations=replay_check_gpu(kernel))
  if args.suite=='actual':
    m,n,k=2112,2048,9216;rng=np.random.default_rng(415832)
    w=rng.integers(-8,8,(n,k),dtype=np.int32);bits=activation_bits(m,k)
    for mutation in ('original','activation-outlier','weight-tail'):
      if mutation=='activation-outlier':bits=bits.copy();bits[-1,-1]=bf16_bits(131072.)
      if mutation=='weight-tail':w=w.copy();w[-1,-1]=7 if w[-1,-1]!=7 else -8
      a,xs=quantize_check_gpu(kernel,bits);_,ws,bias=scales(m,n)
      # Keep even the maxfinite-BF16 outlier's scaled outputs finite. Quantizer
      # maxfinite/subnormal controls remain exact, without incidental NaN payloads.
      ws=np.multiply(ws,np.float32(1e-6),dtype=np.float32)
      record=integer_check_gpu(kernel,mutation,a,w,xs,ws,bias)
      records.append(record);print(json.dumps(record),flush=True)
    report['timing']=benchmark_gpu(kernel,bits,w,ws,bias,args)
    report['timing_includes']='dynamic row reduction, BF16->FP32 conversion, exact division/RNE, nibble packing, scaled GEMM'
  report['complete']=True;report['all_pass']=True
  return report


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=('cpu','small','actual'),default='cpu')
  p.add_argument('--mode',choices=('cu','wgp','both'),default='both')
  p.add_argument('--runs',type=int,default=40);p.add_argument('--rounds',type=int,default=3)
  p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  if not 40<=args.runs<=80 or not 1<=args.rounds<=5:p.error('Require 40..80 serial calls, 1..5 rounds')
  kernel=local('agent-int4-scaled-kernel')
  if args.suite=='cpu':report=cpu_checks(kernel)
  else:
    modes=('cu','wgp') if args.mode=='both' else (args.mode,)
    reports={mode:gpu_checks(ModeKernel(kernel,mode=='wgp'),args) for mode in modes}
    report=dict(cpu_only=False,complete=True,all_pass=True,modes=reports)
  sources=[Path(__file__),Path(__file__).with_name('agent-int4-scaled-kernel.py'),Path(__file__).with_name('agent-int4k32-kernel.py'),
           Path(__file__).parent.parent/'amd_gemm.py',Path(__file__).with_name('wgp_compiler.py')]
  report['source_hashes']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(dict(cpu_only=report['cpu_only'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
