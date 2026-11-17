"""Strict alias/RAW/WAR/WAW audit of the actual residual/norm HCQ batch. No kernel changes."""
from contextlib import contextmanager


def overlaps(left,right):
  return left[0]==right[0] and left[1]<right[1]+right[2] and right[1]<left[1]+left[2]


def resolver(input_uops):
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  def resolve(arg):
    base,lane,offset=unwrap_lane(arg)
    assert lane is None,'Multi-device lanes are outside this single-device proof'
    size=arg.max_numel()*arg.dtype.itemsize
    if base.op is Ops.PARAM:
      assert 0<=base.arg.slot<len(input_uops),'Unbound input parameter'
      root,begin,length=resolve(input_uops[base.arg.slot])
      assert offset>=0 and offset+size<=length,'Parameter view exceeds its input'
      return root,begin+offset,size
    if base.op is Ops.BUFFER:
      buffer=base.buffer
      root,begin,length=('BUFFER',buffer.base),buffer.offset,buffer.base.nbytes
    elif base.op is Ops.ALLOC:
      root,begin,length=('ALLOC',base),0,base.max_numel()*base.dtype.itemsize
    else:raise AssertionError('Unsupported HCQ argument root: '+base.op.name)
    assert offset>=0 and begin+offset+size<=length,'Buffer view exceeds allocation'
    return root,begin+offset,size
  return resolve


def audit(ctx,queues,label,inputs):
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.engine.realize import get_call_outs_ins
  assert label in ('baseline','candidate')
  count=16 if label=='candidate' else 48
  assert len(ctx.batch)==count,'Wrong complete-operation kernel count'
  calls=[call for call,_,_ in ctx.batch]
  assert all(call.op is Ops.CALL and call.body.op is Ops.PROGRAM for call in calls)
  assert len(queues)==1 and len({(device,queue) for _,device,queue in ctx.batch})==1
  queue_key=next(iter(queues));assert queue_key==ctx.batch[0][1:]
  assert queue_key[1]=='COMPUTE:0'
  input_uops=_prepare_jit_inputs(tuple(inputs),{})[0]
  assert len(input_uops)==4
  resolve=resolver(input_uops);input_spans=[resolve(value) for value in input_uops]
  assert [span[2] for span in input_spans]==[8388608,8388608,4096,4]
  assert not any(overlaps(a,b) for i,a in enumerate(input_spans) for b in input_spans[i+1:])
  roots=[]
  def serial(span):
    if span[0] not in roots:roots.append(span[0])
    return dict(root=roots.index(span[0]),root_kind=span[0][0],offset=span[1],nbytes=span[2])
  spans=[[resolve(arg) for arg in call.src[1:]] for call in calls]
  assert all(not arg.is_variable for call in calls for arg in call.src[1:])
  residuals=[];packed=[];records=[];accesses=[]
  for tag,(call,args) in enumerate(zip(calls,spans)):
    outs,ins=get_call_outs_ins(call)
    if label=='candidate':
      assert call.body.src[0].arg.name=='agent_residual_norm'
      assert (outs,ins)==((0,1),(2,3,4,5)) and len(args)==6,'Both fused outputs must be writes'
      r,p,a,b,w,s=args
      assert a==(input_spans[0] if tag==0 else residuals[-1]),'Missing exact running-residual dependency'
      assert [b,w,s]==input_spans[1:]
      assert r[2]==8388608 and p[2]==4194304
      assert not overlaps(r,p),'The two fused outputs overlap'
      residuals.append(r);packed.append(p)
    elif tag%3==0:
      assert (outs,ins)==((0,),(1,2)) and len(args)==3
      r,a,b=args
      assert a==(input_spans[0] if tag==0 else residuals[-1]) and b==input_spans[1]
      assert r[2]==8388608
      residuals.append(r)
    elif tag%3==1:
      assert call.body.src[0].arg.name=='alpamayo_norm'
      assert (outs,ins)==((0,),(1,2,3)) and len(args)==4
      n,r,w,unused_beta=args
      assert r==residuals[-1] and w==unused_beta==input_spans[2] and n[2]==8388608
    else:
      assert (outs,ins)==((0,),(1,2)) and len(args)==3
      p,n,s=args
      assert n==spans[tag-1][0] and s==input_spans[3] and p[2]==4194304
      packed.append(p)
    writes=[args[i] for i in outs];reads=[args[i] for i in ins]
    assert not any(overlaps(a,b) for i,a in enumerate(writes) for b in writes[i+1:])
    assert not any(overlaps(out,inp) for out in writes for inp in reads),'Kernel output aliases a current input'
    assert not any(overlaps(out,inp) for out in writes for inp in input_spans),'Kernel overwrites an original input'
    accesses.append((writes,reads))
    records.append(dict(index=tag,name=call.body.src[0].arg.name,outs=list(outs),ins=list(ins),args=[serial(a) for a in args]))
  assert len(residuals)==len(packed)==16
  assert not any(overlaps(a,b) for i,a in enumerate(packed) for b in packed[i+1:]),'Packed outputs alias'
  assert not any(overlaps(a,b) for a in packed for b in residuals),'Packed output aliases a residual allocation'
  # Independent pairwise byte-interval oracle. Does not use hcq2.DepsTracker.
  edges=[]
  for target,(tw,tr) in enumerate(accesses):
    for source,(sw,sr) in enumerate(accesses[:target]):
      kinds=[]
      if any(overlaps(a,b) for a in sw for b in tr):kinds.append('RAW')
      if any(overlaps(a,b) for a in sr for b in tw):kinds.append('WAR')
      if any(overlaps(a,b) for a in sw for b in tw):kinds.append('WAW')
      if kinds:edges.append(dict(source=source,target=target,kinds=kinds))
  # Inspect the actual builder's program order and barriers, retaining every command.
  seen=[];barriers=[]
  for command in queues[queue_key]:
    if command.op is Ops.CALL and command.body.op is Ops.PROGRAM:
      assert len(seen)<len(calls) and command is calls[len(seen)],'HCQ reordered or substituted a program'
      seen.append(command)
    elif command.op is Ops.CALL and command.body.op is Ops.CUSTOM_FUNCTION and command.body.arg.name=='exec_barrier':
      assert 0<len(seen)<len(calls) and len(seen) not in barriers
      barriers.append(len(seen))
  assert seen==calls
  for edge in edges:
    assert any(edge['source']<barrier<=edge['target'] for barrier in barriers),'Unordered '+','.join(edge['kinds'])+' hazard'
  if label=='candidate':assert barriers==list(range(1,16)),'Every fused operation must wait for its predecessor'
  assert all(any(kind in edge['kinds'] for edge in edges) for kind in ('RAW','WAR','WAW')),'Expected actual arena-reuse hazards are absent'
  return dict(label=label,kernel_calls=count,dependent_operations=16,outputs_per_operation=2,
              actual_queue_builder=True,resolved_input_buffers=True,all_required_hazards_ordered=True,
              barrier_before_tags=barriers,hazard_counts={kind:sum(kind in e['kinds'] for e in edges) for kind in ('RAW','WAR','WAW')},
              input_spans=[serial(a) for a in input_spans],calls=records,hazards=edges)


@contextmanager
def observe(proofs,label,inputs,audit_log=None):
  from tinygrad.runtime.support import hcq2
  original=hcq2._build_queues;seen=[]
  def build(ctx):
    queues=original(ctx)
    proof=audit(ctx,queues,label,inputs)
    proofs.append(proof);seen.append(proof)
    if audit_log is not None:audit_log.append(proof)
    return queues
  hcq2._build_queues=build
  try:
    yield
    assert len(seen)==1,'Expected exactly one complete-operation HCQ batch'
  finally:hcq2._build_queues=original
