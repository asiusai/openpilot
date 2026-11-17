"""Prove 40 serial full-boundary invocations from real pinned HCQ metadata."""
from contextlib import contextmanager
from types import SimpleNamespace


def prove(batch,queues,phases,count=40):
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import DepsTracker,unwrap_lane
  calls=[c.without_after for c,_,_ in batch]
  assert len(calls)==count*len(phases) and len(queues)==1
  queue_of=[(d,q)for _,d,q in batch];assert len(set(queue_of))==1
  tracker=DepsTracker(queue_of);records=[];output_sets=[];dependencies=[]
  def span(value):
    base,lane,offset=unwrap_lane(value)
    assert base.op is Ops.PARAM and lane is None,'Every timed buffer must be an explicit unsharded JIT PARAM'
    return base.key,base.arg.slot,offset,value.max_numel()*value.dtype.itemsize
  def key(spans):return tuple((p[0],p[2],p[3])for p in spans)
  for tag,call in enumerate(calls):
    assert call.op is Ops.CALL and call.body.op is Ops.PROGRAM
    name=call.body.src[0].arg.name
    assert name==phases[tag%len(phases)],(name,phases)
    outs,ins=get_call_outs_ins(call)
    assert len(outs)==(3 if name=='agent_sparse_fc2_compress' else 1)
    written=tuple(span(call.src[1+i])for i in outs)
    read=tuple(span(call.src[1+i])for i in ins)
    assert len(set(key(written)))==len(written)
    waits=tracker.access_resources([(b,i in outs)for i,b in enumerate(call.src[1:])if not b.is_variable],tag)
    if tag:assert tag-1 in waits,('Missing adjacent RAW/WAR/WAW serialization',tag,waits)
    if tag>=len(phases):assert tag-len(phases) in waits,('Missing same-phase WAW dependency',tag,waits)
    if len(phases)==2 and tag%2:
      assert set(key(output_sets[-1])).issubset(set(key(read))),'GEMM must read every actual compression output'
    output_sets.append(written);dependencies.append(waits)
    global_args=[span(call.src[1+i])for i in call.body.arg.globals]
    records.append(dict(name=name,device=batch[tag][1],program_key=call.body.key.hex(),
      global_input_slots=[v[1]for v in global_args],output_argument_indices=list(outs),input_argument_indices=list(ins),
      output_spans=[dict(input_slot=p[1],byte_offset=p[2],bytes=p[3])for p in written]))
  for phase in range(len(phases)):
    assert len({key(s)for s in output_sets[phase::len(phases)]})==1,'Each phase must reuse the same shared output spans'
  if len(phases)==2:
    all_writes=(*output_sets[0],*output_sets[1])
    assert all(a[0]!=b[0] or a[2]+a[3]<=b[2] or b[2]+b[3]<=a[2] for i,a in enumerate(all_writes)for b in all_writes[i+1:])
  ordered=[]
  for command in next(iter(queues.values())):
    command=command.without_after
    if command.op is Ops.CALL and command.body.op is Ops.PROGRAM:ordered.append(command)
    elif command.op is Ops.CALL and command.body.op is Ops.CUSTOM_FUNCTION and command.body.arg.name=='exec_barrier':ordered.append('barrier')
  expected=[item for i,c in enumerate(calls)for item in ((c,)if i==0 else('barrier',c))]
  assert ordered==expected,'Every adjacent call must have an ordered HCQ execution barrier'
  return dict(captured_kernel_calls=len(calls),invocations=count,phases=list(phases),dependencies=dependencies,
    ordered_execution_barriers=len(calls)-1,actual_queue_builder_observed=True,all_phase_outputs_shared=True,
    actual_output_indices_derived=True,kernel_arguments=records)


@contextmanager
def observe(jit,phases):
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  original=hcq2._build_queues;jit._sparse_fc2_proofs=[]
  def build(ctx):
    queues=original(ctx)
    calls=[c.without_after for c,_,_ in ctx.batch]
    named=[c for c in calls if c.op is Ops.CALL and c.body.op is Ops.PROGRAM and c.body.src[0].arg.name in phases]
    if named and all(hcq2.unwrap_lane(c.src[1+i])[0].op is Ops.PARAM for c in named for i in c.body.arg.globals):
      jit._sparse_fc2_proofs.append(prove(ctx.batch,queues,phases))
    return queues
  hcq2._build_queues=build
  try:yield
  finally:hcq2._build_queues=original


def inspect(jit,phases):
  from tinygrad.runtime.support import hcq2
  entries=[c.without_after for c in jit.captured._linear.src]
  graphs=[c for c in entries if isinstance(getattr(c.arg,'aux',None),hcq2.HCQInfo)]
  if graphs:
    assert len(graphs)==len(entries)
    assert len(jit._sparse_fc2_proofs)==1,'Expected exactly one observed capture-time HCQ queue'
    proof=jit._sparse_fc2_proofs[0];actual=[k for c in graphs for k in c.arg.aux.kernels]
    assert len(actual)==proof['captured_kernel_calls']
    for got,want in zip(actual,proof['kernel_arguments'],strict=True):
      assert got[0]==want['device'] and got[1]==want['name'] and got[4]==bytes.fromhex(want['program_key'])
      assert list(got[5])==want['global_input_slots']
      assert list(got[6][0])==want['output_argument_indices'] and list(got[6][1])==want['input_argument_indices']
    return dict(proof,captured_representation='HCQInfo',hcq_metadata_verified=True)
  # CPU controls use the same pinned dependency/barrier builder on real calls.
  batch=[(c,'CPU','COMPUTE:0')for c in entries]
  ctx=SimpleNamespace(batch=batch,calls={('CPU','COMPUTE:0'):list(range(len(batch)))},
    peers={('CPU','COMPUTE:0'):set()},queues={'CPU':['COMPUTE:0']},stamps=lambda *_:())
  return dict(prove(batch,hcq2._build_queues(ctx),phases),captured_representation='PROGRAM',hcq_metadata_verified=False)
