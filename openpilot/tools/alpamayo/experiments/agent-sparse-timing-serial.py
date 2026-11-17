"""Actual pinned HCQ proof, with explicit PARAM and physical Buffer bindings."""
from contextlib import contextmanager
import hashlib
from types import SimpleNamespace


def bindings(args):
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.uop.ops import Ops
  bufs,_,names,_=_prepare_jit_inputs(tuple(args),{})
  assert names==list(range(len(args)))and len(bufs)==len(args)
  assert all(b.op is Ops.BUFFER and b.buffer is not None and b.buffer.is_allocated()for b in bufs)
  assert len({(id(b.buffer.base),b.buffer.offset)for b in bufs})==len(bufs),'Aliased graph inputs'
  return bufs


def span(value,bound):
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  base,lane,off=unwrap_lane(value);assert lane is None,'No sharded timing buffers'
  if base.op is Ops.PARAM:
    assert base.arg.name is None,'Link-time PARAM needs an explicit physical binding'
    slot=base.arg.slot;assert 0<=slot<len(bound)
    kind='PARAM'
  elif base.op is Ops.BUFFER:
    storage=base.buffer;assert storage is not None and storage.is_allocated()
    candidates=[i for i,b in enumerate(bound)if storage.base is b.buffer.base and storage.offset==b.buffer.offset]
    assert len(candidates)==1,'Unbound or independent physical buffer'
    slot=candidates[0];kind='BUFFER'
  else:raise AssertionError('Unbound timing storage: '+base.op.name)
  size=value.max_numel()*value.dtype.itemsize
  assert off>=0 and off+size<=bound[slot].buffer.nbytes
  return slot,off,size,kind


def layouts(hybrid):
  if hybrid:return {
    'agent_sparse_fc2_compress':dict(slots=[4,5,6,0],output_slots=[4,5,6]),
    'agent_sparse_numeric_gemm':dict(slots=[7,0,1,2,3,4,5,6,8],output_slots=[7])}
  return {'alpamayo_fp8_gemm':dict(slots=[4,0,1,2,3],output_slots=[4])}


def prove(batch,queues,phases,bound,rules,count=40,binaries=None):
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import DepsTracker
  calls=[c.without_after for c,_,_ in batch]
  assert len(calls)==count*len(phases)and len(queues)==1
  queue_of=[(d,q)for _,d,q in batch];assert len(set(queue_of))==1
  tracker=DepsTracker(queue_of);outputs=[];reads=[];descriptions=[];dependencies=[]
  def key(values):return tuple(v[:3]for v in values)
  for tag,c in enumerate(calls):
    assert c.op is Ops.CALL and c.body.op is Ops.PROGRAM,'Unexpected copy/host/kernel operation'
    name=c.body.src[0].arg.name;assert name==phases[tag%len(phases)]
    out_indices,in_indices=get_call_outs_ins(c)
    globals_=list(c.body.arg.globals)
    actual=[span(c.src[1+i],bound)for i in globals_]
    assert [x[0]for x in actual]==rules[name]['slots'],'Actual program arguments differ from declared buffer API'
    written=[span(c.src[1+i],bound)for i in out_indices];read=[span(c.src[1+i],bound)for i in in_indices]
    assert [x[0]for x in written]==rules[name]['output_slots'],'Actual output indices/slots differ'
    assert set(x[0]for x in read)==set(rules[name]['slots'])-set(rules[name]['output_slots']),'Missing actual declared input read'
    assert len(set(key(written)))==len(written)
    assert all(x[1]==0 and x[2]==bound[x[0]].buffer.nbytes for x in written+read),'Partial program arguments cannot qualify'
    waits=tracker.access_resources([(b,i in out_indices)for i,b in enumerate(c.src[1:])if not b.is_variable],tag)
    if tag:assert tag-1 in waits,'Missing adjacent RAW/WAR/WAW dependency'
    if tag>=len(phases):assert tag-len(phases)in waits,'Missing same-phase WAW dependency'
    if len(phases)==2 and tag%2:assert set(key(outputs[-1])).issubset(set(key(read)))
    binary=next(s.arg for s in c.body.src if s.op is Ops.BINARY);binary_hash=hashlib.sha256(binary).hexdigest()
    if binaries is not None:assert binary_hash==binaries[name],'Compiled timing binary differs from qualified binary'
    all_param=all(x[3]=='PARAM'for x in actual)
    descriptions.append(dict(name=name,device=batch[tag][1],program_key=c.body.key.hex(),binary_sha256=binary_hash,
      actual_global_indices=globals_,global_input_slots=[x[0]for x in actual],global_storage_kinds=[x[3]for x in actual],
      HCQ_metadata_slots=[x[0]for x in actual]if all_param else[],
      output_argument_indices=list(out_indices),input_argument_indices=list(in_indices),
      output_spans=[dict(input_slot=x[0],byte_offset=x[1],bytes=x[2])for x in written]))
    outputs.append(written);reads.append(read);dependencies.append(waits)
  for phase in range(len(phases)):
    assert len({key(o)for o in outputs[phase::len(phases)]})==1,'Independent phase outputs'
  workspace=[]
  if len(phases)==2:
    written=outputs[0]+outputs[1]
    assert all(a[0]!=b[0]or a[1]+a[2]<=b[1]or b[1]+b[2]<=a[1]for i,a in enumerate(written)for b in written[i+1:])
    for v in outputs[0]:
      for i in range(count):
        assert v[:3]in key(reads[2*i+1]),'Missing complete workspace RAW access'
        if i:
          assert 2*i-1 in dependencies[2*i]and 2*i-2 in dependencies[2*i]
      workspace.append(dict(input_slot=v[0],bytes=v[2],RAW_edges=count,WAR_edges=count-1,WAW_edges=count-1))
  ordered=[]
  for c in next(iter(queues.values())):
    c=c.without_after
    if c.op is Ops.CALL and c.body.op is Ops.PROGRAM:ordered.append(c)
    elif c.op is Ops.CALL and c.body.op is Ops.CUSTOM_FUNCTION and c.body.arg.name=='exec_barrier':ordered.append('barrier')
  expected=[x for i,c in enumerate(calls)for x in((c,)if i==0 else('barrier',c))]
  assert ordered==expected,'Actual HCQ queue lacks ordered execution barriers'
  return dict(captured_kernel_calls=len(calls),invocations=count,phases=list(phases),dependencies=dependencies,
    ordered_execution_barriers=len(calls)-1,workspace_edges=workspace,kernel_arguments=descriptions,
    actual_queue_builder_observed=True,physical_BUFFER_and_PARAM_supported=True,all_phase_outputs_shared=True)


@contextmanager
def observe(jit,args,phases,rules,count=40,binaries=None):
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  bound=bindings(args);original=hcq2._build_queues;jit._sparse_timing_proofs=[]
  def build(ctx):
    queues=original(ctx);calls=[c.without_after for c,_,_ in ctx.batch]
    if len(calls)==count*len(phases)and all(c.op is Ops.CALL and c.body.op is Ops.PROGRAM and c.body.src[0].arg.name in phases for c in calls):
      jit._sparse_timing_proofs.append(prove(ctx.batch,queues,phases,bound,rules,count,binaries))
    return queues
  hcq2._build_queues=build
  try:yield
  finally:hcq2._build_queues=original


def inspect(jit,args,phases,rules,count=40,binaries=None):
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support import hcq2
  bound=bindings(args);entries=[c.without_after for c in jit.captured._linear.src]
  graphs=[c for c in entries if isinstance(getattr(c.arg,'aux',None),hcq2.HCQInfo)]
  if graphs:
    assert len(graphs)==len(entries)==1,'One complete HCQ graph required'
    candidates=getattr(jit,'_sparse_timing_proofs',())
    actual=[k for g in graphs for k in g.arg.aux.kernels]
    matches=[]
    for proof in candidates:
      if len(actual)!=proof['captured_kernel_calls']:continue
      if all(got[0]==want['device']and got[1]==want['name']and got[4]==bytes.fromhex(want['program_key'])
             and list(got[5])==want['HCQ_metadata_slots']and list(got[6][0])==want['output_argument_indices']
             and list(got[6][1])==want['input_argument_indices']for got,want in zip(actual,proof['kernel_arguments'],strict=True)):
        matches.append(proof)
    assert matches,'No actual HCQ argument/program proof matched the captured graph'
    return dict(matches[-1],captured_representation='HCQInfo',hcq_metadata_verified=True)
  assert all(c.op is Ops.CALL and c.body.op is Ops.PROGRAM for c in entries)
  device=args[0].device;batch=[(c,device,'COMPUTE:0')for c in entries]
  ctx=SimpleNamespace(batch=batch,calls={(device,'COMPUTE:0'):list(range(len(batch)))},
    peers={(device,'COMPUTE:0'):set()},queues={device:['COMPUTE:0']},stamps=lambda *_:())
  return dict(prove(batch,hcq2._build_queues(ctx),phases,bound,rules,count,binaries),captured_representation='PROGRAM',hcq_metadata_verified=False)
