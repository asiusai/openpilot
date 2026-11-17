#!/usr/bin/env python3
"""CPU/NULL execution only: real buffers, pinned graphs, invalid graph rejection."""
import argparse,copy,hashlib,importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name):
  s=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def queue_context(calls,device='CPU'):
  return SimpleNamespace(batch=[(c.without_after,device,'COMPUTE:0')for c in calls],
    calls={(device,'COMPUTE:0'):list(range(len(calls)))},peers={(device,'COMPUTE:0'):set()},
    queues={device:['COMPUTE:0']},stamps=lambda *_:())


def toy():
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp,KernelInfo
  def comp(v,m,f,x):
    value=x.flatten().index(0).load().cast(dtypes.uint32)
    stores=[]
    for index,(out,add)in enumerate(((v,0),(m,1),(f,2))):
      r=UOp.range(out.max_numel(),index)
      stores.append(out.flatten().index(r).store(value+add).end(r))
    return UOp.sink(*stores,arg=KernelInfo(name='agent_sparse_fc2_compress'))
  def gemm_program(out,x,w,s,t,v,m,f,seed):
    value=sum(a.flatten().index(0).load().cast(dtypes.float32)for a in(x,w,s,t,v,m,f,seed))
    r=UOp.range(out.max_numel(),0)
    return out.flatten().index(r).store(value.cast(out.dtype)).end(r).sink(arg=KernelInfo(name='agent_sparse_numeric_gemm'))
  def original_program(out,x,w,s,t):
    value=sum(a.flatten().index(0).load().cast(dtypes.float32)for a in(x,w,s,t))
    r=UOp.range(out.max_numel(),0)
    return out.flatten().index(r).store(value.cast(out.dtype)).end(r).sink(arg=KernelInfo(name='alpamayo_fp8_gemm'))
  def compress(x,outputs):return tuple(Tensor.custom_kernel(*outputs,x,fxn=comp)[:3])
  def gemm(x,w,s,t,workspace,out,seed):return Tensor.custom_kernel(out,x,w,s,t,*workspace,seed,fxn=gemm_program)[0]
  def reference(x,w,s,t,out):return Tensor.custom_kernel(out,x,w,s,t,fxn=original_program)[0]
  return SimpleNamespace(compress=compress,gemm=gemm,reference=reference)


def cpu_graphs(R,S):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.runtime.support import hcq2
  shared=tuple(R.physical(a,'CPU')for a in(np.full((64,32),2,np.uint8),np.full((64,32),3,np.uint8),np.array([4.],np.float32),np.array([5.],np.float32)))
  kernel=toy();records=[]
  for hybrid in(False,True):
    expected=np.full((64,64),0x41b8 if hybrid else 0x4160,np.uint16) # 23 or 14
    graph=R.make_graph(kernel,S,shared,expected,hybrid)
    R.check_output(graph)
    calls=[c.without_after for c in graph.jit.captured._linear.src]
    bound=S.bindings(graph.args)
    raw=[c.replace(src=(c.body,*[a.substitute({u:bound[u.arg.slot]for u in a.toposort()if u.op is Ops.PARAM},walk=True)for a in c.src[1:]]))for c in calls]
    ctx=queue_context(raw);queues=hcq2._build_queues(ctx)
    buffer_proof=S.prove([(c.after(UOp.sink()),d,q)for c,d,q in ctx.batch],queues,graph.phases,bound,graph.rules)
    assert all(k=='BUFFER'for d in buffer_proof['kernel_arguments']for k in d['global_storage_kinds'])
    # Exact current mutable scalars and arrays, with output poison.
    for tensor,value in zip(shared,(7,7,7,7),strict=True):R.write(tensor,np.full(tensor.shape,value,np.uint8 if tensor.dtype==dtypes.uint8 else np.float32))
    changed=np.full((64,64),0x4250 if hybrid else 0x41e0,np.uint16) # 52 or 28
    R.poison(graph);graph.jit(*graph.args);R.check_output(graph,changed)
    for tensor,array in zip(shared,(np.full((64,32),2,np.uint8),np.full((64,32),3,np.uint8),np.array([4.],np.float32),np.array([5.],np.float32)),strict=True):R.write(tensor,array)
    bad=raw.copy();out_index=buffer_proof['kernel_arguments'][0]['output_argument_indices'][0]
    rogue=R.physical(np.zeros(graph.args[graph.output_slot].shape,np.uint16),'CPU',dtypes.bfloat16)
    source=list(bad[0].src);source[1+out_index]=rogue.uop.base
    bad[0]=bad[0].replace(src=tuple(source))
    invalid=[('independent-output',bad,None),
      ('missing-barrier',raw,{key:[c for c in value if c.without_after.op is Ops.CALL and c.without_after.body.op is Ops.PROGRAM]for key,value in queues.items()})]
    if hybrid:invalid.append(('reordered',list(reversed(raw)),None))
    for label,changed_calls,changed_queues in invalid:
      caught=False
      try:
        changed_ctx=queue_context(changed_calls)
        S.prove(changed_ctx.batch,changed_queues or hcq2._build_queues(changed_ctx),graph.phases,bound,graph.rules)
      except AssertionError:caught=True
      assert caught,'Invalid graph accepted: '+label
    records.append(dict(hybrid=hybrid,PARAM_proof=graph.proof,BUFFER_proof=buffer_proof,
      AFTER_supported=True,current_input_mutation_exact=True,independent_outputs_rejected=True,
      hybrid_reordered_calls_rejected=hybrid,missing_barriers_rejected=True))
  return records


def null_graphs(C,R,S,N,kernel):
  """Actual full-shape native PROGRAMs and HCQ capture; NULL never runs GPU ISA."""
  from tinygrad import Tensor,Device,dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'));renderer.compiler=LLVMCompiler('AMDGPU','gfx1200','-cumode')
  def original_program(*params):return to_program(kernel.original().original_program(*params,renderer=renderer),renderer)
  def original(x,w,s,t,out):return Tensor.custom_kernel(out,x,w,s,t,fxn=original_program)[0]
  shared=tuple(R.physical(a,'NULL')for a in(np.zeros((2112,9216),np.uint8),np.zeros((2048,9216),np.uint8),np.array([1.],np.float32),np.array([1.],np.float32)))
  records=[]
  for hybrid in(False,True):
    inputs=shared if hybrid else(R.typed_view(shared[0],dtypes.fp8e4m3),R.typed_view(shared[1],dtypes.fp8e4m3),*shared[2:])
    graph=R.make_graph(kernel,S,inputs,np.zeros((2112,2048),np.uint16),hybrid,R.binaries(N),original_kernel=original)
    graph.jit(*graph.args);Device['NULL'].synchronize()
    assert graph.proof['hcq_metadata_verified']and graph.proof['captured_representation']=='HCQInfo'
    assert graph.proof['captured_kernel_calls']==(80 if hybrid else 40)
    records.append(dict(hybrid=hybrid,actual_native_binaries=True,actual_full_shape=True,
      actual_NULL_HCQ_capture=True,numeric_execution=False,proof=graph.proof))
  return records


def nonzero_output(R,S):
  from tinygrad import Tensor,TinyJit,dtypes
  from tinygrad.uop.ops import UOp,KernelInfo
  def program(x,out,w,s,t):
    value=sum(a.flatten().index(0).load().cast(dtypes.float32)for a in(x,w,s,t))
    r=UOp.range(out.max_numel(),0)
    return out.flatten().index(r).store(value.cast(out.dtype)).end(r).sink(arg=KernelInfo(name='alpamayo_fp8_gemm'))
  shared=tuple(R.physical(a,'CPU')for a in(np.full((64,32),2,np.uint8),np.full((64,32),3,np.uint8),np.array([4.],np.float32),np.array([5.],np.float32)))
  out=R.physical(np.zeros((64,64),np.uint16),'CPU',dtypes.bfloat16);args=(*shared,out)
  def run(x,w,s,t,output):
    for _ in range(40):output=Tensor.custom_kernel(x,output,w,s,t,fxn=program)[1].realize()
    return output
  jit=TinyJit(run);rules={'alpamayo_fp8_gemm':dict(slots=[0,4,1,2,3],output_slots=[4])};phases=('alpamayo_fp8_gemm',)
  jit(*args)
  with S.observe(jit,args,phases,rules):jit(*args)
  proof=S.inspect(jit,args,phases,rules)
  assert all(x['output_argument_indices']==[1]and x['output_spans'][0]['input_slot']==4 for x in proof['kernel_arguments'])
  np.testing.assert_array_equal(R.output_bits(jit(*args)),np.full((64,64),0x4160,np.uint16))
  return proof


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  if os.environ.get('DEV')!='CPU':raise ValueError('Run with DEV=CPU')
  C,R,S=map(load,('agent-sparse-timing-common','agent-sparse-timing-run','agent-sparse-timing-serial'))
  report=dict(cpu_only=True,all_pass=False)
  try:
    N,old,numeric,capture,records,_,_,evidence=C.numeric_inputs()
    report['templates']=[dict(layer=r['layer'],changed_elements=e['hybrid_vs_original_BF16']['differing_bits_elements'],
      reconstructed_full_output_sha256=C.array_sha(expected),all_element_inventory_complete=True)for r,d,expected,e in records]
    entry=records[1][3]
    for name in('incomplete','bad-bits','repeated-coordinate'):
      changed=copy.deepcopy(entry)
      if name=='incomplete':changed['hybrid_vs_original_BF16']['worst_values'].pop(0)
      elif name=='bad-bits':changed['hybrid_vs_original_BF16']['worst_values'][0]['actual_bits']='0x3ebc'
      else:changed['hybrid_vs_original_BF16']['worst_values'][1]=changed['hybrid_vs_original_BF16']['worst_values'][0]
      try:C.candidate_expected(records[1][1]['original_bf16'],changed)
      except AssertionError:pass
      else:raise AssertionError('Invalid characterized output accepted: '+name)
    from tinygrad import dtypes,Device
    scalar=R.physical(np.array(1.,np.float32),'CPU');S.bindings((scalar,))
    R.write(scalar,np.array(2.,np.float32));assert scalar.numpy()==2.
    report['physical_scalar_0d_and_current_write']=True
    raw=R.physical(np.array([56,64],np.uint8),'CPU');view=R.typed_view(raw,dtypes.fp8e4m3)
    assert raw.uop.base.buffer is view.uop.base.buffer
    R.write(raw,np.array([68,72],np.uint8));np.testing.assert_array_equal(view.bitcast(dtypes.uint8).numpy(),np.array([68,72],np.uint8))
    report['typed_view_shares_current_physical_bytes']=True
    report['CPU_graphs']=cpu_graphs(R,S)
    report['nonzero_output_argument_1']=nonzero_output(R,S)
    q=np.zeros((64,64),np.uint8);q[17,:4]=[56,64,68,72]
    eligibility=R.eligibility(q,old.fmt().compress(q)[2],17)
    assert eligibility['physical_fraction']==.875 and eligibility['real_rows_only_fraction']==1.
    assert eligibility['partial_real_tile_rows']==1 and eligibility['padded_rows']==47
    report['padding_mask_independent_control']=eligibility
    kernel=N.load(C.NUMERIC/'agent-sparse-numeric-kernel.py','timing_cpu_qualified_native')
    report['NULL_native_graphs']=null_graphs(C,R,S,N,kernel)
    ticks=iter((100,40100));events=[]
    timing=R.measure(lambda *args:events.append('graph'),(),lambda:events.append('completion'),lambda:next(ticks))
    assert events==['graph','completion']and timing['completion_fences_inside_interval']==1 and timing['amortized_ms']==.001
    report['timer_only_graph_and_one_completion']=True
    assert not any(d.startswith('AMD')for d in Device._opened_devices)
    report.update(all_pass=True,opened_devices=sorted(Device._opened_devices),source_hashes={p.name:C.sha(p)for p in HERE.glob('agent-sparse-timing-*.py')})
  except BaseException as error:report['error']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(a.output,report)
  print(json.dumps(dict(cpu_only=True,all_pass=True,CPU_graphs=2,NULL_actual_shape_graphs=2,opened_devices=report['opened_devices'])))


if __name__=='__main__':main()
