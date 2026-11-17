"""CPU regressions for real TinyJit captures and pinned HCQ graph lowering.

The small CPU kernels exercise graph structure only. Numerical INT4 kernels and
GPU controls remain unchanged. No GPU device may be opened by this test.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def matrix(out,a,w,xs,ws,bias):
  from tinygrad.uop.ops import UOp,KernelInfo
  r=UOp.range(out.numel(),0)
  value=sum(v.flatten().index(0).load().float() for v in (a,w,xs,ws,bias))
  return out.flatten().index(r).store(value.cast(out.dtype)).end(r).sink(arg=KernelInfo(name='agent_int4_scaled'))


def quantizer(packed,scale,x):
  from tinygrad.uop.ops import UOp,KernelInfo
  rp,rs=UOp.range(packed.numel(),0),UOp.range(scale.numel(),1)
  p=packed.flatten().index(rp).store((x.flatten().index(rp%x.numel()).load().float()+1).cast(packed.dtype)).end(rp)
  s=scale.flatten().index(rs).store(x.flatten().index(rs%x.numel()).load().float()+2).end(rs)
  return UOp.sink(p,s,arg=KernelInfo(name='agent_int4_quantize'))


def second_output(a,out):
  from tinygrad.uop.ops import UOp,KernelInfo
  r=UOp.range(out.numel(),0)
  return out.flatten().index(r).store((a.flatten().index(0).load()*2).cast(out.dtype)).end(r).sink(arg=KernelInfo(name='serial_cpu_output_second'))


def grouped_matrix(out,a,w,xs,ws):
  result=matrix(out,a,w,xs,ws,ws)
  return result.replace(arg=replace(result.arg,name='agent_int4_grouped'))


def grouped_quantizer(packed,scale,x):
  result=quantizer(packed,scale,x)
  return result.replace(arg=replace(result.arg,name='agent_int4_grouped_quantize'))


def fixture(kind):
  from tinygrad import Tensor,TinyJit,dtypes
  grouped=kind.startswith('group-');base_kind=kind.removeprefix('group-')
  qfn=grouped_quantizer if grouped else quantizer
  qname='agent_int4_grouped_quantize' if grouped else 'agent_int4_quantize'
  gname='agent_int4_grouped' if grouped else 'agent_int4_scaled'
  x=Tensor(np.array([2,3,4,5],np.float32),device='CPU').cast(dtypes.bfloat16).realize()
  q=Tensor.ones(32,dtype=dtypes.uint8,device='CPU').realize()
  s=Tensor.ones(4,dtype=dtypes.float32,device='CPU').realize()
  w=Tensor.ones(16,dtype=dtypes.float32,device='CPU').realize()
  ws=Tensor.ones(4,dtype=dtypes.float32,device='CPU').realize()
  bias=Tensor.zeros(4,dtype=dtypes.bfloat16,device='CPU').realize()
  out=Tensor.zeros(4,4,dtype=dtypes.bfloat16,device='CPU').realize()
  if base_kind=='quantizer':
    def run(x,q,s):
      for _ in range(40):q,s=Tensor.custom_kernel(q,s,x,fxn=qfn)[:2]
      Tensor.realize(q,s);return q,s
    args=(x,q,s);names=(qname,);count=40;full=False
  elif base_kind=='matrix' and grouped:
    def run(q,w,s,ws,out):
      for _ in range(40):out=Tensor.custom_kernel(out,q,w,s,ws,fxn=grouped_matrix)[0]
      return out.realize()
    args=(q,w,s,ws,out);names=(gname,);count=40;full=False
  elif base_kind=='matrix':
    def run(q,w,s,ws,bias,out):
      for _ in range(40):out=Tensor.custom_kernel(out,q,w,s,ws,bias,fxn=matrix)[0]
      return out.realize()
    args=(q,w,s,ws,bias,out);names=(gname,);count=40;full=False
  elif base_kind=='full' and grouped:
    def run(x,w,ws,q,s,out):
      for _ in range(40):
        q,s=Tensor.custom_kernel(q,s,Tensor(x.uop.after(out.uop)),fxn=qfn)[:2]
        out=Tensor.custom_kernel(out,q,w,s,ws,fxn=grouped_matrix)[0]
      return out.realize()
    args=(x,w,ws,q,s,out);names=(qname,gname);count=80;full=True
  elif base_kind=='full':
    def run(x,w,ws,bias,q,s,out):
      for _ in range(40):
        q,s=Tensor.custom_kernel(q,s,Tensor(x.uop.after(out.uop)),fxn=qfn)[:2]
        out=Tensor.custom_kernel(out,q,w,s,ws,bias,fxn=matrix)[0]
      return out.realize()
    args=(x,w,ws,bias,q,s,out);names=(qname,gname);count=80;full=True
  else:
    def run(a,out):
      for _ in range(40):out=Tensor.custom_kernel(a,out,fxn=second_output)[1]
      return out.realize()
    args=(w,out);names=('serial_cpu_output_second',);count=40;full=False
  jit=TinyJit(run)
  for _ in range(3):jit(*args)
  if base_kind=='quantizer':np.testing.assert_array_equal(jit.captured.ret[0].numpy(),np.tile([3,4,5,6],8))
  else:np.testing.assert_array_equal(jit.captured.ret.float().numpy(),10 if kind=='group-full' else 5 if kind=='group-matrix' else 9 if kind=='full' else 4 if kind=='matrix' else 2)
  return jit,names,count,full


def holder(linear,proofs=()):return SimpleNamespace(captured=SimpleNamespace(_linear=linear),_int4_serial_proofs=list(proofs))


def inspect(check,grouped,jit,names,count,full):
  if names[0].startswith('agent_int4_grouped'):
    return grouped.serial_proof(jit.captured._linear,count,full,getattr(jit,'_int4_serial_proofs',()),expected_name=names)
  return check.inspect_serial(jit,names,count,full)


def rejects(label,fn,records):
  try:fn()
  except AssertionError as exc:records.append(dict(test=label,rejected=True,reason=str(exc)));return
  raise AssertionError(f'{label}: invalid graph was accepted')


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  from tinygrad import Device
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.runtime.support import hcq2
  check=local('agent-int4-scaled-check');grouped=local('agent-int4-grouped-check');records=[];fixtures={}
  original_builder=hcq2._build_queues
  for kind in ('quantizer','matrix','full','second-output','group-quantizer','group-matrix','group-full'):
    jit,names,count,full=fixture(kind);fixtures[kind]=(jit,names,count,full)
    proof=inspect(check,grouped,jit,names,count,full)
    assert proof['ordered_execution_barriers']==count-1
    slots=proof['kernel_arguments'][0]['output_spans']
    assert all(s['input_slot']!=0 for s in slots),slots
    if kind=='second-output':assert proof['metadata_outs'][0]==[1]
    records.append(dict(test='actual-CPU-TinyJit-'+kind,proof=proof))
    wrapped=jit.captured._linear.replace(src=tuple(c.after(UOp(Ops.NOOP)) for c in jit.captured._linear.src))
    after_proof=inspect(check,grouped,holder(wrapped),names,count,full)
    assert after_proof['kernel_arguments']==proof['kernel_arguments']
    records.append(dict(test='AFTER-wrapped-PROGRAM-'+kind,pass_=True))
    # Invoke the real pinned finalizer on actual captured PARAM calls. It builds
    # real HCQInfo and submit bodies, but no queue program is linked or executed.
    calls=[c.without_after for c in jit.captured._linear.src]
    ctx=hcq2.BatchCtx([(c,'CPU','COMPUTE:0') for c in calls],False)
    observed=holder(None)
    with check.observe_serial_queue(observed,names,count,full):graph=hcq2._finalize_batch(ctx)
    assert hcq2._build_queues is original_builder
    hcq_linear=UOp(Ops.LINEAR,src=(graph.after(UOp(Ops.NOOP)),))
    hcq_jit=holder(hcq_linear,observed._int4_serial_proofs)
    got=inspect(check,grouped,hcq_jit,names,count,full)
    assert got['captured_representation']=='HCQInfo' and got['hcq_metadata_verified']
    records.append(dict(test='real-HCQ-finalizer-AFTER-'+kind,proof=got))
    rejects('HCQ-missing-observed-proof-'+kind,lambda:inspect(check,grouped,holder(hcq_linear),names,count,full),records)
    aux=graph.arg.aux;bad=list(aux.kernels);first=list(bad[0]);first[5]=(*first[5][:-1],9999);bad[0]=tuple(first)
    changed=graph.replace(arg=replace(graph.arg,aux=replace(aux,kernels=tuple(bad))))
    rejects('HCQ-corrupted-buffer-map-'+kind,lambda:inspect(check,grouped,holder(UOp(Ops.LINEAR,src=(changed,)),observed._int4_serial_proofs),names,count,full),records)
    bad=list(aux.kernels);first=list(bad[0]);first[6]=((999,),first[6][1]);bad[0]=tuple(first)
    changed=graph.replace(arg=replace(graph.arg,aux=replace(aux,kernels=tuple(bad))))
    rejects('HCQ-corrupted-write-metadata-'+kind,lambda:inspect(check,grouped,holder(UOp(Ops.LINEAR,src=(changed,)),observed._int4_serial_proofs),names,count,full),records)
    rejects('HCQ-wrong-count-'+kind,lambda:inspect(check,grouped,hcq_jit,names,count-1,full),records)
  for kind in ('matrix','full','group-matrix','group-full'):
    jit,names,count,full=fixtures[kind];calls=[c.without_after for c in jit.captured._linear.src];changed=[]
    for i,c in enumerate(calls):
      if c.body.src[0].arg.name in ('agent_int4_scaled','agent_int4_grouped'):
        src=list(c.src);value=src[1];base,_,_=hcq2.unwrap_lane(value)
        src[1]=value.substitute({base:UOp.param(100+i,base.dtype,base.max_numel(),base.device)},walk=True)
        c=c.replace(src=tuple(src))
      changed.append(c)
    rejects('independent-output-storage-'+kind,lambda:inspect(check,grouped,holder(UOp(Ops.LINEAR,src=tuple(changed))),names,count,full),records)
    batch=[(c,'CPU','COMPUTE:0') for c in changed];ctx=hcq2.BatchCtx(batch,False)
    observer=holder(None)
    rejects('observer-independent-storage-'+kind,lambda:_observe_bad(check,observer,names,count,full,ctx),records)
    assert hcq2._build_queues is original_builder
  for kind in ('full','group-full'):
    jit,names,count,full=fixtures[kind];calls=[c.without_after for c in jit.captured._linear.src]
    for argument,label in ((1,'packed'),(3,'scale')):
      changed=list(calls);src=list(changed[1].src);value=src[1+argument];base,_,_=hcq2.unwrap_lane(value)
      src[1+argument]=value.substitute({base:UOp.param(700+argument,base.dtype,base.max_numel(),base.device)},walk=True)
      changed[1]=changed[1].replace(src=tuple(src))
      rejects('missing-quantizer-RAW-'+kind+'-'+label,lambda:inspect(check,grouped,holder(UOp(Ops.LINEAR,src=tuple(changed))),names,count,full),records)
    changed=list(calls);body=changed[0].body.replace(arg=replace(changed[0].body.arg,outs=(0,)))
    changed[0]=changed[0].replace(src=(body,*changed[0].src[1:]))
    rejects('missing-scale-write-metadata-'+kind,lambda:inspect(check,grouped,holder(UOp(Ops.LINEAR,src=tuple(changed))),names,count,full),records)
    ctx=hcq2.BatchCtx([(c,'CPU','COMPUTE:0') for c in calls],False);queues=original_builder(ctx)
    broken={key:[c for c in cmds if not (c.op is Ops.CALL and c.body.op is Ops.CUSTOM_FUNCTION and c.body.arg.name=='exec_barrier')]
            for key,cmds in queues.items()}
    rejects('missing-real-execution-barriers-'+kind,lambda:check.serial_queue_proof(ctx.batch,broken,names,count,full),records)
  assert hcq2._build_queues is original_builder
  assert not any(d.startswith(('AMD','NV','CUDA')) for d in Device._opened_devices),Device._opened_devices
  files=[Path(__file__),Path(__file__).with_name('agent-int4-scaled-check.py'),Path(__file__).with_name('agent-int4-grouped-check.py'),
         Path(__file__).with_name('agent-int4-scaled-kernel.py'),Path(__file__).with_name('agent-int4-grouped-kernel.py'),
         Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad/tinygrad/runtime/support/hcq2.py')]
  immutable={}
  for name in ('agent-int4-scaled','agent-int4-grouped'):
    kernel=Path(__file__).with_name(name+'-kernel.py');manifest=Path(__file__).with_name(name+'-compiled-v5')/'report.json'
    expected=json.loads(manifest.read_text())['source_hashes'][str(kernel)]
    actual=hashlib.sha256(kernel.read_bytes()).hexdigest();assert actual==expected,(kernel,actual,expected)
    immutable[str(kernel)]=actual
  report=dict(cpu_only=True,all_pass=True,gpu_devices_opened=False,cases=records,immutable_kernel_hashes=immutable,
              source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(dict(all_pass=True,cases=len(records),output=str(args.output))),flush=True)


def _observe_bad(check,observer,names,count,full,ctx):
  from tinygrad.runtime.support import hcq2
  with check.observe_serial_queue(observer,names,count,full):hcq2._finalize_batch(ctx)


if __name__=='__main__':main()
