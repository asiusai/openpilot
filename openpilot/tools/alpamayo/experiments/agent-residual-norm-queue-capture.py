"""Actual frozen residual/norm graph capture on NULL:AMDLLVM:gfx1200. Never computes numeric outputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OLD=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-residual-norm-retry-source-final')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_queue',path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def capture(artifacts,auditor=None):
  os.environ.update(DEV='NULL:AMDLLVM:gfx1200',PROFILE='0',TC_OPT='2',CACHEDB=str(artifacts/'agent-residual-norm-queue-cache.db'))
  sys.path.insert(0,str(TINY))
  from tinygrad import Device,Tensor,TinyJit,dtypes
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support import hcq2
  retry=load(OLD/'agent-residual-norm-retry.py')
  binding,samples,plan,old=retry.verify_capture()
  bundle=Path(binding['source_bundle']);sys.path[:0]=[str(bundle),str(bundle/'experiments')]
  device=Device[Device.DEFAULT];assert device.device=='NULL' and device.renderer.target.arch=='gfx1200'
  # NULL has no physical arch. Use its already selected offline compiler target for the original installer.
  device.arch=device.renderer.target.arch
  load(bundle/'experiments/wgp_compiler.py').install(device)
  kernel=load(bundle/'agents/agent-residual-norm-kernel.py')
  norm=load(bundle/'experiments/fused-norm.py');quant=load(bundle/'experiments/quantize-packed.py')
  compiled={}
  for label,function,args in (('candidate',kernel.compiled,()),('frozen_norm',norm.compiled,(2048,True,1e-5))):
    source,binary=function(*args)
    compiled[label+'_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    compiled[label+'_binary_sha256']=hashlib.sha256(binary).hexdigest()
  assert compiled==binding['compiled']
  def baseline(a,b,w,s):
    residual=(a+b).realize()
    return residual,quant.quantize(norm.norm(residual,w,rms=True,eps=1e-5),s)[0]
  def batch(function):
    def run(a,b,w,s):
      outputs=[]
      for _ in range(16):a,p=function(a,b,w,s);outputs.append(p)
      return (a,*outputs)
    return TinyJit(run)
  inputs=(Tensor.empty(1,2048,2048,dtype=dtypes.bfloat16),Tensor.empty(1,2048,2048,dtype=dtypes.bfloat16),
          Tensor.empty(2048,dtype=dtypes.bfloat16),Tensor.empty((),dtype=dtypes.float32))
  for value in inputs:value.realize();value.uop.buffer.ensure_allocated()
  assert len(_prepare_jit_inputs(inputs,{})[0])==4
  original=hcq2._build_queues;records=[];contexts=[];active=[None]
  def build(ctx):
    queues=original(ctx)
    if active[0] is not None and len(ctx.batch)>1:
      label=active[0]
      roots=[]
      def describe(arg):
        base,lane,offset=hcq2.unwrap_lane(arg)
        if base not in roots:roots.append(base)
        return dict(op=arg.op.name,base_op=base.op.name,root=roots.index(base),lane=lane,offset=offset,
                    nbytes=arg.max_numel()*arg.dtype.itemsize,dtype=str(arg.dtype),
                    old_param_guard=base.op.name=='PARAM' and lane is None)
      calls=[dict(index=i,name=c.body.src[0].arg.name,outs=list(get_call_outs_ins(c)[0]),ins=list(get_call_outs_ins(c)[1]),
                  args=[describe(arg) for arg in c.src[1:] if not arg.is_variable]) for i,(c,_,_) in enumerate(ctx.batch)]
      row=dict(label=label,kernel_calls=len(calls),calls=calls)
      if auditor is not None:row['proof']=auditor.audit(ctx,queues,label,inputs)
      records.append(row);contexts.append((ctx,queues,label,inputs))
    return queues
  hcq2._build_queues=build
  try:
    for label,function in (('baseline',baseline),('candidate',kernel.fused)):
      jit=batch(function);jit(*inputs);active[0]=label
      if auditor is None:jit(*inputs)
      else:
        observed=[]
        with auditor.observe(observed,label,inputs):jit(*inputs)
        assert len(observed)==1
      active[0]=None
      assert len(jit.captured.ret)==17
  finally:hcq2._build_queues=original
  assert [r['kernel_calls'] for r in records]==[48,16]
  assert not any(name.startswith(('AMD','USB')) for name in Device._opened_devices)
  result=dict(schema=1,gpu_executed=False,numeric_execution=False,actual_tinyjit_capture=True,
              actual_hcq_queue_builder=True,strict_observe_hook_tested=auditor is not None,
              device=device.device,opened_devices=sorted(Device._opened_devices),records=records,compiled=compiled,
              original_npz_sha256=binding['failed_evidence_sha256']['report.agent-residual-norm-inputs.npz'])
  return result,contexts


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifacts',type=Path,default=HERE/'agent-residual-norm-queue-cpu')
  parser.add_argument('--audit',action='store_true')
  args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
  result,_=capture(args.artifacts,load(HERE/'agent-residual-norm-queue-proof.py') if args.audit else None)
  path=args.artifacts/'agent-residual-norm-queue-capture.json'
  path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps(dict(output=str(path),gpu_executed=False,numeric_execution=False,
    graphs=[dict(label=r['label'],kernel_calls=r['kernel_calls'],first_call=r['calls'][0]) for r in result['records']]),indent=2))


if __name__=='__main__':main()
