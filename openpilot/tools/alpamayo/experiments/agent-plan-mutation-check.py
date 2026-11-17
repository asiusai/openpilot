"""CPU numeric mask and real Tensor/JIT execution of frozen mutation setup code."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116');EXP=Path(__file__).resolve().parent

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  if args.output.exists() or not args.output.resolve().is_relative_to(WORK): raise ValueError('Use a new WORK output')
  for key in ('TMPDIR','CACHEDB'):
    if not Path(os.environ.get(key,'')).resolve().is_relative_to(WORK): raise ValueError('All caches and artifacts must stay on WORK')
  import numpy as np
  from tinygrad import Tensor,TinyJit,Device,dtypes
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.helpers import Context
  adapter=local(EXP/'agent-plan-mutation-adapter.py');original_source,modified,record=adapter.source()
  original_getitem=type(Device).__getitem__
  def cpu_only(self,key):
    if key is None or key.split(':')[0] not in ('CPU','NPY','PYTHON'): raise AssertionError('Hardware device forbidden: '+str(key))
    return original_getitem(self,key)
  report=dict(gpu_executed=False,scope='Mutation/alias/poison API setup, not Alpamayo numerical validation',adapter=record)
  with Context(DEV='CPU:LLVM'),patch.object(type(Device),'__getitem__',cpu_only):
    try: Tensor.arange(2112,device='AMD')
    except TypeError as error:
      if "unexpected keyword argument 'device'" not in str(error): raise
      report['original_arange_error_reproduced']=str(error)
    else: raise AssertionError('Original API failure did not reproduce')
    array=((np.arange(2112,dtype=np.int32)%33)-16).astype(np.float32).reshape(1,1,2112)
    first=Tensor(array,device='CPU').cast(dtypes.bfloat16).realize()
    expression=ast.parse(adapter.NEW,mode='eval')
    values=eval(compile(expression,'<fixed-mask-call>','eval'),dict(Tensor=Tensor,first=first)).realize()
    np.testing.assert_array_equal(values.numpy(),np.arange(2112,dtype=np.int32))
    mask=(values==2111).reshape(1,1,2112).realize()
    np.testing.assert_array_equal(mask.numpy(),(np.arange(2112)==2111).reshape(1,1,2112))
    changed=mask.where(first.abs().max()+256,first).cast(dtypes.bfloat16).realize()
    reference=array.copy();reference[0,0,-1]=272
    np.testing.assert_array_equal(changed.bitcast(dtypes.uint16).numpy(),(reference.view(np.uint32)>>16).astype(np.uint16))
    report.update(fixed_arange_exact_values=2112,fixed_mask_true_indices=[2111],changed_outlier_exact=272,
      untouched_logit_bits_equal=2111)

    # Execute the original setup functions and six-case loop on real CPU Tensors
    # and real TinyJits. Only the active-arena device assertion is translated to
    # CPU for this fixture. The production adapter changes no assertion.
    definitions={n.name:n for n in original_source.body if isinstance(n,ast.FunctionDef)}
    block=next(n for n in modified.body if isinstance(n,ast.Try))
    nested={n.name:n for n in block.body if isinstance(n,ast.FunctionDef)}
    selected=[copy.deepcopy(definitions[n]) for n in ('exact','reset','clone','same','kv_equal','variables')]
    selected += [copy.deepcopy(nested[n]) for n in ('active_storage','poisoned_run')]
    active=selected[-2];translated=0
    for node in ast.walk(active):
      if isinstance(node,ast.Constant) and node.value=='AMD': node.value='CPU';translated+=1
    if translated!=1: raise AssertionError('Unexpected test device adaptation')
    start=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='mask')
    selected+=copy.deepcopy(block.body[start:])
    tail=compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'<frozen-mutation-setup-CPU>','exec')
    zeros=np.zeros((2,1,8,2304,128),np.uint16)
    cache=[Tensor(zeros.copy(),device='CPU').bitcast(dtypes.bfloat16).realize() for _ in range(28)]
    model=SimpleNamespace(cache=cache,synchronize=lambda:Device['CPU'].synchronize())
    snapshots=[t.clone().realize() for t in cache]
    identity=lambda:tuple(id(t.uop.base.buffer.base) for t in model.cache)
    dc=Tensor(np.zeros((1,1,64,128),np.uint16),device='CPU').bitcast(dtypes.bfloat16).realize()
    ds=dc.clone().realize();prompt=2027
    arenas=[];events=[]
    def make_jit():
      def run(x,c,s,p,step):
        # Deliberately small CPU probe: real input-dependent outputs exercise
        # TinyJit's actual output alias/staging behavior, not the model kernels.
        return (x+1).cast(dtypes.bfloat16).realize(),x[:,:,-1:].cast(dtypes.float32).reshape(1,1).reshape(1).realize()
      jit=TinyJit(run)
      def invoke(x,c,s,p,step):
        ret=jit(x,c,s,UOp.variable('block_position',0,2303).bind(p),UOp.variable('block_step',0,63).bind(step))
        if jit.captured:
          for alias,(linear,staging) in list(jit.captured._linears.items()):
            if any(u.op is Ops.BUFFER and u.buffer.nbytes==10499328 for u in linear.toposort()): continue
            arena=Buffer('CPU',10499328,preallocate=True);arenas.append(arena)
            ref=UOp.from_buffer(arena,dtypes.int8)
            jit.captured._linears[alias]=(linear.replace(src=(linear.src[0].after(ref),*linear.src[1:])),staging)
        return ret
      for _ in range(3): Tensor.realize(*invoke(first,dc,ds,prompt,0))
      return jit,invoke
    original,original_invoke=make_jit();candidate,candidate_invoke=make_jit()
    class Loaded:
      def __init__(self): self.jit,self.ret=candidate,candidate.captured.ret
      def __call__(self,*a): return candidate_invoke(*a)
    decoder=SimpleNamespace(decode=lambda x,c,s,p,st,count:original_invoke(x,c,s,p,st))
    namespace=dict(Tensor=Tensor,dtypes=dtypes,UOp=UOp,Ops=Ops,_prepare_jit_inputs=_prepare_jit_inputs,
      model=model,decoder=decoder,original=original,original_ret=original.captured.ret,loaded=Loaded(),
      first=first,logits=first,dc=dc,ds=ds,prompt=prompt,snapshots=snapshots,identity=identity,original_identity=identity(),
      result={'cases':[]},stage=lambda name:events.append(name))
    try: exec(tail,namespace)
    except AttributeError as error:
      if 'copyin' not in str(error): raise
      report['next_original_API_error_reproduced']=str(error)
    else: raise AssertionError('Expected frozen poison API to fail')

    # Test-only transport probe, not a production adapter or a skipped write.
    # It exercises every remaining setup branch using the actual pinned copy API.
    writes=[];original_copy_from=Buffer.copy_from
    def copyin_probe(buffer,value):
      if buffer.device!='CPU': raise AssertionError('CPU transport probe received another device')
      source=Buffer('CPU',len(value),initial_value=bytes(value))
      original_copy_from(buffer.ensure_allocated(),source)
      actual=bytes(buffer.host[:buffer.nbytes])
      if actual!=bytes(value): raise AssertionError('Poison transfer did not write every byte')
      writes.append(dict(nbytes=buffer.nbytes,pattern=value[0],sha256=hashlib.sha256(actual).hexdigest()))
      return buffer
    namespace['result']={'cases':[]}
    with patch.object(Buffer,'copyin',copyin_probe,create=True): exec(tail,namespace)
    if hasattr(Buffer,'copyin'): raise AssertionError('CPU-only compatibility probe leaked')
    if len(namespace['result']['cases'])!=6: raise AssertionError('All six CPU setup cases must execute')
    for current,saved in zip(model.cache,snapshots,strict=True): current.assign(saved).realize()
    for current,saved in zip(model.cache,snapshots,strict=True): namespace['exact'](current,saved)
    if identity()!=namespace['original_identity']: raise AssertionError('CPU setup changed owner allocation')
    report.update(cpu_setup_cases=namespace['result']['cases'],cpu_poison_writes=writes,cpu_stage_events=events,
      cpu_original_alias_variants=len(original.captured._linears),cpu_candidate_alias_variants=len(candidate.captured._linears),
      all28_cpu_KV_restored=True,test_only_transport='Buffer.copy_from with verified host bytes; production remains mask-only',
      production_retry_ready=False,blocker='Frozen poison call uses absent Buffer.copyin; additional correction awaits authorization')
  report['opened_devices']=sorted(Device._opened_devices)
  if any(d.split(':')[0] not in ('CPU','NPY','PYTHON') for d in Device._opened_devices): raise AssertionError('A hardware device opened')
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('cpu_setup_cases','cpu_poison_writes','adapter')},indent=2))

if __name__=='__main__': main()
