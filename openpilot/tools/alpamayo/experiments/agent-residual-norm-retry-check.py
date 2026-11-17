"""CPU execution of real TinyJit input, mutation, replay and poison checks."""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

HERE=Path(__file__).resolve().parent


def load(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_cpu',path)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifacts',type=Path,default=HERE/'agent-residual-norm-retry-cpu')
  args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
  os.environ.update(DEV='CPU:LLVM',PROFILE='0',CACHEDB=str(args.artifacts/'agent-residual-norm-retry-cache.db'))
  sys.path[:0]=['/tmp/opencode/alpamayo-asius-20261114/tinygrad']
  import numpy as np
  from tinygrad import Tensor,TinyJit,Device,dtypes
  from tinygrad.engine.jit import _prepare_jit_inputs,JitError
  retry=load(HERE/'agent-residual-norm-retry.py')
  buffers=load(HERE/'agent-residual-norm-retry-buffers.py')
  binding,samples,plan,old=retry.verify_capture()
  records=[];_,adaptation=retry.adapted_micro(old,buffers,binding,records)
  results=[]
  def picked(value):
    flat=np.asarray(value).reshape(-1)
    return np.concatenate((flat[:8],flat[-8:]))
  def expected(sample):
    return tuple(np.add(picked(sample[key]),np.uint16(delta),dtype=np.uint16) for key,delta in
                 (('a',1),('b',3),('weight',5)))+(np.asarray([np.float32(sample['scale']*np.float32(2)+np.float32(1))],np.float32),)
  def probe(a,b,w,s):
    def pick(value,delta):
      flat=value.bitcast(dtypes.uint16).flatten()
      return (flat[:8].cat(flat[-8:])+delta).realize()
    return pick(a,1),pick(b,3),pick(w,5),(s*2+1).reshape(1).realize()
  tree=ast.parse(textwrap.dedent(inspect.getsource(old.gpu_micro)))
  mutate_node=copy.deepcopy(next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='mutate'))
  mutate_ns=dict(dev=Device['CPU:LLVM'],tensors=lambda sample:buffers.materialize(sample,'CPU:LLVM'))
  exec(compile(ast.Module(body=[mutate_node],type_ignores=[]),'<actual frozen mutate on CPU>','exec'),mutate_ns)
  for prefix,sample in samples.items():
    original=[];before=[]
    for key in ('a','b','weight','scale'):
      value=Tensor(sample[key],device='CPU:LLVM')
      if key!='scale':value=value.bitcast(dtypes.bfloat16)
      value.realize();original.append(value);before.append(dict(argument=key,**buffers.physical_info(value)))
    virtual=[row['argument'] for row in before if row['virtual']]
    assert virtual==['scale']
    invoked=[]
    old_jit=TinyJit(lambda *values:invoked.append(True))
    try:old_jit(*original)
    except JitError as error:assert str(error)=='JIT inputs must be real buffers; use .clone()'
    else:raise AssertionError('Original constructor must reproduce the failure')
    assert not invoked
    inputs=buffers.materialize(sample,'CPU:LLVM',records)
    roots=[value.uop.buffer.base for value in inputs]
    assert len(set(roots))==4 and len(_prepare_jit_inputs(inputs,{})[0])==4
    jit=TinyJit(probe)
    for phase in ('first','capture','replay'):
      outputs=jit(*inputs)
      for actual,wanted in zip(outputs,expected(sample)):np.testing.assert_array_equal(actual.numpy(),wanted)
    cases=[]
    for key in ('a','b','weight','scale','restore'):
      changed={name:sample[name].copy() for name in ('a','b','weight','scale')}
      if key=='scale':changed[key]=np.asarray(changed[key]*np.float32(2),np.float32)
      elif key!='restore':
        changed[key].reshape(-1)[:8]=0x3f80
        changed[key].reshape(-1)[-8:]=0xbf80
      mutate_ns['mutate'](inputs,changed)
      assert [value.uop.buffer.base for value in inputs]==roots
      assert len(_prepare_jit_inputs(inputs,{})[0])==4
      outputs=jit(*inputs)
      for actual,wanted in zip(outputs,expected(changed)):np.testing.assert_array_equal(actual.numpy(),wanted)
      for value,name in zip(inputs,('a','b','weight','scale')):
        actual=value.bitcast(dtypes.uint16 if name!='scale' else dtypes.uint32).numpy()
        np.testing.assert_array_equal(actual,changed[name] if name!='scale' else changed[name].view(np.uint32))
      cases.append(key)
    results.append(dict(prefix=prefix,original_virtual_arguments=virtual,before=before,
                        four_distinct_physical_buffers=True,first_capture_replay=True,
                        original_mutate_function_used=True,fresh_mutations=cases,stable_buffer_roots=True))
  # Actual CPU TinyJit with both output storage types used by the poison wrapper.
  small=dict(a=np.full((1,16),0x3f80,np.uint16),b=np.full((1,16),0x3f00,np.uint16),
             weight=np.full((16,),0x3f80,np.uint16),scale=np.asarray(.25,np.float32))
  inputs=buffers.materialize(small,'CPU:LLVM')
  def operation(a,b,w,s):
    r=(a+b).realize()
    return r,(r.float()*s*8).cast(dtypes.uint8).realize()
  state=dict(poisoned_replays=0,verified_poison_bytes=0,unpoisoned_timing_calls=0)
  wrapped=buffers.PoisonedJit(TinyJit(operation),'real-cpu-poison',state,2)
  try:wrapped.run_without_poison(*inputs)
  except AssertionError:pass
  else:raise AssertionError('Unqualified timing must be rejected')
  for _ in range(3):outputs=wrapped(*inputs)
  wanted=tuple(value.bitcast(dtypes.uint16 if value.dtype==dtypes.bfloat16 else dtypes.uint8).numpy().copy() for value in outputs)
  assert np.all(wanted[0]==0x3fc0) and np.all(wanted[1]==3)
  assert state['poisoned_replays']>=1
  writes=state['verified_poison_bytes'];wrapped.run_without_poison(*inputs)
  assert state['verified_poison_bytes']==writes and state['unpoisoned_timing_calls']==1
  original_captured=wrapped.inner.captured
  class Stale:
    ret=original_captured.ret
    def __getattr__(self,key):return getattr(original_captured,key)
    def __call__(self,*args,**kwargs):return self.ret
  try:
    wrapped.inner.captured=Stale()
    stale=wrapped(*inputs)
    got=tuple(value.bitcast(dtypes.uint16 if value.dtype==dtypes.bfloat16 else dtypes.uint8).numpy().copy() for value in stale)
    assert not np.array_equal(got[0],wanted[0]) and not np.array_equal(got[1],wanted[1])
    assert np.all(got[0]==0x7fc1) and np.all(got[1]==0x7f)
  finally:wrapped.inner.captured=original_captured
  # A genuine replay must overwrite every poisoned residual bit and byte again.
  outputs=wrapped(*inputs)
  for value,bits in zip(outputs,wanted):
    np.testing.assert_array_equal(value.bitcast(dtypes.uint16 if value.dtype==dtypes.bfloat16 else dtypes.uint8).numpy(),bits)
  assert not any('AMD' in name or 'USB' in name for name in Device._opened_devices)
  result=dict(gpu_executed=False,bound_capture_arrays=21,layers=results,ast_adaptation=adaptation,
              poison=dict(real_cpu_jit=True,verified_output_poison=True,stale_execution_detected=True,
                          timing_rejects_unqualified_jit=True,timed_call_excludes_poison=True,stats=state),
              opened_devices=sorted(Device._opened_devices),numeric_kernel_unchanged=True)
  retry.write(args.artifacts/'agent-residual-norm-retry-check.json',result)
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()
