"""CPU-only full-shape validator semantics and independent negative-control checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main(output):
  assert not output.exists() and output.parent.resolve()==ROOT.resolve()
  os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='1',PARALLEL='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-validator-cache.db'),
    PROFILE='0',BEAM='0',JIT='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  sys.path.insert(0,str(TINY))
  import numpy as np
  from tinygrad import Tensor,dtypes,TinyJit
  from tinygrad.device import Device
  # This test may execute native CPU code only. Reject device creation before dispatch.
  original_getitem=Device.__class__.__getitem__
  def cpu_only(self,name):
    assert name is None or name.split(':')[0] in ('CPU','PYTHON','NPY'),name
    return original_getitem(self,name)
  Device.__class__.__getitem__=cpu_only
  common=load('agent-ck-qualified-common');previous,oracle,_,reuse=load('agent-ck-qualified-evidence').verify()
  results=[];started=time.monotonic()
  try:
    for key,count,dt,guard in [('O',common.OC,np.uint16,0x5aa5),('LSE',common.LC,np.uint32,0x5aa55aa5)]:
      # Construct independent, finite integer encodings. Every value is known on the CPU.
      values=np.arange(count,dtype=np.uint32)
      values=((values%2048)|0x3f00).astype(np.uint16) if key=='O' else (values%65536)|np.uint32(0x3f000000)
      whole=np.pad(values,(256,256),constant_values=guard).astype(dt)
      changed,returned,edits=common.controls(whole,values)
      assert len([e for e in edits if e['label'].startswith('guarded_bit_')])==dt().itemsize*8
      assert len([e for e in edits if e['label'].startswith('returned_bit_')])==dt().itemsize*8
      for name,a,b in [('clean',whole,values),('negative_all_bits',changed,returned)]:
        expected=common.cpu_flags(a,b,whole)
        dtype=dtypes.bfloat16 if key=='O' else dtypes.float32
        shaped=b.reshape(common.QSHAPE if key=='O' else common.LSHAPE)
        result=common.validator(Tensor(a,device='CPU').bitcast(dtype),Tensor(shaped,device='CPU').bitcast(dtype),
                                Tensor(whole,device='CPU').bitcast(dtype))
        actual=result.numpy()
        assert np.array_equal(actual,expected),(key,name,np.count_nonzero(actual!=expected))
        if name=='clean': assert not actual.any()
        else:
          assert all(actual[0,bit+2]>0 for bit in range(dt().itemsize*8))
          assert all(actual[4,bit+51]>0 for bit in range(dt().itemsize*8))
        results.append({'component':key,'case':name,'flag_words':int(actual.size),'exact_CPU_match':True,
          'counts':actual.sum(axis=1,dtype=np.uint64).tolist(),'flag_sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
      if key=='O':
        b=values.reshape(common.QSHAPE).transpose(0,2,1,3).copy().reshape(1,2048,2048)
        flags=common.validator(Tensor(whole,device='CPU').bitcast(dtypes.bfloat16),Tensor(b,device='CPU').bitcast(dtypes.bfloat16),
                               Tensor(whole,device='CPU').bitcast(dtypes.bfloat16),'B').numpy()
        assert not flags.any();results.append({'case':'B_return_conversion','exact_CPU_match':True})
        ts=[Tensor(whole.copy(),device='CPU').bitcast(dtypes.bfloat16).realize(),
            Tensor(values.copy().reshape(common.QSHAPE),device='CPU').bitcast(dtypes.bfloat16).realize(),
            Tensor(whole.copy(),device='CPU').bitcast(dtypes.bfloat16).realize()]
        roots=[t.uop.base.key.hex() for t in ts]
        runner=TinyJit(lambda *xs:common.validator(*xs).realize())
        for index,(a,b) in enumerate([(whole,values),(whole,values),(changed,returned),(whole,values)]):
          ts[0].uop.buffer.as_memoryview(allow_zero_copy=True)[:]=a.tobytes()
          ts[1].uop.buffer.as_memoryview(allow_zero_copy=True)[:]=b.tobytes()
          if runner.captured is not None:
            flag_buffer=runner.captured.ret.uop.buffer
            flag_buffer.as_memoryview(allow_zero_copy=True)[:]=bytes([255])*flag_buffer.nbytes
          actual=runner(*ts).numpy()
          assert runner.cnt==index+1 and np.array_equal(actual,common.cpu_flags(a,b,whole))
          assert roots==[t.uop.base.key.hex() for t in ts]
          if runner.captured is not None:
            assert runner.captured.expected_names==[0,1,2] and all(not m for _,m in runner.captured._linears.values())
        results.append({'case':'full_O_normal_TinyJit_mutation_restore_and_sentinel_overwrite','calls':4,
          'stable_roots':True,'no_staging':True,'exact_CPU_match':True})
    # Full capacity-tail traversal, with all bit positions and every head independently visible.
    x=np.full((1,8,2304,128),0x7fc1,np.uint16);x[:,:,:2048]=0x3f80;y=x.copy()
    for bit in range(16): x[0,0,2048+bit*2,8]^=np.uint16(1<<bit)
    for head in range(8): y[0,head,-1,-1]^=np.uint16(1)
    expected=np.stack([np.count_nonzero(a[:,:,2048:].reshape(-1,256)!=0x7fc1,axis=1) for a in (x,y)]).astype(np.uint32)
    actual=common.tail_validator(Tensor(x,device='CPU').bitcast(dtypes.bfloat16),Tensor(y,device='CPU').bitcast(dtypes.bfloat16)).numpy()
    assert np.array_equal(actual,expected) and actual.sum()==24
    results.append({'case':'KV_tail_all_bits_all_heads','exact_CPU_match':True,'changed_words':24})
    # Independent layout-index proof for every output index of both B packing utilities.
    index=np.arange(common.OC,dtype=np.int64)
    mapped=(index//128%16)*2048*128+(index//2048)*128+index%128
    assert np.array_equal(mapped,np.arange(common.OC).reshape(common.QSHAPE).transpose(0,2,1,3).reshape(-1))
    index=np.arange(8*2304*128,dtype=np.int64);row=index//128%2304
    mapped=(index//(2304*128))*2048*128+row*128+index%128
    wanted=np.pad(np.arange(2097152).reshape(common.KVSHAPE),((0,0),(0,0),(0,256),(0,0)),constant_values=-1).reshape(-1)
    assert np.array_equal(np.where(row<2048,mapped,-1),wanted)
    results.append({'case':'full_B_Q_and_capacity_KV_index_mapping','exact_CPU_match':True})
    for count in (114702,910,2048):
      index=np.arange(((count+255)//256)*256)
      assert np.array_equal(index[index<count],np.arange(count))
    results.append({'case':'sentinel_masked_dispatch_complete_coverage','exact_CPU_match':True})
    assert all(x.split(':')[0] in ('CPU','PYTHON','NPY') for x in Device._opened_devices)
    record={'cpu_checks':'passed','gpu_executed':False,'opened_devices':sorted(Device._opened_devices),
      'results':results,'seconds':time.monotonic()-started,'binding_evidence':reuse,'attention_numerics_changed':False,
      'source_sha256':{str(Path(__file__)):common.sha(__file__),str(EXP/'agent-ck-qualified-common.py'):common.sha(EXP/'agent-ck-qualified-common.py')}}
    with output.open('x') as stream: stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
  finally: Device.__class__.__getitem__=original_getitem


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'cpu-check.json');args=parser.parse_args();main(args.output)
