"""Exercise the real scoped compiler's physical-pointer branch with anonymous host allocations."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_model_common',EXP/'agent-ck-model-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def main():
  os.environ.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='0',CACHEDB=str(C.ROOT/'cache.db'),TMPDIR=str(C.ROOT/'tmp'),
    TC_OPT='2',LLVMOPT='1',BEAM='0',JITBEAM='0',PROFILE='0')
  sys.path.insert(0,str(C.TINY))
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Buffer,HostAllocator
  from tinygrad.uop.ops import UOp
  from tinygrad.engine import realize
  dev=SimpleNamespace(device='AMD',host='CPU',synchronize=lambda:None)
  dev.allocator=HostAllocator(dev)
  def getitem(self,name): assert name=='AMD';return dev
  Device.__class__.__getitem__=getitem
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/bus/usb','/dev/dri','/dev/kfd'))
  sys.addaudithook(audit)
  def tensor(shape,dt=dtypes.bfloat16):
    return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(shape)*dt.itemsize).ensure_allocated(),dt)).reshape(shape)
  helper=C.load('agent-ck-output-common');adapter=helper.load('agent-ck-attention-v2-adapter');guard=C.load('agent-ck-model-guard')
  catalog=json.loads((C.PREPARED/'catalog.json').read_text());checks=[]
  for incorrect in (False,True):
    xs=[tensor(shape) for shape in (helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE)]
    stores=[tensor((helper.OC+512,)),tensor((helper.LC+512,),dtypes.float32)]
    pointers=[int(stores[0].uop.buffer.get_storage().buf)+512,int(stores[1].uop.buffer.get_storage().buf)+1024,
      *[int(t.uop.buffer.get_storage().buf) for t in xs]]
    if incorrect: pointers[2]+=16
    # The internal flag selects the production branch. These addresses are host-only test allocations.
    simulated={'gpu_executed':True,'phase':'anonymous_host_pointer_simulation','expected_candidate_pointers':pointers}
    with adapter.abi_scope():
      result=helper.outputs('candidate',True,*xs,*stores);linear,_=result[0].linear_with_vars(*result[1:])
      with guard.guard(C,catalog,simulated):
        try: realize.lower_and_compile(linear)
        except AssertionError:
          assert incorrect;checks.append('physical_Q_pointer_mutation_rejected')
        else:
          assert not incorrect and len(simulated['actual_candidate_pointer_bindings'])==1
          assert [r['offset_bytes'] for r in simulated['actual_candidate_pointer_bindings'][0]][:2]==[512,1024]
          checks.append('actual_Buffer_view_physical_offsets_and_native_argument_order')
  assert not Device._opened_devices
  output=C.ROOT/'cpu-physical-pointer-proof.json'
  C.write_json(output,{'gpu_executed':False,'opened_devices':[],'arithmetic_executed':False,'checks':checks,
    'source_sha256':{str(p):C.sha(p) for p in [Path(__file__),EXP/'agent-ck-model-guard.py',C.PROTOCOL]},
    'storage':'Anonymous host allocations behind pinned Buffer and HostAllocator APIs; no GPU memory or USB.'})
  print(json.dumps({'gpu_executed':False,'checks':checks,'proof':str(output)}))

if __name__=='__main__': main()
