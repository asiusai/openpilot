"""CPU controls for scoped hooks, physical ranges, source mutations, and production runpy."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_model_common',EXP/'agent-ck-model-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def main(output,protocol):
  os.environ.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='0',PYTHONDONTWRITEBYTECODE='1',CACHEDB=str(C.ROOT/'cache.db'),TMPDIR=str(C.ROOT/'tmp'))
  sys.path[:0]=[str(C.TINY),str(C.BASE),str(C.BASE/'experiments')]
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp
  class PreOpen(Exception): pass
  def forbidden(self,name): raise PreOpen(name)
  assert not Device._opened_devices;Device.__class__.__getitem__=forbidden
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/bus/usb','/dev/dri','/dev/kfd'))
  sys.addaudithook(audit)
  hook=C.load('agent-ck-model-hook');checks=[]
  def tensor(shape,dt=dtypes.bfloat16): return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(shape)*dt.itemsize),dt)).reshape(shape)
  q,k,v=[tensor(shape) for shape in (hook.QSHAPE,hook.KVSHAPE,hook.KVSHAPE)]
  class Fake:
    def __init__(self):
      self.config={'vlm_config':{'text_config':{'num_hidden_layers':28}}}
      self.cache=[tensor((2,1,8,2304,128)) for _ in range(28)];self.args=(q,k,v);self.fallbacks=0
    def attention(self,q,k,v,mask=None,causal=False): self.fallbacks+=1;return 'fallback'
    def text_layer(self,x,cos,sin,prefix,cache=None,mask=None,cache_pos=None):
      return self.attention(*self.args,mask,cache_pos==0)
  m=Fake();seen=[]
  def callback(h,c,q,k,v,original): seen.append(c['layer']);return 'selected'
  h=hook.Hook(m,callback);x=tensor((1,2048,2048));cs=tensor((1,1,2048,128))
  for index in range(28): assert m.text_layer(x,cs,cs,f'vlm.model.language_model.layers.{index}',m.cache[index],cache_pos=0)=='selected'
  assert seen==list(range(28));checks.append('all_28_contexts_delegate_and_select_in_order_CPU_only')
  for args,mask,causal in [((tensor((1,16,1,128)),k,v),None,True),((q,k,v),None,False),((q,k,v),object(),True),
                           ((tensor(hook.QSHAPE,dtypes.float32),k,v),None,True)]:
    context={'cache_pos':0,'cache':m.cache[0]}
    assert hook.eligible(context,q,k,v,None,True)
    assert not hook.eligible(context,*args,mask,causal)
    before=len(seen);assert m.attention(*args,mask,causal)=='fallback' and len(seen)==before
  checks.append('outside_scope_wrong_shape_dtype_mask_causal_fallback')
  before=len(seen);assert m.text_layer(x,cs,cs,'expert.layers.0',m.cache[0],cache_pos=0)=='fallback' and len(seen)==before
  m.args=(tensor((1,16,2048,72)),k,v)
  assert m.text_layer(x,cs,cs,'vlm.model.language_model.layers.0',m.cache[0],cache_pos=0)=='fallback'
  m.args=(q,k,v)
  try: m.text_layer(x,cs,cs,'vlm.model.language_model.layers.0',m.cache[1],cache_pos=0)
  except AssertionError: checks.append('wrong_cache_owner_rejected')
  else: raise AssertionError('Wrong owner accepted')
  h.close();assert not any(n in vars(m) for n in ('attention','text_layer','_agent_ck_model_hook'))
  checks.append('inherited_methods_restored')
  def fail(*args): raise RuntimeError('deliberate callback control')
  h=hook.Hook(m,fail)
  try: m.text_layer(x,cs,cs,'vlm.model.language_model.layers.0',m.cache[0],cache_pos=0)
  except RuntimeError: pass
  else: raise AssertionError('Missing callback failure')
  assert h.context is None;h.close();checks.append('exception_restores_context_and_ownership')
  rows=[dict(bytes=n,offset=512,base_bytes=n+1024,gpu_address=a+512,base_gpu_address=a)
    for n,a in [(8388608,0x10000000),(4194304,0x20000000),(4194304,0x30000000)]]
  assert C.storage_gate(rows,[8388608,4194304,4194304])
  for mode in ('alias','pointer','extent','alignment','size'):
    bad=copy.deepcopy(rows)
    if mode=='alias': bad[1].update(gpu_address=bad[0]['gpu_address'],base_gpu_address=bad[0]['base_gpu_address'])
    elif mode=='pointer': bad[0]['gpu_address']+=16
    elif mode=='extent': bad[0]['base_bytes']=bad[0]['bytes']
    elif mode=='alignment': bad[0]['base_gpu_address']+=1;bad[0]['gpu_address']+=1
    else: bad[0]['bytes']-=16
    try: C.storage_gate(bad,[8388608,4194304,4194304])
    except AssertionError: checks.append('storage_'+mode+'_rejected')
    else: raise AssertionError(mode)
  # Full program identity mutations must fail exact authorization, including source/AST sharing one ELF.
  identity=C.load('agent-ck-source-identity');catalog=json.loads((C.PREPARED/'catalog.json').read_text())
  batches=[b for b in catalog['batches'].values() if catalog['associations'][b[0]['association']]['name']=='alpamayo_ck_d128']
  assert len(batches)==1 and len(batches[0])==3
  native,ocopy,lcopy=batches[0]
  assert native['outs']==[0,1] and native['ins']==[2,3,4]
  assert native['arguments'][0]['offset_bytes']==512 and native['arguments'][1]['offset_bytes']==1024
  assert ocopy['dependencies']==lcopy['dependencies']==[0]
  assert ocopy['arguments'][1]==native['arguments'][0] and lcopy['arguments'][1]==native['arguments'][1]
  checks.append('O_LSE_native_stores_and_both_returned_copy_dependencies')
  item=next(iter(catalog['associations'].values()))
  for field in ('source_sha256','ast_key','linear_key','binary_sha256','target','descriptor','outs','ins','params'):
    mutated=copy.deepcopy(item);mutated[field]=['deliberate_invalid_association',field]
    from unittest.mock import patch
    with patch.object(identity,'identity',return_value=mutated):
      try: identity.authorize(object(),catalog)
      except AssertionError: checks.append('identity_'+field+'_mutation_rejected')
      else: raise AssertionError('Changed program authorized')
  # Run the exact canonical launcher and pipeline to the first real Device lookup.
  plan=json.loads(protocol.read_text());invocation=json.loads((C.PREPARED/'invocation.json').read_text())
  preopen=output.with_suffix('.runpy');preopen.mkdir(exist_ok=False)
  command=plan['launcher_command'][:];command[command.index('--output')+1]=str(preopen/'unused.json')
  old_argv=sys.argv;old_spec=importlib.util.spec_from_file_location;old_environment=dict(os.environ)
  try:
    sys.argv=command
    try: runpy.run_path(command[0],run_name='__main__')
    except PreOpen as error: checks.append('actual_production_runpy_preopen_blocked_'+str(error))
    else: raise AssertionError('Production launcher did not reach blocked device boundary')
  finally:
    sys.argv=old_argv;importlib.util.spec_from_file_location=old_spec
    os.environ.clear();os.environ.update(old_environment)
  assert not Device._opened_devices
  C.write_json(output,{'gpu_executed':False,'opened_devices':[],'checks':checks,'count':len(checks),
    'runpy_target':command[0],'pipeline':invocation['command'][1],
    'limitation':'Control-flow, metadata, and source rejection tests. No arithmetic or live-layout attestation.'})
  print(json.dumps({'cpu_checks':len(checks),'gpu_executed':False,'output':str(output)}))

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--protocol',type=Path,default=C.PROTOCOL);args=parser.parse_args();main(args.output,args.protocol)
