"""CPU expression controls and actual native six-token fallback/capture lifecycle."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from types import MethodType

os.environ.update(DEV='CPU',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
                  CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/opencode/alpamayo-asius-20261114');WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
sys.path.insert(0,str(BASE/'tinygrad'))
import numpy as np
from tinygrad import Tensor,TinyJit,Device,dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp
P=runpy.run_path(str(HERE/'agent-distill-objective-position-run.py'))
C=P['load']();resolve=P['resolved_position']


def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
    path=os.fsdecode(args[0])
    if path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):raise RuntimeError('CPU-only position check')
sys.addaudithook(guard)


def controls():
  pos=UOp.variable('block_position',0,2298).bind(2027)
  assert [resolve(pos+i) for i in range(6)]==list(range(2027,2033))
  offset=UOp.variable('block_offset',0,64).bind(4)
  assert resolve((pos+offset*2)-3)==2032 and resolve(UOp.const(17))==17 and resolve(0)==0
  rejects={
    'unbound':UOp.variable('unknown_position',0,2298)+1,
    'mixed bound and unbound':pos+UOp.variable('unknown_position',0,10),
    'cancelled unbound':UOp.variable('unknown_position',0,10)*0,
    'conflicting same variable bindings':pos+UOp.variable('block_position',0,2298).bind(2000),
    'ambiguous same name':pos+UOp.variable('block_position',0,10).bind(2),
    'negative':UOp.variable('negative_position',-10,10).bind(-1),
    'outside capacity':pos+300,
    'float expression':UOp.const(1.5),
    'python float':1.0,'python boolean':True}
  for name,value in rejects.items():
    try:resolve(value)
    except ValueError:pass
    else:raise AssertionError('Accepted '+name)
  return dict(actual_pos_plus_i=list(range(2027,2033)),mixed_bound_arithmetic=True,rejected=list(rejects),bound_extrema_never_used=True)


def lifecycle(root):
  runtime=Path(C.read(C.PLAN)['runtime_root'])
  # Import the established upstream adapters before native kernels can shadow optional packages.
  sys.path.insert(0,str(BASE/'upstream/src'))
  from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeModel
  sys.path[:0]=[str(runtime),str(runtime/'experiments')]
  Native=C.module('position_native',runtime/'model.py').EdgeModel
  Decode=C.module('position_native_decoder',runtime/'experiments/block-decode.py').BlockDecoder
  model=Native.__new__(Native);model.config=C.read(BASE/'model/config.json');model.dtype=dtypes.bfloat16
  model.fp8_wmma=True;model.cuda_kernels=False;model.trajectory_jits={}
  model.cache=[Tensor.zeros(2,1,8,2304,128,dtype=model.dtype).realize() for _ in range(28)]
  model.norm=lambda x,*a,**kw:x
  model.linear=lambda x,name,*a,**kw:x[:,:,:2]*.01 if name=='expert.action_out_proj' else x
  model.action_projection=lambda a,t:(a+t.reshape(1,1,1)).pad(((0,0),(0,0),(0,510))).cast(model.dtype)
  positions=[]
  def layer(self,x,cos,sin,prefix,cache=None,mask=None,cache_pos=None):
    if prefix.startswith('vlm.'):
      start=resolve(cache_pos)
      if prefix.endswith('.0'):positions.append(start)
      cache[:,:,:,start:start+x.shape[1]].assign(Tensor.full((2,1,8,x.shape[1],128),float(x[0,0,0].float().item()),dtype=model.dtype)).realize()
      return x
    return (x.float()+.001).cast(model.dtype).realize()
  model.text_layer=MethodType(layer,model)
  symbolic=[]
  def greedy(self,logits,cos,sin,position,step):
    # Neural head is mocked; the actual BlockDecoder constructs position+i and offset+i.
    offset=resolve(step);symbolic.append(position)
    value=.5 if offset<6 else .75 if offset<22 else 1.
    x=Tensor.full((1,1,2048),value,dtype=model.dtype)
    result=self._language(x,cos[:,:,step:step+1],sin[:,:,step:step+1],position)
    token=self.config['traj_ids']['future_start'] if offset==21 else offset
    return result,Tensor([token],dtype=dtypes.float32)
  model._greedy_decode=MethodType(greedy,model)
  decoder=Decode(model,6);root.mkdir();saved={}
  def save(name,array,bf16):
    path=root/(name+'.npy');np.save(path,array,allow_pickle=False)
    item=dict(path=str(path),sha256=C.sha(path),shape=list(array.shape),dtype=str(array.dtype),bf16_bits=bf16)
    saved[name]=item;return item
  original=model.text_layer;tokens=[]
  with Context(JIT=0),C.aligned_targets(model,save) as target:
    logits=model._language(Tensor.full((1,2048,2048),.25,dtype=model.dtype),Tensor.ones(1,1,2048,128,dtype=model.dtype),
                           Tensor.zeros(1,1,2048,128,dtype=model.dtype),0)
    dc=Tensor.ones(1,1,64,128,dtype=model.dtype);ds=Tensor.zeros(1,1,64,128,dtype=model.dtype)
    for start in range(0,24,6):
      logits,block=decoder.decode(logits,dc,ds,2027+start,start,6);tokens.extend(block.numpy().astype(int).tolist())
    model._trajectory(Tensor.zeros(1,64,2),Tensor.ones(1,1,128,128,dtype=model.dtype),
                      Tensor.zeros(1,1,128,128,dtype=model.dtype),2049,3,22).realize()
    prediction=dict(prompt_tokens=2027,prefill_tokens=2048,generated_tokens=22,generated_token_ids=tokens[:22])
    target['finish'](prediction,dict(actual_prompt_length=2027),(np.zeros((1,2048),np.int32),))
  assert positions==[0]+list(range(2027,2051)) and len(symbolic)==24
  assert model.text_layer==original and target['complete']
  assert target['valid_rows_last_write'][2027:]==list(range(1,23))
  record=dict(synthetic_fixture=True,split='train',route_id=C.ROUTES['train'],targets=target)
  data=C.load_record(record,fixture=True)
  x=data['language']['x'].numpy()
  assert np.all(x[:,2027:2033]==.5) and np.all(x[:,2033:2049]==.75) and np.all(x[:,2049:2051]==1.)
  for i in range(1,3):np.testing.assert_allclose(data['expert'][i]['actions'].numpy(),data['expert'][i-1]['actions'].numpy()+data['expert'][i-1]['velocity'].numpy()/3,atol=2e-6)
  return dict(actual_native_block_decoder=True,actual_native_language_and_euler=True,neural_head_mocked=True,
    native_fallback_calls=24,accepted_reasoning_rows=22,valid_context_length=2049,
    all_layer_orders_preserved=True,last_writer_rows_correct=True,final_kv_and_euler_checks_passed=True,hook_restored=True)


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(WORK/'agent-distill-objective-position-artifacts'):raise ValueError('Fresh owned output required')
  root.mkdir(parents=True)
  result=dict(gpu_executed=False,optimizer_updates=0,controls=controls(),lifecycle=lifecycle(root/'capture'))
  assert all(name.startswith(('CPU','NPY','PYTHON')) for name in Device._opened_devices),Device._opened_devices
  result.update(all_passed=True,opened_devices=list(Device._opened_devices),original_source_sha256=P['ORIGINAL_SHA'],
    source_sha256={str(p):C.sha(p) for p in HERE.glob('agent-distill-objective-position-*.py')})
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
