"""CPU policy, original lookup, actual install/run path, and past-history controls."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE=WORK/'agent-fused-mlp-2048-protocol-v2/alpamayo'


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Array:
  """Neural-only NumPy double. Stores, indexing, BF16 bits, and masks are real."""
  def __init__(self,value,device='CPU',dtype=None):
    from tinygrad import dtypes
    self.device=device
    self.data=np.asarray(value.data if isinstance(value,Array) else value)
    self.dtype=dtype or {np.dtype('int32'):dtypes.int32,np.dtype('uint16'):dtypes.uint16,
      np.dtype('bool'):dtypes.bool}.get(self.data.dtype,dtypes.float32)
    if dtype is not None:self.data=self.cast(dtype).data
  @property
  def shape(self):return self.data.shape
  def realize(self):return self
  def to(self,device):return self
  def numpy(self):return self.data.copy()
  def reshape(self,*shape):return Array(self.data.reshape(*shape),dtype=self.dtype)
  def __getitem__(self,key):
    if isinstance(key,Array):key=key.data
    return Array(self.data[key],dtype=self.dtype)
  def __ne__(self,other):return Array(self.data!=other.data)
  def sum(self):return Array(self.data.sum())
  def all(self):return Array(self.data.all())
  def argmax(self,axis=None):return Array(self.data.argmax(axis=axis).astype(np.int32))
  def isfinite(self):return Array(np.isfinite(self.data))
  def cast(self,dtype):
    from tinygrad import dtypes
    a=self.data
    if dtype==dtypes.bfloat16:
      a=a.astype(np.float32);u=a.view(np.uint32);a=((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
    else:a=a.astype({dtypes.int32:np.int32,dtypes.uint16:np.uint16,dtypes.float32:np.float32,dtypes.bool:np.bool_}[dtype])
    result=object.__new__(Array);result.data=a;result.dtype=dtype;result.device=self.device;return result
  def bitcast(self,dtype):
    from tinygrad import dtypes
    if self.dtype==dtypes.bfloat16 and dtype==dtypes.uint16:return Array((self.data.view(np.uint32)>>16).astype(np.uint16),dtype=dtype)
    if self.dtype==dtypes.uint16 and dtype==dtypes.bfloat16:
      result=object.__new__(Array);result.data=(self.data.astype(np.uint32)<<16).view(np.float32);result.dtype=dtype;result.device=self.device;return result
    raise AssertionError((self.dtype,dtype))
  def contiguous(self):return self
  def clone(self):return Array(self.data.copy(),dtype=self.dtype)
  def expand(self,*shape):return Array(np.broadcast_to(self.data,shape),dtype=self.dtype)
  def cat(self,*others,dim=0):return Array(np.concatenate([a.data for a in (self,*others)],axis=dim),dtype=self.dtype)
  @classmethod
  def stack(cls,*values):return cls(np.stack([a.data for a in values]),dtype=values[0].dtype)
  @classmethod
  def zeros(cls,*shape,**kwargs):return cls(np.zeros(shape,dtype=np.float32),**kwargs)
  @classmethod
  def ones(cls,*shape,**kwargs):return cls(np.ones(shape,dtype=np.float32),**kwargs)


class DirectJit:
  def __init__(self,fn):self.fn=fn;self.cnt=0;self.captured=None
  def reset(self):self.cnt=0;self.captured=None
  def __call__(self,*args):
    from tinygrad.uop.ops import UOp
    self.cnt+=1
    return self.fn(*(a.unbind()[1] if isinstance(a,UOp) else a for a in args))


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  assert not args.output.exists()
  sys.path[:0]=[str(SOURCE/'experiments'),str(SOURCE)]
  H=load(HERE/'agent-adaptive-prefill-hook.py','adaptive_cpu_hook');P=H.P
  from tinygrad import Device,dtypes
  assert not Device._opened_devices
  original_device_get=type(Device).__getitem__
  rejected=[]
  def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError,AssertionError,KeyError,AttributeError):rejected.append(name)
    else:raise AssertionError('Expected rejection: '+name)
  def blocked(*args,**kwargs):raise AssertionError('Device/network access forbidden in CPU proof')
  cases=[]
  with patch.object(type(Device),'__getitem__',blocked),patch.object(socket.socket,'connect',blocked):
    lookup=load(SOURCE/'experiments/lookup-decode.py','adaptive_cpu_original_lookup')
    resume,ast_proof=P.resume_lookup(lookup.LookupDecoder.generate)
    anchor=P.NOMINAL[-1];draft=tuple(range(10,26));payload=np.array([*draft,26,*([1]*17)],dtype=np.int32)
    for offset in (0,3,15,16):
      values=payload.copy()
      if offset<16:values[offset]=900
      assert P.prefix(values,draft,anchor)==(offset,True)
    for offset in (0,3,16):
      values=payload.copy();values[17+offset]=0
      assert P.prefix(values,draft,anchor)==(0,False)
    reject('missing finite flag',lambda:P.prefix(payload[:-1],draft,anchor))
    reject('floating target IDs',lambda:P.prefix(payload.astype(float),draft,anchor))
    reject('incomplete previous output',lambda:P.choose((10,11),2031,2048,64,anchor))
    reject('interior anchor',lambda:P.complete((10,anchor,11,anchor),anchor))
    reject('oversized prefill',lambda:P.choose((*draft,anchor),2031,2112,64,anchor))
    assert P.choose(None,2031,2048,64,anchor)==P.choose(P.NOMINAL,2031,2048,64,anchor)==()
    assert P.choose((*draft,anchor),2040,2048,64,anchor)==P.choose((*draft,anchor),2031,2048,8,anchor)==()

    # Actual original LookupDecoder loop, with only neural decode/verify doubled.
    for matched in (1,3,15,16):
      expected=[*draft[:matched],901,902,anchor];events=[]
      d=SimpleNamespace(history=[],statistics=[],_adaptive_seed=list(draft[:matched]),compact_io=True,
        carry_verified_token=True,fallback_block=6,warmup_graphs=False)
      d.continuation=lambda context,limit:[]
      d.first_token=lambda logits:int(logits.data[0]);d.read_tokens=lambda x:x.data
      def decode(logits,c,s,pos,step,count):
        assert pos==2031+step and step>=matched
        events.append((pos,step,count));tokens=expected[step:step+count]
        return Array([0]),Array(np.asarray(tokens,dtype=np.int32))
      d.decode=decode
      result=resume(d,SimpleNamespace(cache=[Array.zeros(2,1,1,2304,1)]),SimpleNamespace(max_tokens=64),
        SimpleNamespace(traj_ids={'future_start':anchor}),SimpleNamespace(decode=str),Array([901]),None,None,2031)
      assert result==expected and d.history==[expected] and events[0][0]==2031+matched
      cases.append({'retained':matched,'first_continuation':events[0],'tokens':result})
    # max_tokens and an anchor at the exact last allowed step retain the old loop.
    d._adaptive_seed=[10]*15;d.history=[];d.statistics=[];d.first_token=lambda _:anchor
    d.decode=lambda logits,c,s,pos,step,count:(Array([0]),Array(np.array([anchor],dtype=np.int32)))
    assert resume(d,SimpleNamespace(),SimpleNamespace(max_tokens=16),SimpleNamespace(traj_ids={'future_start':anchor}),
      SimpleNamespace(decode=str),Array([0]),None,None,2031)==[10]*15+[anchor]
    d.decode=lambda logits,c,s,pos,step,count:(Array([0]),Array(np.array([999],dtype=np.int32)))
    reject('max_tokens without anchor',lambda:resume(d,SimpleNamespace(),SimpleNamespace(max_tokens=16),
      SimpleNamespace(traj_ids={'future_start':anchor}),SimpleNamespace(decode=str),Array([0]),None,None,2031))

    # Execute the ACTUAL original padded-prefill install and original BlockDecoder
    # run entry. Tensor arithmetic is a neural double; selection, state, input
    # origin, cache writes, head slicing, and the 17 independent calls are real.
    original=load(SOURCE/'experiments/agent-speculation-prefill.py','adaptive_cpu_speculation')
    inputs=load(SOURCE/'inputs.py','adaptive_cpu_inputs')
    block=lookup.module.module
    mappings=np.array([*P.NOMINAL,*range(10,50)],dtype=np.int32)
    embed=np.zeros((anchor+1,1),dtype=np.float32)
    for i,t in enumerate(mappings):embed[t,0]=i
    mapping_index={int(t):i for i,t in enumerate(mappings)}
    chain={int(a):int(b) for a,b in zip((*draft,26,anchor),(*draft[1:],26,anchor,anchor))}
    chain.update({int(a):int(b) for a,b in zip(P.NOMINAL,(*P.NOMINAL[1:],anchor))})
    calls=[]
    model=SimpleNamespace(config={'vlm_config':{'text_config':{'num_hidden_layers':1}}},dtype=dtypes.bfloat16,
      weights={'vlm.model.language_model.embed_tokens.weight':Array(embed,dtype=dtypes.bfloat16)},
      output_token_mapping=Array(mappings),cache=[Array.zeros(2,1,1,2304,1,dtype=dtypes.bfloat16)],
      prefill_jit=DirectJit(lambda:None),decode_jit=DirectJit(lambda:None),greedy_decode_jit=DirectJit(lambda:None),
      trajectory_jits={},norm=lambda x,*a,**k:x)
    def text(x,cos,sin,name,cache,mask,cache_pos):
      assert cache_pos==0 and mask is None
      cache.data[:,:,:,cache_pos:cache_pos+x.shape[1],:]=x.data[:,None,:,:]
      return x
    def linear(x,name):
      assert x.shape==(1,1,1) and name=='vlm.lm_head'
      token=int(mappings[int(x.data.item())]);target=chain.get(token,10)
      out=np.full((1,1,len(mappings)),-1,dtype=np.float32);out[0,0,mapping_index[target]]=1
      calls.append(token);return Array(out,dtype=dtypes.bfloat16)
    model.text_layer=text;model.linear=linear
    state=H.State(anchor,True)
    with patch('tinygrad.Tensor',Array),patch('tinygrad.TinyJit',DirectJit):
      # local('padded-prefill') is loaded under these doubles, as in construction.
      base=original.install_prefill(model,P.NOMINAL,anchor)
      adaptive=H.make_prefill(model,base,state,inputs.rotary);model.padded_prefill=adaptive
      args0=SimpleNamespace(max_tokens=64,greedy=True,coarse_timing=False,diffusion_steps=3)
      prepared={'actual_prompt_length':2031,'ego_history_xyz':None,'ego_history_rot':None}
      packed=(np.zeros((1,2048),dtype=np.int32),np.zeros((3,1,2048),dtype=np.int64),-1800,np.zeros((1,1)))
      state.previous=(*draft,26,anchor);state.completed=1;state.enter(args0,prepared,packed)
      x=np.zeros((1,2048,1),np.float32);x[:,2030]=mapping_index[49]
      chain[49]=10
      x[:,2031:2037,0]=[mapping_index[t] for t in P.NOMINAL]
      c,s=(Array(a,dtype=dtypes.bfloat16) for a in inputs.rotary(np.zeros((3,1,2048),np.int64)))
      first=adaptive(Array(x,dtype=dtypes.bfloat16),c,s,64,2031)
      assert adaptive.last_payload.shape==(34,) and P.prefix(adaptive.last_payload.data,draft,anchor)==(16,True)
      assert calls[-17:]==[49,*draft]
      assert state.audit_done and adaptive.last_record['prefix_audit']['all_layer_prefix_kv_exact']
      assert adaptive.last_record['rotary_origin']==248
      np.testing.assert_array_equal(model.cache[0].data[0,0,0,2031:2047,0],[mapping_index[t] for t in draft])
      state.running=False;state.current=None;state.active=None

      # Real run method controls order. Only upload/model/GPU/action math mocked.
      decoder=SimpleNamespace(model=model,history=[],statistics=[],before_diffusion=lambda:None,readback=lambda a:a.data,
        read_tokens=lambda a:a.data,select_logits=lambda values,index:values[:,index:index+1],prefill_draft_records=[])
      decoder.generate=lambda *a: [*draft,26,anchor]
      decoder.run=MethodType(block.BlockDecoder.run,decoder)
      def original_generate(model,args,config,tokenizer,logits,dc,ds,prompt):
        result=[10,anchor];decoder.history=[result];decoder.statistics.append({'lookup_rounds':0,'verified_tokens':0,'serial_tokens':2});return result
      def resumed(self,model,args,config,tokenizer,logits,dc,ds,prompt):
        assert self._adaptive_seed==list(draft)
        result=[*self._adaptive_seed,26,anchor];self.history=[result]
        self.statistics.append({'lookup_rounds':0,'verified_tokens':0,'serial_tokens':2});return result
      close=H.attach_decoder(decoder,model,state,original_generate,resumed)
      model.synchronize=lambda:None;model.encode=lambda *a:Array(x,dtype=dtypes.bfloat16)
      trajectory_calls=[]
      def trajectory(actions,ac,ass,length,steps,offset):
        assert state.previous==(*draft,26,anchor) and state.completed==1
        assert (length,steps,offset)==(2049,3,18)
        trajectory_calls.append((length,steps,offset));return Array(np.zeros((1,64,2),np.float32))
      model.trajectory=trajectory
      upload=SimpleNamespace(batches=(),spans=(),upload=lambda noise:(None,None,Array(np.zeros((1,2048),np.int32)),c,s,None,None,None,None,None))
      import torch
      class Action:
        def __init__(self,**kwargs):pass
        def action_to_traj(self,*args):return torch.zeros(1,64,3),torch.eye(3).expand(1,64,3,3)
      prepared['ego_history_xyz']=torch.zeros(1,1,20,3);prepared['ego_history_rot']=torch.eye(3).expand(1,1,20,3,3)
      config=SimpleNamespace(traj_ids={'future_start':anchor},expert_config=SimpleNamespace(action_space_cfg={'_target_':'neural-double'}))
      with patch.object(block,'UnicycleAccelCurvatureCourseActionSpace',Action):
        result=decoder.run(model,args0,config,SimpleNamespace(decode=str),prepared,packed,0,upload)
        assert state.completed==2 and state.previous==tuple(result['generated_token_ids'])
        assert trajectory_calls==[(2049,3,18)] and state.records[-1]['next_decode_rope_step']==16
        state.completed=1
        previous=state.previous;history=copy.deepcopy(decoder.history)
        def fail(*args):raise RuntimeError('deliberate trajectory failure')
        model.trajectory=fail
        reject('failed trajectory not committed',lambda:decoder.run(model,args0,config,SimpleNamespace(decode=str),prepared,packed,0,upload))
        assert state.previous==previous and state.completed==1 and decoder.history==history and not state.running
        model.trajectory=trajectory
        with state.current_input_only(decoder):
          decoder.run(model,args0,config,SimpleNamespace(decode=str),prepared,packed,0,upload)
        assert state.previous==previous and state.completed==1 and decoder.history==history
      close();assert decoder.run.__func__ is block.BlockDecoder.run

    # Full actual import/install path exercises the precise original owner guard.
    overlay=H.Overlay(SOURCE/'experiments')
    old_spec=importlib.util.spec_from_file_location
    with patch('tinygrad.Tensor',Array),patch('tinygrad.TinyJit',DirectJit):
      with overlay:
        wrapped=load(SOURCE/'experiments/agent-speculation-prefill.py','adaptive_cpu_overlay_install')
        assert wrapped.install_prefill.__wrapped__.__code__.co_filename==str(SOURCE/'experiments/agent-speculation-prefill.py')
        wrapped.install_prefill(model,P.NOMINAL,anchor)
        actual_lookup=lookup.LookupDecoder(model)
        original_run=actual_lookup.run
        wrapped.install_decoder(actual_lookup,P.NOMINAL,anchor)
        assert len(overlay.states)==len(overlay.prefills)==1 and overlay.ast_proof['source_reversible']
        assert actual_lookup.run.__func__ is not original_run.__func__
        reject('foreign lookup wrapper',lambda:wrapped.install_decoder(SimpleNamespace(generate=lambda:None),P.NOMINAL,anchor))
      assert actual_lookup.run.__func__ is original_run.__func__
    assert overlay.restored and importlib.util.spec_from_file_location is old_spec

    # Real adaptive decoder branch controls, including all mismatch positions,
    # a short completed draft, and rejection after a nonfinite unused row.
    branch_rows=[]
    for mismatch in (0,3,15,16,'nonfinite','anchor'):
      state=H.State(anchor);state.running=True;state.active=True
      selected=[];fallback=[];resumed=[]
      proposed=list(draft if mismatch!='anchor' else (10,11,anchor))
      targets=np.array([*draft,26,*([1]*17)],dtype=np.int32)
      if isinstance(mismatch,int) and mismatch<16:targets[mismatch]=999
      if mismatch=='nonfinite':targets[-1]=0
      if mismatch=='anchor':targets[:3]=proposed
      prefill=SimpleNamespace(last_record={'actual_prompt':2031,'draft':proposed},last_payload=Array(targets),
        last_logits=Array(np.arange(17,dtype=np.float32).reshape(1,17,1)))
      mdl=SimpleNamespace(padded_prefill=prefill,cache=[Array.zeros(2,1,1,2304,1)])
      dec=SimpleNamespace(history=[],statistics=[],generate=lambda *a:None,run=lambda *a:None,read_tokens=lambda x:x.data)
      def select(values,index):selected.append(index);return values[:,index:index+1]
      dec.select_logits=select
      def fallback_generate(*args):
        fallback.append(True);dec.statistics.append({'lookup_rounds':0,'verified_tokens':0,'serial_tokens':2});return [10,anchor]
      def continuation(self,model,args,config,tokenizer,logits,dc,ds,prompt):
        accepted=len(self._adaptive_seed)
        assert logits.data.item()==accepted
        resumed.append(accepted);self.statistics.append({'lookup_rounds':0,'verified_tokens':0,'serial_tokens':2})
        return [*self._adaptive_seed,999,anchor]
      close=H.attach_decoder(dec,mdl,state,fallback_generate,continuation)
      result=dec.generate(mdl,SimpleNamespace(max_tokens=64),SimpleNamespace(),SimpleNamespace(),Array([-1]),None,None,2031)
      expected=0 if mismatch in (0,'nonfinite') else 3 if mismatch=='anchor' else mismatch
      assert state.records[-1]['accepted_prefix']==expected
      assert bool(fallback)==(expected==0) and selected==([] if expected==0 or mismatch=='anchor' else [expected])
      assert state.records[-1]['final_cache_length']==2031+len(result)
      assert state.records[-1]['action_rotary_offset']==len(result) and not hasattr(dec,'_adaptive_seed')
      close();branch_rows.append({'case':mismatch,'accepted':expected,'selected_logit_rows':selected,'fallback':bool(fallback)})

  assert type(Device).__getitem__ is original_device_get and not Device._opened_devices
  report_path=WORK/'agent-turn-paced-111-119/report.json'
  assert sha(report_path)=='1274905c40cf01d5985f8a8ecd7a2ddb6e7dc02f90fc48c6b17f4494b2d65dc3'
  canonical=json.loads(report_path.read_text())
  saved=json.loads((report_path.parent/'turn-benchmark-summary.json').read_text())
  assert saved['canonical_report_sha256']==sha(report_path)
  assert len(canonical['rows'])==len(saved['queries'])==18
  for direct,summary in zip(canonical['rows'],saved['queries'],strict=True):
    assert direct['query_ns']==summary['query_ns'] and direct['speculative_record']==summary['speculative_record']
  previous=None;rows=[]
  for index,q in enumerate(saved['queries']):
    tokens=q['speculative_record']['generated'];draft=P.choose(previous,q['speculative_record']['actual_prompt'],2048,64,anchor)
    count=0
    for a,b in zip(draft,tokens):
      if a!=b:break
      count+=1
    rows.append({'index':index,'query_ns':q['query_ns'],'prompt':q['speculative_record']['actual_prompt'],
      'navigation':q['navigation'],'output_tokens':len(tokens),'draft_tokens':len(draft),'maximum_matching_prefix':count})
    previous=P.complete(tokens,anchor)
  assert len(rows)==18 and sum(r['maximum_matching_prefix'] for r in rows)==214
  result={'complete':True,'gpu_executed':False,'numerical_candidate':True,'width':16,'rejections':rejected,
    'original_lookup_resume_cases':cases,'ast':ast_proof,'actual_original_prefill_and_block_run':True,
    'actual_adaptive_decoder_branch_cases':branch_rows,
    'seventeen_independent_head_rows':True,'mock_causal_prefix_audit':True,'completion_only_history':True,
    'failed_and_current_input_only_restoration':True,'actual_overlay_import_path':True,'opened_devices':[],
    'replay':{'source_report':str(report_path),'sha256':sha(report_path),'rows':rows,
      'total_output_tokens':455,'maximum_reused_tokens':214,'timing_or_gpu_accuracy_claim':False},
    'sources':{str(p):sha(p) for p in HERE.glob('agent-adaptive-prefill-*.py')}}
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'complete':True,'gpu_executed':False,'rejected':len(rejected),'replay_tokens':214}))


if __name__=='__main__':main()
