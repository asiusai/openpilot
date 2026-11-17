"""CPU checks for padding, RoPE, target acceptance, cache masks, and fallback."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def policy_checks(policy):
  rng=np.random.default_rng(21122027)
  draft=policy.CHECKPOINT_DRAFT_IDS; anchor=draft[-1]
  cases=0
  for prompt in (1,17,2026,2027,2033,2106,2107,2111,2112):
    for delta in (-1800,-1,0,37):
      ids=rng.integers(1,100000,(1,prompt),dtype=np.int32)
      positions=rng.integers(0,1000,(3,1,prompt),dtype=np.int64)
      vision=rng.normal(size=(2,4)).astype(np.float32)
      prepared={'camera_sentinel':object()}
      original_ids,original_positions=ids.copy(),positions.copy()
      metadata,(new_ids,new_positions,new_delta,new_vision)=policy.padded_inputs(prepared,(ids,positions,delta,vision),2112,draft,anchor)
      np.testing.assert_array_equal(ids,original_ids);np.testing.assert_array_equal(positions,original_positions)
      np.testing.assert_array_equal(new_ids[:,:prompt],ids)
      np.testing.assert_array_equal(new_positions[:,:,:prompt],positions)
      assert new_vision is vision and metadata['camera_sentinel'] is prepared['camera_sentinel']
      assert metadata['actual_prompt_length']==prompt
      assert metadata['prefill_draft_eligible']==(prompt+6<=2112)
      np.testing.assert_array_equal(np.arange(128)+2112+new_delta,np.arange(128)+prompt+delta)
      if metadata['prefill_draft_eligible']:
        np.testing.assert_array_equal(new_ids[0,prompt:prompt+6],draft)
        expected=np.broadcast_to((np.arange(6)+prompt+delta)[None,None,:],(3,1,6))
        np.testing.assert_array_equal(new_positions[:,:,prompt:prompt+6],expected)
        assert np.all(new_ids[:,prompt+6:]==0) and np.all(new_positions[:,:,prompt+6:]==0)
      else:
        assert np.all(new_ids[:,prompt:]==0) and np.all(new_positions[:,:,prompt:]==0)
      cases+=1
  assert policy.accept_targets(np.array(draft,dtype=np.int32),draft,anchor)
  for index in range(6):
    wrong=np.array(draft,dtype=np.int32);wrong[index]+=1
    assert not policy.accept_targets(wrong,draft,anchor)
  assert not policy.accept_targets(np.array(draft),draft,anchor,finite=False)
  assert not policy.accept_targets(np.array(draft[:-1]),draft,anchor)
  assert not policy.accept_targets(np.array(draft,dtype=np.float32),draft,anchor)
  for prompt in (17,2027,2106):
    np.testing.assert_array_equal(policy.target_rows(prompt),np.arange(prompt-1,prompt+5))
    for generated in (1,2,6,64):
      contract=policy.cache_contract(prompt,generated,2304)
      assert contract['cache_length']==prompt+generated and contract['action_rotary_offset']==generated
      assert np.all(contract['valid_text'][:prompt+generated])
      assert not np.any(contract['valid_text'][prompt+generated:])
      assert np.all(contract['expert_mask'][2304:])
    # Each rejected speculative row is masked until normal decoding overwrites
    # its current position. No accepted speculative prefix is retained.
    poison=np.arange(2304,dtype=np.int64)+1000000
    for offset in range(6):
      position=prompt+offset
      poison[position]=offset
      visible=np.arange(2304)<=position
      assert not np.any(visible[position+1:])
      np.testing.assert_array_equal(poison[prompt:position+1],np.arange(offset+1))
  return dict(layout_cases=cases,mismatch_positions_rejected=list(range(6)),finite_gate=True,cache_masks=True)


def toy_checks(implementation,policy):
  from tinygrad import Device,Tensor,TinyJit,dtypes
  if not Device.DEFAULT.startswith('CPU'): raise ValueError('CPU checks require DEV=CPU')
  class ToyModel:
    def __init__(self):
      self.dtype=dtypes.bfloat16
      self.config={'vlm_config':{'text_config':{'num_hidden_layers':2}}}
      self.weights={'vlm.model.language_model.embed_tokens.weight':Tensor.eye(16).cast(self.dtype).realize()}
      self.output_token_mapping=Tensor.arange(16,dtype=dtypes.int32).realize()
      table=np.zeros((16,16),dtype=np.float32)
      for before,after in ((8,1),(1,2),(2,3),(3,4),(4,5),(5,7)):table[before,after]=10
      self.head=Tensor(table).cast(self.dtype).realize()
      self.cache=None;self.prompt_length=None;self.trajectory_jits={}
      self.prefill_jit=TinyJit(lambda x:x);self.decode_jit=TinyJit(lambda x:x);self.greedy_decode_jit=TinyJit(lambda x:x)
    def text_layer(self,x,cos,sin,prefix,cache,mask,cache_pos):
      assert cache_pos==0 and mask is None
      # Small causal CPU layer. Future tokens cannot alter previous hidden rows.
      y=(x.float()+x.float().cumsum(axis=1)*.001).cast(self.dtype).realize()
      values=(y.float()*Tensor.arange(16)).sum(axis=-1).reshape(1,1,x.shape[1],1)
      values=(values+cos[:,:,:,:1].float()*.01).cast(self.dtype).expand(1,8,x.shape[1],128)
      store=cache[:,:,:,:x.shape[1]].uop.store(Tensor.stack(values,values).uop)
      return Tensor(y.uop.after(store)).realize()
    def norm(self,x,name,rms=False):return x
    def linear(self,x,name):return (x@self.head).cast(self.dtype).realize()
  draft=(1,2,3,4,5,7);anchor=7
  model=ToyModel();prefill=implementation.install_prefill(model,draft,anchor,audit=True)
  checked=[]
  for prompt in (3,7,5,3):
    ids=np.full((1,prompt),9,dtype=np.int32);ids[0,-1]=8
    positions=np.broadcast_to(np.arange(prompt),(3,1,prompt)).copy()
    _,packed=policy.padded_inputs({},(ids,positions,0,np.zeros((1,1))),32,draft,anchor)
    x=model.weights['vlm.model.language_model.embed_tokens.weight'][Tensor(packed[0])].realize()
    angles=np.repeat(packed[1][0][:,None,:,None],128,axis=-1).astype(np.float32)*.01
    cos,sin=(Tensor(value).cast(model.dtype).realize() for value in (np.cos(angles),np.sin(angles)))
    first=prefill(x,cos,sin,64,prompt)
    payload=prefill.last_targets.numpy()
    np.testing.assert_array_equal(payload[:6],draft)
    assert np.all(payload[6:]==1) and prefill.prefix_audit['prefix_kv_exact']
    assert int(first.argmax().item())==draft[0]
    checked.append(prompt)
  # Exercise every rejection position through the real wrapper with a separate
  # old-decoder stub. Check forwarding and complete restart rather than prefixes.
  class OldDecoder:
    def __init__(self):self.history=[];self.statistics=[];self.calls=[]
    def read_tokens(self,t):return t.numpy()
    def generate(self,m,args,config,tokenizer,logits,dc,ds,prompt):
      self.calls.append((logits,dc,ds,prompt))
      generated=getattr(self,'result',[11,7])[:]
      for index in range(len(generated)):
        for cache in m.cache:
          cache[:,:,:,prompt+index:prompt+index+1].assign(Tensor.full((2,1,8,1,128),42+index,dtype=m.dtype)).realize()
      self.history.append(generated[:]);self.statistics.append({'serial_tokens':len(generated)})
      return generated
  for mismatch in (None,*range(6),'finite'):
    decoder=OldDecoder();implementation.install_decoder(decoder,draft,anchor)
    payload=np.array([*draft,*([1]*6)],dtype=np.int32)
    if mismatch=='finite':payload[6]=0
    elif mismatch is not None:payload[mismatch]+=1
    prefill.last_targets=Tensor(payload).realize();prefill.last_prompt=5
    dc,ds=object(),object()
    result=decoder.generate(model,SimpleNamespace(max_tokens=64),SimpleNamespace(),None,first,dc,ds,5)
    if mismatch is None:assert result==list(draft) and not decoder.calls
    else:
      assert result==[11,7] and decoder.calls==[(first,dc,ds,5)]
      assert decoder.prefill_draft_records[-1]['final_cache_length']==7
      assert decoder.prefill_draft_records[-1]['action_rotary_offset']==2
  for accepted in (True,False):
    decoder=OldDecoder();decoder.history=[[12,7]];decoder.result=list(draft) if accepted else [11,7]
    implementation.install_decoder(decoder,draft,anchor,audit=True)
    payload=np.array([*draft,*([1]*6)],dtype=np.int32)
    if not accepted:payload[-7]+=1
    prefill.last_targets=Tensor(payload).realize();prefill.last_prompt=5
    saved=[cache[:,:,:,5:11].numpy().copy() for cache in model.cache]
    result=decoder.generate(model,SimpleNamespace(max_tokens=64),SimpleNamespace(),None,first,object(),object(),5)
    assert decoder.prefill_draft_records[-1]['old_decoder_tokens']==decoder.result
    assert decoder.prefill_draft_old_history==[decoder.result]
    assert decoder.history==[[12,7],decoder.result]
    if accepted:
      assert result==list(draft)
      for expected,cache in zip(saved,model.cache,strict=True):np.testing.assert_array_equal(cache[:,:,:,5:11].numpy(),expected)
    else:assert result==[11,7] and len(decoder.calls)==2
  return dict(cpu_jit_prompt_lengths=checked,prefix_logits_and_kv_exact=True,complete_restart_cases=7,
              audit_suffix_restored=True,audit_old_history_independent=True)


def real_checks(args,policy):
  import torch
  torch.set_num_threads(4)
  adapter=local('fast-inputs').FastInputs(args.checkpoint,camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True)
  compact=local('compact-inputs').CompactInputs
  loader=local('nvidia-reference').load_raw
  draft=tuple(adapter.tokenizer.encode(policy.DRAFT_TEXT,add_special_tokens=False))
  assert draft==policy.CHECKPOINT_DRAFT_IDS
  settings=SimpleNamespace(max_tokens=64,byte_pixels=True)
  samples=json.loads(args.samples_json.read_text())
  if args.limit:samples=samples[:args.limit]
  results=[]
  for sample in samples:
    data=loader(Path(sample));_,_,prepared=adapter.prepare(data);packed=adapter.native_inputs(prepared)
    new_prepared,new_packed=policy.padded_inputs(prepared,packed,2112,draft,draft[-1])
    actual=packed[0].shape[1]
    before,after=(compact.arrays(settings,p,k) for p,k in ((prepared,packed),(new_prepared,new_packed)))
    for index,(original,changed) in enumerate(zip(before,after,strict=True)):
      if index==2:changed=changed[:,:actual]
      elif index in (3,4):changed=changed[:,:,:actual]
      np.testing.assert_array_equal(original,changed,err_msg=f'{sample} array {index}')
    if new_prepared['prefill_draft_eligible']:
      for prefill_index,decode_index in ((3,5),(4,6)):
        np.testing.assert_array_equal(after[prefill_index][:,:,actual:actual+6],before[decode_index][:,:,:6])
    results.append(dict(sample=str(sample),actual_prompt=actual,eligible=new_prepared['prefill_draft_eligible'],
                        prefix_ids_rotary_exact=True,decode_action_rotary_exact=True,camera_history_arrays_exact=True))
    print(json.dumps(results[-1]),flush=True)
  return results


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--checkpoint')
  parser.add_argument('--samples-json',type=Path)
  parser.add_argument('--limit',type=int,default=0)
  parser.add_argument('--skip-toy',action='store_true')
  args=parser.parse_args()
  for path in Path(__file__).parent.glob('agent-speculation*.py'):ast.parse(path.read_text())
  policy=local('agent-speculation-policy')
  result=dict(cpu_only=True,policy=policy_checks(policy),gpu_validation='not run')
  if not args.skip_toy:result['toy']=toy_checks(local('agent-speculation-prefill'),policy)
  if args.samples_json:
    if not args.checkpoint:parser.error('--samples-json requires --checkpoint')
    result['real_inputs']=real_checks(args,policy)
  args.output.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({key:value for key,value in result.items() if key!='real_inputs'},indent=2),flush=True)


if __name__=='__main__':main()
