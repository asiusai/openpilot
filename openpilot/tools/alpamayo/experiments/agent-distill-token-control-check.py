"""CPU-only token-control entrypoint, native causal cache/rotary/Euler lifecycle, and failure cleanup checks."""
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from types import MethodType,SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
os.environ.update(DEV='CPU',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',
  OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def no_gpu(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):
    raise RuntimeError('Token control checks are CPU-only')
sys.addaudithook(no_gpu);sys.path.insert(0,str(BASE/'tinygrad'))
import numpy as np
from tinygrad import Tensor,Device,dtypes
from tinygrad.helpers import Context
T=runpy.run_path(str(HERE/'agent-distill-token-control-run.py'))
old=T['module']('token_control_prior_fixture',HERE/'agent-distill-composed-check.py')
position=T['module']('token_control_bound_position',HERE/'agent-distill-objective-position-run.py').resolved_position
read,sha=T['read'],T['sha']

def rejected(call):
  try:call()
  except (ValueError,AssertionError):return
  raise AssertionError('Negative control passed')


def controls():
  config=read(BASE/'model/config.json');tokens=read(T['AUTONOMOUS'])['teacher']['prediction']['generated_token_ids']
  validate=lambda seq:T['validate_tokens'](seq,config,64,config['vlm_config']['text_config']['vocab_size'])
  validate(tokens)
  for bad in ([],tokens[:-1],[tokens[-1]]+tokens,tokens+[tokens[-1]],tokens[:-1]+[-1],tokens[:-1]+[True],
              [config['vlm_config']['image_token_id']]+tokens,[config['traj_ids']['history_id0']]+tokens):rejected(lambda:validate(bad))
  entry=runpy.run_path(str(HERE/'agent-distill-token-control-run.py'));original=entry['AUTONOMOUS_SHA']
  # Reuse production checkpoint admission once, then check provenance before any neural initialization.
  paired,receipt=entry['E'].selected_joint(entry['E'].ACTUAL_ROOT,entry['E'].ACTUAL_REPORT_SHA)
  plan=read(entry['C'].PLAN);row=entry['P'].selected_rows(entry['P'].INPUTS,['train-command-0'])[0]
  refs=entry['references'](plan,row,receipt)
  entry['references'].__globals__['AUTONOMOUS_SHA']='0'*64;rejected(lambda:entry['references'](plan,row,receipt))
  entry['references'].__globals__['AUTONOMOUS_SHA']=original
  changed=deepcopy(row);changed['query_ns']+=1;rejected(lambda:entry['references'](plan,changed,receipt))
  changed=deepcopy(receipt);changed['best_sha256']='0'*64;rejected(lambda:entry['references'](plan,row,changed))
  return paired,refs,dict(token_boundaries_and_reserved_ids_rejected=True,actual_selected25_and_references_admitted=True,
    changed_report_input_checkpoint_rejected=True)


def control_layer(model,language_sources,expert_sources,objects,tokens):
  written={};prompt=2027
  def layer(self,x,c,s,prefix,cache=None,mask=None,cache_pos=None):
    source=int(prefix.rsplit('.',1)[1]);student=prefix.startswith('agent_joint.');base=source+1+(.25 if student else 0)
    if prefix.startswith(('expert.','agent_joint.expert.')):
      assert cache_pos is None and cache[0].shape==(1,8,2304,128)
      np.testing.assert_array_equal(cache[0][0,0,prompt:prompt+20,0].float().numpy(),written[source])
      assert float(cache[0][0,0,0,0].float().item())==base
      actual=mask.numpy();assert actual.shape==(2368,)
      np.testing.assert_array_equal(actual,np.r_[np.zeros(2047),np.full(2304-2047,-np.inf),np.zeros(64)])
      expert_sources.append(source)
      context=cache[0][0,0,:2047,0].float().mean()
      return (x.float()+(context+base)*.001).cast(self.dtype).realize()
    language_sources.append(source)
    if type(cache_pos) is int and cache_pos==0:
      assert x.shape==(1,2048,2) and mask is None
      cache.assign(Tensor.full(cache.shape,base,dtype=self.dtype)).realize();written[source]=[]
      return (x.float()+base*.001).cast(self.dtype).realize()
    pos=position(cache_pos);step=pos-prompt
    assert x.shape==(1,1,2) and 0<=step<20 and len(written[source])==step
    # Use native _language's gfx1200 full-capacity mask branch, with neural arithmetic mocked on CPU.
    actual=mask.numpy();assert actual.shape==(2304,)
    np.testing.assert_array_equal(actual,np.r_[np.zeros(pos+1),np.full(2303-pos,-np.inf)])
    if source==0:
      uploads=objects['uploaded']
      np.testing.assert_array_equal(c.float().numpy(),uploads[5][:,:,step:step+1].float().numpy())
      np.testing.assert_array_equal(s.float().numpy(),uploads[6][:,:,step:step+1].float().numpy())
      expected=self.weights['vlm.model.language_model.embed_tokens.weight'][tokens[step]].reshape(1,1,2)
      np.testing.assert_array_equal(x.float().numpy(),expected.float().numpy())
      objects['placement'].append(dict(step=step,position=pos,token_id=tokens[step]))
    value=(x.float().mean()+base).cast(self.dtype).realize()
    # Deliberately poison every uncommitted row. Reading the future would change the output and valid cache signatures.
    cache[:,:,:,pos:].assign(Tensor.full((2,1,8,2304-pos,128),-40,dtype=self.dtype)).realize()
    cache[:,:,:,pos:pos+1].assign(value.expand(2,1,8,1,128)).realize()
    written[source].append(float(value.float().item()))
    causal=mask.isfinite().where(cache[0,0,0,:,0].float(),0).sum()/(pos+1)
    return (x.float()+causal*.001+base*.001).cast(self.dtype).realize()
  return MethodType(layer,model)


def control_inputs(base,objects):
  class Inputs(base):
    def upload(self,noise):
      result=super().upload(noise);objects['uploaded']=result;return result
  return Inputs


def integration(root,paired,refs):
  root.mkdir();namespace=runpy.run_path(str(HERE/'agent-distill-token-control-run.py'));entry=namespace['main'];globals_=entry.__globals__
  original=globals_['execute'];audit=[];objects=dict(placement=[])
  source=inspect.getsource(old.initializer)
  replacements={
    'm.fp8_wmma=m.cuda_kernels=False;':'m.fp8_wmma=True;m.cuda_kernels=False;',
    "m.weights={p+'12.input_layernorm.weight':Tensor.ones(1,dtype=m.dtype) for p in (R.LANGUAGE,R.EXPERT)}":
      "m.weights={prefix+'12.'+name:Tensor.empty(value.shape,dtype=m.dtype) for prefix,role in ((R.LANGUAGE,'language'),(R.EXPERT,'expert')) for name,value in selected_pair[role].items()}",
    "Tensor.zeros(145000,2,dtype=m.dtype).realize()":
      "Tensor.stack(Tensor.arange(145000).float()/131072,(Tensor.arange(145000)%127).float()/128,dim=1).cast(m.dtype).realize()",
    "compact=module('composed_check_compact',root/'experiments/compact-inputs.py').CompactInputs":
      "compact=control_inputs(module('composed_check_compact',root/'experiments/compact-inputs.py').CompactInputs,control_objects)"}
  for before,after in replacements.items():assert source.count(before)==1;source=source.replace(before,after)
  begin=source.index('    def layer(self,');end=source.index('    m.encode=',begin)
  source=source[:begin]+'    m.text_layer=control_layer(m,language_sources,expert_sources,control_objects,control_tokens)\n'+source[end:]
  begin=source.index('    before=decoder.run');end=source.index('    audit.append(dict(model=',begin)
  source=source[:begin]+'''    before=decoder.run
    def run(*a,**kw):
      assert m.cache is None and not decoder.history and not decoder.statistics and not vc.cache and not vc.encoders
      assert m.padded_prefill.shape is None and not m.trajectory_jits
      result=before(*a,**kw)
      sources=list(range(13))+list(range(16,28))
      assert language_sources==sources*21 and expert_sources==sources*3
      assert m.padded_prefill.last_targets.shape==(12,) and decoder.prefill_draft_records==[]
      assert decoder.statistics==[dict(offline_forced_tokens=20,target_verified=False,autonomous=False)]
      assert len(vc.keys)==10 and vc.misses==10 and vc.hits==0
      audit.append(dict(language_calls=21,language_layers=25,expert_calls=75,twenty_current_causal_token_rows=True,
        image_slots=10,first_call_empty=True,spec6_padding_retained_but_acceptance_bypassed=True))
      return result
'''+source[end:]
  old.selected_pair=paired;old.control_layer=control_layer;old.control_inputs=control_inputs
  old.control_objects=objects;old.control_tokens=refs['tokens'];exec(compile(source,str(Path(__file__))+':causal-fixture','exec'),old.__dict__)
  def execute(args,plan,ready,actual_pair,receipt,actual_refs):
    assert receipt['best_sha256']=='8073b781a84b8f27dc9a64b8ffe7fe1aa82fbbead91ab431918396716414e29a'
    mock_refs=deepcopy(actual_refs);mock_refs['settings']['runtime_environment']['DEV']='CPU'
    return original(args,plan,ready,actual_pair,receipt,mock_refs,cpu_mock=True,initialize=old.initializer(plan,ready[1],audit))
  globals_['execute']=execute
  previous=sys.argv[:];sys.argv=[str(HERE/'agent-distill-token-control-run.py'),'--output',str(root/'report.json'),'--offline-teacher-token-control','--execute-gpu']
  try:entry()
  finally:sys.argv=previous
  result=read(root/'report.json');state=audit.pop(0)
  assert result['complete'] and result['adapter_restored'] and result['decoder_restored'] and not result['gpu_executed']
  assert not result['usable_inference_configuration'] and not result['target_verified_speculation'] and not result['quality_qualified']
  assert result['forced']['prediction']['generated_token_ids']==refs['tokens'] and len(result['token_trace'])==20
  assert objects['placement']==[dict(step=i,position=2027+i,token_id=t) for i,t in enumerate(refs['tokens'])]
  assert result['comparison']['teacher_vs_forced']['tokens_equal'] and not result['comparison']['autonomous_vs_forced']['tokens_equal']
  assert all(s['context_valid_length']==2047 for s in result['forced']['rollout']['steps'])
  assert all('context_k' in s['conditional_block']['differing_tensors'] for s in result['comparison']['teacher_vs_forced']['euler'])
  model,decoder,vc=(state[k] for k in ('model','decoder','vision_cache'))
  assert model.cache is None and not decoder.history and not vc.cache and not model.trajectory_jits
  assert not hasattr(decoder,'token_control_teacher_text') and not any(k.startswith('agent_joint.') for k in model.weights)
  return dict(actual_runpy_entrypoint=True,actual_checkpoint_admission=True,all16_original_width_masters_loaded=True,
    native_prefill_language_mask_rotary_cache_expert_Euler_loops=True,native_gfx1200_full_capacity_mask_branch=True,
    neural_blocks_mocked=True,all20_genuine_embedding_indices_checked=True,
    future_cache_poisoned_and_excluded=True,exact_teacher_sequence_with_own25_layer_KV=True,cleanup_passed=True,arms=audit,report=str(root/'report.json'))


def cleanup_failure():
  # An exception inside the installed callback must restore the original generate method.
  model=SimpleNamespace();decoder=SimpleNamespace(generate=lambda:None,token_control_teacher_text='fixture');before=decoder.generate
  try:
    with T['force_tokens'](decoder,model,[1],1,[]):raise RuntimeError('injected')
  except RuntimeError:pass
  assert decoder.generate is before
  return dict(forced_callback_restored_on_exception=True)


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(T['ROOT']):raise ValueError('Fresh owned output required')
  root.mkdir(parents=True);paired,refs,checks=controls()
  result=dict(gpu_executed=False,optimizer_updates=0,admission=checks,integration=integration(root/'integration',paired,refs),cleanup_failure=cleanup_failure())
  T['P'].no_devices();assert all(name.startswith(('CPU','NPY','PYTHON','DISK')) for name in Device._opened_devices)
  result.update(all_passed=True,opened_devices=list(Device._opened_devices),source_sha256={str(p):sha(p) for p in HERE.glob('agent-distill-token-control-*.py')})
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
