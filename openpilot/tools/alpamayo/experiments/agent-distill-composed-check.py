"""CPU integration: actual launcher/spec6/inputs/cache/decoder/Euler, with mocked neural blocks.

Synthetic completed checkpoint files are explicitly marked and rejected by the normal loader.
No optimizer or GPU operation occurs. This checks composition and receipts, not prediction quality.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import MethodType,SimpleNamespace

os.environ.update(DEV='CPU',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',
  PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
sys.path.insert(0,str(BASE/'tinygrad'))
import numpy as np
from tinygrad import Tensor,TinyJit,Device,dtypes
from tinygrad.helpers import Context

C=runpy.run_path(str(HERE/'agent-distill-composed-run.py'))
module,sha,read=C['module'],C['sha'],C['read']
P=module('composed_check_parent',HERE/'agent-distill-joint-parent.py')
B=module('composed_check_expert',HERE/'agent-distill-expert-block.py')
S=module('composed_check_state',HERE/'agent-distill-fit-state.py')
L=module('composed_check_ledger',HERE/'agent-distill-expert-ledger-run.py')
R=module('composed_check_adapter',HERE/'agent-distill-joint-runtime.py')


def forbidden_devices(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
    name=os.fsdecode(args[0])
    if name.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')):raise RuntimeError('CPU check prohibited device: '+name)


sys.addaudithook(forbidden_devices)


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def rejects(call):
  try:call()
  except (ValueError,AssertionError):return True
  raise AssertionError('Negative control unexpectedly passed')


def fixture(root):
  root.mkdir();seed=B.seed();shapes={k:v.shape for k,v in seed.items()}
  records,manifest=B.target_headers([C['TEACHERS']['train-command-0'][0]],[C['TEACHERS']['validation-command-0'][0]])
  digest=hashlib.sha256()
  for name,value in sorted(seed.items()):digest.update(name.encode());digest.update(value.tobytes())
  context=dict(manifest_sha256=manifest['manifest_sha256'],hyperparameters=B.HYPER,seed_sha256=digest.hexdigest(),
    block_sha256=sha(HERE/'agent-distill-expert-block.py'),synthetic_only=False)
  samples=[r['id'] for r in records if r['split']=='validation']
  validation=[dict(step=step,mean=mean,losses=[mean]*3,samples=samples) for step,mean in ((0,1.),(25,2.))]
  with (root/'best.npz').open('xb') as stream:np.savez(stream,**seed)
  metadata=dict(step=25,best_step=0,best_validation=1.,validation=validation,best_artifact_sha256=sha(root/'best.npz'))
  arrays={prefix+'/'+name:(value.copy() if prefix=='weight' else np.zeros_like(value))
          for prefix in ('weight','moment1','moment2') for name,value in seed.items()}
  arrays.update(beta1_power=np.array([np.float32(.9)**25],np.float32),beta2_power=np.array([np.float32(.999)**25],np.float32),
                learning_rate=np.array([3.5e-6],np.float32))
  S.save(root,arrays,metadata,context,shapes,initial_recovery=True)
  report=dict(synthetic_fixture=True,complete=True,gpu_executed=True,committed_step=25,
    allocator_guards_retained_through_finalizers=True,qualified_ledger=dict(sha256=L.QUALIFIED_SHA),unchanged_trainer_sha256=L.TRAINER_SHA,
    conditioning='full detached teacher KV',hyperparameters=B.HYPER,metadata=metadata,manifest=manifest,best_sha256=sha(root/'best.npz'))
  write(root/'report.json',report);write(root/'wall.json',dict(returncode=0,timed_out=False))
  write(root/'best.json',dict(file='best.npz',sha256=sha(root/'best.npz'),selected_step=0,validation_mean=1.))
  return report


def selection(root):
  report=fixture(root)
  load=lambda:C['selected_expert'](root,sha(root/'report.json'),fixture=True)
  weights,seed_receipt=load()
  assert seed_receipt['selected_step']==0 and not seed_receipt['expert_trained'] and not seed_receipt['validation_improved']
  negatives=[]
  rejects(lambda:C['selected_expert'](root,sha(root/'report.json')));negatives.append('synthetic artifact rejected without test-only admission')
  original=read(root/'journal.json');write(root/'journal.json',dict(committed=25,intended=26))
  rejects(load);write(root/'journal.json',original);negatives.append('unclean optimizer journal')
  report['complete']=False;write(root/'report.json',report);rejects(load)
  report['complete']=True;write(root/'report.json',report);negatives.append('incomplete pilot')
  rejects(lambda:C['selected_expert'](root,'0'*64,fixture=True));negatives.append('changed report checksum')
  best=read(root/'best.json');best['sha256']='0'*64;write(root/'best.json',best);rejects(load)
  best['sha256']=sha(root/'best.npz');write(root/'best.json',best);negatives.append('changed best checksum')
  saved_best=(root/'best.npz').read_bytes();pointer=read(root/'latest.json')
  bad={k:v.copy() for k,v in weights.items()};bad[next(iter(bad))].flat[0]=np.nan
  with (root/'best.npz').open('wb') as stream:np.savez(stream,**bad)
  bad_sha=sha(root/'best.npz');best['sha256']=bad_sha;report['best_sha256']=bad_sha
  report['metadata']['best_artifact_sha256']=bad_sha;pointer['metadata']=report['metadata']
  write(root/'best.json',best);write(root/'latest.json',pointer);write(root/'report.json',report);rejects(load)
  (root/'best.npz').write_bytes(saved_best);restored_sha=sha(root/'best.npz')
  best['sha256']=restored_sha;report['best_sha256']=restored_sha;report['metadata']['best_artifact_sha256']=restored_sha
  pointer['metadata']=report['metadata'];write(root/'best.json',best);write(root/'latest.json',pointer);write(root/'report.json',report)
  negatives.append('nonfinite best with internally consistent checksums')
  # Reuse one marked fixture: validate the improving branch without any fitting.
  metadata=report['metadata'];metadata['validation'][1].update(mean=.5,losses=[.5]*3)
  metadata.update(best_step=25,best_validation=.5)
  pointer=read(root/'latest.json');pointer['metadata']=metadata;write(root/'latest.json',pointer)
  best.update(selected_step=25,validation_mean=.5);write(root/'best.json',best);write(root/'report.json',report)
  _,trained_receipt=load();assert trained_receipt['selected_step']==25 and trained_receipt['expert_trained']
  assert trained_receipt['validation_improved'] and not trained_receipt['quality_qualified']
  # Restore the seed-selected case for full integration. This proves a complete pilot need not select trained weights.
  metadata['validation'][1].update(mean=2.,losses=[2.]*3);metadata.update(best_step=0,best_validation=1.)
  pointer['metadata']=metadata;write(root/'latest.json',pointer);best.update(selected_step=0,validation_mean=1.)
  write(root/'best.json',best);write(root/'report.json',report)
  return dict(seed_selected_receipt=seed_receipt,update25_receipt=trained_receipt,negative_controls=negatives)


def initializer(plan,prepared,audit):
  root=Path(plan['runtime_root'])
  def initialize(args,*,sources,source_dir):
    # This callback runs inside the actual selected launcher's import hooks.
    assert source_dir==root/'experiments'
    os.environ['DEV']='CPU'
    import torch
    from transformers import AutoConfig,AutoTokenizer
    native=module('composed_check_native',root/'model.py').EdgeModel
    config=AutoConfig.from_pretrained(BASE/'model',local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(BASE/'model',local_files_only=True)
    m=native.__new__(native);m.config=deepcopy(read(BASE/'model/config.json'));m.dtype=dtypes.bfloat16
    m.fp8_wmma=m.cuda_kernels=False;m.cache=None;m.prompt_length=m.greedy_shape=None
    m.weights={p+'12.input_layernorm.weight':Tensor.ones(1,dtype=m.dtype) for p in (R.LANGUAGE,R.EXPERT)}
    m.weights['vlm.model.language_model.embed_tokens.weight']=Tensor.zeros(145000,2,dtype=m.dtype).realize()
    m.weight_scales={};m.trajectory_jits={};m.encode_jits={}
    m.prefill_jit=TinyJit(m._language);m.decode_jit=TinyJit(m._language);m.greedy_decode_jit=TinyJit(m._greedy_decode)
    m.minimum_cache_capacity=2304
    m.norm=lambda x,*a,**kw:x
    m.action_projection=lambda actions,timestep: (actions+timestep.reshape(1,1,1)).pad(((0,0),(0,0),(0,510))).cast(m.dtype)
    policy=module('composed_check_policy',root/'experiments/agent-speculation-policy.py')
    m.output_token_mapping=Tensor(np.array(policy.CHECKPOINT_DRAFT_IDS,np.int32)).realize()
    head_calls=[0];expert_sources=[];language_sources=[]
    def linear(x,name,*a,**kw):
      if name=='vlm.lm_head':
        i=head_calls[0]%6;head_calls[0]+=1
        logits=np.zeros((1,1,6),np.float32);logits[0,0,i]=1
        return Tensor(logits).cast(m.dtype)
      assert name=='expert.action_out_proj';return x[:,:,:2]*.01
    m.linear=linear
    def layer(self,x,c,s,prefix,cache=None,mask=None,cache_pos=None):
      source=int(prefix.rsplit('.',1)[1]);student=prefix.startswith('agent_joint.')
      expected=source+1+(.25 if student else 0)
      if prefix.startswith(('expert.','agent_joint.expert.')):
        assert cache_pos is None and cache[0].shape==(1,8,2304,128)
        assert float(cache[0][0,0,0,0].float().item())==expected
        assert float(cache[1][0,0,0,0].float().item())==expected
        expert_sources.append(source)
        return (x.float()+expected*.001).cast(m.dtype).realize()
      assert cache_pos==0 and x.shape==(1,2048,2)
      language_sources.append(source)
      cache.assign(Tensor.full(cache.shape,expected,dtype=m.dtype)).realize()
      return (x.float()+expected*.001).cast(m.dtype).realize()
    m.text_layer=MethodType(layer,m)
    m.encode=lambda *a:None
    def vision(pixels,positions):
      assert pixels.shape[1]==720
      return (pixels.float().mean()*.01+Tensor.zeros(180,2)).cast(m.dtype).realize()
    m._vision=vision
    # All installers below come from the selected source tree under the real launcher hook.
    padding=module('composed_check_padding',root/'experiments/padded-prefill.py');padding.install(m)
    assert type(m.padded_prefill).__name__=='SpeculativePrefill'
    decoder=module('composed_check_decoder',root/'experiments/lookup-decode.py').LookupDecoder(m)
    decoder.readback=lambda tensor:tensor.numpy();decoder.read_tokens=lambda tensor:tensor.numpy()
    decoder.history=[[991]];decoder.statistics=[dict(stale=True)]
    vc=module('composed_check_visioncache',root/'experiments/vision-cache.py').VisionCache(m,grouped=True)
    vc.cache['stale']=Tensor.ones(1);vc.encoders['stale']=TinyJit(lambda x:x)
    compact=module('composed_check_compact',root/'experiments/compact-inputs.py').CompactInputs
    settings=SimpleNamespace(max_tokens=args.max_tokens,diffusion_steps=args.steps,vision_batch=args.vision_batch,greedy=True,
      byte_pixels=args.byte_pixels,cache_rotary_inputs=args.cache_rotary_inputs,coarse_timing=args.coarse_timing,
      initial_noise=np.load(args.initial_noise,allow_pickle=False))
    report=deepcopy(read(C['TEACHERS']['train-command-0'][0])['settings'])
    # Mark the actual fixture environment; reuse tests below check saved teachers separately.
    report['runtime_environment']['DEV']='CPU'
    before=decoder.run
    def run(*a,**kw):
      assert m.cache is None and not decoder.history and not decoder.statistics and not vc.cache and not vc.encoders
      assert m.padded_prefill.shape is None and not m.trajectory_jits
      start_l,start_e=len(language_sources),len(expert_sources)
      result=before(*a,**kw)
      sources=list(range(13))+list(range(16,28)) if hasattr(m,'joint_depth_mapping') else list(range(28))
      assert language_sources[start_l:]==sources and expert_sources[start_e:]==sources*3
      assert m.padded_prefill.last_targets.shape==(12,) and decoder.prefill_draft_records[-1]['accepted']
      assert len(vc.keys)==10 and vc.misses==len(set(vc.keys)) and vc.hits==0
      audit.append(dict(layers=len(m.cache),language_sources=sources,expert_calls=len(sources)*3,
        actual_spec6_accepted=True,current_cache_values_checked_for_every_expert_layer=True,
        image_slots=len(vc.keys),fresh_unique_images=vc.misses,first_call_empty=True))
      return result
    audit.append(dict(model=m,decoder=decoder,vision_cache=vc))
    return SimpleNamespace(model=m,settings=settings,config=config,tokenizer=tokenizer,vision_cache=vc,decoder=decoder,
      input_class=compact,pad_inputs=padding.pad_inputs,runner=run,report=report,Device=Device)
  return initialize


def integration(root,fixture_root):
  root.mkdir()
  # Test a complete CLI path from runpy-loaded source, injecting only checkpoint-test admission and neural initialization.
  namespace=runpy.run_path(str(HERE/'agent-distill-composed-run.py'));entry=namespace['main'];globals_=entry.__globals__
  actual_selected=globals_['selected_expert'];actual_execute=globals_['execute_prediction'];audit=[]
  globals_['selected_expert']=lambda path,digest:actual_selected(path,digest,fixture=True)
  def execute(args,selected_plan,actual_ready,language,expert,receipt):
    assert len(language)==8 and sum(v.size for v in language.values())==50335744
    assert sum(v.size for v in expert.values())==7078912
    toy={'input_layernorm.weight':np.ones(1,np.float32)}
    return actual_execute(args,selected_plan,actual_ready,toy,toy,receipt,cpu_mock=True,
                          initialize=initializer(selected_plan,actual_ready[1],audit))
  globals_['execute_prediction']=execute
  old_argv=sys.argv[:]
  sys.argv=[str(HERE/'agent-distill-composed-run.py'),'--output',str(root/'report.json'),'--sample-id','train-command-0',
    '--expert-root',str(fixture_root),'--expert-report-sha',sha(fixture_root/'report.json'),'--rerun-teacher','--execute-gpu']
  try:entry()
  finally:sys.argv=old_argv
  report=read(root/'report.json');objects=audit.pop(0)
  assert report['complete'] and report['cpu_integration_mock'] and not report['gpu_executed']
  assert [arm['layers'] for arm in audit]==[28,25]
  assert report['adapter']['expert_selected_step']==0 and report['adapter']['original_seed_selected']
  assert not report['adapter']['expert_trained'] and not report['adapter']['teacher_cache_injected']
  assert all(not row['conditional_block']['matched'] for row in report['comparison']['euler'])
  assert report['comparison']['tokens_equal'] and report['comparison']['xyz_max_absolute_m']>0
  assert objects['model'].cache is None and not objects['decoder'].history and not objects['vision_cache'].cache
  assert objects['model'].config['expert_config']['llm_config']['num_hidden_layers']==28
  assert not any(k.startswith('agent_joint.') for k in objects['model'].weights)
  assert not objects['model'].trajectory_jits and objects['model'].padded_prefill.shape is None
  return dict(actual_runpy_source_and_cli_main=True,actual_selected_launcher_and_hooks=True,
    actual_compact_input_upload=True,actual_ten_image_cache=True,actual_block_decoder_and_action_decode=True,
    actual_native_expert_and_three_euler_loops=True,neural_blocks_mocked=True,checkpoint_fixture=True,
    arms=audit,adapter_restored=True,report=str(root/'report.json'))


def reuse_checks():
  plan=read(C['PLAN']);result=[]
  for sample,(path,digest) in C['TEACHERS'].items():
    row=P.selected_rows(P.INPUTS,[sample])[0];settings=read(path)['settings']
    value=C['saved_teacher'](path,digest,row,settings,plan)
    inv=read(plan['invocation_path'])['command'];noise=np.load(inv[inv.index('--initial-noise')+1],allow_pickle=False)
    C['validate_rollout'](value['rollout'],28,noise,value['prediction']['prompt_tokens']+value['prediction']['generated_tokens'])
    changed=deepcopy(settings);changed['initial_noise_sha256']='0'*64
    rejects(lambda:C['saved_teacher'](path,digest,row,changed,plan))
    wrong=deepcopy(row);wrong['actual_prompt_tokens']+=1
    rejects(lambda:C['saved_teacher'](path,digest,wrong,settings,plan))
    result.append(dict(sample=sample,reuse_checks_passed=True,original_reasoning_tokens=value['prediction']['generated_tokens'],
      noise_mismatch_rejected=True,input_mismatch_rejected=True,full_context_retained=True))
  return result


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(WORK):raise ValueError('Fresh owned output directory required')
  root.mkdir(parents=True)
  result=dict(gpu_executed=False,optimizer_updates=0,quality_qualified=False,
    selection=selection(root/'marked-checkpoint-fixture'),reuse=reuse_checks())
  result['integration']=integration(root/'integration',root/'marked-checkpoint-fixture')
  P.no_devices();assert all(name.startswith(('CPU','NPY','DISK')) for name in Device._opened_devices),Device._opened_devices
  result.update(all_passed=True,opened_devices=list(Device._opened_devices),
    source_sha256={str(p):sha(p) for p in HERE.glob('agent-distill-composed-*.py')})
  write(root/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
