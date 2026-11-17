"""One-sample composed diagnostic. GPU execution belongs to the normal parent ACL/lock wrapper."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import MethodType

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
PLAN=WORK/'agent-distill-command-artifacts/capture-v1/agent-distill-command-capture-plan.json'
TEACHERS={
  'train-command-0':(WORK/'agent-distill-joint-teacher-v1/report.json','5b796b1490bf234cf874a007be575612cb03599d61ba4cde041302e606b4052c'),
  'validation-command-0':(WORK/'agent-distill-joint-validation-v1/report.json','fdfc6a47355e9f164bc63df1ec240062931be43a0e05a56794593e6ade234d7c')}
SETTINGS=('prefill_calibration_sha256','vision_calibration_sha256','initial_noise_sha256','settings','runtime_environment')
CONDITIONS=('x','cos','sin','context_k','context_v','mask')


def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def selected_expert(root,report_sha,*,fixture=False):
  """Admit only a completed clean pilot and its actual validation-selected artifact."""
  import numpy as np
  root=Path(root).resolve()
  allowed=WORK/('agent-distill-composed-artifacts' if fixture else 'agent-distill-expert-artifacts')
  if not root.is_relative_to(allowed):raise ValueError('Expected an owned expert pilot directory')
  report=read(root/'report.json')
  if report.get('synthetic_fixture') and not fixture:raise ValueError('Synthetic fixture is not a trained expert artifact')
  if sha(root/'report.json')!=report_sha:raise ValueError('Expert report changed')
  if not report.get('complete') or not report.get('gpu_executed') or report.get('committed_step')!=25:
    raise ValueError('Expert pilot is incomplete')
  if not report.get('allocator_guards_retained_through_finalizers') or report.get('conditioning')!='full detached teacher KV':
    raise ValueError('Unexpected expert pilot')
  L=module('composed_qualified_ledger_entry',HERE/'agent-distill-expert-ledger-run.py')
  if report.get('qualified_ledger',{}).get('sha256')!=L.QUALIFIED_SHA or report.get('unchanged_trainer_sha256')!=L.TRAINER_SHA:
    raise ValueError('Unexpected expert training implementation')
  if sha(HERE/'agent-distill-expert-train.py')!=L.TRAINER_SHA:raise ValueError('Original expert trainer changed')
  wall=read(root/'wall.json')
  if wall['returncode']!=0 or wall['timed_out']:raise ValueError('Expert parent process did not finish successfully')
  B=module('composed_expert_block',HERE/'agent-distill-expert-block.py')
  S=module('composed_expert_state',HERE/'agent-distill-fit-state.py')
  A=module('composed_expert_best',HERE/'agent-distill-continue-admit.py')
  records,manifest=B.target_headers([TEACHERS['train-command-0'][0]],[TEACHERS['validation-command-0'][0]])
  if report['manifest']!=manifest:raise ValueError('Wrong approved expert training/validation targets')
  seed=B.seed();shapes={k:v.shape for k,v in seed.items()}
  pointer=read(root/'latest.json');context=pointer['context']
  if context.get('synthetic_only') is not False or context['hyperparameters']!=B.HYPER:
    raise ValueError('Wrong expert training context')
  if context['block_sha256']!=sha(HERE/'agent-distill-expert-block.py') or report['hyperparameters']!=B.HYPER:
    raise ValueError('Expert graph/hyperparameters changed')
  digest=hashlib.sha256()
  for name,value in sorted(seed.items()):digest.update(name.encode());digest.update(value.tobytes())
  if context['seed_sha256']!=digest.hexdigest():raise ValueError('Expert seed changed')
  if context['manifest_sha256']!=report['manifest']['manifest_sha256']:raise ValueError('Training manifest mismatch')
  arrays,metadata=S.load(root,context,shapes)
  if metadata!=report['metadata'] or metadata['step']!=25:raise ValueError('Committed metadata differs from completed report')
  checks=metadata['validation']
  expected_validation=[r['id'] for r in records if r['split']=='validation']
  if [r['step'] for r in checks]!=[0,25] or any(r['samples']!=expected_validation for r in checks):
    raise ValueError('Missing independent validation selection')
  for row in checks:
    if not row['samples'] or any(not name.startswith('validation-') for name in row['samples']):
      raise ValueError('Wrong validation samples')
    if len(row['samples'])!=len(row['losses']) or not np.isfinite(row['losses']).all() or min(row['losses'])<0:
      raise ValueError('Invalid validation losses')
    if not np.isclose(row['mean'],np.mean(row['losses']),rtol=0,atol=1e-12):raise ValueError('Invalid validation mean')
  winner=min(checks,key=lambda row:row['mean'])
  if (metadata['best_step'],metadata['best_validation'])!=(winner['step'],winner['mean']):
    raise ValueError('Best checkpoint was not selected by independent validation')
  weights,best=A.read_best(root,metadata,shapes)
  if report['best_sha256']!=best['sha256']:raise ValueError('Best checksum differs from completed report')
  for name,value in weights.items():
    expected=seed[name] if winner['step']==0 else arrays['weight/'+name]
    np.testing.assert_array_equal(value.view(np.uint32),expected.view(np.uint32))
  return weights,dict(root=str(root),report_sha256=report_sha,best_sha256=best['sha256'],selected_step=winner['step'],
    selected_optimizer_updates=winner['step'],completed_pilot_updates=25,expert_trained=winner['step']>0,
    validation_mean=winner['mean'],validation_improved=winner['mean']<checks[0]['mean'],master_dtype='float32',
    selection='original BF16 seed retained' if winner['step']==0 else 'independent validation selected update25',
    manifest_sha256=context['manifest_sha256'],training_conditioning='full detached teacher KV',
    evaluation_conditioning='current student language KV and own Euler actions',quality_qualified=False)


def array_fingerprint(array,bf16=False):
  import numpy as np
  stream=io.BytesIO();np.save(stream,array,allow_pickle=False)
  return dict(sha256=hashlib.sha256(stream.getvalue()).hexdigest(),shape=list(array.shape),dtype=str(array.dtype),bf16_bits=bf16)


def capture_tensor(tensor):
  import numpy as np
  from tinygrad import dtypes
  bf16=tensor.dtype==dtypes.bfloat16
  raw=tensor.contiguous().bitcast(dtypes.uint16).numpy().copy() if bf16 else tensor.numpy().copy()
  numeric=(raw.astype(np.uint32)<<16).view(np.float32) if bf16 else raw
  return numeric,array_fingerprint(raw,bf16)


@contextmanager
def rollout(model,compressed):
  """Observe only actual expert calls and compact slot12. Never replace any input/cache."""
  old={name:getattr(model,name) for name in ('expert_step','text_layer','trajectory')}
  state=dict(steps=[],final_actions=None);active=[None]
  def text(self,x,cos,sin,prefix,cache=None,mask=None,cache_pos=None):
    index=int(prefix.rsplit('.',1)[1]) if prefix.startswith('expert.expert.layers.') else None
    if index==12:
      if active[0] is None:raise ValueError('Expert group outside current rollout')
      row=state['steps'][active[0]]
      values=dict(x=x,cos=cos,sin=sin,context_k=cache[0],context_v=cache[1],mask=mask)
      row['conditioning']={name:capture_tensor(value)[1] for name,value in values.items()}
      import numpy as np
      mask_values=capture_tensor(mask)[0]
      if mask_values.shape!=(2368,):raise ValueError('Expected full context/action mask')
      valid=int(np.sum(mask_values[:2304]==0))
      expected=np.full(2368,-np.inf,np.float32);expected[:valid]=0;expected[2304:]=0
      np.testing.assert_array_equal(mask_values,expected)
      row['context_valid_length']=valid
    y=old['text_layer'](x,cos,sin,prefix,cache,mask,cache_pos)
    if index==(12 if compressed else 15):
      numeric,fingerprint=capture_tensor(y)
      state['steps'][active[0]].update(group_output=fingerprint,_group_output=numeric)
    return y
  def expert(self,actions,timestep,cos,sin,flat_cache,mask):
    if active[0] is not None:raise ValueError('Nested rollout')
    i=len(state['steps']);active[0]=i
    a,_=capture_tensor(actions);t,_=capture_tensor(timestep)
    state['steps'].append(dict(actions=a.tolist(),timestep=t.tolist(),cache_layers=len(flat_cache)//2))
    try:
      velocity=old['expert_step'](actions,timestep,cos,sin,flat_cache,mask)
      v,_=capture_tensor(velocity);state['steps'][i]['velocity']=v.tolist()
      return velocity
    finally:active[0]=None
  def trajectory(self,*args,**kwargs):
    result=old['trajectory'](*args,**kwargs)
    state['final_actions']=capture_tensor(result)[0].tolist();return result
  for name,fn in (('text_layer',text),('expert_step',expert),('trajectory',trajectory)):
    setattr(model,name,MethodType(fn,model))
  try:yield state
  finally:
    for name,fn in old.items():setattr(model,name,fn)


def validate_rollout(state,expected_layers,noise,context_length):
  import numpy as np
  if len(state['steps'])!=3:raise ValueError('Expected three actual own Euler steps')
  np.testing.assert_array_equal(np.asarray(state['steps'][0]['actions'],np.float32),np.asarray(noise,np.float32))
  for i,row in enumerate(state['steps']):
    if row['cache_layers']!=expected_layers:raise ValueError('Wrong current cache count')
    if row['context_valid_length']!=context_length:raise ValueError('Wrong own reasoning cache length')
    expected=dict(x=[1,64,512],cos=[1,1,64,128],sin=[1,1,64,128],context_k=[1,8,2304,128],context_v=[1,8,2304,128],mask=[2368])
    if {k:v['shape'] for k,v in row['conditioning'].items()}!=expected:raise ValueError('Wrong full expert conditioning shapes')
    np.testing.assert_allclose(row['timestep'],[i/3],rtol=0,atol=1e-7)
    if np.asarray(row['actions']).shape!=(1,64,2) or np.asarray(row['velocity']).shape!=(1,64,2):raise ValueError('Wrong rollout shape')
    if not np.isfinite(row['actions']).all() or not np.isfinite(row['velocity']).all():raise ValueError('Nonfinite rollout')
    after=np.asarray(row['actions'],np.float32)+np.asarray(row['velocity'],np.float32)/np.float32(3)
    actual=state['steps'][i+1]['actions'] if i<2 else state['final_actions']
    np.testing.assert_allclose(actual,after,rtol=2e-6,atol=2e-6)


def saved_teacher(path,digest,row,settings,plan):
  """Use completed eager teacher evidence only under the same input, sources and settings."""
  import numpy as np
  if sha(path)!=digest:raise ValueError('Teacher report changed')
  report=read(path)
  if not report.get('complete') or not report.get('gpu_executed') or report.get('mode')!='capture':raise ValueError('Incomplete teacher capture')
  if report['settings']!=settings:raise ValueError('Teacher settings/noise differ')
  core=[HERE/'agent-distill-joint-parent.py',HERE/'agent-distill-joint-runtime.py',Path(plan['runtime_root'])/'model.py']
  for source in core:
    if report['source_sha256'].get(str(source))!=sha(source):raise ValueError('Teacher source changed: '+str(source))
  matches=[r for r in report['rows'] if r['id']==row['id']]
  if len(matches)!=1:raise ValueError('Teacher sample missing/duplicated')
  saved=matches[0]
  for key in ('split','route_id','query_ns','input_file_sha256','actual_prompt_tokens'):
    if saved[key]!=row[key]:raise ValueError('Teacher input differs: '+key)
  targets=saved['targets'];steps=[]
  if len(targets['expert_steps'])!=3 or any(r['layer_order']!=list(range(28)) for r in targets['expert_steps']):
    raise ValueError('Teacher depth/solver changed')
  B=module('composed_target_reader',HERE/'agent-distill-expert-block.py')
  for i in range(3):
    values={name:B.tensor_array(targets['tensors'][f'expert{i}.'+name],shape)
            for name,shape in (('actions',(1,64,2)),('velocity',(1,64,2)),('timestep',(1,)))}
    conditioning={name:{k:item[k] for k in ('sha256','shape','dtype','bf16_bits')}
                  for name in CONDITIONS for item in [targets['tensors'][f'expert{i}.group12.'+name]]}
    y=targets['tensors'][f'expert{i}.group12.y']
    steps.append(dict(**{k:v.tolist() for k,v in values.items()},conditioning=conditioning,cache_layers=28,
                      context_valid_length=saved['teacher']['prompt_tokens']+saved['teacher']['generated_tokens'],
                      group_output={k:y[k] for k in ('sha256','shape','dtype','bf16_bits')},_output_file=y))
  final=np.asarray(steps[-1]['actions'],np.float32)+np.asarray(steps[-1]['velocity'],np.float32)/np.float32(3)
  return dict(prediction=saved['teacher'],rollout=dict(steps=steps,final_actions=final.tolist(),
                 final_actions_source='derived from captured final actions and velocity; trajectory output is original'),
               vision_input_keys=saved['vision_input_keys'],
              reused=True,report_sha256=digest,input_and_settings_equal=True,
              source_equality='original capture parent, joint runtime, frozen model; selected frozen modules checked against original plan',
              exactness='same eager arithmetic and inputs; reuse is diagnostic, not a performance comparison')


def compare(teacher,candidate):
  import numpy as np
  a,b=(value['prediction'] for value in (teacher,candidate))
  xyz=np.asarray(a['xyz'])-np.asarray(b['xyz']);rot=np.asarray(a['rot'])-np.asarray(b['rot'])
  distances=np.linalg.norm(xyz,axis=-1)
  result=dict(tokens_equal=a['generated_token_ids']==b['generated_token_ids'],reasoning_equal=a['reasoning']==b['reasoning'],
    xyz_mean_distance_m=float(distances.mean()),xyz_final_distance_m=float(distances[:,-1].mean()),
    xyz_max_absolute_m=float(np.abs(xyz).max()),rotation_max_absolute=float(np.abs(rot).max()),euler=[])
  for old,new in zip(teacher['rollout']['steps'],candidate['rollout']['steps'],strict=True):
    matched=old['conditioning']==new['conditioning']
    conditional=dict(matched=matched,differing_tensors=[k for k in CONDITIONS if old['conditioning'][k]!=new['conditioning'][k]],
      reason='all actual block conditioning tensors equal' if matched else 'actual conditioning differs; residual reconstruction is not matched')
    if matched:
      if '_group_output' in old:target=old['_group_output']
      else:
        B=module('composed_conditional_reader',HERE/'agent-distill-expert-block.py')
        target=B.tensor_array(old['_output_file'],(1,64,512))
      conditional['residual_mse']=float(np.mean((target-new['_group_output'])**2))
    result['euler'].append(dict(velocity_max_absolute=float(np.abs(np.asarray(old['velocity'])-np.asarray(new['velocity'])).max()),
      action_state_max_absolute=float(np.abs(np.asarray(old['actions'])-np.asarray(new['actions'])).max()),conditional_block=conditional))
  return result


def public(value):
  if isinstance(value,dict):return {k:public(v) for k,v in value.items() if not k.startswith('_')}
  if isinstance(value,list):return [public(v) for v in value]
  return value


def execute_prediction(options,plan,ready,language,expert,expert_receipt,*,cpu_mock=False,initialize=None):
  """The CPU integration test injects arithmetic components at initialize, after the actual launcher."""
  import numpy as np
  from tinygrad.helpers import Context
  P=module('composed_parent_entry',HERE/'agent-distill-joint-parent.py')
  R=module('composed_joint_adapter',HERE/'agent-distill-joint-runtime.py')
  row,prepared,packed=ready
  outcome=dict(complete=False,gpu_executed=not cpu_mock,cpu_integration_mock=cpu_mock,quality_qualified=False,
    timings_valid_for_throughput=False,sample_id=row['id'],expert=expert_receipt,
    input={key:row[key] for key in ('split','route_id','query_ns','input_file_sha256','actual_prompt_tokens')},
    language=dict(checkpoint_sha256=R.PILOT_SHA,selected_step=100,master_dtype='float32'),
    source_sha256={str(p):sha(p) for p in HERE.glob('agent-distill-composed-*.py')})
  def pipeline(value,args):
    components=(initialize or value.initialize)(args,sources=outcome['source_sha256'],source_dir=Path(plan['runtime_root'])/'experiments')
    model=components.model
    if not cpu_mock and components.Device[components.Device.DEFAULT].arch!='gfx1200':raise ValueError('Expected Chestnut gfx1200')
    settings={k:components.report[k] for k in SETTINGS}
    outcome['settings']=settings
    padded,pack=components.pad_inputs(prepared,packed,2048)
    assert padded['actual_prompt_length']==row['actual_prompt_tokens']
    def predict(compressed):
      if model.cache is not None or components.decoder.history or components.vision_cache.cache:raise ValueError('Arm did not start fresh')
      components.vision_cache.prepare(padded)
      inputs=components.input_class(components.settings,components.config,padded,pack)
      if args.selective_pixels:inputs.selective_cache=components.vision_cache
      with rollout(model,compressed) as observed:
        prediction=components.runner(model,components.settings,components.config,components.tokenizer,padded,pack,seed=0,inputs=inputs)
      if prediction['prefill_tokens']!=2048 or prediction['generated_tokens']!=len(prediction['generated_token_ids']):
        raise ValueError('Changed full prefill or incomplete reasoning result')
      validate_rollout(observed,25 if compressed else 28,components.settings.initial_noise,
                       prediction['prompt_tokens']+prediction['generated_tokens'])
      if len(model.cache)!=(25 if compressed else 28):raise ValueError('Wrong completed cache count')
      for key,shape in (('xyz',(1,64,3)),('rot',(1,64,3,3))):
        if np.asarray(prediction[key]).shape!=shape or not np.isfinite(prediction[key]).all():raise ValueError('Invalid complete trajectory')
      image_keys=[dict(grid=[int(n) for n in key[0]],pixel_sha256=key[1].hex()) for key in components.vision_cache.keys]
      if len(image_keys)!=10:raise ValueError('Expected all ten calibrated history images')
      return dict(prediction=prediction,rollout=observed,reused=False,started_with_empty_caches=True,vision_input_keys=image_keys)
    # Deliberate eager diagnostic: only small rollout outputs and one group's fingerprints are read back.
    with Context(JIT=0):
      R.reset_state(model,components.decoder,components.vision_cache)
      if options.rerun_teacher:
        teacher=predict(False)
      else:
        teacher=saved_teacher(options.teacher_report,options.teacher_sha,row,settings,plan)
        validate_rollout(teacher['rollout'],28,components.settings.initial_noise,
                         teacher['prediction']['prompt_tokens']+teacher['prediction']['generated_tokens'])
      outcome['teacher']=teacher
      with R.joint_pilot(model,language,expert,allow_untrained_expert=True,
                         decoder=components.decoder,vision_cache=components.vision_cache) as ignored_untrained_receipt:
        # Create a new receipt from the verified selection. Never publish the old seed-only receipt as trained.
        outcome['adapter']=dict(mapping=R.mapping(),language_runtime_dtype=str(model.dtype),expert_runtime_dtype=str(model.dtype),
          language_namespace='agent_joint.language.12',expert_namespace='agent_joint.expert.12',
          expert_selected_step=expert_receipt['selected_step'],expert_trained=expert_receipt['expert_trained'],
          original_seed_selected=expert_receipt['selected_step']==0,teacher_cache_injected=False,quality_qualified=False)
        outcome['candidate']=predict(True)
      if teacher['vision_input_keys']!=outcome['candidate']['vision_input_keys']:
        raise ValueError('Teacher and student calibrated image history differs')
      if model.cache is not None or model.config['expert_config']['llm_config']['num_hidden_layers']!=28:
        raise ValueError('Adapter did not restore original model')
      outcome['comparison']=compare(teacher,outcome['candidate'])
      outcome.update(complete=True,adapter_restored=True)
  try:P.launch(plan,options.output,[row['sample']],pipeline)
  except BaseException as error:
    outcome['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:options.output.write_text(json.dumps(public(outcome),indent=2,allow_nan=False)+'\n')
  return public(outcome)


def checked_plan(root):
  """Keep the selected protocol; only private compilation/temp locations change."""
  plan=read(PLAN);runtime=Path(plan['runtime_root'])
  names=['model.py','runtime.py','benchmark.py','inputs.py','amd_gemm.py','amd_attention.py','amd_decode.py']
  names += ['experiments/'+name+'.py' for name in ('benchmark-pipeline','benchmark-fast','agent-speculation-run','agent-speculation-policy',
    'agent-speculation-prefill','agent-attention-rowsoftmax','padded-prefill','block-decode','draft-decode','lookup-decode','reduced-vocab',
       'expert-kernels','restore-precision','bf16-linear','fast-quantize','quantize-reduction','quantize-packed','online-attention','fused-norm',
    'projector-kernels','batched-vision','vision-cache','compact-inputs','fast-inputs','usb-readback','wgp_compiler')]
  for name in names:
    if sha(runtime/name)!=plan['runtime_files_sha256'][name]:raise ValueError('Selected frozen source changed: '+name)
  cache=root/'agent-distill-composed-kernels.db';temporary=root/'agent-distill-composed-tmp';temporary.mkdir()
  seed=plan['compiler_cache_seed']
  if sha(seed['path'])!=seed['sha256']:raise ValueError('Compiler cache seed changed')
  shutil.copyfile(seed['path'],cache)
  invocation=read(plan['invocation_path']);invocation['environment'].update(CACHEDB=str(cache),TMPDIR=str(temporary))
  path=root/'agent-distill-composed-invocation.json';path.write_text(json.dumps(invocation,indent=2)+'\n')
  plan['invocation_path']=str(path)
  for key,value in invocation['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                    CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  return plan


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--sample-id',choices=tuple(TEACHERS),required=True)
  parser.add_argument('--expert-root',type=Path,required=True);parser.add_argument('--expert-report-sha',required=True)
  parser.add_argument('--rerun-teacher',action='store_true')
  parser.add_argument('--execute-gpu',action='store_true')
  args=parser.parse_args();args.output=args.output.resolve();root=args.output.parent
  if args.output.name!='report.json' or not root.is_relative_to(WORK) or not root.is_dir() or args.output.exists():
    raise ValueError('Use a fresh normal-wrapper output directory and report.json on the 500 GB workspace')
  if any(root.glob('agent-distill-composed-*')):raise ValueError('Output already contains composed artifacts')
  expert,receipt=selected_expert(args.expert_root,args.expert_report_sha)
  args.teacher_report,args.teacher_sha=TEACHERS[args.sample_id]
  plan=checked_plan(root)
  if not args.execute_gpu:os.environ['DEV']='CPU'
  P=module('composed_parent_prepare',HERE/'agent-distill-joint-parent.py')
  rows=P.selected_rows(P.INPUTS,[args.sample_id]);ready=P.prepare(plan,rows)[0]
  R=module('composed_load_language',HERE/'agent-distill-joint-runtime.py');language=R.load_pilot(P.PILOT)
  sys.path.insert(0,str(BASE/'tinygrad'))
  if not args.execute_gpu:
    args.output.write_text(json.dumps(dict(complete=True,gpu_executed=False,preflight_only=True,
      expert=receipt,language_sha256=R.PILOT_SHA,sample_id=args.sample_id),indent=2)+'\n')
    P.no_devices();return
  execute_prediction(args,plan,ready,language,expert,receipt)


if __name__=='__main__':main()
