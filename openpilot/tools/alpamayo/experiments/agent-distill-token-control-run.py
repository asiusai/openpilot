"""OFFLINE causal diagnostic: force recorded teacher tokens through the selected student's own language model.

This is not target-verified speculation, autonomous inference, or a speed/quality candidate.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import MethodType

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-token-control-artifacts'
AUTONOMOUS=WORK/'agent-distill-joint-eval-train-v1/report.json'
AUTONOMOUS_SHA='2bfd7393d728e4eb034ab2fca539bf0f03aa30d18e277a21fa8fee532c5c7757'
EVAL_SHA='aef10712e1e529fc0752d994a8623f58fad838f3019533365df1d3af2a849979'
DIAGNOSTIC='offline teacher-token causal conditioning only; not autonomous or target-verified speculation'

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value
if sha(HERE/'agent-distill-joint-eval-run.py')!=EVAL_SHA:raise ValueError('Reviewed joint evaluator changed')
E=module('token_control_eval',HERE/'agent-distill-joint-eval-run.py');C=E.C
P=module('token_control_parent',HERE/'agent-distill-joint-parent.py')
R=module('token_control_adapter',HERE/'agent-distill-joint-runtime.py')


def validate_tokens(tokens,config,max_tokens,vocab_size):
  anchor=config['traj_ids']['future_start']
  if not isinstance(tokens,list) or not tokens or len(tokens)>max_tokens:raise ValueError('Invalid teacher sequence length')
  if any(type(t) is not int or not 0<=t<vocab_size for t in tokens):raise ValueError('Invalid token ID')
  if tokens[-1]!=anchor or anchor in tokens[:-1]:raise ValueError('Require exactly one terminal trajectory anchor')
  for token in tokens:
    if token in (config['vlm_config']['image_token_id'],config['vlm_config']['text_config']['eos_token_id']):raise ValueError('Non-reasoning token')
    for role in ('history','future'):
      if config['traj_ids'][role+'_id0']<=token<config['traj_ids'][role+'_id0']+config[role+'_vocab_size']:raise ValueError('Trajectory-code token')


def references(plan,row,receipt):
  import numpy as np
  if sha(AUTONOMOUS)!=AUTONOMOUS_SHA:raise ValueError('Completed autonomous comparison changed')
  report=read(AUTONOMOUS)
  if not all(report.get(k) for k in ('complete','gpu_executed','adapter_restored')) or report.get('cpu_integration_mock'):raise ValueError('Expected completed actual autonomous run')
  if report['sample_id']!='train-command-0' or report['paired_checkpoint_sha256']!=receipt['best_sha256'] or report['joint_selected_step']!=25:
    raise ValueError('Wrong autonomous candidate')
  if report['joint_selection']!=receipt or report['adapter']['mapping']!=R.mapping():raise ValueError('Changed candidate selection/mapping')
  if report['source_sha256'].get(str(HERE/'agent-distill-joint-eval-run.py'))!=EVAL_SHA:raise ValueError('Wrong autonomous source')
  if report['input']!={key:row[key] for key in report['input']}:raise ValueError('Autonomous input differs')
  path,digest=C.TEACHERS['train-command-0'];teacher=C.saved_teacher(path,digest,row,report['settings'],plan)
  if teacher['prediction']!=report['teacher']['prediction'] or teacher['vision_input_keys']!=report['candidate']['vision_input_keys']:
    raise ValueError('Teacher prediction/history lineage differs')
  command=read(plan['invocation_path'])['command'];noise=np.load(command[command.index('--initial-noise')+1],allow_pickle=False)
  E.teacher_preflight(plan,row)
  for arm,layers in ((teacher,28),(report['candidate'],25)):
    prediction=arm['prediction'];C.validate_rollout(arm['rollout'],layers,noise,prediction['prompt_tokens']+prediction['generated_tokens'])
    for name,shape in (('xyz',(1,64,3)),('rot',(1,64,3,3))):
      array=np.asarray(prediction[name]);assert array.shape==shape and np.isfinite(array).all()
  tokens=teacher['prediction']['generated_token_ids']
  config=read(E.BASE/'model/config.json');validate_tokens(tokens,config,report['settings']['settings']['max_tokens'],config['vlm_config']['text_config']['vocab_size'])
  assert len(tokens)==teacher['prediction']['generated_tokens']==20
  return dict(teacher=teacher,autonomous=report['candidate'],settings=report['settings'],tokens=tokens,
    provenance=dict(teacher_report=str(path),teacher_report_sha256=digest,autonomous_report=str(AUTONOMOUS),autonomous_report_sha256=AUTONOMOUS_SHA,
      input=report['input'],paired_checkpoint_sha256=receipt['best_sha256'],teacher_token_ids=tokens,teacher_text=teacher['prediction']['reasoning']))


@contextmanager
def force_tokens(decoder,model,tokens,prompt,trace):
  """Replace only token choice. Keep genuine embeddings, native single-row language, rotary slices and cache writes."""
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp
  original=decoder.generate
  expected=list(tokens)
  def generate(self,current,args,config,tokenizer,logits,dc,ds,actual_prompt):
    if current is not model or actual_prompt!=prompt or trace:raise ValueError('Stale/repeated forced decode')
    embedding=model.weights['vlm.model.language_model.embed_tokens.weight']
    validate_tokens(expected,model.config,args.max_tokens,embedding.shape[0])
    if len(model.cache)!=25 or any(t.shape!=(2,1,8,2304,128) for t in model.cache):raise ValueError('Expected fresh paired25 full-capacity caches')
    if dc.shape!=ds.shape or dc.shape[2]<len(expected) or prompt+len(expected)>2304:raise ValueError('Insufficient native rotary/cache capacity')
    if model.padded_prefill.last_prompt!=prompt or model.prompt_length!=2048:raise ValueError('Wrong original padded prefill')
    if tokenizer.decode(expected)!=trace_text[0]:raise ValueError('Teacher token/text mismatch')
    for begin in range(0,len(expected),6):
      # Match the original six-token fallback's bound position expressions and single-row arithmetic.
      count=min(6,args.max_tokens-begin)
      pos=UOp.variable('block_position',0,2304-count).bind(prompt+begin)
      offset=UOp.variable('block_step',0,dc.shape[2]-count).bind(begin)
      for i in range(min(count,len(expected)-begin)):
        step=begin+i;token=expected[step]
        # These predictions are observations under a forced prefix, never acceptance checks.
        own_choice=self.first_token(logits)
        token_tensor=Tensor(token,device=embedding.device,dtype=dtypes.int32).realize()
        x=embedding[token_tensor].reshape(1,1,-1)
        logits=model._language(x,dc[:,:,offset+i:offset+i+1],ds[:,:,offset+i:offset+i+1],pos+i)
        trace.append(dict(step=step,token_id=token,cache_position=prompt+step,rotary_offset=step,
          student_choice_under_forced_prefix=int(own_choice),same_choice=int(own_choice)==token))
    self.statistics.append(dict(offline_forced_tokens=len(expected),target_verified=False,autonomous=False))
    return list(expected)
  # The caller binds text from the hash-checked teacher record; it is checked against the actual tokenizer.
  trace_text=[getattr(decoder,'token_control_teacher_text',None)]
  if trace_text[0] is None:raise ValueError('Missing teacher text provenance')
  decoder.generate=MethodType(generate,decoder)
  try:yield
  finally:decoder.generate=original


def comparison(reference,current):
  import numpy as np
  old,new=reference['prediction'],current['prediction'];distances=np.linalg.norm(np.asarray(old['xyz'])-new['xyz'],axis=-1)
  a,b=E.motion(old),E.motion(new)
  result=dict(tokens_equal=old['generated_token_ids']==new['generated_token_ids'],reasoning_equal=old['reasoning']==new['reasoning'],
    reference_tokens=old['generated_tokens'],current_tokens=new['generated_tokens'],xyz_mean_distance_m=float(distances.mean()),
    xyz_final_distance_m=float(distances[:,-1].mean()),rotation_max_absolute=float(np.max(np.abs(np.asarray(old['rot'])-new['rot']))),
    vehicle_motion=dict(reference=a,current=b,speed_mean_absolute_delta_m_s=float(np.mean(np.abs(np.asarray(a['speed_m_s'])-b['speed_m_s']))),
      acceleration_mean_absolute_delta_m_s2=float(np.mean(np.abs(np.asarray(a['acceleration_m_s2'])-b['acceleration_m_s2'])))),euler=[])
  for before,after in zip(reference['rollout']['steps'],current['rollout']['steps'],strict=True):
    unequal=[key for key in C.CONDITIONS if before['conditioning'][key]!=after['conditioning'][key]]
    result['euler'].append(dict(reference_valid_context=before['context_valid_length'],current_valid_context=after['context_valid_length'],
      velocity_max_absolute=float(np.max(np.abs(np.asarray(before['velocity'])-after['velocity']))),
      action_state_max_absolute=float(np.max(np.abs(np.asarray(before['actions'])-after['actions']))),
      conditional_block=dict(matched=not unequal,differing_tensors=unequal,residual_mse_reported=False,
        kv_fingerprints_include_masked_tail=True,valid_context_only_kv_comparison_available=False,
        reason='Full-capacity conditioning fingerprints include masked tails; this diagnostic does not score internal reconstruction')))
  return result


def execute(options,plan,ready,paired,receipt,refs,*,initialize=None,cpu_mock=False):
  import numpy as np
  from tinygrad.helpers import Context
  row,prepared,packed=ready
  outcome=dict(complete=False,gpu_executed=not cpu_mock,cpu_integration_mock=cpu_mock,diagnostic=DIAGNOSTIC,
    usable_inference_configuration=False,target_verified_speculation=False,quality_qualified=False,timings_valid_for_throughput=False,
    sample_id=row['id'],token_provenance=refs['provenance'],joint_selection=receipt,token_trace=[],
    teacher=refs['teacher'],autonomous=refs['autonomous'],source_sha256={str(p):sha(p) for p in HERE.glob('agent-distill-token-control-*.py')})
  def pipeline(value,args):
    components=(initialize or value.initialize)(args,sources=outcome['source_sha256'],source_dir=Path(plan['runtime_root'])/'experiments')
    model,decoder=components.model,components.decoder
    if not cpu_mock and components.Device[components.Device.DEFAULT].arch!='gfx1200':raise ValueError('Expected Chestnut gfx1200')
    actual_settings={key:components.report[key] for key in C.SETTINGS}
    if actual_settings!=refs['settings']:raise ValueError('Actual settings differ from completed comparison')
    outcome['settings']=actual_settings
    assert model.config['vlm_config']['vision_config']['num_hidden_layers']==27
    assert model.config['expert_config']['action_space_cfg']['dt']==.1 and model.config['expert_config']['action_space_cfg']['n_waypoints']==64
    padded,pack=components.pad_inputs(prepared,packed,2048)
    if padded['actual_prompt_length']!=row['actual_prompt_tokens']:raise ValueError('Prompt changed')
    original_generate=decoder.generate
    try:
      with Context(JIT=0),R.joint_pilot(model,paired['language'],paired['expert'],allow_untrained_expert=True,decoder=decoder,vision_cache=components.vision_cache):
        if model.cache is not None or decoder.history or components.vision_cache.cache:raise ValueError('Candidate did not start fresh')
        decoder.token_control_teacher_text=refs['provenance']['teacher_text']
        with force_tokens(decoder,model,refs['tokens'],row['actual_prompt_tokens'],outcome['token_trace']):
          components.vision_cache.prepare(padded)
          inputs=components.input_class(components.settings,components.config,padded,pack)
          if args.selective_pixels:inputs.selective_cache=components.vision_cache
          with C.rollout(model,True) as observed:
            prediction=components.runner(model,components.settings,components.config,components.tokenizer,padded,pack,seed=0,inputs=inputs)
          if prediction['generated_token_ids']!=refs['tokens'] or prediction['reasoning']!=refs['provenance']['teacher_text']:raise ValueError('Forced sequence changed')
          if prediction['prefill_tokens']!=2048 or prediction['generated_tokens']!=len(refs['tokens']):raise ValueError('Changed prefill/sequence length')
          C.validate_rollout(observed,25,components.settings.initial_noise,prediction['prompt_tokens']+len(refs['tokens']))
          for name,shape in (('xyz',(1,64,3)),('rot',(1,64,3,3))):
            array=np.asarray(prediction[name]);assert array.shape==shape and np.isfinite(array).all()
          keys=[dict(grid=[int(n) for n in key[0]],pixel_sha256=key[1].hex()) for key in components.vision_cache.keys]
          if len(keys)!=10 or keys!=refs['teacher']['vision_input_keys'] or keys!=refs['autonomous']['vision_input_keys']:raise ValueError('Calibrated image history differs')
          outcome['forced']=dict(prediction=prediction,rollout=observed,vision_input_keys=keys,started_with_empty_caches=True,
            autonomous=False,forced_tokens_only=True,teacher_cache_injected=False,teacher_actions_injected=False)
          outcome['adapter']=dict(mapping=R.mapping(),runtime_dtype=str(model.dtype),selected_step=receipt['selected_step'],vision_layers=27)
      assert decoder.generate==original_generate and model.cache is None and not decoder.history and not components.vision_cache.cache
      assert model.config['vlm_config']['text_config']['num_hidden_layers']==model.config['expert_config']['llm_config']['num_hidden_layers']==28
      outcome['comparison']=dict(teacher_vs_forced=comparison(refs['teacher'],outcome['forced']),
        autonomous_vs_forced=comparison(refs['autonomous'],outcome['forced']),teacher_vs_autonomous=comparison(refs['teacher'],refs['autonomous']))
      outcome.update(complete=True,adapter_restored=True,decoder_restored=True)
    finally:
      if hasattr(decoder,'token_control_teacher_text'):del decoder.token_control_teacher_text
  try:P.launch(plan,options.output,[row['sample']],pipeline)
  except BaseException as error:
    outcome['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:options.output.write_text(json.dumps(C.public(outcome),indent=2,allow_nan=False)+'\n')
  return outcome


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--offline-teacher-token-control',action='store_true',required=True)
  args=parser.parse_args();args.output=args.output.resolve();root=args.output.parent
  if args.output.name!='report.json' or not root.is_relative_to(WORK) or not root.is_dir() or args.output.exists():raise ValueError('Fresh normal-wrapper output required')
  if any(root.glob('agent-distill-composed-*')):raise ValueError('Existing runtime artifacts')
  paired,receipt=E.selected_joint(E.ACTUAL_ROOT,E.ACTUAL_REPORT_SHA)
  plan=C.checked_plan(root)
  if not args.execute_gpu:os.environ['DEV']='CPU'
  row=P.selected_rows(P.INPUTS,['train-command-0'])[0];ready=P.prepare(plan,[row])[0]
  sys.path.insert(0,str(E.BASE/'tinygrad'));refs=references(plan,row,receipt)
  if not args.execute_gpu:
    P.no_devices();args.output.write_text(json.dumps(dict(complete=True,gpu_executed=False,preflight_only=True,diagnostic=DIAGNOSTIC,
      usable_inference_configuration=False,joint_selection=receipt,token_provenance=refs['provenance']),indent=2)+'\n');return
  execute(args,plan,ready,paired,receipt,refs)


if __name__=='__main__':main()
