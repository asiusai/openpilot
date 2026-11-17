"""Capture missing selected own suffix K/V and fixed-teacher-state native velocity references.

One offline teacher-token query, followed by three independent native expert evaluations.
No fitting or optimizer. Use the existing normal parent ACL/global-lock wrapper.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import MethodType
import importlib.util

sys.dont_write_bytecode=True;HERE=Path(__file__).resolve().parent
def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value
M=module('motion_capture_core',HERE/'agent-distill-motion-objective-core.py')
if M.sha(HERE/'agent-distill-token-control-run.py')!='f146816b0270e3c99a35e023fe03a1a3b0b94b1e8f0584b4d3b2965740f4dab7':raise ValueError('Reviewed token-control source changed')
T=module('motion_token_control',HERE/'agent-distill-token-control-run.py')
BASE_EXECUTE=T.execute
B=module('motion_capture_arrays',HERE/'agent-distill-expert-block.py')


def teacher_conditions(refs):
  import numpy as np
  p=T.E.P;path=M.WORK/'agent-distill-objective-legacy-train-v2/report.json';row=p.target(path,p.TRAIN_SHA,'train')
  if row['teacher']['generated_token_ids']!=refs['tokens'] or row['input_file_sha256']!=refs['provenance']['input']['input_file_sha256']:
    raise ValueError('Teacher state/token/input alignment differs')
  expected=dict(actions=(1,64,2),timestep=(1,),cos=(1,1,64,128),sin=(1,1,64,128),mask=(2368,),velocity=(1,64,2))
  steps=[{name:B.tensor_array(row['targets']['tensors'][f'expert{i}.{name}'],shape,name=='mask') for name,shape in expected.items()} for i in range(3)]
  for i,s in enumerate(steps):
    np.testing.assert_allclose(s['timestep'],[i/3],rtol=0,atol=1e-7)
    np.testing.assert_array_equal(s['mask'],np.r_[np.zeros(2047),np.full(257,-np.inf),np.zeros(64)])
    if i:np.testing.assert_allclose(s['actions'],steps[i-1]['actions']+steps[i-1]['velocity']/3,rtol=2e-6,atol=2e-6)
  return steps,dict(path=str(path),sha256=p.TRAIN_SHA)


def collect(model,steps,save,*,cpu_mock=False):
  """Called after ordinary own Euler observation ends, while the joint adapter remains active."""
  from tinygrad import Tensor,dtypes
  if len(model.cache)!=25 or model.config['expert_config']['llm_config']['num_hidden_layers']!=25:raise ValueError('Need current paired25 candidate')
  state=dict(complete=False,context_slots=M.SLOTS,suffix_sources=M.SOURCES,tensors={},weight_storage={},steps=[],
    fixed_teacher_actions=True,free_running_training=False,trainable_roles=['expert12'],language_frozen=True,
    own_language_context=True,teacher_cache_injected=False,student_value_detached=False,
    frozen_arithmetic='FP32 dequantized deployed weight constants; FP32 activation surrogate, without runtime FP8/BF16 rounding')
  def keep(name,value):
    if name in state['tensors']:raise ValueError('Duplicate motion array')
    state['tensors'][name]=save(name,value)
  for slot in M.SLOTS:
    if model.cache[slot].shape!=(2,1,8,2304,128):raise ValueError('Context was truncated')
    keep(f'context.{slot}',model.cache[slot])
  for source in M.SOURCES:
    for name,shape in M.SHAPES.items():
      full=f'expert.expert.layers.{source}.'+name;weight=model.weights[full]
      if weight.shape!=shape:raise ValueError('Frozen suffix geometry changed')
      scale=model.weight_scales.get(full);value=weight.float() if scale is None else weight.float()*scale.float()
      key=f'layer.{source}.{name}';keep(key,value)
      state['weight_storage'][key]=dict(runtime_dtype=str(weight.dtype),dequantized=scale is not None)
  for name,full in [('norm','expert.expert.norm.weight'),('head_weight','expert.action_out_proj.weight'),('head_bias','expert.action_out_proj.bias')]:
    weight=model.weights[full];scale=model.weight_scales.get(full)
    keep(name,weight.float() if scale is None else weight.float()*scale.float())
    state['weight_storage'][name]=dict(runtime_dtype=str(weight.dtype),dequantized=scale is not None)
  original=model.text_layer;active=[None]
  def layer(self,x,c,s,prefix,cache=None,mask=None,cache_pos=None):
    if not prefix.startswith('expert.expert.layers.'):raise ValueError('Unexpected language execution during fixed-state expert evaluation')
    slot=int(prefix.rsplit('.',1)[1]);i=active[0];state['steps'][i]['compact_layer_order'].append(slot)
    if slot==12:
      for name,value in (('x',x),('cos',c),('sin',s),('mask',mask)):keep(f'step.{i}.{name}',value)
    y=original(x,c,s,prefix,cache,mask,cache_pos)
    if slot==12:keep(f'step.{i}.student_y',y)
    return y
  model.text_layer=MethodType(layer,model)
  device=model.cache[0].device
  flat=[value[i] for value in model.cache for i in range(2)]
  try:
    for i,step in enumerate(steps):
      active[0]=i;state['steps'].append(dict(compact_layer_order=[]))
      tensors={name:Tensor(value.copy(),device=device).cast(model.dtype if name in ('cos','sin') else dtypes.float32).realize() for name,value in step.items()}
      for name in ('actions','timestep'):keep(f'step.{i}.{name}',tensors[name])
      keep(f'step.{i}.target_velocity',tensors['velocity'])
      velocity=model.expert_step(tensors['actions'],tensors['timestep'],tensors['cos'],tensors['sin'],flat,tensors['mask'])
      keep(f'step.{i}.native_velocity',velocity)
      if state['steps'][i]['compact_layer_order']!=list(range(25)):raise ValueError('Incomplete native expert evaluation')
  finally:model.text_layer=original
  state.update(complete=True,action_rows=64,capacity=2304,valid_length=2047,qualification_targets_only=True,cpu_mock=cpu_mock)
  return state


def execute(args,plan,ready,paired,receipt,refs,*,initialize=None,cpu_mock=False):
  import numpy as np
  from tinygrad import dtypes
  control=M.WORK/'agent-distill-token-control-train-v1/report.json'
  if M.sha(control)!=M.CONTROL_SHA:raise ValueError('Completed native token control changed')
  completed=M.read(control)
  if not all(completed[k] for k in ('complete','gpu_executed','adapter_restored','decoder_restored')):raise ValueError('Incomplete causal control')
  if completed['token_provenance']!=refs['provenance'] or completed['joint_selection']!=receipt:raise ValueError('Causal control selection/input mismatch')
  steps,target_source=teacher_conditions(refs);directory=args.output.parent/'motion-targets';directory.mkdir()
  if __import__('shutil').disk_usage(directory).free<M.estimate()['workspace_free_required_bytes']:raise ValueError('Insufficient target workspace')
  def save(name,tensor):
    bf16=tensor.dtype==dtypes.bfloat16
    array=tensor.contiguous().bitcast(dtypes.uint16).numpy().copy() if bf16 else tensor.float().numpy().copy()
    numbers=(array.astype(np.uint32)<<16).view(np.float32) if bf16 else array
    if name.endswith('.mask'):
      if not np.all((numbers==0)|np.isneginf(numbers)):raise ValueError('Invalid mask')
    elif not np.isfinite(numbers).all():raise ValueError('Nonfinite captured motion value')
    path=directory/(name+'.npy');np.save(path,array,allow_pickle=False)
    return dict(path=str(path),sha256=M.sha(path),shape=list(array.shape),dtype=str(array.dtype),bf16_bits=bf16)
  original=T.C.rollout;captured={}
  @contextmanager
  def observed(model,compressed):
    with original(model,compressed) as state:yield state
    # The observer is now removed, so these independent fixed-state calls cannot enter the three-step own rollout.
    captured.update(collect(model,steps,save,cpu_mock=cpu_mock))
  T.C.rollout=observed
  try:return BASE_EXECUTE(args,plan,ready,paired,receipt,refs,initialize=initialize,cpu_mock=cpu_mock)
  finally:
    T.C.rollout=original
    if args.output.exists():
      report=M.read(args.output)
      if report.get('complete') and not cpu_mock and report['forced']['prediction']['xyz']!=completed['forced']['prediction']['xyz']:
        report.update(complete=False,failure=dict(type='ValueError',message='Repeated native causal control trajectory changed'))
      captured.update(sample_id=ready[0]['id'],selected_checkpoint_sha256=M.SELECTED_SHA,token_control_report_sha256=M.CONTROL_SHA,
        teacher_condition_source=target_source,optimizer_updates=0,checkpoint_selection_allowed=False)
      report.update(motion_capture=captured,motion_objective_estimate=M.estimate())
      report['source_sha256'].update({str(p):M.sha(p) for p in HERE.glob('agent-distill-motion-objective-*.py')})
      args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
      if not report.get('complete') or not captured.get('complete'):raise ValueError('Motion capture incomplete; see retained report')


def main():
  # Keep the actual selected checkpoint/input CLI and launcher. Replace only the execution callback.
  namespace=T.main.__globals__;original=namespace['execute'];namespace['execute']=execute
  try:T.main()
  finally:namespace['execute']=original


if __name__=='__main__':main()
