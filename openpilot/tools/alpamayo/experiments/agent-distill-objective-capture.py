"""Complete teacher token-row targets for a connected local joint objective.

Use capture_prediction() with the cohort owner's existing prepared tuple and components.
The standalone parent command selects one existing approved input; it never copies pixels.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
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
FORBIDDEN=('0000008e--321b2ddc03','00000090--1cf3f7d926','sealed_test')
ROUTES={'train':'00000071--858232e52d','validation':'00000073--41474e402a'}


def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def raw(tensor):
  import numpy as np
  from tinygrad import dtypes
  bf16=tensor.dtype==dtypes.bfloat16
  array=tensor.contiguous().bitcast(dtypes.uint16).numpy().copy() if bf16 else tensor.numpy().copy()
  if bf16:
    if not np.isfinite((array.astype(np.uint32)<<16).view(np.float32)).all():raise ValueError('Nonfinite BF16 target')
  elif not np.all(np.isfinite(array)|np.isneginf(array)):raise ValueError('Invalid target')
  return array,bf16


def fingerprint(value):return hashlib.sha256(value.tobytes()).hexdigest()


@contextmanager
def aligned_targets(model,save):
  """Observe every current write. Later accepted decode writes replace provisional draft rows."""
  import numpy as np
  if model.config['vlm_config']['text_config']['num_hidden_layers']!=28 or model.config['expert_config']['llm_config']['num_hidden_layers']!=28:
    raise ValueError('Need the unchanged full teacher')
  if model.config['vlm_config']['vision_config']['num_hidden_layers']!=27:raise ValueError('Changed vision teacher')
  originals={k:getattr(model,k) for k in ('text_layer','expert_step')}
  state=dict(language_writes=[],expert=[],tensors={},complete=False)
  canvases={};covered=np.full(2304,-1,np.int32);pending=[None];active=[None]
  def merge(name,tensor,start,rows):
    value,bf16=raw(tensor);axis=2 if name in ('cos','sin') else 1
    if value.shape[axis]!=rows:raise ValueError('Rotary/query row alignment changed')
    expected=(1,1,rows,128) if axis==2 else (1,rows,2048)
    if value.shape!=expected:raise ValueError('Wrong full-width language shape')
    if name not in canvases:
      shape=list(value.shape);shape[axis]=2304;canvases[name]=(np.zeros(shape,value.dtype),bf16)
    target,kind=canvases[name]
    if kind!=bf16 or target.dtype!=value.dtype:raise ValueError('Target dtype changed')
    index=[slice(None)]*value.ndim;index[axis]=slice(start,start+rows);target[tuple(index)]=value
  def keep(name,tensor):
    if name in state['tensors']:raise ValueError('Duplicate target')
    value,bf16=raw(tensor);state['tensors'][name]=save(name,value,bf16)
  def layer(self,x,cos,sin,prefix,cache=None,mask=None,cache_pos=None):
    language=prefix.startswith('vlm.model.language_model.layers.')
    if not language and not prefix.startswith('expert.expert.layers.'):raise ValueError('Unexpected student layer')
    index=int(prefix.rsplit('.',1)[1])
    if language:
      if index==0:
        start=cache_pos if isinstance(cache_pos,int) else cache_pos.unbind()[1]
        start=int(start);rows=x.shape[1]
        if not 0<=start<start+rows<=2304:raise ValueError('Invalid cache write')
        pending[0]=dict(start=start,rows=rows,layer_order=[])
      write=pending[0]
      if write is None or index!=len(write['layer_order']):raise ValueError('Incomplete teacher language call')
      write['layer_order'].append(index)
      if index==12:
        for name,tensor in (('x',x),('cos',cos),('sin',sin)):merge(name,tensor,write['start'],write['rows'])
    else:
      if active[0] is None:raise ValueError('Expert layer outside actual Euler step')
      step=state['expert'][active[0]];step['layer_order'].append(index)
      if index==12:
        for name,tensor in (('x',x),('cos',cos),('sin',sin),('mask',mask)):keep(f'expert{active[0]}.'+name,tensor)
        step['teacher_context_fingerprints']=[fingerprint(raw(t)[0]) for t in cache]
    y=originals['text_layer'](x,cos,sin,prefix,cache,mask,cache_pos)
    if language and index==15:
      merge('y',y,write['start'],write['rows'])
    if language and index==27:
      covered[write['start']:write['start']+write['rows']]=len(state['language_writes'])
      state['language_writes'].append(write);pending[0]=None
    if not language and index==15:keep(f'expert{active[0]}.y',y)
    return y
  def expert_step(self,actions,timestep,cos,sin,flat_cache,mask):
    if active[0] is not None:raise ValueError('Nested Euler step')
    i=len(state['expert']);active[0]=i;state['expert'].append(dict(layer_order=[]))
    keep(f'expert{i}.actions',actions);keep(f'expert{i}.timestep',timestep)
    try:
      velocity=originals['expert_step'](actions,timestep,cos,sin,flat_cache,mask)
      keep(f'expert{i}.velocity',velocity);return velocity
    finally:active[0]=None
  def finish(prediction,prepared,packed):
    prompt=int(prepared['actual_prompt_length']);generated=prediction['generated_token_ids'];valid=prompt+len(generated)
    if not 0<prompt<valid<=2304 or not 0<len(generated)<=64 or prediction['generated_tokens']!=len(generated):raise ValueError('Invalid complete teacher tokens')
    if prediction['prompt_tokens']!=prompt or generated[-1]!=model.config['traj_ids']['future_start']:raise ValueError('Missing aligned trajectory anchor')
    if prediction['prefill_tokens']!=2048 or packed[0].shape!=(1,2048):raise ValueError('Changed full prefill')
    if pending[0] is not None or not state['language_writes'] or np.any(covered[:valid]<0):raise ValueError('Missing actual accepted language rows')
    if len(state['expert'])!=3 or any(r['layer_order']!=list(range(28)) for r in state['expert']):raise ValueError('Incomplete teacher Euler capture')
    if len(model.cache)!=28 or model.cache[12].shape!=(2,1,8,2304,128):raise ValueError('Wrong current teacher cache')
    for name,(array,bf16) in canvases.items():state['tensors']['language.'+name]=save('language.'+name,array,bf16)
    context=[]
    for i,name in enumerate(('k','v')):
      value,bf16=raw(model.cache[12][i]);context.append(fingerprint(value))
      state['tensors']['language.'+name]=save('language.'+name,value,bf16)
    if any(row['teacher_context_fingerprints']!=context for row in state['expert']):raise ValueError('Euler context differs from final actual teacher sequence')
    state.update(complete=True,prompt_length=prompt,valid_length=valid,capacity=2304,
      valid_token_ids=packed[0][0,:prompt].astype(int).tolist()+generated,
      valid_rows_last_write=covered[:valid].astype(int).tolist(),
      target_meanings=dict(language='input to layer12, output after layer15, final current layer12 post-RoPE K/V',
        expert='input to expert12, output after expert15 at each actual full-teacher Euler state'),
      teacher_forcing='Replay this complete teacher token sequence; current student-produced KV replaces teacher KV only inside the differentiable objective',
      padding='Unwritten rows are marked invalid; no valid row may be synthesized',quality_qualified=False)
    return state
  state['finish']=finish
  model.text_layer=MethodType(layer,model);model.expert_step=MethodType(expert_step,model)
  try:yield state
  finally:
    for name,value in originals.items():setattr(model,name,value)
    state.pop('finish',None)


def capture_prediction(components,options,row,prepared,packed,save):
  """Cohort interface: reuse its existing tuple and initialized selected native components."""
  from tinygrad.helpers import Context
  if row['split'] not in ROUTES or row['route_id']!=ROUTES[row['split']] or any(word in json.dumps(row) for word in FORBIDDEN):
    raise ValueError('Unapproved target input')
  R=module('objective_capture_reset',HERE/'agent-distill-joint-runtime.py')
  prepared,packed=components.pad_inputs(prepared,packed,2048)
  R.reset_state(components.model,components.decoder,components.vision_cache)
  with Context(JIT=0),aligned_targets(components.model,save) as targets:
    components.vision_cache.prepare(prepared)
    if len(components.vision_cache.keys)!=10:raise ValueError('Keep all ten calibrated image-history slots')
    inputs=components.input_class(components.settings,components.config,prepared,packed)
    if options.selective_pixels:inputs.selective_cache=components.vision_cache
    prediction=components.runner(components.model,components.settings,components.config,components.tokenizer,prepared,packed,seed=0,inputs=inputs)
    targets['finish'](prediction,prepared,packed)
  return dict(id=row['id'],split=row['split'],route_id=row['route_id'],query_ns=row['query_ns'],
    input_file_sha256=row['input_file_sha256'],targets=targets,teacher=prediction)


def load_record(record,device='CPU',*,fixture=False):
  import numpy as np
  from tinygrad import Tensor
  if record.get('synthetic_fixture') and not fixture:raise ValueError('Synthetic capture is not a training target')
  target=record['targets']
  if not target['complete'] or record['route_id']!=ROUTES.get(record['split']):raise ValueError('Need complete independent approved targets')
  if any(word in json.dumps(record) for word in FORBIDDEN):raise ValueError('Forbidden target')
  if target['capacity']!=2304 or not 0<target['prompt_length']<target['valid_length']<=2304:raise ValueError('Invalid full context')
  if len(target['valid_token_ids'])!=target['valid_length'] or len(target['valid_rows_last_write'])!=target['valid_length']:
    raise ValueError('Missing accepted teacher sequence')
  if any(not 0<=i<len(target['language_writes']) for i in target['valid_rows_last_write']):raise ValueError('Missing current language row')
  if any(row['layer_order']!=list(range(28)) for row in target['language_writes']+target['expert']) or len(target['expert'])!=3:
    raise ValueError('Changed teacher execution')
  expected={f'language.{k}':shape for k,shape in dict(x=(1,2304,2048),y=(1,2304,2048),k=(1,8,2304,128),v=(1,8,2304,128),
             cos=(1,1,2304,128),sin=(1,1,2304,128)).items()}
  for i in range(3):expected.update({f'expert{i}.{k}':shape for k,shape in dict(x=(1,64,512),y=(1,64,512),cos=(1,1,64,128),
    sin=(1,1,64,128),mask=(2368,),actions=(1,64,2),timestep=(1,),velocity=(1,64,2)).items()})
  if set(target['tensors'])!=set(expected):raise ValueError('Incomplete aligned target tensors')
  arrays={}
  for name,item in target['tensors'].items():
    path=Path(item['path']).resolve()
    if not path.is_relative_to(WORK) or sha(path)!=item['sha256']:raise ValueError('Changed target')
    array=np.load(path,allow_pickle=False)
    if list(array.shape)!=item['shape'] or array.shape!=expected[name] or str(array.dtype)!=item['dtype']:raise ValueError('Changed tensor shape/dtype')
    if item['bf16_bits']:
      if array.dtype!=np.uint16:raise ValueError('Expected BF16 bits')
      array=(array.astype(np.uint32)<<16).view(np.float32)
    elif array.dtype!=np.float32:raise ValueError('Expected FP32 tensor')
    if name.endswith('.mask'):
      expected_mask=np.full(2368,-np.inf,np.float32);expected_mask[:target['valid_length']]=0;expected_mask[2304:]=0
      np.testing.assert_array_equal(array,expected_mask)
    elif not np.isfinite(array).all():raise ValueError('Nonfinite objective input')
    arrays[name]=array
  for i in range(3):
    np.testing.assert_allclose(arrays[f'expert{i}.timestep'],[i/3],rtol=0,atol=1e-7)
    if i:
      np.testing.assert_allclose(arrays[f'expert{i}.actions'],arrays[f'expert{i-1}.actions']+arrays[f'expert{i-1}.velocity']/3,rtol=2e-6,atol=2e-6)
      for key in ('mask','cos','sin'):np.testing.assert_array_equal(arrays[f'expert{i}.{key}'],arrays[f'expert0.{key}'])
  values={name:Tensor(array.copy(),device=device).is_param_(False) for name,array in arrays.items()}
  return dict(prompt_length=target['prompt_length'],valid_length=target['valid_length'],capacity=target['capacity'],
    language={k:values['language.'+k] for k in ('x','y','cos','sin','k','v')},
    expert=[{k:values[f'expert{i}.'+k] for k in ('x','y','cos','sin','mask','actions','timestep','velocity')} for i in range(3)])


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inputs-manifest',type=Path,required=True)
  parser.add_argument('--sample-id',required=True);parser.add_argument('--execute-gpu',action='store_true');args=parser.parse_args()
  root=args.output.resolve().parent
  if args.output.name!='report.json' or not root.is_relative_to(WORK) or not root.is_dir() or args.output.exists():raise ValueError('Fresh parent wrapper directory required')
  P=module('objective_selected_parent',HERE/'agent-distill-joint-parent.py')
  rows=P.selected_rows(args.inputs_manifest,[args.sample_id]);plan=read(PLAN)
  temporary=root/'agent-distill-objective-tmp';temporary.mkdir()
  cache=root/'agent-distill-objective-kernels.db';seed=plan['compiler_cache_seed']
  if sha(seed['path'])!=seed['sha256']:raise ValueError('Changed compiler seed')
  shutil.copyfile(seed['path'],cache)
  invocation=read(plan['invocation_path']);invocation['environment'].update(CACHEDB=str(cache),TMPDIR=str(temporary))
  owned=root/'agent-distill-objective-invocation.json';owned.write_text(json.dumps(invocation,indent=2)+'\n');plan['invocation_path']=str(owned)
  for key,value in invocation['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  if not args.execute_gpu:os.environ['DEV']='CPU'
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
                    CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  ready=P.prepare(plan,rows)[0];sys.path.insert(0,str(BASE/'tinygrad'))
  if 'prepared' in rows[0] and 'packed' in rows[0]:
    common=module('objective_existing_input_fingerprints',HERE/'agent-distill-command-common.py')
    if common.fingerprint(ready[1])!=rows[0]['prepared'] or common.fingerprint(ready[2])!=rows[0]['packed']:
      raise ValueError('Changed cohort prepared inputs or mRoPE')
  outcome=dict(complete=False,gpu_executed=False,quality_qualified=False,timings_valid_for_throughput=False,
    input_manifest=dict(path=str(args.inputs_manifest),sha256=sha(args.inputs_manifest)),
    source_sha256={str(p):sha(p) for p in HERE.glob('agent-distill-objective-*.py')})
  native_source=Path(plan['runtime_root'])/'model.py'
  outcome['source_sha256'][str(native_source)]=sha(native_source)
  def pipeline(value,options):
    if not args.execute_gpu:
      P.no_devices();outcome.update(complete=True,preflight_only=True,sample_id=rows[0]['id']);return
    import numpy as np
    components=value.initialize(options,sources=outcome['source_sha256'],source_dir=Path(plan['runtime_root'])/'experiments')
    if components.Device[components.Device.DEFAULT].arch!='gfx1200':raise ValueError('Expected parent-owned Chestnut')
    outcome['gpu_executed']=True
    outcome['settings']={k:components.report[k] for k in ('settings','runtime_environment','initial_noise_sha256','prefill_calibration_sha256','vision_calibration_sha256')}
    directory=root/'agent-distill-objective-targets';directory.mkdir()
    def save(name,array,bf16):
      path=directory/(name+'.npy');np.save(path,array,allow_pickle=False)
      return dict(path=str(path),sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype),bf16_bits=bf16)
    outcome['record']=capture_prediction(components,options,*ready,save)
    outcome['complete']=True
  try:P.launch(plan,args.output,[rows[0]['sample']],pipeline)
  except BaseException as error:outcome['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:args.output.write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
