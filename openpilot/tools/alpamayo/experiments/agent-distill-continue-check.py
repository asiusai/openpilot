"""Actual CPU restoration and tiny synthetic continuation. Never fits captured examples."""
import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

sys.dont_write_bytecode=True
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-continue-artifacts'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,default=ROOT/'live-v1')
  parser.add_argument('--output-dir',type=Path,default=ROOT/'cpu-controls-v1');args=parser.parse_args()
  root=args.output_dir.resolve();assert root.is_relative_to(ROOT) and root.name.startswith('cpu-controls-');root.mkdir(exist_ok=False)
  def guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      path=os.fsdecode(values[0])
      assert not path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb','/sys/devices/virtual/kfd'))
      assert not ('sealed_test' in path and path.endswith('.npz'))
      assert '0000008e--321b2ddc03' not in path
  sys.addaudithook(guard)
  bundle=args.bundle.resolve();frozen=bundle/'frozen';protocol=bundle/'agent-distill-continue-protocol.json';digest=sha(protocol)
  runner=frozen/'agent-distill-continue-train.py'
  namespace=runpy.run_path(str(runner),run_name='agent_distill_continue_actual_factory_cpu');C=namespace['C'];p=C.protocol(protocol)
  A=C.module('agent_distill_continue_check_admit',frozen/'agent-distill-continue-admit.py')
  S=C.module('agent_distill_continue_check_state',frozen/'agent-distill-continue-state.py')
  oldS=C.module('agent_distill_continue_check_old_state',Path(p['predecessor']['state_source']))
  failures=[]
  def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,FileNotFoundError,TypeError,RuntimeError):failures.append(name)
    else:raise AssertionError('Negative control accepted: '+name)
  def cli(arguments):
    previous=sys.argv;sys.argv=[str(runner),'--protocol-sha',digest,*arguments]
    try:runpy.run_path(str(runner),run_name='__main__')
    finally:sys.argv=previous
  assert 'tinygrad' not in sys.modules
  cli([]);assert 'tinygrad' not in sys.modules
  reject('missing reviewed availability before tinygrad import',lambda:cli(['--execute-training']))
  stale=root/'agent-distill-continue-stale-availability.json';write(stale,dict(captured_unix_ns=0,synthetic_stale_fixture=True))
  reject('stale reviewed availability before tinygrad import',lambda:cli(['--availability-snapshot',str(stale),'--reviewed-availability-sha',sha(stale),'--execute-training']))
  assert 'tinygrad' not in sys.modules and not (ROOT/'pilot-100-v1').exists()
  # Pin actual production arithmetic/factory. Only the schedule, admission, recovery cap, and reporting change.
  old_runner=Path(p['predecessor']['state_source']).with_name('agent-distill-fit-live-train.py')
  old_tree,new_tree=ast.parse(old_runner.read_text()),ast.parse(runner.read_text())
  def function(tree,name):return next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name)
  for name in ('seed_arrays','environment','tensors','objective','validate'):
    assert ast.dump(function(old_tree,name),include_attributes=False)==ast.dump(function(new_tree,name),include_attributes=False),name
  def optimizer(tree):return next(node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='AdamW')
  assert ast.dump(optimizer(old_tree),include_attributes=False)==ast.dump(optimizer(new_tree),include_attributes=False)
  def arithmetic(tree):
    child=function(tree,'child');loop=next(node for node in ast.walk(child) if isinstance(node,ast.For) and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id=='range')
    return ast.dump(next(node for node in loop.body if isinstance(node,ast.With)),include_attributes=False)
  assert arithmetic(old_tree)==arithmetic(new_tree)
  assert (frozen/'agent-distill-continue-ledger.py').read_bytes()==old_runner.with_name('agent-distill-fit-ledger.py').read_bytes()
  # Only two numerical-state module differences are the admitted total-update bounds.
  expected=Path(p['predecessor']['state_source']).read_text().replace('0<=step<=25','0<=step<=100')
  assert expected==(frozen/'agent-distill-continue-state.py').read_text()
  for name,mutate in [
    ('changed learning rate',lambda value:value['hyperparameters'].update(learning_rate=1e-5)),
    ('changed total updates',lambda value:value['hyperparameters'].update(updates=101)),
    ('changed validation period',lambda value:value['hyperparameters'].update(validate_every=50)),
    ('expanded process allocation ceiling',lambda value:value.update(cumulative_allocation_limit=512*1024**3)),
    ('wrong predecessor checkpoint',lambda value:value['predecessor']['files'].update(**{'checkpoint-0025.npz':'0'*64}))]:
    changed=copy.deepcopy(p);mutate(changed);path=root/('agent-distill-continue-negative-'+str(len(failures))+'.json');write(path,changed)
    reject(name,lambda path=path:C.protocol(path))
  wrong=copy.deepcopy(p);wrong['predecessor']['files']['latest.json']='0'*64
  reject('stale predecessor pointer',lambda:A.predecessor(wrong,oldS))
  wrong=copy.deepcopy(p);wrong['predecessor']['context']['hyperparameters']['updates']=24
  reject('wrong predecessor context',lambda:A.predecessor(wrong,oldS))
  os.environ.update(DEV='CPU:LLVM',TC='0',ALLOW_TF32='0',TRAINING='0',PROFILE='0',PMC='',SQTT='',VIZ='0',
    OPTIM_DTYPE='float32',FUSE_OPTIM='0',CONST_LR='0',LRU='0',NO_MEMORY_PLANNER='1',
    PYTHONDONTWRITEBYTECODE='1',CACHEDB=str(root/'cpu-cache.db'),TMPDIR=str(root),
    CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  sys.path.insert(0,str(BASE/'tinygrad'))
  import numpy as np
  from tinygrad import Tensor,Device
  from tinygrad.helpers import Context
  from tinygrad.nn.optim import AdamW
  # Actual runpy factory + actual BF16 seeds, actual native optimizer, all 27 restoration arrays.
  arrays,metadata,transition=A.predecessor(p,oldS);assert transition['next_command_index']==1
  B=C.graph();values=namespace['seed_arrays'](p);block=B.Block(values,p['block_config'],'CPU')
  del values
  block.weights={name:block.weights[name].is_param_(True).realize() for name in sorted(block.weights)}
  opt=AdamW(list(block.weights.values()),lr=3.5e-6,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)
  def actual_fit_forbidden(*args,**kwargs):raise AssertionError('No actual checkpoint forward or optimizer update is authorized')
  B.Block.__call__=actual_fit_forbidden;opt.step=actual_fit_forbidden
  S.restore(block.weights,opt,arrays);restored=S.snapshot(block.weights,opt)
  for name in arrays:np.testing.assert_array_equal(restored[name].view(np.uint32),arrays[name].view(np.uint32))
  context=A.context(p,digest);shapes={name:tuple(item['shape']) for name,item in p['master_weights'].items()}
  actual=root/'actual-restoration-only';actual.mkdir();A.copy_best(Path(p['predecessor']['root']),actual,metadata,S)
  S.save(actual,restored,metadata,context,shapes,initial_recovery=True)
  loaded,loaded_metadata=S.load(actual,context,shapes);A.history(loaded_metadata,25);A.read_best(actual,loaded_metadata,shapes)
  for name in arrays:np.testing.assert_array_equal(loaded[name].view(np.uint32),arrays[name].view(np.uint32))
  reject('new checkpoint with stale predecessor context',lambda:S.load(actual,p['predecessor']['context'],shapes))
  actual_restoration=dict(array_count=len(arrays),all_bits_equal=True,actual_model_forward_calls=0,actual_model_optimizer_updates=0,
    runpy_factory_used=True,original_BF16_seed_factory_used=True,new_context_checkpoint_sha256=sha(actual/'checkpoint-0025.npz'))
  del arrays,restored,loaded,block,opt;gc.collect()
  # Synthetic native AdamW equivalence across the real 25-to-100 boundary and command cycle.
  start=np.array([[.25,-.375],[.125,.75]],np.float32)
  inputs=[np.array([[1.,.5],[-.5,2.],[.25,-1.]],np.float32)+np.float32(i/8) for i in range(3)]
  def make(initial):
    weights={'toy':Tensor(initial.copy(),device='CPU').is_param_(True).realize()}
    optimizer=AdamW(list(weights.values()),lr=3.5e-6,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)
    return weights,optimizer
  def update(weights,optimizer,step):
    index=(step-1)%3
    with Context(TRAINING=1):
      optimizer.zero_grad();prediction=(Tensor(inputs[index],device='CPU').is_param_(False)@weights['toy']).reshape(1,3,2)
      kv=prediction.reshape(1,3,2,1).transpose(1,2)
      targets=[Tensor.zeros(*item.shape,device='CPU').is_param_(False) for item in (prediction,kv,kv)]
      objective=C.loss(B,(prediction,kv,kv*2),targets,3)
      gradient=objective.gradient(weights['toy'])[0];weights['toy'].grad=gradient.realize();optimizer.step()
    return index
  def validation(weights,step):
    # Synthetic-only independent validation values; never actual teacher target losses.
    value=float(np.mean((np.array([[.75,-.25]],np.float32)@weights['toy'].numpy())**2))
    return dict(step=step,mean=value,samples=['validation-command-'+str(i) for i in range(3)])
  reference,refopt=make(start);meta=dict(step=0,best_step=0,validation=[validation(reference,0)])
  meta['best_validation']=meta['validation'][0]['mean']
  old_toy=root/'synthetic-old-step25';old_toy.mkdir();toy_shapes={'toy':(2,2)}
  for step in range(1,101):
    update(reference,refopt,step);meta['step']=step
    if step%25==0:
      item=validation(reference,step);meta['validation'].append(item)
      if item['mean']<meta['best_validation']:meta.update(best_validation=item['mean'],best_step=step)
    if step==25:oldS.save(old_toy,S.snapshot(reference,refopt),copy.deepcopy(meta),p['predecessor']['context'],toy_shapes,initial_recovery=True)
  uninterrupted=S.snapshot(reference,refopt)
  saved,resume_meta=oldS.load(old_toy,p['predecessor']['context'],toy_shapes)
  resumed,resopt=make(np.zeros_like(start));S.restore(resumed,resopt,saved)
  new_toy=root/'synthetic-new-context';new_toy.mkdir();S.save(new_toy,S.snapshot(resumed,resopt),resume_meta,context,toy_shapes,initial_recovery=True)
  commands=[]
  for step in range(26,101):
    S.intent(new_toy,step-1);commands.append(update(resumed,resopt,step));resume_meta['step']=step
    if step%25==0:
      item=validation(resumed,step);resume_meta['validation'].append(item)
      if item['mean']<resume_meta['best_validation']:resume_meta.update(best_validation=item['mean'],best_step=step)
    S.save(new_toy,S.snapshot(resumed,resopt),resume_meta,context,toy_shapes)
    if step in (26,49,50,74,75,99,100):
      recovered,recovered_meta=S.load(new_toy,context,toy_shapes);assert recovered_meta['step']==step;A.history(recovered_meta,step)
  final=S.snapshot(resumed,resopt)
  for name in uninterrupted:np.testing.assert_array_equal(uninterrupted[name].view(np.uint32),final[name].view(np.uint32))
  assert commands[0]==1 and commands==[(step-1)%3 for step in range(26,101)] and resume_meta==meta
  reject('wrong new-context predecessor',lambda:S.load(new_toy,p['predecessor']['context'],toy_shapes))
  corrupt=root/'synthetic-corrupt';shutil.copytree(new_toy,corrupt)
  with (corrupt/'checkpoint-0100.npz').open('ab') as stream:stream.write(b'corrupt synthetic fixture')
  reject('corrupt recovery payload',lambda:S.load(corrupt,context,toy_shapes))
  pending=root/'synthetic-uncommitted';shutil.copytree(new_toy,pending);S.intent(pending,100)
  reject('uncommitted update replay',lambda:S.load(pending,context,toy_shapes))
  stale_pointer=root/'synthetic-stale-boundary';shutil.copytree(new_toy,stale_pointer)
  pointer=json.loads((stale_pointer/'latest.json').read_text());pointer['metadata']['step']=99
  (stale_pointer/'latest.json').write_text(json.dumps(pointer))
  reject('stale checkpoint boundary',lambda:S.load(stale_pointer,context,toy_shapes))
  rejected=copy.deepcopy(final);rejected['learning_rate']*=2
  reject('changed restored learning rate',lambda:S.validate_arrays(rejected,toy_shapes))
  rejected=copy.deepcopy(final);del rejected['moment2/toy']
  reject('missing optimizer moment',lambda:S.validate_arrays(rejected,toy_shapes))
  rejected=dict(step=25,best_step=25,best_validation=1.,validation=[
    dict(step=step,mean=1.,samples=['validation-command-'+str(i) for i in range(3)]) for step in (0,25)])
  reject('validation tie chooses later checkpoint',lambda:A.history(rejected,25))
  rejected=copy.deepcopy(resume_meta);rejected['validation'].append(dict(step=101,mean=0,samples=[]))
  reject('extra stopping validation',lambda:A.history(rejected,100))
  rejected=copy.deepcopy(resume_meta);rejected['best_validation']=0
  reject('selection outside separate validation',lambda:A.history(rejected,100))
  assert set(Device._opened_devices)<= {'CPU','NPY','PYTHON'};C.no_devices()
  # Recheck all predecessor bytes after every CPU operation.
  for name,digest2 in p['predecessor']['files'].items():assert sha(Path(p['predecessor']['root'])/name)==digest2
  result=dict(complete=True,all_passed=True,gpu_executed=False,actual_model_optimizer_updates=0,actual_student_forward_calls=0,
    actual_restoration=actual_restoration,predecessor_unchanged=True,actual_default_runpy_preflight_passed=True,
    production_arithmetic_and_optimizer_AST_identical=True,allocation_ledger_byte_identical=True,
    synthetic_only_optimizer_updates=175,synthetic_resume_25_to_100_all_state_bits_equal=True,
    next_step=26,next_command_index=commands[0],fixed_order_correct=True,validation_steps=[0,25,50,75,100],
    sealed_target_arrays_opened=0,negative_controls_rejected=failures,opened_devices=sorted(Device._opened_devices),
    protocol_sha256=digest,source_sha256={str(path):sha(path) for path in [Path(__file__),*frozen.glob('*.py')]})
  write(root/'agent-distill-continue-check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
