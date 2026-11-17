"""CPU fixed schedule, independent validation selection, full-shape graph identity, and native launcher mock."""
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
os.environ.update(DEV='CPU',TC='0',ALLOW_TF32='0',NO_MEMORY_PLANNER='1',LRU='0',PROFILE='0',
  MAX_BUFFER_SIZE=str(320*1024**2),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
sys.path.insert(0,str(BASE/'tinygrad'))
def no_gpu(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):
    raise RuntimeError('Pilot checks are CPU-only')
sys.addaudithook(no_gpu)
import numpy as np
from tinygrad import Tensor,Device
from tinygrad.helpers import Context
R=runpy.run_path(str(HERE/'agent-distill-objective-pilot-run.py'));P=R['P']
prior=P.module('pilot_reused_qualified_mock',HERE/'agent-distill-objective-qualify-check.py')
base_fixture=prior.fixture


def fixture():
  weights,train=base_fixture()
  def validation(device):
    value=train(device)
    value['language']['x']=value['language']['x']+.03;value['language']['y']=value['language']['y']+.11
    for step in value['expert']:step['y']=step['y']-.07
    return value
  def both(device):return dict(train=train(device),validation=validation(device))
  return weights,train,validation,both


def full_shape():
  weights={}
  for role,width,hidden in (('language',2048,9216),('expert',512,3840)):
    shapes={'input_layernorm.weight':(width,),'post_attention_layernorm.weight':(width,),
      'self_attn.q_proj.weight':(2048,width),'self_attn.k_proj.weight':(1024,width),'self_attn.v_proj.weight':(1024,width),
      'self_attn.o_proj.weight':(width,2048),'mlp.fc1.weight':(hidden,width),'mlp.fc2.weight':(width,hidden)}
    weights.update({role+'.'+name:Tensor.empty(shape).is_param_(True) for name,shape in shapes.items()})
  data=dict(prompt_length=2027,valid_length=2047,capacity=2304,
    language={k:Tensor.empty(shape) for k,shape in dict(x=(1,2304,2048),y=(1,2304,2048),k=(1,8,2304,128),v=(1,8,2304,128),
      cos=(1,1,2304,128),sin=(1,1,2304,128)).items()},expert=[])
  for _ in range(3):data['expert'].append({k:Tensor.empty(shape) for k,shape in dict(x=(1,64,512),y=(1,64,512),
    cos=(1,1,64,128),sin=(1,1,64,128),mask=(2368,)).items()})
  l={k.removeprefix('language.'):v for k,v in weights.items() if k.startswith('language.')}
  e={k.removeprefix('expert.'):v for k,v in weights.items() if k.startswith('expert.')}
  with Context(TRAINING=1):
    a,ao=P.graph(weights,data);b,bo=P.Q.O.objective(l,e,data,query_chunk=128)
    assert ao['language_extent']==bo['language_extent']==2048
    assert ao['context_k'].shape==ao['context_v'].shape==(1,8,2304,128)
    assert [v.shape for v in ao['expert']]==[(1,64,512)]*3
    ag=a['total'].gradient(*weights.values(),ao['context_k'],ao['context_v'])
    bg=b['total'].gradient(*weights.values(),bo['context_k'],bo['context_v'])
    for left,right in zip([*a.values(),*ao['language'],*ao['expert'],*ag],[*b.values(),*bo['language'],*bo['expert'],*bg],strict=True):
      assert left.uop.key==right.uop.key
  assert P.O is P.Q.O
  return dict(qualified_graph_identical=True,objective_source_sha256=P.sha(HERE/'agent-distill-objective-core.py'),
    full_shape_forward_and_18_gradient_graphs_identical=True,shape=dict(extent=2048,capacity=2304,euler=3,action_rows=64,query_chunk=128),
    kernels_executed=0,actual_target_arrays_read=False)


def entrypoint(root):
  Device['CPU']
  prior.Q=P;prior.R=R
  weights,train,validation,both=fixture();prior.fixture=lambda:(deepcopy(weights),train);prior.pilot_validation=validation
  source=inspect.getsource(prior.entrypoint)
  source=source.replace('agent-distill-objective-qualify-run.py','agent-distill-objective-pilot-run.py')
  source=source.replace('agent-distill-objective-qualify-core.py','agent-distill-objective-pilot-core.py')
  source=source.replace('--aligned-report','--train-report').replace('--execute-qualification','--execute-pilot')
  source=source.replace("'--output-dir',str(root/'entrypoint')", "'--validation-report',str(root/'synthetic-val.json'),'--validation-report-sha','synthetic-validation','--output-dir',str(root/'entrypoint')")
  before="real_work(values,load,directory,ctx,'CPU',phase,fixture=True,**kwargs)";assert source.count(before)==1
  source=source.replace(before,"real_work(values,lambda device:dict(train=load(device),validation=pilot_validation(device)),directory,ctx,'CPU',phase,fixture=True,**kwargs)")
  source=source.replace('one-update-and-recovery','fixed-25-updates-and-recovery')
  exec(compile(source,str(Path(__file__))+':qualified-entrypoint-mock','exec'),prior.__dict__)
  with Context(MAX_BUFFER_SIZE=320*1024**2):result=prior.entrypoint(root)
  report=P.read(root/'entrypoint/report.json')
  assert report['complete'] and report['updates_this_run']==25 and report['recovery_arrays']==51
  assert report['bounds']==P.BOUNDS and report['initial_weights'].startswith('Immutable language100/expert25')
  assert [v['step'] for v in report['validation']]==[0,25]
  assert all(v['masked_context_gradient_exactly_zero'] for v in report['training_losses'])
  return dict(result,actual_synthetic_updates=25,validation_steps=[0,25],first_weight_and_finalizer=True,selected_step=report['best_step'])


def schedule_and_resume(root):
  weights,_,_,both=fixture();reference=root/'entrypoint';context=P.read(reference/'latest.json')['context']
  interrupted=root/'interrupt-13';interrupted.mkdir()
  part=P.run_step(deepcopy(weights),both,interrupted,context,'CPU',lambda name:None,fixture=True,stop_after=13)
  assert not part['complete'] and [v['step'] for v in part['validation']]==[0]
  resumed_root=root/'resume-13';resumed_root.mkdir()
  resumed=P.run_step(deepcopy(weights),both,resumed_root,context,'CPU',lambda name:None,resume=interrupted,fixture=True)
  shapes={name:value.shape for name,value in weights.items()}
  a,ma=P.S.load(reference,context,shapes);b,mb=P.S.load(resumed_root,context,shapes)
  for name in a:np.testing.assert_array_equal(a[name].view(np.uint32),b[name].view(np.uint32))
  assert ma==mb and resumed['updates_this_run']==12 and len(a)==51
  av,_=P.A.read_best(reference,ma,shapes);bv,_=P.A.read_best(resumed_root,mb,shapes)
  for name in av:np.testing.assert_array_equal(av[name].view(np.uint32),bv[name].view(np.uint32))
  rejected=[]
  def fail(name,call):
    try:call()
    except (ValueError,AssertionError):rejected.append(name)
    else:raise AssertionError('Accepted '+name)
  fail('qualification checkpoint context',lambda:P.S.load(interrupted,dict(context,kind='joint-resource-qualification-v1'),shapes))
  journal=P.read(interrupted/'journal.json');P.S.atomic_json(interrupted/'journal.json',dict(committed=13,intended=14))
  fail('dirty update journal',lambda:P.S.load(interrupted,context,shapes));P.S.atomic_json(interrupted/'journal.json',journal)
  pointer=P.read(interrupted/'latest.json');changed=deepcopy(pointer);changed['sha256']='0'*64
  P.S.atomic_json(interrupted/'latest.json',changed);fail('changed recovery checksum',lambda:P.S.load(interrupted,context,shapes))
  P.S.atomic_json(interrupted/'latest.json',pointer)
  for initial,final,best in ((.5,.6,0),(.5,.5,0),(.5,.4,25)):
    metadata=dict(step=25,kind=P.KIND,best_step=best,best_validation=min(initial,final),training=[dict(step=i) for i in range(1,26)],
      validation=[dict(step=step,mean=value,components=dict(prompt=.2,reasoning=.2,expert=value-.2,total=value),samples=['validation-command-0'])
        for step,value in ((0,initial),(25,final))])
    P.validate_metadata(metadata)
    wrong=deepcopy(metadata);wrong['best_step']=25 if best==0 else 0
    fail('wrong strict validation selection '+str(final),lambda:P.validate_metadata(wrong))
  incomplete=deepcopy(ma);incomplete['validation'][1]['mean']=float('nan')
  fail('nonfinite validation',lambda:P.validate_metadata(incomplete))
  return dict(validation_steps=[0,25],fixed_updates=25,interrupted_at=13,resumed_updates=12,all51_arrays_bit_exact=True,
    selected_weights_bit_exact=True,metadata_identical=True,strict_selection_cases=['worse retains zero','tie retains zero','improvement selects25'],
    negatives=rejected,synthetic_only=True)


def target_controls():
  calls=[]
  with patch.object(P,'target',lambda *args:calls.append(args)):
    try:P.control('completed-train',P.TRAIN_SHA,None,None)
    except ValueError:pass
    else:raise AssertionError('Accepted missing validation')
  assert not calls
  rows=dict(train=dict(route_id='same-drive'),validation=dict(route_id='same-drive'))
  with patch.object(P,'target',lambda path,digest,split:rows[split]):
    try:P.control('train',P.TRAIN_SHA,'validation',P.VALIDATION_SHA)
    except ValueError:pass
    else:raise AssertionError('Accepted same drive')
  return dict(missing_validation_blocks_before_any_target_read=True,same_drive_rejected=True,pending_outputs_read=False)


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(P.ROOT):raise ValueError('Fresh owned CPU output required')
  root.mkdir(parents=True)
  result=dict(gpu_executed=False,actual_data_optimizer_updates=0,full_shape=full_shape(),target_controls=target_controls(),
    entrypoint=entrypoint(root),schedule=schedule_and_resume(root),budget=P.estimate())
  assert all(name.startswith(('CPU','NPY','PYTHON')) for name in Device._opened_devices)
  result.update(all_passed=True,new_bounds=P.BOUNDS,opened_devices=list(Device._opened_devices),
    source_sha256={str(p):P.sha(p) for p in (*HERE.glob('agent-distill-objective-pilot-*.py'),
      HERE/'agent-distill-objective-qualify-core.py',HERE/'agent-distill-objective-qualify-run.py',HERE/'agent-distill-objective-core.py')})
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
