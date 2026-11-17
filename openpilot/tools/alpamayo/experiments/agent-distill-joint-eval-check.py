"""CPU admission and full selected-launcher inference mock; no training or GPU execution."""
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
os.environ.update(DEV='CPU',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',
  OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(BASE/'tinygrad'))
def no_gpu(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):
    raise RuntimeError('Joint evaluation checks are CPU-only')
sys.addaudithook(no_gpu)
import numpy as np
from tinygrad import Device,dtypes
E=runpy.run_path(str(HERE/'agent-distill-joint-eval-run.py'))
P=E['P'];C=E['C'];sha,read=E['sha'],E['read']
old=E['module']('joint_eval_prior_integration',HERE/'agent-distill-composed-check.py')


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rejected(call):
  try:call()
  except (ValueError,AssertionError):return
  raise AssertionError('Negative control passed')


def selection(root):
  # Actual completed selected25 is admitted through the production loader, including every recovery array.
  paired,actual=E['selected_joint'](E['ACTUAL_ROOT'],E['ACTUAL_REPORT_SHA'])
  assert actual['selected_step']==25 and actual['best_sha256']=='8073b781a84b8f27dc9a64b8ffe7fe1aa82fbbead91ab431918396716414e29a'
  root.mkdir();report=deepcopy(read(E['ACTUAL_ROOT']/'report.json'));report['synthetic_fixture']=True
  pointer=deepcopy(read(E['ACTUAL_ROOT']/'latest.json'));metadata=report['metadata']
  seeds=P.seed_arrays()
  with (root/'best.npz').open('xb') as stream:np.savez(stream,**seeds)
  final=metadata['validation'][1];components=final['components']
  components['expert']+=metadata['validation'][0]['mean']-final['mean']+.1
  final['mean']=components['total']=.5*(components['prompt']+components['reasoning'])+components['expert']
  metadata.update(best_step=0,best_validation=metadata['validation'][0]['mean'],best_artifact_sha256=sha(root/'best.npz'))
  report.update(validation=metadata['validation'],best_step=0,strict_validation_improvement=False,best_sha256=metadata['best_artifact_sha256'])
  pointer['metadata']=metadata
  # The immutable completed state is only read through this fixture symlink; tests edit fixture metadata and selected masters only.
  (root/'checkpoint-0025.npz').symlink_to(E['ACTUAL_ROOT']/'checkpoint-0025.npz')
  for name,value in [('report.json',report),('latest.json',pointer),('journal.json',dict(committed=25,intended=None)),
      ('wall.json',dict(returncode=0,timed_out=False)),('best.json',dict(file='best.npz',sha256=metadata['best_artifact_sha256'],selected_step=0,validation_mean=metadata['best_validation']))]:write(root/name,value)
  load=lambda:E['selected_joint'](root,sha(root/'report.json'),fixture=True)
  _,baseline=load();assert baseline['selected_step']==0 and baseline['baseline_pair_selected']
  assert baseline['origins']['language_updates']==100 and baseline['origins']['expert_updates']==25
  negatives=[]
  rejected(lambda:E['selected_joint'](root,sha(root/'report.json')));negatives.append('marked fixture rejected by production admission')
  rejected(lambda:E['selected_joint'](E['ACTUAL_ROOT'],'0'*64));negatives.append('unreviewed report checksum')
  write(root/'journal.json',dict(committed=25,intended=26));rejected(load);write(root/'journal.json',dict(committed=25,intended=None));negatives.append('dirty journal')
  report['complete']=False;write(root/'report.json',report);rejected(load);report['complete']=True;write(root/'report.json',report);negatives.append('incomplete pilot')
  report['context']['hyperparameters']['max_committed_updates']=24;write(root/'report.json',report);rejected(load)
  report['context']['hyperparameters']['max_committed_updates']=25;write(root/'report.json',report);negatives.append('changed training context')
  best=read(root/'best.json');best['sha256']='0'*64;write(root/'best.json',best);rejected(load)
  best['sha256']=sha(root/'best.npz');write(root/'best.json',best);negatives.append('changed selected checksum')
  report['metadata']['best_step']=25;pointer['metadata']=report['metadata'];write(root/'report.json',report);write(root/'latest.json',pointer)
  rejected(load);report['metadata']['best_step']=0;write(root/'report.json',report);write(root/'latest.json',pointer);negatives.append('wrong validation winner')
  return paired,dict(actual_selected25=actual,marked_baseline_selection=baseline,negatives=negatives,optimizer_updates=0)


def reuse():
  plan=read(C.PLAN);result=[]
  for sample in C.TEACHERS:
    row=old.P.selected_rows(old.P.INPUTS,[sample])[0]
    item=E['teacher_preflight'](plan,row);assert item['reusable']
    path,digest=C.TEACHERS[sample];settings=deepcopy(read(path)['settings']);settings['initial_noise_sha256']='0'*64
    rejected(lambda:C.saved_teacher(path,digest,row,settings,plan))
    wrong=deepcopy(row);wrong['query_ns']+=1
    rejected(lambda:C.saved_teacher(path,digest,wrong,read(path)['settings'],plan))
    result.append(dict(sample=sample,**item,changed_noise_rejected=True,changed_input_rejected=True))
  return result


def motion_checks():
  times=np.arange(1,65,dtype=np.float64)*.1
  def prediction(x):return dict(xyz=np.column_stack([x,np.zeros((64,2))])[None].tolist())
  constant=E['motion'](prediction(10*times));braking=E['motion'](prediction(12*times-.5*times**2))
  np.testing.assert_allclose(constant['speed_m_s'],10,atol=1e-11)
  np.testing.assert_allclose(constant['acceleration_m_s2'],0,atol=1e-10)
  np.testing.assert_allclose(braking['acceleration_m_s2'],-1,atol=1e-10)
  assert all(abs(value-10)<1e-12 for value in constant['mean_speed_m_s'].values())
  return dict(constant_speed_and_zero_acceleration=True,one_m_s2_braking=True,all64_speeds_and63_accelerations=True)


def integration(root,paired):
  root.mkdir();namespace=runpy.run_path(str(HERE/'agent-distill-joint-eval-run.py'));entry=namespace['main'];globals_=entry.__globals__
  original_execute=globals_['execute_prediction'];audit=[]
  # Retain all actual16 selected tensors and original widths in the runtime adapter. Only neural arithmetic is mocked.
  source=inspect.getsource(old.initializer)
  before="m.weights={p+'12.input_layernorm.weight':Tensor.ones(1,dtype=m.dtype) for p in (R.LANGUAGE,R.EXPERT)}"
  after="m.weights={prefix+'12.'+name:Tensor.empty(value.shape,dtype=m.dtype) for prefix,role in ((R.LANGUAGE,'language'),(R.EXPERT,'expert')) for name,value in joint_expected[role].items()}"
  assert source.count(before)==1;source=source.replace(before,after)
  before='start_l,start_e=len(language_sources),len(expert_sources)'
  after="""if hasattr(m,'joint_depth_mapping'):
        for role,values in joint_expected.items():
          for name,value in values.items():
            tensor=m.weights['agent_joint.'+role+'.12.'+name]
            assert tensor.shape==value.shape and tensor.dtype==dtypes.bfloat16
            expected=float(torch.tensor(float(value.flat[0])).to(torch.bfloat16).float())
            assert float(tensor.flatten()[0].float().item())==expected
      start_l,start_e=len(language_sources),len(expert_sources)"""
  assert source.count(before)==1;source=source.replace(before,after)
  old.joint_expected=paired;exec(compile(source,str(Path(__file__))+':selected-width-initializer','exec'),old.__dict__)
  def execute(args,plan,ready,language,expert,receipt):
    assert sum(value.size for value in language.values())==50335744 and sum(value.size for value in expert.values())==7078912
    assert receipt['selected_step']==25 and receipt['best_sha256']=='8073b781a84b8f27dc9a64b8ffe7fe1aa82fbbead91ab431918396716414e29a'
    # Both architectures execute only in this CPU mock. The real CLI always reuses the immutable teacher and executes one candidate query.
    args.rerun_teacher=True
    return original_execute(args,plan,ready,language,expert,receipt,cpu_mock=True,initialize=old.initializer(plan,ready[1],audit))
  globals_['execute_prediction']=execute
  argv=sys.argv[:];sys.argv=[str(HERE/'agent-distill-joint-eval-run.py'),'--output',str(root/'report.json'),'--sample-id','train-command-0',
    '--joint-root',str(E['ACTUAL_ROOT']),'--joint-report-sha',E['ACTUAL_REPORT_SHA'],'--execute-gpu']
  try:entry()
  finally:sys.argv=argv
  report=read(root/'report.json');objects=audit.pop(0)
  assert report['complete'] and report['cpu_integration_mock'] and not report['gpu_executed']
  assert [arm['layers'] for arm in audit]==[28,25]
  assert report['adapter']['joint_selected_step']==25 and not report['adapter']['baseline_pair_selected']
  assert report['adapter']['paired_students'] and report['adapter']['vision_layers']==27 and not report['adapter']['teacher_cache_injected']
  assert report['paired_checkpoint_sha256']=='8073b781a84b8f27dc9a64b8ffe7fe1aa82fbbead91ab431918396716414e29a'
  assert all(not row['conditional_block']['matched'] for row in report['comparison']['euler'])
  assert report['comparison']['xyz_max_absolute_m']>0 and len(report['comparison']['vehicle_motion']['candidate']['speed_m_s'])==64
  assert objects['model'].cache is None and not objects['decoder'].history and not objects['vision_cache'].cache
  assert objects['model'].config['expert_config']['llm_config']['num_hidden_layers']==28
  assert objects['model'].config['vlm_config']['text_config']['num_hidden_layers']==28
  assert not any(name.startswith('agent_joint.') for name in objects['model'].weights)
  assert not objects['model'].trajectory_jits and objects['model'].padded_prefill.shape is None
  return dict(actual_runpy_entrypoint=True,actual_completed_checkpoint_admission=True,all16_original_width_tensors_loaded=True,
    actual_selected_launcher_spec6_inputs_cache_decoder_and_Euler=True,all16_runtime_bf16_samples_match_selected_masters=True,
    neural_arithmetic_mocked=True,arms=audit,full_trajectory_and_motion_report=True,cleanup_passed=True,report=str(root/'report.json'))


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(E['ROOT']):raise ValueError('Fresh owned CPU output required')
  root.mkdir(parents=True)
  paired,selected=selection(root/'marked-baseline-fixture')
  result=dict(gpu_executed=False,optimizer_updates=0,actual_selected_checkpoint=selected,reuse=reuse(),motion=motion_checks())
  result['integration']=integration(root/'integration',paired)
  old.P.no_devices();assert all(name.startswith(('CPU','NPY','PYTHON','DISK')) for name in Device._opened_devices)
  result.update(all_passed=True,opened_devices=list(Device._opened_devices),
    source_sha256={str(path):sha(path) for path in HERE.glob('agent-distill-joint-eval-*.py')})
  write(root/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
