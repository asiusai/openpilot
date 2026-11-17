"""One free-running candidate query using the completed selected paired joint checkpoint."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-joint-eval-artifacts'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
ACTUAL_ROOT=WORK/'agent-distill-objective-pilot-artifacts/live-v1'
ACTUAL_REPORT_SHA='3597bcf5678a0211cf0eaac5ffa07a92d5fcbb5dd29420c1aa40fbe280f37087'


def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value
def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())

for name,digest in [('agent-distill-composed-run.py','907d46bdb1b1f76d529cebbe5365200979b41ac1a81854594e02d26db5c0b8c1'),
    ('agent-distill-joint-runtime.py','4f098eb5ddda5ae4c5cfe0b9e01956f5dc05177aa4994fb9f8c83b86dcec939e')]:
  if sha(HERE/name)!=digest:raise ValueError('Changed reviewed composed runtime')
C=module('joint_eval_composed',HERE/'agent-distill-composed-run.py')
P=module('joint_eval_pilot',HERE/'agent-distill-objective-pilot-core.py')


def selected_joint(root,report_sha,*,fixture=False):
  """Read completed selection, all51 recovery arrays and the exact16 selected masters."""
  import numpy as np
  root=Path(root).resolve()
  if not re.fullmatch('[0-9a-f]{64}',report_sha):raise ValueError('Explicit completed pilot report SHA256 required')
  if fixture:
    if not root.is_relative_to(ROOT):raise ValueError('Expected owned marked checkpoint fixture')
  elif root!=ACTUAL_ROOT or report_sha!=ACTUAL_REPORT_SHA:raise ValueError('Use the parent-reviewed completed joint pilot')
  if sha(root/'report.json')!=report_sha:raise ValueError('Joint pilot report checksum changed')
  report=read(root/'report.json')
  if report.get('synthetic_fixture') and not fixture:raise ValueError('Synthetic fixture cannot admit a live candidate')
  if not report.get('complete') or not report.get('gpu_executed') or report.get('committed_step')!=25 or report.get('kind')!=P.KIND:
    raise ValueError('Completed fixed25 connected pilot required')
  wall=read(root/'wall.json')
  if wall['returncode']!=0 or wall['timed_out']:raise ValueError('Pilot parent did not complete successfully')
  if report['bounds']!=P.BOUNDS or not report.get('allocator_guards_retained_through_finalizers'):
    raise ValueError('Unexpected pilot resource guards')
  if report.get('qualified_ledger_sha256')!='f070ca77dedbc6f22027a2dcb063189b6c66647ae1436d022d0cbf7386b1663d':raise ValueError('Unexpected ledger')
  if report.get('recovery_arrays')!=51 or not report.get('recovery_bit_exact') or report.get('additional_updates_after_recovery')!=0:
    raise ValueError('Missing complete verified recovery')
  _,context=P.control(WORK/'agent-distill-objective-legacy-train-v2/report.json',P.TRAIN_SHA,
    WORK/'agent-distill-objective-validation-v1/report.json',P.VALIDATION_SHA)
  if report['context']!=context or report.get('qualified_resource_report_sha256')!=P.QUALIFIED_SHA:raise ValueError('Changed connected training context')
  for name in ('agent-distill-objective-pilot-core.py','agent-distill-objective-pilot-run.py','agent-distill-objective-core.py'):
    if report['source_sha256'].get(str(HERE/name))!=sha(HERE/name):raise ValueError('Changed actual pilot/objective source')
  seeds=P.seed_arrays();shapes={name:value.shape for name,value in seeds.items()}
  arrays,metadata=P.S.load(root,context,shapes);P.validate_metadata(metadata)
  if metadata!=report['metadata'] or report['validation']!=metadata['validation']:raise ValueError('Committed selection metadata differs')
  if report['checkpoint_sha256']!=read(root/'latest.json')['sha256']:raise ValueError('Final recovery checkpoint differs')
  if report['journal']!=dict(committed=25,intended=None):raise ValueError('Completed report does not have a clean final journal')
  selected,best=P.A.read_best(root,metadata,shapes)
  step=metadata['best_step']
  if report['best_step']!=step or report['strict_validation_improvement']!=(step==25) or report['best_sha256']!=best['sha256']:
    raise ValueError('Completed report differs from actual validation-selected best')
  for name,value in selected.items():
    expected=seeds[name] if step==0 else arrays['weight/'+name]
    np.testing.assert_array_equal(value.view(np.uint32),expected.view(np.uint32))
  paired={role:{name.removeprefix(role+'.'):value for name,value in selected.items() if name.startswith(role+'.')} for role in ('language','expert')}
  if [len(paired[role]) for role in paired]!=[8,8]:raise ValueError('Expected all16 paired tensors')
  receipt=dict(root=str(root),report_sha256=report_sha,best_sha256=best['sha256'],recovery_sha256=report['checkpoint_sha256'],
    selected_step=step,joint_optimizer_updates_selected=step,completed_joint_updates=25,baseline_pair_selected=step==0,
    origins=dict(language_updates=100,expert_updates=25,checkpoint_sha256={k:v[1] for k,v in P.SEEDS.items()}),
    selection='original language100/expert25 pair retained' if step==0 else 'strict independent joint validation selected25',
    validation=metadata['validation'],training_objective=P.HYPER['objective'],master_dtype='float32',selected_tensor_names=sorted(selected),
    training_conditioning='aligned teacher token sequence and teacher Euler states with differentiable student K/V',
    evaluation_conditioning='own free-running reasoning, matching current language K/V and own three Euler action states',
    recovery_arrays_verified=51,quality_qualified=False)
  return paired,receipt


def motion(prediction):
  import numpy as np
  xyz=np.asarray(prediction['xyz'],np.float64)
  if xyz.shape!=(1,64,3) or not np.isfinite(xyz).all():raise ValueError('Expected complete finite trajectory')
  speed=np.linalg.norm(np.diff(np.vstack([np.zeros(2),xyz[0,:,:2]]),axis=0),axis=1)/.1
  acceleration=np.diff(speed)/.1;midpoints=(np.arange(64)+.5)*.1
  return dict(method='XY position-derived interval speed from ego origin; finite-difference acceleration, dt0.1s',
    speed_midpoints_s=midpoints.tolist(),speed_m_s=speed.tolist(),acceleration_times_s=(np.arange(1,64)*.1).tolist(),
    acceleration_m_s2=acceleration.tolist(),mean_speed_m_s={name:float(speed[lo:hi].mean()) for name,lo,hi in
      [('0-1',0,10),('1-3',10,30),('3-6.4',30,64)]},
    minimum_acceleration_m_s2=float(acceleration.min()),maximum_acceleration_m_s2=float(acceleration.max()),
    final_xyz_m=xyz[0,-1].tolist(),path_length_m=float(speed.sum()*.1))


original_compare=C.compare
def compare(teacher,candidate):
  import numpy as np
  result=original_compare(teacher,candidate);old,new=motion(teacher['prediction']),motion(candidate['prediction'])
  result.update(vehicle_motion=dict(teacher=old,candidate=new,
    speed_mean_absolute_delta_m_s=float(np.abs(np.asarray(old['speed_m_s'])-new['speed_m_s']).mean()),
    acceleration_mean_absolute_delta_m_s2=float(np.abs(np.asarray(old['acceleration_m_s2'])-new['acceleration_m_s2']).mean())),
    conditioning_summary=dict(teacher_reasoning_tokens=teacher['prediction']['generated_tokens'],candidate_reasoning_tokens=candidate['prediction']['generated_tokens'],
      teacher_valid_context=[x['context_valid_length'] for x in teacher['rollout']['steps']],
      candidate_valid_context=[x['context_valid_length'] for x in candidate['rollout']['steps']],
      internal_reconstruction_is_not_full_prediction_quality=True))
  return result
C.compare=compare

# Reuse every original numerical call, reset, observer, comparison boundary and cleanup.
# Replace only stale separate-checkpoint receipts and add invariant checks for motion timing and vision depth.
source=inspect.getsource(C.execute_prediction)
replacements={
  'sample_id=row[\'id\'],expert=expert_receipt,':"sample_id=row['id'],joint_selection=expert_receipt,",
  "language=dict(checkpoint_sha256=R.PILOT_SHA,selected_step=100,master_dtype='float32'),":
    "paired_checkpoint_sha256=expert_receipt['best_sha256'],joint_selected_step=expert_receipt['selected_step'],",
  "HERE.glob('agent-distill-composed-*.py')":"HERE.glob('agent-distill-joint-eval-*.py')",
  'model=components.model':"model=components.model\n    assert model.config['vlm_config']['vision_config']['num_hidden_layers']==27\n    assert model.config['expert_config']['action_space_cfg']['dt']==.1 and model.config['expert_config']['action_space_cfg']['n_waypoints']==64",
  "expert_selected_step=expert_receipt['selected_step'],expert_trained=expert_receipt['expert_trained'],\n          original_seed_selected=expert_receipt['selected_step']==0,":
    "joint_selected_step=expert_receipt['selected_step'],paired_students=True,vision_layers=27,\n           baseline_pair_selected=expert_receipt['selected_step']==0,language_origin_updates=100,expert_origin_updates=25,"}
for before,after in replacements.items():
  if source.count(before)!=1:raise ValueError('Composed integration boundary changed')
  source=source.replace(before,after)
exec(compile(source,str(Path(__file__))+':composed-receipt-overlay','exec'),C.__dict__)
execute_prediction=C.execute_prediction


def teacher_preflight(plan,row):
  import numpy as np
  path,digest=C.TEACHERS[row['id']];settings=read(path)['settings']
  teacher=C.saved_teacher(path,digest,row,settings,plan)
  command=read(plan['invocation_path'])['command'];noise=Path(command[command.index('--initial-noise')+1])
  noise_values=np.load(noise,allow_pickle=False)
  if noise_values.shape!=(1,64,2) or noise_values.dtype!=np.float32 or not np.isfinite(noise_values).all():raise ValueError('Invalid initial noise')
  if hashlib.sha256(noise_values.tobytes()).hexdigest()!=settings['initial_noise_sha256']:raise ValueError('Current initial noise differs from saved teacher')
  C.validate_rollout(teacher['rollout'],28,noise_values,teacher['prediction']['prompt_tokens']+teacher['prediction']['generated_tokens'])
  return dict(reusable=True,teacher_report=str(path),teacher_report_sha256=digest,
    input_sources_noise_checked=True,live_runtime_settings_equality_required=True,generated_tokens=teacher['prediction']['generated_tokens'])


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--sample-id',choices=tuple(C.TEACHERS),required=True)
  parser.add_argument('--joint-root',type=Path,required=True);parser.add_argument('--joint-report-sha',required=True)
  parser.add_argument('--execute-gpu',action='store_true');args=parser.parse_args();args.output=args.output.resolve();root=args.output.parent
  if args.output.name!='report.json' or not root.is_relative_to(WORK) or not root.is_dir() or args.output.exists():raise ValueError('Fresh normal-wrapper output required')
  if any(root.glob('agent-distill-composed-*')):raise ValueError('Output contains an earlier composed run')
  paired,receipt=selected_joint(args.joint_root,args.joint_report_sha)
  args.teacher_report,args.teacher_sha=C.TEACHERS[args.sample_id];args.rerun_teacher=False
  plan=C.checked_plan(root)
  if not args.execute_gpu:os.environ['DEV']='CPU'
  parent=module('joint_eval_input_parent',HERE/'agent-distill-joint-parent.py')
  rows=parent.selected_rows(parent.INPUTS,[args.sample_id]);ready=parent.prepare(plan,rows)[0]
  sys.path.insert(0,str(BASE/'tinygrad'));reuse=teacher_preflight(plan,ready[0])
  if not args.execute_gpu:
    args.output.write_text(json.dumps(dict(complete=True,gpu_executed=False,preflight_only=True,joint_selection=receipt,
      sample_id=args.sample_id,teacher_reuse=reuse),indent=2)+'\n');parent.no_devices();return
  execute_prediction(args,plan,ready,paired['language'],paired['expert'],receipt)


if __name__=='__main__':main()
