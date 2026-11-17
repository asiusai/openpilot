"""Guarded CPU-only middle-language-block recovery skeleton. Default is preflight, never training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
PROTECTED='0000008e--321b2ddc03'
ART=WORK/'agent-distill-feasibility-artifacts'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def local(name):
  spec=importlib.util.spec_from_file_location('distill_'+name,HERE/('agent-distill-feasibility-'+name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def protocol():
  selected=WORK/'agent-exact-live-paced/report.launcher-sources'
  sources=[Path(__file__),HERE/'agent-distill-feasibility-block.py',
    BASE/'model/config.json',BASE/'model/model.safetensors.index.json',
    BASE/'upstream/.venv/lib/python3.12/site-packages/transformers/models/cosmos3_edge/modeling_cosmos3_edge.py',
    WORK/'prefill-calibration/report.prefill-scales.json',Path('/dev/shm/alpamayo-9060-fp8/index.json'),
    ART/'agent-distill-feasibility-cpu-gradient.json']
  sources += [selected/name for name in ('model.py','quantize-packed.py','batched-vision.py','agent-speculation-prefill.py')]
  measured=json.loads((ART/'agent-distill-feasibility-cpu-gradient.json').read_text())
  master_hashes={name:value for name,value in measured['checkpoint_tensor_sha256'].items() if name.startswith('vlm.model.language_model.layers.12.')}
  assert len(master_hashes)==8
  return dict(schema=1,group=[12,13,14,15],student_initial_layer=12,eventual_student_layer=3,master_tensor_sha256=master_hashes,
    master_checkpoint=str(BASE/'model'),source_sha256={str(p):sha(p) for p in sources},
    protected_routes=[PROTECTED],device='cpu',student_compute_dtype='float32',max_updates=200,validate_every=25,learning_rate=3.5e-6,
    teacher_contract=dict(runtime='frozen selected tinygrad FP8/BF16',dynamic_vision=True,fixed_language=True,
      padded_prompt=2048,full_history_images=10,patches_per_image=720,features_per_image=180,expert_euler_steps=3,
      group_input='h entering layer 12',group_target='h leaving layer 15',kv_target='post-RoPE K and V from layer 12',
      fit_rows='actual prompt only; exclude all proposed draft and END padding rows'),
    split_minimum='One train, one validation, one sealed test record, each from a distinct evidenced drive; plumbing only.',
    missing=['No independent full-history/navigation split manifest has been verified.',
      'No selected-runtime h12/h16/K12/V12 captures from such drives exist in the reviewed artifacts.'],
    applicability='CPU recovery only. GPU backward, selected-precision student export, complete quality and latency remain unverified.')


def verify_sources(plan):
  assert plan['schema']==1 and plan['group']==[12,13,14,15] and plan['student_initial_layer']==12
  assert plan['device']=='cpu' and 1<=plan['max_updates']<=200
  assert plan['protected_routes']==[PROTECTED]
  assert plan['validate_every']==25 and plan['learning_rate']==3.5e-6 and plan['master_checkpoint']==str(BASE/'model')
  assert plan['source_sha256']==protocol()['source_sha256'] and plan['teacher_contract']==protocol()['teacher_contract']
  assert plan['master_tensor_sha256']==protocol()['master_tensor_sha256']
  for path,expected in plan['source_sha256'].items():assert sha(path)==expected,'Pinned source changed: '+path


def verify_records(manifest):
  assert manifest['schema']==1 and manifest['teacher_contract']==protocol()['teacher_contract']
  assert manifest['teacher_source_sha256']==protocol()['source_sha256'],'Capture source identities must match the frozen protocol'
  rows=manifest['samples'];assert rows and len(rows)<=64
  drives={};routes={};ids=set();captures=set();splits=set()
  for row in rows:
    assert row['id'] not in ids;ids.add(row['id'])
    assert row['split'] in ('train','validation','test');splits.add(row['split'])
    assert PROTECTED not in json.dumps(row),'Protected route cannot enter fitting or stopping, including renamed captures with original-source provenance'
    drive=row['drive_id'];assert drive and row['drive_evidence']['path'] and row['drive_evidence']['sha256']
    assert drives.setdefault(drive,row['split'])==row['split'],'Drive overlaps splits'
    assert routes.setdefault(row['route_id'],row['split'])==row['split'],'Route overlaps splits'
    assert row['capture_sha256'] not in captures,'Duplicate capture across records';captures.add(row['capture_sha256'])
    assert row['history_start_ns']<=row['query_ns'] and row['query_ns']-row['history_start_ns']>=2_000_000_000
    assert len(row['frames'])==10 and [frame['camera'] for frame in row['frames']]==[1]*5+[6]*5
    for frame in row['frames']:
      assert frame['capture_ns']<=frame['requested_ns']<=row['query_ns']
      assert frame['route_id']==row['route_id'],'Frame origin differs from its declared recording'
    nav=row['navigation']
    assert nav['text'] and nav['available_ns']<=row['query_ns'] and nav['source_sha256']
    assert nav['basis'] in ('logged_driver_instruction','recorded_driver_intent')
    assert row['input_modalities']==['images','capture_times','ego_history','navigation']
    assert 1800<row['actual_prompt_tokens']<=2048
  assert splits=={'train','validation','test'} and len(drives)>=3
  return rows


def verify_inputs(rows):
  for row in rows:
    capture=Path(row['capture_path'])
    assert capture.is_relative_to(Path('/mnt/nvme-500')) and sha(capture)==row['capture_sha256']
    evidence=row['drive_evidence'];assert sha(evidence['path'])==evidence['sha256']
    nav=row['navigation'];assert sha(nav['source_path'])==nav['source_sha256']
    for frame in row['frames']:assert sha(frame['source_path'])==frame['sha256']
    with zipfile.ZipFile(capture) as archive:assert sum(info.file_size for info in archive.infolist())<32*1024*1024
    # Test captures remain sealed: checksum/header checks do not load their numerical arrays for fitting or selection.
    if row['split']!='test':read_capture(row)


def read_capture(row):
  import numpy as np
  assert sha(row['capture_path'])==row['capture_sha256'],'Capture changed after preflight'
  shapes=dict(x=(1,2048,2048),y=(1,2048,2048),k=(1,8,2048,128),v=(1,8,2048,128),cos=(1,2048,128),sin=(1,2048,128))
  with np.load(row['capture_path'],allow_pickle=False) as archive:
    assert set(archive.files)==set(shapes),'Only current-context block tensors are permitted; no future labels'
    result={name:archive[name].copy() for name in shapes}
  for name,value in result.items():
    assert value.shape==shapes[name] and value.dtype==np.uint16,'Expected original BF16 bit arrays'
    assert np.all((value&0x7f80)!=0x7f80),'Nonfinite captured BF16 value'
  return result


def train_cpu(plan,rows,output):
  """Optional executable skeleton, intentionally CPU-only. This function was not run during feasibility work."""
  implementation=local('block');torch=implementation.torch_cpu()
  block,config=implementation.make(12);master_hashes=implementation.initialize(block,12)
  assert master_hashes==plan['master_tensor_sha256'],'Original BF16 master payload changed'
  optimizer=torch.optim.AdamW(block.parameters(),lr=plan['learning_rate'])
  train=[row for row in rows if row['split']=='train'];validation=[row for row in rows if row['split']=='validation']
  def evaluate(row,grad):
    arrays=read_capture(row)
    tensors={name:torch.from_numpy(value).view(torch.bfloat16).float() for name,value in arrays.items()}
    with torch.set_grad_enabled(grad):
      result=implementation.forward(block,config,tensors['x'],tensors['cos'],tensors['sin'])
      loss,_=implementation.recovery_loss(result,tuple(tensors[name] for name in ('y','k','v')),row['actual_prompt_tokens'])
    return loss
  output.mkdir(parents=True,exist_ok=False)
  write(output/'agent-distill-feasibility-run-protocol.json',dict(plan=plan,master_tensor_sha256=master_hashes))
  from safetensors.torch import save_file
  best=float('inf');history=[]
  def validate(step):
    nonlocal best
    value=sum(float(evaluate(row,False)) for row in validation)/len(validation)
    history.append(dict(update=step,validation_loss=value))
    if value<best:
      best=value;save_file({name:value.detach().contiguous() for name,value in block.state_dict().items()},
                          str(output/'agent-distill-feasibility-best.safetensors'))
    write(output/'agent-distill-feasibility-loss.json',history)
  validate(0)
  for update in range(1,plan['max_updates']+1):
    optimizer.zero_grad(set_to_none=True)
    loss=evaluate(train[(update-1)%len(train)],True)
    assert torch.isfinite(loss);loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in block.parameters())
    optimizer.step()
    if update%plan['validate_every']==0 or update==plan['max_updates']:validate(update)
  implementation.assert_cpu_only()
  write(output/'agent-distill-feasibility-complete.json',dict(optimizer_updates=plan['max_updates'],device='cpu',
    test_opened_for_model_selection=False,whole_model_quality_qualified=False,selected_precision_export_qualified=False))


def main(argv=None):
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol',type=Path,default=ART/'agent-distill-feasibility-protocol-v3.json')
  parser.add_argument('--write-protocol',action='store_true')
  parser.add_argument('--split-manifest',type=Path)
  parser.add_argument('--execute-training',action='store_true',default=False)
  parser.add_argument('--output-dir',type=Path,default=WORK/'agent-distill-feasibility-proposed-recovery')
  args=parser.parse_args(argv)
  if args.write_protocol:
    assert not args.execute_training
    assert args.protocol.resolve().is_relative_to(WORK) and not args.protocol.exists()
    args.protocol.parent.mkdir(parents=True,exist_ok=True);write(args.protocol,protocol())
  plan=json.loads(args.protocol.read_text());verify_sources(plan)
  if args.split_manifest is None:
    if args.execute_training:raise ValueError('Training blocked: independent drive split and selected-runtime block captures are missing')
    print(json.dumps(dict(training_executed=False,status='blocked',missing=plan['missing'],protocol_sha256=sha(args.protocol))))
    return
  manifest=json.loads(args.split_manifest.read_text());rows=verify_records(manifest);verify_inputs(rows)
  summary=dict(training_executed=False,status='input_preflight_passed',split_manifest_sha256=sha(args.split_manifest),
               protocol_sha256=sha(args.protocol),samples=len(rows),device='cpu')
  if not args.execute_training:print(json.dumps(summary));return
  output=args.output_dir.resolve()
  assert output.is_relative_to(WORK) and output.name.startswith('agent-distill-feasibility-') and not output.exists()
  train_cpu(plan,rows,output)


if __name__=='__main__':main()
