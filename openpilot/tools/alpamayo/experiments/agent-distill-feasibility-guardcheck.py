"""CPU/std-library preflight controls. No numerical training and no model imports."""
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ART=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-distill-feasibility-artifacts')


def main():
  spec=importlib.util.spec_from_file_location('distill_guarded_train',HERE/'agent-distill-feasibility-train.py')
  trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
  protocol=ART/'agent-distill-feasibility-protocol-v3.json'
  capture=io.StringIO()
  with redirect_stdout(capture):trainer.main(['--write-protocol'] if not protocol.exists() else [])
  declared=json.loads(protocol.read_text());trainer.verify_sources(declared)
  assert json.loads(capture.getvalue())['training_executed'] is False
  cases=[]
  def rejects(name,fn,errors=(AssertionError,ValueError)):
    try:fn()
    except errors as error:cases.append(dict(case=name,rejected=True,error=type(error).__name__))
    else:raise AssertionError('Guard accepted '+name)
  rejects('explicit-training-without-inputs',lambda:trainer.main(['--execute-training']))
  bad=copy.deepcopy(declared);bad['max_updates']=201
  rejects('over-200-updates',lambda:trainer.verify_sources(bad))
  bad=copy.deepcopy(declared);first=next(iter(bad['source_sha256']));bad['source_sha256'][first]='0'*64
  rejects('source-pin-mutation',lambda:trainer.verify_sources(bad))
  bad=copy.deepcopy(declared);first=next(iter(bad['master_tensor_sha256']));bad['master_tensor_sha256'][first]='0'*64
  rejects('bf16-master-pin-mutation',lambda:trainer.verify_sources(bad))
  rows=[]
  for index,split in enumerate(('train','validation','test')):
    query=10_000_000_000
    frames=[dict(camera=c,capture_ns=query+offset,requested_ns=query+offset,route_id='not-an-existing-route-'+str(index)) for c in (1,6)
            for offset in (-2_000_000_000,-1_000_000_000,-500_000_000,-100_000_000,0)]
    rows.append(dict(id='fixture-'+str(index),split=split,route_id='not-an-existing-route-'+str(index),drive_id='fixture-drive-'+str(index),
      drive_evidence=dict(path='/nonexistent/metadata-fixture',sha256='1'*64),capture_sha256=str(index+1)*64,
      capture_path=str(ART/'nonexistent-fixture-capture.npz'),history_start_ns=query-2_000_000_000,query_ns=query,frames=frames,
      navigation=dict(text='Fixture instruction',available_ns=query,source_sha256='2'*64,basis='logged_driver_instruction'),
      input_modalities=['images','capture_times','ego_history','navigation'],actual_prompt_tokens=2027))
  manifest=dict(schema=1,teacher_contract=declared['teacher_contract'],teacher_source_sha256=declared['source_sha256'],samples=rows)
  assert len(trainer.verify_records(manifest))==3
  rejects('metadata-fixture-is-not-real-data',lambda:trainer.verify_inputs(rows),(FileNotFoundError,))
  for name,change in (
      ('protected-entire-route',lambda m:m['samples'][0].update(route_id=trainer.PROTECTED)),
      ('protected-renamed-source',lambda m:m['samples'][0]['frames'][0].update(source_path='/source/'+trainer.PROTECTED+'/frame')),
      ('frame-origin-mismatch',lambda m:m['samples'][0]['frames'][0].update(route_id='another-route')),
      ('drive-across-splits',lambda m:m['samples'][1].update(drive_id=m['samples'][0]['drive_id'])),
      ('route-across-splits',lambda m:m['samples'][1].update(route_id=m['samples'][0]['route_id'])),
      ('duplicate-capture',lambda m:m['samples'][1].update(capture_sha256=m['samples'][0]['capture_sha256'])),
      ('future-camera',lambda m:m['samples'][0]['frames'][0].update(capture_ns=10_000_000_001)),
      ('future-navigation',lambda m:m['samples'][0]['navigation'].update(available_ns=10_000_000_001)),
      ('future-input-modality',lambda m:m['samples'][0].update(input_modalities=['images','future_trajectory'])),
      ('unmatched-teacher-source',lambda m:m.update(teacher_source_sha256={})),
      ('unknown-navigation-origin',lambda m:m['samples'][0]['navigation'].update(basis='inferred_from_future_ground_truth'))):
    bad=copy.deepcopy(manifest);change(bad)
    rejects(name,lambda:trainer.verify_records(bad))
  assert 'torch' not in sys.modules and not any(name.startswith('tinygrad.runtime.ops_') for name in sys.modules)
  result=dict(schema=1,training_executed=False,optimizer_updates=0,gpu_initialized=False,
    default_without_training=True,explicit_training_missing_data_rejected=True,metadata_fixture_is_not_dataset=True,
    real_split_verified=False,negative_controls=cases,protocol_sha256=trainer.sha(protocol),
    source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   (Path(__file__),HERE/'agent-distill-feasibility-train.py')})
  path=ART/'agent-distill-feasibility-guardcheck-v3.json';trainer.write(path,result)
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()
