"""Explicit read-only predecessor admission and durable continuation selection."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())

def history(metadata,step):
  import numpy as np
  assert metadata['step']==step and type(step) is int and 25<=step<=100
  items=metadata['validation'];assert [x['step'] for x in items]==list(range(0,step//25*25+1,25))
  for item in items:
    assert item['samples']==['validation-command-'+str(i) for i in range(3)]
    assert type(item['mean']) in (float,int) and np.isfinite(item['mean']) and item['mean']>=0
  # Python min retains the first equal element, matching strict-improvement selection.
  winner=min(items,key=lambda item:item['mean'])
  assert metadata['best_step']==winner['step'] and metadata['best_validation']==winner['mean']

def read_best(root,metadata,shapes,expected_sha=None):
  import numpy as np
  root=Path(root);item=read(root/'best.json');assert item['file']=='best.npz'
  digest=sha(root/'best.npz');assert item['sha256']==digest
  assert item['selected_step']==metadata['best_step'] and item['validation_mean']==metadata['best_validation']
  if expected_sha is not None:assert digest==expected_sha
  if 'best_artifact_sha256' in metadata:assert digest==metadata['best_artifact_sha256']
  assert (root/'best.npz').stat().st_size<256*1024**2
  with np.load(root/'best.npz',allow_pickle=False) as archive:
    assert set(archive.files)==set(shapes)
    values={name:archive[name].copy() for name in shapes}
  assert all(v.dtype==np.float32 and v.shape==tuple(shapes[k]) and np.isfinite(v).all() for k,v in values.items())
  return values,item

def predecessor(p,old_state):
  import numpy as np
  pred=p['predecessor'];root=Path(pred['root']);assert pred['committed_step']==25
  expected={'report.json','latest.json','journal.json','checkpoint-0025.npz','best.json','best.npz','wall.json'}
  assert set(pred['files'])==expected
  for name,digest in pred['files'].items():assert sha(root/name)==digest,name
  report=read(root/'report.json');pointer=read(root/'latest.json');wall=read(root/'wall.json')
  assert report['complete'] is True and report['gpu_executed'] is True
  assert report['real_optimizer_updates']==report['committed_step']==report['best_step']==25
  assert report['protocol_sha256']==pred['protocol_sha256'] and not report['sealed_targets_opened'] and not report['driving_model_modified']
  assert not wall['timed_out'] and wall['returncode']==0
  assert read(root/'journal.json')==dict(committed=25,intended=None)
  assert pointer['context']==pred['context'] and pointer['context']['protocol_sha256']==pred['protocol_sha256']
  assert pointer['context']['manifest_sha256']==p['training_manifest_sha256']
  old_hyper=pointer['context']['hyperparameters'];assert old_hyper['updates']==25 and p['hyperparameters']['updates']==100
  assert {k:v for k,v in old_hyper.items() if k!='updates'}=={k:v for k,v in p['hyperparameters'].items() if k!='updates'}
  shapes={name:tuple(v['shape']) for name,v in p['master_weights'].items()}
  arrays,metadata=old_state.load(root,pred['context'],shapes)
  assert metadata==pred['metadata'] and metadata['validation']==report['validation'];history(metadata,25)
  best,item=read_best(root,metadata,shapes,pred['files']['best.npz'])
  for name in shapes:np.testing.assert_array_equal(best[name].view(np.uint32),arrays['weight/'+name].view(np.uint32))
  assert len(arrays)==3*len(shapes)+3
  return arrays,copy.deepcopy(metadata),dict(root=str(root),checkpoint_sha256=pred['files']['checkpoint-0025.npz'],
    latest_sha256=pred['files']['latest.json'],old_context=pred['context'],committed_step=25,next_step=26,next_command_index=1,
    restored_arrays=len(arrays),best_weights_equal_committed=True)

def context(p,protocol_sha):
  return dict(protocol_sha256=protocol_sha,manifest_sha256=p['training_manifest_sha256'],hyperparameters=p['hyperparameters'],
    predecessor=dict(protocol_sha256=p['predecessor']['protocol_sha256'],latest_sha256=p['predecessor']['files']['latest.json'],
      checkpoint_sha256=p['predecessor']['files']['checkpoint-0025.npz'],committed_step=25))

def copy_best(old_root,new_root,metadata,state):
  old_root,new_root=Path(old_root),Path(new_root);item=read(old_root/'best.json');assert item['file']=='best.npz'
  with (old_root/'best.npz').open('rb') as source,(new_root/'best.npz').open('xb') as target:
    shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())
  assert sha(new_root/'best.npz')==item['sha256']
  metadata['best_artifact_sha256']=item['sha256'];state.atomic_json(new_root/'best.json',item)

def commit_best(root,weights,metadata,state):
  import numpy as np
  root=Path(root);temporary=root/'best-next.npz'
  with temporary.open('xb') as stream:
    np.savez(stream,**{name:value.numpy() for name,value in weights.items()});stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,root/'best.npz');metadata['best_artifact_sha256']=sha(root/'best.npz')
  state.atomic_json(root/'best.json',dict(file='best.npz',sha256=metadata['best_artifact_sha256'],
    selected_step=metadata['best_step'],validation_mean=metadata['best_validation']))
