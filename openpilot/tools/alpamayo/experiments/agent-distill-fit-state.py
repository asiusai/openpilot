"""FP32 AdamW recovery snapshots. Atomic commits reject interrupted update replay."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def atomic_json(path,value):
  temporary=path.with_name(path.name+'.tmp')
  with temporary.open('x') as stream:
    stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,path)
  descriptor=os.open(path.parent,os.O_RDONLY)
  try:os.fsync(descriptor)
  finally:os.close(descriptor)

def snapshot(weights,optimizer):
  result={}
  for index,(name,weight) in enumerate(weights.items()):
    for prefix,value in [('weight',weight),('moment1',optimizer.m[index]),('moment2',optimizer.v[index])]:
      result[prefix+'/'+name]=value.numpy().copy()
  result.update(beta1_power=optimizer.b1_t.numpy().copy(),beta2_power=optimizer.b2_t.numpy().copy(),learning_rate=optimizer.lr.numpy().copy())
  return result

def validate_arrays(arrays,shapes):
  import numpy as np
  expected={prefix+'/'+name:tuple(shape) for prefix in ('weight','moment1','moment2') for name,shape in shapes.items()}
  expected.update(beta1_power=(1,),beta2_power=(1,),learning_rate=(1,))
  assert set(arrays)==set(expected)
  for name,value in arrays.items():assert value.shape==expected[name] and value.dtype==np.float32 and np.isfinite(value).all(),name
  assert arrays['learning_rate'].tobytes()==np.array([3.5e-6],np.float32).tobytes()
  assert all(0<float(arrays[name][0])<=1 for name in ('beta1_power','beta2_power'))

def intent(root,step):
  journal=root/'journal.json'
  current=json.loads(journal.read_text())
  assert current==dict(committed=step,intended=None),'An uncommitted optimizer update cannot be replayed'
  atomic_json(journal,dict(committed=step,intended=step+1))

def save(root,arrays,metadata,context,shapes,initial_recovery=False):
  import numpy as np
  validate_arrays(arrays,shapes);step=metadata['step'];assert 0<=step<=25
  pointer=root/'latest.json';prior=json.loads(pointer.read_text()) if pointer.exists() else None
  if prior:
    assert prior['metadata']['step']+1==step and prior['context']==context
    assert json.loads((root/'journal.json').read_text())==dict(committed=step-1,intended=step)
  else:assert (step==0 or initial_recovery) and not (root/'journal.json').exists()
  destination=root/f'checkpoint-{step:04d}.npz'
  with destination.open('xb') as stream:np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
  assert destination.stat().st_size<700*1024**2
  atomic_json(pointer,dict(schema=1,context=context,metadata=metadata,file=destination.name,sha256=sha(destination)))
  atomic_json(root/'journal.json',dict(committed=step,intended=None))
  if prior:
    old=root/prior['file'];assert old.name==f'checkpoint-{step-1:04d}.npz' and sha(old)==prior['sha256'];old.unlink()

def load(root,context,shapes):
  import numpy as np
  pointer=json.loads((root/'latest.json').read_text())
  assert pointer['schema']==1 and pointer['context']==context
  step=pointer['metadata']['step'];assert isinstance(step,int) and 0<=step<=25
  assert json.loads((root/'journal.json').read_text())==dict(committed=step,intended=None),'Uncommitted update requires explicit investigation'
  assert pointer['file']==f'checkpoint-{step:04d}.npz'
  path=root/pointer['file'];assert path.stat().st_size<700*1024**2 and sha(path)==pointer['sha256']
  with zipfile.ZipFile(path) as archive:assert sum(i.file_size for i in archive.infolist())<700*1024**2
  with np.load(path,allow_pickle=False) as archive:arrays={name:archive[name].copy() for name in archive.files}
  validate_arrays(arrays,shapes)
  assert arrays['beta1_power'].tobytes()==np.array([np.float32(.9)**step],np.float32).tobytes() or np.isclose(arrays['beta1_power'][0],np.float32(.9)**step,rtol=2e-6)
  assert np.isclose(arrays['beta2_power'][0],np.float32(.999)**step,rtol=2e-6)
  return arrays,pointer['metadata']

def restore(weights,optimizer,arrays):
  from tinygrad import Tensor
  for index,(name,weight) in enumerate(weights.items()):
    for prefix,destination in [('weight',weight),('moment1',optimizer.m[index]),('moment2',optimizer.v[index])]:
      destination.assign(Tensor(arrays[prefix+'/'+name].copy(),device=destination.device).is_param_(False)).realize()
  for name,destination in [('beta1_power',optimizer.b1_t),('beta2_power',optimizer.b2_t),('learning_rate',optimizer.lr)]:
    destination.assign(Tensor(arrays[name].copy(),device=destination.device).is_param_(False)).realize()
