"""Fixed160 state adapter. Reuse qualified atomic writes, intents, snapshots, and restoration."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

BASE=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-distill-fit-artifacts/live-v4/frozen/agent-distill-fit-state.py')
spec=importlib.util.spec_from_file_location('vision_cohort_fit_original_state',BASE)
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
atomic_json=original.atomic_json
snapshot=original.snapshot
restore=original.restore
intent=original.intent
sha=original.sha


def validate_arrays(arrays,shapes,step):
  import numpy as np
  assert isinstance(step,int) and 0<=step<=160
  expected={prefix+'/'+name:tuple(shape) for prefix in ('weight','moment1','moment2') for name,shape in shapes.items()}
  expected.update(beta1_power=(1,),beta2_power=(1,),learning_rate=(1,))
  assert set(arrays)==set(expected)
  for name,value in arrays.items():
    assert value.shape==expected[name] and value.dtype==np.float32 and np.isfinite(value).all(),name
    if name.startswith('moment2/'):assert np.all(value>=0),name
  assert arrays['learning_rate'].tobytes()==np.array([3.5e-5],np.float32).tobytes()
  for key,beta in (('beta1_power',.9),('beta2_power',.999)):
    power=np.float32(1)
    for _ in range(step):power=np.float32(power*np.float32(beta))
    assert np.array_equal(arrays[key],np.array([power],np.float32)),key


def save(root,arrays,metadata,context,shapes,initial_recovery=False):
  import numpy as np
  step=metadata['step'];validate_arrays(arrays,shapes,step)
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
  pointer=json.loads((root/'latest.json').read_text());assert pointer['schema']==1 and pointer['context']==context
  step=pointer['metadata']['step'];assert isinstance(step,int) and 0<=step<=160
  assert json.loads((root/'journal.json').read_text())==dict(committed=step,intended=None),'Dirty journal cannot be resumed'
  assert pointer['file']==f'checkpoint-{step:04d}.npz'
  path=root/pointer['file'];assert path.stat().st_size<700*1024**2 and sha(path)==pointer['sha256']
  with zipfile.ZipFile(path) as archive:assert sum(i.file_size for i in archive.infolist())<700*1024**2
  with np.load(path,allow_pickle=False) as archive:arrays={name:archive[name].copy() for name in archive.files}
  validate_arrays(arrays,shapes,step)
  return arrays,pointer['metadata']


def save_best(root,weights,metadata):
  import numpy as np
  temporary=root/'best-next.npz'
  with temporary.open('xb') as stream:
    np.savez(stream,**{name:value.numpy() for name,value in weights.items()});stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,root/'best.npz')
  atomic_json(root/'best.json',dict(file='best.npz',sha256=sha(root/'best.npz'),
    selected_step=metadata['best_step'],validation_mean=metadata['best_validation']))


def check_best(root,metadata,shapes):
  import numpy as np
  receipt=json.loads((root/'best.json').read_text())
  assert receipt['file']=='best.npz' and receipt['sha256']==sha(root/'best.npz')
  assert receipt['selected_step']==metadata['best_step'] and receipt['validation_mean']==metadata['best_validation']
  with np.load(root/'best.npz',allow_pickle=False) as archive:
    assert set(archive.files)==set(shapes)
    for name,shape in shapes.items():
      value=archive[name];assert value.shape==tuple(shape) and value.dtype==np.float32 and np.isfinite(value).all()
  return receipt
