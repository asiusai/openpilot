"""Single L2048 FP32 language-block feasibility; synthetic inputs and no optimizer."""
import importlib.util
from pathlib import Path
import sys

EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-full-artifacts'
OLD=WORK/'agent-distill-tinygrad-artifacts'
KFD=WORK/'agent-distill-kfd1-artifacts'
SUCCESS=WORK/'agent-distill-kfd1-import-v2-artifacts'
spec=importlib.util.spec_from_file_location('agent-distill-full-kfd-base',KFD/'frozen/agent-distill-kfd1-common.py')
K=importlib.util.module_from_spec(spec);sys.modules[spec.name]=K;spec.loader.exec_module(K)
B=K.B
file_sha,read,write,no_gpu_fds=K.file_sha,K.read,K.write,K.no_gpu_fds
TINY,PY,TARGET=K.TINY,K.PY,K.TARGET
LENGTH=2048
VALID_ROWS=2047
LIVE_LIMIT=12*1024**3
TOTAL_LIMIT=16*1024**3
MAX_BUFFER=256*1024**2
FREE_RESERVE=5*1024**3
TOLERANCES=B.TOLERANCES

def module(path,name):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
  sys.modules[name]=m;spec.loader.exec_module(m);return m
def weights_metadata():return {name[2:]:item for name,item in read(OLD/'oracle.json')['arrays'].items() if name.startswith('w:')}
def block(values,device,placeholders=False):
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp
  b=B.Block.__new__(B.Block);b.eps=read(OLD/'oracle.json')['config']['rms_norm_eps'];assert b.eps==1e-5
  b.weights={name:(Tensor(UOp.new_buffer(device,__import__('math').prod(item['shape']),dtypes.float32).reshape(item['shape']))
      if placeholders else Tensor(values['w:'+name].copy(),device=device)) for name,item in weights_metadata().items()}
  assert sum(w.numel() for w in b.weights.values())==50335744
  return b
def loss(outputs,targets,valid=VALID_ROWS):
  assert 0<valid<=LENGTH
  return B.recovery_loss(outputs,targets,valid)
def graph(device='AMD'):
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp
  import math
  leaf=lambda shape:Tensor(UOp.new_buffer(device,math.prod(shape),dtypes.float32).reshape(shape))
  b=block(None,device,True);x=leaf((1,LENGTH,2048));cos,sin=(leaf((1,LENGTH,128)) for _ in range(2))
  targets=(leaf((1,LENGTH,2048)),leaf((1,8,LENGTH,128)),leaf((1,8,LENGTH,128)))
  outputs,activation=b(x,cos,sin);objective=loss(outputs,targets)
  gradients=objective.gradient(*b.weights.values(),x)
  return b,x,cos,sin,targets,outputs,activation,objective,gradients
def array_manifest():
  import numpy as np
  oracle=read(ROOT/'oracle.json');assert oracle['complete'] and oracle['length']==LENGTH and oracle['optimizer_updates']==0
  for path,digest in oracle['source_sha256'].items():assert file_sha(path)==digest,path
  arrays={}
  for name,item in oracle['arrays'].items():
    path=Path(item['path']);assert file_sha(path)==item['sha256'],name
    arrays[name]=np.load(path,allow_pickle=False)
    assert list(arrays[name].shape)==item['shape'] and str(arrays[name].dtype)==item['dtype']
  return oracle,arrays
