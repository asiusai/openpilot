"""Bounded CPU-only profile of the selected live adapter; no scheduling changes."""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pstats
import socket
import sys
import time
from types import FunctionType
from unittest.mock import patch

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
EXP=SOURCE/'experiments'
HERE=Path(__file__).resolve().parent
ARTIFACTS=WORK/'agent-live-history-fast-artifacts'
CHECKPOINT=Path('/tmp/opencode/alpamayo-asius-20261114/model')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')

def sha(path):
  with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
  sys.modules[name]=module;spec.loader.exec_module(module);return module

def configure():
  for path in (ARTIFACTS,ARTIFACTS/'tmp',ARTIFACTS/'cache'):path.mkdir(exist_ok=True)
  os.environ.update(DEV='CPU',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',CACHEDB=str(ARTIFACTS/'compiler-cache.db'),TMPDIR=str(ARTIFACTS/'tmp'),XDG_CACHE_HOME=str(ARTIFACTS/'cache'))
  sys.dont_write_bytecode=True
  sys.path[:0]=[str(TINY),str(EXP),str(SOURCE),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']

def cpu_guards(stack):
  from tinygrad import Device
  original=type(Device).__getitem__
  def cpu_device(self,name):
    if str(name).split(':')[0] not in ('CPU','NPY'):raise AssertionError('GPU forbidden: '+str(name))
    return original(self,name)
  def no_network(*a,**kw):raise AssertionError('Network forbidden')
  stack.enter_context(patch.object(type(Device),'__getitem__',cpu_device))
  stack.enter_context(patch.object(socket.socket,'connect',no_network))

def sample(row):
  import numpy as np
  import torch
  return {p.stem:[str(np.load(p,allow_pickle=False).item())] if p.stem=='nav_text' else torch.from_numpy(np.load(p,allow_pickle=False))
    for p in Path(row['sample']).glob('*.npy')}

def stats(values):
  import numpy as np
  return dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),min_ms=min(values),max_ms=max(values))

def exact(a,b,label='value'):
  import numpy as np
  import torch
  if isinstance(a,torch.Tensor):
    assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype,label
    assert a.contiguous().view(torch.uint8).numpy().tobytes()==b.contiguous().view(torch.uint8).numpy().tobytes(),label
  elif isinstance(a,np.ndarray):
    assert isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
  elif isinstance(a,dict):
    assert a.keys()==b.keys(),label
    for key in a:exact(a[key],b[key],label+'/'+str(key))
  elif isinstance(a,(tuple,list)):
    assert type(a) is type(b) and len(a)==len(b),label
    for i,(x,y) in enumerate(zip(a,b,strict=True)):exact(x,y,label+'/'+str(i))
  else:assert a==b,label

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();assert args.output.resolve().is_relative_to(WORK) and not args.output.exists()
  configure()
  import torch
  torch.set_num_threads(4)
  with ExitStack() as stack:
    cpu_guards(stack)
    A=load(EXP/'fast-inputs.py','history_fast_profile_adapter')
    adapter=A.FastInputs(CHECKPOINT,camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True,cache_resize=True)
    spans={}
    def timed(label,fn):
      def wrapped(*a,**kw):
        start=time.perf_counter_ns()
        try:return fn(*a,**kw)
        finally:spans[label]=spans.get(label,0)+(time.perf_counter_ns()-start)/1e6
      return wrapped
    for owner,attr,label in [(adapter.processor,'apply_chat_template','render'),(adapter.processor,'_process_images','images'),(adapter.processor.image_processor,'patchify','patchify')]:
      setattr(owner,attr,timed(label,getattr(owner,attr)))
    original=adapter.tokenize
    adapter.tokenize=FunctionType(original.__code__,dict(original.__globals__,resize=timed('resize',original.__globals__['resize'])),original.__name__,original.__defaults__)
    adapter.tokenize.__kwdefaults__=original.__kwdefaults__
    schedule=json.loads((WORK/'stream-10hz/floor/manifest.json').read_text())['schedule'][:80:4]
    rows=[]
    for i,row in enumerate(schedule):
      data=sample(row);spans.clear();start=time.perf_counter_ns();adapter.prepare(data);elapsed=(time.perf_counter_ns()-start)/1e6
      rows.append(dict(index=i,query_ns=row['query_ns'],measured=i>=4,total_ms=elapsed,spans=dict(spans)))
    profile=cProfile.Profile();profile.enable();adapter.prepare(data);profile.disable()
    text=io.StringIO();pstats.Stats(profile,stream=text).strip_dirs().sort_stats('cumulative').print_stats(45)
    measured=[r for r in rows if r['measured']]
    report=dict(complete=True,cpu_only=True,gpu_executed=False,rows=rows,torch_threads=4,
      timing={'total':stats([r['total_ms'] for r in measured]),**{key:stats([r['spans'][key] for r in measured]) for key in spans}},
      profile=text.getvalue(),cache=dict(resize_entries=len(adapter.resize_cache)),source_sha256={str(p):sha(p) for p in (Path(__file__),EXP/'fast-inputs.py')})
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['timing']),flush=True)

if __name__=='__main__':main()
