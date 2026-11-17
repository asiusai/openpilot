"""Actual quality entrypoint to a blocked device boundary, plus strict parity rejection controls."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import runpy
import shutil
import socket
import sys
from unittest.mock import patch

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_selected_common',EXP/'agent-ck-selected-common.py')
S=importlib.util.module_from_spec(s);s.loader.exec_module(S)

def main(output):
  assert not output.exists();root=output.parent/output.stem;root.mkdir()
  os.environ['DEV']='USB+AMD:LLVM'  # Target metadata only; every device lookup below is blocked.
  S.configure()
  import numpy as np
  from tinygrad import Device
  from tinygrad.codegen import to_program_context
  from tinygrad.helpers import PARALLEL
  expected=S.selected()[2].pv()[1]
  assert {v.key:str(v.value) for v in to_program_context}==expected['compiler_context']
  assert PARALLEL.value==expected['parallel']==32
  original=S.read(S.QUALITY);assert S.exact(original,original)['exact']
  base=S.read(original['pipeline']);side=Path(original['pipeline']).with_suffix('.speculation.json');rejected=[]
  for name in ('one_float_bit','tokens','history','unknown_prediction','sixth_target'):
    report=copy.deepcopy(original);pipeline=copy.deepcopy(base);report['pipeline']=str(root/(name+'.json'))
    speculative=S.read(side)
    if name=='one_float_bit':
      v=pipeline['runs'][0]['prediction']['xyz']
      while isinstance(v[0],list):v=v[0]
      x=np.array([v[0]],np.float64);x.view(np.uint64)[0]^=np.uint64(1);v[0]=float(x[0])
    elif name=='tokens':pipeline['runs'][0]['prediction']['generated_token_ids'][0]^=1
    elif name=='history':report['decoder_histories'][1]['before'].append([1])
    elif name=='unknown_prediction':pipeline['runs'][0]['prediction']['unreviewed']=0
    else:speculative['records'][0]['targets'][5]^=1
    S.write(report['pipeline'],pipeline);S.write(Path(report['pipeline']).with_suffix('.speculation.json'),speculative)
    try:S.exact(original,report)
    except (AssertionError,ValueError):rejected.append(name)
    else:raise AssertionError('Accepted mutated exact-composition result: '+name)
  class Boundary(Exception):pass
  attempts=[]
  def blocked(self,name):attempts.append(name);raise Boundary(name)
  old_spec=importlib.util.spec_from_file_location
  def spec(name,location,*args,**kwargs):
    value=old_spec(name,location,*args,**kwargs)
    if location is None or Path(location).resolve()!=EXP/'agent-ck-selected-common.py':return value
    loader=value.loader
    class Loader:
      def create_module(self,spec):return loader.create_module(spec)
      def exec_module(self,m):
        loader.exec_module(m)
        # Admission occurs before final source freeze. This sole test substitute is explicitly recorded.
        m.verify=lambda:{'CPU_PREFLIGHT_ONLY':True}
        m.PREPARED=root/'prepared';m.C.PREPARED=m.PREPARED
    value.loader=Loader();return value
  prepared=root/'prepared';prepared.mkdir()
  for name in ('candidate-invocation.json','baseline-invocation.json','samples.json','ck-identity.json'):
    shutil.copyfile(S.PREPARED/name,prepared/name)
  S.write(prepared/'protocol.json',{'CPU_PREFLIGHT_ONLY':True})
  target=root/'entry/report.json';argv=[str(EXP/'agent-ck-selected-quality.py'),'--parent-run','--output',str(target)]
  before=dict(os.environ)
  try:
    with patch.object(type(Device),'__getitem__',blocked),patch.object(socket.socket,'connect',lambda *a:(_ for _ in ()).throw(AssertionError('No network'))),\
         patch.object(importlib.util,'spec_from_file_location',spec),patch.object(sys,'argv',argv):
      try:runpy.run_path(argv[0],run_name='__main__')
      except Boundary:pass
      else:raise AssertionError('Actual quality entry did not reach the device boundary')
  finally:os.environ.clear();os.environ.update(before)
  assert attempts and not Device._opened_devices and importlib.util.spec_from_file_location is old_spec
  result=S.read(target);assert not result['completed'] and not result['gpu_executed'] and result['hooks_restored']
  S.write(output,dict(complete=True,gpu_executed=False,actual_quality_runpy_preopen=True,device_lookup_blocked=attempts,
    parity_rejections=rejected,compiler_context=expected['compiler_context'],selected_parallel=32,
    manifest_verifier_mocked_before_freeze=True,no_device_opened=True))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
