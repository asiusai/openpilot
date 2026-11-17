"""CPU-only negative controls for exact program and batch authorization."""
import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-source-work')
def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main(catalog,output):
  assert not output.exists()
  os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0',PARALLEL='0',PROFILE='0',BEAM='0',JITBEAM='0',LLVMOPT='1',GMMU='0',TC_OPT='2',
    TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'))
  sys.path.insert(0,'/tmp/opencode/alpamayo-asius-20261114/tinygrad')
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer,Device
  from tinygrad.uop.ops import UOp
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.codegen import to_program
  common=load('agent-ck-qualified-common');identity=load('agent-ck-source-identity');plan=json.loads(catalog.read_text())
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  def tensor(): return Tensor(UOp.from_buffer(Buffer('AMD',8388608),dtypes.bfloat16)).reshape(common.QSHAPE)
  inputs=[tensor(),tensor()]
  output_tensor=common.utility(*inputs)
  raw,_=output_tensor.linear_with_vars()
  programs=[to_program(c.body,renderer) for c in raw.src]
  assert len(programs)==2
  integer=programs[1]
  floating=to_program(inputs[0].contiguous().clone().linear_with_vars()[0].src[0].body,renderer)
  a,ia=identity.authorize(integer,plan);b,ib=identity.authorize(floating,plan)
  assert a!=b and ia['binary_sha256']==ib['binary_sha256'] and ia['source_sha256']!=ib['source_sha256']
  old=json.loads(Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work/prepared-final/protocol.json').read_text())
  assert ia['source_sha256']!=old['programs'][ia['binary_sha256']]['source_sha256']
  controls=[]
  def reject(name,program):
    try: identity.authorize(program,plan)
    except (AssertionError,ValueError): controls.append({'name':name,'rejected':True})
    else: raise AssertionError(('Unexpected authorization',name))
  reject('unlisted_source_same_binary',integer.replace(src=(*integer.src[:2],integer.src[2].replace(arg=integer.src[2].arg+'\n'),integer.src[3])))
  reject('legitimate_source_wrong_AST',integer.replace(src=(integer.src[0],integer.src[1],floating.src[2],integer.src[3])))
  reject('legitimate_source_wrong_lowered_AST',integer.replace(src=(integer.src[0],floating.src[1],integer.src[2],integer.src[3])))
  reject('wrong_target',integer.replace(arg=replace(integer.arg,target=Target('AMD','LLVM','gfx1100','USB'))))
  reject('wrong_launch_ABI',integer.replace(arg=replace(integer.arg,local_size=(64,1,1))))
  reject('wrong_binary',integer.replace(src=(*integer.src[:3],programs[0].src[3])))
  rows=next(v for v in plan['batches'].values() if len(v)==2 and v[1]['dependencies']==[0])
  for name,change in (
      ('removed_dependency',lambda r:r[1].update(dependencies=[])),
      ('wrong_pointer_offset',lambda r:r[1]['arguments'][1].update(offset_bytes=512)),
      ('wrong_alias_binding',lambda r:r[1]['arguments'][1].update(root=999)),
      ('wrong_call_association',lambda r:r[1].update(association=b if r[1]['association']!=b else a))):
    bad=copy.deepcopy(rows);change(bad);assert identity.key(bad) not in plan['batches'];controls.append({'name':name,'rejected':True})
  assert not Device._opened_devices
  result={'gpu_executed':False,'opened_devices':[],'old_guard_failure_reproduced':True,
    'shared_binary_sha256':ia['binary_sha256'],'legitimate_associations':[a,b],'negative_controls':controls,
    'catalog_sha256':hashlib.sha256(catalog.read_bytes()).hexdigest()}
  with output.open('x') as stream: stream.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result))


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();main(args.catalog,args.output)
