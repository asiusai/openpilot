"""Compile the actual L2048 generic forward/backward ASTs offline. No Device opening."""
import importlib
import json
import os
from pathlib import Path
import sys
import time

C=importlib.import_module('agent-distill-full-common')

def main():
  assert not (C.ROOT/'offline-compile.json').exists();C.no_gpu_fds();start=time.monotonic()
  def deny(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/'))
  sys.addaudithook(deny)
  from tinygrad import Device
  from tinygrad.helpers import ALLOW_DEVICE_USAGE,Target,USE_TC,ALLOW_TF32,NO_MEMORY_PLANNER
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import Ops
  assert ALLOW_DEVICE_USAGE.value==USE_TC.value==ALLOW_TF32.value==0 and NO_MEMORY_PLANNER.value==1
  renderer=AMDLLVMRenderer(Target.parse(C.TARGET));assert repr(renderer.target)==C.TARGET
  b,x,c,s,t,o,a,l,g=C.graph();linear=l.schedule_linear(*o,a,*g)
  records=[]
  for index,call in enumerate(linear.src):
    ast=call.body;assert ast.op is Ops.SINK
    prg=to_program(ast,renderer);blob=prg.src[3].arg
    prefix=C.ROOT/'offline-programs'/f'{index:02d}';prefix.parent.mkdir(exist_ok=True)
    prefix.with_suffix('.ll').write_text(prg.src[2].arg);prefix.with_suffix('.elf').write_bytes(blob)
    records.append({'call_index':index,'input_ast':ast.key.hex(),'program_key':prg.key.hex(),
      'source_sha256':C.B.sha(prg.src[2].arg.encode()),'binary_sha256':C.B.sha(blob),
      'binary_bytes':len(blob),'global_size':prg.arg.global_size,'local_size':prg.arg.local_size,
      'wmma_uops':sum(u.op is Ops.WMMA for u in prg.src[1].src)})
  assert not Device._opened_devices and not any(r['wmma_uops'] for r in records);C.no_gpu_fds()
  report={'complete':True,'gpu_executed':False,'opened_devices':[],'target':C.TARGET,'renderer':type(renderer).__name__,
    'compiler':type(renderer.compiler).__name__,'compiler_cachekey':renderer.compiler.cachekey,'length':C.LENGTH,
    'calls':len(records),'unique_input_asts':len({r['input_ast'] for r in records}),'records':records,
    'seconds':time.monotonic()-start,'TC':0,'ALLOW_TF32':0,'NO_MEMORY_PLANNER':1,
    'source_sha256':{str(p):C.file_sha(p) for p in (Path(__file__),C.EXP/'agent-distill-full-common.py')}}
  C.write(C.ROOT/'offline-compile.json',report)
  print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
