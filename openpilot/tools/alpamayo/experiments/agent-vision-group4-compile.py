"""CPU compiler/resources evidence for unchanged4x4 GEMM shapes and renderer mode loss."""
import argparse
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('agent_vision_group4_compile_common',HERE/'agent-vision-group4-common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  if args.output.exists():raise FileExistsError('Keep compiler evidence immutable')
  c.require(os.environ.get('ALLOW_DEVICE_USAGE')=='0','Set ALLOW_DEVICE_USAGE=0')
  sys.path[:0]=[str(c.BASE),str(c.BASE/'experiments')]
  from tinygrad import Device,dtypes
  from tinygrad.codegen import do_to_program,to_program_key
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.uop.ops import UOp
  target=Target('AMD','LLVM','gfx1200','USB')
  renderer=AMDLLVMRenderer(target)
  wgp=c.module(c.BASE/'experiments/wgp_compiler.py','agent_vision_group4_original_wgp')
  renderer.compiler=wgp.WGPCompiler('gfx1200')
  worker=pickle.loads(pickle.dumps(renderer))
  tiled=c.module(c.BASE/'experiments/tiled-gemm.py','agent_vision_group4_original_tiled')
  resources=c.module(Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-gemm-compile.py'),'agent_vision_group4_resources')
  shapes=(('attention_linear',720,1152,1152),('fc1',720,4304,1152),('fc2',720,1152,4304),
          ('projector_fc1',180,11520,4608),('projector_fc2',180,2048,11520))
  args.output.parent.mkdir(parents=True,exist_ok=True)
  report=dict(cpu_only=True,gpu_executed=False,complete=False,target=repr(target),tile=[4,4],records=[],
    renderer_serialization=dict(parent_compiler=type(renderer.compiler).__name__,parent_cachekey=renderer.compiler.cachekey,
      worker_compiler=type(worker.compiler).__name__,worker_cachekey=worker.compiler.cachekey,
      worker_loses_instance_WGP=True,mode_changed_for_GPU_pair=False),
    source_sha256={str(path):c.sha(path) for path in (Path(__file__),c.BASE/'experiments/tiled-gemm.py',c.BASE/'experiments/batched-vision.py',
      c.BASE/'experiments/projector-kernels.py',c.BASE/'experiments/wgp_compiler.py')})
  for images in (1,2,3,4):
    for name,rows,columns,width in shapes:
      row_scale=images>1 if name.startswith('projector') else True
      spec=[((images*rows,columns),dtypes.bfloat16),((images*rows,width),dtypes.fp8e4m3),((columns,width),dtypes.fp8e4m3),
        (((images*rows,) if row_scale else (1,)),dtypes.float32),((1,),dtypes.float32),((columns,),dtypes.bfloat16)]
      values=[UOp.placeholder(shape,dtype,i) for i,(shape,dtype) in enumerate(spec)]
      class Proxy:
        DEFAULT='CPU_COMPILER_ONLY'
        def __getitem__(self,key):return SimpleNamespace(renderer=renderer)
      with patch.object(tiled,'Device',Proxy()):ast=tiled.kernel(*values,tile_m=4,tile_n=4,row_scale=row_scale)
      assert to_program_key(ast,renderer)==to_program_key(ast,worker)
      row=dict(images=images,name=name,rows_per_image=rows,mnk=[images*rows,columns,width],row_scale=row_scale,
        ast=ast.key.hex(),same_cache_key_despite_mode_loss=True,modes={})
      for mode,r in (('requested_parent_WGP',renderer),('reconstructed_worker_CU',worker)):
        program=do_to_program(ast,r);source,binary=program.src[2].arg,program.src[3].arg
        prefix=args.output.parent/f'agent-vision-group4-b{images}-{name}-{mode}'
        prefix.with_suffix('.ll').write_text(source);prefix.with_suffix('.hsaco').write_bytes(binary)
        row['modes'][mode]=dict(source_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
          binary_sha256=__import__('hashlib').sha256(binary).hexdigest(),resources=resources.metadata(binary),
          binary_bytes=len(binary),global_size=program.arg.global_size,local_size=program.arg.local_size)
      report['records'].append(row);print(json.dumps(row),flush=True)
  assert not Device._opened_devices
  report.update(complete=True,opened_devices=[],no_new_tile_or_arithmetic=True,
    limits='CPU compilation is not GPU arithmetic parity. Actual vision binaries/modes must be captured in both parent arms.')
  c.write(args.output,report)


if __name__=='__main__':main()
