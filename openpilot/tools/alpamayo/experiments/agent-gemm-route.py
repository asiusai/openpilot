"""Replay one recorded pipeline invocation with optional standalone GEMM hooks.

This runner edits no runtime source. Use only after exact random checks pass.
It keeps the recorded inputs, model settings, and pipeline timing boundary.
"""
import argparse
from collections import Counter
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--invocation',type=Path,required=True)
  parser.add_argument('--mode',choices=('reference','direct','transpose','pipeline'),default='reference')
  parser.add_argument('--warmup',type=int,default=12)
  parser.add_argument('--runs',type=int,default=24)
  parser.add_argument('--compare-reference',type=Path)
  args = parser.parse_args()
  recorded = json.loads(args.invocation.read_text())
  command = recorded['command'][1:]
  for key,value in [('output',str(args.output)),('warmup',str(args.warmup)),('runs',str(args.runs))]:
    flag = '--'+key
    assert flag in command
    command[command.index(flag)+1] = value
  for key,value in recorded['environment'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key] = value
  counts = Counter()
  if args.mode != 'reference':
    import amd_gemm
    from tinygrad import dtypes
    original_spec = importlib.util.spec_from_file_location
    spec = original_spec('agent_gemm_kernel',Path(__file__).with_name('agent-gemm-kernel.py'))
    kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)
    original = amd_gemm._gemm
    def standard(out,x,w,xs,ws,*bias,packed_input=True):
      if not packed_input or out.dtype != dtypes.bfloat16 or w.shape[-1]%16 or out.shape[-1]%16:
        return original(out,x,w,xs,ws,*bias,packed_input=packed_input)
      counts['standard'] += 1
      return kernel.program(out,x,w,xs,ws,*bias,mode=args.mode)
    amd_gemm._gemm = standard

    def patch_module(module, path):
      if path.name == 'tiled-gemm.py':
        original_kernel = module.kernel
        def tiled(out,x,w,xs,ws,*bias,**options):
          extra = {key:value for key,value in options.items() if key not in ('tile_m','tile_n','row_scale','column_scale')}
          defaults = dict(local_m=1,local_n=1,swap=False,unroll_k=1,group_m=1)
          eligible = (options.get('tile_m') == options.get('tile_n') == 4 and not options.get('column_scale',False)
                      and all(value == defaults.get(key) for key,value in extra.items())
                      and out.dtype == dtypes.bfloat16 and out.shape[-1]%16 == w.shape[-1]%16 == 0)
          if not eligible: return original_kernel(out,x,w,xs,ws,*bias,**options)
          counts['tiled'] += 1
          return kernel.program(out,x,w,xs,ws,*bias,mode=args.mode,row_scale=options.get('row_scale',False))
        module.kernel = tiled
      elif path.name == 'image-gemm.py':
        original_kernel = module.kernel
        def image(out,x,w,xs,ws,*bias,images,rows):
          if out.dtype != dtypes.bfloat16 or out.shape[-1]%16 or w.shape[-1]%16:
            return original_kernel(out,x,w,xs,ws,*bias,images=images,rows=rows)
          counts['image'] += 1
          return kernel.program(out,x,w,xs,ws,*bias,rows=rows,images=images,mode=args.mode)
        module.kernel = image

    def hooked_spec(name,location,*values,**options):
      spec = original_spec(name,location,*values,**options)
      path = Path(location)
      if path.resolve().parent != Path(__file__).resolve().parent or path.name not in ('tiled-gemm.py','image-gemm.py'):
        return spec
      original_loader = spec.loader
      class Loader:
        def create_module(self,value): return original_loader.create_module(value)
        def exec_module(self,value):
          original_loader.exec_module(value)
          patch_module(value,path)
      spec.loader = Loader()
      return spec
    importlib.util.spec_from_file_location = hooked_spec
  sys.argv = command
  try: runpy.run_path(command[0],run_name='__main__')
  finally:
    args.output.with_suffix('.agent-gemm-hooks.json').write_text(json.dumps(dict(mode=args.mode,compiled_calls=counts),indent=2)+'\n')
  if args.compare_reference:
    import numpy as np
    old = json.loads(args.compare_reference.read_text())
    new = json.loads(args.output.read_text())
    old_by_sample = {entry['sample']:entry['prediction'] for entry in old['warmup']+old['runs']}
    checked = 0
    for entry in new['warmup']+new['runs']:
      expected,actual = old_by_sample[entry['sample']],entry['prediction']
      assert expected['generated_token_ids'] == actual['generated_token_ids']
      assert expected['reasoning'] == actual['reasoning']
      for key in ('xyz','rot'):
        assert np.isfinite(actual[key]).all()
        np.testing.assert_array_equal(actual[key],expected[key],err_msg=f'{entry["sample"]} {key}')
      checked += 1
    assert checked
    args.output.with_suffix('.agent-gemm-parity.json').write_text(json.dumps(dict(exact=True,predictions=checked),indent=2)+'\n')


if __name__ == '__main__': main()
